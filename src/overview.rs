//! `overview` verb core: repository structure at a glance.
//!
//! One bounded answer to "what does this repo look like": the directory
//! skeleton with per-dir file counts and dominant extensions, served from a
//! live gitignore-aware walk (the indexing walker's ignore semantics, but
//! over ALL files, not just indexable extensions) so it never depends on
//! the vector backend or index freshness. Rendering is pruned
//! deterministically to a token budget (bytes/4, the same estimate the
//! usage telemetry ships): depth is reduced until it fits, then at depth 1
//! the smallest dirs are dropped. Ordering is fully deterministic — BTreeMap
//! aggregation over a sorted file list — so identical trees render
//! identically.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use anyhow::Result;
use serde_json::json;
use tracing::info;

/// Default rendered-output token budget; 0 disables pruning.
pub const DEFAULT_TOKEN_BUDGET: usize = 1200;

/// Extension entries kept in the root line / JSON languages list.
const LANG_JSON_TOP: usize = 6;
const LANG_HUMAN_TOP: usize = 5;

/// Per-directory aggregates during the single pass over the file list.
#[derive(Default)]
struct DirStats {
    /// Extension (lowercased; extensionless files keyed by lowercase
    /// filename) -> direct file count.
    files: BTreeMap<String, usize>,
    subtree_files: usize,
    subtree_dirs: usize,
}

/// One directory row of the skeleton.
#[derive(Clone)]
pub struct DirEntry {
    /// Relative path from the repo root, no trailing slash.
    pub path: String,
    /// 1 = top level under the root.
    pub depth: usize,
    /// Files directly inside this directory.
    pub files: usize,
    /// Files anywhere in this subtree.
    pub subtree_files: usize,
    /// Directories anywhere in this subtree.
    pub subtree_dirs: usize,
    /// Dominant extension among direct files, "" when the dir holds none.
    pub ext: String,
}

/// The full overview result, before the CLI picks a rendering.
pub struct Overview {
    pub root: String,
    pub path: String,
    pub total_files: usize,
    pub total_dirs: usize,
    pub root_files: usize,
    /// All extensions sorted by file count desc, then name asc.
    pub languages: Vec<(String, usize)>,
    /// Directories shown, in DFS path order (parents before descendants).
    pub dirs: Vec<DirEntry>,
    pub requested_depth: usize,
    /// Depth actually rendered after budget pruning.
    pub effective_depth: usize,
    /// Directories cut at depth 1 for the token budget (depth-cutoff dirs
    /// are not counted; the depth fields already describe that).
    pub omitted_dirs: usize,
    pub token_estimate: usize,
}

/// All non-ignored files under `root`, sorted. Same ignore semantics as
/// indexing, but without the extension filter and size cap: a structure
/// overview must count docs and config files too.
pub fn walk_all_files(root: &Path) -> Result<Vec<PathBuf>> {
    use ignore::WalkState;
    use std::sync::{Arc, Mutex};

    let start = std::time::Instant::now();
    let walker = crate::walker::walk_builder(root, false).build_parallel();
    let files: Arc<Mutex<Vec<PathBuf>>> = Arc::new(Mutex::new(Vec::new()));
    let files_ref = files.clone();
    let skip = crate::walker::should_skip_path;
    let ignore_patterns: Vec<String> = crate::config::DEFAULT_IGNORE_PATTERNS
        .iter()
        .map(|pattern| (*pattern).to_string())
        .collect();
    walker.run(move || {
        let files = files_ref.clone();
        let ignore_patterns = ignore_patterns.clone();
        Box::new(move |entry| {
            if let Ok(entry) = entry {
                if entry.file_type().is_some_and(|ft| ft.is_file())
                    && !skip(root, entry.path(), &ignore_patterns)
                {
                    files.lock().unwrap().push(entry.into_path());
                }
            }
            WalkState::Continue
        })
    });
    let mut files = files
        .lock()
        .map_err(|e| anyhow::anyhow!("Failed to get mutex: {}", e))?
        .clone();
    files.sort();
    info!(
        files_found = files.len(),
        elapsed_ms = start.elapsed().as_millis() as u64,
        "Overview walk completed"
    );
    Ok(files)
}

/// Walk, aggregate, and budget-prune in one call.
pub fn overview(
    root: &Path,
    requested_depth: usize,
    token_budget: usize,
    human: bool,
) -> Result<Overview> {
    let files = walk_all_files(root)?;
    Ok(build(root, &files, requested_depth, token_budget, human))
}

/// Aggregate a (sorted or unsorted) file list into the pruned skeleton.
fn build(
    root: &Path,
    files: &[PathBuf],
    requested_depth: usize,
    token_budget: usize,
    human: bool,
) -> Overview {
    let mut dirs: BTreeMap<String, DirStats> = BTreeMap::new();
    let mut languages: BTreeMap<String, usize> = BTreeMap::new();
    let mut root_files = 0usize;
    for file in files {
        let rel = file.strip_prefix(root).unwrap_or(file);
        let label = extension_label(rel);
        *languages.entry(label.clone()).or_insert(0) += 1;
        // The direct parent is "" for root-level files.
        let Some(parent) = rel.parent() else {
            root_files += 1;
            continue;
        };
        if parent.to_string_lossy().is_empty() {
            root_files += 1;
            continue;
        }
        let mut current = Some(parent);
        let mut direct = true;
        while let Some(dir) = current {
            let key = dir.to_string_lossy().to_string();
            if key.is_empty() {
                break;
            }
            let stats = dirs.entry(key).or_default();
            stats.subtree_files += 1;
            if direct {
                *stats.files.entry(label.clone()).or_insert(0) += 1;
                direct = false;
            }
            current = dir.parent();
        }
    }

    // Each directory contributes 1 to subtree_dirs of every proper ancestor.
    let keys: Vec<String> = dirs.keys().cloned().collect();
    for key in keys {
        let mut ancestor = Path::new(&key).parent();
        while let Some(dir) = ancestor {
            let key = dir.to_string_lossy().to_string();
            if key.is_empty() {
                break;
            }
            if let Some(stats) = dirs.get_mut(&key) {
                stats.subtree_dirs += 1;
            }
            ancestor = dir.parent();
        }
    }

    let mut entries: Vec<DirEntry> = dirs
        .iter()
        .map(|(path, stats)| {
            let files: usize = stats.files.values().sum();
            // BTreeMap iterates name asc; strictly-greater keeps the
            // lexicographically smallest name on count ties.
            let mut best = "";
            let mut best_count = 0;
            for (name, count) in &stats.files {
                if *count > best_count {
                    best = name;
                    best_count = *count;
                }
            }
            DirEntry {
                depth: path.matches('/').count() + 1,
                path: path.clone(),
                files,
                subtree_files: stats.subtree_files,
                subtree_dirs: stats.subtree_dirs,
                ext: best.to_string(),
            }
        })
        .collect();
    // BTreeMap iteration is path-sorted, which is DFS preorder ("a" before
    // "a/b" before "b"): parents are immediately followed by their subtree,
    // so the basename-only human lines read grouped like `tree` output.
    entries.sort_by(|a, b| a.path.cmp(&b.path));

    let mut languages: Vec<(String, usize)> = languages.into_iter().collect();
    languages.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));

    let total_files = files.len();
    let total_dirs = entries.len();
    let mut result = Overview {
        root: root
            .file_name()
            .map(|name| name.to_string_lossy().to_string())
            .unwrap_or_else(|| root.display().to_string()),
        path: root.display().to_string(),
        total_files,
        total_dirs,
        root_files,
        languages,
        dirs: Vec::new(),
        requested_depth,
        effective_depth: requested_depth,
        omitted_dirs: 0,
        token_estimate: 0,
    };

    let mut effective_depth = requested_depth;
    loop {
        result.dirs = entries
            .iter()
            .filter(|entry| entry.depth <= effective_depth)
            .cloned()
            .collect();
        result.effective_depth = effective_depth;
        result.token_estimate = token_estimate_of(&render(&result, human));
        if token_budget == 0 || result.token_estimate <= token_budget || effective_depth == 1 {
            break;
        }
        effective_depth -= 1;
    }

    if token_budget > 0 && result.token_estimate > token_budget {
        // Depth 1 still over budget: keep the largest dirs that fit. Rank by
        // subtree files desc (ties alphabetical), drop from the smallest.
        let mut ranked = result.dirs.clone();
        ranked.sort_by(|a, b| {
            b.subtree_files
                .cmp(&a.subtree_files)
                .then(a.path.cmp(&b.path))
        });
        while ranked.len() > 1 {
            ranked.pop();
            let mut candidate = ranked.clone();
            candidate.sort_by(|a, b| a.path.cmp(&b.path));
            result.dirs = candidate;
            result.token_estimate = token_estimate_of(&render(&result, human));
            if result.token_estimate <= token_budget {
                break;
            }
        }
        let depth_one_total = entries.iter().filter(|entry| entry.depth == 1).count();
        result.omitted_dirs = depth_one_total - result.dirs.len();
        result.token_estimate = token_estimate_of(&render(&result, human));
    }

    result
}

/// Lowercased extension, or the lowercased filename for extensionless
/// files (dockerfile, makefile, ...) so they still group in the stats.
fn extension_label(rel: &Path) -> String {
    rel.extension()
        .map(|ext| ext.to_string_lossy().to_lowercase())
        .unwrap_or_else(|| {
            rel.file_name()
                .map(|name| name.to_string_lossy().to_lowercase())
                .unwrap_or_default()
        })
}

/// Tokens estimated as bytes/4 rounded up, matching the usage telemetry
/// methodology.
fn token_estimate_of(text: &str) -> usize {
    text.len().div_ceil(4)
}

fn render(overview: &Overview, human: bool) -> String {
    if human {
        render_human(overview)
    } else {
        render_json(overview)
    }
}

/// Compact JSON rendering, one object per invocation like every other verb.
pub fn render_json(overview: &Overview) -> String {
    let dirs: Vec<serde_json::Value> = overview
        .dirs
        .iter()
        .map(|d| {
            json!({
                "path": d.path,
                "depth": d.depth,
                "files": d.files,
                "subtree_files": d.subtree_files,
                "subtree_dirs": d.subtree_dirs,
                "ext": d.ext,
            })
        })
        .collect();
    let languages: Vec<serde_json::Value> = overview
        .languages
        .iter()
        .take(LANG_JSON_TOP)
        .map(|(name, count)| json!({ "ext": name, "files": count }))
        .collect();
    json!({
        "root": overview.root,
        "path": overview.path,
        "total_files": overview.total_files,
        "total_dirs": overview.total_dirs,
        "root_files": overview.root_files,
        "languages": languages,
        "requested_depth": overview.requested_depth,
        "depth": overview.effective_depth,
        "omitted_dirs": overview.omitted_dirs,
        "token_estimate": overview.token_estimate,
        "dirs": dirs,
    })
    .to_string()
}

/// Indented two spaces per level, no box-drawing art: the cheapest
/// readable skeleton for pasting straight into an agent context.
pub fn render_human(overview: &Overview) -> String {
    let mut languages: Vec<String> = overview
        .languages
        .iter()
        .take(LANG_HUMAN_TOP)
        .map(|(name, count)| format!("{name} {count}"))
        .collect();
    if overview.languages.len() > LANG_HUMAN_TOP {
        languages.push(format!("+{}", overview.languages.len() - LANG_HUMAN_TOP));
    }
    let mut out = format!(
        "{} — {} files, {} dirs",
        overview.root, overview.total_files, overview.total_dirs
    );
    if !languages.is_empty() {
        out.push_str(&format!(" — {}", languages.join(" · ")));
    }
    for d in &overview.dirs {
        let name = d.path.rsplit('/').next().unwrap_or(&d.path);
        let ext = if d.ext.is_empty() {
            String::new()
        } else {
            format!(" ({})", d.ext)
        };
        let subtree = if d.subtree_files > d.files {
            format!(" · {} in subtree", d.subtree_files)
        } else {
            String::new()
        };
        out.push_str(&format!(
            "\n{}{}/ {} {}{}",
            "  ".repeat(d.depth),
            name,
            d.files,
            if d.files == 1 { "file" } else { "files" },
            format!("{ext}{subtree}")
        ));
    }
    if overview.omitted_dirs > 0 {
        out.push_str(&format!(
            "\n(+{} dirs omitted for token budget)",
            overview.omitted_dirs
        ));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn paths(root: &str, rels: &[&str]) -> Vec<PathBuf> {
        rels.iter().map(|rel| Path::new(root).join(rel)).collect()
    }

    #[test]
    fn totals_subtrees_and_dominant_extension() {
        let files = paths(
            "/r",
            &[
                "src/a.rs",
                "src/b.rs",
                "src/deep/c.rs",
                "README.md",
                "docs/x.md",
                "docs/y.md",
            ],
        );
        let overview = build(Path::new("/r"), &files, 9, 0, false);
        assert_eq!(overview.total_files, 6);
        assert_eq!(overview.total_dirs, 3);
        assert_eq!(overview.root_files, 1);
        let src = overview.dirs.iter().find(|d| d.path == "src").unwrap();
        assert_eq!((src.files, src.subtree_files, src.subtree_dirs), (2, 3, 1));
        assert_eq!(src.ext, "rs");
        let deep = overview.dirs.iter().find(|d| d.path == "src/deep").unwrap();
        assert_eq!(deep.depth, 2);
        assert_eq!((deep.files, deep.subtree_files), (1, 1));
        // rs:3 and md:3 tie on count; name asc puts md first.
        assert_eq!(overview.languages[0], ("md".to_string(), 3));
    }

    #[test]
    fn extensionless_files_group_by_filename() {
        let files = paths("/r", &["Makefile", "Dockerfile", "src/m.rs"]);
        let overview = build(Path::new("/r"), &files, 2, 0, false);
        assert_eq!(overview.root_files, 2);
        let names: Vec<&str> = overview.languages.iter().map(|(n, _)| n.as_str()).collect();
        assert_eq!(names, ["dockerfile", "makefile", "rs"]);
    }

    #[test]
    fn budget_pruning_is_deterministic_and_bounded() {
        let mut rels = Vec::new();
        for i in 0..40 {
            rels.push(format!("pkg{i}/mod.rs"));
            rels.push(format!("pkg{i}/src/deep/inner/file{i}.rs"));
        }
        let files: Vec<PathBuf> = rels.iter().map(|rel| Path::new("/r").join(rel)).collect();
        let first = build(Path::new("/r"), &files, 3, 400, false);
        let second = build(Path::new("/r"), &files, 3, 400, false);
        assert_eq!(render_json(&first), render_json(&second));
        assert!(
            first.token_estimate <= 400 || first.effective_depth == 1,
            "estimate {} depth {}",
            first.token_estimate,
            first.effective_depth
        );
        assert!(first.effective_depth < 3 || first.omitted_dirs > 0);
    }

    #[test]
    fn human_render_shows_counts_and_subtrees() {
        let files = paths("/r", &["src/a.rs", "src/deep/b.rs", "top.md"]);
        let overview = build(Path::new("/r"), &files, 2, 0, true);
        let text = render_human(&overview);
        assert!(text.contains("3 files, 2 dirs"), "{text}");
        assert!(text.contains("src/ 1 file (rs) · 2 in subtree"), "{text}");
        assert!(text.contains("  deep/ 1 file (rs)"), "{text}");
    }

    #[test]
    fn empty_tree_renders() {
        let overview = build(Path::new("/r"), &[], 2, 1200, false);
        assert_eq!(overview.total_files, 0);
        let text = render_human(&overview);
        assert!(text.contains("0 files, 0 dirs"), "{text}");
    }
}
