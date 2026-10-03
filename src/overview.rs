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

use std::collections::{BTreeMap, BTreeSet};
use std::ffi::OsString;
use std::path::{Path, PathBuf};

use anyhow::Result;
use serde_json::json;
use tracing::{info, warn};

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
#[derive(Clone)]
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
    /// Deepest directory discovered before budget pruning.
    pub tree_depth: usize,
    /// All directories not shown because of the requested depth or budget.
    pub omitted_dirs: usize,
    /// Walk entries that could not be read; zero means the walk is complete.
    pub walk_errors: usize,
    /// True when even the smallest supported rendering exceeds the budget.
    pub over_budget: bool,
    /// Budget requested by the caller; zero disables pruning.
    pub token_budget: usize,
    pub token_estimate: usize,
}

/// Results of the live walk used to build an overview.
#[derive(Default, Clone)]
pub struct WalkedTree {
    pub files: Vec<PathBuf>,
    pub dirs: Vec<PathBuf>,
    pub walk_errors: usize,
}

/// All non-ignored files and directories under `root`, sorted. The walk has
/// the same ignore and symlink semantics as indexing, but no extension filter
/// or size cap: a structure overview must also show docs, config files, and
/// empty scaffolding directories.
pub fn walk_tree(root: &Path, follow_links: bool) -> Result<WalkedTree> {
    use ignore::WalkState;
    use std::sync::{Arc, Mutex};

    let start = std::time::Instant::now();
    let walker = crate::walker::walk_builder(root, follow_links).build_parallel();
    let tree: Arc<Mutex<WalkedTree>> = Arc::new(Mutex::new(WalkedTree::default()));
    let tree_ref = tree.clone();
    let skip = crate::walker::should_skip_path;
    let ignore_patterns: Vec<String> = crate::config::DEFAULT_IGNORE_PATTERNS
        .iter()
        .map(|pattern| (*pattern).to_string())
        .collect();
    walker.run(move || {
        let tree = tree_ref.clone();
        let ignore_patterns = ignore_patterns.clone();
        Box::new(move |entry| {
            match entry {
                Ok(entry) => {
                    if skip(root, entry.path(), &ignore_patterns) {
                        return WalkState::Continue;
                    }
                    let Some(file_type) = entry.file_type() else {
                        return WalkState::Continue;
                    };
                    if !follow_links && file_type.is_symlink() {
                        return WalkState::Continue;
                    }
                    let mut tree = match tree.lock() {
                        Ok(tree) => tree,
                        Err(poisoned) => poisoned.into_inner(),
                    };
                    if file_type.is_file() {
                        tree.files.push(entry.into_path());
                    } else if file_type.is_dir() && entry.path() != root {
                        tree.dirs.push(entry.into_path());
                    }
                }
                Err(_) => {
                    let mut tree = match tree.lock() {
                        Ok(tree) => tree,
                        Err(poisoned) => poisoned.into_inner(),
                    };
                    tree.walk_errors += 1;
                }
            }
            WalkState::Continue
        })
    });
    let mut walked = match tree.lock() {
        Ok(walked) => walked.clone(),
        Err(poisoned) => poisoned.into_inner().clone(),
    };
    walked.files.sort();
    walked.dirs.sort();
    if walked.walk_errors > 0 {
        warn!(
            walk_errors = walked.walk_errors,
            files_found = walked.files.len(),
            dirs_found = walked.dirs.len(),
            "Overview walk was incomplete"
        );
    }
    info!(
        files_found = walked.files.len(),
        dirs_found = walked.dirs.len(),
        walk_errors = walked.walk_errors,
        elapsed_ms = start.elapsed().as_millis() as u64,
        "Overview walk completed"
    );
    Ok(walked)
}

/// Walk, aggregate, and budget-prune in one call.
pub fn overview(
    root: &Path,
    requested_depth: usize,
    token_budget: usize,
    human: bool,
    follow_links: bool,
) -> Result<Overview> {
    let tree = walk_tree(root, follow_links)?;
    Ok(build_tree(
        root,
        &tree.files,
        &tree.dirs,
        tree.walk_errors,
        requested_depth,
        token_budget,
        human,
    ))
}

/// Aggregate a (sorted or unsorted) file list into the pruned skeleton.
fn build_tree(
    root: &Path,
    files: &[PathBuf],
    walked_dirs: &[PathBuf],
    walk_errors: usize,
    requested_depth: usize,
    token_budget: usize,
    human: bool,
) -> Overview {
    let mut dirs: BTreeMap<Vec<OsString>, DirStats> = BTreeMap::new();
    let mut languages: BTreeMap<String, usize> = BTreeMap::new();
    let mut root_files = 0usize;
    for file in files {
        let components = relative_components(file, root);
        let rel: PathBuf = components.iter().collect();
        let label = extension_label(&rel);
        *languages.entry(label.clone()).or_insert(0) += 1;
        if components.len() <= 1 {
            root_files += 1;
            continue;
        }
        let parent = &components[..components.len() - 1];
        let mut current = parent;
        let mut direct = true;
        while !current.is_empty() {
            let key = current.to_vec();
            let stats = dirs.entry(key).or_default();
            stats.subtree_files += 1;
            if direct {
                *stats.files.entry(label.clone()).or_insert(0) += 1;
                direct = false;
            }
            current = &current[..current.len() - 1];
        }
    }

    // Empty directories are structural content too. Ensure every walked dir
    // and every ancestor of a file exists before counting subtree_dirs.
    let mut known_dirs: BTreeSet<Vec<OsString>> = dirs.keys().cloned().collect();
    for dir in walked_dirs {
        let components = relative_components(dir, root);
        for depth in 1..=components.len() {
            known_dirs.insert(components[..depth].to_vec());
        }
    }
    for key in known_dirs {
        dirs.entry(key).or_default();
    }

    // Each directory contributes 1 to subtree_dirs of every proper ancestor.
    let keys: Vec<Vec<OsString>> = dirs.keys().cloned().collect();
    for key in keys {
        for depth in 1..key.len() {
            let ancestor = key[..depth].to_vec();
            if let Some(stats) = dirs.get_mut(&ancestor) {
                stats.subtree_dirs += 1;
            }
        }
    }

    let entries: Vec<DirEntry> = dirs
        .iter()
        .map(|(key, stats)| {
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
                depth: key.len(),
                path: public_path(key),
                files,
                subtree_files: stats.subtree_files,
                subtree_dirs: stats.subtree_dirs,
                ext: best.to_string(),
            }
        })
        .collect();
    // Vec<OsString> has component-wise order, unlike a joined string where
    // `src-tauri` sorts between `src` and `src/bin`. The BTreeMap iteration
    // therefore gives the DFS preorder required by the human renderer.

    let mut languages: Vec<(String, usize)> = languages.into_iter().collect();
    languages.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));

    let total_files = files.len();
    let total_dirs = entries.len();
    let tree_depth = entries.iter().map(|entry| entry.depth).max().unwrap_or(1);
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
        tree_depth,
        effective_depth: requested_depth,
        omitted_dirs: 0,
        walk_errors,
        over_budget: false,
        token_budget,
        token_estimate: 0,
    };

    // Clamp once to the tree's real height: the filter cannot change above
    // it, and depth 1 is the public library floor (CLI also validates >= 1).
    result.effective_depth = requested_depth.clamp(1, tree_depth);
    result.dirs = entries
        .iter()
        .filter(|entry| entry.depth <= result.effective_depth)
        .cloned()
        .collect();
    result.omitted_dirs = total_dirs - result.dirs.len();
    if token_budget == 0 {
        measured(&mut result, human);
        return result;
    }

    // Render size is monotonic in depth, so binary-search the smallest depth
    // that fits. This bounds pruning by log(tree depth), even when a caller
    // passes usize::MAX or a pathological tree has thousands of levels.
    let mut fitting_depth = None;
    let mut low = 1;
    let mut high = result.effective_depth;
    while low <= high {
        let depth = (low + high) / 2;
        let mut candidate = result.clone();
        candidate.dirs = entries
            .iter()
            .filter(|entry| entry.depth <= depth)
            .cloned()
            .collect();
        candidate.effective_depth = depth;
        candidate.omitted_dirs = total_dirs - candidate.dirs.len();
        measured(&mut candidate, human);
        if candidate.token_estimate <= token_budget {
            fitting_depth = Some(depth);
            high = depth.saturating_sub(1);
        } else {
            low = depth + 1;
        }
    }
    result.effective_depth = fitting_depth.unwrap_or(1);
    result.dirs = entries
        .iter()
        .filter(|entry| entry.depth <= result.effective_depth)
        .cloned()
        .collect();
    result.omitted_dirs = total_dirs - result.dirs.len();
    measured(&mut result, human);
    if !result.over_budget {
        return result;
    }

    // Depth 1 is still over budget: keep the largest possible prefix of the
    // subtree-size ranking. Render size is monotonic in keep-count, so binary
    // search instead of cloning/rendering once per top-level directory.
    let mut ranked = result.dirs.clone();
    ranked.sort_by(|a, b| {
        b.subtree_files
            .cmp(&a.subtree_files)
            .then(a.path.cmp(&b.path))
    });
    let depth_one_total = entries.iter().filter(|entry| entry.depth == 1).count();
    if depth_one_total == 0 {
        return result;
    }
    let deeper_omitted = total_dirs - depth_one_total;
    let mut fitting_count = None;
    let mut low = 1;
    let mut high = ranked.len().saturating_sub(1);
    while low <= high {
        let keep = (low + high) / 2;
        let mut candidate = result.clone();
        candidate.dirs = ranked[..keep].to_vec();
        candidate.dirs.sort_by(|a, b| a.path.cmp(&b.path));
        candidate.omitted_dirs = deeper_omitted + depth_one_total - keep;
        measured(&mut candidate, human);
        if candidate.token_estimate <= token_budget {
            fitting_count = Some(keep);
            high = keep.saturating_sub(1);
        } else {
            low = keep + 1;
        }
    }
    let keep = fitting_count.unwrap_or(1);
    result.dirs = ranked[..keep].to_vec();
    result.dirs.sort_by(|a, b| a.path.cmp(&b.path));
    result.omitted_dirs = deeper_omitted + depth_one_total - keep;
    measured(&mut result, human);

    result
}

fn relative_components(path: &Path, root: &Path) -> Vec<OsString> {
    let relative = path.strip_prefix(root).unwrap_or(path);
    relative
        .components()
        .filter(|component| {
            !matches!(
                component,
                std::path::Component::RootDir | std::path::Component::CurDir
            )
        })
        .map(|component| component.as_os_str().to_os_string())
        .collect()
}

fn public_path(components: &[OsString]) -> String {
    components
        .iter()
        .map(|component| component.to_string_lossy())
        .collect::<Vec<_>>()
        .join("/")
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

/// Set the estimate and over-budget status for the exact bytes that will be
/// rendered. JSON's embedded estimate is part of its own payload, so iterate
/// to the fixed point rather than measuring a smaller pre-trailer rendering.
fn measured(overview: &mut Overview, human: bool) -> usize {
    overview.token_estimate = 0;
    overview.over_budget = overview.token_budget > 0;
    loop {
        let rendered = render(overview, human);
        let next_estimate = token_estimate_of(&rendered);
        let next_over_budget = overview.token_budget > 0 && next_estimate > overview.token_budget;
        if next_estimate == overview.token_estimate && next_over_budget == overview.over_budget {
            return next_estimate;
        }
        overview.token_estimate = next_estimate;
        overview.over_budget = next_over_budget;
    }
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
        "languages_total": overview.languages.len(),
        "languages_omitted": overview.languages.len().saturating_sub(LANG_JSON_TOP),
        "requested_depth": overview.requested_depth,
        "depth": overview.effective_depth,
        "omitted_dirs": overview.omitted_dirs,
        "walk_errors": overview.walk_errors,
        "over_budget": overview.over_budget,
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
    if overview.effective_depth < overview.tree_depth {
        out.push_str(&format!(
            " — depth {} of {}",
            overview.effective_depth, overview.tree_depth
        ));
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
            "\n{}{}/ {} {}{}{}",
            "  ".repeat(d.depth),
            name,
            d.files,
            if d.files == 1 { "file" } else { "files" },
            ext,
            subtree
        ));
    }
    if overview.omitted_dirs > 0 {
        out.push_str(&format!(
            "\n(+{} dirs omitted by depth/token budget)",
            overview.omitted_dirs
        ));
    }
    if overview.walk_errors > 0 {
        out.push_str(&format!(
            " ({} walk entries skipped; output is incomplete)",
            overview.walk_errors
        ));
    }
    if overview.over_budget {
        out.push_str(&format!(
            " (! over budget: {} estimated tokens exceeds {})",
            overview.token_estimate, overview.token_budget
        ));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn build(
        root: &Path,
        files: &[PathBuf],
        requested_depth: usize,
        token_budget: usize,
        human: bool,
    ) -> Overview {
        build_tree(root, files, &[], 0, requested_depth, token_budget, human)
    }

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
        let rendered = render_json(&first);
        assert_eq!(first.token_estimate, rendered.len().div_ceil(4));
        assert_eq!(first.over_budget, first.token_estimate > 400);
        if !first.over_budget {
            assert!(first.token_estimate <= 400);
        }
        assert!(first.effective_depth < 3 || first.omitted_dirs > 0);
    }

    #[test]
    fn component_order_is_true_dfs_preorder() {
        let files = paths(
            "/r",
            &["src/bin/app.rs", "src/main.rs", "src-tauri/config.json"],
        );
        let overview = build(Path::new("/r"), &files, 3, 0, true);
        let paths: Vec<&str> = overview
            .dirs
            .iter()
            .map(|entry| entry.path.as_str())
            .collect();
        assert_eq!(paths, ["src", "src/bin", "src-tauri"]);
        let rendered = render_human(&overview);
        assert!(rendered.contains("\n    bin/"), "{rendered}");
    }

    #[test]
    fn empty_directories_and_walk_errors_are_counted() {
        let empty = Path::new("/r").join("empty");
        let files = paths("/r", &["src/a.rs"]);
        let overview = build_tree(Path::new("/r"), &files, &[empty], 2, 9, 0, true);
        assert_eq!(overview.total_dirs, 2);
        assert_eq!(overview.walk_errors, 2);
        assert!(render_human(&overview).contains("2 walk entries skipped"));
        assert!(render_json(&overview).contains(r#""walk_errors":2"#));
    }

    #[test]
    fn language_truncation_is_disclosed_in_both_modes() {
        let rels: Vec<String> = (0..9).map(|i| format!("file.{i}")).collect();
        let refs: Vec<&str> = rels.iter().map(String::as_str).collect();
        let files = paths("/r", &refs);
        let overview = build(Path::new("/r"), &files, 1, 0, false);
        let json = render_json(&overview);
        assert!(json.contains(r#""languages_total":9"#), "{json}");
        assert!(json.contains(r#""languages_omitted":3"#), "{json}");
        let human = render_human(&overview);
        assert!(human.ends_with(" +4"), "{human}");
    }

    #[test]
    fn top_level_pruning_keeps_largest_and_measures_the_trailer() {
        let mut rels = Vec::new();
        for i in 0..12 {
            for file in 0..=i {
                rels.push(format!("pkg{i:02}/file{file}.rs"));
            }
            rels.push(format!("pkg{i:02}/deep/inner/file.rs"));
        }
        let files: Vec<PathBuf> = rels.iter().map(|rel| Path::new("/r").join(rel)).collect();
        let overview = build(Path::new("/r"), &files, 2, 180, true);
        let rendered = render_human(&overview);
        assert_eq!(overview.token_estimate, rendered.len().div_ceil(4));
        assert!(rendered.contains("(+"), "{rendered}");
        assert_eq!(
            overview.omitted_dirs,
            overview.total_dirs - overview.dirs.len()
        );
        let retained: Vec<usize> = overview
            .dirs
            .iter()
            .map(|entry| entry.path.trim_start_matches("pkg").parse().unwrap())
            .collect();
        assert_eq!(retained, (12 - retained.len()..12).collect::<Vec<_>>());
    }

    #[test]
    fn depth_requests_are_clamped_to_the_real_tree_height() {
        let files = paths("/r", &["a/b/c/file.rs"]);
        let overview = build(Path::new("/r"), &files, usize::MAX, 1, true);
        assert_eq!(overview.tree_depth, 3);
        assert_eq!(overview.effective_depth, 1);
        assert_eq!(overview.omitted_dirs, 2);
        assert!(overview.over_budget);
    }

    #[test]
    fn empty_tree_with_tiny_budget_is_over_budget_not_empty() {
        let overview = build(Path::new("/r"), &[], 1, 1, true);
        assert!(overview.dirs.is_empty());
        assert_eq!(overview.omitted_dirs, 0);
        assert!(overview.over_budget);
        assert_eq!(
            render_human(&overview).len().div_ceil(4),
            overview.token_estimate
        );
    }

    #[cfg(unix)]
    #[test]
    fn non_utf8_directory_keys_do_not_merge_counts() {
        use std::ffi::OsStr;
        use std::os::unix::ffi::OsStrExt;

        let first_name = OsStr::from_bytes(b"bad\xe9");
        let second_name = OsStr::from_bytes(b"bad\xea");
        let root = Path::new("/r");
        let files = vec![
            root.join(first_name).join("a.rs"),
            root.join(second_name).join("b.rs"),
        ];
        let overview = build(root, &files, 2, 0, false);
        assert_eq!(overview.total_files, 2);
        assert_eq!(overview.total_dirs, 2);
        let lossy = "bad\u{FFFD}";
        let lossy_rows = overview
            .dirs
            .iter()
            .filter(|entry| entry.path == lossy)
            .count();
        let lossy_files = overview
            .dirs
            .iter()
            .filter(|entry| entry.path == lossy)
            .map(|entry| entry.files)
            .sum::<usize>();
        // The public String renderer necessarily lossy-collapses both names,
        // but the component keys keep their counts and rows distinct.
        assert_eq!(lossy_rows, 2);
        assert_eq!(lossy_files, 2);
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
