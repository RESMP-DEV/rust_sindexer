//! File walker for discovering code files in a codebase.

use std::path::{Path, PathBuf};
use std::time::Instant;

use anyhow::Result;
use tracing::{debug, info, instrument};

use crate::config::{Config, DEFAULT_IGNORE_PATTERNS, EXTENSIONLESS_FILES, SUPPORTED_EXTENSIONS};

/// Walks a codebase to discover indexable files.
///
/// Uses the `ignore` crate internally for parallel, gitignore-aware traversal.
pub struct CodeWalker {
    /// File extensions to include (e.g., ["rs", "py", "ts"]).
    pub extensions: Vec<String>,
    /// Additional patterns to ignore beyond .gitignore.
    pub ignore_patterns: Vec<String>,
    /// Maximum file size in bytes to index (0 = unlimited).
    pub max_file_size: u64,
    /// Whether traversal follows symlinks.
    pub follow_symlinks: bool,
}

impl CodeWalker {
    /// Create a new walker with default settings.
    pub fn new() -> Self {
        Self::from_config(&Config::default())
    }

    /// Create a walker from application configuration.
    pub fn from_config(config: &Config) -> Self {
        Self {
            extensions: SUPPORTED_EXTENSIONS
                .iter()
                .map(|s| (*s).to_string())
                .collect(),
            ignore_patterns: DEFAULT_IGNORE_PATTERNS
                .iter()
                .map(|s| (*s).to_string())
                .collect(),
            max_file_size: config.max_file_size,
            follow_symlinks: config.follow_symlinks,
        }
    }

    /// Walk the codebase at the given path and return all indexable files.
    ///
    /// This runs in parallel using the ignore crate's parallel walker.
    #[instrument(skip(self), fields(path = %path.display()))]
    pub async fn walk(&self, path: &Path) -> Result<Vec<PathBuf>> {
        use ignore::WalkBuilder;
        use std::sync::Mutex;

        let start = Instant::now();
        debug!(
            extensions = self.extensions.len(),
            ignore_patterns = self.ignore_patterns.len(),
            "Starting file walk"
        );
        let files = std::sync::Arc::new(Mutex::new(Vec::new()));
        let extensions = self.extensions.clone();
        let mut builder = WalkBuilder::new(path);

        builder
            .hidden(true)
            .follow_links(self.follow_symlinks)
            .git_ignore(true)
            .git_global(true)
            .git_exclude(true);
        builder.add_custom_ignore_filename(".contextignore");

        if let Some(home) = std::env::var_os("HOME") {
            let global_ignore = std::path::PathBuf::from(home)
                .join(".context")
                .join(".contextignore");
            if global_ignore.exists() {
                builder.add_ignore(global_ignore);
            }
        }

        let walker = builder.build_parallel();

        let files_ref = files.clone();
        walker.run(|| {
            let files = files_ref.clone();
            let extensions = extensions.clone();
            let ignore_patterns = self.ignore_patterns.clone();
            let max_file_size = self.max_file_size;
            Box::new(move |entry| {
                if let Ok(entry) = entry {
                    if entry.file_type().is_some_and(|ft| ft.is_file()) {
                        if should_skip_path(path, entry.path(), &ignore_patterns) {
                            return ignore::WalkState::Continue;
                        }
                        if max_file_size > 0 {
                            match entry.metadata() {
                                Ok(metadata) if metadata.len() > max_file_size => {
                                    debug!(
                                        path = %entry.path().display(),
                                        size = metadata.len(),
                                        max_file_size,
                                        "Skipping oversized file"
                                    );
                                    return ignore::WalkState::Continue;
                                }
                                Err(err) => {
                                    debug!(
                                        path = %entry.path().display(),
                                        error = %err,
                                        "Skipping file with unreadable metadata"
                                    );
                                    return ignore::WalkState::Continue;
                                }
                                _ => {}
                            }
                        }
                        let include = if let Some(ext) = entry.path().extension() {
                            extensions
                                .iter()
                                .any(|e| e == ext.to_string_lossy().as_ref())
                        } else if let Some(name) = entry.path().file_name() {
                            let name_lower = name.to_string_lossy().to_lowercase();
                            EXTENSIONLESS_FILES.contains(&name_lower.as_str())
                        } else {
                            false
                        };
                        if include {
                            files.lock().unwrap().push(entry.into_path());
                        }
                    }
                }
                ignore::WalkState::Continue
            })
        });

        let result = files
            .lock()
            .map_err(|e| anyhow::anyhow!("Failed to get mutex: {}", e))?
            .clone();

        info!(
            files_found = result.len(),
            elapsed_ms = start.elapsed().as_millis() as u64,
            "File walk completed"
        );
        Ok(result)
    }
}

fn should_skip_path(root: &Path, path: &Path, ignore_patterns: &[String]) -> bool {
    let relative = path.strip_prefix(root).unwrap_or(path);
    let components = relative
        .components()
        .filter_map(|component| component.as_os_str().to_str())
        .collect::<Vec<_>>();

    ignore_patterns.iter().any(|pattern| {
        let pattern = pattern.trim().trim_matches('/');
        !pattern.is_empty()
            && !pattern.contains('/')
            && !pattern.contains('*')
            && components.contains(&pattern)
    })
}

impl Default for CodeWalker {
    fn default() -> Self {
        Self::new()
    }
}
