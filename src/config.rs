use std::collections::HashSet;
use std::env;
use tracing::{debug, info};

const EMBEDDING_URL_ENV_KEYS: &[&str] = &["EMBEDDING_URL", "OPENAI_BASE_URL"];
const EMBEDDING_API_KEY_ENV_KEYS: &[&str] = &["EMBEDDING_API_KEY", "OPENAI_API_KEY"];
const MILVUS_URL_ENV_KEYS: &[&str] = &["MILVUS_URL", "MILVUS_ADDRESS"];

/// Supported file extensions for indexing (without leading dot).
pub const SUPPORTED_EXTENSIONS: &[&str] = &[
    "py", "rs", "js", "ts", "tsx", "jsx", "go", "java", "cpp", "cc", "cxx", "c", "h", "hpp", "rb",
    "php", "swift", "kt", "scala", "cs", "fs", "ml", "mli", "hs", "lua", "sh", "bash", "zsh", "pl",
    "pm", "r", "jl", "ex", "exs", "erl", "hrl", "clj", "cljs", "lisp", "el", "vim", "sql",
    "graphql", "proto", "thrift", "yaml", "yml", "toml", "json", "xml", "html", "css", "scss",
    "sass", "less", "md", "rst", "tex",
];

/// Extensionless files to include.
pub const EXTENSIONLESS_FILES: &[&str] = &["dockerfile", "makefile", "justfile", "rakefile"];

/// Default patterns to ignore during directory traversal.
pub const DEFAULT_IGNORE_PATTERNS: &[&str] = &[
    "node_modules",
    ".git",
    ".hg",
    ".svn",
    ".sindexer",
    ".rust_sindexer",
    ".worktrees",
    "agent_workspace",
    "logs",
    "target",
    "dist",
    "build",
    "__pycache__",
    ".pytest_cache",
    ".venv",
    "venv",
    ".tox",
    ".eggs",
];

/// Application configuration loaded from environment variables with sensible defaults.
#[derive(Debug, Clone)]
pub struct Config {
    /// URL for the embedding service. Empty disables semantic search.
    pub embedding_url: String,
    /// Model name for embeddings.
    pub embedding_model: String,
    /// Optional API key for embedding providers that require bearer auth.
    pub embedding_api_key: Option<String>,
    /// URL for Milvus vector database. Empty uses the local vector store.
    pub milvus_url: String,
    /// Optional bearer token for authenticated Milvus-compatible endpoints.
    pub milvus_token: Option<String>,
    /// Size of text chunks in characters.
    pub chunk_size: usize,
    /// Overlap between adjacent chunks in characters.
    pub chunk_overlap: usize,
    /// Number of items per batch for bulk operations.
    pub batch_size: usize,
    /// Maximum concurrent operations.
    pub concurrency: usize,
    /// Maximum file size in bytes to process (0 = unlimited).
    pub max_file_size: u64,
    /// Whether to follow symbolic links during traversal.
    pub follow_symlinks: bool,
    /// Number of threads for parallel operations (0 = auto-detect).
    pub parallelism: usize,
    /// Embedding vector dimension.
    pub embedding_dimension: usize,
    /// Embedding API RPM limit (0 = unlimited).
    pub embedding_rpm: u32,
    /// Embedding API TPM limit (0 = unlimited).
    pub embedding_tpm: u64,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            embedding_url: String::new(),
            embedding_model: "all-minilm".to_string(),
            embedding_api_key: None,
            milvus_url: String::new(),
            milvus_token: None,
            chunk_size: 512,
            chunk_overlap: 64,
            batch_size: 32,
            concurrency: 32,
            max_file_size: 1024 * 1024, // 1 MB
            follow_symlinks: false,
            parallelism: 0,
            embedding_dimension: 384,
            embedding_rpm: 400,
            embedding_tpm: 1_600_000,
        }
    }
}

impl Config {
    /// Load configuration from environment variables, falling back to defaults.
    pub fn from_env() -> Self {
        let defaults = Self::default();

        let config = Self {
            embedding_url: first_non_empty_env(EMBEDDING_URL_ENV_KEYS)
                .unwrap_or(defaults.embedding_url),
            embedding_model: first_non_empty_env(&["EMBEDDING_MODEL"])
                .unwrap_or(defaults.embedding_model),
            embedding_api_key: first_non_empty_env(EMBEDDING_API_KEY_ENV_KEYS),
            milvus_url: first_non_empty_env(MILVUS_URL_ENV_KEYS).unwrap_or(defaults.milvus_url),
            milvus_token: first_non_empty_env(&["MILVUS_TOKEN"]),
            chunk_size: env::var("CHUNK_SIZE")
                .ok()
                .and_then(|s| s.parse().ok())
                .unwrap_or(defaults.chunk_size),
            chunk_overlap: env::var("CHUNK_OVERLAP")
                .ok()
                .and_then(|s| s.parse().ok())
                .unwrap_or(defaults.chunk_overlap),
            batch_size: env::var("BATCH_SIZE")
                .ok()
                .and_then(|s| s.parse().ok())
                .unwrap_or(defaults.batch_size),
            concurrency: env::var("INDEXING_CONCURRENCY")
                .or_else(|_| env::var("CONCURRENCY"))
                .ok()
                .and_then(|s| s.parse().ok())
                .unwrap_or(defaults.concurrency),
            max_file_size: env::var("MAX_FILE_SIZE")
                .ok()
                .and_then(|s| s.parse().ok())
                .unwrap_or(defaults.max_file_size),
            follow_symlinks: env::var("FOLLOW_SYMLINKS")
                .map(|s| s == "1" || s.eq_ignore_ascii_case("true"))
                .unwrap_or(defaults.follow_symlinks),
            parallelism: env::var("PARALLELISM")
                .ok()
                .and_then(|s| s.parse().ok())
                .unwrap_or(defaults.parallelism),
            embedding_dimension: env::var("EMBEDDING_DIMENSION")
                .ok()
                .and_then(|s| s.parse().ok())
                .unwrap_or(defaults.embedding_dimension),
            embedding_rpm: env::var("EMBEDDING_RPM")
                .ok()
                .and_then(|s| s.parse().ok())
                .unwrap_or(defaults.embedding_rpm),
            embedding_tpm: env::var("EMBEDDING_TPM")
                .ok()
                .and_then(|s| s.parse().ok())
                .unwrap_or(defaults.embedding_tpm),
        };
        info!(
            embedder = if config.has_embedding_url() { "http" } else { "disabled" },
            embedding_url = if config.has_embedding_url() {
                config.embedding_url.as_str()
            } else {
                "disabled"
            },
            embedding_model = %config.embedding_model,
            vector_store = if config.has_milvus_url() { "milvus" } else { "local" },
            milvus_url = if config.has_milvus_url() {
                config.milvus_url.as_str()
            } else {
                "local"
            },
            chunk_size = config.chunk_size,
            batch_size = config.batch_size,
            concurrency = config.concurrency,
            embedding_dimension = config.embedding_dimension,
            "configuration loaded from environment"
        );
        config
    }

    #[inline]
    pub fn has_embedding_url(&self) -> bool {
        !self.embedding_url.is_empty()
    }

    #[inline]
    pub fn has_milvus_url(&self) -> bool {
        !self.milvus_url.is_empty()
    }

    /// Get effective thread count for parallel operations.
    #[inline]
    pub fn thread_count(&self) -> usize {
        let count = if self.parallelism == 0 {
            std::thread::available_parallelism()
                .map(|p| p.get())
                .unwrap_or(4)
        } else {
            self.parallelism
        };
        debug!(thread_count = count, "effective parallelism");
        count
    }

    /// Get supported extensions as a HashSet for O(1) lookup.
    pub fn extension_set(&self) -> HashSet<&'static str> {
        SUPPORTED_EXTENSIONS.iter().copied().collect()
    }

    /// Get extensionless files as a HashSet for O(1) lookup.
    pub fn extensionless_set(&self) -> HashSet<&'static str> {
        EXTENSIONLESS_FILES.iter().copied().collect()
    }

    /// Get skip directories as a HashSet for O(1) lookup.
    pub fn skip_dirs_set(&self) -> HashSet<&'static str> {
        DEFAULT_IGNORE_PATTERNS.iter().copied().collect()
    }

    /// Check if a file extension should be included.
    #[inline]
    pub fn should_include_extension(&self, ext: &str) -> bool {
        let ext_lower = ext.to_lowercase();
        SUPPORTED_EXTENSIONS.contains(&ext_lower.as_str())
    }

    /// Check if an extensionless filename should be included.
    #[inline]
    pub fn should_include_extensionless(&self, name: &str) -> bool {
        let name_lower = name.to_lowercase();
        EXTENSIONLESS_FILES.contains(&name_lower.as_str())
    }
}

fn first_non_empty_env(keys: &[&str]) -> Option<String> {
    keys.iter().find_map(|key| {
        env::var(key)
            .ok()
            .map(|value| value.trim().to_string())
            .filter(|value| !value.is_empty())
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use once_cell::sync::Lazy;
    use std::sync::Mutex;

    static ENV_LOCK: Lazy<Mutex<()>> = Lazy::new(|| Mutex::new(()));
    const TEST_ENV_KEYS: &[&str] = &[
        "EMBEDDING_URL",
        "OPENAI_BASE_URL",
        "EMBEDDING_API_KEY",
        "OPENAI_API_KEY",
        "MILVUS_URL",
        "MILVUS_ADDRESS",
        "MILVUS_TOKEN",
    ];

    struct EnvGuard {
        saved: Vec<(&'static str, Option<String>)>,
    }

    impl EnvGuard {
        fn new(overrides: &[(&str, Option<&str>)]) -> Self {
            let saved = TEST_ENV_KEYS
                .iter()
                .map(|&key| (key, env::var(key).ok()))
                .collect::<Vec<_>>();

            for &key in TEST_ENV_KEYS {
                env::remove_var(key);
            }
            for (key, value) in overrides {
                match value {
                    Some(value) => env::set_var(key, value),
                    None => env::remove_var(key),
                }
            }

            Self { saved }
        }
    }

    impl Drop for EnvGuard {
        fn drop(&mut self) {
            for (key, value) in &self.saved {
                match value {
                    Some(value) => env::set_var(key, value),
                    None => env::remove_var(key),
                }
            }
        }
    }

    #[test]
    fn from_env_prefers_canonical_names() {
        let _lock = ENV_LOCK.lock().unwrap();
        let _guard = EnvGuard::new(&[
            ("EMBEDDING_URL", Some("https://api.openai.com/v1")),
            ("EMBEDDING_API_KEY", Some("secret")),
            ("MILVUS_URL", Some("https://cluster.example.com:443")),
            ("MILVUS_TOKEN", Some("token")),
        ]);

        let config = Config::from_env();

        assert_eq!(config.embedding_url, "https://api.openai.com/v1");
        assert_eq!(config.embedding_api_key.as_deref(), Some("secret"));
        assert_eq!(config.milvus_url, "https://cluster.example.com:443");
        assert_eq!(config.milvus_token.as_deref(), Some("token"));
        assert!(config.has_embedding_url());
        assert!(config.has_milvus_url());
    }

    #[test]
    fn from_env_accepts_legacy_aliases() {
        let _lock = ENV_LOCK.lock().unwrap();
        let _guard = EnvGuard::new(&[
            ("OPENAI_BASE_URL", Some("https://api.jina.ai/v1")),
            ("OPENAI_API_KEY", Some("jina-secret")),
            (
                "MILVUS_ADDRESS",
                Some("https://cluster.zillizcloud.com:443"),
            ),
        ]);

        let config = Config::from_env();

        assert_eq!(config.embedding_url, "https://api.jina.ai/v1");
        assert_eq!(config.embedding_api_key.as_deref(), Some("jina-secret"));
        assert_eq!(config.milvus_url, "https://cluster.zillizcloud.com:443");
        assert!(config.has_embedding_url());
        assert!(config.has_milvus_url());
    }

    #[test]
    fn from_env_treats_empty_urls_as_unset() {
        let _lock = ENV_LOCK.lock().unwrap();
        let _guard = EnvGuard::new(&[
            ("EMBEDDING_URL", Some("  ")),
            ("OPENAI_BASE_URL", Some("")),
            ("EMBEDDING_API_KEY", Some("")),
            ("MILVUS_URL", Some(" ")),
            ("MILVUS_ADDRESS", Some("")),
        ]);

        let config = Config::from_env();

        assert_eq!(config.embedding_url, "");
        assert_eq!(config.embedding_api_key, None);
        assert_eq!(config.milvus_url, "");
        assert!(!config.has_embedding_url());
        assert!(!config.has_milvus_url());
    }
}
