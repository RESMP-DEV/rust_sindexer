pub mod client;

pub use client::{Document, InsertRow, MilvusClient, SearchHit};

use sha2::{Digest, Sha256};
use std::path::Path;

/// Generate a sanitized, hashed collection name from a filesystem path.
///
/// Milvus collection names must:
/// - Start with a letter or underscore
/// - Contain only alphanumeric characters and underscores
/// - Be at most 255 characters
///
/// This function takes the last path component as a human-readable prefix,
/// sanitizes it, and appends a truncated SHA-256 hash of the full path
/// for uniqueness.
pub fn collection_name_from_path(path: &Path) -> String {
    let full_path = path.to_string_lossy();

    // Hash the full path for uniqueness
    let mut hasher = Sha256::new();
    hasher.update(full_path.as_bytes());
    let hash = hex::encode(hasher.finalize());
    let hash_prefix = &hash[..16];

    // Extract and sanitize the last component for readability
    let prefix = path
        .file_name()
        .and_then(|s| s.to_str())
        .unwrap_or("collection");

    let sanitized: String = prefix
        .chars()
        .map(|c| if c.is_ascii_alphanumeric() { c } else { '_' })
        .collect();

    // Ensure it starts with a letter or underscore
    let sanitized = if sanitized
        .chars()
        .next()
        .map(|c| c.is_ascii_digit())
        .unwrap_or(true)
    {
        format!("_{}", sanitized)
    } else {
        sanitized
    };

    // Truncate prefix if needed to fit within 255 chars with hash
    let max_prefix_len = 255 - 1 - hash_prefix.len(); // 1 for underscore separator
    let prefix_part = if sanitized.len() > max_prefix_len {
        &sanitized[..max_prefix_len]
    } else {
        &sanitized
    };

    format!("{}_{}", prefix_part, hash_prefix)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    #[test]
    fn test_collection_name_basic() {
        let path = PathBuf::from("/home/user/my-project");
        let name = collection_name_from_path(&path);
        assert!(name.starts_with("my_project_"));
        assert!(name.len() <= 255);
    }

    #[test]
    fn test_collection_name_numeric_start() {
        let path = PathBuf::from("/home/user/123project");
        let name = collection_name_from_path(&path);
        assert!(name.starts_with('_'));
    }

    #[test]
    fn test_collection_name_special_chars() {
        let path = PathBuf::from("/home/user/my.project-name@v2");
        let name = collection_name_from_path(&path);
        assert!(!name.contains('.'));
        assert!(!name.contains('-'));
        assert!(!name.contains('@'));
    }

    #[test]
    fn test_collection_name_uniqueness() {
        let path1 = PathBuf::from("/home/user/project");
        let path2 = PathBuf::from("/home/other/project");
        let name1 = collection_name_from_path(&path1);
        let name2 = collection_name_from_path(&path2);
        assert_ne!(name1, name2);
    }
}
