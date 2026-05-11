pub mod client;
pub mod local;

pub use client::{CollectionStats, Document, InsertRow, MilvusClient, SearchHit};
pub use local::LocalStore;

use anyhow::Result;
use sha2::{Digest, Sha256};
use std::borrow::Cow;
use std::env;
use std::path::Path;
use tracing::{debug, info};

const COLLECTION_IDENTITY_ENV: &str = "SINDEXER_COLLECTION_IDENTITY";

/// Selects between a local brute-force vector store and a remote Milvus instance.
pub enum VectorStore {
    Local(LocalStore),
    Milvus(MilvusClient),
}

impl VectorStore {
    pub async fn create_collection(&self, name: &str, dimension: usize) -> Result<()> {
        info!(collection = name, dimension, "creating collection");
        match self {
            Self::Local(store) => store.create_collection(name, dimension),
            Self::Milvus(client) => client.create_collection(name, dimension).await,
        }
    }

    pub async fn has_collection(&self, name: &str) -> Result<bool> {
        match self {
            Self::Local(store) => store.has_collection(name),
            Self::Milvus(client) => client.has_collection(name).await,
        }
    }

    pub async fn drop_collection(&self, name: &str) -> Result<()> {
        info!(collection = name, "dropping collection");
        match self {
            Self::Local(store) => store.drop_collection(name),
            Self::Milvus(client) => client.drop_collection(name).await,
        }
    }

    pub async fn insert_batch(&self, collection: &str, data: &[InsertRow]) -> Result<usize> {
        debug!(collection, batch_size = data.len(), "inserting batch");
        match self {
            Self::Local(store) => {
                let docs: Vec<_> = data
                    .iter()
                    .map(|row| local::LocalDoc {
                        id: row.id.to_string(),
                        content: row.content.clone(),
                        vector: row.vector.clone(),
                        metadata: row.metadata.clone(),
                    })
                    .collect();
                let inserted = docs.len();
                store.insert_docs(collection, docs)?;
                Ok(inserted)
            }
            Self::Milvus(client) => client.insert_batch(collection, data).await,
        }
    }

    pub async fn search(
        &self,
        collection: &str,
        vector: &[f32],
        top_k: usize,
    ) -> Result<Vec<SearchHit>> {
        debug!(collection, top_k, "searching vector store");
        match self {
            Self::Local(store) => store.search(collection, vector, top_k),
            Self::Milvus(client) => client.search(collection, vector, top_k).await,
        }
    }

    pub async fn list_collections(&self) -> Result<Vec<String>> {
        match self {
            Self::Local(store) => Ok(store.list_collections()),
            Self::Milvus(client) => client.list_collections().await,
        }
    }

    pub async fn collection_stats(&self, name: &str) -> Result<CollectionStats> {
        match self {
            Self::Local(store) => Ok(CollectionStats {
                row_count: store.collection_size(name) as u64,
            }),
            Self::Milvus(client) => client.collection_stats(name).await,
        }
    }

    pub async fn delete_by_relative_paths(
        &self,
        collection: &str,
        relative_paths: &[String],
    ) -> Result<()> {
        if relative_paths.is_empty() {
            return Ok(());
        }
        debug!(
            collection,
            path_count = relative_paths.len(),
            "deleting by relative paths"
        );
        match self {
            Self::Local(store) => store.delete_by_filter(collection, relative_paths),
            Self::Milvus(client) => {
                let filter = build_relative_path_milvus_filter(relative_paths);
                client.delete(collection, &filter).await
            }
        }
    }
}

fn build_relative_path_milvus_filter(relative_paths: &[String]) -> String {
    let serialized = relative_paths
        .iter()
        .map(|path| serde_json::to_string(path).expect("relative path must serialize"))
        .collect::<Vec<_>>()
        .join(", ");
    format!("metadata[\"relative_path\"] in [{serialized}]")
}

/// Generate a sanitized, hashed collection name from a filesystem path or
/// an operator-provided stable collection identity.
///
/// Milvus collection names must:
/// - Start with a letter or underscore
/// - Contain only alphanumeric characters and underscores
/// - Be at most 255 characters
///
/// By default, this function takes the last path component as a human-readable
/// prefix, sanitizes it, and appends a truncated SHA-256 hash of the full path
/// for uniqueness. Set `SINDEXER_COLLECTION_IDENTITY` to make multiple hosts
/// with different checkout paths share the same backing collection.
pub fn collection_name_from_path(path: &Path) -> String {
    let identity = env::var(COLLECTION_IDENTITY_ENV).ok();
    let identity = identity.as_deref().and_then(non_empty_trimmed);
    collection_name_from_path_with_identity(path, identity)
}

fn collection_name_from_path_with_identity(path: &Path, identity: Option<&str>) -> String {
    let identity = identity.and_then(non_empty_trimmed);
    let collection_identity = identity
        .map(Cow::Borrowed)
        .unwrap_or_else(|| Cow::Owned(path.to_string_lossy().into_owned()));

    // Hash the collection identity (provided identity or full path) for uniqueness.
    let mut hasher = Sha256::new();
    hasher.update(collection_identity.as_bytes());
    let hash = hex::encode(hasher.finalize());
    let hash_prefix = &hash[..16];

    // Extract and sanitize the last component for readability
    let prefix = identity
        .and_then(identity_prefix)
        .or_else(|| path.file_name().and_then(|s| s.to_str()))
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

fn non_empty_trimmed(value: &str) -> Option<&str> {
    let value = value.trim();
    if value.is_empty() {
        None
    } else {
        Some(value)
    }
}

fn identity_prefix(identity: &str) -> Option<&str> {
    Path::new(identity)
        .file_name()
        .and_then(|s| s.to_str())
        .and_then(non_empty_trimmed)
        .or_else(|| non_empty_trimmed(identity))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    #[test]
    fn test_collection_name_basic() {
        let path = PathBuf::from("/home/user/my-project");
        let name = collection_name_from_path_with_identity(&path, None);
        assert!(name.starts_with("my_project_"));
        assert!(name.len() <= 255);
    }

    #[test]
    fn test_collection_name_numeric_start() {
        let path = PathBuf::from("/home/user/123project");
        let name = collection_name_from_path_with_identity(&path, None);
        assert!(name.starts_with('_'));
    }

    #[test]
    fn test_collection_name_special_chars() {
        let path = PathBuf::from("/home/user/my.project-name@v2");
        let name = collection_name_from_path_with_identity(&path, None);
        assert!(!name.contains('.'));
        assert!(!name.contains('-'));
        assert!(!name.contains('@'));
    }

    #[test]
    fn test_collection_name_uniqueness() {
        let path1 = PathBuf::from("/home/user/project");
        let path2 = PathBuf::from("/home/other/project");
        let name1 = collection_name_from_path_with_identity(&path1, None);
        let name2 = collection_name_from_path_with_identity(&path2, None);
        assert_ne!(name1, name2);
    }

    #[test]
    fn test_collection_name_identity_overrides_absolute_path() {
        let mac_path = PathBuf::from("/Users/kearm/AlphaHENG");
        let linux_path = PathBuf::from("/home/kearm/AlphaHENG");

        let mac_name = collection_name_from_path_with_identity(&mac_path, Some("AlphaHENG"));
        let linux_name = collection_name_from_path_with_identity(&linux_path, Some("AlphaHENG"));

        assert_eq!(mac_name, linux_name);
        assert!(mac_name.starts_with("AlphaHENG_"));
    }

    #[test]
    fn test_collection_name_blank_identity_falls_back_to_path() {
        let path1 = PathBuf::from("/home/user/project");
        let path2 = PathBuf::from("/home/other/project");
        let name1 = collection_name_from_path_with_identity(&path1, Some("  "));
        let name2 = collection_name_from_path_with_identity(&path2, Some(""));

        assert_ne!(name1, name2);
    }
}
