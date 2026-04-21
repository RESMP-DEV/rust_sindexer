pub mod client;
pub mod local;

pub use client::{CollectionStats, Document, InsertRow, MilvusClient, SearchHit};
pub use local::LocalStore;

use anyhow::Result;
use sha2::{Digest, Sha256};
use std::path::Path;

/// Selects between a local brute-force vector store and a remote Milvus instance.
pub enum VectorStore {
    Local(LocalStore),
    Milvus(MilvusClient),
}

impl VectorStore {
    pub async fn create_collection(&self, name: &str, dimension: usize) -> Result<()> {
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
        match self {
            Self::Local(store) => store.drop_collection(name),
            Self::Milvus(client) => client.drop_collection(name).await,
        }
    }

    pub async fn insert_batch(&self, collection: &str, data: &[InsertRow]) -> Result<()> {
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
                store.insert_docs(collection, docs)
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
