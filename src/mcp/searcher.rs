//! Code search functionality using vector embeddings and Milvus.
//!
//! This module provides semantic code search by embedding queries and
//! searching a Milvus vector database for similar code chunks.

use std::path::Path;

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};

use crate::types::CodeChunk;

/// A search result containing a matched code chunk with relevance score.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SearchResult {
    /// The matched code chunk.
    pub chunk: CodeChunk,
    /// Similarity score (higher is more relevant, typically 0.0 to 1.0 for cosine similarity).
    pub score: f32,
}

/// Shared application state containing embedding model and vector database clients.
///
/// This struct is expected to be provided by the application and contains
/// initialized clients for the embedding service and Milvus.
pub struct ContextState {
    /// Client for generating text embeddings.
    pub embedding_client: EmbeddingClient,
    /// Client for Milvus vector database operations.
    pub milvus_client: MilvusClient,
    /// Name of the Milvus collection to search.
    pub collection_name: String,
}

/// Client for generating text embeddings.
///
/// Wraps the embedding service (e.g., OpenAI, local model) to generate
/// vector representations of text.
pub struct EmbeddingClient {
    /// Base URL for the embedding API.
    pub base_url: String,
    /// Model identifier for embeddings.
    pub model: String,
    /// HTTP client for API requests.
    client: reqwest::Client,
}

/// Request payload for embedding API.
#[derive(Serialize)]
struct EmbeddingRequest<'a> {
    model: &'a str,
    input: &'a str,
}

/// Response from embedding API.
#[derive(Deserialize)]
struct EmbeddingResponse {
    data: Vec<EmbeddingData>,
}

#[derive(Deserialize)]
struct EmbeddingData {
    embedding: Vec<f32>,
}

impl EmbeddingClient {
    /// Create a new embedding client.
    pub fn new(base_url: impl Into<String>, model: impl Into<String>) -> Self {
        Self {
            base_url: base_url.into(),
            model: model.into(),
            client: reqwest::Client::new(),
        }
    }

    /// Generate an embedding vector for the given text.
    pub async fn embed(&self, text: &str) -> Result<Vec<f32>> {
        let url = format!("{}/embeddings", self.base_url.trim_end_matches('/'));
        let request = EmbeddingRequest {
            model: &self.model,
            input: text,
        };

        let response = self
            .client
            .post(&url)
            .json(&request)
            .send()
            .await
            .context("Failed to send embedding request")?;

        let status = response.status();
        if !status.is_success() {
            let body = response.text().await.unwrap_or_default();
            anyhow::bail!("Embedding API returned {}: {}", status, body);
        }

        let embedding_response: EmbeddingResponse = response
            .json()
            .await
            .context("Failed to parse embedding response")?;

        embedding_response
            .data
            .into_iter()
            .next()
            .map(|d| d.embedding)
            .context("No embedding data in response")
    }
}

/// Client for Milvus vector database operations.
pub struct MilvusClient {
    /// Milvus server address.
    pub address: String,
    /// Internal gRPC client (placeholder for actual milvus-sdk-rust).
    _client: (),
}

/// Search parameters for Milvus queries.
#[derive(Clone, Debug)]
pub struct SearchParams {
    /// Number of results to return.
    pub limit: u32,
    /// Metric type for similarity (e.g., "IP" for inner product, "L2" for Euclidean).
    pub metric_type: String,
    /// Optional filter expression.
    pub filter: Option<String>,
}

impl Default for SearchParams {
    fn default() -> Self {
        Self {
            limit: 10,
            metric_type: "IP".to_string(),
            filter: None,
        }
    }
}

/// Raw search hit from Milvus.
#[derive(Debug)]
pub struct MilvusHit {
    pub id: String,
    pub score: f32,
    pub content: String,
    pub file_path: String,
    pub relative_path: String,
    pub start_line: u32,
    pub end_line: u32,
    pub language: String,
}

impl MilvusClient {
    /// Create a new Milvus client.
    pub fn new(address: impl Into<String>) -> Self {
        Self {
            address: address.into(),
            _client: (),
        }
    }

    /// Search for similar vectors in the specified collection.
    ///
    /// This is a placeholder implementation. In production, this would use
    /// the milvus-sdk-rust crate to perform the actual vector search.
    pub async fn search(
        &self,
        collection: &str,
        vector: &[f32],
        params: &SearchParams,
    ) -> Result<Vec<MilvusHit>> {
        // In a real implementation, this would:
        // 1. Connect to Milvus using gRPC
        // 2. Construct a search request with the query vector
        // 3. Execute the search with the specified parameters
        // 4. Parse and return the results

        // Placeholder that simulates the Milvus search API structure
        let _ = (collection, vector, params);

        // The actual implementation would look something like:
        //
        // use milvus::client::Client;
        // use milvus::search::{SearchRequest, SearchResult};
        //
        // let mut client = Client::connect(&self.address).await?;
        //
        // let request = SearchRequest::new(collection)
        //     .with_vectors(vec![vector.to_vec()])
        //     .with_top_k(params.limit as i64)
        //     .with_metric_type(&params.metric_type)
        //     .with_output_fields(vec![
        //         "id", "content", "file_path", "relative_path",
        //         "start_line", "end_line", "language"
        //     ]);
        //
        // if let Some(filter) = &params.filter {
        //     request = request.with_filter(filter);
        // }
        //
        // let results = client.search(request).await?;
        //
        // results.into_iter().map(|hit| MilvusHit { ... }).collect()

        Ok(vec![])
    }
}

/// Search for code chunks semantically similar to the given query.
///
/// This function:
/// 1. Embeds the query text using the configured embedding model
/// 2. Searches Milvus for vectors similar to the query embedding
/// 3. Returns ranked results with code snippets and file paths
///
/// # Arguments
///
/// * `state` - Application state containing embedding and Milvus clients
/// * `path` - Base path for resolving relative file paths (typically repo root)
/// * `query` - Natural language or code query to search for
/// * `limit` - Maximum number of results to return
///
/// # Returns
///
/// A vector of `SearchResult` ordered by relevance (highest score first).
///
/// # Example
///
/// ```ignore
/// let results = search_code(&state, Path::new("/repo"), "authentication logic", 10).await?;
/// for result in results {
///     println!("{}: {} (score: {:.3})",
///         result.chunk.relative_path,
///         result.chunk.start_line,
///         result.score
///     );
/// }
/// ```
pub async fn search_code(
    state: &ContextState,
    path: &Path,
    query: &str,
    limit: u32,
) -> Result<Vec<SearchResult>> {
    // Embed the query text
    let query_embedding = state
        .embedding_client
        .embed(query)
        .await
        .context("Failed to embed search query")?;

    // Build search parameters
    let search_params = SearchParams {
        limit,
        metric_type: "IP".to_string(), // Inner product for normalized embeddings
        filter: None,
    };

    // Search Milvus for similar vectors
    let hits = state
        .milvus_client
        .search(&state.collection_name, &query_embedding, &search_params)
        .await
        .context("Failed to search Milvus")?;

    // Convert hits to SearchResults
    let results: Vec<SearchResult> = hits
        .into_iter()
        .map(|hit| {
            let file_path = path.join(&hit.relative_path);
            SearchResult {
                chunk: CodeChunk {
                    id: hit.id,
                    content: hit.content,
                    file_path,
                    relative_path: hit.relative_path,
                    start_line: hit.start_line,
                    end_line: hit.end_line,
                    language: hit.language,
                },
                score: hit.score,
            }
        })
        .collect();

    Ok(results)
}

/// Search with an optional path filter.
///
/// Restricts search results to files within the specified directory.
pub async fn search_code_in_directory(
    state: &ContextState,
    base_path: &Path,
    directory: &Path,
    query: &str,
    limit: u32,
) -> Result<Vec<SearchResult>> {
    // Build a filter expression for the path prefix
    let dir_str = directory.to_string_lossy();
    let filter = format!(r#"relative_path like "{}%""#, dir_str);

    let query_embedding = state
        .embedding_client
        .embed(query)
        .await
        .context("Failed to embed search query")?;

    let search_params = SearchParams {
        limit,
        metric_type: "IP".to_string(),
        filter: Some(filter),
    };

    let hits = state
        .milvus_client
        .search(&state.collection_name, &query_embedding, &search_params)
        .await
        .context("Failed to search Milvus")?;

    let results: Vec<SearchResult> = hits
        .into_iter()
        .map(|hit| {
            let file_path = base_path.join(&hit.relative_path);
            SearchResult {
                chunk: CodeChunk {
                    id: hit.id,
                    content: hit.content,
                    file_path,
                    relative_path: hit.relative_path,
                    start_line: hit.start_line,
                    end_line: hit.end_line,
                    language: hit.language,
                },
                score: hit.score,
            }
        })
        .collect();

    Ok(results)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_search_params_default() {
        let params = SearchParams::default();
        assert_eq!(params.limit, 10);
        assert_eq!(params.metric_type, "IP");
        assert!(params.filter.is_none());
    }

    #[test]
    fn test_search_result_serialization() {
        let result = SearchResult {
            chunk: CodeChunk {
                id: "test-id".to_string(),
                content: "fn main() {}".to_string(),
                file_path: "/repo/src/main.rs".into(),
                relative_path: "src/main.rs".to_string(),
                start_line: 1,
                end_line: 1,
                language: "rust".to_string(),
            },
            score: 0.95,
        };

        let json = serde_json::to_string(&result).expect("serialization should succeed");
        assert!(json.contains("test-id"));
        assert!(json.contains("0.95"));
    }
}
