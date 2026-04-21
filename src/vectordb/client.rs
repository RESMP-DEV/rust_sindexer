use anyhow::{Context, Result};
use reqwest::Client;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

/// A document to be inserted into Milvus.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Document {
    /// Unique identifier for the document.
    pub id: String,
    /// Text content of the document.
    pub content: String,
    /// Embedding vector.
    pub vector: Vec<f32>,
    /// Additional metadata as JSON.
    #[serde(default)]
    pub metadata: serde_json::Value,
}

/// A search result from Milvus.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SearchHit {
    /// Document ID.
    pub id: String,
    /// Similarity score (distance).
    pub score: f32,
    /// Document content.
    pub content: String,
    /// Metadata.
    pub metadata: serde_json::Value,
}

/// HTTP client for Milvus vector database.
#[derive(Clone)]
pub struct MilvusClient {
    client: Client,
    base_url: String,
}

#[derive(Serialize)]
struct CreateCollectionRequest {
    #[serde(rename = "dbName")]
    db_name: String,
    #[serde(rename = "collectionName")]
    collection_name: String,
    dimension: usize,
    #[serde(rename = "metricType")]
    metric_type: String,
    #[serde(rename = "primaryFieldName")]
    primary_field_name: String,
    #[serde(rename = "vectorFieldName")]
    vector_field_name: String,
}

#[derive(Serialize)]
struct HasCollectionRequest {
    #[serde(rename = "dbName")]
    db_name: String,
    #[serde(rename = "collectionName")]
    collection_name: String,
}

#[derive(Deserialize)]
struct HasCollectionResponse {
    code: i32,
    data: Option<HasCollectionData>,
    message: Option<String>,
}

#[derive(Deserialize)]
struct HasCollectionData {
    has: bool,
}

#[derive(Serialize)]
struct InsertRequest {
    #[serde(rename = "dbName")]
    db_name: String,
    #[serde(rename = "collectionName")]
    collection_name: String,
    data: Vec<InsertRow>,
}

/// A row to be inserted into Milvus.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct InsertRow {
    /// Unique identifier.
    pub id: i64,
    /// Text content.
    pub content: String,
    /// Embedding vector.
    pub vector: Vec<f32>,
    /// Additional metadata as JSON.
    pub metadata: serde_json::Value,
}

#[derive(Serialize)]
struct SearchRequest {
    #[serde(rename = "dbName")]
    db_name: String,
    #[serde(rename = "collectionName")]
    collection_name: String,
    data: Vec<Vec<f32>>,
    #[serde(rename = "annsField")]
    anns_field: String,
    limit: usize,
    #[serde(rename = "outputFields")]
    output_fields: Vec<String>,
}

#[derive(Deserialize)]
struct SearchResponse {
    code: i32,
    data: Option<SearchResultsData>,
    message: Option<String>,
}

#[derive(Deserialize)]
#[serde(untagged)]
enum SearchResultsData {
    Flat(Vec<SearchResultItem>),
    Nested(Vec<Vec<SearchResultItem>>),
}

#[derive(Deserialize)]
struct SearchResultItem {
    id: serde_json::Value,
    distance: f32,
    content: Option<String>,
    metadata: Option<serde_json::Value>,
}

#[derive(Serialize)]
struct DeleteRequest {
    #[serde(rename = "dbName")]
    db_name: String,
    #[serde(rename = "collectionName")]
    collection_name: String,
    filter: String,
}

#[derive(Deserialize)]
struct MilvusResponse {
    code: i32,
    message: Option<String>,
}

#[derive(Serialize)]
struct DropCollectionRequest {
    #[serde(rename = "dbName")]
    db_name: String,
    #[serde(rename = "collectionName")]
    collection_name: String,
}

#[derive(Serialize)]
struct ListCollectionsRequest {
    #[serde(rename = "dbName")]
    db_name: String,
}

#[derive(Deserialize)]
struct ListCollectionsResponse {
    code: i32,
    data: Option<Vec<String>>,
    message: Option<String>,
}

#[derive(Serialize)]
struct CollectionStatsRequest {
    #[serde(rename = "dbName")]
    db_name: String,
    #[serde(rename = "collectionName")]
    collection_name: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CollectionStats {
    pub row_count: u64,
}

#[derive(Deserialize)]
struct CollectionStatsResponse {
    code: i32,
    data: Option<CollectionStatsData>,
    message: Option<String>,
}

#[derive(Deserialize)]
struct CollectionStatsData {
    #[serde(rename = "rowCount", default)]
    row_count: u64,
}

impl MilvusClient {
    /// Create a new Milvus client.
    ///
    /// # Arguments
    /// * `base_url` - Base URL for Milvus REST API (e.g., "http://localhost:19530")
    pub fn new(base_url: &str, token: Option<String>) -> Self {
        let mut headers = reqwest::header::HeaderMap::new();
        if let Some(ref token) = token {
            headers.insert(
                reqwest::header::AUTHORIZATION,
                format!("Bearer {}", token)
                    .parse()
                    .expect("valid header value"),
            );
        }

        let client = Client::builder()
            .pool_idle_timeout(std::time::Duration::from_secs(90))
            .pool_max_idle_per_host(32)
            .connect_timeout(std::time::Duration::from_secs(10))
            .timeout(std::time::Duration::from_secs(60))
            .default_headers(headers)
            .build()
            .expect("failed to create HTTP client");

        Self {
            client,
            base_url: base_url.to_string(),
        }
    }

    /// Create a collection with the specified name and vector dimension.
    ///
    /// Uses cosine similarity as the metric type. The collection will have:
    /// - `id` as the primary field (VARCHAR)
    /// - `content` for document text (VARCHAR)
    /// - `vector` for embeddings (FLOAT_VECTOR)
    /// - `metadata` for additional data (JSON)
    pub async fn create_collection(&self, name: &str, dimension: usize) -> Result<()> {
        let url = format!("{}/v2/vectordb/collections/create", self.base_url);

        let request = CreateCollectionRequest {
            db_name: "default".to_string(),
            collection_name: name.to_string(),
            dimension,
            metric_type: "COSINE".to_string(),
            primary_field_name: "id".to_string(),
            vector_field_name: "vector".to_string(),
        };

        let response: MilvusResponse = self
            .client
            .post(&url)
            .json(&request)
            .send()
            .await
            .context("failed to send create collection request")?
            .json()
            .await
            .context("failed to parse create collection response")?;

        if response.code != 0 {
            anyhow::bail!(
                "create collection failed: {}",
                response
                    .message
                    .unwrap_or_else(|| "unknown error".to_string())
            );
        }

        Ok(())
    }

    /// Check if a collection exists.
    pub async fn has_collection(&self, name: &str) -> Result<bool> {
        let url = format!("{}/v2/vectordb/collections/has", self.base_url);

        let request = HasCollectionRequest {
            db_name: "default".to_string(),
            collection_name: name.to_string(),
        };

        let response: HasCollectionResponse = self
            .client
            .post(&url)
            .json(&request)
            .send()
            .await
            .context("failed to send has collection request")?
            .json()
            .await
            .context("failed to parse has collection response")?;

        if response.code != 0 {
            anyhow::bail!(
                "has collection failed: {}",
                response
                    .message
                    .unwrap_or_else(|| "unknown error".to_string())
            );
        }

        Ok(response.data.map(|d| d.has).unwrap_or(false))
    }

    /// Insert documents into a collection.
    pub async fn insert(&self, collection: &str, docs: Vec<Document>) -> Result<()> {
        if docs.is_empty() {
            return Ok(());
        }

        let url = format!("{}/v2/vectordb/entities/insert", self.base_url);

        let data: Vec<InsertRow> = docs
            .into_iter()
            .map(|doc| InsertRow {
                id: milvus_id_for_chunk_id(&doc.id),
                content: doc.content,
                vector: doc.vector,
                metadata: doc.metadata,
            })
            .collect();

        let request = InsertRequest {
            db_name: "default".to_string(),
            collection_name: collection.to_string(),
            data,
        };

        let response: MilvusResponse = self
            .client
            .post(&url)
            .json(&request)
            .send()
            .await
            .context("failed to send insert request")?
            .json()
            .await
            .context("failed to parse insert response")?;

        if response.code != 0 {
            anyhow::bail!(
                "insert failed: {}",
                response
                    .message
                    .unwrap_or_else(|| "unknown error".to_string())
            );
        }

        Ok(())
    }

    /// Search for similar vectors in a collection.
    ///
    /// # Arguments
    /// * `collection` - Collection name
    /// * `vector` - Query vector
    /// * `top_k` - Number of results to return
    pub async fn search(
        &self,
        collection: &str,
        vector: &[f32],
        top_k: usize,
    ) -> Result<Vec<SearchHit>> {
        let url = format!("{}/v2/vectordb/entities/search", self.base_url);

        let request = SearchRequest {
            db_name: "default".to_string(),
            collection_name: collection.to_string(),
            data: vec![vector.to_vec()],
            anns_field: "vector".to_string(),
            limit: top_k,
            output_fields: vec![
                "id".to_string(),
                "content".to_string(),
                "metadata".to_string(),
            ],
        };

        let response: SearchResponse = self
            .client
            .post(&url)
            .json(&request)
            .send()
            .await
            .context("failed to send search request")?
            .json()
            .await
            .context("failed to parse search response")?;

        if response.code != 0 {
            anyhow::bail!(
                "search failed: {}",
                response
                    .message
                    .unwrap_or_else(|| "unknown error".to_string())
            );
        }

        let hits = response
            .data
            .map(SearchResultsData::into_hits)
            .unwrap_or_default()
            .into_iter()
            .map(|item| SearchHit {
                id: milvus_search_id_to_string(item.id),
                score: item.distance,
                content: item.content.unwrap_or_default(),
                metadata: item.metadata.unwrap_or(serde_json::Value::Null),
            })
            .collect();

        Ok(hits)
    }

    /// Delete entities from a collection using a filter expression.
    ///
    /// # Arguments
    /// * `collection` - Collection name
    /// * `filter` - Filter expression (e.g., `id in ["id1", "id2"]`)
    pub async fn delete(&self, collection: &str, filter: &str) -> Result<()> {
        let url = format!("{}/v2/vectordb/entities/delete", self.base_url);

        let request = DeleteRequest {
            db_name: "default".to_string(),
            collection_name: collection.to_string(),
            filter: filter.to_string(),
        };

        let response: MilvusResponse = self
            .client
            .post(&url)
            .json(&request)
            .send()
            .await
            .context("failed to send delete request")?
            .json()
            .await
            .context("failed to parse delete response")?;

        if response.code != 0 {
            anyhow::bail!(
                "delete failed: {}",
                response
                    .message
                    .unwrap_or_else(|| "unknown error".to_string())
            );
        }

        Ok(())
    }

    /// Batch insert rows into a collection.
    ///
    /// # Arguments
    /// * `collection` - Collection name
    /// * `data` - Rows to insert
    pub async fn insert_batch(&self, collection: &str, data: &[InsertRow]) -> Result<()> {
        if data.is_empty() {
            return Ok(());
        }

        const MAX_RETRIES: u32 = 3;
        let url = format!("{}/v2/vectordb/entities/insert", self.base_url);

        let request_body = InsertRequest {
            db_name: "default".to_string(),
            collection_name: collection.to_string(),
            data: data.to_vec(),
        };
        let body_bytes = serde_json::to_vec(&request_body)
            .context("failed to serialize insert batch request")?;

        let mut last_err = None;
        for attempt in 0..=MAX_RETRIES {
            if attempt > 0 {
                let backoff = std::time::Duration::from_millis(500 * 2u64.pow(attempt - 1));
                tokio::time::sleep(backoff).await;
            }

            let response = match self
                .client
                .post(&url)
                .header("content-type", "application/json")
                .body(body_bytes.clone())
                .send()
                .await
            {
                Ok(r) => r,
                Err(e) => {
                    last_err = Some(format!("request failed: {e}"));
                    continue;
                }
            };

            let status = response.status();
            if status == reqwest::StatusCode::TOO_MANY_REQUESTS || status.is_server_error() {
                let body = response.text().await.unwrap_or_default();
                last_err = Some(format!("Milvus returned {status}: {body}"));
                continue;
            }

            let response: MilvusResponse = response
                .json()
                .await
                .context("failed to parse insert batch response")?;

            if response.code != 0 {
                anyhow::bail!(
                    "insert batch failed: {}",
                    response
                        .message
                        .unwrap_or_else(|| "unknown error".to_string())
                );
            }

            return Ok(());
        }

        anyhow::bail!(
            "insert batch failed after {} retries: {}",
            MAX_RETRIES,
            last_err.unwrap_or_else(|| "unknown error".to_string())
        )
    }

    /// List all collections in the database.
    pub async fn list_collections(&self) -> Result<Vec<String>> {
        let url = format!("{}/v2/vectordb/collections/list", self.base_url);

        let request = ListCollectionsRequest {
            db_name: "default".to_string(),
        };

        let response: ListCollectionsResponse = self
            .client
            .post(&url)
            .json(&request)
            .send()
            .await
            .context("failed to send list collections request")?
            .json()
            .await
            .context("failed to parse list collections response")?;

        if response.code != 0 {
            anyhow::bail!(
                "list collections failed: {}",
                response
                    .message
                    .unwrap_or_else(|| "unknown error".to_string())
            );
        }

        Ok(response.data.unwrap_or_default())
    }

    /// Get statistics for a collection (row count).
    pub async fn collection_stats(&self, name: &str) -> Result<CollectionStats> {
        let url = format!("{}/v2/vectordb/collections/get_stats", self.base_url);

        let request = CollectionStatsRequest {
            db_name: "default".to_string(),
            collection_name: name.to_string(),
        };

        let response: CollectionStatsResponse = self
            .client
            .post(&url)
            .json(&request)
            .send()
            .await
            .context("failed to send collection stats request")?
            .json()
            .await
            .context("failed to parse collection stats response")?;

        if response.code != 0 {
            anyhow::bail!(
                "collection stats failed: {}",
                response
                    .message
                    .unwrap_or_else(|| "unknown error".to_string())
            );
        }

        let data = response.data.unwrap_or(CollectionStatsData { row_count: 0 });
        Ok(CollectionStats {
            row_count: data.row_count,
        })
    }

    /// Drop a collection.
    ///
    /// # Arguments
    /// * `name` - Collection name to drop
    pub async fn drop_collection(&self, name: &str) -> Result<()> {
        let url = format!("{}/v2/vectordb/collections/drop", self.base_url);

        let request = DropCollectionRequest {
            db_name: "default".to_string(),
            collection_name: name.to_string(),
        };

        let response: MilvusResponse = self
            .client
            .post(&url)
            .json(&request)
            .send()
            .await
            .context("failed to send drop collection request")?
            .json()
            .await
            .context("failed to parse drop collection response")?;

        if response.code != 0 {
            anyhow::bail!(
                "drop collection failed: {}",
                response
                    .message
                    .unwrap_or_else(|| "unknown error".to_string())
            );
        }

        Ok(())
    }
}

pub(crate) fn milvus_id_for_chunk_id(id: &str) -> i64 {
    let digest = Sha256::digest(id.as_bytes());
    let mut bytes = [0_u8; 8];
    bytes.copy_from_slice(&digest[..8]);
    (u64::from_be_bytes(bytes) & (i64::MAX as u64)) as i64
}

fn milvus_search_id_to_string(value: serde_json::Value) -> String {
    match value {
        serde_json::Value::String(s) => s,
        serde_json::Value::Number(n) => n.to_string(),
        other => other.to_string(),
    }
}

impl SearchResultsData {
    fn into_hits(self) -> Vec<SearchResultItem> {
        match self {
            SearchResultsData::Flat(items) => items,
            SearchResultsData::Nested(groups) => groups.into_iter().next().unwrap_or_default(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{SearchResponse, milvus_id_for_chunk_id};

    #[test]
    fn milvus_ids_are_stable_positive_i64s() {
        let id = milvus_id_for_chunk_id("chunk-1");
        assert!(id >= 0);
        assert_eq!(id, milvus_id_for_chunk_id("chunk-1"));
        assert_ne!(id, milvus_id_for_chunk_id("chunk-2"));
    }

    #[test]
    fn search_response_accepts_flat_results() {
        let response: SearchResponse = serde_json::from_value(serde_json::json!({
            "code": 0,
            "data": [
                { "id": 123, "distance": 0.0, "content": "hello", "metadata": {} }
            ]
        }))
        .unwrap();

        assert_eq!(response.data.unwrap().into_hits().len(), 1);
    }
}
