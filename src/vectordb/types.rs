//! Milvus API types for vector database operations.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Error types for Milvus operations.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(tag = "type", content = "message")]
pub enum MilvusError {
    /// Collection already exists.
    CollectionAlreadyExists(String),
    /// Collection not found.
    CollectionNotFound(String),
    /// Invalid schema definition.
    InvalidSchema(String),
    /// Invalid field type or value.
    InvalidField(String),
    /// Vector dimension mismatch.
    DimensionMismatch { expected: usize, got: usize },
    /// Index not built or not ready.
    IndexNotReady(String),
    /// Connection or network error.
    ConnectionError(String),
    /// Authentication or authorization failure.
    AuthenticationError(String),
    /// Rate limit exceeded.
    RateLimitExceeded { retry_after_ms: u64 },
    /// Invalid search parameters.
    InvalidSearchParams(String),
    /// Data insertion failure.
    InsertionFailed(String),
    /// Deletion failure.
    DeletionFailed(String),
    /// Internal server error.
    InternalError(String),
    /// Request timeout.
    Timeout { operation: String, elapsed_ms: u64 },
    /// Unknown or unclassified error.
    Unknown(String),
}

impl std::fmt::Display for MilvusError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::CollectionAlreadyExists(name) => {
                write!(f, "collection already exists: {name}")
            }
            Self::CollectionNotFound(name) => write!(f, "collection not found: {name}"),
            Self::InvalidSchema(msg) => write!(f, "invalid schema: {msg}"),
            Self::InvalidField(msg) => write!(f, "invalid field: {msg}"),
            Self::DimensionMismatch { expected, got } => {
                write!(f, "dimension mismatch: expected {expected}, got {got}")
            }
            Self::IndexNotReady(name) => write!(f, "index not ready: {name}"),
            Self::ConnectionError(msg) => write!(f, "connection error: {msg}"),
            Self::AuthenticationError(msg) => write!(f, "authentication error: {msg}"),
            Self::RateLimitExceeded { retry_after_ms } => {
                write!(f, "rate limit exceeded, retry after {retry_after_ms}ms")
            }
            Self::InvalidSearchParams(msg) => write!(f, "invalid search params: {msg}"),
            Self::InsertionFailed(msg) => write!(f, "insertion failed: {msg}"),
            Self::DeletionFailed(msg) => write!(f, "deletion failed: {msg}"),
            Self::InternalError(msg) => write!(f, "internal error: {msg}"),
            Self::Timeout {
                operation,
                elapsed_ms,
            } => write!(f, "timeout on {operation} after {elapsed_ms}ms"),
            Self::Unknown(msg) => write!(f, "unknown error: {msg}"),
        }
    }
}

impl std::error::Error for MilvusError {}

/// Supported data types for Milvus fields.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum DataType {
    Bool,
    Int8,
    Int16,
    Int32,
    Int64,
    Float,
    Double,
    String,
    VarChar,
    Json,
    Array,
    BinaryVector,
    FloatVector,
    Float16Vector,
    BFloat16Vector,
    SparseFloatVector,
}

/// Index type for vector fields.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum IndexType {
    /// Flat index (brute force).
    Flat,
    /// Inverted file index.
    IvfFlat,
    /// IVF with scalar quantization.
    IvfSq8,
    /// IVF with product quantization.
    IvfPq,
    /// Hierarchical Navigable Small World graph.
    Hnsw,
    /// RHNSW with flat refinement.
    RhnswFlat,
    /// RHNSW with PQ.
    RhnswPq,
    /// RHNSW with SQ.
    RhnswSq,
    /// Approximate Nearest Neighbor Oh Yeah.
    Annoy,
    /// DiskANN for large-scale datasets.
    DiskAnn,
    /// Auto-index (server decides).
    AutoIndex,
    /// GPU IVF Flat.
    GpuIvfFlat,
    /// GPU IVF PQ.
    GpuIvfPq,
    /// GPU Brute Force.
    GpuBruteForce,
    /// GPU CAGRA.
    GpuCagra,
    /// Sparse inverted index.
    SparseInvertedIndex,
    /// Sparse WAND.
    SparseWand,
}

/// Metric type for similarity search.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum MetricType {
    /// L2 (Euclidean) distance.
    L2,
    /// Inner product.
    Ip,
    /// Cosine similarity.
    Cosine,
    /// Hamming distance (binary vectors).
    Hamming,
    /// Jaccard distance (binary vectors).
    Jaccard,
    /// Tanimoto distance (binary vectors).
    Tanimoto,
    /// Substructure (binary vectors).
    Substructure,
    /// Superstructure (binary vectors).
    Superstructure,
}

/// Schema definition for a single field.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FieldSchema {
    /// Field name.
    pub name: String,
    /// Data type.
    pub data_type: DataType,
    /// Whether this field is the primary key.
    #[serde(default)]
    pub is_primary_key: bool,
    /// Whether to auto-generate IDs (only for primary key fields).
    #[serde(default)]
    pub auto_id: bool,
    /// Description of the field.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    /// Maximum length for VarChar fields.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_length: Option<u32>,
    /// Dimension for vector fields.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub dim: Option<u32>,
    /// Element data type for Array fields.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub element_type: Option<DataType>,
    /// Maximum capacity for Array fields.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_capacity: Option<u32>,
    /// Whether this field can be null.
    #[serde(default)]
    pub nullable: bool,
    /// Default value for the field.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub default_value: Option<serde_json::Value>,
}

impl FieldSchema {
    /// Create a new primary key field with Int64 type.
    pub fn primary_key(name: impl Into<String>, auto_id: bool) -> Self {
        Self {
            name: name.into(),
            data_type: DataType::Int64,
            is_primary_key: true,
            auto_id,
            description: None,
            max_length: None,
            dim: None,
            element_type: None,
            max_capacity: None,
            nullable: false,
            default_value: None,
        }
    }

    /// Create a new float vector field.
    pub fn float_vector(name: impl Into<String>, dim: u32) -> Self {
        Self {
            name: name.into(),
            data_type: DataType::FloatVector,
            is_primary_key: false,
            auto_id: false,
            description: None,
            max_length: None,
            dim: Some(dim),
            element_type: None,
            max_capacity: None,
            nullable: false,
            default_value: None,
        }
    }

    /// Create a new VarChar field.
    pub fn varchar(name: impl Into<String>, max_length: u32) -> Self {
        Self {
            name: name.into(),
            data_type: DataType::VarChar,
            is_primary_key: false,
            auto_id: false,
            description: None,
            max_length: Some(max_length),
            dim: None,
            element_type: None,
            max_capacity: None,
            nullable: false,
            default_value: None,
        }
    }

    /// Create a new JSON field.
    pub fn json(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            data_type: DataType::Json,
            is_primary_key: false,
            auto_id: false,
            description: None,
            max_length: None,
            dim: None,
            element_type: None,
            max_capacity: None,
            nullable: false,
            default_value: None,
        }
    }

    /// Set a description for this field.
    pub fn with_description(mut self, description: impl Into<String>) -> Self {
        self.description = Some(description.into());
        self
    }

    /// Set this field as nullable.
    pub fn nullable(mut self) -> Self {
        self.nullable = true;
        self
    }
}

/// Schema definition for a collection.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CollectionSchema {
    /// Collection name.
    pub name: String,
    /// Description of the collection.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    /// Field definitions.
    pub fields: Vec<FieldSchema>,
    /// Enable dynamic fields (schema-free mode).
    #[serde(default)]
    pub enable_dynamic_field: bool,
}

impl CollectionSchema {
    /// Create a new collection schema.
    pub fn new(name: impl Into<String>, fields: Vec<FieldSchema>) -> Self {
        Self {
            name: name.into(),
            description: None,
            fields,
            enable_dynamic_field: false,
        }
    }

    /// Set a description for this collection.
    pub fn with_description(mut self, description: impl Into<String>) -> Self {
        self.description = Some(description.into());
        self
    }

    /// Enable dynamic fields.
    pub fn with_dynamic_fields(mut self) -> Self {
        self.enable_dynamic_field = true;
        self
    }
}

/// Consistency level for read operations.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Default)]
#[serde(rename_all = "PascalCase")]
pub enum ConsistencyLevel {
    /// Strong consistency (linearizable).
    Strong,
    /// Bounded staleness.
    Bounded,
    /// Session consistency.
    Session,
    /// Eventual consistency (fastest).
    #[default]
    Eventually,
}

/// Request to create a new collection.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CreateCollectionRequest {
    /// Collection schema.
    pub schema: CollectionSchema,
    /// Number of shards.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub shards_num: Option<u32>,
    /// Consistency level.
    #[serde(default)]
    pub consistency_level: ConsistencyLevel,
    /// Index parameters for auto-indexing.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub index_params: Option<IndexParams>,
    /// Additional properties.
    #[serde(default, skip_serializing_if = "HashMap::is_empty")]
    pub properties: HashMap<String, String>,
}

impl CreateCollectionRequest {
    /// Create a new collection creation request.
    pub fn new(schema: CollectionSchema) -> Self {
        Self {
            schema,
            shards_num: None,
            consistency_level: ConsistencyLevel::default(),
            index_params: None,
            properties: HashMap::new(),
        }
    }

    /// Set the number of shards.
    pub fn with_shards(mut self, shards: u32) -> Self {
        self.shards_num = Some(shards);
        self
    }

    /// Set the consistency level.
    pub fn with_consistency(mut self, level: ConsistencyLevel) -> Self {
        self.consistency_level = level;
        self
    }

    /// Set index parameters.
    pub fn with_index(mut self, params: IndexParams) -> Self {
        self.index_params = Some(params);
        self
    }
}

/// Index parameters for vector fields.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IndexParams {
    /// Name of the field to index.
    pub field_name: String,
    /// Index type.
    pub index_type: IndexType,
    /// Metric type for similarity.
    pub metric_type: MetricType,
    /// Index-specific parameters.
    #[serde(default, skip_serializing_if = "HashMap::is_empty")]
    pub params: HashMap<String, serde_json::Value>,
}

impl IndexParams {
    /// Create HNSW index parameters.
    pub fn hnsw(field_name: impl Into<String>, metric_type: MetricType, m: u32, ef_construction: u32) -> Self {
        let mut params = HashMap::new();
        params.insert("M".to_string(), serde_json::json!(m));
        params.insert("efConstruction".to_string(), serde_json::json!(ef_construction));
        Self {
            field_name: field_name.into(),
            index_type: IndexType::Hnsw,
            metric_type,
            params,
        }
    }

    /// Create IVF_FLAT index parameters.
    pub fn ivf_flat(field_name: impl Into<String>, metric_type: MetricType, nlist: u32) -> Self {
        let mut params = HashMap::new();
        params.insert("nlist".to_string(), serde_json::json!(nlist));
        Self {
            field_name: field_name.into(),
            index_type: IndexType::IvfFlat,
            metric_type,
            params,
        }
    }

    /// Create flat (brute force) index parameters.
    pub fn flat(field_name: impl Into<String>, metric_type: MetricType) -> Self {
        Self {
            field_name: field_name.into(),
            index_type: IndexType::Flat,
            metric_type,
            params: HashMap::new(),
        }
    }
}

/// A single row of data for insertion.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(transparent)]
pub struct Row {
    /// Field values keyed by field name.
    pub fields: HashMap<String, serde_json::Value>,
}

impl Row {
    /// Create a new empty row.
    pub fn new() -> Self {
        Self {
            fields: HashMap::new(),
        }
    }

    /// Insert a field value.
    pub fn insert(mut self, key: impl Into<String>, value: impl Serialize) -> Self {
        self.fields.insert(
            key.into(),
            serde_json::to_value(value).unwrap_or(serde_json::Value::Null),
        );
        self
    }
}

impl Default for Row {
    fn default() -> Self {
        Self::new()
    }
}

/// Request to insert data into a collection.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InsertRequest {
    /// Collection name.
    pub collection_name: String,
    /// Partition name (optional).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub partition_name: Option<String>,
    /// Rows of data to insert.
    pub rows: Vec<Row>,
}

impl InsertRequest {
    /// Create a new insert request.
    pub fn new(collection_name: impl Into<String>, rows: Vec<Row>) -> Self {
        Self {
            collection_name: collection_name.into(),
            partition_name: None,
            rows,
        }
    }

    /// Specify a partition.
    pub fn with_partition(mut self, partition: impl Into<String>) -> Self {
        self.partition_name = Some(partition.into());
        self
    }
}

/// Result of an insert operation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InsertResult {
    /// Number of rows inserted.
    pub insert_count: u64,
    /// IDs of inserted entities (if auto_id is false or IDs are returned).
    #[serde(default)]
    pub ids: Vec<i64>,
    /// Timestamp of the insert operation.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub timestamp: Option<u64>,
}

/// Search parameters for vector similarity search.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchParams {
    /// Metric type for similarity.
    pub metric_type: MetricType,
    /// Index-specific search parameters.
    #[serde(default, skip_serializing_if = "HashMap::is_empty")]
    pub params: HashMap<String, serde_json::Value>,
}

impl SearchParams {
    /// Create search parameters for HNSW.
    pub fn hnsw(metric_type: MetricType, ef: u32) -> Self {
        let mut params = HashMap::new();
        params.insert("ef".to_string(), serde_json::json!(ef));
        Self { metric_type, params }
    }

    /// Create search parameters for IVF indexes.
    pub fn ivf(metric_type: MetricType, nprobe: u32) -> Self {
        let mut params = HashMap::new();
        params.insert("nprobe".to_string(), serde_json::json!(nprobe));
        Self { metric_type, params }
    }

    /// Create basic search parameters.
    pub fn basic(metric_type: MetricType) -> Self {
        Self {
            metric_type,
            params: HashMap::new(),
        }
    }
}

/// Request to perform a vector similarity search.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchRequest {
    /// Collection name.
    pub collection_name: String,
    /// Partition names to search (empty = all partitions).
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub partition_names: Vec<String>,
    /// Name of the vector field to search.
    pub anns_field: String,
    /// Query vectors.
    pub data: Vec<Vec<f32>>,
    /// Number of nearest neighbors to return.
    pub limit: u32,
    /// Search parameters.
    pub search_params: SearchParams,
    /// Fields to return in results.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub output_fields: Vec<String>,
    /// Filter expression.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub filter: Option<String>,
    /// Consistency level for this search.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub consistency_level: Option<ConsistencyLevel>,
    /// Offset for pagination.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub offset: Option<u32>,
}

impl SearchRequest {
    /// Create a new search request.
    pub fn new(
        collection_name: impl Into<String>,
        anns_field: impl Into<String>,
        vectors: Vec<Vec<f32>>,
        limit: u32,
        search_params: SearchParams,
    ) -> Self {
        Self {
            collection_name: collection_name.into(),
            partition_names: Vec::new(),
            anns_field: anns_field.into(),
            data: vectors,
            limit,
            search_params,
            output_fields: Vec::new(),
            filter: None,
            consistency_level: None,
            offset: None,
        }
    }

    /// Specify output fields to return.
    pub fn with_output_fields(mut self, fields: Vec<String>) -> Self {
        self.output_fields = fields;
        self
    }

    /// Add a filter expression.
    pub fn with_filter(mut self, filter: impl Into<String>) -> Self {
        self.filter = Some(filter.into());
        self
    }

    /// Set pagination offset.
    pub fn with_offset(mut self, offset: u32) -> Self {
        self.offset = Some(offset);
        self
    }

    /// Specify partitions to search.
    pub fn with_partitions(mut self, partitions: Vec<String>) -> Self {
        self.partition_names = partitions;
        self
    }
}

/// A single search hit.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchHit {
    /// Entity ID.
    pub id: i64,
    /// Distance/score (interpretation depends on metric type).
    pub distance: f32,
    /// Output field values.
    #[serde(default)]
    pub entity: HashMap<String, serde_json::Value>,
}

/// Results for a single query vector.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchQueryResult {
    /// Hits for this query.
    pub hits: Vec<SearchHit>,
}

/// Result of a search operation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchResult {
    /// Results for each query vector.
    pub results: Vec<SearchQueryResult>,
    /// Total number of entities searched.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub num_queries: Option<u32>,
}

impl SearchResult {
    /// Get results for the first (or only) query vector.
    pub fn first(&self) -> Option<&SearchQueryResult> {
        self.results.first()
    }

    /// Flatten all hits from all queries.
    pub fn all_hits(&self) -> impl Iterator<Item = &SearchHit> {
        self.results.iter().flat_map(|r| r.hits.iter())
    }
}

/// Request to delete entities from a collection.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeleteRequest {
    /// Collection name.
    pub collection_name: String,
    /// Partition name (optional).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub partition_name: Option<String>,
    /// Filter expression to match entities for deletion.
    pub filter: String,
}

impl DeleteRequest {
    /// Create a delete request with a filter expression.
    pub fn new(collection_name: impl Into<String>, filter: impl Into<String>) -> Self {
        Self {
            collection_name: collection_name.into(),
            partition_name: None,
            filter: filter.into(),
        }
    }

    /// Delete by primary key IDs.
    pub fn by_ids(collection_name: impl Into<String>, pk_field: &str, ids: &[i64]) -> Self {
        let ids_str = ids
            .iter()
            .map(|id| id.to_string())
            .collect::<Vec<_>>()
            .join(", ");
        Self {
            collection_name: collection_name.into(),
            partition_name: None,
            filter: format!("{pk_field} in [{ids_str}]"),
        }
    }

    /// Specify a partition.
    pub fn with_partition(mut self, partition: impl Into<String>) -> Self {
        self.partition_name = Some(partition.into());
        self
    }
}

/// Result of a delete operation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeleteResult {
    /// Number of entities deleted.
    pub delete_count: u64,
    /// Timestamp of the delete operation.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub timestamp: Option<u64>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_field_schema_builders() {
        let pk = FieldSchema::primary_key("id", true);
        assert!(pk.is_primary_key);
        assert!(pk.auto_id);
        assert_eq!(pk.data_type, DataType::Int64);

        let vec = FieldSchema::float_vector("embedding", 768);
        assert_eq!(vec.dim, Some(768));
        assert_eq!(vec.data_type, DataType::FloatVector);

        let text = FieldSchema::varchar("content", 65535).nullable();
        assert!(text.nullable);
        assert_eq!(text.max_length, Some(65535));
    }

    #[test]
    fn test_collection_schema_serialization() {
        let schema = CollectionSchema::new(
            "documents",
            vec![
                FieldSchema::primary_key("id", true),
                FieldSchema::float_vector("embedding", 1536),
                FieldSchema::varchar("text", 65535),
            ],
        )
        .with_description("Document embeddings")
        .with_dynamic_fields();

        let json = serde_json::to_string_pretty(&schema).unwrap();
        assert!(json.contains("\"name\": \"documents\""));
        assert!(json.contains("\"enable_dynamic_field\": true"));

        let deserialized: CollectionSchema = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized.name, "documents");
        assert_eq!(deserialized.fields.len(), 3);
    }

    #[test]
    fn test_insert_request() {
        let rows = vec![
            Row::new()
                .insert("text", "hello world")
                .insert("embedding", vec![0.1f32; 768]),
            Row::new()
                .insert("text", "goodbye world")
                .insert("embedding", vec![0.2f32; 768]),
        ];

        let request = InsertRequest::new("documents", rows).with_partition("partition_a");

        let json = serde_json::to_string(&request).unwrap();
        assert!(json.contains("partition_a"));
    }

    #[test]
    fn test_search_request() {
        let request = SearchRequest::new(
            "documents",
            "embedding",
            vec![vec![0.1f32; 768]],
            10,
            SearchParams::hnsw(MetricType::Cosine, 64),
        )
        .with_output_fields(vec!["text".to_string(), "metadata".to_string()])
        .with_filter("category == 'science'")
        .with_offset(20);

        let json = serde_json::to_string_pretty(&request).unwrap();
        assert!(json.contains("\"limit\": 10"));
        assert!(json.contains("category == 'science'"));
    }

    #[test]
    fn test_delete_by_ids() {
        let request = DeleteRequest::by_ids("documents", "id", &[1, 2, 3, 4, 5]);
        assert!(request.filter.contains("id in [1, 2, 3, 4, 5]"));
    }

    #[test]
    fn test_milvus_error_display() {
        let err = MilvusError::DimensionMismatch {
            expected: 768,
            got: 512,
        };
        assert_eq!(
            err.to_string(),
            "dimension mismatch: expected 768, got 512"
        );

        let err = MilvusError::RateLimitExceeded { retry_after_ms: 5000 };
        assert_eq!(err.to_string(), "rate limit exceeded, retry after 5000ms");
    }

    #[test]
    fn test_index_params_builders() {
        let hnsw = IndexParams::hnsw("embedding", MetricType::Cosine, 16, 256);
        assert_eq!(hnsw.index_type, IndexType::Hnsw);
        assert_eq!(hnsw.params.get("M"), Some(&serde_json::json!(16)));

        let ivf = IndexParams::ivf_flat("embedding", MetricType::L2, 128);
        assert_eq!(ivf.index_type, IndexType::IvfFlat);
        assert_eq!(ivf.params.get("nlist"), Some(&serde_json::json!(128)));
    }
}
