//! MCP tool definitions for codebase indexing and semantic search.

use std::path::PathBuf;

use rmcp::{
    handler::server::{tool::ToolRouter, wrapper::Parameters},
    model::{ServerCapabilities, ServerInfo},
    tool, tool_router, ErrorData as McpError, Json, ServerHandler,
};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use crate::types::{IndexState, IndexStatus};

// ============================================================================
// Tool Input Schemas
// ============================================================================

/// Parameters for indexing a codebase.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct IndexCodebaseParams {
    /// Absolute path to the codebase directory to index.
    pub path: String,
    /// Force re-indexing even if an index already exists.
    #[serde(default)]
    pub force: bool,
}

/// Parameters for searching indexed code.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct SearchCodeParams {
    /// Absolute path to the indexed codebase.
    pub path: String,
    /// Natural language or code query to search for.
    pub query: String,
    /// Maximum number of results to return.
    #[serde(default = "default_limit")]
    pub limit: u32,
}

fn default_limit() -> u32 {
    10
}

/// Parameters for checking indexing status.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct GetIndexingStatusParams {
    /// Absolute path to the codebase to check.
    pub path: String,
}

/// Parameters for clearing an index.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct ClearIndexParams {
    /// Absolute path to the codebase whose index should be cleared.
    pub path: String,
}

// ============================================================================
// Tool Output Schemas
// ============================================================================

/// Result of an indexing operation.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct IndexResult {
    /// Whether the indexing operation succeeded.
    pub success: bool,
    /// Human-readable message describing the result.
    pub message: String,
    /// Path that was indexed.
    pub path: PathBuf,
    /// Number of files indexed.
    pub files_indexed: usize,
    /// Number of code chunks created.
    pub chunks_created: usize,
}

/// A single search result from semantic code search.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct SearchResultItem {
    /// Path to the file containing the match.
    pub file_path: PathBuf,
    /// Path relative to the repository root.
    pub relative_path: String,
    /// The matching code snippet.
    pub content: String,
    /// Starting line number (1-indexed).
    pub start_line: u32,
    /// Ending line number (1-indexed, inclusive).
    pub end_line: u32,
    /// Programming language of the code.
    pub language: String,
    /// Similarity score (0.0 to 1.0, higher is more relevant).
    pub score: f32,
}

/// Result of a search operation.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct SearchResults {
    /// The search results.
    pub results: Vec<SearchResultItem>,
    /// Number of results returned.
    pub count: usize,
}

/// Result of clearing an index.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct ClearResult {
    /// Whether the clear operation succeeded.
    pub success: bool,
    /// Human-readable message describing the result.
    pub message: String,
    /// Path whose index was cleared.
    pub path: PathBuf,
}

// ============================================================================
// Tool Handler
// ============================================================================

/// MCP tool handler for codebase indexing and semantic search.
#[derive(Clone)]
pub struct CodebaseTools {
    tool_router: ToolRouter<Self>,
}

impl CodebaseTools {
    /// Create a new tool handler instance.
    pub fn new() -> Self {
        Self {
            tool_router: Self::tool_router(),
        }
    }

    /// Get the tool router for this handler.
    pub fn router(&self) -> &ToolRouter<Self> {
        &self.tool_router
    }
}

impl Default for CodebaseTools {
    fn default() -> Self {
        Self::new()
    }
}

impl ServerHandler for CodebaseTools {
    fn get_info(&self) -> ServerInfo {
        ServerInfo {
            instructions: Some("Semantic code indexing and search MCP server".into()),
            capabilities: ServerCapabilities::builder().enable_tools().build(),
            ..Default::default()
        }
    }
}

#[tool_router]
impl CodebaseTools {
    /// Index a codebase for semantic search.
    ///
    /// Walks the directory tree, extracts code chunks using tree-sitter,
    /// generates embeddings, and stores them in a vector database.
    #[tool(
        name = "index_codebase",
        description = "Index a codebase directory for semantic code search. Walks the directory, \
                       parses source files, extracts code chunks, and generates embeddings. \
                       Use force=true to re-index an already indexed codebase."
    )]
    async fn index_codebase(
        &self,
        params: Parameters<IndexCodebaseParams>,
    ) -> Result<Json<IndexResult>, McpError> {
        let params = params.0;
        let path = PathBuf::from(&params.path);

        // Validate path exists and is a directory
        if !path.exists() {
            return Err(McpError::invalid_params(
                format!("Path does not exist: {}", params.path),
                None,
            ));
        }
        if !path.is_dir() {
            return Err(McpError::invalid_params(
                format!("Path is not a directory: {}", params.path),
                None,
            ));
        }

        // TODO: Implement actual indexing logic
        // This is a placeholder that will be connected to the indexing pipeline
        Ok(Json(IndexResult {
            success: true,
            message: format!(
                "Indexing {} (force={})",
                params.path, params.force
            ),
            path,
            files_indexed: 0,
            chunks_created: 0,
        }))
    }

    /// Search indexed code using semantic similarity.
    ///
    /// Converts the query to an embedding and finds the most similar
    /// code chunks in the vector database.
    #[tool(
        name = "search_code",
        description = "Search indexed code using natural language or code queries. \
                       Returns the most semantically similar code chunks from the indexed codebase."
    )]
    async fn search_code(
        &self,
        params: Parameters<SearchCodeParams>,
    ) -> Result<Json<SearchResults>, McpError> {
        let params = params.0;
        let path = PathBuf::from(&params.path);

        // Validate path
        if !path.exists() {
            return Err(McpError::invalid_params(
                format!("Path does not exist: {}", params.path),
                None,
            ));
        }

        // Validate limit
        if params.limit == 0 {
            return Err(McpError::invalid_params(
                "Limit must be greater than 0".to_string(),
                None,
            ));
        }

        // TODO: Implement actual search logic
        // This is a placeholder that will be connected to the vector database
        Ok(Json(SearchResults {
            results: vec![],
            count: 0,
        }))
    }

    /// Get the current indexing status for a codebase.
    #[tool(
        name = "get_indexing_status",
        description = "Check the indexing status of a codebase. Returns information about \
                       whether indexing is in progress, completed, or not started."
    )]
    async fn get_indexing_status(
        &self,
        params: Parameters<GetIndexingStatusParams>,
    ) -> Result<Json<IndexStatus>, McpError> {
        let params = params.0;
        let path = PathBuf::from(&params.path);

        // Validate path
        if !path.exists() {
            return Err(McpError::invalid_params(
                format!("Path does not exist: {}", params.path),
                None,
            ));
        }

        // TODO: Implement actual status retrieval
        // This is a placeholder that will be connected to the indexing state
        Ok(Json(IndexStatus {
            total_files: 0,
            processed_files: 0,
            total_chunks: 0,
            status: IndexState::Idle,
        }))
    }

    /// Clear the index for a codebase.
    #[tool(
        name = "clear_index",
        description = "Remove the index for a codebase, freeing up storage. \
                       The codebase will need to be re-indexed before searching."
    )]
    async fn clear_index(
        &self,
        params: Parameters<ClearIndexParams>,
    ) -> Result<Json<ClearResult>, McpError> {
        let params = params.0;
        let path = PathBuf::from(&params.path);

        // Validate path
        if !path.exists() {
            return Err(McpError::invalid_params(
                format!("Path does not exist: {}", params.path),
                None,
            ));
        }

        // TODO: Implement actual index clearing
        // This is a placeholder that will be connected to the vector database
        Ok(Json(ClearResult {
            success: true,
            message: format!("Cleared index for {}", params.path),
            path,
        }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_limit() {
        assert_eq!(default_limit(), 10);
    }

    #[test]
    fn test_index_params_default_force() {
        let json = r#"{"path": "/tmp/test"}"#;
        let params: IndexCodebaseParams = serde_json::from_str(json).unwrap();
        assert!(!params.force);
    }

    #[test]
    fn test_search_params_default_limit() {
        let json = r#"{"path": "/tmp/test", "query": "find main function"}"#;
        let params: SearchCodeParams = serde_json::from_str(json).unwrap();
        assert_eq!(params.limit, 10);
    }

    #[test]
    fn test_codebase_tools_creation() {
        let tools = CodebaseTools::new();
        let all_tools = tools.router().list_all();
        assert_eq!(all_tools.len(), 4); // index_codebase, search_code, get_indexing_status, clear_index
    }
}
