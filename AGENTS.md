# rust_sindexer (Rust Semantic Indexer)

High-performance Rust MCP server for semantic code indexing. Drop-in replacement for the JS-based Claude Context MCP — single native binary, no Node.js overhead.

Implements a walker → splitter → embedder → vector DB pipeline with BM25 lexical search and hybrid fusion.

## Architecture

```
Walker (files) → Splitter (chunks) → Embedder (vectors) → Vector DB (store)
                                          │
                                     Lexical (BM25)
                                          │
                                     Hybrid Fusion (RRF)
```

## Components

**Walker** (`src/walker/mod.rs`) — Parallel file discovery using the `ignore` crate with native .gitignore support. Filters by extension during traversal. Auto-detects CPU cores.

**Splitter** (`src/splitter/`) — Tree-sitter AST parsing for semantic code chunking. Extracts functions, classes, structs, traits, impl blocks per language. Splits oversized chunks at line boundaries with configurable overlap.
- `parsers.rs` — language parser initialization
- `node_types.rs` — AST node type definitions per language
- `extractor.rs` — recursive AST walker for chunk extraction
- `refine.rs` — chunk size management and splitting

Supported languages: Python, JavaScript, TypeScript, TSX, Rust, Go, Java, C++, C, Ruby, PHP, Swift, Scala, C#

**Embedder** (`src/embedding/mod.rs`) — HTTP client for any OpenAI-compatible embedding API. Batches 100 texts per request. Connection pool: 32 idle per host. Works with any provider that speaks the `/v1/embeddings` format.

Tested providers:
- Jina AI (`jina-code-embeddings-1.5b`) — free tier, good for code
- OpenAI (`text-embedding-3-small`, `text-embedding-3-large`)
- Local servers (ollama, vLLM, TEI, sentence-transformers)
- Any OpenAI-compatible endpoint

**Vector DB** (`src/vectordb/`) — RESTful client for Milvus vector database. COSINE similarity metric. Batches 500 documents per insert. Schema: id, content, vector, metadata (JSON).

Tested backends:
- Zilliz Cloud (managed Milvus, free tier available)
- Self-hosted Milvus via Docker
- Any Milvus-compatible REST API

**Lexical Search** (`src/lexical/mod.rs`) — Tantivy-based BM25 index for keyword/symbol search. Stored in-memory per collection.

**Hybrid Fusion** (`src/mcp/hybrid.rs`) — Reciprocal Rank Fusion (RRF) combining semantic and lexical results. Configurable weights, top-k, and fusion constant.

**Incremental Indexing** (`src/mcp/manifest.rs`) — File-hash manifest stored at `.rust_sindexer/index-manifest.json`. Tracks SHA-256 per file to skip unchanged files on reindex. Pass `force: true` to bypass.

## Key Files

- `src/main.rs` — MCP server entry point, stdio transport
- `src/types.rs` — CodeChunk, EmbeddingVector, IndexStatus
- `src/config.rs` — walker/splitter configuration
- `src/mcp/state.rs` — shared async state, concurrent index tracking
- `src/mcp/indexer.rs` — 3-phase indexing pipeline with incremental support
- `src/mcp/hybrid.rs` — hybrid search (semantic + lexical fusion)
- `src/mcp/manifest.rs` — index manifest for incremental reindexing
- `src/mcp/tools.rs` — MCP tool definitions and JSON schemas

## MCP Tools

- **index_codebase** — walk, split, embed, and store a directory (incremental by default)
- **search_code** — hybrid search (semantic + lexical) over indexed codebase
- **get_indexing_status** — check indexing progress
- **clear_index** — remove indexed data (vector + lexical)

## Environment Variables

- `EMBEDDING_URL` — embedding API base URL (default: `http://localhost:8080/v1`). Any OpenAI-compatible endpoint.
- `EMBEDDING_API_KEY` — API key for the embedding endpoint. Not needed for local servers.
- `EMBEDDING_MODEL` — model name to request from the embedding API.
- `EMBEDDING_DIMENSION` — vector dimension (default: `384`). Must match your model's output.
- `MILVUS_URL` — Milvus/Zilliz Cloud endpoint (default: `http://localhost:19530`).
- `MILVUS_TOKEN` — authentication token for Milvus. Not needed for local unauthenticated instances.
- `MAX_FILE_SIZE` — maximum file size to process in bytes (default: `1048576`, 1MB).
- `LOG_LEVEL` — logging verbosity: trace/debug/info/warn/error (default: `info`).

## Running

```bash
cargo build --release
./target/release/rust_sindexer
```

Communicates via stdin/stdout (MCP stdio transport).

## Tests

```bash
cargo test              # all tests
cargo test walker       # file discovery
cargo test splitter     # AST parsing
cargo test embedding    # embedding client
```

## Performance

- Walker: `ignore` crate's thread pool with extension filtering during traversal
- Splitter: rayon par_chunks for CPU-parallel AST parsing
- Embedding: 100 texts per API batch
- Vector insert: 500 docs per Milvus batch
- Connection pooling: 32 idle HTTP connections per host
- LTO enabled in release builds

Defaults: 1MB max file size, 2500-char chunks, 300-char overlap, auto-detect CPU cores.

## Dependencies

- **Core:** rmcp, tokio, rayon, ignore
- **Parsing:** tree-sitter + language grammars
- **HTTP:** reqwest with connection pooling and rustls-tls
- **Concurrency:** dashmap, parking_lot
- **Lexical:** tantivy

## rmcp Usage Patterns

This project uses rmcp 0.12. Key pattern:

```rust
#[derive(Clone)]
pub struct MyHandler {
    tool_router: ToolRouter<Self>,
}

#[tool_router]
impl MyHandler {
    fn new() -> Self {
        Self { tool_router: Self::tool_router() }
    }

    #[tool(description = "Does something")]
    async fn my_tool(&self, params: Parameters<MyParams>) -> Result<Json<MyResult>, McpError> {
        // ...
    }
}

// Main: Router wraps handler and serves over stdio
let router = Router::new(MyHandler::new());
let service = router.serve((tokio::io::stdin(), tokio::io::stdout())).await?;
service.waiting().await?;
```

Key imports: `rmcp::{tool, tool_router, ServiceExt, handler::server::{tool::ToolRouter, router::Router}}`.

## Code Quality

**Less code is better.** This codebase should be minimal and focused:

- Prefer single-file modules over directory structures
- Delete unused code rather than commenting it out
- No placeholder/stub implementations — working code only
- Every line must serve a purpose
- If something can be done in 10 lines instead of 100, do it in 10
- Avoid abstractions until needed 3+ times
- No backwards compatibility shims — just change the code

**The goal is a fast, minimal MCP server — not a framework.**
