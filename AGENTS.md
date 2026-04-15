# RClaude-Context

High-performance Rust MCP server for semantic code indexing. Replaces the JS-based Claude Context MCP with a single native binary — no Node.js overhead, no per-agent CPU cost. One persistent process handles indexing + search for the entire fleet.

Implements a walker → splitter → embedder → Milvus pipeline with BM25 lexical search and hybrid fusion for codebase search via Claude Code.

## Architecture

```
┌─────────┐    ┌──────────┐    ┌──────────┐    ┌────────┐
│ Walker  │───>│ Splitter │───>│ Embedder │───>│ Milvus │
│ (files) │    │ (chunks) │    │ (vectors)│    │ (store)│
└─────────┘    └──────────┘    └──────────┘    └────────┘
                                   │
                              ┌────────────┐
                              │  Lexical   │
                              │  (BM25)    │
                              └────────────┘
                                   │
                              ┌────────────┐
                              │  Hybrid    │
                              │  Fusion    │
                              └────────────┘
```

### Walker (`src/walker/mod.rs`)

Parallel file discovery using the `ignore` crate with native .gitignore support. Filters by extension during traversal. Auto-detects CPU cores for parallelism. Returns sorted paths for deterministic indexing.

### Splitter (`src/splitter/`)

Tree-sitter AST parsing for semantic code chunking. Extracts functions, classes, structs, traits, impl blocks per language. Splits oversized chunks at line boundaries with configurable overlap.

**Supported languages:** Python, JavaScript, TypeScript, TSX, Rust, Go, Java, C++, C

**Key modules:**
- `parsers.rs` - Language parser initialization
- `node_types.rs` - AST node type definitions per language
- `extractor.rs` - Recursive AST walker for chunk extraction
- `refine.rs` - Chunk size management and splitting

### Embedder (`src/embedding/mod.rs`)

HTTP client for text-to-vector embeddings. Default endpoint: `http://localhost:8100/v1/embeddings`. Batches 100 texts per request. Connection pool: 32 idle per host.

### Milvus (`src/vectordb/`)

RESTful client for Milvus vector database. COSINE similarity metric. Batches 500 documents per insert. Schema: id, content, vector, metadata (JSON).

### Lexical Search (`src/lexical/mod.rs`)

Tantivy-based BM25 index for keyword/symbol search. Stored in-memory per collection. Supports insert, search, delete, and clear operations.

### Hybrid Fusion (`src/mcp/hybrid.rs`)

Reciprocal Rank Fusion (RRF) combining semantic (Milvus) and lexical (BM25) results. Configurable weights, top-k, and fusion constant.

### Incremental Indexing (`src/mcp/manifest.rs`)

File-hash manifest stored at `.rclaude-context/index-manifest.json`. Tracks SHA-256 per file to skip unchanged files on reindex. Supports force-reindex to bypass the manifest.

## Key Files

- `src/main.rs` - MCP server entry point, stdio transport
- `src/types.rs` - CodeChunk, EmbeddingVector, IndexStatus
- `src/config.rs` - Walker/splitter configuration
- `src/mcp/state.rs` - Shared async state, concurrent index tracking
- `src/mcp/indexer.rs` - 3-phase indexing pipeline with incremental support
- `src/mcp/hybrid.rs` - Hybrid search (semantic + lexical fusion)
- `src/mcp/manifest.rs` - Index manifest for incremental reindexing
- `src/mcp/tools.rs` - MCP tool definitions and JSON schemas

## MCP Tools

1. **index_codebase** - Walk, split, embed, and store a directory (incremental by default)
2. **search_code** - Hybrid search (semantic + lexical) over indexed codebase
3. **get_indexing_status** - Check indexing progress
4. **clear_index** - Remove indexed data (vector + lexical)

## Running

```bash
# Build release binary
cargo build --release

# Run as MCP server (stdio transport)
./target/release/rclaude-context
```

The server communicates via stdin/stdout. Configure in Claude Code's MCP settings.

## Tests

Unit tests are embedded in source modules:

```bash
# Run all tests
cargo test

# Run specific module tests
cargo test walker
cargo test splitter
cargo test embedding
```

## Performance

### Batch Sizes

- File processing (rayon): 64 files — L3 cache efficiency
- Embedding API: 100 texts — API throughput
- Milvus insert: 500 docs — network efficiency

### Optimizations

- **Walker:** ignore crate's internal thread pool, filters during traversal
- **Splitter:** rayon par_chunks for CPU-parallel AST parsing
- **Status updates:** Every 10 files to reduce lock contention
- **Connection pooling:** 32 idle HTTP connections per host
- **LTO:** Link-time optimization in release builds

### Configuration Defaults

- `max_file_size`: 1MB
- `chunk_size`: 2500 chars
- `overlap`: 300 chars
- `parallelism`: 0 (auto-detect CPU cores)

## Dependencies

**Core:** rmcp, tokio, rayon, ignore
**Parsing:** tree-sitter + 9 language grammars
**HTTP:** reqwest with connection pooling
**Concurrency:** dashmap, parking_lot
**Lexical:** tantivy

## rmcp Usage Patterns

This project uses rmcp 0.12. Key patterns:

```rust
// Tool handler with router
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

// Main: use Router to wrap handler and serve
let handler = MyHandler::new();
let router = Router::new(handler);
let service = router.serve((tokio::io::stdin(), tokio::io::stdout())).await?;
service.waiting().await?;
```

Key imports: `rmcp::{tool, tool_router, ServiceExt, handler::server::{tool::ToolRouter, router::Router}}`.

The Router wraps the handler and provides Service<RoleServer> implementation.

## Code Quality

**Less code is better.** This codebase should be minimal and focused:

- Prefer single-file modules over directory structures
- Delete unused code rather than commenting it out
- No placeholder/stub implementations - working code only
- Every line must serve a purpose
- If something can be done in 10 lines instead of 100, do it in 10
- Avoid abstractions until needed 3+ times
- No backwards compatibility shims - just change the code

**Before adding code, ask:**
1. Does this need to exist at all?
2. Can this be simpler?
3. Is there duplication to eliminate?

**The goal is a fast, minimal MCP server - not a framework.**
