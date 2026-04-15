# rust_sindexer (Rust Semantic Indexer)

High-performance Rust MCP server for semantic code context retrieval. Indexes codebases using tree-sitter for AST-aware chunking, generates embeddings, and stores them in Milvus for fast similarity search.

## Why Rust?

The JavaScript reference implementation processes files sequentially. rust_sindexer uses Rayon for parallel file walking, parsing, and chunk extraction, achieving significant speedups on multi-core systems:

- **File discovery:** Parallel with `ignore` crate (vs sequential glob in JS)
- **AST parsing:** Parallel across all cores (vs one file at a time)
- **Chunk extraction:** Parallel per-file (vs sequential)
- **Embedding batching:** Adaptive concurrent batches (vs fixed batch size)

For a 10,000-file codebase on an 8-core machine, expect roughly 4-6x faster indexing compared to the sequential approach.

## Supported Languages

Tree-sitter parsers are included for:
- Python
- JavaScript / TypeScript / TSX / JSX
- Rust
- Go
- Java
- C / C++

Each language has defined "splittable" node types (functions, classes, methods, etc.) that are extracted as semantic chunks rather than arbitrary line-based splits.

## Prerequisites

### Milvus

rust_sindexer requires a running Milvus instance for vector storage:

```bash
# Using Docker
docker run -d --name milvus \
  -p 19530:19530 \
  -p 9091:9091 \
  milvusdb/milvus:latest

# Or using Docker Compose (recommended for production)
# See: https://milvus.io/docs/install_standalone-docker.md
```

### Embedding Server

An HTTP embedding server must be running to generate vectors. The server should accept POST requests with JSON body `{"texts": ["..."]}` and return `{"embeddings": [[...]]}`.

Example using a local model:

```bash
# Using sentence-transformers
uvx sentence-transformers serve --model all-MiniLM-L6-v2 --port 8080
```

## Installation

### From Source

```bash
git clone https://github.com/RESMP-DEV/rust_sindexer
cd rust_sindexer
cargo build --release

# Binary will be at target/release/rust-sindexer
```

### Using Cargo

```bash
cargo install rust-sindexer
```

## Configuration

Environment variables:

- `MILVUS_HOST` — Milvus server hostname (default: `localhost`)
- `MILVUS_PORT` — Milvus gRPC port (default: `19530`)
- `EMBEDDING_URL` — Embedding server endpoint (default: `http://localhost:8080/embed`)
- `EMBEDDING_DIM` — Embedding vector dimension (default: `384`)
- `INDEX_PARALLELISM` — Number of indexing threads, 0 = auto (default: `0`)
- `MAX_FILE_SIZE` — Maximum file size to process in bytes (default: `1048576`)
- `LOG_LEVEL` — Logging verbosity: trace/debug/info/warn/error (default: `info`)

## Usage with Claude Code

Add to your Claude Code MCP configuration (`~/.claude/claude_desktop_config.json` or project `.mcp.json`):

```json
{
  "mcpServers": {
    "code-context": {
      "command": "rust-sindexer",
      "args": [],
      "env": {
        "MILVUS_HOST": "localhost",
        "EMBEDDING_URL": "http://localhost:8080/embed"
      }
    }
  }
}
```

### Available Tools

Once configured, the MCP server exposes:

- `index_codebase` - Index a directory, creating embeddings for all code chunks
- `search_code` - Hybrid semantic + lexical search across indexed code
- `get_indexing_status` - Check indexing progress
- `clear_index` - Remove indexed data

## Architecture

```
                     ┌─────────────────┐
                     │   Claude Code   │
                     │   (MCP Client)  │
                     └────────┬────────┘
                              │ stdio
                     ┌────────▼────────┐
                     │  rust_sindexer  │
                     │   (MCP Server)  │
                     └────────┬────────┘
              ┌───────────────┼───────────────┐
              │               │               │
     ┌────────▼────────┐ ┌────▼────┐ ┌────────▼────────┐
     │  File Walker    │ │ Splitter│ │   Embedding     │
     │ (ignore + rayon)│ │ (tree-  │ │   Client        │
     │                 │ │ sitter) │ │   (reqwest)     │
     └─────────────────┘ └─────────┘ └────────┬────────┘
                                              │
                                     ┌────────▼────────┐
                                     │     Milvus      │
                                     │  (Vector Store) │
                                     └─────────────────┘
```

## License

MIT
