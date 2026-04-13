# RClaude-Context

High-performance Rust MCP server for semantic code context retrieval. Indexes codebases using tree-sitter for AST-aware chunking, generates embeddings, and stores them in Milvus for fast similarity search.

## Why Rust?

The JavaScript reference implementation processes files sequentially. RClaude-Context uses Rayon for parallel file walking, parsing, and chunk extraction, achieving significant speedups on multi-core systems:

| Operation | JavaScript | Rust |
|-----------|------------|------|
| File discovery | Sequential glob | Parallel with `ignore` crate |
| AST parsing | One file at a time | Parallel across all cores |
| Chunk extraction | Sequential | Parallel per-file |
| Embedding batching | Fixed batch size | Adaptive concurrent batches |

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

RClaude-Context requires a running Milvus instance for vector storage:

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
git clone https://github.com/your-org/rclaude-context
cd rclaude-context
cargo build --release

# Binary will be at target/release/rclaude-context
```

### Using Cargo

```bash
cargo install rclaude-context
```

## Configuration

Environment variables:

| Variable | Description | Default |
|----------|-------------|---------|
| `MILVUS_HOST` | Milvus server hostname | `localhost` |
| `MILVUS_PORT` | Milvus gRPC port | `19530` |
| `EMBEDDING_URL` | Embedding server endpoint | `http://localhost:8080/embed` |
| `EMBEDDING_DIM` | Embedding vector dimension | `384` |
| `INDEX_PARALLELISM` | Number of indexing threads (0 = auto) | `0` |
| `MAX_FILE_SIZE` | Maximum file size to process (bytes) | `1048576` |
| `LOG_LEVEL` | Logging verbosity (trace/debug/info/warn/error) | `info` |

## Usage with Claude Code

Add to your Claude Code MCP configuration (`~/.claude/claude_desktop_config.json` or project `.mcp.json`):

```json
{
  "mcpServers": {
    "code-context": {
      "command": "rclaude-context",
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
- `search_code` - Semantic search across indexed code
- `get_index_status` - Check indexing progress
- `list_collections` - List indexed codebases

## Architecture

```
                     ┌─────────────────┐
                     │   Claude Code   │
                     │   (MCP Client)  │
                     └────────┬────────┘
                              │ stdio
                     ┌────────▼────────┐
                     │  RClaude-Context│
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

## Performance Benchmarks

*TODO: Add benchmarks comparing indexing speed and query latency against the JavaScript implementation.*

## License

MIT
