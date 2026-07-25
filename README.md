# (Rust Semantic Indexer) rust_sindexer

A high-performance Rust MCP server for semantic code indexing and search. Drop-in replacement for [`@zilliz/claude-context-mcp`](https://www.npmjs.com/package/@zilliz/claude-context-mcp) — single native binary, no Node.js required.

Works with zero configuration: by default it serves BM25 lexical search backed by a local on-disk vector store. Set `EMBEDDING_URL` to enable semantic search, and `MILVUS_URL` to back it with Milvus/Zilliz Cloud at scale.

## Why rust_sindexer?

The official JS-based Claude Context MCP has several pain points:

- **Node.js overhead** — each `npx` invocation spawns a full Node.js process
- **No real incremental indexing** — re-indexes everything on each run
- **Shallow .gitignore support** — only root-level, no nested ignore files
- **Startup latency** — npm download/cache step on cold starts
- **gRPC timeout issues** — `DEADLINE_EXCEEDED` errors during indexing

rust_sindexer fixes all of these:

- **Single binary** — no runtime dependencies, ~37MB native executable
- **Incremental updates** — SHA-256 manifest tracks file changes, and `update_index` touches only changed/deleted files without falling back to a full rebuild
- **Full .gitignore support** — via the `ignore` crate with nested directory support
- **Instant startup** — native binary, no package manager involved
- **REST API for Milvus** — no gRPC, no timeout issues
- **Parallel everything** — Rayon-based parallel file walking, AST parsing, and chunk extraction
- **Hybrid search** — BM25 lexical + semantic vector search with Reciprocal Rank Fusion
- **Any embedding provider** — works with any OpenAI-compatible API (cloud or local)

## Supported Languages

Tree-sitter parsers for AST-aware code chunking:

Python, JavaScript, TypeScript, TSX, Rust, Go, Java, C, C++, Ruby, PHP, Swift, Scala, C#

## Installation

### From source

```bash
git clone https://github.com/RESMP-DEV/rust_sindexer
cd rust_sindexer
cargo build --release
# Binary at target/release/sindexer
```

### Via cargo install

```bash
cargo install --path .
```

## Host compatibility

- **Apple Silicon macOS (including M-series / M4)** — supported via the standard `cargo build` / `cargo test` flow.
- **Linux infra hosts** — supported via the same source build and CI matrix.
- **GPU hosts** — supported without CUDA- or Metal-specific code in this binary; point `EMBEDDING_URL` (or `OPENAI_BASE_URL`) at any OpenAI-compatible GPU-backed embeddings service such as vLLM, TEI, Ollama, or Jina.

The binary itself stays CPU-only and talks to embedding providers over HTTP, so the same build works on local laptops, infra boxes, and GPU-serving hosts.

## Operating modes

- **Lexical only (default)** — no environment variables needed. BM25 keyword/symbol search over a Tantivy index. Good for exact matches and code navigation.
- **Semantic + lexical** — set `EMBEDDING_URL`. Hybrid RRF fusion of semantic similarity and BM25. The local vector store handles project-scale indexes (roughly up to 50K chunks).
- **Full scale** — set `EMBEDDING_URL` and `MILVUS_URL` to use Milvus/Zilliz Cloud as the vector backend for large deployments.

## Embedding provider (optional)

Semantic search needs an embedding service — any service that speaks the OpenAI `/v1/embeddings` format:

**Cloud providers:**

```bash
# OpenAI
EMBEDDING_URL=https://api.openai.com/v1
EMBEDDING_API_KEY=sk-xxx
EMBEDDING_MODEL=text-embedding-3-small
EMBEDDING_DIMENSION=1536

# Jina AI (free tier available)
EMBEDDING_URL=https://api.jina.ai/v1
EMBEDDING_API_KEY=jina_xxx
EMBEDDING_MODEL=jina-code-embeddings-1.5b
EMBEDDING_DIMENSION=1536

# Voyage AI
EMBEDDING_URL=https://api.voyageai.com/v1
EMBEDDING_API_KEY=pa-xxx
EMBEDDING_MODEL=voyage-code-3
EMBEDDING_DIMENSION=1024

# Cohere (via OpenAI-compatible proxy)
# Google Gemini (via OpenAI-compatible proxy)
# Any other provider with an OpenAI-compatible endpoint
```

**Local / self-hosted:**

```bash
# Ollama
ollama pull nomic-embed-text
EMBEDDING_URL=http://localhost:11434/v1
EMBEDDING_MODEL=nomic-embed-text
EMBEDDING_DIMENSION=768

# vLLM
EMBEDDING_URL=http://localhost:8000/v1
EMBEDDING_MODEL=BAAI/bge-base-en-v1.5
EMBEDDING_DIMENSION=768

# Text Embeddings Inference (TEI)
EMBEDDING_URL=http://localhost:8080/v1
EMBEDDING_MODEL=BAAI/bge-base-en-v1.5
EMBEDDING_DIMENSION=768

# sentence-transformers
EMBEDDING_URL=http://localhost:8080/v1
EMBEDDING_MODEL=all-MiniLM-L6-v2
EMBEDDING_DIMENSION=384
```

No API key is needed for local servers — just omit `EMBEDDING_API_KEY`.

## Vector database (optional)

By default vectors persist to a local JSON store under `~/.cache/sindexer/` (or `$XDG_CACHE_HOME/sindexer/`) — no external database needed. For large deployments, point `MILVUS_URL` at Milvus, either managed or self-hosted:

```bash
# Zilliz Cloud (managed Milvus, free tier available)
MILVUS_URL=https://your-cluster.zillizcloud.com:443
MILVUS_TOKEN=your-api-key

# Self-hosted Milvus via Docker
docker run -d --name milvus -p 19530:19530 -p 9091:9091 milvusdb/milvus:latest
MILVUS_URL=http://localhost:19530
# No MILVUS_TOKEN needed for local unauthenticated instances
```

## Configuration

All configuration is via environment variables. Canonical names are preferred, and these compatibility aliases are also accepted:

- `OPENAI_BASE_URL` → `EMBEDDING_URL`
- `OPENAI_API_KEY` → `EMBEDDING_API_KEY`
- `MILVUS_ADDRESS` → `MILVUS_URL`

Empty values are treated as unset.

**Connection:**

- `EMBEDDING_URL` — embedding API base URL. When unset or empty, semantic search is disabled.
- `EMBEDDING_API_KEY` — optional API key for the embedding endpoint (omit for local servers)
- `EMBEDDING_MODEL` — model name to request (default: `all-minilm`)
- `EMBEDDING_DIMENSION` — vector dimension (default: `384`). Must match your model's output.
- `MILVUS_URL` — Milvus endpoint. When unset or empty, the local vector store is used.
- `MILVUS_TOKEN` — optional Milvus auth token (omit for local unauthenticated instances)
- `SINDEXER_COLLECTION_IDENTITY` — stable collection namespace so multiple hosts can intentionally share collections for the same codebases; requires `SINDEXER_COLLECTION_ROOT`
- `SINDEXER_COLLECTION_ROOT` — host-local root path governed by the stable identity. The root gets the identity directly, while separately indexed descendants append their relative paths and receive distinct collections. Paths outside the root use their own absolute-path identities.

**Tuning:**

- `MAX_FILE_SIZE` — max file size to process in bytes (default: `1048576` / 1MB)
- `CHUNK_SIZE` — maximum chunk size in bytes (default: `512`)
- `CHUNK_OVERLAP` — chunk overlap in characters, applied as whole lines at an assumed 80 characters per line (`CHUNK_OVERLAP / 80` lines). Values below `80` — including the default `64` — round down to zero overlap; set `160` for two lines of overlap, etc.
- `BATCH_SIZE` — texts per embedding API request (default: `32`)
- `INDEXING_CONCURRENCY` — max concurrent embedding/insert operations (default: `32`; `CONCURRENCY` is accepted as an alias)
- `EMBEDDING_RPM` / `EMBEDDING_TPM` — client-side rate limits for the embedding API, enforced with token buckets plus retry/backoff on 429s (defaults: `400` requests/min, `1600000` tokens/min; `0` = unlimited)
- `FOLLOW_SYMLINKS` — follow symbolic links during traversal (default: `false`)
- `RUST_LOG` — standard tracing filter, e.g. `RUST_LOG=debug` (default level: `info`; logs go to stderr, keeping stdout clean for MCP)

## Usage

### Claude Code (user scope)

```bash
claude mcp add claude-context --scope user -- /path/to/sindexer
```

Or add the server block to `~/.claude.json` directly:

```json
{
  "mcpServers": {
    "claude-context": {
      "type": "stdio",
      "command": "/path/to/sindexer",
      "args": [],
      "env": {
        "EMBEDDING_URL": "https://api.openai.com/v1",
        "EMBEDDING_API_KEY": "your-key",
        "EMBEDDING_MODEL": "text-embedding-3-small",
        "EMBEDDING_DIMENSION": "1536",
        "MILVUS_URL": "https://your-cluster.zillizcloud.com:443",
        "MILVUS_TOKEN": "your-token"
      }
    }
  }
}
```

The `env` block is optional — with no environment variables the server runs in lexical-only mode.

### Project-level

Add the same `mcpServers` block to `.mcp.json` in the project root.

### Other MCP clients

Any stdio MCP client (Codex, Copilot, Claude Desktop, etc.) works with the same command; declare it wherever that client configures MCP servers (e.g. `~/.codex/config.toml` for Codex).

### MCP Tools

Once configured, the server exposes eight tools:

- **`index_codebase`** — Indexes a directory. It may perform an initial full build when no compatible index exists; pass `force: true` only when a full rebuild is intended. Poll `get_indexing_status` until the status becomes `completed` or `failed`.
- **`update_index`** — Incrementally updates an existing compatible index by touching only changed/deleted files. It refuses to perform an initial or full rebuild if the manifest or backing collection is missing or incompatible.
- **`search_code`** — Hybrid search (semantic + BM25 lexical) with Reciprocal Rank Fusion. Returns code chunks with file paths, line numbers, and relevance scores. `limit` defaults to 10, and an optional `extensions` filter (e.g. `["rs", "py"]`) restricts results by file type. If semantic search is unavailable but the lexical index exists, the server falls back to lexical-only results.
- **`get_indexing_status`** — Report the current state for a path: `idle`, `indexing`, `completed`, or `failed`, with progress counters.
- **`clear_index`** — Remove all indexed data (vectors + lexical index) for a codebase.
- **`list_collections`** — List all vector collections with row counts.
- **`collection_stats`** — Row count for a specific collection.
- **`drop_collection`** — Permanently delete a specific collection by name.

### MCP-specific behavior

`index_codebase` intentionally returns quickly instead of holding the MCP stdio request open until the whole repository is indexed. Long-running MCP tool calls are a common source of client-side failures and timeouts even when the underlying indexing pipeline works correctly.

Use this pattern:

1. Call `index_codebase` with an absolute repository path.
2. Poll `get_indexing_status` for the same absolute path.
3. Start calling `search_code` once the status is `completed`.

All tool paths must be absolute filesystem paths. Relative paths are rejected.

## Migrating from @zilliz/claude-context-mcp

### 1. Build the binary

```bash
git clone https://github.com/RESMP-DEV/rust_sindexer
cd rust_sindexer
cargo build --release
```

### 2. Replace your MCP configuration

The server name stays `claude-context` so existing tool references keep working.

**Before** (JS version):

```json
{
  "mcpServers": {
    "claude-context": {
      "command": "npx",
      "args": ["@zilliz/claude-context-mcp@latest"],
      "env": {
        "OPENAI_API_KEY": "sk-...",
        "MILVUS_ADDRESS": "https://your-cluster.zillizcloud.com",
        "MILVUS_TOKEN": "your-token"
      }
    }
  }
}
```

**After** (rust_sindexer):

```json
{
  "mcpServers": {
    "claude-context": {
      "type": "stdio",
      "command": "/path/to/sindexer",
      "args": [],
      "env": {
        "EMBEDDING_URL": "https://api.openai.com/v1",
        "EMBEDDING_API_KEY": "sk-...",
        "EMBEDDING_MODEL": "text-embedding-3-small",
        "EMBEDDING_DIMENSION": "1536",
        "MILVUS_URL": "https://your-cluster.zillizcloud.com:443",
        "MILVUS_TOKEN": "your-token"
      }
    }
  }
}
```

### 3. Clear and re-index

Since the embedding model and chunk boundaries differ, clear your existing index and re-index:

```
> clear_index for /path/to/project
> index_codebase for /path/to/project
> get_indexing_status for /path/to/project
```

For routine refreshes after that first build, call `update_index` so only
changed/deleted files are touched and the tool refuses full rebuild fallback.

### Environment variable mapping

The JS version uses different variable names. Here's the mapping:

- `OPENAI_API_KEY` → `EMBEDDING_API_KEY` (still accepted directly for compatibility)
- `OPENAI_BASE_URL` → `EMBEDDING_URL` (still accepted directly for compatibility)
- `EMBEDDING_MODEL` → `EMBEDDING_MODEL` (same name)
- (new) `EMBEDDING_DIMENSION` — defaults to `384`; must match your model's output dimension
- `MILVUS_ADDRESS` → `MILVUS_URL` (still accepted directly for compatibility)
- `MILVUS_TOKEN` → `MILVUS_TOKEN` (same name)

### What stays the same

- Tool names: `index_codebase`, `search_code`, `get_indexing_status`, `clear_index`
- Core parameters: `path`, `query`, `limit`, `force`
- Project-level `.mcp.json` files

### What changes

- New tools: `update_index`, `list_collections`, `collection_stats`, `drop_collection`
- `extensionFilter: [".ts", ".py"]` becomes `extensions: ["ts", "py"]` (no dot prefix)
- No `EMBEDDING_PROVIDER` selection — uses any OpenAI-compatible HTTP endpoint
- No required `~/.context/.env` file — any environment injection mechanism works
- Everything is optional — with no environment variables at all, the server still serves lexical search

## Architecture

```
                     ┌─────────────────┐
                     │   MCP Client    │
                     │ (Claude, Codex, │
                     │  Copilot, etc.) │
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
                              ┌───────────────┼──────────────┐
                              │                              │
                     ┌────────▼────────┐           ┌────────▼────────┐
                     │  Vector Store   │           │    Tantivy      │
                     │ (Local / Milvus)│           │   (BM25 Index)  │
                     └─────────────────┘           └─────────────────┘
                              │                              │
                              └──────────────┬───────────────┘
                                    ┌────────▼────────┐
                                    │  Hybrid Fusion  │
                                    │   (RRF Merge)   │
                                    └─────────────────┘
```

## On-disk layout

- `<repo>/.sindexer/index-manifest.json` — SHA-256 per-file manifest driving incremental updates
- `<repo>/.sindexer/index-status.json` — persisted indexing status
- `~/.cache/sindexer/` (or `$XDG_CACHE_HOME/sindexer/`) — Tantivy lexical indexes and local vector store persistence, keyed by codebase path

## Tests

```bash
cargo test              # All tests
cargo test walker       # File discovery
cargo test splitter     # AST parsing
cargo test embedding    # Embedding client
cargo test local        # Local vector store
cargo test lexical      # BM25 search
```

## License

MIT
