# sindexer

**sindexer** is a fast, self-contained code search engine for AI coding
assistants. It runs as an
[MCP (Model Context Protocol) server](https://modelcontextprotocol.io): point
it at a repository and it builds an index of your code, which any MCP-compatible
client — Claude Code, Claude Desktop, Codex, Copilot, and others — can then
search to find the functions, classes, and files relevant to whatever you're
working on. It's a single native binary with no Node.js, no Docker, and no
accounts required: with zero configuration it serves keyword (BM25) search out
of a local on-disk index, and you can optionally plug in an embedding service
for natural-language semantic search and Milvus/Zilliz for large-scale
deployments. It is the repository's supported Rust Index CLI and the only
indexing runtime.

**Who it's for:** developers using AI coding tools who want their assistant to
retrieve *actual relevant code* from large repositories instead of guessing.
If you've used "codebase indexing" features in AI editors, this is the
self-hosted, bring-your-own-provider version of that.

## Highlights

- **Zero-config by default** — no environment variables needed. Keyword/symbol
  search (BM25 via Tantivy) backed by a local on-disk vector store.
- **Single native binary** — ~37 MB executable, no runtime dependencies,
  instant startup. No `npx` download step, no Node.js process overhead.
- **Optional semantic search** — set `EMBEDDING_URL` to any OpenAI-compatible
  embeddings API (OpenAI, Jina, Voyage, Ollama, vLLM, TEI, ...) and get hybrid
  search: semantic similarity + BM25 fused with Reciprocal Rank Fusion.
- **Incremental indexing** — a SHA-256 per-file manifest means refreshes touch
  only changed and deleted files.
- **AST-aware chunking** — tree-sitter parsers split code along function,
  class, and impl boundaries for 14 languages, so search hits are whole
  logical units, not arbitrary line windows.
- **Full .gitignore support** — nested ignore files are honored during
  traversal, plus a built-in skip list (`node_modules`, `target`, `dist`, ...).
- **Parallel pipeline** — Rayon-based file walking, parsing, and chunk
  extraction; concurrent embedding with client-side rate limiting.
- **Scales when you need it to** — set `MILVUS_URL` to store vectors in
  Milvus/Zilliz Cloud over a plain REST API (no gRPC, no timeout surprises).

## Supported languages

Tree-sitter AST parsing: Python, JavaScript, TypeScript, TSX, Rust, Go, Java,
C, C++, Ruby, PHP, Swift, Scala, C#.

60+ file types in total — including Kotlin, Lua, SQL, YAML, Markdown,
Dockerfile, and Makefile — are indexed with line/heading-based chunking. See
`SUPPORTED_EXTENSIONS` and `EXTENSIONLESS_FILES` in `src/config.rs` for the
full list.

## Installation

### Prebuilt binaries

Release builds are published to
[GitHub Releases](https://github.com/RESMP-DEV/rust_sindexer/releases) for:

| Platform | Asset suffix |
| --- | --- |
| Linux x86_64 | `x86_64-unknown-linux-gnu.tar.gz` |
| Linux ARM64 | `aarch64-unknown-linux-gnu.tar.gz` |
| macOS Intel | `x86_64-apple-darwin.tar.gz` |
| macOS Apple Silicon | `aarch64-apple-darwin.tar.gz` |
| Windows x86_64 | `x86_64-pc-windows-msvc.zip` |

Assets are named `sindexer-<tag>-<target>.{tar.gz|zip}` with a matching
`.sha256` checksum file. Download the archive for your platform from the
latest release page, verify it, and extract:

```bash
# Example: Linux x86_64 (copy the real asset URL from the release page)
curl -LO https://github.com/RESMP-DEV/rust_sindexer/releases/download/<tag>/sindexer-<tag>-x86_64-unknown-linux-gnu.tar.gz
curl -LO https://github.com/RESMP-DEV/rust_sindexer/releases/download/<tag>/sindexer-<tag>-x86_64-unknown-linux-gnu.tar.gz.sha256
sha256sum -c sindexer-*-x86_64-unknown-linux-gnu.tar.gz.sha256
tar xzf sindexer-*-x86_64-unknown-linux-gnu.tar.gz
sudo install sindexer-*/sindexer /usr/local/bin/
sindexer --version
```

### From source

Requires [Rust](https://rustup.rs) (stable).

```bash
git clone https://github.com/RESMP-DEV/rust_sindexer
cd rust_sindexer
cargo build --release
./target/release/sindexer --version
```

### Via cargo install

```bash
git clone https://github.com/RESMP-DEV/rust_sindexer
cd rust_sindexer
cargo install --path .
# Installs the binary as `sindexer` into ~/.cargo/bin
```

## Quickstart

The binary is an MCP *server*: it communicates over stdin/stdout and is
launched by an MCP client. You don't interact with it like a normal CLI.

### 1. Register it with your AI tool

**Claude Code** (user scope, available in all projects):

```bash
claude mcp add code-indexer --scope user -- /usr/local/bin/sindexer
```

**Any client that reads an `mcpServers` block** (project-level `.mcp.json`,
`~/.claude.json`, Claude Desktop config, ...):

```json
{
  "mcpServers": {
    "code-indexer": {
      "type": "stdio",
      "command": "/usr/local/bin/sindexer",
      "args": []
    }
  }
}
```

More ready-to-edit configs live in [`examples/mcp-servers.json`](examples/mcp-servers.json).

### 2. Index your project, then search

Once your client is connected, just ask it — the tools below are exposed
automatically. Conceptually:

```
> index_codebase for /home/you/projects/myapp
> get_indexing_status for /home/you/projects/myapp      # poll until "completed"
> search_code for "where are HTTP retries handled?" in /home/you/projects/myapp
```

In plain chat: *"Index this repository, then find where authentication tokens
are validated."* Results come back with file paths, line ranges, code
snippets, and relevance scores.

That's it — no environment variables, no external services. Keyword search
works immediately; add `EMBEDDING_URL` later if you want natural-language
matching (see [Configuration](#configuration)).

### No MCP client handy? Try the raw demo

[`examples/demo.sh`](examples/demo.sh) drives the server directly over raw
JSON-RPC: it creates a tiny throwaway project, indexes it, runs a search, and
prints every response.

```bash
cargo build --release
./examples/demo.sh
```

## Usage

### MCP tools

The server exposes eight tools to connected clients. All `path` parameters
must be **absolute** filesystem paths; relative paths are rejected.

| Tool | Parameters | What it does |
| --- | --- | --- |
| `index_codebase` | `path`, `force` (default `false`) | Index a directory. Performs an initial full build when no compatible index exists; pass `force: true` only when a full rebuild is intended. Returns quickly — poll `get_indexing_status` for completion. |
| `update_index` | `path` | Refresh an index incrementally when compatible; if this codebase's manifest or scoped collection is missing or incompatible, safely rebuild only that codebase. |
| `search_code` | `path`, `query`, `limit` (default `10`), `extensions` (e.g. `["rs", "py"]`) | Hybrid search (semantic + BM25, fused via RRF). Returns chunks with file paths, line numbers, language, and scores. Falls back to lexical-only when no embedding service is configured. |
| `get_indexing_status` | `path` | Current state for a path: `idle`, `indexing`, `completed`, or `failed`, with progress counters. |
| `clear_index` | `path` | Remove all indexed data (vectors + lexical index) for a codebase. |
| `list_collections` | — | List all vector collections with row counts. |
| `collection_stats` | `collection_name` | Row count for a specific collection. |
| `drop_collection` | `collection_name` | Permanently delete a collection by name. |

### Typical workflow

`index_codebase` intentionally returns quickly instead of holding the MCP
request open until the whole repository is indexed — long-running tool calls
are a common source of client-side timeouts. Use this pattern:

1. Call `index_codebase` with an absolute repository path.
2. Poll `get_indexing_status` for the same path.
3. Start calling `search_code` once the status is `completed`.
4. After code changes, call `update_index` for a fast incremental refresh.

### Search examples

```jsonc
// Natural-language query (semantic mode), Rust files only, top 5
{
  "path": "/home/you/projects/myapp",
  "query": "how does the retry backoff work",
  "limit": 5,
  "extensions": ["rs"]
}

// Exact symbol lookup works great in lexical-only mode
{
  "path": "/home/you/projects/myapp",
  "query": "parse_config_file"
}
```

Each hit includes `file_path`, `relative_path`, `content`, `start_line`,
`end_line`, `language`, and `score`.

## Configuration

All configuration is via environment variables, and **all of it is optional**.
Empty values are treated as unset. Compatibility aliases are accepted:
`OPENAI_BASE_URL` → `EMBEDDING_URL`, `OPENAI_API_KEY` → `EMBEDDING_API_KEY`,
`MILVUS_ADDRESS` → `MILVUS_URL`.

### Operating modes

| Mode | Env vars | Behavior |
| --- | --- | --- |
| Lexical only (default) | none | BM25 keyword/symbol search over a Tantivy index. Great for exact matches and code navigation. |
| Semantic + lexical | `EMBEDDING_URL` | Hybrid search: semantic similarity + BM25 fused with RRF. The local vector store comfortably handles project-scale indexes (~up to 50K chunks). |
| Full scale | `EMBEDDING_URL` + `MILVUS_URL` | Same hybrid search, with vectors stored in Milvus/Zilliz Cloud for large deployments. |

### Environment variables

**Connection:**

| Variable | Default | Description |
| --- | --- | --- |
| `EMBEDDING_URL` | unset | Embedding API base URL. Setting this enables semantic search. Any OpenAI-compatible `/v1/embeddings` endpoint works. |
| `EMBEDDING_API_KEY` | unset | API key for the embedding endpoint. Omit for local servers. |
| `EMBEDDING_MODEL` | `all-minilm` | Model name to request. |
| `EMBEDDING_DIMENSION` | `384` | Vector dimension. **Must match your model's output.** |
| `EMBEDDING_QUERY_PREFIX` | unset | Optional text prepended to semantic search queries; whitespace is preserved. |
| `EMBEDDING_PASSAGE_PREFIX` | unset | Optional text prepended to indexed code chunks; whitespace is preserved. |
| `MILVUS_URL` | unset | Milvus/Zilliz endpoint. When unset, the local vector store is used. |
| `MILVUS_TOKEN` | unset | Milvus/Zilliz auth token. Omit for local unauthenticated instances. |
| `SINDEXER_COLLECTION_IDENTITY` | unset | Stable collection namespace so multiple hosts can intentionally share collections for the same codebase. Requires `SINDEXER_COLLECTION_ROOT`. |
| `SINDEXER_COLLECTION_ROOT` | unset | Host-local root path governed by the stable identity. The root gets the identity directly; separately indexed descendants append their relative paths and receive distinct collections. Paths outside the root use their own absolute-path identities. |

**Tuning:**

| Variable | Default | Description |
| --- | --- | --- |
| `MAX_FILE_SIZE` | `1048576` (1 MB) | Max file size to process, in bytes. |
| `CHUNK_SIZE` | `512` | Maximum chunk size in bytes. |
| `CHUNK_OVERLAP` | `64` | Chunk overlap in characters, applied as whole lines at an assumed 80 chars/line (`CHUNK_OVERLAP / 80` lines). Values below `80` — including the default `64` — round down to zero overlap; set `160` for two lines of overlap, etc. |
| `BATCH_SIZE` | `32` | Texts per embedding API request. |
| `INDEXING_CONCURRENCY` | `32` | Max concurrent embedding/insert operations (`CONCURRENCY` also accepted). |
| `EMBEDDING_RPM` | `400` | Client-side embedding requests/min rate limit (`0` = unlimited). Enforced with token buckets plus retry/backoff on 429s. |
| `EMBEDDING_TPM` | `1600000` | Client-side embedding tokens/min rate limit (`0` = unlimited). |
| `FOLLOW_SYMLINKS` | `false` | Follow symbolic links during traversal (`1`/`true` to enable). |
| `RUST_LOG` | `info` | Standard tracing filter, e.g. `RUST_LOG=debug`. Logs go to **stderr**, keeping stdout clean for MCP traffic. |

### Embedding providers (optional)

Semantic search needs any service that speaks the OpenAI `/v1/embeddings`
format.

**Cloud providers:**

```bash
# OpenAI
EMBEDDING_URL=https://api.openai.com/v1
EMBEDDING_API_KEY=sk-xxx
EMBEDDING_MODEL=text-embedding-3-small
EMBEDDING_DIMENSION=1536

# Jina AI (free tier available; code-specialized model)
EMBEDDING_URL=https://api.jina.ai/v1
EMBEDDING_API_KEY=jina_xxx
EMBEDDING_MODEL=jina-code-embeddings-1.5b
EMBEDDING_DIMENSION=1536

# Voyage AI
EMBEDDING_URL=https://api.voyageai.com/v1
EMBEDDING_API_KEY=pa-xxx
EMBEDDING_MODEL=voyage-code-3
EMBEDDING_DIMENSION=1024
```

**Local / self-hosted** (no API key needed — just omit `EMBEDDING_API_KEY`):

```bash
# Ollama
ollama pull nomic-embed-text
EMBEDDING_URL=http://localhost:11434/v1
EMBEDDING_MODEL=nomic-embed-text
EMBEDDING_DIMENSION=768

# vLLM / Text Embeddings Inference / sentence-transformers
EMBEDDING_URL=http://localhost:8000/v1
EMBEDDING_MODEL=BAAI/bge-base-en-v1.5
EMBEDDING_DIMENSION=768

# Native RESMP.DEV calibrated Jina Code MXFP4 server
EMBEDDING_URL=http://localhost:1235/v1
EMBEDDING_MODEL=jina-code-embeddings-1.5b-block-gptq-mxfp4-32k
EMBEDDING_DIMENSION=1536
EMBEDDING_QUERY_PREFIX=$'Find the most relevant code snippet given the following query:\n'
EMBEDDING_PASSAGE_PREFIX=$'Candidate code snippet:\n'
```

The `$'...'` form is Bash syntax for embedding a real newline; ordinary quoted
`"...\n"` text would pass a literal backslash and `n` to the embedding server.
Changing `EMBEDDING_PASSAGE_PREFIX` invalidates the index manifest. `update_index`
detects the incompatibility and performs a full rebuild scoped to that codebase,
without touching unrelated collections.

### Vector storage (optional)

By default vectors persist to a local JSON store under `~/.cache/sindexer/`
(or `$XDG_CACHE_HOME/sindexer/`) — no external database needed (~75 MB for
50K chunks at 384 dimensions). For large deployments, point `MILVUS_URL` at
Milvus, managed or self-hosted:

```bash
# Zilliz Cloud (managed Milvus, free tier available)
MILVUS_URL=https://your-cluster.zillizcloud.com:443
MILVUS_TOKEN=your-api-key

# Self-hosted Milvus via Docker
docker run -d --name milvus -p 19530:19530 -p 9091:9091 milvusdb/milvus:latest
MILVUS_URL=http://localhost:19530
```

For the exact REST endpoints sindexer uses, see [`docs/milvus-api.md`](docs/milvus-api.md).

## Building from source & development

```bash
cargo build --release     # binary at target/release/sindexer
cargo test                # all tests
cargo test walker         # file discovery
cargo test splitter       # AST parsing
cargo test embedding      # embedding client
cargo test local          # local vector store
cargo test lexical        # BM25 search
cargo clippy -- -D warnings
cargo fmt -- --check
```

CI runs the test suite on Linux x86_64 and macOS Apple Silicon, plus clippy
and rustfmt gates (`.github/workflows/ci.yml`), and a weekly supply-chain
audit with cargo-deny/cargo-audit (`.github/workflows/supply-chain.yml`).

**Host compatibility:** the binary itself is CPU-only and talks to embedding
providers over HTTP, so the same build works on laptops, infra boxes, and
GPU-serving hosts (point `EMBEDDING_URL` at vLLM/TEI/Ollama/Jina on the GPU
side). Apple Silicon macOS and Linux are covered by CI; Windows builds are
produced by the release workflow.

**Project layout:**

| Path | Purpose |
| --- | --- |
| `src/main.rs` | MCP server entry point (rmcp stdio transport) |
| `src/api.rs` | Library API (`sindexer::Sindexer`) for embedding indexing/search in Rust programs |
| `src/config.rs` | Environment configuration, supported extensions, ignore patterns |
| `src/walker/` | Parallel file discovery (`ignore` crate, .gitignore-aware) |
| `src/splitter/` | Tree-sitter AST chunking |
| `src/embedding/` | OpenAI-compatible embedding client with rate limiting |
| `src/lexical/` | Tantivy BM25 index |
| `src/vectordb/` | Local JSON vector store + Milvus/Zilliz REST client |
| `src/mcp/` | MCP tools, indexing pipeline, hybrid fusion, incremental manifest |
| `examples/` | Ready-to-use MCP configs and a raw JSON-RPC demo script |

## Troubleshooting & FAQ

**I ran `./sindexer` in a terminal and nothing happens.**
That's expected. sindexer is an MCP *server*: it reads JSON-RPC requests from
stdin and writes responses to stdout, so it sits quietly until an MCP client
drives it. Register it with an MCP client (see [Quickstart](#quickstart)), or
try it without one via `./examples/demo.sh`. `sindexer --help` prints a
summary; `RUST_LOG=debug ./sindexer` shows lifecycle logs on stderr.

**"Path must be absolute" errors.**
All tool paths must be absolute filesystem paths (`/home/you/project`, not
`./project` or `~/project`). Relative paths are rejected by design.

**`search_code` returns no results.**
In order: (1) index the project first — call `index_codebase`, then poll
`get_indexing_status` until it reports `completed`; (2) check your
`extensions` filter isn't excluding everything (no dot prefix: `["rs"]`, not
`[".rs"]`); (3) files larger than `MAX_FILE_SIZE` (default 1 MB) are skipped;
(4) gitignored files and built-in skip directories (`node_modules`, `target`,
`dist`, `build`, `__pycache__`, virtualenvs, ...) are never indexed.

**How do I get natural-language ("semantic") search?**
Set `EMBEDDING_URL` (plus `EMBEDDING_API_KEY`/`EMBEDDING_MODEL`/
`EMBEDDING_DIMENSION` as needed) and re-index. If you see the message
*"(lexical only; set EMBEDDING_URL for semantic search)"* after indexing, the
server is running in keyword-only mode. Lexical-only search is still quite
good for exact symbol names.

**Embedding requests fail or vectors don't insert.**
`EMBEDDING_DIMENSION` must exactly match your model's output dimension
(e.g. `1536` for `text-embedding-3-small` and Jina code embeddings, `768` for
`nomic-embed-text`). A mismatch is the most common cause of broken semantic
indexes — after fixing it, re-index with `force: true`.

**Where is my index stored? How do I start over?**
Per-repo metadata lives in `<repo>/.sindexer/` (file-hash manifest and
status); vector and lexical data live under `~/.cache/sindexer/` (or
`$XDG_CACHE_HOME/sindexer/`). To reset cleanly: call the `clear_index` tool
for the path, or delete those directories. With Milvus configured, vectors
live in a collection named after the path — use `list_collections` /
`drop_collection` to manage them.

**How do I see what's going on?**
Logs go to stderr (stdout is reserved for MCP protocol traffic). Set
`RUST_LOG=debug` for verbose output, e.g. in your MCP client's `env` block.

**Milvus/Zilliz authentication errors.**
Set `MILVUS_TOKEN` to your API key. Self-hosted Milvus without auth doesn't
need it. Note sindexer talks to Milvus over its REST API (port `19530`
serves both), so no gRPC client libraries are required.

**Can multiple machines share one index?**
With Milvus/Zilliz, yes: set `SINDEXER_COLLECTION_IDENTITY` and
`SINDEXER_COLLECTION_ROOT` so different checkout paths on different hosts map
to the same collection. The local on-disk store is per-machine.

## Configure the Rust Index CLI

### 1. Get the binary

From [GitHub Releases](https://github.com/RESMP-DEV/rust_sindexer/releases)
or build from source:

```bash
git clone https://github.com/RESMP-DEV/rust_sindexer
cd rust_sindexer
cargo build --release
```

### 2. Replace your MCP configuration

Register the server as `sindexer`; do not retain stale wrapper or provider names.

**Before** (JS version):

```json
{
  "mcpServers": {
    "sindexer": {
      "command": "npx",
      "args": [],
      "env": {
        "OPENAI_API_KEY": "sk-...",
        "MILVUS_ADDRESS": "https://your-cluster.zillizcloud.com",
        "MILVUS_TOKEN": "your-token"
      }
    }
  }
}
```

**After** (sindexer):

```json
{
  "mcpServers": {
    "sindexer": {
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

Since the embedding model and chunk boundaries differ, clear your existing
index and re-index:

```
> clear_index for /path/to/project
> index_codebase for /path/to/project
> get_indexing_status for /path/to/project
```

For routine refreshes after that first build, call `update_index` so only
changed/deleted files are touched.

### Environment variable mapping

The old variable names are still accepted directly, so strictly speaking you
only *need* to add `EMBEDDING_DIMENSION`:

- `OPENAI_API_KEY` → `EMBEDDING_API_KEY` (old name still accepted)
- `OPENAI_BASE_URL` → `EMBEDDING_URL` (old name still accepted)
- `EMBEDDING_MODEL` → `EMBEDDING_MODEL` (same)
- (new) `EMBEDDING_DIMENSION` — defaults to `384`; must match your model's output
- `MILVUS_ADDRESS` → `MILVUS_URL` (old name still accepted)
- `MILVUS_TOKEN` → `MILVUS_TOKEN` (same)

### What stays the same

- Tool names: `index_codebase`, `search_code`, `get_indexing_status`, `clear_index`
- Core parameters: `path`, `query`, `limit`, `force`
- Project-level `.mcp.json` files

### What changes

- New tools: `update_index`, `list_collections`, `collection_stats`, `drop_collection`
- `extensionFilter: [".ts", ".py"]` becomes `extensions: ["ts", "py"]` (no dot prefix)
- No `EMBEDDING_PROVIDER` selection — any OpenAI-compatible HTTP endpoint works
- No required `~/.context/.env` file — any environment injection mechanism works
- Everything is optional — with no environment variables at all, the server still serves lexical search

## Architecture

```
                     ┌─────────────────┐
                     │   MCP Client    │
                     │ (Claude, Codex, │
                     │  Copilot, etc.) │
                     └────────┬────────┘
                              │ stdio (newline-delimited JSON-RPC)
                     ┌────────▼────────┐
                     │    sindexer     │
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

When embeddings are disabled, the pipeline stops after splitting and only
populates the lexical index.

## On-disk layout

- `<repo>/.sindexer/index-manifest.json` — SHA-256 per-file manifest driving incremental updates
- `<repo>/.sindexer/index-status.json` — persisted indexing status
- `~/.cache/sindexer/` (or `$XDG_CACHE_HOME/sindexer/`) — Tantivy lexical indexes and local vector store persistence, keyed by codebase path

## License

MIT
