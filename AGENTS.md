# sindexer (Semantic Indexer)

High-performance Rust MCP server + CLI for semantic code indexing. Single
native binary, no Node.js overhead. This file is the repo's only instruction
file (`CLAUDE.md` symlinks to it) and doubles as the **new-machine setup
guide for the full-scale deployment**: Jina code embeddings + Milvus/Zilliz
Cloud vector backend. Lexical-only and local-vector modes work with zero
configuration and are documented in README.md.

## New machine setup (full-scale: Jina + Zilliz)

Prerequisites: macOS (Apple Silicon primary), git, Rust stable, `~/.local/bin`
on PATH, `/usr/local/bin` on PATH, admin rights once (the served-binary
symlink lives in a root-owned directory). Steps are ordered; verify each
before moving on.

1. **Clone to the canonical path.** Collections key on the absolute path
   string (symlinks are NOT resolved), so the checkout location is the index
   identity for CLI and MCP alike:

   ```bash
   git clone git@github.com:RESMP-DEV/rust_sindexer.git \
     ~/AlphaHENG/contrib/rust_sindexer
   ```

   A different path works but orphans existing collections for this repo;
   after moving a checkout, `sindexer clear <old-abs-path>` cleans up (it
   works on deleted paths). `SINDEXER_COLLECTION_ROOT` /
   `SINDEXER_COLLECTION_IDENTITY` scope identity per checkout when needed.

2. **Build and serve the release binary** (LTO profile; this artifact is the
   served binary — rebuild it after pulling changes):

   ```bash
   cargo build --release
   sudo ln -sf "$PWD/target/release/sindexer" /usr/local/bin/sindexer
   ```

3. **Provision `~/.context/.env`** (never committed; the repo holds no
   secrets — copy values from the existing machine's file or the user's
   secret store). The binary reads process env only; the PATH wrapper is
   what sources this file.

   | Variable | Full-scale value | Meaning |
   |---|---|---|
   | `EMBEDDING_URL` | `https://api.jina.ai/v1` | OpenAI-compatible embedding endpoint (`OPENAI_BASE_URL` alias) |
   | `EMBEDDING_API_KEY` | Jina API key | Bearer auth (`OPENAI_API_KEY` alias) |
   | `EMBEDDING_MODEL` | `jina-code-embeddings-1.5b` | Model name |
   | `EMBEDDING_DIMENSION` | `1536` | Must match model output; changing it later invalidates manifests and forces full rebuilds |
   | `MILVUS_URL` | Zilliz cluster endpoint | Enables the Milvus/Zilliz vector store (`MILVUS_ADDRESS` alias) |
   | `MILVUS_TOKEN` | cluster token | Bearer auth |

   Optional: `EMBEDDING_QUERY_PREFIX` / `EMBEDDING_PASSAGE_PREFIX` (task
   prefixes), `EMBEDDING_BATCH_SIZE`, `SINDEXER_COLLECTION_ROOT`,
   `SINDEXER_COLLECTION_IDENTITY`. Other variables that live in the same
   file (`ZILLIZ_CLOUD_*`, `SPLITTER_TYPE`, `EMBEDDING_PROVIDER`, …) belong
   to other fleet tooling; this binary ignores them.

4. **Install the local ops kit** (PATH wrapper, health monitor, watchdog):
   follow `deploy/local/README.md` verbatim — symlink `sindexer`,
   `sindexer-doctor`, and the legacy `rust-indexer` alias into
   `~/.local/bin`, then copy and bootstrap the
   `com.local.sindexer-doctor` LaunchAgent (one `--watch` line every 30
   minutes to `~/Library/Logs/sindexer-doctor.log`). The wrapper sources
   `~/.context/.env` with caller-env-wins semantics and execs
   `/usr/local/bin/sindexer`.

5. **MCP clients (optional).** With no arguments the binary speaks
   newline-delimited JSON-RPC on stdio:

   ```json
   { "mcpServers": { "sindexer": { "command": "/usr/local/bin/sindexer" } } }
   ```

6. **Verify the full path, in order:**

   ```bash
   command -v sindexer            # → ~/.local/bin/sindexer
   sindexer --version
   sindexer collections           # talks to Zilliz: proves MILVUS_URL + token
   sindexer index ~/some/repo     # full build: Jina embeddings land in Zilliz
   sindexer search ~/some/repo "main entry point"   # hybrid hits with scores
   sindexer overview ~/some/repo --human           # structure at a glance
   sindexer-doctor --json         # exit 0 = healthy; ~60s cycle is normal
   ```

**GPU host role:** `deploy/b550/` is the Linux compute-machine variant —
local CUDA Jina embeddings (3090 Ti selection), remote Milvus reached over a
loopback-only SSH tunnel. Start there, not here, when provisioning the
second tier.

## Agent operating contract

- `sindexer search <repo> "<concept>"` is the index-first move; grep only to
  confirm an exact string. After substantial edits to an indexed repo, run
  `sindexer update <repo>` to keep the index warm.
- CLI verbs: `index`, `update`, `search`, `overview`, `status`, `clear`,
  `collections`, `stats`, `drop`, `usage` (`sindexer --help` is canonical).
- Output is compact JSON on stdout, logs on stderr; usage mistakes exit 2,
  runtime failures exit 1.
- Every search and index/update appends a best-effort telemetry event to
  `~/.context/usage/sindexer.jsonl`; `sindexer usage --human` reports
  estimated token savings.
- Collection identity: absolute path string, symlinks not resolved — keep
  CLI and MCP callers passing the same path form.

## Operating modes (reference)

- **Lexical only (default)** — zero config. BM25 (tantivy) with local store.
- **Semantic + lexical** — `EMBEDDING_URL` set; local vector store handles
  project scale (<50K chunks).
- **Full scale** — `EMBEDDING_URL` + `MILVUS_URL`: Zilliz/Milvus backend.
  The production configuration described above.
- **Dev fallback** — `EMBEDDING_URL` unset and an OpenAI-compatible server
  answers on 127.0.0.1:1234 (LM Studio): used automatically for
  index/update/search unless `SINDEXER_AUTO_EMBEDDING=0`. Never fires when
  `EMBEDDING_URL` is explicitly set.

## Architecture

```
Walker (files) → Splitter (chunks) → Embedder (vectors) → Vector Store
                                          │
                                     Lexical (BM25)
                                          │
                                     Hybrid Fusion (RRF)
```

When embeddings are disabled, the pipeline stops after splitting and only
populates the lexical index.

## Components

**Walker** (`src/walker/mod.rs`) — Parallel file discovery using the
`ignore` crate with native .gitignore support; `walk_builder` is the shared
ignore-semantics base (hidden entries skipped, repo/global/exclude
gitignores, `.contextignore` overlays). Filters by extension and
extensionless filenames (Dockerfile, Makefile, etc.) via
`config::SUPPORTED_EXTENSIONS` (60+ types).

**Overview** (`src/overview.rs`) — Token-bounded repo structure at a
glance: dir skeleton with per-dir file counts and dominant extensions from
a live walk over all file types; bytes/4 budget prunes by depth, then by
subtree size.

**Splitter** (`src/splitter/`) — Tree-sitter AST parsing for semantic code
chunking. Extracts functions, classes, structs, traits, impl blocks per
language. Falls back to markdown heading or line-based splitting for
unsupported languages.

Supported AST languages: Python, JavaScript, TypeScript, TSX, Rust, Go,
Java, C++, C, Ruby, PHP, Swift, Scala, C#

**Embedder** (`src/embedding/mod.rs`) — `Embedder` enum: `Http` for
OpenAI-compatible APIs, or `Disabled` for lexical-only. Auto-detected from
`EMBEDDING_URL` / `OPENAI_BASE_URL`. Batches 32 texts per request.

**Vector Store** (`src/vectordb/`) — `VectorStore` enum: `Local`
(brute-force cosine, JSON disk persistence) or `Milvus`
(`src/vectordb/client.rs`, Milvus/Zilliz v2 REST API — see
`docs/milvus-api.md`). Auto-detected from `MILVUS_URL` / `MILVUS_ADDRESS`.

**Lexical Search** (`src/lexical/mod.rs`) — Tantivy-based BM25 index.

**Hybrid Fusion** (`src/mcp/hybrid.rs`) — Reciprocal Rank Fusion (RRF);
works when either source is empty.

**Incremental Indexing** (`src/mcp/manifest.rs`) — SHA-256 file-hash
manifest at `.sindexer/index-manifest.json`. `update_index` refreshes
changed files only and refuses full-rebuild fallback; `force: true` on
`index_codebase` is the deliberate full rebuild.

## Key Files

- `src/main.rs` — entry point: MCP stdio server (no args) / CLI dispatch
- `src/cli.rs` — CLI verbs (thin parsers over the MCP-tool cores)
- `src/overview.rs` — repo-structure overview core
- `src/types.rs` — CodeChunk, EmbeddingVector, IndexStatus
- `src/config.rs` — walker/splitter configuration
- `src/mcp/state.rs` — shared async state with Embedder/VectorStore enums
- `src/mcp/indexer.rs` — indexing pipeline (lexical-only or full)
- `src/mcp/hybrid.rs` — hybrid search fusion
- `src/mcp/manifest.rs` — incremental reindexing manifest
- `src/mcp/tools.rs` — MCP tool definitions and JSON schemas
- `src/usage.rs` — usage telemetry (JSONL log + `usage` verb)
- `src/vectordb/local.rs`, `src/vectordb/client.rs` — vector stores
- `deploy/local/` — PATH wrapper, doctor, watchdog LaunchAgent
- `deploy/b550/` — Linux GPU-host deployment kit

## MCP Tools

index_codebase, update_index, search_code, get_indexing_status,
clear_index, list_collections, collection_stats, drop_collection.

AlphaHENG invokes the Rust Index CLI binary directly and supplies its
runtime environment from the control machine.

## Tests

```bash
cargo test              # all tests
cargo test walker       # file discovery
cargo test splitter     # AST parsing
cargo test overview     # structure overview
cargo test lexical      # BM25 search
```

Known flake: `mcp::tools::tests::test_index_codebase_updates_shared_status`
(mock embedding server under full-suite parallel load) can fail
intermittently; it passes in isolation. Re-run before investigating.

## Dependencies

- **Core:** rmcp 1.5.0, tokio, rayon, ignore
- **Parsing:** tree-sitter + language grammars
- **HTTP:** reqwest (rustls-tls)
- **Lexical:** tantivy

## Code Quality

**Less code is better.** This codebase should be minimal and focused:

- Prefer single-file modules over directory structures
- Delete unused code rather than commenting it out
- No placeholder/stub implementations — working code only
- Every line must serve a purpose
- If something can be done in 10 lines instead of 100, do it in 10
- Avoid abstractions until needed 3+ times
- No backwards compatibility shims — just change the code

**The goal is a fast, minimal indexing tool — not a framework.**
