# Changelog

All notable changes to `rust_sindexer` will be documented in this file.

## [Unreleased]

### Changed

- Batch vector writes use the Milvus `upsert` endpoint (parity with
  rust-indexer): retried batches replace rows for existing primary keys
  instead of duplicating them. The response parser accepts both
  `upsertCount` and `insertCount`.

### Fixed

- `update_index` now self-heals a missing or incompatible per-codebase manifest
  or vector collection by rebuilding only that scoped index. Unrelated
  collections are never used as a fallback or removed.

### Documentation

- Restructured the README for first-time users: added a plain-language
  project description, feature highlights, prebuilt-binary installation
  instructions (GitHub Releases), a prominent Quickstart, an MCP tool
  reference table with parameters, search examples, a "Building from source
  & development" section with project layout, and a Troubleshooting & FAQ
  section covering the most common failure modes.
- Added an `examples/` directory: ready-to-edit `mcp-servers.json` configs
  (zero-config and semantic variants) and `demo.sh`, a self-contained
  end-to-end smoke test that drives the server over raw stdio JSON-RPC
  without an MCP client.
- Rewrote the README for accuracy: documented the zero-config lexical-only
  default mode and local vector store, corrected the binary name to `sindexer`
  in MCP config examples, replaced the nonexistent `LOG_LEVEL` variable with
  `RUST_LOG`, listed all eight MCP tools (including `list_collections`,
  `collection_stats`, and `drop_collection`), documented previously missing
  environment variables (`CHUNK_SIZE`, `CHUNK_OVERLAP`, `BATCH_SIZE`,
  `INDEXING_CONCURRENCY`, `EMBEDDING_RPM`, `EMBEDDING_TPM`,
  `FOLLOW_SYMLINKS`, `SINDEXER_COLLECTION_IDENTITY`), fixed MCP config file
  locations, and added an on-disk layout section. `PARALLELISM` is parsed but
  not yet wired to the walker or rayon, so it stays undocumented.

### Added

- Added separate query and passage prefix settings for task-aware embedding
  models, including calibrated Jina code embeddings.
- The binary now accepts `-h`/`--help` and `-V`/`--version`, printing usage
  (with MCP registration examples) and the crate version respectively.
  Unknown arguments are still ignored, so existing MCP client launches are
  unaffected.
- Added the `update_index` MCP tool for incremental-only refreshes of an
  existing compatible codebase index. It touches only changed/deleted files and
  fails instead of falling back to a full rebuild when the manifest or backing
  collection is missing or incompatible.
- Added `SINDEXER_COLLECTION_IDENTITY` so multiple hosts with different checkout
  paths can intentionally share the same Milvus collection for one codebase.
- `get_indexing_status` now reconciles idle or completed shared-collection
  status from live vector-store row counts when a host did not perform the
  indexing run, without masking failed runs as completed.
- Hardened status reconciliation so delayed Milvus stats cannot lower completed
  counters, while later higher live row counts raise all row-backed totals
  together.

### Fixed

- Updated vulnerable transitive Rust dependencies and aligned the
  `cargo-deny` advisory policy with the current configuration schema.
- Include the passage-prefix fingerprint in manifest compatibility so prefix
  changes require a full rebuild instead of mixing incompatible vectors.
- Acquire RPM and TPM capacity atomically so waiting on one bucket cannot
  consume tokens from the other; avoid prefix allocations when prefixes are
  disabled.
- Updated manifest-test helpers for the Rust 1.97 clippy rules used by CI.
- Resolved new Rust 1.97 clippy lints (collapsible if, redundant
  `.into_iter()`, manual `contains`, and an allowed too-many-arguments on an
  internal helper) so the CI lint gate passes on current stable.
- Deduplicate semantic and lexical hits by canonicalizing the backend-specific
  ID formats that Milvus and Tantivy expose for the same chunk.
- Scoped `SINDEXER_COLLECTION_IDENTITY` with `SINDEXER_COLLECTION_ROOT` and
  namespace separately indexed descendants by relative path, preventing
  AlphaHENG, its nested projects, and unrelated repositories from aliasing and
  overwriting one Milvus collection.
- Reset the exact path-scoped collection during an intentional full rebuild
  instead of relying on asynchronous per-path deletes that can leave stale
  rows or inflated statistics.
- Preserve the accepted Milvus `insertCount` when a fresh or updating Zilliz
  collection is immediately searchable but its stats endpoint reports a
  stale lower or higher row count.
- Keep a completed nonzero local status authoritative instead of inflating it
  from a stale higher shared-collection stats response after a rebuild.
- Preserve that completed nonzero accepted-insert count during a no-op
  `update_index` instead of replacing it with a lagging live stats response.
- Treat an explicitly indexed codebase as the ignore-file boundary so an
  ancestor `.contextignore` can exclude a child from an aggregate index without
  making that child project unindexable on its own.
- Disabled Tantivy's automatic lexical-index merge policy for sindexer-owned
  writers so large incremental refreshes cannot abort the MCP process with a
  merge-worker stack overflow.
- Pinned `tantivy` to `0.26.1` so the lexical-index dependency chain resolves
  to a patched `lru` release instead of the vulnerable `0.12.x` line flagged
  by Dependabot.

- Load legacy manifests without `max_file_size` as the historical 1MB default
  instead of zero, so `update_index` does not reject old compatible manifests
  solely because the manifest schema grew.
- Ignore `.sindexer`, `.rust_sindexer`, `.worktrees`, `agent_workspace`, and
  `logs` during file discovery so refreshes do not index generated metadata,
  local agent sandboxes, or runtime logs.
- Prevent forced AlphaHENG indexing runs from walking common generated or vendor
  directories, following symlinks by default, or processing files above the
  configured `MAX_FILE_SIZE`.
- Split single-line chunks on UTF-8 boundaries instead of sending oversized
  payloads to embedding providers.
- Bisect failed embedding batches and skip only chunks that remain
  unembeddable, preserving the rest of a large index rebuild without turning
  each failed provider batch into a full serial retry.
- Count Zilliz/Milvus inserts using the provider's accepted `insertCount`
  rather than attempted rows, and surface a warning when accepted vector rows
  trail generated embeddings.
- Report persisted non-idle indexing status when the in-memory MCP state is a
  stale idle placeholder, and keep MCP wrapper progress mirrors from
  overwriting completed status with zero-count startup snapshots.
