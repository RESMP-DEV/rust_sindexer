# Changelog

All notable changes to `rust_sindexer` will be documented in this file.

## [Unreleased]

### Added

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
- Avoid dropping a whole shared Milvus collection during full reindex when
  `SINDEXER_COLLECTION_IDENTITY` is active; use targeted relative-path deletes
  before reinserting local files instead.
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
