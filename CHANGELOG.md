# Changelog

All notable changes to `rust_sindexer` will be documented in this file.

## [Unreleased]

### Added
- Added `SINDEXER_COLLECTION_IDENTITY` so multiple hosts with different checkout
  paths can intentionally share the same Milvus collection for one codebase.
- `get_indexing_status` now reconciles idle or completed shared-collection
  status from live vector-store row counts when a host did not perform the
  indexing run, without masking failed runs as completed.
- Hardened status reconciliation so delayed Milvus stats cannot lower completed
  counters, while later higher live row counts raise all row-backed totals
  together.

### Fixed
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
