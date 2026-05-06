# Changelog

All notable changes to sindexer will be documented in this file.

## [Unreleased]

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
