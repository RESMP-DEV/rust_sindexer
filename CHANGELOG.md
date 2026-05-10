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
