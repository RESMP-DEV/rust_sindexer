use anyhow::{Context, Result};
use sha2::{Digest, Sha256};
use std::fs;
use std::path::{Path, PathBuf};
use std::time::Instant;
use tantivy::collector::TopDocs;
use tantivy::query::QueryParser;
use tantivy::schema::{
    Field, IndexRecordOption, Schema, TantivyDocument, TextFieldIndexing, TextOptions, Value, FAST,
    INDEXED, STORED, STRING,
};
use tantivy::{Index, IndexReader, IndexWriter, Term};
use tracing::debug;

use crate::mcp::HybridHit;
use crate::types::CodeChunk;

pub struct LexicalIndex {
    index: Index,
    reader: IndexReader,
}

impl LexicalIndex {
    pub fn create(path: &Path) -> Result<Self> {
        debug!(path = %path.display(), "creating lexical index");
        let index_path = cache_dir_for(path)?;
        fs::create_dir_all(&index_path).with_context(|| {
            format!(
                "failed to create lexical index directory at {}",
                index_path.display()
            )
        })?;

        let index = if metadata_exists(&index_path) {
            Index::open_in_dir(&index_path).context("failed to open existing lexical index")?
        } else {
            Index::create_in_dir(&index_path, schema()).context("failed to create lexical index")?
        };
        let reader = index
            .reader()
            .context("failed to create lexical index reader")?;

        Ok(Self { index, reader })
    }

    pub fn open(path: &Path) -> Result<Self> {
        let index_path = cache_dir_for(path)?;
        if !metadata_exists(&index_path) {
            anyhow::bail!(
                "failed to open lexical index at {}: metadata missing",
                index_path.display()
            );
        }
        let index = Index::open_in_dir(&index_path)
            .with_context(|| format!("failed to open lexical index at {}", index_path.display()))?;
        let reader = index
            .reader()
            .context("failed to create lexical index reader")?;

        Ok(Self { index, reader })
    }

    pub fn exists(path: &Path) -> Result<bool> {
        Ok(metadata_exists(&cache_dir_for(path)?))
    }

    pub fn index(&self) -> &Index {
        &self.index
    }

    pub fn reader(&self) -> &IndexReader {
        &self.reader
    }

    pub fn writer(&self, heap_size_bytes: usize) -> Result<IndexWriter> {
        self.index
            .writer::<TantivyDocument>(heap_size_bytes)
            .context("failed to create lexical index writer")
    }

    pub fn insert_chunks(&self, chunks: &[CodeChunk]) -> Result<()> {
        if chunks.is_empty() {
            return Ok(());
        }
        debug!(
            chunk_count = chunks.len(),
            "inserting chunks into lexical index"
        );

        let start = Instant::now();
        tracing::debug!(chunks = chunks.len(), "Inserting chunks into lexical index");

        let fields = LexicalFields::new(self.index.schema())?;
        let mut writer = self
            .index
            .writer::<TantivyDocument>(50_000_000)
            .context("failed to create lexical index writer")?;

        for chunk in chunks {
            let mut doc = TantivyDocument::default();
            doc.add_text(fields.id, &chunk.id);
            doc.add_text(fields.content, &chunk.content);
            doc.add_text(fields.relative_path, &chunk.relative_path);
            doc.add_u64(fields.start_line, u64::from(chunk.start_line));
            doc.add_u64(fields.end_line, u64::from(chunk.end_line));
            doc.add_text(fields.language, &chunk.language);
            writer.add_document(doc)?;
        }

        writer.commit().context("failed to commit lexical index")?;
        self.reader
            .reload()
            .context("failed to reload lexical index reader")?;

        tracing::info!(
            chunks = chunks.len(),
            elapsed_ms = start.elapsed().as_millis() as u64,
            "Lexical index insert completed"
        );
        Ok(())
    }

    pub fn delete_by_paths(&self, relative_paths: &[String]) -> Result<()> {
        if relative_paths.is_empty() {
            return Ok(());
        }
        debug!(
            path_count = relative_paths.len(),
            "deleting paths from lexical index"
        );

        tracing::debug!(
            paths = relative_paths.len(),
            "Deleting paths from lexical index"
        );
        let fields = LexicalFields::new(self.index.schema())?;
        let mut writer = self
            .index
            .writer::<TantivyDocument>(50_000_000)
            .context("failed to create lexical index writer")?;

        for relative_path in relative_paths {
            writer.delete_term(Term::from_field_text(fields.relative_path, relative_path));
        }

        writer.commit().context("failed to commit lexical index")?;
        self.reader
            .reload()
            .context("failed to reload lexical index reader")?;
        Ok(())
    }

    /// Performs a lexical search over the index and returns the top matching hits.
    ///
    /// The query is parsed and executed against the indexed `content` field; results are
    /// ordered by score and limited by `limit`. If `limit` is zero, an empty vector is returned.
    ///
    /// # Examples
    ///
    /// ```
    /// // Assume `index` is a ready `LexicalIndex`.
    /// let hits = index.search("target_keyword", 10).unwrap();
    /// // At most `limit` results are returned and each hit has a `chunk` and a `score`.
    /// assert!(hits.len() <= 10);
    /// ```
    pub fn search(&self, query: &str, limit: usize) -> Result<Vec<HybridHit>> {
        debug!(query_len = query.len(), limit, "lexical search");
        if limit == 0 {
            return Ok(Vec::new());
        }

        let start = Instant::now();
        tracing::debug!(query_len = query.len(), limit, "Lexical search starting");

        let fields = LexicalFields::new(self.index.schema())?;
        let parsed_query = QueryParser::for_index(&self.index, vec![fields.content])
            .parse_query(query)
            .context("failed to parse lexical search query")?;
        let searcher = self.reader.searcher();
        let collector = TopDocs::with_limit(limit).order_by_score();
        let top_docs = searcher
            .search(&parsed_query, &collector)
            .context("failed to search lexical index")?;

        tracing::debug!(
            raw_hits = top_docs.len(),
            elapsed_us = start.elapsed().as_micros() as u64,
            "Lexical search query executed"
        );

        top_docs
            .into_iter()
            .map(|(score, address)| {
                let doc = searcher
                    .doc::<TantivyDocument>(address)
                    .context("failed to load lexical search result document")?;

                Ok(HybridHit {
                    chunk: CodeChunk {
                        id: string_value(&doc, fields.id).unwrap_or_default(),
                        content: string_value(&doc, fields.content).unwrap_or_default(),
                        file_path: PathBuf::new(),
                        relative_path: string_value(&doc, fields.relative_path).unwrap_or_default(),
                        start_line: u64_value(&doc, fields.start_line).unwrap_or_default() as u32,
                        end_line: u64_value(&doc, fields.end_line).unwrap_or_default() as u32,
                        language: string_value(&doc, fields.language).unwrap_or_default(),
                    },
                    score,
                })
            })
            .collect()
    }

    pub fn clear(&self) -> Result<()> {
        tracing::info!("Clearing lexical index");
        let mut writer = self
            .index
            .writer::<TantivyDocument>(50_000_000)
            .context("failed to create lexical index writer")?;
        writer
            .delete_all_documents()
            .context("failed to delete all lexical index documents")?;
        writer.commit().context("failed to commit lexical index")?;
        self.reader
            .reload()
            .context("failed to reload lexical index reader")?;
        Ok(())
    }
}

struct LexicalFields {
    id: Field,
    content: Field,
    relative_path: Field,
    start_line: Field,
    end_line: Field,
    language: Field,
}

impl LexicalFields {
    fn new(schema: Schema) -> Result<Self> {
        Ok(Self {
            id: schema
                .get_field("id")
                .context("missing id field in lexical index schema")?,
            content: schema
                .get_field("content")
                .context("missing content field in lexical index schema")?,
            relative_path: schema
                .get_field("relative_path")
                .context("missing relative_path field in lexical index schema")?,
            start_line: schema
                .get_field("start_line")
                .context("missing start_line field in lexical index schema")?,
            end_line: schema
                .get_field("end_line")
                .context("missing end_line field in lexical index schema")?,
            language: schema
                .get_field("language")
                .context("missing language field in lexical index schema")?,
        })
    }
}

/// Retrieves the first stored string value for a field from a Tantivy document.
///
/// If the field has a stored string value, returns that value as a `String`; otherwise returns `None`.
///
/// # Examples
///
/// ```
/// use tantivy::schema::{Schema, SchemaBuilder, TextOptions, STORED, TEXT};
/// use tantivy::doc;
/// use tantivy::schema::Field;
///
/// // Build a minimal schema with a stored text field.
/// let mut schema_builder = SchemaBuilder::new();
/// let text_opts = TextOptions::default().set_stored();
/// let field: Field = schema_builder.add_text_field("content", text_opts);
/// let schema: Schema = schema_builder.build();
///
/// // Create a document with the stored field.
/// let document = doc!(field => "hello world");
///
/// // Call the helper (assumes `string_value` is in scope).
/// let value = crate::lexical::string_value(&document, field);
/// assert_eq!(value.as_deref(), Some("hello world"));
/// ```
fn string_value(doc: &TantivyDocument, field: Field) -> Option<String> {
    doc.get_first(field)
        .and_then(|value| value.as_str().map(ToString::to_string))
}

/// Extracts the first stored unsigned integer value for a field from a Tantivy document.
///
/// # Examples
///
/// ```
/// use tantivy::schema::{SchemaBuilder, STORED, FAST};
/// use tantivy::doc;
///
/// // Build a schema with a stored u64 field
/// let mut schema_builder = SchemaBuilder::default();
/// let field = schema_builder.add_u64_field("num", STORED | FAST);
/// let _schema = schema_builder.build();
///
/// // Create a document containing the stored u64 value
/// let document = doc!(field => 42u64);
///
/// // Read the value back
/// let v = crate::u64_value(&document, field);
/// assert_eq!(v, Some(42u64));
/// ```
fn u64_value(doc: &TantivyDocument, field: Field) -> Option<u64> {
    doc.get_first(field).and_then(|value| value.as_u64())
}

/// Constructs the Tantivy schema used by the lexical index.
///
/// Fields:
/// - `id`: stored string identifier.
/// - `content`: stored full-text field indexed with positions and term frequencies using the `"default"` tokenizer.
/// - `relative_path`: stored string containing the file path relative to the project root.
/// - `start_line`, `end_line`: stored, indexed, and fast `u64` fields for chunk line ranges.
/// - `language`: stored string indicating the chunk's programming language.
///
/// # Examples
///
/// ```
/// let s = schema();
/// assert!(s.get_field("id").is_some());
/// assert!(s.get_field("content").is_some());
/// assert!(s.get_field("relative_path").is_some());
/// assert!(s.get_field("start_line").is_some());
/// assert!(s.get_field("end_line").is_some());
/// assert!(s.get_field("language").is_some());
/// ```
fn schema() -> Schema {
    let mut builder = Schema::builder();
    builder.add_text_field("id", STRING | STORED);
    builder.add_text_field("content", content_field_options());
    builder.add_text_field("relative_path", STRING | STORED);
    builder.add_u64_field("start_line", STORED | INDEXED | FAST);
    builder.add_u64_field("end_line", STORED | INDEXED | FAST);
    builder.add_text_field("language", STRING | STORED);
    builder.build()
}

fn content_field_options() -> TextOptions {
    let indexing = TextFieldIndexing::default()
        .set_tokenizer("default")
        .set_index_option(IndexRecordOption::WithFreqsAndPositions);

    TextOptions::default()
        .set_indexing_options(indexing)
        .set_stored()
}

fn cache_dir_for(path: &Path) -> Result<PathBuf> {
    let mut hasher = Sha256::new();
    hasher.update(path.to_string_lossy().as_bytes());
    let digest = hex::encode(hasher.finalize());

    Ok(home_cache_dir()?
        .join("sindexer")
        .join("lexical-indexes")
        .join(digest))
}

fn home_cache_dir() -> Result<PathBuf> {
    if let Some(cache_home) = std::env::var_os("XDG_CACHE_HOME") {
        return Ok(PathBuf::from(cache_home));
    }

    let home = std::env::var_os("HOME").context("HOME is not set")?;
    Ok(PathBuf::from(home).join(".cache"))
}

fn metadata_exists(index_path: &Path) -> bool {
    index_path.join("meta.json").exists()
}

#[cfg(test)]
pub(crate) mod test_support {
    use once_cell::sync::Lazy;
    use std::path::Path;
    use tokio::sync::{Mutex, MutexGuard};

    static CACHE_ENV_LOCK: Lazy<Mutex<()>> = Lazy::new(|| Mutex::new(()));
    pub(crate) type CacheEnvGuard = MutexGuard<'static, ()>;

    pub(crate) fn set_test_cache_dir(path: &Path) -> CacheEnvGuard {
        let guard = CACHE_ENV_LOCK.blocking_lock();
        std::env::set_var("XDG_CACHE_HOME", path);
        guard
    }

    pub(crate) async fn set_test_cache_dir_async(path: &Path) -> CacheEnvGuard {
        let guard = CACHE_ENV_LOCK.lock().await;
        std::env::set_var("XDG_CACHE_HOME", path);
        guard
    }
}

#[cfg(test)]
mod tests {
    use super::LexicalIndex;
    use crate::lexical::test_support::{set_test_cache_dir, CacheEnvGuard};
    use crate::types::CodeChunk;
    use std::path::PathBuf;
    use tempfile::TempDir;

    fn code_chunk(id: &str, content: &str, relative_path: &str) -> CodeChunk {
        CodeChunk {
            id: id.to_string(),
            content: content.to_string(),
            file_path: PathBuf::from(format!("/repo/{relative_path}")),
            relative_path: relative_path.to_string(),
            start_line: 1,
            end_line: 1,
            language: "rust".to_string(),
        }
    }

    fn test_index() -> (CacheEnvGuard, TempDir, TempDir, LexicalIndex) {
        let cache_dir = TempDir::new().unwrap();
        let cache_lock = set_test_cache_dir(cache_dir.path());

        let repo_dir = TempDir::new().unwrap();
        let index = LexicalIndex::create(repo_dir.path()).unwrap();
        (cache_lock, cache_dir, repo_dir, index)
    }

    #[test]
    fn test_insert_and_search() {
        let (_home_lock, _home_dir, _repo_dir, index) = test_index();
        let chunks = vec![
            code_chunk("chunk-1", "fn alpha() {}", "src/alpha.rs"),
            code_chunk(
                "chunk-2",
                "fn target_keyword() -> bool { true }",
                "src/target.rs",
            ),
            code_chunk("chunk-3", "fn omega() {}", "src/omega.rs"),
        ];

        index.insert_chunks(&chunks).unwrap();

        let results = index.search("target_keyword", 10).unwrap();

        assert!(!results.is_empty());
        assert_eq!(results[0].chunk.id, "chunk-2");
        assert!(results[0].score > 0.0);
    }

    #[test]
    fn test_search_empty_index() {
        let (_home_lock, _home_dir, _repo_dir, index) = test_index();

        let results = index.search("missing_keyword", 10).unwrap();

        assert!(results.is_empty());
    }

    #[test]
    fn test_bm25_ranks_exact_match_higher() {
        let (_home_lock, _home_dir, _repo_dir, index) = test_index();
        let chunks = vec![
            code_chunk(
                "exact",
                "fn calculate_score(value: i32) -> i32 { value + 1 }",
                "src/exact.rs",
            ),
            code_chunk(
                "partial",
                "the score is computed from several inputs",
                "src/partial.rs",
            ),
        ];

        index.insert_chunks(&chunks).unwrap();

        let results = index.search("calculate_score", 10).unwrap();

        assert!(!results.is_empty());
        assert_eq!(results[0].chunk.id, "exact");
        assert!(results[0].score > 0.0);
    }

    #[test]
    fn test_delete_by_paths() {
        let (_home_lock, _home_dir, _repo_dir, index) = test_index();
        let chunks = vec![
            code_chunk("chunk-a", "alpha unique token", "src/a.rs"),
            code_chunk("chunk-b", "beta unique token", "src/b.rs"),
            code_chunk("chunk-c", "gamma unique token", "src/c.rs"),
        ];

        index.insert_chunks(&chunks).unwrap();
        index.delete_by_paths(&[String::from("src/b.rs")]).unwrap();

        assert!(index.search("beta", 10).unwrap().is_empty());

        let alpha_hits = index.search("alpha", 10).unwrap();
        assert_eq!(alpha_hits.len(), 1);
        assert_eq!(alpha_hits[0].chunk.relative_path, "src/a.rs");

        let gamma_hits = index.search("gamma", 10).unwrap();
        assert_eq!(gamma_hits.len(), 1);
        assert_eq!(gamma_hits[0].chunk.relative_path, "src/c.rs");
    }

    #[test]
    fn test_clear() {
        let (_home_lock, _home_dir, _repo_dir, index) = test_index();
        let chunks = vec![
            code_chunk("chunk-a", "alpha unique token", "src/a.rs"),
            code_chunk("chunk-b", "beta unique token", "src/b.rs"),
        ];

        index.insert_chunks(&chunks).unwrap();
        index.clear().unwrap();

        assert!(index.search("alpha", 10).unwrap().is_empty());
        assert!(index.search("beta", 10).unwrap().is_empty());
    }
}
