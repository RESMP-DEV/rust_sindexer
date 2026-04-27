use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Instant;

use anyhow::{bail, Context, Result};
use serde::{Deserialize, Serialize};
use tokio::task;
use tokio::time::{sleep, Duration};
use tracing::{debug, error, info, instrument, warn};

use crate::config::Config;
use crate::embedding::{Embedder, EmbeddingClient, EmbeddingConfig, RateLimiter};
use crate::lexical::LexicalIndex;
use crate::mcp::hybrid::{fuse_hybrid_hits, HybridFusionOptions, HybridHit};
use crate::mcp::indexer::IndexerState;
use crate::mcp::state::{ContextState, SharedState};
use crate::splitter::{CodeSplitter, Config as SplitterConfig};
use crate::types::{IndexState, IndexStatus};
use crate::vectordb::{collection_name_from_path, LocalStore, MilvusClient, VectorStore};
use crate::walker::CodeWalker;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IndexResult {
    pub files_indexed: usize,
    pub chunks_created: usize,
    pub lexical_only: bool,
    pub duration_ms: u64,
    pub warnings: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchHit {
    pub file_path: PathBuf,
    pub relative_path: String,
    pub content: String,
    pub start_line: u32,
    pub end_line: u32,
    pub language: String,
    pub score: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CollectionInfo {
    pub name: String,
    pub row_count: u64,
}

pub struct Sindexer {
    state: SharedState,
}

impl Sindexer {
    pub fn new(config: Config) -> Self {
        let embedder_mode = if std::env::var("EMBEDDING_URL").map(|v| !v.is_empty()).unwrap_or(false) {
            "http"
        } else {
            "disabled"
        };
        let vector_mode = if std::env::var("MILVUS_URL").map(|v| !v.is_empty()).unwrap_or(false) {
            "milvus"
        } else {
            "local"
        };
        info!(
            embedder = embedder_mode,
            vector_store = vector_mode,
            concurrency = config.concurrency,
            chunk_size = config.chunk_size,
            embedding_dimension = config.embedding_dimension,
            "Sindexer initialized"
        );
        Self {
            state: Arc::new(ContextState::new(config)),
        }
    }

    pub fn from_env() -> Self {
        Self::new(Config::from_env())
    }

    pub fn with_components(config: Config, embedder: Embedder, vector_store: VectorStore) -> Self {
        info!(
            concurrency = config.concurrency,
            chunk_size = config.chunk_size,
            "Sindexer initialized with explicit components"
        );
        Self {
            state: Arc::new(ContextState::with_components(config, embedder, vector_store)),
        }
    }

    #[cfg(feature = "mcp")]
    pub(crate) fn shared_state(&self) -> &SharedState {
        &self.state
    }

    #[instrument(skip(self), fields(path = %path.display()))]
    pub async fn index(&self, path: &Path, force: bool) -> Result<IndexResult> {
        validate_directory(path)?;

        if self.state.is_indexing(path) && !force {
            warn!(path = %path.display(), "Index request rejected: already indexing");
            bail!(
                "Indexing is already running for {}. Use force=true to rebuild.",
                path.display()
            );
        }

        info!(force, "Starting index operation");
        let start = Instant::now();

        let indexer_state = create_indexer_state(&self.state, path);
        self.state.set_status(
            path.to_path_buf(),
            IndexStatus {
                status: IndexState::Indexing,
                ..Default::default()
            },
        );

        let state_for_mirror = self.state.clone();
        let is_clone = indexer_state.clone();
        let path_clone = path.to_path_buf();
        let status_mirror = tokio::spawn(async move {
            loop {
                let status = is_clone.get_status().await;
                let done = !matches!(status.status, IndexState::Indexing);
                state_for_mirror.set_status(path_clone.clone(), status);
                if done {
                    break;
                }
                sleep(Duration::from_millis(250)).await;
            }
        });

        let result = agent::index_codebase(&indexer_state, path, force).await;
        let _ = status_mirror.await;

        match &result {
            Ok(r) => info!(
                files = r.files_processed,
                chunks = r.chunks_created,
                embeddings = r.embeddings_generated,
                vectors = r.vectors_inserted,
                lexical_only = r.lexical_only,
                duration_ms = r.duration_ms,
                warnings = r.warnings.len(),
                "Index operation completed"
            ),
            Err(e) => error!(error = %e, elapsed_ms = start.elapsed().as_millis() as u64, "Index operation failed"),
        }

        let result = result.context("indexing failed")?;

        Ok(IndexResult {
            files_indexed: result.files_processed,
            chunks_created: result.chunks_created,
            lexical_only: result.lexical_only,
            duration_ms: result.duration_ms,
            warnings: result.warnings,
        })
    }

    #[instrument(skip(self, query), fields(path = %path.display(), query_len = query.len()))]
    pub async fn search(
        &self,
        path: &Path,
        query: &str,
        limit: usize,
        extensions: &[String],
    ) -> Result<Vec<SearchHit>> {
        validate_directory(path)?;
        if limit == 0 {
            bail!("limit must be greater than 0");
        }

        let start = Instant::now();
        debug!(limit, extensions = ?extensions, "Search request");

        let collection = collection_name_from_path(path);

        let semantic_start = Instant::now();
        let vector_hits = if self.state.embedder.is_enabled() {
            let hits = self.state
                .search(&collection, query, limit)
                .await?;
            debug!(
                count = hits.len(),
                elapsed_ms = semantic_start.elapsed().as_millis() as u64,
                "Semantic search completed"
            );
            hits.into_iter()
                .map(|r| HybridHit {
                    chunk: r.chunk,
                    score: r.score,
                })
                .collect()
        } else {
            debug!("Semantic search skipped (embeddings disabled)");
            Vec::new()
        };

        let lexical_path = path.to_path_buf();
        let lexical_query = query.to_string();
        let lexical_start = Instant::now();
        let lexical_hits = task::spawn_blocking(move || -> Result<Vec<HybridHit>> {
            if !LexicalIndex::exists(&lexical_path)? {
                debug!("No lexical index found at {}", lexical_path.display());
                return Ok(Vec::new());
            }
            let idx = LexicalIndex::open(&lexical_path)?;
            let mut hits = idx.search(&lexical_query, limit)?;
            for hit in &mut hits {
                if hit.chunk.file_path.as_os_str().is_empty() {
                    hit.chunk.file_path = lexical_path.join(&hit.chunk.relative_path);
                }
            }
            Ok(hits)
        })
        .await
        .context("lexical search task panicked")?
        .context("lexical search failed")?;

        debug!(
            lexical_hits = lexical_hits.len(),
            lexical_ms = lexical_start.elapsed().as_millis() as u64,
            "Lexical search completed"
        );

        let options = HybridFusionOptions {
            limit,
            extension_filter: extensions.to_vec(),
        };
        let fused = fuse_hybrid_hits(query, vector_hits, lexical_hits, &options);

        info!(
            results = fused.len(),
            elapsed_ms = start.elapsed().as_millis() as u64,
            "Search completed"
        );

        Ok(fused
            .into_iter()
            .map(|hit| SearchHit {
                file_path: hit.chunk.file_path,
                relative_path: hit.chunk.relative_path,
                content: hit.chunk.content,
                start_line: hit.chunk.start_line,
                end_line: hit.chunk.end_line,
                language: hit.chunk.language,
                score: hit.score,
            })
            .collect())
    }

    pub fn status(&self, path: &Path) -> IndexStatus {
        let status = self.state.get_status(path);
        debug!(path = %path.display(), status = ?status.status, "Status query");
        status
    }

    #[instrument(skip(self), fields(path = %path.display()))]
    pub async fn clear(&self, path: &Path) -> Result<()> {
        info!("Clearing index");
        let start = Instant::now();

        let collection_name = collection_name_from_path(path);
        let had_vector = self
            .state
            .vector_store
            .has_collection(&collection_name)
            .await?;
        if had_vector {
            self.state
                .vector_store
                .drop_collection(&collection_name)
                .await?;
            debug!(collection = collection_name, "Dropped vector collection");
        }
        self.state
            .set_status(path.to_path_buf(), IndexStatus::default());
        let _ = self.state.manifest_store.clear_status(path);
        self.state.indexing_status.remove(&path.to_path_buf());

        let lexical_path = path.to_path_buf();
        task::spawn_blocking(move || -> Result<()> {
            let idx = LexicalIndex::create(&lexical_path)?;
            idx.clear()?;
            Ok(())
        })
        .await
        .context("lexical clear task panicked")?
        .context("failed to clear lexical index")?;

        info!(
            had_vector_collection = had_vector,
            elapsed_ms = start.elapsed().as_millis() as u64,
            "Index cleared"
        );
        Ok(())
    }

    #[instrument(skip(self))]
    pub async fn list_collections(&self) -> Result<Vec<CollectionInfo>> {
        let names = self.state.vector_store.list_collections().await?;
        debug!(count = names.len(), "Listed collections");
        let mut out = Vec::with_capacity(names.len());
        for name in &names {
            let row_count = self
                .state
                .vector_store
                .collection_stats(name)
                .await
                .map(|s| s.row_count)
                .unwrap_or(0);
            out.push(CollectionInfo {
                name: name.clone(),
                row_count,
            });
        }
        Ok(out)
    }

    pub async fn collection_stats(&self, name: &str) -> Result<u64> {
        let stats = self.state.vector_store.collection_stats(name).await?;
        debug!(collection = name, row_count = stats.row_count, "Collection stats");
        Ok(stats.row_count)
    }

    #[instrument(skip(self))]
    pub async fn drop_collection(&self, name: &str) -> Result<bool> {
        if !self.state.vector_store.has_collection(name).await? {
            debug!(collection = name, "Drop requested but collection does not exist");
            return Ok(false);
        }
        self.state.vector_store.drop_collection(name).await?;
        info!(collection = name, "Collection dropped");
        Ok(true)
    }
}

fn validate_directory(path: &Path) -> Result<()> {
    if !path.is_absolute() {
        bail!("path must be absolute: {}", path.display());
    }
    if !path.exists() {
        bail!("path does not exist: {}", path.display());
    }
    if !path.is_dir() {
        bail!("path is not a directory: {}", path.display());
    }
    Ok(())
}

fn create_indexer_state(state: &SharedState, root_path: &Path) -> Arc<IndexerState> {
    let config = &state.config;
    let splitter = CodeSplitter::new(SplitterConfig {
        root_path: root_path.to_path_buf(),
        max_chunk_bytes: config.chunk_size,
        overlap_lines: config.chunk_overlap / 80,
        ..SplitterConfig::default()
    });

    let embedder = if state.embedder.is_enabled() {
        let rate_limiter = RateLimiter::new(config.embedding_rpm, config.embedding_tpm);
        Embedder::Http(EmbeddingClient::with_rate_limiter(
            EmbeddingConfig::from_config(config),
            rate_limiter,
        ))
    } else {
        Embedder::Disabled
    };

    let vector_store = if matches!(state.vector_store, VectorStore::Milvus(_)) {
        VectorStore::Milvus(MilvusClient::new(
            &config.milvus_url,
            config.milvus_token.clone(),
        ))
    } else {
        VectorStore::Local(LocalStore::new())
    };

    Arc::new(IndexerState::with_concurrency(
        CodeWalker::new(),
        splitter,
        embedder,
        vector_store,
        config.embedding_dimension,
        config.concurrency,
    ))
}

use crate::mcp::indexer as agent;
