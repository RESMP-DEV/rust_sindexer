use std::sync::Arc;
use std::time::Instant;

use anyhow::{Context, Result};
use parking_lot::Mutex;
use reqwest::Client;
use reqwest::header::{AUTHORIZATION, HeaderMap};
use serde::{Deserialize, Serialize};
use tracing::{debug, info};

use crate::config::Config;
use crate::types::EmbeddingVector;

/// Token-bucket rate limiter for RPM and TPM limits.
struct TokenBucket {
    capacity: f64,
    tokens: f64,
    refill_rate: f64, // tokens per second
    last_refill: Instant,
}

impl TokenBucket {
    fn new(per_minute: f64) -> Self {
        let refill_rate = per_minute / 60.0;
        Self {
            capacity: per_minute,
            tokens: per_minute,
            refill_rate,
            last_refill: Instant::now(),
        }
    }

    fn refill(&mut self) {
        let now = Instant::now();
        let elapsed = now.duration_since(self.last_refill).as_secs_f64();
        self.tokens = (self.tokens + elapsed * self.refill_rate).min(self.capacity);
        self.last_refill = now;
    }

    fn try_acquire(&mut self, cost: f64) -> Option<std::time::Duration> {
        self.refill();
        if self.tokens >= cost {
            self.tokens -= cost;
            None
        } else {
            let deficit = cost - self.tokens;
            Some(std::time::Duration::from_secs_f64(deficit / self.refill_rate))
        }
    }
}

#[derive(Clone)]
pub struct RateLimiter {
    rpm: Arc<Mutex<TokenBucket>>,
    tpm: Arc<Mutex<TokenBucket>>,
}

impl RateLimiter {
    pub fn new(rpm_limit: u32, tpm_limit: u64) -> Self {
        Self {
            rpm: Arc::new(Mutex::new(TokenBucket::new(rpm_limit as f64))),
            tpm: Arc::new(Mutex::new(TokenBucket::new(tpm_limit as f64))),
        }
    }

    pub fn unlimited() -> Self {
        Self::new(100_000, 1_000_000_000)
    }

    pub async fn acquire(&self, estimated_tokens: u64) {
        loop {
            let wait = {
                let rpm_wait = self.rpm.lock().try_acquire(1.0);
                let tpm_wait = self.tpm.lock().try_acquire(estimated_tokens as f64);
                match (rpm_wait, tpm_wait) {
                    (None, None) => break,
                    (Some(a), Some(b)) => a.max(b),
                    (Some(a), None) | (None, Some(a)) => a,
                }
            };
            debug!("Rate limiter: sleeping {}ms", wait.as_millis());
            tokio::time::sleep(wait).await;
        }
    }

    /// Drain buckets on 429/5xx to force all concurrent requests to back off.
    pub fn penalize(&self, retry_after_secs: f64) {
        {
            let mut rpm = self.rpm.lock();
            let drain = retry_after_secs * rpm.refill_rate;
            rpm.tokens = (rpm.tokens - drain).max(0.0);
        }
        {
            let mut tpm = self.tpm.lock();
            let drain = retry_after_secs * tpm.refill_rate;
            tpm.tokens = (tpm.tokens - drain).max(0.0);
        }
    }
}

/// Configuration for the embedding client.
#[derive(Clone, Debug)]
pub struct EmbeddingConfig {
    /// Base URL for the embedding API.
    pub url: String,
    /// Model name to use for embeddings.
    pub model: String,
    /// Maximum number of texts per batch request.
    pub batch_size: usize,
    /// Optional API key for providers that require bearer auth.
    pub api_key: Option<String>,
}

impl Default for EmbeddingConfig {
    fn default() -> Self {
        Self {
            url: "http://localhost:8100/v1/embeddings".to_string(),
            model: "all-minilm".to_string(),
            batch_size: 100,
            api_key: None,
        }
    }
}

impl EmbeddingConfig {
    /// Creates an EmbeddingConfig from the application Config.
    pub fn from_config(config: &Config) -> Self {
        let base_url = config.embedding_url.trim_end_matches('/');
        let url = if base_url.ends_with("/v1") {
            format!("{base_url}/embeddings")
        } else {
            format!("{base_url}/v1/embeddings")
        };

        Self {
            url,
            model: config.embedding_model.clone(),
            batch_size: config.batch_size,
            api_key: config.embedding_api_key.clone(),
        }
    }
}

/// Request payload for the embeddings API.
#[derive(Serialize)]
struct EmbeddingRequest<'a> {
    input: &'a [String],
    model: &'a str,
}

/// Single embedding in the API response.
#[derive(Deserialize)]
struct EmbeddingData {
    embedding: Vec<f32>,
}

/// Response from the embeddings API.
#[derive(Deserialize)]
struct EmbeddingResponse {
    data: Vec<EmbeddingData>,
}

/// Client for generating text embeddings via HTTP API.
pub struct EmbeddingClient {
    client: Client,
    config: EmbeddingConfig,
    rate_limiter: RateLimiter,
}

impl EmbeddingClient {
    /// Creates a new embedding client with the given configuration.
    pub fn new(config: EmbeddingConfig) -> Self {
        Self::with_rate_limiter(config, RateLimiter::unlimited())
    }

    pub fn with_rate_limiter(config: EmbeddingConfig, rate_limiter: RateLimiter) -> Self {
        let mut headers = HeaderMap::new();
        if let Some(ref key) = config.api_key {
            headers.insert(
                AUTHORIZATION,
                format!("Bearer {}", key)
                    .parse()
                    .expect("valid header value"),
            );
        }

        let client = Client::builder()
            .default_headers(headers)
            .pool_max_idle_per_host(64)
            .connect_timeout(std::time::Duration::from_secs(10))
            .timeout(std::time::Duration::from_secs(120))
            .build()
            .expect("failed to build HTTP client");

        Self { client, config, rate_limiter }
    }

    /// Creates a new embedding client with default configuration.
    pub fn with_defaults() -> Self {
        Self::new(EmbeddingConfig::default())
    }

    /// Generates an embedding for a single text.
    pub async fn embed(&self, text: &str) -> Result<EmbeddingVector> {
        let texts = vec![text.to_string()];
        let mut results = self.embed_batch(&texts).await?;

        results
            .pop()
            .context("embedding API returned empty response")
    }

    /// Generates embeddings for multiple texts in batches.
    pub async fn embed_batch(&self, texts: &[String]) -> Result<Vec<EmbeddingVector>> {
        if texts.is_empty() {
            return Ok(Vec::new());
        }

        let start = std::time::Instant::now();
        let total_texts = texts.len();
        let num_batches = (total_texts + self.config.batch_size - 1) / self.config.batch_size;
        debug!(
            texts = total_texts,
            batches = num_batches,
            batch_size = self.config.batch_size,
            model = %self.config.model,
            "Embedding batch starting"
        );

        let mut all_embeddings = Vec::with_capacity(texts.len());

        for (i, chunk) in texts.chunks(self.config.batch_size).enumerate() {
            let chunk_start = std::time::Instant::now();
            let embeddings = self.embed_chunk(chunk).await?;
            debug!(
                batch = i + 1,
                batch_texts = chunk.len(),
                elapsed_ms = chunk_start.elapsed().as_millis() as u64,
                "Embedding batch chunk completed"
            );
            all_embeddings.extend(embeddings);
        }

        info!(
            texts = total_texts,
            embeddings = all_embeddings.len(),
            elapsed_ms = start.elapsed().as_millis() as u64,
            "Embedding batch completed"
        );
        Ok(all_embeddings)
    }

    /// Embeds a single chunk of texts (up to batch_size) with retry + backoff.
    async fn embed_chunk(&self, texts: &[String]) -> Result<Vec<EmbeddingVector>> {
        const MAX_RETRIES: u32 = 5;
        const BYTES_PER_TOKEN: u64 = 4;
        const BASE_BACKOFF_MS: u64 = 1000;

        let estimated_tokens: u64 = texts.iter().map(|t| t.len() as u64 / BYTES_PER_TOKEN).sum();

        let mut last_err = None;
        for attempt in 0..=MAX_RETRIES {
            self.rate_limiter.acquire(estimated_tokens).await;

            let request = EmbeddingRequest {
                input: texts,
                model: &self.config.model,
            };

            let response = match self.client.post(&self.config.url).json(&request).send().await {
                Ok(r) => r,
                Err(e) => {
                    let backoff_secs = (BASE_BACKOFF_MS * 2u64.pow(attempt)) as f64 / 1000.0;
                    self.rate_limiter.penalize(backoff_secs);
                    last_err = Some(format!("request failed: {e}"));
                    tokio::time::sleep(std::time::Duration::from_secs_f64(backoff_secs)).await;
                    continue;
                }
            };

            let status = response.status();
            if status == reqwest::StatusCode::TOO_MANY_REQUESTS
                || status.is_server_error()
            {
                let retry_after = response
                    .headers()
                    .get("retry-after")
                    .and_then(|v| v.to_str().ok())
                    .and_then(|v| v.parse::<f64>().ok());

                let backoff_secs = retry_after
                    .unwrap_or((BASE_BACKOFF_MS * 2u64.pow(attempt)) as f64 / 1000.0);

                self.rate_limiter.penalize(backoff_secs);
                let body = response.text().await.unwrap_or_default();
                tracing::warn!(
                    "Embedding API {status} (attempt {}/{MAX_RETRIES}), backing off {backoff_secs:.1}s: {body}",
                    attempt + 1,
                );
                last_err = Some(format!("embedding API returned {status}: {body}"));
                tokio::time::sleep(std::time::Duration::from_secs_f64(backoff_secs)).await;
                continue;
            }

            if !status.is_success() {
                let body = response.text().await.unwrap_or_default();
                anyhow::bail!("embedding API returned {status}: {body}");
            }

            let response: EmbeddingResponse = response
                .json()
                .await
                .context("failed to parse embedding response")?;

            let embeddings = response
                .data
                .into_iter()
                .map(|d| {
                    let dimension = d.embedding.len();
                    EmbeddingVector {
                        vector: d.embedding,
                        dimension,
                    }
                })
                .collect();

            return Ok(embeddings);
        }

        anyhow::bail!(
            "embedding failed after {} retries: {}",
            MAX_RETRIES,
            last_err.unwrap_or_else(|| "unknown error".to_string())
        )
    }
}

/// Embedding backend selector. When no embedding service is configured,
/// the server operates in lexical-only mode.
pub enum Embedder {
    Http(EmbeddingClient),
    Disabled,
}

impl Embedder {
    pub fn is_enabled(&self) -> bool {
        matches!(self, Self::Http(_))
    }

    pub async fn embed(&self, text: &str) -> Result<EmbeddingVector> {
        match self {
            Self::Http(client) => client.embed(text).await,
            Self::Disabled => anyhow::bail!("Embedding is disabled. Set EMBEDDING_URL to enable semantic search."),
        }
    }

    pub async fn embed_batch(&self, texts: &[String]) -> Result<Vec<EmbeddingVector>> {
        match self {
            Self::Http(client) => client.embed_batch(texts).await,
            Self::Disabled => anyhow::bail!("Embedding is disabled. Set EMBEDDING_URL to enable semantic search."),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = EmbeddingConfig::default();
        assert_eq!(config.url, "http://localhost:8100/v1/embeddings");
        assert_eq!(config.model, "all-minilm");
        assert_eq!(config.batch_size, 100);
        assert_eq!(config.api_key, None);
    }

    #[test]
    fn test_from_config_includes_api_key() {
        let app_config = Config {
            embedding_url: "https://api.openai.com/v1".to_string(),
            embedding_model: "text-embedding-3-small".to_string(),
            embedding_api_key: Some("secret".to_string()),
            batch_size: 32,
            ..Config::default()
        };

        let config = EmbeddingConfig::from_config(&app_config);

        assert_eq!(config.url, "https://api.openai.com/v1/embeddings");
        assert_eq!(config.model, "text-embedding-3-small");
        assert_eq!(config.batch_size, 32);
        assert_eq!(config.api_key.as_deref(), Some("secret"));
    }
}
