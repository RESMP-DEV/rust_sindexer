use anyhow::{Context, Result};
use reqwest::Client;
use reqwest::header::{AUTHORIZATION, HeaderMap};
use serde::{Deserialize, Serialize};

use crate::config::Config;
use crate::types::EmbeddingVector;

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
}

impl EmbeddingClient {
    /// Creates a new embedding client with the given configuration.
    pub fn new(config: EmbeddingConfig) -> Self {
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
            .pool_max_idle_per_host(32)
            .connect_timeout(std::time::Duration::from_secs(10))
            .timeout(std::time::Duration::from_secs(120))
            .build()
            .expect("failed to build HTTP client");

        Self { client, config }
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

        let mut all_embeddings = Vec::with_capacity(texts.len());

        for chunk in texts.chunks(self.config.batch_size) {
            let embeddings = self.embed_chunk(chunk).await?;
            all_embeddings.extend(embeddings);
        }

        Ok(all_embeddings)
    }

    /// Embeds a single chunk of texts (up to batch_size) with retry + backoff.
    async fn embed_chunk(&self, texts: &[String]) -> Result<Vec<EmbeddingVector>> {
        const MAX_RETRIES: u32 = 3;

        let mut last_err = None;
        for attempt in 0..=MAX_RETRIES {
            if attempt > 0 {
                let backoff = std::time::Duration::from_millis(500 * 2u64.pow(attempt - 1));
                tokio::time::sleep(backoff).await;
            }

            let request = EmbeddingRequest {
                input: texts,
                model: &self.config.model,
            };

            let response = match self.client.post(&self.config.url).json(&request).send().await {
                Ok(r) => r,
                Err(e) => {
                    last_err = Some(format!("request failed: {e}"));
                    continue;
                }
            };

            let status = response.status();
            if status == reqwest::StatusCode::TOO_MANY_REQUESTS
                || status.is_server_error()
            {
                let body = response.text().await.unwrap_or_default();
                last_err = Some(format!("embedding API returned {status}: {body}"));
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
