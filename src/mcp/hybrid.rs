use crate::types::CodeChunk;
use std::collections::HashMap;
use tracing::{debug, info};

/// Result item for combined lexical / semantic retrieval flows.
#[derive(Clone, Debug)]
pub struct HybridHit {
    pub chunk: CodeChunk,
    pub score: f32,
}

/// Configuration for hybrid retrieval fusion.
#[derive(Clone, Debug, Default)]
pub struct HybridFusionOptions {
    pub limit: usize,
    pub extension_filter: Vec<String>,
}

/// Fuse semantic and lexical hits using reciprocal rank fusion.
pub fn fuse_hybrid_hits(
    _query: &str,
    semantic_hits: Vec<HybridHit>,
    lexical_hits: Vec<HybridHit>,
    options: &HybridFusionOptions,
) -> Vec<HybridHit> {
    debug!(
        semantic_count = semantic_hits.len(),
        lexical_count = lexical_hits.len(),
        limit = options.limit,
        "fusing hybrid search hits"
    );
    if options.limit == 0 {
        return Vec::new();
    }

    let mut fused = HashMap::new();
    merge_ranked_hits(
        &mut fused,
        filter_hits(semantic_hits, &options.extension_filter),
    );
    merge_ranked_hits(
        &mut fused,
        filter_hits(lexical_hits, &options.extension_filter),
    );

    let mut fused_hits: Vec<HybridHit> = fused.into_values().collect();
    fused_hits.sort_by(|left, right| right.score.total_cmp(&left.score));
    fused_hits.truncate(options.limit);
    info!(result_count = fused_hits.len(), "hybrid fusion complete");
    fused_hits
}

fn filter_hits(hits: Vec<HybridHit>, extension_filter: &[String]) -> Vec<HybridHit> {
    if extension_filter.is_empty() {
        return hits;
    }

    let normalized = extension_filter
        .iter()
        .map(|ext| ext.trim_start_matches('.').to_ascii_lowercase())
        .collect::<Vec<_>>();

    hits.into_iter()
        .filter(|hit| {
            hit.chunk
                .relative_path
                .rsplit('.')
                .next()
                .filter(|ext| *ext != hit.chunk.relative_path)
                .map(|ext| {
                    normalized
                        .iter()
                        .any(|candidate| candidate == &ext.to_ascii_lowercase())
                })
                .unwrap_or(false)
        })
        .collect()
}

fn merge_ranked_hits(fused: &mut HashMap<String, HybridHit>, hits: Vec<HybridHit>) {
    for (rank, hit) in hits.into_iter().enumerate() {
        let key = hit_key(&hit.chunk);
        let rrf_score = 1.0 / (60.0 + rank as f32 + 1.0);

        fused
            .entry(key)
            .and_modify(|existing| {
                existing.score += rrf_score;
                if existing.chunk.file_path.as_os_str().is_empty()
                    && !hit.chunk.file_path.as_os_str().is_empty()
                {
                    existing.chunk.file_path = hit.chunk.file_path.clone();
                }
            })
            .or_insert(HybridHit {
                chunk: hit.chunk,
                score: rrf_score,
            });
    }
}

fn hit_key(chunk: &CodeChunk) -> String {
    if !chunk.id.is_empty() {
        return chunk.id.clone();
    }

    format!(
        "{}:{}:{}:{}",
        chunk.relative_path, chunk.start_line, chunk.end_line, chunk.language
    )
}

#[cfg(test)]
mod tests {
    use super::{fuse_hybrid_hits, HybridFusionOptions, HybridHit};
    use crate::types::CodeChunk;
    use std::path::PathBuf;

    fn hit(
        id: &str,
        relative_path: &str,
        file_path: &str,
        start_line: u32,
        score: f32,
    ) -> HybridHit {
        HybridHit {
            chunk: CodeChunk {
                id: id.to_string(),
                content: format!("fn {id}() {{}}"),
                file_path: PathBuf::from(file_path),
                relative_path: relative_path.to_string(),
                start_line,
                end_line: start_line,
                language: "rust".to_string(),
            },
            score,
        }
    }

    #[test]
    fn test_fuse_hybrid_hits_merges_duplicate_hits() {
        let semantic_hits = vec![hit("chunk-1", "src/lib.rs", "/repo/src/lib.rs", 10, 0.9)];
        let lexical_hits = vec![hit("chunk-1", "src/lib.rs", "", 10, 12.0)];

        let fused = fuse_hybrid_hits(
            "lib",
            semantic_hits,
            lexical_hits,
            &HybridFusionOptions {
                limit: 10,
                ..Default::default()
            },
        );

        assert_eq!(fused.len(), 1);
        assert_eq!(fused[0].chunk.id, "chunk-1");
        assert_eq!(fused[0].chunk.file_path, PathBuf::from("/repo/src/lib.rs"));
    }

    #[test]
    fn test_fuse_hybrid_hits_respects_limit() {
        let semantic_hits = vec![
            hit("chunk-1", "src/a.rs", "/repo/src/a.rs", 1, 0.9),
            hit("chunk-2", "src/b.rs", "/repo/src/b.rs", 2, 0.8),
        ];
        let lexical_hits = vec![hit("chunk-3", "src/c.rs", "", 3, 10.0)];

        let fused = fuse_hybrid_hits(
            "lib",
            semantic_hits,
            lexical_hits,
            &HybridFusionOptions {
                limit: 2,
                ..Default::default()
            },
        );

        assert_eq!(fused.len(), 2);
    }

    #[test]
    fn test_fuse_hybrid_hits_filters_extensions() {
        let semantic_hits = vec![
            hit("chunk-1", "src/a.rs", "/repo/src/a.rs", 1, 0.9),
            hit("chunk-2", "src/b.py", "/repo/src/b.py", 2, 0.8),
        ];

        let fused = fuse_hybrid_hits(
            "lib",
            semantic_hits,
            Vec::new(),
            &HybridFusionOptions {
                limit: 10,
                extension_filter: vec!["rs".to_string()],
            },
        );

        assert_eq!(fused.len(), 1);
        assert_eq!(fused[0].chunk.relative_path, "src/a.rs");
    }
}
