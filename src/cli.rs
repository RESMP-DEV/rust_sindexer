//! Native CLI mode: `sindexer <verb> ...`.
//!
//! Shares the exact indexing/search cores with the MCP tool layer (state,
//! indexer, hybrid fusion, lexical index, and the live-rows helpers in
//! indexer.rs) — the verbs below are thin arg parsers around the same calls
//! the MCP wrappers make. Output is compact JSON on stdout, one object per
//! invocation, matching the MCP tool result shapes.
//!
//! CLI-only convenience: for the embedding verbs (index/update/search), when
//! EMBEDDING_URL is unset, SINDEXER_AUTO_EMBEDDING is not "0", and
//! 127.0.0.1:1234 accepts a TCP connection, EMBEDDING_URL is set to a local
//! OpenAI-compatible endpoint (LM Studio nomic-embed on this machine). The
//! probe runs once in main(), before the async runtime starts, so no env
//! mutation races worker threads.

use std::path::{Path, PathBuf};
use std::time::Instant;

use anyhow::{anyhow, Result};
use serde_json::json;
use tokio::task;

use crate::config::Config;
use crate::lexical::LexicalIndex;
use crate::mcp::hybrid::{fuse_hybrid_hits, HybridFusionOptions, HybridHit};
use crate::mcp::indexer;
use crate::mcp::state::{create_shared_state, SharedState};
use crate::types::IndexStatus;
use crate::usage::{self, HitMetrics, IndexLog, SearchLog};
use crate::vectordb::collection_name_from_path;

pub const VERBS: &[&str] = &[
    "index",
    "update",
    "search",
    "status",
    "clear",
    "collections",
    "stats",
    "drop",
    "usage",
];

const EMBEDDING_VERBS: &[&str] = &["index", "update", "search"];

/// True when `arg` names one of the CLI verbs (main uses this to pick CLI
/// mode over MCP server mode).
pub fn is_verb(arg: &str) -> bool {
    VERBS.contains(&arg)
}

pub const HELP: &str = r#"sindexer — semantic code indexing (CLI + MCP server)

USAGE:
    sindexer <verb> [args]          CLI mode (verbs below)
    sindexer                        MCP server mode (stdio JSON-RPC)

CLI VERBS:
    index <path> [--force]              Index a codebase (full build when no
                                        compatible index exists).
    update <path>                       Incrementally update changed files.
    search <path> <query> [--limit N]   Hybrid semantic+lexical search;
                                        [--ext rs,py] filters extensions;
                                        `--` ends flag parsing.
    status <path>                       Indexing status for a codebase.
    clear <path>                        Remove a codebase's index (the path
                                        need not exist; use it to clean up
                                        after a repo is deleted).
    collections                         List indexed collections + row counts.
    stats <collection>                  Row count for one collection.
    drop <collection>                   Permanently drop one collection
                                        (exit 1 if it did not exist).
    usage [--since SPEC] [--repo TEXT] [--human]
                                        Report usage telemetry and estimated
                                        token savings. SPEC: Nh/Nd/Nw/Ny,
                                        YYYY-MM-DD, or epoch seconds/ms.

Paths may be relative; they are absolutized against the working directory but
symlinks are NOT resolved, so CLI and MCP modes key the same collection for
the same absolute path string. Output is compact JSON on stdout, logs on
stderr. For index/update/search, when EMBEDDING_URL is unset,
SINDEXER_AUTO_EMBEDDING != 0, and 127.0.0.1:1234 accepts connections,
EMBEDDING_URL defaults to http://127.0.0.1:1234/v1; otherwise lexical-only.

Every search and index/update run appends a usage event to
~/.context/usage/sindexer.jsonl (best-effort, never fails the command);
SINDEXER_USAGE_LOG overrides the path, SINDEXER_USAGE_LOG=0 disables.

MCP MODE:
    With no arguments sindexer speaks newline-delimited JSON-RPC on stdio,
    meant to be launched by an MCP client. Set EMBEDDING_URL to enable
    semantic search, and MILVUS_URL to store vectors in Milvus/Zilliz.

OPTIONS:
    -h, --help       Print this help message and exit
    -V, --version    Print version information and exit

See README.md for the full configuration and usage reference.
"#;

/// Print a usage mistake to stderr and return the usage exit code (2).
fn usage_error(message: &str) -> Result<i32> {
    eprintln!("sindexer: {message}\nRun 'sindexer --help' for usage.");
    Ok(2)
}

/// Absolutize without resolving symlinks, so collection identity (a hash of
/// the path string) matches what an MCP client would pass for the same repo.
fn absolutize(raw: &str) -> PathBuf {
    std::path::absolute(raw).unwrap_or_else(|_| PathBuf::from(raw))
}

/// Mirror of the MCP validate_directory_path: must exist and be a directory.
fn directory_path(raw: &str) -> Result<PathBuf> {
    let path = Path::new(raw);
    if !path.exists() {
        return Err(anyhow!("Path does not exist: {raw}"));
    }
    if !path.is_dir() {
        return Err(anyhow!("Path is not a directory: {raw}"));
    }
    Ok(absolutize(raw))
}

/// Run before the async runtime starts (from main), for embedding verbs only.
/// Sync code, so the blocking connect and env mutation are both safe here.
pub fn prepare_environment(verb: &str) {
    if !EMBEDDING_VERBS.contains(&verb) {
        return;
    }
    let is_set = |key: &str| std::env::var(key).is_ok_and(|v| !v.trim().is_empty());
    // config.rs accepts OPENAI_BASE_URL as an alias for EMBEDDING_URL; an
    // explicit configuration of either must never be overridden.
    if is_set("EMBEDDING_URL") || is_set("OPENAI_BASE_URL") {
        return;
    }
    if std::env::var("SINDEXER_AUTO_EMBEDDING").is_ok_and(|v| v == "0") {
        return;
    }
    if let Some(dimension) = probe_local_embeddings() {
        std::env::set_var("EMBEDDING_URL", "http://127.0.0.1:1234/v1");
        std::env::set_var("EMBEDDING_DIMENSION", dimension.to_string());
    }
}

/// Verify 127.0.0.1:1234 is actually an embeddings endpoint and measure the
/// vector dimension, so the auto-default cannot mismatch the collection
/// schema. Best-effort: any failure means lexical-only mode (no env set).
fn probe_local_embeddings() -> Option<usize> {
    use std::io::{Read, Write};
    use std::net::{SocketAddr, TcpStream};
    let addr: SocketAddr = "127.0.0.1:1234".parse().ok()?;
    let mut stream = TcpStream::connect_timeout(&addr, std::time::Duration::from_secs(1)).ok()?;
    let _ = stream.set_read_timeout(Some(std::time::Duration::from_secs(2)));
    let _ = stream.set_write_timeout(Some(std::time::Duration::from_secs(2)));
    let body = r#"{"input":["sindexer dimension probe"]}"#;
    let request = format!(
        "POST /v1/embeddings HTTP/1.1\r\nHost: 127.0.0.1:1234\r\nContent-Type: application/json\r\nConnection: close\r\nContent-Length: {}\r\n\r\n{body}",
        body.len()
    );
    stream.write_all(request.as_bytes()).ok()?;
    let mut response = String::new();
    stream.read_to_string(&mut response).ok()?;
    let payload = response.split_once("\r\n\r\n")?.1;
    // Skip the HTTP preamble by finding the JSON object start, then parse
    // exactly one JSON value so trailing chunked-transfer metadata
    // (`\r\n0\r\n\r\n`) is ignored.
    let json_start = payload.find('{')?;
    let value: serde_json::Value = serde_json::Deserializer::from_str(&payload[json_start..])
        .into_iter::<serde_json::Value>()
        .next()?
        .ok()?;
    let dimension = value
        .pointer("/data/0/embedding")
        .and_then(|array| array.as_array())
        .map(|array| array.len())?;
    if dimension == 0 {
        return None;
    }
    Some(dimension)
}

/// Per-invocation shared state built from the environment configuration.
fn shared_state() -> SharedState {
    create_shared_state(Config::from_env())
}

/// `index` verb: build or rebuild the index, print the JSON summary.
async fn cmd_index(path: &Path, force: bool) -> Result<i32> {
    let start = Instant::now();
    let state = shared_state();
    let indexer_state = indexer::create_indexer_state(&state, path);
    let outcome = indexer::index_codebase(&indexer_state, path, force).await;
    let duration_ms = start.elapsed().as_millis() as u64;
    match outcome {
        Ok(result) => {
            println!(
                "{}",
                json!({
                    "success": true,
                    "message": format!("Indexed {}", path.display()),
                    "path": path.display().to_string(),
                    "files_indexed": result.files_processed,
                    "chunks_created": result.chunks_created,
                    "lexical_only": result.lexical_only,
                    "warnings": result.warnings,
                })
            );
            usage::log_index(IndexLog {
                mode: "cli",
                verb: "index",
                repo: path,
                files_processed: result.files_processed,
                chunks_created: result.chunks_created,
                duration_ms,
                lexical_only: result.lexical_only,
                error: None,
            });
            Ok(0)
        }
        Err(err) => {
            let message = err.to_string();
            usage::log_index(IndexLog {
                mode: "cli",
                verb: "index",
                repo: path,
                files_processed: 0,
                chunks_created: 0,
                duration_ms,
                lexical_only: false,
                error: Some(&message),
            });
            Err(err)
        }
    }
}

/// `update` verb: incremental refresh, print the JSON summary.
async fn cmd_update(path: &Path) -> Result<i32> {
    let start = Instant::now();
    let state = shared_state();
    let indexer_state = indexer::create_indexer_state(&state, path);
    let outcome = indexer::update_codebase_index(&indexer_state, path).await;
    let duration_ms = start.elapsed().as_millis() as u64;
    match outcome {
        Ok(result) => {
            println!(
                "{}",
                json!({
                    "success": true,
                    "message": format!("Updated {}", path.display()),
                    "path": path.display().to_string(),
                    "files_indexed": result.files_processed,
                    "chunks_created": result.chunks_created,
                    "lexical_only": result.lexical_only,
                    "warnings": result.warnings,
                })
            );
            usage::log_index(IndexLog {
                mode: "cli",
                verb: "update",
                repo: path,
                files_processed: result.files_processed,
                chunks_created: result.chunks_created,
                duration_ms,
                lexical_only: result.lexical_only,
                error: None,
            });
            Ok(0)
        }
        Err(err) => {
            let message = err.to_string();
            usage::log_index(IndexLog {
                mode: "cli",
                verb: "update",
                repo: path,
                files_processed: 0,
                chunks_created: 0,
                duration_ms,
                lexical_only: false,
                error: Some(&message),
            });
            Err(err)
        }
    }
}

/// `search` verb: hybrid semantic+lexical search, print fused hits as JSON.
async fn cmd_search(
    path: &Path,
    query: &str,
    limit: usize,
    extensions: Vec<String>,
) -> Result<i32> {
    let start = Instant::now();
    let outcome = perform_search(path, query, limit, extensions).await;
    let duration_ms = start.elapsed().as_millis() as u64;
    match outcome {
        Ok((payload, metrics)) => {
            let output_bytes = payload.len();
            println!("{payload}");
            usage::log_search(SearchLog {
                mode: "cli",
                repo: path,
                query,
                limit,
                duration_ms,
                metrics,
                output_bytes,
                error: None,
            });
            Ok(0)
        }
        Err(err) => {
            let message = err.to_string();
            usage::log_search(SearchLog {
                mode: "cli",
                repo: path,
                query,
                limit,
                duration_ms,
                metrics: HitMetrics::default(),
                output_bytes: 0,
                error: Some(&message),
            });
            Err(err)
        }
    }
}

/// The search core without output or telemetry: returns the serialized
/// stdout payload plus the hit measurements for the usage log.
async fn perform_search(
    path: &Path,
    query: &str,
    limit: usize,
    extensions: Vec<String>,
) -> Result<(String, HitMetrics)> {
    let state = shared_state();
    let collection = collection_name_from_path(path);

    let vector_hits = if state.embedder.is_enabled() {
        state
            .search(&collection, query, limit)
            .await?
            .into_iter()
            .map(|result| HybridHit {
                chunk: result.chunk,
                score: result.score,
            })
            .collect::<Vec<_>>()
    } else {
        Vec::new()
    };

    let lexical_path = path.to_path_buf();
    let lexical_query = query.to_string();
    let lexical_hits = task::spawn_blocking(move || -> anyhow::Result<Vec<HybridHit>> {
        if !LexicalIndex::exists(&lexical_path)? {
            return Ok(Vec::new());
        }
        let lexical_index = LexicalIndex::open(&lexical_path)?;
        let mut hits = lexical_index.search(&lexical_query, limit)?;
        for hit in &mut hits {
            if hit.chunk.file_path.as_os_str().is_empty() {
                hit.chunk.file_path = lexical_path.join(&hit.chunk.relative_path);
            }
        }
        Ok(hits)
    })
    .await
    .map_err(|err| anyhow!("failed to join lexical search task: {err}"))?
    .map_err(|err| anyhow!("failed to search lexical index: {err}"))?;

    let options = HybridFusionOptions {
        limit,
        extension_filter: extensions,
    };
    let fused = fuse_hybrid_hits(query, vector_hits, lexical_hits, &options);
    let metrics = usage::measure_hits(path, &fused);
    let results: Vec<serde_json::Value> = fused
        .into_iter()
        .map(|hit| {
            json!({
                "content": hit.chunk.content,
                "end_line": hit.chunk.end_line,
                "file_path": hit.chunk.file_path.display().to_string(),
                "language": hit.chunk.language,
                "relative_path": hit.chunk.relative_path,
                "score": hit.score,
                "start_line": hit.chunk.start_line,
            })
        })
        .collect();
    Ok((
        json!({ "count": results.len(), "results": results }).to_string(),
        metrics,
    ))
}

/// `status` verb: index status with live-rows reconciliation, as JSON.
async fn cmd_status(path: &Path) -> Result<i32> {
    let state = shared_state();
    let mut status = state.get_status(path);
    if indexer::should_request_live_rows(&status) {
        let requested_from_idle = status.status == crate::types::IndexState::Idle;
        let collection = collection_name_from_path(path);
        if let Ok(stats) = state.vector_store.collection_stats(&collection).await {
            let live_rows = indexer::checked_live_row_count(stats.row_count);
            status = state.get_status(path);
            if live_rows > 0
                && indexer::can_apply_live_rows(&status, requested_from_idle)
                && (status.vectors_inserted < live_rows
                    || status.embeddings_generated < live_rows
                    || status.total_chunks < live_rows
                    || status.status == crate::types::IndexState::Idle)
            {
                status.vectors_inserted = status.vectors_inserted.max(live_rows);
                status.embeddings_generated = status.embeddings_generated.max(live_rows);
                status.total_chunks = status.total_chunks.max(live_rows);
                status.status = crate::types::IndexState::Completed;
                state.set_status(path.to_path_buf(), status.clone());
            }
        }
    }
    println!("{}", serde_json::to_string(&status)?);
    Ok(0)
}

/// `clear` verb: drop the vector collection and lexical index for a path and
/// reset its status.
async fn cmd_clear(path: &Path) -> Result<i32> {
    let state = shared_state();
    let collection_name = collection_name_from_path(path);
    let had_collection = state.vector_store.has_collection(&collection_name).await?;
    if had_collection {
        state.vector_store.drop_collection(&collection_name).await?;
    }
    state.set_status(path.to_path_buf(), IndexStatus::default());
    let _ = state.manifest_store.clear_status(path);
    state.indexing_status.remove(path);
    let lexical_path = path.to_path_buf();
    task::spawn_blocking(move || -> anyhow::Result<()> {
        let lexical_index = LexicalIndex::create(&lexical_path)?;
        lexical_index.clear()?;
        Ok(())
    })
    .await
    .map_err(|err| anyhow!("failed to join lexical clear task: {err}"))?
    .map_err(|err| anyhow!("failed to clear lexical index: {err}"))?;
    println!(
        "{}",
        json!({
            "success": true,
            "message": if had_collection {
                format!("Cleared index for {}", path.display())
            } else {
                format!("No index existed for {}; status reset to idle", path.display())
            },
            "path": path.display().to_string(),
        })
    );
    Ok(0)
}

/// `collections` verb: list vector collections with row counts.
async fn cmd_collections() -> Result<i32> {
    let state = shared_state();
    let names = state.vector_store.list_collections().await?;
    let mut collections = Vec::with_capacity(names.len());
    for name in &names {
        let row_count = state
            .vector_store
            .collection_stats(name)
            .await
            .map(|stats| stats.row_count)
            .unwrap_or(0);
        collections.push(json!({ "name": name, "row_count": row_count }));
    }
    println!(
        "{}",
        json!({ "collections": collections, "count": names.len() })
    );
    Ok(0)
}

/// `stats` verb: row count for one collection.
async fn cmd_stats(collection: &str) -> Result<i32> {
    let state = shared_state();
    let stats = state.vector_store.collection_stats(collection).await?;
    println!(
        "{}",
        json!({ "collection_name": collection, "row_count": stats.row_count })
    );
    Ok(0)
}

/// `drop` verb: permanently drop one collection; exit 1 when it is missing.
async fn cmd_drop(collection: &str) -> Result<i32> {
    let state = shared_state();
    let exists = state.vector_store.has_collection(collection).await?;
    if !exists {
        // Data parity with the MCP tool, but a CLI exit code must be
        // detectable in scripts: report the miss and fail.
        println!(
            "{}",
            json!({
                "success": false,
                "message": format!("Collection '{collection}' does not exist"),
                "collection_name": collection,
            })
        );
        return Ok(1);
    }
    state.vector_store.drop_collection(collection).await?;
    println!(
        "{}",
        json!({
            "success": true,
            "message": format!("Dropped collection '{collection}'"),
            "collection_name": collection,
        })
    );
    Ok(0)
}

async fn cmd_usage(rest: &[&String]) -> Result<i32> {
    let mut since: Option<String> = None;
    let mut repo: Option<String> = None;
    let mut human = false;
    let mut iter = rest.iter();
    while let Some(arg) = iter.next() {
        match arg.as_str() {
            "--since" => {
                let Some(value) = iter.next() else {
                    return usage_error("--since needs a value");
                };
                since = Some((*value).clone());
            }
            "--repo" => {
                let Some(value) = iter.next() else {
                    return usage_error("--repo needs a value");
                };
                repo = Some((*value).clone());
            }
            "--human" => human = true,
            other => {
                return usage_error(&format!(
                    "usage takes --since/--repo/--human, got '{other}'"
                ));
            }
        }
    }
    let since_ms = match since.as_deref() {
        None => None,
        Some(spec) => match usage::parse_since(spec, usage::now_ms()) {
            Some(bound) => Some(bound),
            None => {
                return usage_error(&format!(
                    "invalid --since value '{spec}' (expected Nh/Nd/Nw/Ny, YYYY-MM-DD, or epoch)"
                ));
            }
        },
    };
    let filter = usage::UsageFilter {
        since_ms,
        since_spec: since,
        repo_contains: repo,
    };
    let report = usage::report_json(&filter);
    if human {
        println!("{}", usage::render_human(&report));
    } else {
        println!("{report}");
    }
    Ok(0)
}

/// Entry point for CLI mode. `args` excludes the program name and starts with
/// a verb. Returns the process exit code. Usage mistakes return Ok(2);
/// runtime failures propagate as Err (exit 1 in main).
pub async fn run(args: &[String]) -> Result<i32> {
    let mut iter = args.iter();
    let verb = iter.next().ok_or_else(|| anyhow!("missing verb"))?.as_str();
    let rest: Vec<&String> = iter.collect();

    match verb {
        "index" | "update" => {
            let mut path: Option<String> = None;
            let mut force = false;
            for arg in &rest {
                if arg.as_str() == "--force" {
                    if verb == "update" {
                        return usage_error("update takes no --force flag");
                    }
                    force = true;
                } else if arg.starts_with("--") {
                    return usage_error(&format!("unknown flag '{arg}' for {verb}"));
                } else if path.is_none() {
                    path = Some((*arg).clone());
                } else {
                    return usage_error(&format!("{verb} takes exactly one <path>"));
                }
            }
            let Some(path) = path else {
                return usage_error(&format!("{verb} takes <path>"));
            };
            let resolved = directory_path(&path)?;
            if verb == "index" {
                cmd_index(&resolved, force).await
            } else {
                cmd_update(&resolved).await
            }
        }
        "search" => {
            let mut path: Option<String> = None;
            let mut query: Option<String> = None;
            let mut limit = 10usize;
            let mut extensions: Vec<String> = Vec::new();
            let mut end_of_flags = false;
            let mut index = 0;
            while index < rest.len() {
                let arg = rest[index].as_str();
                if !end_of_flags && arg == "--" {
                    end_of_flags = true;
                } else if !end_of_flags && (arg == "--limit" || arg == "-l") {
                    index += 1;
                    let Some(value) = rest.get(index) else {
                        return usage_error("--limit needs a value");
                    };
                    let parsed: usize = match value.parse() {
                        Ok(parsed) => parsed,
                        Err(_) => {
                            return usage_error(&format!("invalid --limit value '{value}'"));
                        }
                    };
                    limit = parsed;
                    if limit == 0 {
                        return usage_error("--limit must be > 0");
                    }
                } else if !end_of_flags && (arg == "--ext" || arg == "-e") {
                    index += 1;
                    let Some(value) = rest.get(index) else {
                        return usage_error("--ext needs a value");
                    };
                    extensions.extend(
                        value
                            .split(',')
                            .map(str::trim)
                            .filter(|s| !s.is_empty())
                            .map(str::to_string),
                    );
                } else if !end_of_flags && arg.starts_with('-') && arg.len() > 1 {
                    return usage_error(&format!("unknown flag '{arg}'"));
                } else if path.is_none() {
                    path = Some(arg.to_string());
                } else if query.is_none() {
                    query = Some(arg.to_string());
                } else {
                    return usage_error("search takes <path> <query>");
                }
                index += 1;
            }
            let (Some(path), Some(query)) = (path, query) else {
                return usage_error("search takes <path> <query>");
            };
            // Parity with the MCP tool: validation failures are logged as
            // error searches too (they measure fallback-to-grep incidents).
            let resolved = match directory_path(&path) {
                Ok(resolved) => resolved,
                Err(err) => {
                    let message = err.to_string();
                    usage::log_search(SearchLog {
                        mode: "cli",
                        repo: Path::new(&path),
                        query: &query,
                        limit,
                        duration_ms: 0,
                        metrics: HitMetrics::default(),
                        output_bytes: 0,
                        error: Some(&message),
                    });
                    return Err(err);
                }
            };
            cmd_search(&resolved, &query, limit, extensions).await
        }
        "status" | "clear" => {
            if rest.len() != 1 {
                return usage_error(&format!("{verb} takes exactly one <path>"));
            }
            let raw = rest[0];
            if verb == "status" {
                // Mirror of the MCP tool: status requires the path to exist.
                if !Path::new(raw).exists() {
                    return Err(anyhow!("Path does not exist: {raw}"));
                }
                cmd_status(&absolutize(raw)).await
            } else {
                // Mirror of the MCP tool: clear must work for deleted repos
                // (orphan cleanup), so no existence check here.
                cmd_clear(&absolutize(raw)).await
            }
        }
        "collections" => {
            if !rest.is_empty() {
                return usage_error("collections takes no arguments");
            }
            cmd_collections().await
        }
        "stats" | "drop" => {
            if rest.len() != 1 || rest[0].starts_with("--") {
                return usage_error(&format!("{verb} takes exactly one <collection>"));
            }
            let collection = rest[0];
            if verb == "stats" {
                cmd_stats(collection).await
            } else {
                cmd_drop(collection).await
            }
        }
        "usage" => cmd_usage(&rest).await,
        _ => usage_error(&format!("unknown verb '{verb}'")),
    }
}

#[cfg(test)]
mod tests {
    use super::run;

    /// Build a `Vec<String>` argument vector for `run`.
    fn args(parts: &[&str]) -> Vec<String> {
        parts.iter().map(|part| part.to_string()).collect()
    }

    /// Unknown flags are usage errors (exit 2).
    #[tokio::test]
    async fn unknown_flag_is_a_usage_error() {
        assert_eq!(run(&args(&["index", "--bogus"])).await.unwrap(), 2);
        assert_eq!(
            run(&args(&["search", "p", "q", "--nope"])).await.unwrap(),
            2
        );
    }

    /// Extra positionals are usage errors (exit 2).
    #[tokio::test]
    async fn extra_positionals_are_usage_errors() {
        assert_eq!(run(&args(&["index", "a", "b"])).await.unwrap(), 2);
        assert_eq!(run(&args(&["status", "a", "b"])).await.unwrap(), 2);
        assert_eq!(run(&args(&["collections", "x"])).await.unwrap(), 2);
        assert_eq!(run(&args(&["search", "a", "b", "c"])).await.unwrap(), 2);
    }

    /// Missing required arguments are usage errors (exit 2).
    #[tokio::test]
    async fn missing_arguments_are_usage_errors() {
        assert_eq!(run(&args(&["index"])).await.unwrap(), 2);
        assert_eq!(run(&args(&["search", "only-path"])).await.unwrap(), 2);
        assert_eq!(run(&args(&["drop"])).await.unwrap(), 2);
    }

    /// `update` rejects a `--force` flag (exit 2).
    #[tokio::test]
    async fn update_rejects_force_flag() {
        assert_eq!(run(&args(&["update", "--force"])).await.unwrap(), 2);
    }

    /// A query starting with `-` after `--` is kept as the query, not a flag.
    #[tokio::test]
    async fn usage_verb_validates_flags() {
        assert_eq!(run(&args(&["usage", "--bogus"])).await.unwrap(), 2);
        assert_eq!(run(&args(&["usage", "--since"])).await.unwrap(), 2);
        assert_eq!(run(&args(&["usage", "--repo"])).await.unwrap(), 2);
        assert_eq!(run(&args(&["usage", "positional"])).await.unwrap(), 2);
        assert_eq!(
            run(&args(&["usage", "--since", "not-a-spec"]))
                .await
                .unwrap(),
            2
        );
    }

    #[tokio::test]
    // The env guard must stay held for the whole `run` call, which reads
    // SINDEXER_USAGE_LOG while awaiting; a scoped lock would drop protection.
    #[allow(clippy::await_holding_lock)]
    async fn usage_verb_reports_from_configured_log() {
        let lock = crate::usage::test_support::ENV_LOCK
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());
        let dir = tempfile::tempdir().unwrap();
        let log_path = dir.path().join("usage.jsonl");
        std::fs::write(
            &log_path,
            concat!(
                r#"{"v":"0.1.0","ts_ms":1000,"ts":"1970-01-01T00:00:01Z","event":"search","mode":"cli","repo":"/r","query":"q","limit":5,"results":1,"excerpt_chars":10,"hit_files":1,"hit_file_bytes":800,"top_hit_file_bytes":800,"output_bytes":200,"duration_ms":5,"error":null}"#,
                "\n"
            ),
        )
        .unwrap();
        std::env::set_var(crate::usage::USAGE_LOG_ENV, log_path.to_str().unwrap());

        let code = run(&args(&["usage"])).await.unwrap();
        std::env::remove_var(crate::usage::USAGE_LOG_ENV);
        drop(lock);
        assert_eq!(code, 0);
    }

    #[tokio::test]
    async fn query_starting_with_dash_after_separator_is_kept() {
        // Path does not exist, so this is a runtime error (exit via Err),
        // proving the query was not parsed as a flag.
        assert!(
            run(&args(&["search", "/definitely/not/here", "--", "--limit"]))
                .await
                .is_err()
        );
    }
}
