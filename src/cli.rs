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

use anyhow::{anyhow, Result};
use serde_json::json;
use tokio::task;

use crate::config::Config;
use crate::lexical::LexicalIndex;
use crate::mcp::hybrid::{fuse_hybrid_hits, HybridFusionOptions, HybridHit};
use crate::mcp::indexer;
use crate::mcp::state::{create_shared_state, SharedState};
use crate::types::IndexStatus;
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
];

const EMBEDDING_VERBS: &[&str] = &["index", "update", "search"];

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

Paths may be relative; they are absolutized against the working directory but
symlinks are NOT resolved, so CLI and MCP modes key the same collection for
the same absolute path string. Output is compact JSON on stdout, logs on
stderr. For index/update/search, when EMBEDDING_URL is unset,
SINDEXER_AUTO_EMBEDDING != 0, and 127.0.0.1:1234 accepts connections,
EMBEDDING_URL defaults to http://127.0.0.1:1234/v1; otherwise lexical-only.

MCP MODE:
    With no arguments sindexer speaks newline-delimited JSON-RPC on stdio,
    meant to be launched by an MCP client. Set EMBEDDING_URL to enable
    semantic search, and MILVUS_URL to store vectors in Milvus/Zilliz.

OPTIONS:
    -h, --help       Print this help message and exit
    -V, --version    Print version information and exit

See README.md for the full configuration and usage reference.
"#;

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
    if std::env::var("EMBEDDING_URL").is_ok_and(|v| !v.trim().is_empty()) {
        return;
    }
    if std::env::var("SINDEXER_AUTO_EMBEDDING").is_ok_and(|v| v == "0") {
        return;
    }
    let addr = "127.0.0.1:1234".parse().expect("valid socket addr");
    if std::net::TcpStream::connect_timeout(&addr, std::time::Duration::from_millis(300)).is_ok() {
        // A local OpenAI-compatible embedding server is assumed to be
        // listening (LM Studio on this machine).
        std::env::set_var("EMBEDDING_URL", "http://127.0.0.1:1234/v1");
    }
}

fn shared_state() -> SharedState {
    create_shared_state(Config::from_env())
}

async fn cmd_index(path: &Path, force: bool) -> Result<i32> {
    let state = shared_state();
    let indexer_state = indexer::create_indexer_state(&state, path);
    let result = indexer::index_codebase(&indexer_state, path, force).await?;
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
    Ok(0)
}

async fn cmd_update(path: &Path) -> Result<i32> {
    let state = shared_state();
    let indexer_state = indexer::create_indexer_state(&state, path);
    let result = indexer::update_codebase_index(&indexer_state, path).await?;
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
    Ok(0)
}

async fn cmd_search(
    path: &Path,
    query: &str,
    limit: usize,
    extensions: Vec<String>,
) -> Result<i32> {
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
    println!("{}", json!({ "count": results.len(), "results": results }));
    Ok(0)
}

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

async fn cmd_stats(collection: &str) -> Result<i32> {
    let state = shared_state();
    let stats = state.vector_store.collection_stats(collection).await?;
    println!(
        "{}",
        json!({ "collection_name": collection, "row_count": stats.row_count })
    );
    Ok(0)
}

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
            let resolved = directory_path(&path)?;
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
        _ => usage_error(&format!("unknown verb '{verb}'")),
    }
}

#[cfg(test)]
mod tests {
    use super::run;

    fn args(parts: &[&str]) -> Vec<String> {
        parts.iter().map(|part| part.to_string()).collect()
    }

    #[tokio::test]
    async fn unknown_flag_is_a_usage_error() {
        assert_eq!(run(&args(&["index", "--bogus"])).await.unwrap(), 2);
        assert_eq!(
            run(&args(&["search", "p", "q", "--nope"])).await.unwrap(),
            2
        );
    }

    #[tokio::test]
    async fn extra_positionals_are_usage_errors() {
        assert_eq!(run(&args(&["index", "a", "b"])).await.unwrap(), 2);
        assert_eq!(run(&args(&["status", "a", "b"])).await.unwrap(), 2);
        assert_eq!(run(&args(&["collections", "x"])).await.unwrap(), 2);
        assert_eq!(run(&args(&["search", "a", "b", "c"])).await.unwrap(), 2);
    }

    #[tokio::test]
    async fn missing_arguments_are_usage_errors() {
        assert_eq!(run(&args(&["index"])).await.unwrap(), 2);
        assert_eq!(run(&args(&["search", "only-path"])).await.unwrap(), 2);
        assert_eq!(run(&args(&["drop"])).await.unwrap(), 2);
    }

    #[tokio::test]
    async fn update_rejects_force_flag() {
        assert_eq!(run(&args(&["update", "--force"])).await.unwrap(), 2);
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
