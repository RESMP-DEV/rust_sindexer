//! Native CLI mode: `sindexer <verb> ...`.
//!
//! Shares the exact indexing/search cores with the MCP tool layer (state,
//! indexer, hybrid fusion, lexical index) — the verbs below are thin arg
//! parsers around the same calls the MCP wrappers make, so behavior is
//! identical by construction. Output is compact JSON on stdout, one object
//! per invocation, matching the MCP tool result shapes.
//!
//! CLI-only convenience: when EMBEDDING_URL is unset and an OpenAI-compatible
//! embedding server answers on 127.0.0.1:1234, it is used automatically
//! (LM Studio nomic-embed on this machine), so `sindexer search` just works.

use std::net::TcpStream;
use std::path::{Path, PathBuf};
use std::time::Duration;

use anyhow::{anyhow, Result};
use serde_json::json;
use tokio::task;

use crate::config::Config;
use crate::mcp::hybrid::{fuse_hybrid_hits, HybridFusionOptions, HybridHit};
use crate::mcp::indexer;
use crate::mcp::state::{create_shared_state, SharedState};
use crate::lexical::LexicalIndex;
use crate::types::{IndexState, IndexStatus};
use crate::vectordb::collection_name_from_path;

pub const VERBS: &[&str] = &[
    "index", "search", "update", "status", "clear", "collections", "stats", "drop",
];

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
                                        [--ext rs,py] filters extensions.
    status <path>                       Indexing status for a codebase.
    clear <path>                        Remove a codebase's index.
    collections                         List indexed collections + row counts.
    stats <collection>                  Row count for one collection.
    drop <collection>                   Permanently drop one collection.

Paths may be relative; they are resolved to absolute. Output is compact JSON
on stdout. When EMBEDDING_URL is unset and a local embedding server answers
on 127.0.0.1:1234, it is used automatically; otherwise lexical-only mode.

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

fn resolve_path(raw: &str) -> Result<PathBuf> {
    let path = Path::new(raw);
    if !path.exists() {
        return Err(anyhow!("Path does not exist: {raw}"));
    }
    path.canonicalize().map_err(|err| anyhow!("cannot resolve {raw}: {err}"))
}

fn default_embedding_env() {
    if std::env::var("EMBEDDING_URL").is_ok_and(|v| !v.trim().is_empty()) {
        return;
    }
    if let Ok(stream) = TcpStream::connect_timeout(
        &"127.0.0.1:1234".parse().expect("valid socket addr"),
        Duration::from_millis(300),
    ) {
        drop(stream);
        // Local OpenAI-compatible embedding server detected (LM Studio).
        std::env::set_var("EMBEDDING_URL", "http://127.0.0.1:1234/v1");
    }
}

fn shared_state() -> SharedState {
    default_embedding_env();
    create_shared_state(Config::from_env())
}

fn should_request_live_rows(status: &IndexStatus) -> bool {
    match status.status {
        IndexState::Idle => true,
        IndexState::Completed => {
            status.vectors_inserted == 0
                || status.embeddings_generated == 0
                || status.total_chunks == 0
        }
        IndexState::Indexing | IndexState::Failed => false,
    }
}

fn can_apply_live_rows(status: &IndexStatus, requested_from_idle: bool) -> bool {
    match status.status {
        IndexState::Idle => true,
        IndexState::Completed => {
            requested_from_idle
                || status.vectors_inserted == 0
                || status.embeddings_generated == 0
                || status.total_chunks == 0
        }
        IndexState::Indexing | IndexState::Failed => false,
    }
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
        })
    );
    Ok(0)
}

async fn cmd_search(path: &Path, query: &str, limit: usize, extensions: Vec<String>) -> Result<i32> {
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
    .await??;

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
    if should_request_live_rows(&status) {
        let requested_from_idle = status.status == IndexState::Idle;
        let collection = collection_name_from_path(path);
        if let Ok(stats) = state.vector_store.collection_stats(&collection).await {
            let live_rows = usize::try_from(stats.row_count).unwrap_or(0);
            status = state.get_status(path);
            if live_rows > 0
                && can_apply_live_rows(&status, requested_from_idle)
                && (status.vectors_inserted < live_rows
                    || status.embeddings_generated < live_rows
                    || status.total_chunks < live_rows
                    || status.status == IndexState::Idle)
            {
                status.vectors_inserted = status.vectors_inserted.max(live_rows);
                status.embeddings_generated = status.embeddings_generated.max(live_rows);
                status.total_chunks = status.total_chunks.max(live_rows);
                status.status = IndexState::Completed;
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
    .await??;
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
    println!("{}", json!({ "collections": collections, "count": names.len() }));
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
        println!(
            "{}",
            json!({
                "success": false,
                "message": format!("Collection '{collection}' does not exist"),
                "collection_name": collection,
            })
        );
        return Ok(0);
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
/// a verb. Returns the process exit code.
pub async fn run(args: &[String]) -> Result<i32> {
    let mut iter = args.iter();
    let verb = iter.next().ok_or_else(|| anyhow!("missing verb"))?.as_str();
    let rest: Vec<&String> = iter.collect();

    match verb {
        "index" => {
            let (path, force) = parse_path_flag(&rest)?;
            cmd_index(&resolve_path(&path)?, force).await
        }
        "update" => {
            let (path, _) = parse_path_flag(&rest)?;
            cmd_update(&resolve_path(&path)?).await
        }
        "search" => {
            let mut path: Option<String> = None;
            let mut query: Option<String> = None;
            let mut limit = 10usize;
            let mut extensions: Vec<String> = Vec::new();
            let mut index = 0;
            while index < rest.len() {
                let arg = rest[index].as_str();
                match arg {
                    "--limit" | "-l" => {
                        index += 1;
                        let Some(value) = rest.get(index) else {
                            return usage_error("--limit needs a value");
                        };
                        limit = value.parse().map_err(|_| anyhow!("invalid --limit"))?;
                    }
                    "--ext" | "-e" => {
                        index += 1;
                        let Some(value) = rest.get(index) else {
                            return usage_error("--ext needs a value");
                        };
                        extensions = value.split(',').map(str::trim).filter(|s| !s.is_empty()).map(str::to_string).collect();
                    }
                    _ if path.is_none() => path = Some(arg.to_string()),
                    _ if query.is_none() => query = Some(arg.to_string()),
                    _ => return usage_error("search takes <path> <query> [--limit N] [--ext a,b]"),
                }
                index += 1;
            }
            let (Some(path), Some(query)) = (path, query) else {
                return usage_error("search takes <path> <query>");
            };
            if limit == 0 {
                return usage_error("--limit must be > 0");
            }
            cmd_search(&resolve_path(&path)?, &query, limit, extensions).await
        }
        "status" | "clear" => {
            let Some(path) = rest.first() else {
                return usage_error(&format!("{verb} takes <path>"));
            };
            let resolved = resolve_path(path)?;
            if verb == "status" {
                cmd_status(&resolved).await
            } else {
                cmd_clear(&resolved).await
            }
        }
        "collections" => cmd_collections().await,
        "stats" => match rest.first() {
            Some(collection) => cmd_stats(collection).await,
            None => usage_error("stats takes <collection>"),
        },
        "drop" => match rest.first() {
            Some(collection) => cmd_drop(collection).await,
            None => usage_error("drop takes <collection>"),
        },
        _ => usage_error(&format!("unknown verb '{verb}'")),
    }
}

fn parse_path_flag(rest: &[&String]) -> Result<(String, bool)> {
    let positional: Vec<&String> = rest.iter().filter(|arg| !arg.starts_with("--")).copied().collect();
    let Some(path) = positional.first() else {
        return Err(anyhow!("missing <path>"));
    };
    Ok(((*path).clone(), rest.iter().any(|arg| arg.as_str() == "--force")))
}
