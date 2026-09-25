//! Best-effort usage telemetry measuring how much agent context the
//! index-first search flow saves compared to reading whole files.
//!
//! Every search (CLI `search` verb and MCP `search_code` tool) and every CLI
//! index/update run appends one JSON line to the usage log. The default log
//! is `~/.context/usage/sindexer.jsonl`; `SINDEXER_USAGE_LOG` overrides the
//! path, and `SINDEXER_USAGE_LOG=0` (or `off`/`false`) disables logging
//! entirely. Telemetry writes never fail the command being measured: all
//! errors are swallowed after a debug log line.
//!
//! The `usage` verb aggregates the log into token-savings estimates. Tokens
//! are approximated as bytes/4, the search side is the measured output
//! payload agents actually read, and the counterfactual baselines model the
//! grep-first flow the index replaces (reading the hit files in full). See
//! `METHODOLOGY` for the exact wording shipped in every report.

use std::collections::BTreeMap;
use std::fs::OpenOptions;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use serde_json::{json, Value};

use crate::mcp::hybrid::HybridHit;
use crate::types::CodeChunk;

pub const USAGE_LOG_ENV: &str = "SINDEXER_USAGE_LOG";

const METHODOLOGY: &str = "Tokens are estimated as bytes/4. search_output is \
the measured JSON payload agents read (CLI stdout or MCP tool result). The \
baselines model the grep-first flow this index replaces: the agent reads the \
files containing the hits — 'read all hit files' assumes every unique file \
among the results is read in full, 'read top hit file' assumes only the \
best-ranked hit's file is read. Targeted follow-up reads after a search are \
not counted, so true per-search savings fall between the two baselines. \
Index maintenance costs local compute, not context tokens.";

/// Per-search measurements derived from the fused hit list.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct HitMetrics {
    pub results: usize,
    pub excerpt_chars: usize,
    pub hit_files: usize,
    pub hit_file_bytes: u64,
    pub top_hit_file_bytes: u64,
}

/// Measure a fused hit list: excerpt volume returned versus the on-disk size
/// of the files the hits point at. File sizes are best-effort — missing
/// files contribute zero.
pub fn measure_hits(root: &Path, hits: &[HybridHit]) -> HitMetrics {
    let mut sizes: BTreeMap<&str, u64> = BTreeMap::new();
    let mut excerpt_chars = 0usize;
    let mut top_hit_file_bytes = 0u64;
    for (index, hit) in hits.iter().enumerate() {
        excerpt_chars += hit.chunk.content.chars().count();
        let size = file_size(root, &hit.chunk);
        if index == 0 {
            top_hit_file_bytes = size;
        }
        sizes
            .entry(hit.chunk.relative_path.as_str())
            .or_insert(size);
    }
    HitMetrics {
        results: hits.len(),
        excerpt_chars,
        hit_files: sizes.len(),
        hit_file_bytes: sizes.values().sum(),
        top_hit_file_bytes,
    }
}

fn file_size(root: &Path, chunk: &CodeChunk) -> u64 {
    std::fs::metadata(&chunk.file_path)
        .or_else(|_| std::fs::metadata(root.join(&chunk.relative_path)))
        .map(|metadata| metadata.len())
        .unwrap_or(0)
}

/// One search event, from either the CLI verb or the MCP tool.
pub struct SearchLog<'a> {
    /// "cli" or "mcp"
    pub mode: &'a str,
    pub repo: &'a Path,
    pub query: &'a str,
    pub limit: usize,
    pub duration_ms: u64,
    pub metrics: HitMetrics,
    /// Serialized payload size the caller read (stdout JSON / tool result).
    pub output_bytes: usize,
    pub error: Option<&'a str>,
}

pub fn log_search(log: SearchLog<'_>) {
    let value = json!({
        "event": "search",
        "mode": log.mode,
        "repo": log.repo.display().to_string(),
        "query": log.query,
        "limit": log.limit,
        "results": log.metrics.results,
        "excerpt_chars": log.metrics.excerpt_chars,
        "hit_files": log.metrics.hit_files,
        "hit_file_bytes": log.metrics.hit_file_bytes,
        "top_hit_file_bytes": log.metrics.top_hit_file_bytes,
        "output_bytes": log.output_bytes,
        "duration_ms": log.duration_ms,
        "error": log.error,
    });
    append_event(value);
}

/// One index/update run (CLI verbs; MCP index tools where wired).
pub struct IndexLog<'a> {
    pub mode: &'a str,
    /// "index" or "update"
    pub verb: &'a str,
    pub repo: &'a Path,
    pub files_processed: usize,
    pub chunks_created: usize,
    pub duration_ms: u64,
    pub lexical_only: bool,
    pub error: Option<&'a str>,
}

pub fn log_index(log: IndexLog<'_>) {
    let value = json!({
        "event": "index",
        "mode": log.mode,
        "verb": log.verb,
        "repo": log.repo.display().to_string(),
        "files_processed": log.files_processed,
        "chunks_created": log.chunks_created,
        "duration_ms": log.duration_ms,
        "lexical_only": log.lexical_only,
        "error": log.error,
    });
    append_event(value);
}

/// Resolved log destination: `None` means logging is disabled (either
/// explicitly or because no home directory is known).
pub fn usage_log_path() -> Option<PathBuf> {
    if let Ok(value) = std::env::var(USAGE_LOG_ENV) {
        let trimmed = value.trim();
        if trimmed.is_empty() || matches!(trimmed, "0" | "off" | "false" | "no") {
            return None;
        }
        return Some(PathBuf::from(trimmed));
    }
    home_dir().map(|home| home.join(".context/usage/sindexer.jsonl"))
}

fn usage_log_disabled() -> bool {
    std::env::var(USAGE_LOG_ENV).is_ok_and(|value| {
        let trimmed = value.trim();
        trimmed.is_empty() || matches!(trimmed, "0" | "off" | "false" | "no")
    })
}

fn home_dir() -> Option<PathBuf> {
    std::env::var_os("HOME")
        .filter(|home| !home.is_empty())
        .map(PathBuf::from)
}

fn append_event(mut value: Value) {
    // Test binaries must never write the real usage log: integration-style
    // tests call search/index with tempdir repos and mock servers, and their
    // synthetic events would corrupt the token-savings data. Explicit
    // SINDEXER_USAGE_LOG overrides (used by usage.rs's own tests) always go
    // through, so the write path itself stays covered.
    if cfg!(test) && std::env::var_os(USAGE_LOG_ENV).is_none() {
        return;
    }
    let Some(path) = usage_log_path() else {
        return;
    };
    let ts_ms = now_ms();
    if let Some(object) = value.as_object_mut() {
        object.insert("v".into(), json!(env!("CARGO_PKG_VERSION")));
        object.insert("ts_ms".into(), json!(ts_ms));
        object.insert("ts".into(), json!(iso_utc_from_ms(ts_ms)));
    }
    let mut line = value.to_string();
    line.push('\n');
    let write = || -> std::io::Result<()> {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let mut file = OpenOptions::new().create(true).append(true).open(&path)?;
        file.write_all(line.as_bytes())
    };
    if let Err(err) = write() {
        tracing::debug!(error = %err, path = %path.display(), "usage log append failed");
    }
}

pub fn now_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_millis() as u64)
        .unwrap_or(0)
}

/// Format epoch milliseconds as `YYYY-MM-DDTHH:MM:SSZ` (UTC) without a time
/// crate, via Howard Hinnant's civil-from-days algorithm.
pub fn iso_utc_from_ms(ms: u64) -> String {
    let secs = ms / 1000;
    let days = (secs / 86_400) as i64;
    let secs_of_day = secs % 86_400;
    let (year, month, day) = civil_from_days(days);
    format!(
        "{year:04}-{month:02}-{day:02}T{:02}:{:02}:{:02}Z",
        secs_of_day / 3600,
        (secs_of_day / 60) % 60,
        secs_of_day % 60
    )
}

/// Inverse of `days_from_civil`, both from Howard Hinnant's
/// date algorithms. Input is days since 1970-01-01.
fn civil_from_days(days: i64) -> (i64, u32, u32) {
    let z = days + 719_468;
    let era = if z >= 0 { z } else { z - 146_096 } / 146_097;
    let doe = (z - era * 146_097) as u64; // [0, 146096]
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365; // [0, 399]
    let y = yoe as i64 + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100); // [0, 365]
    let mp = (5 * doy + 2) / 153; // [0, 11]
    let day = (doy - (153 * mp + 2) / 5 + 1) as u32; // [1, 31]
    let month = if mp < 10 { mp + 3 } else { mp - 9 } as u32; // [1, 12]
    (if month <= 2 { y + 1 } else { y }, month, day)
}

fn days_from_civil(year: i64, month: u32, day: u32) -> i64 {
    let year = if month <= 2 { year - 1 } else { year };
    let era = if year >= 0 { year } else { year - 399 } / 400;
    let yoe = (year - era * 400) as u64; // [0, 399]
    let mp = if month > 2 { month - 3 } else { month + 9 } as u64; // [0, 11]
    let doy = (153 * mp + 2) / 5 + day as u64 - 1; // [0, 365]
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy; // [0, 146096]
    era * 146_097 + doe as i64 - 719_468
}

/// Parse a `--since` spec against a reference "now" in epoch ms: `Nh`,
/// `Nd`, `Nw`, `Ny` relative windows, a `YYYY-MM-DD` UTC date, or a raw
/// epoch in seconds (< 1e11) or milliseconds. Returns the lower bound in
/// epoch ms. Future dates clamp to the reference time.
pub fn parse_since(spec: &str, now_ms: u64) -> Option<u64> {
    let spec = spec.trim();
    let relative = |suffix: char, unit_ms: u64| -> Option<u64> {
        let digits = spec.strip_suffix(suffix)?;
        if digits.is_empty() || !digits.bytes().all(|b| b.is_ascii_digit()) {
            return None;
        }
        let count: u64 = digits.parse().ok()?;
        // Oversized windows saturate to the epoch, never fail or underflow.
        let span = count.saturating_mul(unit_ms);
        Some(now_ms.saturating_sub(span))
    };
    if let Some(bound) = relative('h', 3_600_000)
        .or_else(|| relative('d', 86_400_000))
        .or_else(|| relative('w', 7 * 86_400_000))
        .or_else(|| relative('y', 365 * 86_400_000))
    {
        return Some(bound);
    }
    if let Ok(epoch) = spec.parse::<u64>() {
        // Below 100 Gs epoch values predate the Unix era by a wide margin;
        // treat anything larger as already-milliseconds.
        let ms = if epoch < 100_000_000_000 {
            epoch * 1000
        } else {
            epoch
        };
        return Some(ms.min(now_ms));
    }
    let date = spec.as_bytes();
    if date.len() == 10
        && date[4] == b'-'
        && date[7] == b'-'
        && date
            .iter()
            .enumerate()
            .all(|(i, b)| i == 4 || i == 7 || b.is_ascii_digit())
    {
        let year: i64 = spec[0..4].parse().ok()?;
        let month: u32 = spec[5..7].parse().ok()?;
        let day: u32 = spec[8..10].parse().ok()?;
        if !(1..=12).contains(&month) || !(1..=31).contains(&day) {
            return None;
        }
        let ms = (days_from_civil(year, month, day) as u64).checked_mul(86_400_000)?;
        return Some(ms.min(now_ms));
    }
    None
}

/// Filters applied to log events before aggregation.
#[derive(Debug, Default, Clone)]
pub struct UsageFilter {
    pub since_ms: Option<u64>,
    pub since_spec: Option<String>,
    pub repo_contains: Option<String>,
}

/// Aggregate the usage log into a report object (shape documented by the
/// `usage` verb). Missing log files produce a zeroed report, not an error —
/// "nothing logged yet" is a valid answer.
pub fn report_json(filter: &UsageFilter) -> Value {
    let path = usage_log_path();
    if usage_log_disabled() {
        return json!({
            "telemetry": "disabled",
            "methodology": METHODOLOGY,
            "hint": format!("set {USAGE_LOG_ENV} to a path to enable"),
        });
    }
    let Some(path) = path else {
        return json!({
            "telemetry": "unavailable",
            "reason": format!("no home directory and no {USAGE_LOG_ENV}"),
            "methodology": METHODOLOGY,
        });
    };

    let (parsed, skipped, events) = match std::fs::read_to_string(&path) {
        Ok(contents) => {
            let mut parsed = 0u64;
            let mut skipped = 0u64;
            let mut events = Vec::new();
            for line in contents.lines() {
                if line.trim().is_empty() {
                    continue;
                }
                match serde_json::from_str::<Value>(line) {
                    Ok(event) => {
                        parsed += 1;
                        events.push(event);
                    }
                    Err(_) => skipped += 1,
                }
            }
            (parsed, skipped, events)
        }
        Err(_) => (0, 0, Vec::new()),
    };

    let matches = |event: &Value| -> bool {
        if let Some(since) = filter.since_ms {
            if event.get("ts_ms").and_then(Value::as_u64).unwrap_or(0) < since {
                return false;
            }
        }
        if let Some(needle) = &filter.repo_contains {
            let repo = event.get("repo").and_then(Value::as_str).unwrap_or("");
            if !repo.contains(needle.as_str()) {
                return false;
            }
        }
        true
    };

    #[derive(Default)]
    struct Totals {
        searches: u64,
        cli: u64,
        mcp: u64,
        zero_result: u64,
        errors: u64,
        output_bytes: u64,
        hit_file_bytes: u64,
        top_hit_file_bytes: u64,
        duration_ms: u64,
        index_runs: u64,
        index_errors: u64,
        files_processed: u64,
        chunks_created: u64,
        index_duration_ms: u64,
    }
    #[derive(Default)]
    struct RepoTotals {
        searches: u64,
        zero_result: u64,
        output_bytes: u64,
        hit_file_bytes: u64,
    }

    let mut totals = Totals::default();
    let mut per_repo: BTreeMap<String, RepoTotals> = BTreeMap::new();
    let mut first_ms: Option<u64> = None;
    let mut last_ms: Option<u64> = None;
    let mut output_bytes_samples: Vec<u64> = Vec::new();
    let mut saved_all_samples: Vec<u64> = Vec::new();
    let mut matched = 0u64;

    for event in events.iter().filter(|event| matches(event)) {
        matched += 1;
        let ts = event.get("ts_ms").and_then(Value::as_u64);
        if let Some(ts) = ts {
            first_ms = Some(first_ms.map_or(ts, |first: u64| first.min(ts)));
            last_ms = Some(last_ms.map_or(ts, |last: u64| last.max(ts)));
        }
        let repo = event.get("repo").and_then(Value::as_str).unwrap_or("");
        match event.get("event").and_then(Value::as_str) {
            Some("search") => {
                totals.searches += 1;
                match event.get("mode").and_then(Value::as_str) {
                    Some("mcp") => totals.mcp += 1,
                    Some("cli") => totals.cli += 1,
                    _ => {}
                }
                let results = event.get("results").and_then(Value::as_u64).unwrap_or(0);
                let error = event.get("error").is_some_and(Value::is_string);
                if error {
                    totals.errors += 1;
                } else if results == 0 {
                    totals.zero_result += 1;
                }
                let output = event
                    .get("output_bytes")
                    .and_then(Value::as_u64)
                    .unwrap_or(0);
                let hit_bytes = event
                    .get("hit_file_bytes")
                    .and_then(Value::as_u64)
                    .unwrap_or(0);
                totals.output_bytes += output;
                totals.hit_file_bytes += hit_bytes;
                totals.top_hit_file_bytes += event
                    .get("top_hit_file_bytes")
                    .and_then(Value::as_u64)
                    .unwrap_or(0);
                totals.duration_ms += event
                    .get("duration_ms")
                    .and_then(Value::as_u64)
                    .unwrap_or(0);
                if !error {
                    output_bytes_samples.push(output);
                    saved_all_samples.push(hit_bytes.saturating_sub(output));
                }
                let entry = per_repo.entry(repo.to_string()).or_default();
                entry.searches += 1;
                if !error && results == 0 {
                    entry.zero_result += 1;
                }
                entry.output_bytes += output;
                entry.hit_file_bytes += hit_bytes;
            }
            Some("index") => {
                totals.index_runs += 1;
                if event.get("error").is_some_and(Value::is_string) {
                    totals.index_errors += 1;
                }
                totals.files_processed += event
                    .get("files_processed")
                    .and_then(Value::as_u64)
                    .unwrap_or(0);
                totals.chunks_created += event
                    .get("chunks_created")
                    .and_then(Value::as_u64)
                    .unwrap_or(0);
                totals.index_duration_ms += event
                    .get("duration_ms")
                    .and_then(Value::as_u64)
                    .unwrap_or(0);
            }
            _ => {}
        }
    }

    let tokens = |bytes: u64| bytes / 4;
    let output_tokens = tokens(totals.output_bytes);
    let baseline_all = tokens(totals.hit_file_bytes);
    let baseline_top = tokens(totals.top_hit_file_bytes);
    let mut per_repo_values: Vec<Value> = per_repo
        .into_iter()
        .map(|(repo, entry)| {
            json!({
                "repo": repo,
                "searches": entry.searches,
                "zero_result": entry.zero_result,
                "search_output_tokens": tokens(entry.output_bytes),
                "baseline_read_all_hit_files_tokens": tokens(entry.hit_file_bytes),
                "saved_tokens_vs_read_all": tokens(entry.hit_file_bytes.saturating_sub(entry.output_bytes)),
            })
        })
        .collect();
    per_repo_values.sort_by_key(|entry| {
        (
            std::cmp::Reverse(entry.get("searches").and_then(Value::as_u64).unwrap_or(0)),
            entry
                .get("repo")
                .and_then(Value::as_str)
                .unwrap_or("")
                .to_string(),
        )
    });

    json!({
        "log_path": path.display().to_string(),
        "log_exists": path.exists(),
        "events": {
            "parsed": parsed,
            "skipped_malformed": skipped,
            "matched": matched,
        },
        "window": {
            "since": filter.since_spec.clone(),
            "since_epoch_ms": filter.since_ms,
            "first_event": first_ms.map(iso_utc_from_ms),
            "last_event": last_ms.map(iso_utc_from_ms),
        },
        "repo_filter": filter.repo_contains.clone(),
        "searches": {
            "total": totals.searches,
            "cli": totals.cli,
            "mcp": totals.mcp,
            "zero_result": totals.zero_result,
            "errors": totals.errors,
            "duration_ms_total": totals.duration_ms,
            "output_bytes_total": totals.output_bytes,
            "output_bytes_median": percentile(&mut output_bytes_samples, 0.50),
            "output_bytes_p90": percentile(&mut output_bytes_samples, 0.90),
        },
        "index_runs": {
            "total": totals.index_runs,
            "errors": totals.index_errors,
            "files_processed": totals.files_processed,
            "chunks_created": totals.chunks_created,
            "duration_ms_total": totals.index_duration_ms,
        },
        "token_estimate": {
            "unit": "bytes/4",
            "search_output_tokens": output_tokens,
            "baseline_read_all_hit_files_tokens": baseline_all,
            "baseline_read_top_hit_file_tokens": baseline_top,
            "saved_tokens_vs_read_all": baseline_all.saturating_sub(output_tokens),
            "saved_tokens_vs_read_top": baseline_top.saturating_sub(output_tokens),
            "median_saved_tokens_vs_read_all_per_search": percentile(&mut saved_all_samples, 0.50).map(tokens),
            "reduction_factor_vs_read_all": if output_tokens == 0 {
                Value::Null
            } else {
                json!((baseline_all as f64 / output_tokens as f64 * 10.0).round() / 10.0)
            },
        },
        "per_repo": per_repo_values,
        "methodology": METHODOLOGY,
    })
}

/// Nearest-rank-ish percentile of a sample set; consumes and sorts it.
fn percentile(values: &mut [u64], fraction: f64) -> Option<u64> {
    if values.is_empty() {
        return None;
    }
    values.sort_unstable();
    let index = ((values.len() - 1) as f64 * fraction).round() as usize;
    values.get(index).copied()
}

/// Render the JSON report for humans: aligned sections, ~k/~M token
/// magnitudes, methodology footnote.
pub fn render_human(report: &Value) -> String {
    if report.get("telemetry").is_some() {
        let state = report
            .get("telemetry")
            .and_then(Value::as_str)
            .unwrap_or("");
        let hint = report.get("hint").and_then(Value::as_str).unwrap_or("");
        return format!("sindexer usage — telemetry {state}. {hint}\n");
    }
    let mut out = String::new();
    let log_path = report
        .get("log_path")
        .and_then(Value::as_str)
        .unwrap_or("?");
    let log_exists = report
        .get("log_exists")
        .and_then(Value::as_bool)
        .unwrap_or(false);
    out.push_str(&format!(
        "sindexer usage — {}{}\n",
        log_path,
        if log_exists {
            ""
        } else {
            " (no log yet — nothing recorded)"
        }
    ));
    if let Some(window) = report.get("window") {
        let first = window.get("first_event").and_then(Value::as_str);
        let last = window.get("last_event").and_then(Value::as_str);
        if first.is_some() || last.is_some() {
            out.push_str(&format!(
                "window   {} → {}{}\n",
                first.unwrap_or("?"),
                last.unwrap_or("now"),
                window
                    .get("since")
                    .and_then(Value::as_str)
                    .map(|since| format!(" (since {since})"))
                    .unwrap_or_default()
            ));
        }
    }
    if let Some(events) = report.get("events") {
        out.push_str(&format!(
            "events   {} parsed, {} matched, {} malformed\n",
            events.get("parsed").and_then(Value::as_u64).unwrap_or(0),
            events.get("matched").and_then(Value::as_u64).unwrap_or(0),
            events
                .get("skipped_malformed")
                .and_then(Value::as_u64)
                .unwrap_or(0),
        ));
    }
    if let Some(searches) = report.get("searches") {
        let get = |key: &str| searches.get(key).and_then(Value::as_u64).unwrap_or(0);
        out.push_str(&format!(
            "searches {} (cli {}, mcp {})   zero-result {}   errors {}\n",
            get("total"),
            get("cli"),
            get("mcp"),
            get("zero_result"),
            get("errors"),
        ));
        if get("total") > 0 {
            let median = searches.get("output_bytes_median").and_then(Value::as_u64);
            let p90 = searches.get("output_bytes_p90").and_then(Value::as_u64);
            out.push_str(&format!(
                "         output bytes: total {}   median {}   p90 {}\n",
                fmt_count(get("output_bytes_total")),
                median.map(fmt_count).unwrap_or_else(|| "-".into()),
                p90.map(fmt_count).unwrap_or_else(|| "-".into()),
            ));
        }
    }
    if let Some(estimate) = report.get("token_estimate") {
        let get = |key: &str| estimate.get(key).and_then(Value::as_u64).unwrap_or(0);
        let factor = estimate
            .get("reduction_factor_vs_read_all")
            .and_then(Value::as_f64);
        out.push_str("\ntoken estimate (bytes/4 ≈ tokens)\n");
        out.push_str(&format!(
            "  search output entered context     {}\n",
            fmt_tokens(get("search_output_tokens"))
        ));
        out.push_str(&format!(
            "  baseline: read all hit files      {}\n",
            fmt_tokens(get("baseline_read_all_hit_files_tokens"))
        ));
        out.push_str(&format!(
            "  baseline: read top hit file       {}\n",
            fmt_tokens(get("baseline_read_top_hit_file_tokens"))
        ));
        out.push_str(&format!(
            "  saved vs read-all                 {}{}\n",
            fmt_tokens(get("saved_tokens_vs_read_all")),
            factor
                .map(|f| format!("  ({f:.1}x reduction)"))
                .unwrap_or_default()
        ));
        out.push_str(&format!(
            "  saved vs read-top                 {}\n",
            fmt_tokens(get("saved_tokens_vs_read_top"))
        ));
        if let Some(median) = estimate
            .get("median_saved_tokens_vs_read_all_per_search")
            .and_then(Value::as_u64)
        {
            out.push_str(&format!(
                "  median saved per search           {}\n",
                fmt_tokens(median)
            ));
        }
    }
    if let Some(repos) = report.get("per_repo").and_then(Value::as_array) {
        if !repos.is_empty() {
            out.push_str("\nper repo\n");
            let shown = repos.len().min(15);
            for entry in &repos[..shown] {
                out.push_str(&format!(
                    "  {:<44} {:>5} searches   {:>9} out   {:>12} saved\n",
                    truncate(entry.get("repo").and_then(Value::as_str).unwrap_or("?"), 44),
                    entry.get("searches").and_then(Value::as_u64).unwrap_or(0),
                    fmt_tokens(
                        entry
                            .get("search_output_tokens")
                            .and_then(Value::as_u64)
                            .unwrap_or(0)
                    ),
                    fmt_tokens(
                        entry
                            .get("saved_tokens_vs_read_all")
                            .and_then(Value::as_u64)
                            .unwrap_or(0)
                    ),
                ));
            }
            if repos.len() > shown {
                out.push_str(&format!("  … and {} more\n", repos.len() - shown));
            }
        }
    }
    if let Some(index_runs) = report.get("index_runs") {
        let get = |key: &str| index_runs.get(key).and_then(Value::as_u64).unwrap_or(0);
        if get("total") > 0 {
            out.push_str(&format!(
                "\nindex runs {} (errors {})   {} files   {} chunks   {} total\n",
                get("total"),
                get("errors"),
                fmt_count(get("files_processed")),
                fmt_count(get("chunks_created")),
                fmt_duration(get("duration_ms_total")),
            ));
        }
    }
    if let Some(methodology) = report.get("methodology").and_then(Value::as_str) {
        out.push_str(&format!("\nmethodology: {methodology}\n"));
    }
    out
}

fn truncate(text: &str, width: usize) -> String {
    if text.chars().count() <= width {
        text.to_string()
    } else {
        let mut cut: String = text.chars().take(width - 1).collect();
        cut.push('…');
        cut
    }
}

fn fmt_count(value: u64) -> String {
    let thousands = value.to_string();
    let mut grouped = String::new();
    for (index, digit) in thousands.chars().enumerate() {
        if index > 0 && (thousands.len() - index).is_multiple_of(3) {
            grouped.push(',');
        }
        grouped.push(digit);
    }
    grouped
}

fn fmt_tokens(tokens: u64) -> String {
    let sign = "~";
    if tokens >= 1_000_000 {
        format!("{sign}{:.1}M", tokens as f64 / 1_000_000.0)
    } else if tokens >= 1_000 {
        format!("{sign}{:.1}k", tokens as f64 / 1_000.0)
    } else {
        format!("{sign}{tokens}")
    }
}

fn fmt_duration(ms: u64) -> String {
    let secs = ms / 1000;
    if secs >= 3600 {
        format!("{}h{:02}m", secs / 3600, (secs % 3600) / 60)
    } else if secs >= 60 {
        format!("{}m{:02}s", secs / 60, secs % 60)
    } else {
        format!("{secs}s")
    }
}

#[cfg(test)]
pub(crate) mod test_support {
    /// Serializes tests that mutate process environment variables.
    pub static ENV_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use std::path::PathBuf;
    use std::sync::MutexGuard;

    use crate::mcp::hybrid::HybridHit;
    use crate::types::CodeChunk;

    struct EnvVarGuard {
        key: &'static str,
        _lock: MutexGuard<'static, ()>,
    }

    impl EnvVarGuard {
        fn set(key: &'static str, value: &str) -> Self {
            let lock = test_support::ENV_LOCK
                .lock()
                .unwrap_or_else(|poison| poison.into_inner());
            std::env::set_var(key, value);
            Self { key, _lock: lock }
        }
    }

    impl Drop for EnvVarGuard {
        fn drop(&mut self) {
            std::env::remove_var(self.key);
        }
    }

    fn chunk(relative_path: &str, content: &str, file_path: Option<PathBuf>) -> HybridHit {
        HybridHit {
            chunk: CodeChunk {
                id: format!("id-{relative_path}"),
                content: content.to_string(),
                file_path: file_path.unwrap_or_else(|| PathBuf::from(relative_path)),
                relative_path: relative_path.to_string(),
                start_line: 1,
                end_line: content.lines().count() as u32,
                language: "rust".to_string(),
            },
            score: 0.5,
        }
    }

    #[test]
    fn test_iso_utc_known_anchors() {
        assert_eq!(iso_utc_from_ms(0), "1970-01-01T00:00:00Z");
        assert_eq!(iso_utc_from_ms(1_789_000_000_000), "2026-09-10T00:26:40Z");
        assert_eq!(iso_utc_from_ms(1_758_672_000_000), "2025-09-24T00:00:00Z");
        assert_eq!(iso_utc_from_ms(4_102_444_800_000), "2100-01-01T00:00:00Z");
    }

    #[test]
    fn test_civil_days_roundtrip() {
        for days in -20_000i64..=20_000 {
            let (year, month, day) = civil_from_days(days);
            assert_eq!(days_from_civil(year, month, day), days, "at days={days}");
        }
    }

    #[test]
    fn test_parse_since_relative_windows() {
        let now = 1_000_000_000_000u64;
        assert_eq!(parse_since("0h", now), Some(now));
        assert_eq!(parse_since("2h", now), Some(now - 7_200_000));
        assert_eq!(parse_since("7d", now), Some(now - 604_800_000));
        assert_eq!(parse_since("1w", now), Some(now - 604_800_000));
        assert_eq!(parse_since("1y", now), Some(now - 31_536_000_000));
        // Windows older than the reference clamp to zero, never underflow.
        assert_eq!(parse_since("100y", now), Some(0));
        assert_eq!(parse_since("999999999999w", now), Some(0));
    }

    #[test]
    fn test_parse_since_epoch_and_date() {
        let now = 2_000_000_000_000u64;
        assert_eq!(parse_since("1789000000", now), Some(1_789_000_000_000));
        assert_eq!(parse_since("1789000000000", now), Some(1_789_000_000_000));
        assert_eq!(parse_since("9999-12-31", now), Some(now)); // future clamps
        let expected = (days_from_civil(2026, 9, 1) as u64) * 86_400_000;
        assert_eq!(parse_since("2026-09-01", now), Some(expected));
        assert_eq!(parse_since("nonsense", now), None);
        assert_eq!(parse_since("2026-13-01", now), None);
        assert_eq!(parse_since("2026-09-32", now), None);
        assert_eq!(parse_since("-3d", now), None);
    }

    #[test]
    fn test_measure_hits_dedupes_and_sums() {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path();
        fs::write(root.join("a.rs"), "x".repeat(100)).unwrap();
        fs::write(root.join("b.rs"), "y".repeat(40)).unwrap();

        let a = PathBuf::from(root.join("a.rs"));
        let b = PathBuf::from(root.join("b.rs"));
        let hits = vec![
            chunk("src/a.rs", "aaaa", Some(a.clone())),
            chunk("src/a.rs", "bbbb", Some(a.clone())),
            chunk("b.rs", "cc", Some(b.clone())),
            chunk("gone.rs", "dddd", Some(root.join("nope.rs"))),
        ];
        let metrics = measure_hits(root, &hits);
        assert_eq!(metrics.results, 4);
        assert_eq!(metrics.excerpt_chars, 14);
        assert_eq!(metrics.hit_files, 3); // a.rs, b.rs, gone.rs (size 0)
        assert_eq!(metrics.hit_file_bytes, 140);
        assert_eq!(metrics.top_hit_file_bytes, 100);
    }

    #[test]
    fn test_measure_hits_falls_back_to_root_join() {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path();
        fs::write(root.join("lib.rs"), "z".repeat(10)).unwrap();
        // file_path empty (lexical hits before fixup) → resolved via root.
        let hits = vec![chunk("lib.rs", "code", None)];
        assert_eq!(
            measure_hits(root, &hits),
            HitMetrics {
                results: 1,
                excerpt_chars: 4,
                hit_files: 1,
                hit_file_bytes: 10,
                top_hit_file_bytes: 10,
            }
        );
    }

    #[test]
    fn test_log_search_writes_jsonl_and_reports() {
        let dir = tempfile::tempdir().unwrap();
        let log_path = dir.path().join("usage.jsonl");
        let _guard = EnvVarGuard::set(USAGE_LOG_ENV, log_path.to_str().unwrap());

        log_search(SearchLog {
            mode: "cli",
            repo: Path::new("/repo/alpha"),
            query: "find the thing",
            limit: 5,
            duration_ms: 42,
            metrics: HitMetrics {
                results: 3,
                excerpt_chars: 300,
                hit_files: 2,
                hit_file_bytes: 8_000,
                top_hit_file_bytes: 4_000,
            },
            output_bytes: 400,
            error: None,
        });
        log_search(SearchLog {
            mode: "mcp",
            repo: Path::new("/repo/beta"),
            query: "miss",
            limit: 5,
            duration_ms: 10,
            metrics: HitMetrics::default(),
            output_bytes: 60,
            error: None,
        });
        log_search(SearchLog {
            mode: "cli",
            repo: Path::new("/repo/alpha"),
            query: "boom",
            limit: 5,
            duration_ms: 7,
            metrics: HitMetrics::default(),
            output_bytes: 0,
            error: Some("collection missing"),
        });
        log_index(IndexLog {
            mode: "cli",
            verb: "update",
            repo: Path::new("/repo/alpha"),
            files_processed: 12,
            chunks_created: 34,
            duration_ms: 500,
            lexical_only: false,
            error: None,
        });

        let contents = fs::read_to_string(&log_path).unwrap();
        let lines: Vec<&str> = contents.lines().collect();
        assert_eq!(lines.len(), 4);
        for line in &lines {
            assert!(serde_json::from_str::<Value>(line).is_ok(), "line: {line}");
        }

        let filter = UsageFilter::default();
        let report = report_json(&filter);
        assert_eq!(report["events"]["parsed"], 4);
        assert_eq!(report["searches"]["total"], 3);
        assert_eq!(report["searches"]["cli"], 2);
        assert_eq!(report["searches"]["mcp"], 1);
        assert_eq!(report["searches"]["zero_result"], 1);
        assert_eq!(report["searches"]["errors"], 1);
        assert_eq!(report["token_estimate"]["search_output_tokens"], 115); // (400+60+0)/4
        assert_eq!(
            report["token_estimate"]["baseline_read_all_hit_files_tokens"],
            2_000
        );
        assert_eq!(
            report["token_estimate"]["baseline_read_top_hit_file_tokens"],
            1_000
        );
        assert_eq!(report["index_runs"]["total"], 1);
        assert_eq!(report["index_runs"]["files_processed"], 12);

        // Repo filter narrows to alpha only.
        let filtered = report_json(&UsageFilter {
            repo_contains: Some("alpha".into()),
            ..UsageFilter::default()
        });
        assert_eq!(filtered["searches"]["total"], 2);
        assert_eq!(filtered["token_estimate"]["search_output_tokens"], 100); // (400+0)/4

        // Since filter in the future matches nothing.
        let future = report_json(&UsageFilter {
            since_ms: Some(u64::MAX),
            ..UsageFilter::default()
        });
        assert_eq!(future["searches"]["total"], 0);

        let human = render_human(&report);
        assert!(human.contains("searches 3 (cli 2, mcp 1)"));
        assert!(human.contains("methodology:"));
    }

    #[test]
    fn test_log_can_be_disabled() {
        let _guard = EnvVarGuard::set(USAGE_LOG_ENV, "0");
        assert!(usage_log_path().is_none());
        let report = report_json(&UsageFilter::default());
        assert_eq!(report["telemetry"], "disabled");
        // Logging is a no-op: nothing panics, no file materializes.
        log_search(SearchLog {
            mode: "cli",
            repo: Path::new("/repo"),
            query: "q",
            limit: 1,
            duration_ms: 0,
            metrics: HitMetrics::default(),
            output_bytes: 0,
            error: None,
        });
        assert!(!Path::new("/repo").join("usage.jsonl").exists());
    }

    #[test]
    fn test_percentile_handles_empty_and_small_sets() {
        let mut empty: Vec<u64> = Vec::new();
        assert_eq!(percentile(&mut empty, 0.5), None);
        let mut single = vec![7u64];
        assert_eq!(percentile(&mut single, 0.5), Some(7));
        let mut values = vec![100u64, 10, 50, 20, 90];
        assert_eq!(percentile(&mut values, 0.5), Some(50));
        assert_eq!(percentile(&mut values, 0.9), Some(100));
    }

    #[test]
    fn test_fmt_helpers() {
        assert_eq!(fmt_count(1_234_567), "1,234,567");
        assert_eq!(fmt_count(999), "999");
        assert_eq!(fmt_tokens(999), "~999");
        assert_eq!(fmt_tokens(1_500), "~1.5k");
        assert_eq!(fmt_tokens(2_500_000), "~2.5M");
        assert_eq!(fmt_duration(3_725_000), "1h02m");
        assert_eq!(fmt_duration(65_000), "1m05s");
        assert_eq!(fmt_duration(59_000), "59s");
        assert_eq!(truncate("abcdefghij", 5), "abcd…");
        assert_eq!(truncate("abc", 5), "abc");
    }
}
