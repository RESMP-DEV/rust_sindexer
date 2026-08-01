use anyhow::Result;
use rmcp::transport::io::stdio;
use rmcp::ServiceExt;
use sindexer::config::Config;
use sindexer::mcp::create_shared_state;
use sindexer::mcp::CodebaseTools;
use tracing_subscriber::{fmt, prelude::*, EnvFilter};

const HELP: &str = r#"sindexer — semantic code indexing MCP server

USAGE:
    sindexer [OPTIONS]

sindexer is a Model Context Protocol (MCP) server. It speaks newline-delimited
JSON-RPC over stdin/stdout and is meant to be launched by an MCP client
(Claude Code, Claude Desktop, Codex, Copilot, ...), not driven by hand.
Run it directly and it will wait, silently, for MCP requests on stdin.

With no environment variables the server works out of the box in
lexical-only mode. Set EMBEDDING_URL to enable semantic search, and
MILVUS_URL to store vectors in Milvus/Zilliz instead of the local store.

OPTIONS:
    -h, --help       Print this help message and exit
    -V, --version    Print version information and exit

EXAMPLES:
    # Register with Claude Code (user scope)
    claude mcp add code-indexer --scope user -- /path/to/sindexer

    # Register with an inline MCP config (~/.claude.json, .mcp.json, ...)
    {
      "mcpServers": {
        "code-indexer": { "command": "/path/to/sindexer" }
      }
    }

    # End-to-end smoke test without an MCP client
    ./examples/demo.sh /path/to/sindexer

See README.md for the full configuration and usage reference.
"#;

#[tokio::main]
async fn main() -> Result<()> {
    match std::env::args().nth(1).as_deref() {
        Some("-h" | "--help") => {
            print!("{HELP}");
            return Ok(());
        }
        Some("-V" | "--version") => {
            println!("sindexer {}", env!("CARGO_PKG_VERSION"));
            return Ok(());
        }
        _ => {}
    }

    tracing_subscriber::registry()
        .with(fmt::layer().with_writer(std::io::stderr))
        .with(EnvFilter::from_default_env().add_directive(tracing::Level::INFO.into()))
        .init();

    tracing::info!("Starting sindexer MCP server");

    let config = Config::from_env();
    let tools = CodebaseTools::with_state(create_shared_state(config));
    let service = tools.serve(stdio()).await?;

    tracing::info!("MCP server initialized, waiting for requests");

    match service.waiting().await {
        Ok(reason) => tracing::info!(?reason, "Server stopped"),
        Err(e) => tracing::error!(?e, "Server task failed"),
    }

    Ok(())
}
