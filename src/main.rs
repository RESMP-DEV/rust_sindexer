use anyhow::Result;
use rust_sindexer::config::Config;
use rust_sindexer::mcp::create_shared_state;
use rust_sindexer::mcp::CodebaseTools;
use rust_sindexer::transport::StdioTransport;
use rmcp::ServiceExt;
use tracing_subscriber::{fmt, prelude::*, EnvFilter};

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::registry()
        .with(fmt::layer().with_writer(std::io::stderr))
        .with(EnvFilter::from_default_env().add_directive(tracing::Level::INFO.into()))
        .init();

    tracing::info!("Starting rust_sindexer MCP server");

    let config = Config::from_env();
    let tools = CodebaseTools::with_state(create_shared_state(config));
    let transport = StdioTransport::new(tokio::io::stdin(), tokio::io::stdout());

    let service = tools.serve(transport).await?;

    tracing::info!("MCP server initialized, waiting for requests");

    match service.waiting().await {
        Ok(reason) => tracing::info!(?reason, "Server stopped"),
        Err(e) => tracing::error!(?e, "Server task failed"),
    }

    Ok(())
}
