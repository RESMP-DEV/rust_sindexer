use anyhow::Result;
use rmcp::transport::io::stdio;
use rmcp::ServiceExt;
use sindexer::cli;
use sindexer::config::Config;
use sindexer::mcp::create_shared_state;
use sindexer::mcp::CodebaseTools;
use tracing_subscriber::{fmt, prelude::*, EnvFilter};

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match args.first().map(String::as_str) {
        Some("-h" | "--help") => {
            print!("{}", cli::HELP);
            use std::io::Write;
            std::io::stdout().flush()?;
            return Ok(());
        }
        Some("-V" | "--version") => {
            println!("sindexer {}", env!("CARGO_PKG_VERSION"));
            return Ok(());
        }
        Some(first) if cli::is_verb(first) => {
            // Env preparation happens before any runtime worker threads exist,
            // so the (rare) env mutation in prepare_environment cannot race.
            cli::prepare_environment(first);
            tracing_subscriber::registry()
                .with(fmt::layer().with_writer(std::io::stderr))
                .with(EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("warn")))
                .init();
            let code = {
                let runtime = tokio::runtime::Runtime::new()?;
                runtime.block_on(cli::run(&args))?
            };
            std::process::exit(code);
        }
        Some(other) => {
            eprintln!("sindexer: unknown command '{other}'");
            eprintln!(
                "Run 'sindexer --help' for usage (verbs) or pass no arguments for MCP server mode."
            );
            std::process::exit(2);
        }
        None => {}
    }

    let runtime = tokio::runtime::Runtime::new()?;
    runtime.block_on(serve())
}

async fn serve() -> Result<()> {
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
