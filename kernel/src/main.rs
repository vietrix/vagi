use std::net::SocketAddr;
use std::path::PathBuf;
use std::sync::Arc;

use anyhow::{Context, Result, bail};
use tokio::net::TcpListener;
use tracing_subscriber::EnvFilter;
use vagi_kernel::KernelContext;
use vagi_kernel::routes::build_router;
use vagi_kernel::web_ui::build_web_ui_router;

#[derive(Debug, Default)]
struct CliArgs {
    web_ui: bool,
}

fn print_usage() {
    println!("Usage: vagi-kernel [--web-ui]");
    println!("  --web-ui   Start an additional OpenWebUI-like frontend at http://127.0.0.1:17071");
}

fn parse_args() -> Result<CliArgs> {
    let mut args = CliArgs::default();
    for arg in std::env::args().skip(1) {
        match arg.as_str() {
            "--web-ui" => args.web_ui = true,
            "--help" | "-h" => {
                print_usage();
                std::process::exit(0);
            }
            unknown => bail!("unknown argument `{unknown}`. Use --help"),
        }
    }
    Ok(args)
}

#[tokio::main]
async fn main() -> Result<()> {
    let cli_args = parse_args()?;

    tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::from_default_env())
        .init();

    let snapshot_db = std::env::var("VAGI_SNAPSHOT_DB")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from("runtime/snapshots.redb"));
    let memory_db = std::env::var("VAGI_MEMORY_DB")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from("runtime/memory.redb"));

    let ctx = Arc::new(KernelContext::new(&snapshot_db, &memory_db)?);
    let api_router = build_router(Arc::clone(&ctx));

    let host = std::env::var("VAGI_KERNEL_HOST").unwrap_or_else(|_| "127.0.0.1".to_string());
    let port: u16 = std::env::var("VAGI_KERNEL_PORT")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(17070);
    let api_addr: SocketAddr = format!("{host}:{port}").parse()?;
    let api_listener = TcpListener::bind(api_addr)
        .await
        .with_context(|| format!("failed to bind kernel API on {api_addr}"))?;

    if cli_args.web_ui {
        let web_ui_host =
            std::env::var("VAGI_WEB_UI_HOST").unwrap_or_else(|_| "127.0.0.1".to_string());
        let web_ui_port: u16 = std::env::var("VAGI_WEB_UI_PORT")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(17071);
        let web_ui_addr: SocketAddr = format!("{web_ui_host}:{web_ui_port}").parse()?;
        let web_ui_listener = TcpListener::bind(web_ui_addr)
            .await
            .with_context(|| format!("failed to bind Web UI on {web_ui_addr}"))?;

        let web_ui_router = build_web_ui_router(Arc::clone(&ctx), PathBuf::from("runtime/web-ui"))
            .context("failed to build Web UI router")?;

        let api_server = async move {
            tracing::info!("vAGI kernel API listening at http://{api_addr}");
            axum::serve(api_listener, api_router)
                .await
                .context("kernel API server terminated unexpectedly")
        };

        let ui_server = async move {
            tracing::info!("vAGI Web UI listening at http://{web_ui_addr}");
            axum::serve(web_ui_listener, web_ui_router)
                .await
                .context("Web UI server terminated unexpectedly")
        };

        tokio::try_join!(api_server, ui_server)?;
    } else {
        tracing::info!("vAGI kernel API listening at http://{api_addr}");
        axum::serve(api_listener, api_router)
            .await
            .context("kernel API server terminated unexpectedly")?;
    }

    Ok(())
}
