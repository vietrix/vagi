use std::net::SocketAddr;
use std::path::PathBuf;
use std::sync::Arc;

use anyhow::Result;
use tokio::net::TcpListener;
use tracing_subscriber::EnvFilter;
use vagi_kernel::KernelContext;
use vagi_kernel::routes::build_router;

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::from_default_env())
        .init();

    let snapshot_db = std::env::var("VAGI_SNAPSHOT_DB")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from("runtime/snapshots.redb"));

    let ctx = Arc::new(KernelContext::new(&snapshot_db)?);
    let router = build_router(ctx);

    let host = std::env::var("VAGI_KERNEL_HOST").unwrap_or_else(|_| "127.0.0.1".to_string());
    let port: u16 = std::env::var("VAGI_KERNEL_PORT")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(7070);
    let addr: SocketAddr = format!("{host}:{port}").parse()?;

    let listener = TcpListener::bind(addr).await?;
    tracing::info!("vAGI kernel listening at http://{addr}");
    axum::serve(listener, router).await?;
    Ok(())
}

