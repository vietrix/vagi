use std::net::SocketAddr;
use std::path::PathBuf;
use std::process::{Child, Command};
use std::sync::Arc;
use std::time::Duration;

use anyhow::{Context, Result, bail};
use tokio::net::TcpListener;
use tracing_subscriber::EnvFilter;
use vagi_kernel::KernelContext;
use vagi_kernel::routes::build_router;

#[derive(Debug, Default)]
struct CliArgs {
    web_ui: bool,
}

fn print_usage() {
    println!("Usage: vagi-kernel [--web-ui]");
    println!("  --web-ui   Start official Open WebUI server at http://127.0.0.1:17071");
    println!("             Requires `open-webui` command installed from upstream project.");
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

fn spawn_open_webui(host: &str, port: u16, kernel_port: u16) -> Result<Child> {
    let mut candidates: Vec<(String, Vec<String>)> = Vec::new();
    if let Ok(cmd) = std::env::var("VAGI_OPEN_WEBUI_CMD") {
        if !cmd.trim().is_empty() {
            candidates.push((cmd, Vec::new()));
        }
    }
    candidates.push(("open-webui".to_string(), Vec::new()));
    candidates.push((
        "python".to_string(),
        vec!["-m".to_string(), "open_webui".to_string()],
    ));
    candidates.push((
        "python3".to_string(),
        vec!["-m".to_string(), "open_webui".to_string()],
    ));

    let mut errors = Vec::new();

    let default_openai_base_url = format!("http://127.0.0.1:{kernel_port}/v1");
    let openai_base_url = std::env::var("VAGI_OPEN_WEBUI_OPENAI_BASE_URL")
        .ok()
        .filter(|v| !v.trim().is_empty())
        .unwrap_or(default_openai_base_url);
    let openai_api_key = std::env::var("VAGI_OPEN_WEBUI_OPENAI_API_KEY")
        .ok()
        .filter(|v| !v.trim().is_empty())
        .unwrap_or_else(|| "sk-vagi-local".to_string());
    let disable_persistent_config =
        std::env::var("VAGI_OPEN_WEBUI_DISABLE_PERSISTENT_CONFIG")
            .map(|v| v != "0")
            .unwrap_or(true);

    for (program, prefix_args) in candidates {
        let mut command = Command::new(&program);
        for arg in &prefix_args {
            command.arg(arg);
        }
        command
            .arg("serve")
            .arg("--host")
            .arg(host)
            .arg("--port")
            .arg(port.to_string());

        command.env("ENABLE_OPENAI_API", "True");
        command.env("OPENAI_API_BASE_URL", &openai_base_url);
        command.env("OPENAI_API_BASE_URLS", &openai_base_url);
        command.env("OPENAI_API_KEY", &openai_api_key);
        command.env("OPENAI_API_KEYS", &openai_api_key);
        if disable_persistent_config {
            command.env("ENABLE_PERSISTENT_CONFIG", "False");
        }

        match command.spawn() {
            Ok(mut child) => {
                std::thread::sleep(Duration::from_millis(700));
                match child.try_wait() {
                    Ok(Some(status)) => {
                        errors.push(format!(
                            "{program}: exited immediately with status {status}"
                        ));
                    }
                    Ok(None) => return Ok(child),
                    Err(err) => errors.push(format!("{program}: failed to inspect status: {err}")),
                }
            }
            Err(err) => errors.push(format!("{program}: {err}")),
        }
    }

    bail!(
        "failed to spawn Open WebUI upstream process.\n\
         Tried commands:\n{}\n\
         Install from upstream repo: https://github.com/open-webui/open-webui",
        errors.join("\n")
    )
}

fn stop_child_process(child: &mut Child) {
    let _ = child.kill();
    let _ = child.wait();
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
        let mut open_webui_child = spawn_open_webui(&web_ui_host, web_ui_port, port)?;
        tracing::info!(
            "Open WebUI (official upstream) started at http://{}:{}",
            web_ui_host,
            web_ui_port
        );
        tracing::info!(
            "Open WebUI OpenAI provider -> http://127.0.0.1:{}/v1 (override with VAGI_OPEN_WEBUI_OPENAI_BASE_URL)",
            port
        );

        let api_server = axum::serve(api_listener, api_router);
        tracing::info!("vAGI kernel API listening at http://{api_addr}");

        tokio::select! {
            api_result = api_server => {
                stop_child_process(&mut open_webui_child);
                api_result.context("kernel API server terminated unexpectedly")?;
            }
            signal_result = tokio::signal::ctrl_c() => {
                if let Err(err) = signal_result {
                    tracing::warn!("failed to listen for ctrl-c: {err}");
                } else {
                    tracing::info!("ctrl-c received, shutting down");
                }
                stop_child_process(&mut open_webui_child);
            }
        }
    } else {
        tracing::info!("vAGI kernel API listening at http://{api_addr}");
        axum::serve(api_listener, api_router)
            .await
            .context("kernel API server terminated unexpectedly")?;
    }

    Ok(())
}
