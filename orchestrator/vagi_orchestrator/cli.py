from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Annotated

import httpx
import typer

from .genesis.pipeline import GenesisConfig, train_and_export
from .memory import MemoryClient

app = typer.Typer(help="vAGI CLI")
genesis_app = typer.Typer(help="Genesis run commands")
memory_app = typer.Typer(help="Vector memory commands")
app.add_typer(genesis_app, name="genesis")
app.add_typer(memory_app, name="memory")


def _api_url(url: str | None) -> str:
    if url:
        return url.rstrip("/")
    return "http://127.0.0.1:8080"


def _kernel_url(url: str | None) -> str:
    if url:
        return url.rstrip("/")
    return "http://127.0.0.1:7070"


def _split_paragraphs(text: str) -> list[str]:
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    chunks = [chunk.strip() for chunk in re.split(r"\n\s*\n+", normalized)]
    return [chunk for chunk in chunks if chunk]


@app.command()
def chat(
    prompt: str,
    model: Annotated[str, typer.Option("--model")] = "vagi-v1",
    session_id: Annotated[str | None, typer.Option("--session-id")] = None,
    api_url: Annotated[str | None, typer.Option("--api-url")] = None,
) -> None:
    url = _api_url(api_url)
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "session_id": session_id,
    }
    response = httpx.post(f"{url}/v1/chat/completions", json=payload, timeout=60)
    response.raise_for_status()
    body = response.json()
    typer.echo(body["choices"][0]["message"]["content"])
    policy = body.get("metadata", {}).get("policy", {})
    if policy:
        typer.echo(
            f"policy_status={policy.get('status')} verifier_pass={policy.get('verifier_pass')}"
        )


@app.command("scan")
def scan_code(
    path: str,
    api_url: Annotated[str | None, typer.Option("--api-url")] = None,
) -> None:
    url = _api_url(api_url)
    response = httpx.post(f"{url}/v1/agents/scan-code", json={"path": path}, timeout=120)
    response.raise_for_status()
    body = response.json()
    typer.echo(f"Scanned files: {body['scanned_files']}")
    typer.echo(f"Issues: {len(body['issues'])}")
    typer.echo(json.dumps(body["issues"][:10], ensure_ascii=False, indent=2))


@app.command("dream")
def dream_run(
    source: Annotated[str, typer.Option("--source")] = "manual",
    api_url: Annotated[str | None, typer.Option("--api-url")] = None,
) -> None:
    url = _api_url(api_url)
    response = httpx.post(
        f"{url}/v1/evolution/run-dream",
        json={"source": source},
        timeout=120,
    )
    response.raise_for_status()
    typer.echo(json.dumps(response.json(), ensure_ascii=False, indent=2))


@app.command()
def benchmark(
    n: Annotated[int, typer.Option("--n")] = 10,
    api_url: Annotated[str | None, typer.Option("--api-url")] = None,
) -> None:
    url = _api_url(api_url)
    latencies: list[float] = []
    for idx in range(1, n + 1):
        payload = {
            "model": "vagi-v1",
            "messages": [{"role": "user", "content": f"benchmark run #{idx}"}],
            "stream": False,
        }
        started = time.perf_counter()
        response = httpx.post(f"{url}/v1/chat/completions", json=payload, timeout=60)
        elapsed = (time.perf_counter() - started) * 1000
        response.raise_for_status()
        latencies.append(elapsed)
    p50 = sorted(latencies)[len(latencies) // 2]
    typer.echo(
        f"benchmark runs={n} p50={p50:.2f}ms avg={(sum(latencies)/len(latencies)):.2f}ms"
    )


@memory_app.command("ingest")
def memory_ingest(
    file_path: str,
    kernel_url: Annotated[str | None, typer.Option("--kernel-url")] = None,
) -> None:
    path = Path(file_path)
    if not path.exists():
        raise typer.BadParameter(f"file not found: {path}")
    if not path.is_file():
        raise typer.BadParameter(f"path is not a file: {path}")

    text = path.read_text(encoding="utf-8")
    paragraphs = _split_paragraphs(text)
    if not paragraphs:
        raise typer.BadParameter("file does not contain non-empty paragraphs")

    client = MemoryClient(kernel_url=_kernel_url(kernel_url))
    success = 0
    failed = 0
    total = len(paragraphs)
    for index, paragraph in enumerate(paragraphs, start=1):
        try:
            if client.add_document(paragraph):
                success += 1
                typer.echo(f"[{index}/{total}] ingested")
            else:
                failed += 1
                typer.echo(f"[{index}/{total}] failed: missing document id", err=True)
        except Exception as exc:
            failed += 1
            typer.echo(f"[{index}/{total}] failed: {exc}", err=True)

    typer.echo(f"ingest_total={total} success={success} failed={failed}")


@memory_app.command("query")
def memory_query(
    question: str,
    top_k: Annotated[int, typer.Option("--top-k")] = 3,
    kernel_url: Annotated[str | None, typer.Option("--kernel-url")] = None,
) -> None:
    client = MemoryClient(kernel_url=_kernel_url(kernel_url))
    results = client.search(question, top_k=top_k)
    if not results:
        typer.echo("No memory hits.")
        return
    for idx, item in enumerate(results, start=1):
        typer.echo(f"{idx}. {item}")


@genesis_app.command("train")
def genesis_train(
    output_dir: Annotated[str, typer.Option("--output-dir")] = "runtime/models/genesis-v0",
    repeats: Annotated[int, typer.Option("--repeats")] = 28,
    epochs: Annotated[int, typer.Option("--epochs")] = 12,
    embed_dim: Annotated[int, typer.Option("--embed-dim")] = 128,
    hidden_dim: Annotated[int, typer.Option("--hidden-dim")] = 256,
    num_layers: Annotated[int, typer.Option("--num-layers")] = 2,
    seq_len: Annotated[int, typer.Option("--seq-len")] = 64,
    batch_size: Annotated[int, typer.Option("--batch-size")] = 32,
    lr: Annotated[float, typer.Option("--lr")] = 3e-3,
) -> None:
    result = train_and_export(
        output_dir=Path(output_dir),
        config=GenesisConfig(
            repeats=repeats,
            epochs=epochs,
            embed_dim=embed_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            seq_len=seq_len,
            batch_size=batch_size,
            lr=lr,
        ),
    )
    typer.echo(json.dumps(result, ensure_ascii=False, indent=2))


@genesis_app.command("load")
def genesis_load(
    model_dir: Annotated[str, typer.Option("--model-dir")] = "runtime/models/genesis-v0",
    kernel_url: Annotated[str | None, typer.Option("--kernel-url")] = None,
) -> None:
    url = _kernel_url(kernel_url)
    response = httpx.post(
        f"{url}/internal/model/load",
        json={"model_dir": model_dir},
        timeout=120,
    )
    response.raise_for_status()
    typer.echo(json.dumps(response.json(), ensure_ascii=False, indent=2))


@genesis_app.command("infer")
def genesis_infer(
    prompt: Annotated[str, typer.Option("--prompt")] = "User: Xin chao\nAssistant:",
    max_new_tokens: Annotated[int, typer.Option("--max-new-tokens")] = 96,
    kernel_url: Annotated[str | None, typer.Option("--kernel-url")] = None,
) -> None:
    url = _kernel_url(kernel_url)
    response = httpx.post(
        f"{url}/internal/infer",
        json={"prompt": prompt, "max_new_tokens": max_new_tokens},
        timeout=120,
    )
    response.raise_for_status()
    typer.echo(json.dumps(response.json(), ensure_ascii=False, indent=2))


@genesis_app.command("run")
def genesis_run(
    model_dir: Annotated[str, typer.Option("--model-dir")] = "runtime/models/genesis-v0",
    kernel_url: Annotated[str | None, typer.Option("--kernel-url")] = None,
    api_url: Annotated[str | None, typer.Option("--api-url")] = None,
) -> None:
    output_path = Path(model_dir)
    train_result = train_and_export(output_dir=output_path, config=GenesisConfig())

    kernel = _kernel_url(kernel_url)
    api = _api_url(api_url)
    load_response = httpx.post(
        f"{kernel}/internal/model/load",
        json={"model_dir": str(output_path)},
        timeout=120,
    )
    load_response.raise_for_status()
    infer_response = httpx.post(
        f"{kernel}/internal/infer",
        json={"prompt": "User: Xin chao\nAssistant:", "max_new_tokens": 80},
        timeout=120,
    )
    infer_response.raise_for_status()
    chat_response = httpx.post(
        f"{api}/v1/chat/completions",
        json={
            "model": "vagi-v1",
            "messages": [{"role": "user", "content": "Xin chao, toi can toi uu API chat."}],
            "stream": False,
        },
        timeout=120,
    )
    chat_response.raise_for_status()
    payload = {
        "train": {
            "model_dir": train_result["model_dir"],
            "best_val_loss": train_result["best_val_loss"],
        },
        "kernel_load": load_response.json(),
        "kernel_infer": infer_response.json(),
        "chat": {
            "status_code": chat_response.status_code,
            "policy": chat_response.json().get("metadata", {}).get("policy", {}),
            "model_runtime": chat_response.json().get("metadata", {}).get("model_runtime", {}),
        },
    }
    typer.echo(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    app()
