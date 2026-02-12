from __future__ import annotations

import json
import time
from typing import Annotated

import httpx
import typer

app = typer.Typer(help="vAGI CLI")


def _api_url(url: str | None) -> str:
    if url:
        return url.rstrip("/")
    return "http://127.0.0.1:8080"


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


if __name__ == "__main__":
    app()
