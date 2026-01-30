from __future__ import annotations

import argparse
import json
import threading
import time
import urllib.request

import uvicorn

from serve.app.config import load_config


def _post(url: str, payload: dict) -> dict:
    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=5) as response:
        return json.loads(response.read().decode("utf-8"))


def _get(url: str) -> dict:
    with urllib.request.urlopen(url, timeout=5) as response:
        return json.loads(response.read().decode("utf-8"))


def _wait_ready(base_url: str, timeout_s: float) -> None:
    start = time.time()
    while time.time() - start < timeout_s:
        try:
            _get(f"{base_url}/v1/models")
            return
        except Exception:
            time.sleep(0.1)
    raise RuntimeError("Server did not become ready in time")


def _start_server(host: str, port: int) -> tuple[uvicorn.Server, threading.Thread]:
    config = uvicorn.Config(
        "serve.app.main:createApp",
        host=host,
        port=port,
        log_level="warning",
        factory=True,
    )
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    return server, thread


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a local smoke test for the vAGI API.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8010)
    parser.add_argument("--timeout", type=float, default=5.0)
    args = parser.parse_args()

    server, thread = _start_server(args.host, args.port)
    base_url = f"http://{args.host}:{args.port}"

    try:
        _wait_ready(base_url, args.timeout)
        cfg = load_config()
        obs = [[0.0 for _ in range(cfg.obs_dim)]]

        init = _post(f"{base_url}/v1/core/init", {"batchSize": 1})
        state_id = init["stateId"]

        step = _post(
            f"{base_url}/v1/core/step",
            {"stateId": state_id, "inputIds": [[0]], "obs": obs},
        )
        plan = _post(
            f"{base_url}/v1/core/plan",
            {"stateId": state_id, "inputIds": [[0]], "obs": obs, "numCandidates": 4, "horizon": 2},
        )

        output = {"init": init, "step": step, "plan": plan}
        print(json.dumps(output, indent=2))
    finally:
        server.should_exit = True
        thread.join(timeout=5)


if __name__ == "__main__":
    main()
