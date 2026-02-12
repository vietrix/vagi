# vAGI Orchestrator

Service điều phối cho vAGI V1:

- OODA reasoning loop (`observe -> orient -> decide -> act`)
- Runtime policy gateway (hard-enforce OODA + verifier gate)
- OpenAI-compatible HTTP API (`/v1/chat/completions`)
- Agent scan code endpoint (`/v1/agents/scan-code`)
- Evolution batch (`/v1/evolution/run-dream`)
- CLI (`vagi chat|scan|dream|benchmark`)

## Chạy local

```bash
pip install -e .[dev]
uvicorn vagi_orchestrator.app:app --host 127.0.0.1 --port 8080
```

## Biến môi trường chính

- `VAGI_KERNEL_URL` (default `http://127.0.0.1:7070`)
- `VAGI_ORCH_HOST` (default `127.0.0.1`)
- `VAGI_ORCH_PORT` (default `8080`)
- `VAGI_RUNTIME_DIR` (default `runtime`)
- `VAGI_DREAM_HOUR` (default `2`)
- `VAGI_DREAM_MINUTE` (default `0`)
