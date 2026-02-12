# Runbook vAGI V1

## 1) Start kernel

```bash
cargo run -p vagi-kernel
```

Health check:

```bash
curl http://127.0.0.1:7070/healthz
```

## 2) Start orchestrator

```bash
cd orchestrator
pip install -e .[dev]
uvicorn vagi_orchestrator.app:app --host 127.0.0.1 --port 8080
```

Health check:

```bash
curl http://127.0.0.1:8080/v1/healthz
```

## 3) Smoke chat

```bash
curl -X POST http://127.0.0.1:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d "{\"model\":\"vagi-v1\",\"messages\":[{\"role\":\"user\",\"content\":\"write secure login\"}]}"
```

## 4) Trigger dream batch

```bash
curl -X POST http://127.0.0.1:8080/v1/evolution/run-dream \
  -H "Content-Type: application/json" \
  -d "{\"source\":\"manual\"}"
```

## 5) Failure handling

- If `/v1/healthz` is `degraded`: check `VAGI_KERNEL_URL` and kernel process status.
- If `POST /v1/chat/completions` returns `422`:
  - inspect `error.code` and `error.details[]`
  - verify OODA stages and verifier gate in audit DB (`episodes`)
  - tighten generation guards if repeated policy failures appear
- If verifier failures increase: inspect `violations` and adjust reasoner draft rules.
- If dream promotion is blocked: inspect `pass_rate` and `regression_fail` from run report.

