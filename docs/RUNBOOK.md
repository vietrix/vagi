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
  -d "{\"model\":\"vagi-v1\",\"messages\":[{\"role\":\"user\",\"content\":\"viết login an toàn\"}]}"
```

## 4) Trigger dream batch

```bash
curl -X POST http://127.0.0.1:8080/v1/evolution/run-dream \
  -H "Content-Type: application/json" \
  -d "{\"source\":\"manual\"}"
```

## 5) Failure handling

- Nếu `/v1/healthz` trả `degraded`: kiểm tra kernel URL (`VAGI_KERNEL_URL`) và trạng thái service kernel.
- Nếu verifier fail tăng cao: kiểm tra payload `violations` và thắt chặt generation guardrails.
- Nếu không promote được trong dream: kiểm tra `pass_rate` và `regression_fail` trong báo cáo run-dream.

