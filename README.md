# vAGI V1 (Greenfield)

Triển khai mới 100% cho kiến trúc vAGI theo 4 tầng:

1. **Substrate (Rust Kernel)**: State Space Memory + Linear Processing + Snapshot store
2. **Cognitive Engine**: Fast intuition, world simulation, logic verifier (WASI)
3. **Reasoning Loop**: OODA điều phối bởi orchestrator
4. **Evolution**: Sleep & Dream batch, trust score, promotion gate

## Cấu trúc

- `kernel/`: service Rust nội bộ (`/internal/*`)
- `orchestrator/`: API ngoài + CLI + scheduler
- `contracts/`: schema interface
- `docs/`: ADR + runbook
- `ops/`: profile chạy local

## Chạy kernel

```bash
cargo run -p vagi-kernel
```

Mặc định chạy tại `http://127.0.0.1:7070`.

## Chạy orchestrator

```bash
cd orchestrator
pip install -e .[dev]
uvicorn vagi_orchestrator.app:app --host 127.0.0.1 --port 8080
```

API ngoài:

- `POST /v1/chat/completions`
- `POST /v1/agents/scan-code`
- `POST /v1/evolution/run-dream`
- `GET /v1/healthz`
- `GET /v1/metrics`

## Test

```bash
cargo test -p vagi-kernel
cd orchestrator && pytest
```

