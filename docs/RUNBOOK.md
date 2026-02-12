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

## 3.1) Smoke JIT (Hyper-Kernel Assembler prototype)

```bash
curl -X POST http://127.0.0.1:7070/internal/jit/execute \
  -H "Content-Type: application/json" \
  -d "{\"input\":7,\"logic\":\"add 5\nmul 2\nxor 3\"}"
```

## 3.2) Smoke HAM (Tuần 2 - Holographic Templates)

Upsert template:

```bash
curl -X POST http://127.0.0.1:7070/internal/hdc/templates/upsert \
  -H "Content-Type: application/json" \
  -d "{\"template_id\":\"python_secure_v1\",\"logic_template\":\"add {{delta}}\nmul 2\nxor {{mask}}\",\"tags\":[\"python\",\"secure\"]}"
```

Query template:

```bash
curl -X POST http://127.0.0.1:7070/internal/hdc/templates/query \
  -H "Content-Type: application/json" \
  -d "{\"query\":\"need secure python patch\",\"top_k\":3}"
```

Bind template:

```bash
curl -X POST http://127.0.0.1:7070/internal/hdc/templates/bind \
  -H "Content-Type: application/json" \
  -d "{\"template_id\":\"python_secure_v1\",\"bindings\":{\"delta\":\"5\",\"mask\":\"3\"}}"
```

Weave + execute (query -> bind -> JIT):

```bash
curl -X POST http://127.0.0.1:7070/internal/hdc/weave/execute \
  -H "Content-Type: application/json" \
  -d "{\"query\":\"secure python patch\",\"input\":9,\"bindings\":{\"delta\":\"5\",\"mask\":\"3\"}}"
```

Weaver plan (query -> multi-candidate -> score -> select):

```bash
curl -X POST http://127.0.0.1:7070/internal/hdc/weave/plan \
  -H "Content-Type: application/json" \
  -d "{\"query\":\"secure python patch\",\"input\":9,\"top_k\":3,\"risk_threshold\":0.65,\"verifier_required\":true,\"bindings\":{\"delta\":\"5\",\"mask\":\"3\"}}"
```

Evolution mutate (genetic trainer):

```bash
curl -X POST http://127.0.0.1:7070/internal/hdc/evolution/mutate \
  -H "Content-Type: application/json" \
  -d "{\"query\":\"secure python patch\",\"generations\":3,\"population_size\":8,\"survivors\":2,\"risk_threshold\":0.65,\"seed_input\":13,\"promote\":true}"
```

Enable orchestrator Weaver mode (hybrid):

```bash
export VAGI_REASONER_MODE=hybrid
export VAGI_WEAVER_TOP_K=3
export VAGI_MUTATION_ENABLED=true
export VAGI_MUTATION_GENERATIONS=3
export VAGI_MUTATION_POPULATION=8
export VAGI_MUTATION_SURVIVORS=2
uvicorn vagi_orchestrator.app:app --host 127.0.0.1 --port 8080
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
