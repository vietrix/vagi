# vAGI

vAGI is a compact, single-checkpoint causal transformer core with a recurrent state and multi-head outputs for text, actions, value, and optional world prediction.

## Quickstart

```bash
python -m pip install -e .
pytest
```

## Pseudo usage (illustrative, not executable)

```text
cfg = VAGIConfig(...)
model = VAGICore(cfg)

state = model.init_state(batch_size=B, device="cpu")

out = model.step(input_ids=token_ids_t, obs=obs_t, state=state)
text_logits = out["text_logits"]
action_logits = out["action_logits"]
value = out["value"]
state = out["state"]
```

## Reproducibility

```bash
python -m scripts.run_all_benchmarks --deterministic
python -m scripts.bench_latency --steps 50
python -m scripts.tune_inference --batch-sizes 1,2,4 --kv-cache-len 16
```

## Privacy

Logs and rollouts are scrubbed for common PII patterns by default. See
`docs/PRIVACY.md` for opt-in, retention, and delete controls.
