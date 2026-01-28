# vAGI-core

vAGI-core is a compact, single-checkpoint causal transformer core with a recurrent state and multi-head outputs for text, actions, value, and optional world prediction.

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
