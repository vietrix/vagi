# Use

## Forward
Call `VAGICore.forward` with `input_ids` and optional `obs` and `state`.
Set `return_features=True` to retrieve hidden and trace tensors for distillation.

## Step
Use `VAGICore.step` to run timestep updates with recurrent memory.

## vAGI-lite preset
```text
from vagi_core import load_vagi_lite_config, VAGICore
cfg = load_vagi_lite_config()
model = VAGICore(cfg)
```

## Toy environment + agent loop
Use the deterministic toy environment to exercise `step()`:

```bash
python -m runtime.agent_loop --steps 10 --log runs/agent/transitions.jsonl
```

This logs transitions (obs, actions, rewards) for offline training.
