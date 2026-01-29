# Use

## Forward
Call `VAGICore.forward` with `input_ids` and optional `obs` and `state`.

## Step
Use `VAGICore.step` to run timestep updates with recurrent memory.

## Toy environment + agent loop
Use the deterministic toy environment to exercise `step()`:

```bash
python -m runtime.agent_loop --steps 10 --log runs/agent/transitions.jsonl
```

This logs transitions (obs, actions, rewards) for offline training.
