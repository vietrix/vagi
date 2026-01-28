# Use

## Forward
Call `VAGICore.forward` with `input_ids` and optional `obs` and `state`.

## Step
Use `VAGICore.step` to run timestep updates with recurrent memory.

## Toy environment + agent loop
Use the deterministic toy environment to exercise `step()`:

```bash
python -m scripts.agent_loop --steps 10 --save transitions.pt
```

This logs transitions (obs, actions, rewards) for offline training.
