# Core Loop Modes

vAGI provides a unified core loop with two operating modes:

## `act(...)` — fast policy-only

Use when latency or compute budget is tight. It runs a single `step` and returns the
greedy policy action.

## `think_then_act(...)` — plan + reflection + info-seeking

Use when accuracy matters. It performs:

1. Budget decision (optional)
2. Reflection heads (error type + info gain)
3. Planning (`plan_step`) with risk-aware uncertainty penalties
4. Action selection

Both modes use the same model weights. The difference is compute budget, not model size.

## Compute budget controller

If enabled (`use_budget_head=True`), the model predicts:

- `mode`: `act` vs `think`
- `horizon`: planning depth
- `num_candidates`: search breadth

This allows the model to adaptively allocate compute per task.
