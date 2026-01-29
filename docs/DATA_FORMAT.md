# vAGI Offline Data Format (JSONL)

This repo uses a strict JSONL schema for offline rollouts to ensure reproducible training.

## Schema version

All records must include:

- `schema_version`: integer (current = `1`)

## Required fields (per line)

Each line is a single JSON object with the following required keys:

- `schema_version` (int)
- `episode_id` (str or int)
- `timestep` (int, >= 0)
- `obs` (list[float], non-empty)
- `action` (int)
- `reward` (float)
- `done` (bool)

## Optional fields

- `obs_next` (list[float]) — must match `obs` length
- `return` (float) — if present, used as precomputed return
- `value` (float) — value estimate at this step
- `task` (str) — task identifier
- `success` (bool) — optional outcome flag
- `info` (object) — arbitrary metadata (must be a JSON object)

## Ordering requirements

The streaming reader assumes:

1. Records are ordered by `episode_id`.
2. `timestep` increases within each episode.
3. Each episode ends with `done: true`.

## Example (single line)

```json
{"schema_version":1,"episode_id":"ep-0001","timestep":0,"obs":[0.1,0.2],"action":2,"reward":1.0,"done":false}
```

## Packing outputs

The offline packer (`utils/data/pack.py`) produces tensors:

- `obs`: `(B, O)`
- `obs_future`: `(B, K, O)` where `K = horizon`
- `actions`: `(B,)`
- `returns`: `(B, 1)`
- `rewards`: `(B, 1)`
- `mask`: `(B, K)` (1.0 for valid future steps, 0.0 for padding)
