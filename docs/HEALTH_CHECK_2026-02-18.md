# Health Check Report - vAGI V1

Date: 2026-02-18
Scope: Runability-first quick health check (repo architecture, setup, test posture, API/contract fit).

## Executive Summary

- Project architecture is clear and coherent: Rust `kernel` + Python `orchestrator` + JSON schema `contracts`.
- Main runability blocker is dependency coupling in orchestrator CLI/genesis path, causing heavy ML deps (`torch`) to be required for non-genesis workflows.
- Core orchestrator tests (excluding genesis/CLI-genesis import path) pass locally after installing minimal dependencies.
- Kernel execution and tests are currently blocked in this environment because Rust toolchain (`cargo`, `rustc`) is not installed.

## What Works

- API shape and policy gate flow in orchestrator are implemented and test-backed.
- `/v1/chat/completions` applies precheck and postcheck policy gates, persists policy audit data, and returns OpenAI-style response object.
- Response contract for non-stream chat aligns with pydantic response models and schema expectations (`choices[].finish_reason`, metadata structure).

## Verified Evidence

- CLI imports genesis pipeline at module load:
  - `orchestrator/vagi_orchestrator/cli.py:12`
- Base orchestrator deps include `sentence-transformers`:
  - `orchestrator/pyproject.toml:16`
- `genesis` extra includes torch:
  - `orchestrator/pyproject.toml:25`
- Policy gate in chat endpoint:
  - `orchestrator/vagi_orchestrator/app.py:129`
  - `orchestrator/vagi_orchestrator/app.py:151`
- Response model has `finish_reason` and metadata object:
  - `orchestrator/vagi_orchestrator/models.py:25`
  - `orchestrator/vagi_orchestrator/models.py:37`
- Response schema requires the same fields:
  - `contracts/schemas/chat-completions-response.schema.json:15`
  - `contracts/schemas/chat-completions-response.schema.json:44`

## Runtime/Test Results

- Initial `pytest` failed due to missing deps (`fastapi`, `typer`, local package import).
- After installing minimal deps + editable package without full ML stack:
  - Passed: `tests/test_app.py tests/test_policy.py tests/test_reasoning.py tests/test_store.py tests/test_memory.py tests/test_dream.py tests/test_scanner.py`
  - Still blocked: `tests/test_cli_memory.py`, `tests/test_genesis_pipeline.py` (because importing CLI/genesis requires `torch`).
- Kernel side could not be executed here because `cargo` is unavailable in current machine image.

## Risk Assessment (Runability First)

1. High: Non-genesis commands are coupled to genesis ML stack.
- Impact: onboarding/setup cost, CI fragility, and test collection failure unless `torch` exists.

2. High: Kernel testability depends on external toolchain not guaranteed by repo bootstrap.
- Impact: no deterministic end-to-end validation in fresh environments.

3. Medium: Heavy dependency in base install (`sentence-transformers`) may be unnecessary for default path.
- Impact: long setup time, larger attack surface, slower CI.

## Prioritized Action List

1. Decouple CLI import path from genesis training stack.
- Change: move genesis imports into `genesis_*` command functions or behind lazy loader.
- Success criteria: `pytest` can collect/run non-genesis tests without `torch` installed.

2. Re-scope orchestrator dependencies.
- Change: move `sentence-transformers` and/or torch-requiring pieces to optional extras tied to memory/genesis features.
- Success criteria: `pip install -e .[dev]` for API-only workflow does not pull GPU-heavy packages.

3. Split test matrix.
- Change: add explicit test groups: `core`, `memory`, `genesis`.
- Success criteria: core CI path runs fast and green without ML extras; genesis path runs with extras in dedicated job.

4. Add deterministic environment bootstrap docs/scripts.
- Change: include separate setup commands for `core` vs `genesis` and explicit Rust toolchain requirement.
- Success criteria: fresh machine can run target test subset with one documented command sequence.

5. Add minimum smoke script for end-to-end contract check.
- Change: script to validate `/v1/healthz`, `/v1/chat/completions`, and 422 policy response shape.
- Success criteria: one command produces pass/fail summary and blocks merges on regression.

## Defaults/Assumptions Used

- Priority chosen: runability before deeper optimization.
- No code changes applied to business logic in this health check.
- Findings reflect repository state and machine environment on 2026-02-18.
