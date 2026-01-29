# AGENTS

## Project quickstart
- Install: `python -m pip install -e .`
- Tests: `pytest` (or `python -m pytest -q`)

## Code quality policy
- Use existing tooling only. No new linters/formatters are required.
- Ruff/Black/Mypy are not configured in `pyproject.toml`, so do not introduce them.

## PR review rubric (strict output format)
1) Summary (3–6 bullets)
2) High-risk issues (security / correctness / data-loss)
3) Performance/regression notes
4) Test gaps & suggested tests
5) Recommended action: APPROVE / REQUEST_CHANGES / COMMENT_ONLY + confidence score 0–100

Be specific and reference filenames + line ranges when possible.

## Repo-specific constraints
- Core structure: `core/`, `runtime/`, `io/`, `scripts/`, `tests/`.
- Do not propose changes that require hidden dependencies; if needed, update
  `pyproject.toml` and `requirements-lock.txt` coherently and explain why.
