# Contributing

## Ownership

This project is owned and maintained by Vietrix.
Contact: zyntherdev7878@gmail.com
Domain: TBD

Thanks for your interest in contributing to vAGI.

## Code of Conduct

By participating, you agree to abide by `CODE_OF_CONDUCT.md`.

## Ways to contribute
- Report bugs and regression issues
- Propose features or improvements
- Improve documentation and examples
- Add tests and benchmarks

## Development setup

```bash
python -m pip install -e .
python -m pytest
```

## Project structure
- `core/`: model backbone, heads, memory, losses
- `scripts/`: training and evaluation helpers
- `envs/`: toy environments
- `runtime/`: agent loop runtime + logging
- `docs/`: documentation
- `tests/`: pytest suite

## Style and quality
- Follow existing naming and formatting conventions in the repo.
- Prefer small, focused changes with clear intent.
- Add or update tests when behavior changes.
- Keep doc updates close to code changes.

## Commit messages

Use conventional prefixes:
- feat:, fix:, refactor:, docs:, test:, chore:, ci:

## Pull requests

Checklist:
- [ ] Changes are scoped and focused
- [ ] Tests added or updated
- [ ] Documentation updated if behavior changed
- [ ] `python -m pytest` passes

## Issues

When filing issues, include:
- Expected vs actual behavior
- Steps to reproduce
- Logs or stack traces
- Environment (OS, Python, PyTorch)

## Security

Please do not report security issues via public issues. See `SECURITY.md`.

## CLA

The CLA in `CLA.md` is required. A bot will request confirmation.
