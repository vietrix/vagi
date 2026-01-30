# Upgrade Guide

This guide documents how to upgrade between releases when **schema** or **checkpoint**
formats change.

## When to update

Update this guide when any of the following changes:

- JSONL `schema_version` in `utils/data/schema.py`
- Checkpoint layout in `io/checkpoint.py`
- Core API signature changes in `docs/CORE_API.md`
- Export metadata format in `scripts/export_utils.py`

## Checklist for breaking changes

1. **Bump semver** in `pyproject.toml`.
2. **Document the change** in `CHANGELOG.md`.
3. **Update** `exports/manifest.json` with new `schema_version` or `config_hash`.
4. **Add migration notes** here (what to re-export, re-train, or re-process).

## Common migrations

### Schema version bump

If `schema_version` changes:

- Re-run data generation scripts (`scripts/collect_*`) to emit the new schema.
- Update any downstream parsers to match new fields.

### Checkpoint format change

If checkpoint metadata or layout changes:

- Re-export checkpoints with `io/checkpoint.py`.
- Update any deployment pipelines consuming `.safetensors` or `meta.json`.

### Export metadata changes

If `scripts/export_utils.py` changes:

- Re-run `scripts/export_onnx.py` and `scripts/quantize_onnx.py`.
- Rebuild `exports/manifest.json` and re-attach assets to releases.
