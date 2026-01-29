# Compatibility Contract

This document defines what is considered stable and how changes are versioned.

## Schema version (JSONL)

- Current `schema_version` is **1**.
- `schema_version: 0` is supported for backward compatibility and normalized to `1`.
- Any future schema change must provide:
  - A migration note
  - A version bump
  - A clear backward-compatibility story

## Checkpoint format

Checkpoints are **safetensors + metadata**:

```
model.safetensors
meta.json
optimizer.pt   (optional)
```

The loader must continue to accept:

- A directory containing `model.safetensors`
- A single `.safetensors` file (with or without `meta.json`)

## Semver rules

- **Patch**: fixes or improvements without changing public contracts.
- **Minor**: new optional fields, new scripts, new docs.
- **Major**: breaking changes to data formats, checkpoints, or core APIs.

Breaking changes must include:

- `docs/COMPATIBILITY.md` update
- Migration notes
- Tests updated to enforce the new behavior
