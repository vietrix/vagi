# CI CODEX

This repository integrates OpenAI Codex for PR review automation with safe controls.

## Triggering a review
- Add label `ai-review` (alias: `codex-review`)
  **or**
- Comment `/ai-review` (alias: `Codex Review`)

The workflow exits early for draft PRs.

## Autofix (optional)
- Add label `ai-autofix` to a **same-repo** PR (not from a fork).
- The workflow runs tests, invokes Codex for minimal fixes, and opens a new PR
  back to the original PR branch if tests pass.

## Security model
- Review workflow is read-only (`contents: read`) and never uses `pull_request_target`.
- Autofix workflow runs only on same-repo PRs and writes to a new branch.
- Secrets are never printed.

## Quota tips
- Use labels or the `/ai-review` comment to avoid running on every push.
- The review input includes only PR metadata, changed file list, and unified diffs.
