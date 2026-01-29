# PRIVACY

This project is designed to run locally without web access. When user-provided
data is involved, follow these privacy guardrails.

## Opt-in
- Logging and rollouts default to PII scrubbing.
- Use `--privacy-opt-in` to acknowledge retention of logs in local storage.

## Scrubbing
- JSONL logs are scrubbed for common PII patterns (emails, IPs, phone numbers,
  API-key-like strings).
- Scrubbing runs automatically via `runtime.privacy.scrub_record`.

## Retention and delete
- Use `--retain-days` to auto-delete logs older than N days.
- Use `--delete-logs` to remove JSONL/log files in the output directory.
- Utilities are implemented in `runtime/privacy.py`.

## User control
- Logs are plain text JSONL and can be deleted at any time.
- No telemetry or remote uploads are performed by default.
