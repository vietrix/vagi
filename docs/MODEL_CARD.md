# MODEL CARD

## Model summary
vAGI is a compact causal transformer core with optional observation tokens and
recurrent memory state. It exposes language, policy, value, and optional world
prediction heads for agent-style experimentation.

## Architecture
- Decoder-only transformer with pre-norm blocks.
- Optional observation tokenizer prepends `K` obs tokens.
- Fast memory slots updated via gated writes.
- KV cache is stubbed for future extension.

## Intended use
- Research prototypes and toy environments.
- Offline supervised training and behavioral cloning.
- Deterministic benchmarking on small CPU tasks.

## Training data
- Minimal examples (sample text and synthetic rollouts).
- No web or external data sources are required.

## Evaluation
- CodeEnv regression benchmarks (20 tasks).
- Metrics: pass rate, average steps, run_tests count, wall time.

## Limitations
- Not optimized for large-scale training or deployment.
- KV cache and rotary attention are simplified or stubbed.
- Not tuned for real-world code agents.

## Privacy
- Logs and rollouts are scrubbed for common PII patterns.
- Opt-in controls and retention/delete utilities are available in `runtime/privacy.py`.

## License
- Apache-2.0 with vAGI Attribution (see `LICENSE`).
