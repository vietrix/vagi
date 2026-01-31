# vAGI

vAGI is a compact causal transformer core with recurrent state for agent-style models.
It produces text logits, action logits (policy), value estimates, and optional world
predictions in a single forward pass.

## Ownership
- Organization: Vietrix
- Contact: zyntherdev7878@gmail.com
- Domain: TBD

## Architecture (diagrams)

High level data flow:

```
tokens --> token_embed --+--> +pos --> Transformer blocks --> hidden states
                         |
obs ----> obs_tokenizer --+--> (optional) special tokens <OBS>/<ACT>/<VAL>

Outputs:
  text_logits  (LM head on sequence)
  action_logits (policy head on h_act or h_last)
  value        (value head on h_act or h_last)
  world_pred   (world head on h_last, optional)
  state        (memory + KV cache)
```

Heads overview:

```
hidden states
  |--> LanguageHead  --> text_logits
  |--> PolicyHead    --> action_logits
  |--> ValueHead     --> value
  |--> WorldHead     --> world_pred (optional)
```

Distillation + QAT pipeline:

```
Teacher -> logits/values/world/trace/uncertainty
       \-> distill losses (KL + MSE)
Student -> optional QAT phase (int8/bf16 simulation)
```

## vAGI-lite default

The current vAGI-lite preset is stored in `core/vagi_lite.json` and loaded via
`load_vagi_lite_config()`.

```json
{
  "vocab_size": 128,
  "hidden_size": 32,
  "n_layers": 1,
  "n_heads": 2,
  "n_kv_heads": 2,
  "mlp_ratio": 2.0,
  "max_seq_len": 16,
  "obs_dim": 16,
  "obs_tokens": 2,
  "action_dim": 8,
  "memory_slots": 4,
  "dropout": 0.0,
  "use_rotary": false,
  "use_gqa": false,
  "use_flash_attn": false,
  "use_world_pred": true,
  "use_special_tokens": true
}
```

## Results and benchmarks

Latest sweep report: 2026-01-31 12:46:09

Benchmark setup:
- Device: CPU
- Environment: ToyEnv
- Eval: 5 episodes, 16 steps
- Latency: 50 steps, 5 warmup steps

Sweep table (params vs latency, pass_rate, memory):

| label | params | latency_ms | pass_rate | memory_mb | pareto | default |
| --- | ---: | ---: | ---: | ---: | :---: | :---: |
| h32_l1_h2 | 20477 | 1.615 | 0.062 | 0.078 | YES | YES |
| h32_l1_h4 | 20477 | 1.810 | 0.062 | 0.078 |  |  |
| h32_l2_h2 | 28893 | 2.815 | 0.188 | 0.110 |  |  |
| h32_l2_h4 | 28893 | 3.171 | 0.188 | 0.110 |  |  |
| h32_l3_h2 | 37309 | 3.887 | 0.062 | 0.142 |  |  |
| h32_l3_h4 | 37309 | 3.638 | 0.062 | 0.142 |  |  |
| h48_l1_h2 | 37613 | 1.773 | 0.188 | 0.143 |  |  |
| h48_l1_h4 | 37613 | 1.771 | 0.188 | 0.143 |  |  |
| h48_l2_h2 | 56381 | 2.607 | 0.062 | 0.215 |  |  |
| h48_l2_h4 | 56381 | 3.001 | 0.062 | 0.215 |  |  |
| h48_l3_h2 | 75149 | 3.866 | 0.062 | 0.287 |  |  |
| h48_l3_h4 | 75149 | 3.432 | 0.062 | 0.287 |  |  |
| h64_l1_h2 | 59357 | 1.793 | 0.188 | 0.226 |  |  |
| h64_l1_h4 | 59357 | 1.750 | 0.188 | 0.226 | YES |  |
| h64_l2_h2 | 92573 | 2.743 | 0.188 | 0.353 |  |  |
| h64_l2_h4 | 92573 | 2.673 | 0.188 | 0.353 |  |  |
| h64_l3_h2 | 125789 | 4.359 | 0.062 | 0.480 |  |  |
| h64_l3_h4 | 125789 | 3.815 | 0.062 | 0.480 |  |  |

Notes:
- This sweep used a small synthetic dataset and a short teacher warmup, so pass_rate
  is intentionally low. Use a real teacher checkpoint for meaningful results.

## Docs
- Quickstart: `docs/QUICKSTART.md`
- Install: `docs/install.md`
- Use: `docs/use.md`
- Train + distill: `docs/train.md`
- Benchmarks: `docs/benchmarks.md`
- Reproducibility: `docs/REPRODUCIBILITY.md`
- Evaluation: `docs/EVALUATION.md`
- Export: `docs/EXPORT.md`
- Privacy: `docs/PRIVACY.md`
- AI review: `docs/CI_CODEX.md`
- Config: `docs/config.md`
- Architecture: `docs/architecture.md`
- API: `docs/api.md`
- Data: `docs/data.md`
- Model card: `docs/MODEL_CARD.md`

## Policies
- Code of Conduct: `CODE_OF_CONDUCT.md`
- Security: `SECURITY.md`
- Contributing: `CONTRIBUTING.md`
- License: `LICENSE`
