# Core API Stability

This document defines the stable vAGI core interface and how we version changes.

## Stable interfaces

The following methods are considered stable and should remain backward compatible:

- `VAGICore.init_state(batch_size, device=None, *, prefill_kv=False, kv_max_seq_len=None)`
- `VAGICore.forward(input_ids, obs=None, state=None, labels=None, targets=None, return_loss=False, return_hidden=False)`
- `VAGICore.step(input_ids, obs, state)`
- `VAGICore.plan_step(input_ids, obs, state, num_candidates=4, horizon=3, uncertainty_weight=1.0, strategy="cem", cem_iters=3, elite_frac=0.2, tree_branching=4)`

### Output contracts

`forward` returns a dict with (at minimum):

- `text_logits` `(B, S, V)`
- `action_logits` `(B, A)`
- `value` `(B, 1)`
- `world_pred` `(B, K, O)` or `None`
- `state` (RecurrentState or `None`)

`step` returns the same keys, plus any uncertainty outputs:

- `value_logvar` (optional)
- `world_logvar` (optional)

If `use_world_pred` is disabled, `world_pred` is `None`.

If `use_uncertainty` is disabled, `value_logvar` and `world_logvar` are `None`.

If `return_hidden=True`, `forward` includes:

- `hidden` `(B, D)` — last-token representation for auxiliary losses

## Semantic versioning rules

We follow semver:

- **Patch**: bug fixes, performance improvements, no API changes.
- **Minor**: additive changes (new optional fields, new scripts, new docs).
- **Major**: breaking changes to the stable interfaces or output contracts.

Any change that:

- removes a method, or
- changes required parameters, or
- changes the meaning or shape of required outputs

requires a **major** version bump and a clear migration note.
