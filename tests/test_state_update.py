import torch

from core.base import VAGIConfig, VAGICore


def test_state_updates_memory() -> None:
    torch.manual_seed(0)
    cfg = VAGIConfig(
        vocab_size=128,
        hidden_size=64,
        n_layers=2,
        n_heads=4,
        n_kv_heads=4,
        mlp_ratio=2.0,
        max_seq_len=16,
        obs_dim=16,
        obs_tokens=2,
        action_dim=8,
        memory_slots=4,
        dropout=0.0,
        use_world_pred=False,
    )
    model = VAGICore(cfg)
    bsz = 2
    obs = torch.randn(bsz, cfg.obs_dim)
    token = torch.randint(0, cfg.vocab_size, (bsz, 1), dtype=torch.long)

    state = model.init_state(bsz)
    mem_before = state.mem.clone()
    out = model.step(input_ids=token, obs=obs, state=state)
    mem_after = out["state"].mem

    assert not torch.allclose(mem_after, mem_before)
    assert mem_after.abs().sum().item() > 0.0
