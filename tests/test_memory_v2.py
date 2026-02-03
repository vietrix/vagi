import torch

from core.base import VAGIConfig, VAGICore


def test_memory_v2_determinism_and_norm() -> None:
    torch.manual_seed(0)
    cfg = VAGIConfig(
        vocab_size=64,
        hidden_size=32,
        n_layers=1,
        n_heads=4,
        n_kv_heads=4,
        mlp_ratio=2.0,
        max_seq_len=8,
        obs_dim=8,
        obs_tokens=1,
        action_dim=4,
        memory_slots=4,
        dropout=0.0,
    )
    model_a = VAGICore(cfg)
    model_b = VAGICore(cfg)
    model_b.load_state_dict(model_a.state_dict())

    obs = torch.randn(2, cfg.obs_dim)
    token = torch.randint(0, cfg.vocab_size, (2, 1), dtype=torch.long)
    state_a = model_a.init_state(2)
    state_b = model_b.init_state(2)

    out_a = model_a.step(input_ids=token, obs=obs, state=state_a)
    out_b = model_b.step(input_ids=token, obs=obs, state=state_b)
    mem_a = out_a["state"].mem
    mem_b = out_b["state"].mem

    assert torch.allclose(mem_a, mem_b, atol=1e-6)
    assert torch.isfinite(mem_a).all()
    assert mem_a.norm().item() < 1e3
