import torch

from vagi_core import VAGIConfig, VAGICore
from vagi_core.backbone import CausalTransformerBackbone


def test_act_token_appended_without_obs() -> None:
    cfg = VAGIConfig(
        vocab_size=32,
        hidden_size=16,
        n_layers=1,
        n_heads=4,
        n_kv_heads=4,
        mlp_ratio=2.0,
        max_seq_len=8,
        obs_dim=4,
        obs_tokens=0,
        action_dim=4,
        memory_slots=0,
        dropout=0.0,
        use_special_tokens=True,
        use_world_pred=False,
    )
    backbone = CausalTransformerBackbone(cfg)
    input_ids = torch.randint(0, cfg.vocab_size, (2, 3), dtype=torch.long)
    x, h_last, h_act, _ = backbone(input_ids=input_ids, obs=None, state=None)
    assert h_act is not None
    assert torch.allclose(h_act, h_last)
    assert x.shape[1] == input_ids.shape[1] + 1


def test_language_loss_alignment_with_act_token() -> None:
    cfg = VAGIConfig(
        vocab_size=64,
        hidden_size=32,
        n_layers=1,
        n_heads=4,
        n_kv_heads=4,
        mlp_ratio=2.0,
        max_seq_len=8,
        obs_dim=8,
        obs_tokens=2,
        action_dim=4,
        memory_slots=2,
        dropout=0.0,
        use_special_tokens=True,
        use_world_pred=False,
    )
    model = VAGICore(cfg)
    input_ids = torch.randint(0, cfg.vocab_size, (2, 4), dtype=torch.long)
    obs = torch.randn(2, cfg.obs_dim)
    labels = input_ids.clone()
    out = model.forward(input_ids=input_ids, obs=obs, labels=labels, return_loss=True)
    loss = out["loss"]
    assert loss is not None
    assert torch.isfinite(loss)
