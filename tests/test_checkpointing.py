import torch

from vagi_core import VAGIConfig, VAGICore

from scripts.checkpoint import load_checkpoint, load_config_from_checkpoint, save_checkpoint
from scripts.data_utils import shift_labels


def test_checkpoint_roundtrip(tmp_path) -> None:
    cfg = VAGIConfig(
        vocab_size=64,
        hidden_size=32,
        n_layers=1,
        n_heads=4,
        n_kv_heads=4,
        mlp_ratio=2.0,
        max_seq_len=16,
        obs_dim=8,
        obs_tokens=1,
        action_dim=4,
        memory_slots=2,
        dropout=0.0,
        use_world_pred=False,
    )
    model = VAGICore(cfg)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    batch_size, seq_len = 2, 4
    input_ids = torch.randint(0, cfg.vocab_size, (batch_size, seq_len), dtype=torch.long)
    obs = torch.randn(batch_size, cfg.obs_dim)
    labels = shift_labels(input_ids)
    state = model.init_state(batch_size)

    out = model.forward(input_ids=input_ids, obs=obs, state=state, labels=labels, return_loss=True)
    loss = out["loss"]
    assert loss is not None
    loss.backward()
    optimizer.step()

    save_checkpoint(tmp_path, model=model, optimizer=optimizer, config=cfg, step=5)
    cfg_loaded = load_config_from_checkpoint(tmp_path)
    assert cfg_loaded is not None
    assert cfg_loaded.vocab_size == cfg.vocab_size

    model2 = VAGICore(cfg)
    optimizer2 = torch.optim.AdamW(model2.parameters(), lr=1e-3)
    meta = load_checkpoint(tmp_path, model=model2, optimizer=optimizer2, device="cpu")
    assert int(meta.get("step", 0)) == 5

    for name, param in model.state_dict().items():
        assert torch.allclose(param, model2.state_dict()[name])
