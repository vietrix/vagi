import torch

from core.base import VAGIConfig, VAGICore

from io.checkpoint import load_checkpoint, save_checkpoint


def test_checkpoint_roundtrip(tmp_path) -> None:
    cfg = VAGIConfig(
        vocab_size=32,
        hidden_size=32,
        n_layers=1,
        n_heads=4,
        n_kv_heads=4,
        mlp_ratio=2.0,
        max_seq_len=8,
        obs_dim=8,
        obs_tokens=1,
        action_dim=4,
        memory_slots=2,
        dropout=0.0,
        use_world_pred=False,
    )
    model = VAGICore(cfg)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    input_ids = torch.randint(0, cfg.vocab_size, (2, 4), dtype=torch.long)
    labels = input_ids.clone()
    state = model.init_state(2)

    out = model.forward(input_ids=input_ids, state=state, labels=labels, return_loss=True)
    loss = out["loss"]
    assert loss is not None
    loss.backward()
    optimizer.step()

    save_checkpoint(model, optimizer, step=7, out_dir=tmp_path, extra={"tag": "roundtrip"})

    model2 = VAGICore(cfg)
    optimizer2 = torch.optim.AdamW(model2.parameters(), lr=1e-3)
    meta = load_checkpoint(model2, optimizer=optimizer2, ckpt_path=tmp_path)
    assert int(meta.get("step", 0)) == 7

    for name, param in model.state_dict().items():
        assert torch.allclose(param, model2.state_dict()[name])
