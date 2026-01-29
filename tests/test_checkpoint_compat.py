import torch
from safetensors.torch import save_file

from io.checkpoint import load_checkpoint
from vagi_core import VAGIConfig, VAGICore


def test_loads_checkpoint_without_meta(tmp_path) -> None:
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
    model_path = tmp_path / "model.safetensors"
    save_file(model.state_dict(), str(model_path))

    model_loaded = VAGICore(cfg)
    meta = load_checkpoint(model_loaded, optimizer=None, ckpt_path=tmp_path)
    assert meta == {}
    for name, param in model.state_dict().items():
        assert torch.allclose(param, model_loaded.state_dict()[name])
