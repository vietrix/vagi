"""Backend parity smoke tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

from vagi_core import VAGIConfig, VAGICore


def test_backend_parity(tmp_path: Path) -> None:
    pytest.importorskip("onnx")
    ort = pytest.importorskip("onnxruntime")

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
        obs_tokens=2,
        action_dim=6,
        memory_slots=2,
        dropout=0.0,
        use_world_pred=True,
        world_model_horizon=1,
        use_uncertainty=True,
    )
    model = VAGICore(cfg).eval()

    input_ids = torch.randint(0, cfg.vocab_size, (1, 2), dtype=torch.long)
    obs = torch.randn(1, cfg.obs_dim)

    with torch.no_grad():
        torch_out = model.forward(input_ids=input_ids, obs=obs, state=None, return_loss=False)
        plan = model.plan_step(
            input_ids=input_ids,
            obs=obs,
            state=model.init_state(batch_size=1),
            horizon=2,
            num_candidates=4,
            uncertainty_fallback=0.0,
        )

    class Wrapper(nn.Module):
        def __init__(self, core: VAGICore) -> None:
            super().__init__()
            self.core = core

        def forward(
            self, input_ids: torch.Tensor, obs: torch.Tensor
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
            outputs = self.core.forward(input_ids=input_ids, obs=obs, state=None, return_loss=False)
            world = outputs["world_pred"]
            if world is None:
                world = torch.zeros((obs.shape[0], cfg.obs_dim), dtype=obs.dtype, device=obs.device)
            if world.ndim == 3:
                world = world[:, 0, :]
            return outputs["text_logits"], outputs["action_logits"], outputs["value"], world

    wrapper = Wrapper(model)
    onnx_path = tmp_path / "parity.onnx"
    torch.onnx.export(
        wrapper,
        (input_ids, obs),
        onnx_path.as_posix(),
        input_names=["input_ids", "obs"],
        output_names=["text_logits", "action_logits", "value", "world_pred"],
        dynamic_axes={
            "input_ids": {0: "batch", 1: "seq"},
            "obs": {0: "batch"},
            "text_logits": {0: "batch", 1: "seq"},
            "action_logits": {0: "batch"},
            "value": {0: "batch"},
            "world_pred": {0: "batch"},
        },
        opset_version=17,
        do_constant_folding=True,
    )

    session = ort.InferenceSession(onnx_path.as_posix(), providers=["CPUExecutionProvider"])
    onnx_out = session.run(
        None,
        {"input_ids": input_ids.numpy(), "obs": obs.detach().numpy()},
    )
    text_logits, action_logits, value, world_pred = onnx_out

    np.testing.assert_allclose(
        torch_out["text_logits"].detach().numpy(), text_logits, rtol=1e-3, atol=1e-3
    )
    np.testing.assert_allclose(
        torch_out["action_logits"].detach().numpy(), action_logits, rtol=1e-3, atol=1e-3
    )
    np.testing.assert_allclose(torch_out["value"].detach().numpy(), value, rtol=1e-3, atol=1e-3)
    torch_world = torch_out["world_pred"]
    assert torch_world is not None
    if torch_world.ndim == 3:
        torch_world = torch_world[:, 0, :]
    np.testing.assert_allclose(torch_world.detach().numpy(), world_pred, rtol=1e-3, atol=1e-3)

    onnx_action = int(np.argmax(action_logits, axis=-1).reshape(-1)[0])
    plan_action = int(plan["action"].reshape(-1)[0].item())
    assert plan_action == onnx_action
