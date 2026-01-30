"""Export vAGI core to ONNX (inference-only heads)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch import nn

from vagi_core import VAGIConfig, VAGICore
from scripts.export_utils import build_metadata, write_metadata


class VAGIOnnxWrapper(nn.Module):
    def __init__(self, model: VAGICore) -> None:
        super().__init__()
        self.model = model

    def forward(self, input_ids: torch.Tensor, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        outputs = self.model.forward(input_ids=input_ids, obs=obs, state=None, return_loss=False)
        return outputs["text_logits"], outputs["action_logits"], outputs["value"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export vAGI to ONNX.")
    parser.add_argument("--out", type=str, default="exports/vagi.onnx")
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--seq-len", type=int, default=4)
    parser.add_argument("--obs-dim", type=int, default=16)
    parser.add_argument("--vocab-size", type=int, default=128)
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--obs-tokens", type=int, default=2)
    parser.add_argument("--action-dim", type=int, default=8)
    parser.add_argument("--memory-slots", type=int, default=4)
    parser.add_argument("--meta-out", type=str, default=None)
    return parser.parse_args()


def main() -> None:
    try:
        import onnx  # noqa: F401
    except ImportError as exc:  # pragma: no cover
        raise SystemExit("onnx package is required for export. Install via `pip install onnx`.") from exc

    args = parse_args()
    cfg = VAGIConfig(
        vocab_size=args.vocab_size,
        hidden_size=args.hidden_size,
        n_layers=args.layers,
        n_heads=args.heads,
        n_kv_heads=args.heads,
        mlp_ratio=2.0,
        max_seq_len=max(args.seq_len, args.obs_tokens + args.seq_len + 4),
        obs_dim=args.obs_dim,
        obs_tokens=args.obs_tokens,
        action_dim=args.action_dim,
        memory_slots=args.memory_slots,
        dropout=0.0,
        use_world_pred=False,
    )
    model = VAGICore(cfg).eval()
    wrapper = VAGIOnnxWrapper(model)

    input_ids = torch.zeros((args.batch_size, args.seq_len), dtype=torch.long)
    obs = torch.zeros((args.batch_size, args.obs_dim), dtype=torch.float32)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    torch.onnx.export(
        wrapper,
        (input_ids, obs),
        out_path.as_posix(),
        input_names=["input_ids", "obs"],
        output_names=["text_logits", "action_logits", "value"],
        dynamic_axes={
            "input_ids": {0: "batch", 1: "seq"},
            "obs": {0: "batch"},
            "text_logits": {0: "batch", 1: "seq"},
            "action_logits": {0: "batch"},
            "value": {0: "batch"},
        },
        opset_version=args.opset,
        do_constant_folding=True,
    )
    meta = build_metadata(cfg=cfg, export_format="onnx")
    if args.meta_out:
        meta_path = Path(args.meta_out)
        if meta_path.suffix == ".json":
            meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
        else:
            write_metadata(meta_path, meta)
    else:
        write_metadata(out_path, meta)
    print(f"Exported ONNX to {out_path}")


if __name__ == "__main__":
    main()
