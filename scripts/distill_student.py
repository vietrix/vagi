"""Distill a smaller vAGI student from a teacher model."""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

from vagi_core import VAGIConfig, VAGICore

from scripts.checkpoint import load_checkpoint, load_config_from_checkpoint, save_checkpoint
from scripts.data_utils import RandomDataset, load_tensor_dataset, move_batch_to_device, shift_labels, validate_batch
from scripts.utils import set_seed


class DistillAdapter(nn.Module):
    """Optional projection layers for distillation."""

    def __init__(self, teacher_trace_dim: int, student_trace_dim: int) -> None:
        super().__init__()
        if teacher_trace_dim == student_trace_dim:
            self.trace_proj: nn.Module = nn.Identity()
        else:
            self.trace_proj = nn.Linear(teacher_trace_dim, student_trace_dim, bias=False)

    def project_trace(self, trace: torch.Tensor) -> torch.Tensor:
        return self.trace_proj(trace)


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Distill a smaller student vAGI model.")
    parser.add_argument("--teacher-checkpoint", type=str, default=None)
    parser.add_argument("--data", type=str, default=None, help="Optional tensor dataset (.pt).")
    parser.add_argument("--out-dir", type=str, default="runs/distill_student")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--qat-epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--seq-len", type=int, default=16)
    parser.add_argument("--num-samples", type=int, default=256)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--temperature", type=float, default=2.0)
    parser.add_argument("--with-obs", action="store_true")
    parser.add_argument("--with-world", action="store_true")
    parser.add_argument("--with-supervised", action="store_true")

    # Teacher config overrides (used if no checkpoint config).
    parser.add_argument("--teacher-vocab-size", type=int, default=256)
    parser.add_argument("--teacher-hidden-size", type=int, default=128)
    parser.add_argument("--teacher-layers", type=int, default=4)
    parser.add_argument("--teacher-heads", type=int, default=8)
    parser.add_argument("--teacher-mlp-ratio", type=float, default=4.0)
    parser.add_argument("--teacher-max-seq-len", type=int, default=128)
    parser.add_argument("--teacher-obs-dim", type=int, default=16)
    parser.add_argument("--teacher-obs-tokens", type=int, default=2)
    parser.add_argument("--teacher-action-dim", type=int, default=8)
    parser.add_argument("--teacher-memory-slots", type=int, default=4)
    parser.add_argument("--teacher-use-world", action="store_true", default=False)
    parser.add_argument("--teacher-use-special-tokens", action="store_true", default=True)
    parser.add_argument("--teacher-no-special-tokens", action="store_false", dest="teacher_use_special_tokens")

    # Student config overrides.
    parser.add_argument("--student-hidden-size", type=int, default=64)
    parser.add_argument("--student-layers", type=int, default=2)
    parser.add_argument("--student-heads", type=int, default=4)
    parser.add_argument("--student-mlp-ratio", type=float, default=2.0)
    parser.add_argument("--student-memory-slots", type=int, default=None)

    # Distillation weights.
    parser.add_argument("--w-text", type=float, default=1.0)
    parser.add_argument("--w-policy", type=float, default=1.0)
    parser.add_argument("--w-value", type=float, default=0.5)
    parser.add_argument("--w-world", type=float, default=0.5)
    parser.add_argument("--w-uncertainty", type=float, default=0.25)
    parser.add_argument("--w-trace", type=float, default=0.5)

    parser.add_argument("--trace-parts", type=str, default="h_last,h_act,mem")
    parser.add_argument("--uncertainty-source", type=str, default="policy", choices=["policy", "text"])

    parser.add_argument("--qat-mode", type=str, default="int8+bf16", choices=["none", "int8", "bf16", "int8+bf16"])
    parser.add_argument("--qat-quantize-text", action="store_true", default=False)
    return parser.parse_args(argv)


def _build_teacher_config(args: argparse.Namespace) -> VAGIConfig:
    cfg = load_config_from_checkpoint(args.teacher_checkpoint) if args.teacher_checkpoint else None
    if cfg is not None:
        return cfg
    obs_dim = max(args.teacher_obs_dim, 1)
    return VAGIConfig(
        vocab_size=args.teacher_vocab_size,
        hidden_size=args.teacher_hidden_size,
        n_layers=args.teacher_layers,
        n_heads=args.teacher_heads,
        n_kv_heads=args.teacher_heads,
        mlp_ratio=args.teacher_mlp_ratio,
        max_seq_len=max(args.teacher_max_seq_len, 8),
        obs_dim=obs_dim,
        obs_tokens=args.teacher_obs_tokens,
        action_dim=args.teacher_action_dim,
        memory_slots=args.teacher_memory_slots,
        dropout=0.0,
        use_world_pred=args.teacher_use_world or args.with_world,
        use_special_tokens=args.teacher_use_special_tokens,
    )


def _build_student_config(args: argparse.Namespace, teacher_cfg: VAGIConfig) -> VAGIConfig:
    memory_slots = args.student_memory_slots
    if memory_slots is None:
        memory_slots = teacher_cfg.memory_slots
    return VAGIConfig(
        vocab_size=teacher_cfg.vocab_size,
        hidden_size=args.student_hidden_size,
        n_layers=args.student_layers,
        n_heads=args.student_heads,
        n_kv_heads=args.student_heads,
        mlp_ratio=args.student_mlp_ratio,
        max_seq_len=teacher_cfg.max_seq_len,
        obs_dim=teacher_cfg.obs_dim,
        obs_tokens=teacher_cfg.obs_tokens,
        action_dim=teacher_cfg.action_dim,
        memory_slots=memory_slots,
        dropout=0.0,
        use_world_pred=teacher_cfg.use_world_pred,
        use_special_tokens=teacher_cfg.use_special_tokens,
    )


def _build_dataloader(args: argparse.Namespace, cfg: VAGIConfig) -> DataLoader:
    if args.data:
        dataset = load_tensor_dataset(args.data)
    else:
        dataset = RandomDataset(
            num_samples=args.num_samples,
            seq_len=args.seq_len,
            vocab_size=cfg.vocab_size,
            obs_dim=cfg.obs_dim,
            action_dim=cfg.action_dim,
            with_obs=args.with_obs,
            with_world=args.with_world,
            seed=args.seed,
        )
    return DataLoader(dataset, batch_size=args.batch_size, shuffle=True)


def _entropy_from_logits(logits: torch.Tensor) -> torch.Tensor:
    probs = torch.softmax(logits, dim=-1)
    log_probs = torch.log_softmax(logits, dim=-1)
    return -(probs * log_probs).sum(dim=-1)


def _kd_loss(student_logits: torch.Tensor, teacher_logits: torch.Tensor, temperature: float) -> torch.Tensor:
    if temperature <= 0:
        raise ValueError("temperature must be > 0")
    student = F.log_softmax(student_logits / temperature, dim=-1)
    teacher = F.softmax(teacher_logits / temperature, dim=-1)
    return F.kl_div(student, teacher, reduction="batchmean") * (temperature**2)


def _extract_trace(
    features: Dict[str, Optional[torch.Tensor]],
    *,
    include_act: bool,
    include_mem: bool,
) -> torch.Tensor:
    h_last = features.get("h_last")
    if h_last is None:
        raise ValueError("features missing h_last for trace distillation")
    h_act = features.get("h_act")
    if h_act is None:
        h_act = h_last
    parts = [h_last]
    if include_act:
        parts.append(h_act)
    if include_mem:
        mem_next = features.get("mem_next")
        if mem_next is None:
            mem_feat = torch.zeros_like(h_last)
        else:
            mem_feat = mem_next.mean(dim=1)
        parts.append(mem_feat)
    return torch.cat(parts, dim=-1)


def _fake_quant_int8_ste(x: torch.Tensor) -> torch.Tensor:
    if x.numel() == 0:
        return x
    max_val = x.detach().abs().max()
    if max_val <= 0:
        return x
    scale = max_val / 127.0
    q = torch.clamp(torch.round(x / scale), -127, 127) * scale
    return x + (q - x).detach()


def _fake_cast_bf16_ste(x: torch.Tensor) -> torch.Tensor:
    q = x.to(torch.bfloat16).to(x.dtype)
    return x + (q - x).detach()


def _apply_quant_mode(x: torch.Tensor, mode: str) -> torch.Tensor:
    if mode == "int8":
        return _fake_quant_int8_ste(x)
    if mode == "bf16":
        return _fake_cast_bf16_ste(x)
    if mode == "int8+bf16":
        return _fake_cast_bf16_ste(_fake_quant_int8_ste(x))
    return x


def train_student(args: argparse.Namespace) -> Tuple[VAGICore, Dict[str, float]]:
    set_seed(args.seed)
    device = torch.device(args.device)

    teacher_cfg = _build_teacher_config(args)
    student_cfg = _build_student_config(args, teacher_cfg)

    if teacher_cfg.vocab_size != student_cfg.vocab_size:
        raise ValueError("Teacher and student must share vocab_size for distillation.")
    if teacher_cfg.action_dim != student_cfg.action_dim:
        raise ValueError("Teacher and student must share action_dim for distillation.")
    if teacher_cfg.obs_dim != student_cfg.obs_dim:
        raise ValueError("Teacher and student must share obs_dim for distillation.")
    if teacher_cfg.obs_tokens != student_cfg.obs_tokens:
        raise ValueError("Teacher and student must share obs_tokens for distillation.")
    if teacher_cfg.use_special_tokens != student_cfg.use_special_tokens:
        raise ValueError("Teacher and student must share use_special_tokens for distillation.")

    teacher = VAGICore(teacher_cfg).to(device)
    if args.teacher_checkpoint:
        load_checkpoint(args.teacher_checkpoint, model=teacher, optimizer=None, device=device)
    teacher.eval()
    for param in teacher.parameters():
        param.requires_grad = False

    student = VAGICore(student_cfg).to(device)
    student.train()

    trace_parts = [part.strip() for part in args.trace_parts.split(",") if part.strip()]
    include_act = "h_act" in trace_parts
    include_mem = "mem" in trace_parts
    teacher_trace_dim = student_trace_dim = student_cfg.hidden_size * (1 + int(include_act) + int(include_mem))
    teacher_trace_dim = teacher_cfg.hidden_size * (1 + int(include_act) + int(include_mem))

    adapter = DistillAdapter(teacher_trace_dim, student_trace_dim).to(device)

    params = list(student.parameters()) + list(adapter.parameters())
    optimizer = torch.optim.AdamW(params, lr=args.lr)
    loader = _build_dataloader(args, teacher_cfg)

    metrics: Dict[str, float] = {
        "loss": 0.0,
        "kd_text": 0.0,
        "kd_policy": 0.0,
        "kd_value": 0.0,
        "kd_world": 0.0,
        "kd_uncertainty": 0.0,
        "kd_trace": 0.0,
    }

    def _run_epoch(phase: str, quant_mode: str) -> None:
        nonlocal metrics
        epoch_metrics = {key: 0.0 for key in metrics}
        step = 0
        for batch in loader:
            batch = move_batch_to_device(batch, device)
            validate_batch(batch, teacher_cfg, require_obs=args.with_obs)

            input_ids = batch["input_ids"]
            labels = batch.get("labels")
            if labels is None:
                labels = shift_labels(input_ids)
            obs = batch.get("obs") if args.with_obs else None
            teacher_state = teacher.init_state(input_ids.shape[0], device=device)
            student_state = student.init_state(input_ids.shape[0], device=device)

            with torch.no_grad():
                teacher_out = teacher.forward(
                    input_ids=input_ids,
                    obs=obs,
                    state=teacher_state,
                    return_loss=False,
                    return_features=True,
                )

            student_out = student.forward(
                input_ids=input_ids,
                obs=obs,
                state=student_state,
                return_loss=False,
                return_features=True,
            )

            if quant_mode != "none":
                student_out = dict(student_out)
                student_out["action_logits"] = _apply_quant_mode(student_out["action_logits"], quant_mode)
                student_out["value"] = _apply_quant_mode(student_out["value"], quant_mode)
                if student_out["world_pred"] is not None:
                    student_out["world_pred"] = _apply_quant_mode(student_out["world_pred"], quant_mode)
                if args.qat_quantize_text:
                    student_out["text_logits"] = _apply_quant_mode(student_out["text_logits"], quant_mode)
                if "features" in student_out and student_out["features"] is not None:
                    features = dict(student_out["features"])
                    for key in ("h_last", "h_act", "mem_next"):
                        if features.get(key) is not None:
                            features[key] = _apply_quant_mode(features[key], quant_mode)
                    student_out["features"] = features

            include_special = teacher_cfg.use_special_tokens and obs is not None
            k_prefix = teacher_cfg.obs_tokens + (1 if include_special else 0) if obs is not None else 0
            k_suffix = 2 if include_special else 0

            text_student = student_out["text_logits"]
            text_teacher = teacher_out["text_logits"]
            end = text_student.shape[1] - k_suffix if k_suffix else text_student.shape[1]
            text_student = text_student[:, k_prefix:end, :]
            text_teacher = text_teacher[:, k_prefix:end, :]

            loss_text = _kd_loss(text_student.reshape(-1, text_student.shape[-1]),
                                 text_teacher.reshape(-1, text_teacher.shape[-1]),
                                 args.temperature)
            loss_policy = _kd_loss(student_out["action_logits"], teacher_out["action_logits"], args.temperature)
            loss_value = F.mse_loss(student_out["value"], teacher_out["value"])

            loss_world = torch.zeros((), device=device)
            if student_out["world_pred"] is not None and teacher_out["world_pred"] is not None:
                loss_world = F.mse_loss(student_out["world_pred"], teacher_out["world_pred"])

            loss_uncertainty = torch.zeros((), device=device)
            if args.uncertainty_source == "policy":
                teacher_unc = _entropy_from_logits(teacher_out["action_logits"])
                student_unc = _entropy_from_logits(student_out["action_logits"])
                loss_uncertainty = F.mse_loss(student_unc, teacher_unc)
            else:
                teacher_unc = _entropy_from_logits(text_teacher)
                student_unc = _entropy_from_logits(text_student)
                loss_uncertainty = F.mse_loss(student_unc, teacher_unc)

            loss_trace = torch.zeros((), device=device)
            if "features" in teacher_out and "features" in student_out:
                teacher_trace = _extract_trace(teacher_out["features"], include_act=include_act, include_mem=include_mem)
                student_trace = _extract_trace(student_out["features"], include_act=include_act, include_mem=include_mem)
                loss_trace = F.mse_loss(student_trace, adapter.project_trace(teacher_trace))

            total = (
                args.w_text * loss_text
                + args.w_policy * loss_policy
                + args.w_value * loss_value
                + args.w_world * loss_world
                + args.w_uncertainty * loss_uncertainty
                + args.w_trace * loss_trace
            )

            if args.with_supervised:
                targets: Dict[str, torch.Tensor] = {}
                if "actions" in batch:
                    targets["actions"] = batch["actions"]
                if "values" in batch:
                    targets["values"] = batch["values"]
                if args.with_world and "obs_next" in batch:
                    targets["obs_next"] = batch["obs_next"]
                sup_out = student.forward(
                    input_ids=input_ids,
                    obs=obs,
                    state=student_state,
                    labels=labels,
                    targets=targets,
                    return_loss=True,
                    return_features=False,
                )
                if sup_out["loss"] is not None:
                    total = total + sup_out["loss"]

            optimizer.zero_grad(set_to_none=True)
            total.backward()
            optimizer.step()

            epoch_metrics["loss"] += float(total.detach().item())
            epoch_metrics["kd_text"] += float(loss_text.detach().item())
            epoch_metrics["kd_policy"] += float(loss_policy.detach().item())
            epoch_metrics["kd_value"] += float(loss_value.detach().item())
            epoch_metrics["kd_world"] += float(loss_world.detach().item())
            epoch_metrics["kd_uncertainty"] += float(loss_uncertainty.detach().item())
            epoch_metrics["kd_trace"] += float(loss_trace.detach().item())

            step += 1
            if args.max_steps is not None and step >= args.max_steps:
                break

        if step > 0:
            metrics = {k: v / step for k, v in epoch_metrics.items()}
            print(
                f"{phase}: loss={metrics['loss']:.6f} "
                f"kd_text={metrics['kd_text']:.6f} kd_policy={metrics['kd_policy']:.6f} "
                f"kd_value={metrics['kd_value']:.6f} kd_world={metrics['kd_world']:.6f} "
                f"kd_uncertainty={metrics['kd_uncertainty']:.6f} kd_trace={metrics['kd_trace']:.6f}"
            )

    _run_epoch("distill", "none")
    if args.qat_epochs > 0 and args.qat_mode != "none":
        for _ in range(args.qat_epochs):
            _run_epoch("qat", args.qat_mode)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    meta = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "teacher": asdict(teacher_cfg),
        "student": asdict(student_cfg),
        "distill": {
            "temperature": args.temperature,
            "weights": {
                "text": args.w_text,
                "policy": args.w_policy,
                "value": args.w_value,
                "world": args.w_world,
                "uncertainty": args.w_uncertainty,
                "trace": args.w_trace,
            },
            "trace_parts": trace_parts,
            "uncertainty_source": args.uncertainty_source,
        },
        "qat": {"mode": args.qat_mode, "epochs": args.qat_epochs},
    }
    (out_dir / "distill_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    save_checkpoint(out_dir, model=student, optimizer=None, config=student_cfg, step=0)
    return student, metrics


def main() -> None:
    args = parse_args()
    train_student(args)


if __name__ == "__main__":
    main()
