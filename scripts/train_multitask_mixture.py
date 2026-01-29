"""Multi-task offline training with mixture sampling."""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import torch
from safetensors.torch import load_file
from torch.nn import functional as F

from io.checkpoint import save_checkpoint
from scripts.utils import get_lr, set_deterministic
from utils.data.pack import pack_batches
from utils.data.reader import read_jsonl
from vagi_core import VAGIConfig, VAGICore


@dataclass
class SourceSpec:
    path: str
    weight: float
    task: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train multi-task mixture from JSONL rollouts.")
    parser.add_argument(
        "--source",
        action="append",
        default=[],
        help="Format: path:weight:task (weight and task optional).",
    )
    parser.add_argument("--out-dir", type=str, default="runs/multitask")
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--horizon", type=int, default=1)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--value-weight", type=float, default=0.5)
    parser.add_argument("--weight-clip", type=float, default=20.0)
    parser.add_argument("--awbc-temp", type=float, default=0.5)
    parser.add_argument("--anchor", type=str, default=None)
    parser.add_argument("--anchor-weight", type=float, default=0.0)
    parser.add_argument("--rep-weight", type=float, default=0.0)
    parser.add_argument("--rep-noise-std", type=float, default=0.0)
    parser.add_argument("--rep-method", type=str, default="mse", choices=["mse", "cosine", "contrastive"])
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--grad-checkpoint", action="store_true")
    parser.add_argument("--obs-noise-std", type=float, default=0.0)
    parser.add_argument("--gae-lambda", type=float, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--vocab-size", type=int, default=128)
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--obs-dim", type=int, default=16)
    parser.add_argument("--obs-tokens", type=int, default=2)
    parser.add_argument("--action-dim", type=int, default=8)
    parser.add_argument("--memory-slots", type=int, default=4)
    parser.add_argument("--log-every", type=int, default=25)
    parser.add_argument("--save-every", type=int, default=100)
    return parser.parse_args()


def _parse_sources(raw_sources: Iterable[str]) -> List[SourceSpec]:
    specs: List[SourceSpec] = []
    for item in raw_sources:
        parts = [part.strip() for part in item.split(":") if part.strip()]
        if not parts:
            continue
        path = parts[0]
        weight = float(parts[1]) if len(parts) > 1 else 1.0
        task = parts[2] if len(parts) > 2 else Path(path).stem
        specs.append(SourceSpec(path=path, weight=weight, task=task))
    if not specs:
        raise ValueError("At least one --source must be provided")
    return specs


def _normalize_weights(weights: List[float], temperature: float) -> List[float]:
    if temperature <= 0.0:
        raise ValueError("temperature must be > 0")
    scaled = [w ** (1.0 / temperature) for w in weights]
    total = sum(scaled)
    if total == 0.0:
        raise ValueError("All sampling weights are zero")
    return [w / total for w in scaled]


def _build_batch_iter(path: str, *, batch_size: int, horizon: int, gamma: float, task_id: int):
    return pack_batches(read_jsonl(path), batch_size=batch_size, horizon=horizon, gamma=gamma, task_id=task_id)


def _load_anchor(path: Optional[str]) -> Optional[Dict[str, torch.Tensor]]:
    if not path:
        return None
    return load_file(str(path))


def _anchor_loss(model: torch.nn.Module, anchor_state: Dict[str, torch.Tensor]) -> torch.Tensor:
    loss = None
    for name, param in model.named_parameters():
        if name not in anchor_state:
            continue
        anchor = anchor_state[name].to(param.device)
        delta = torch.mean((param - anchor) ** 2)
        loss = delta if loss is None else loss + delta
    if loss is None:
        return torch.tensor(0.0, device=next(model.parameters()).device)
    return loss


def _sample_source(rng: torch.Generator, probs: List[float]) -> int:
    dist = torch.tensor(probs)
    return int(torch.multinomial(dist, num_samples=1, generator=rng).item())


def main() -> None:
    args = parse_args()
    set_deterministic(args.seed, args.deterministic)

    sources = _parse_sources(args.source)
    probs = _normalize_weights([spec.weight for spec in sources], args.temperature)
    task_id_map = {spec.task: idx for idx, spec in enumerate(sources)}

    cfg = VAGIConfig(
        vocab_size=max(args.vocab_size, args.action_dim + 1),
        hidden_size=args.hidden_size,
        n_layers=args.layers,
        n_heads=args.heads,
        n_kv_heads=args.heads,
        mlp_ratio=2.0,
        max_seq_len=8,
        obs_dim=args.obs_dim,
        obs_tokens=args.obs_tokens,
        action_dim=args.action_dim,
        memory_slots=args.memory_slots,
        dropout=0.0,
        use_world_pred=False,
        use_grad_checkpoint=args.grad_checkpoint,
        use_task_embedding=len(sources) > 1,
        task_vocab_size=max(len(sources), 1),
    )
    model = VAGICore(cfg)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    model.train()

    anchor_state = _load_anchor(args.anchor)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = out_dir / "metrics.jsonl"

    iterators = [
        _build_batch_iter(
            spec.path,
            batch_size=args.batch_size,
            horizon=args.horizon,
            gamma=args.gamma,
            task_id=task_id_map[spec.task],
            gae_lambda=args.gae_lambda,
        )
        for spec in sources
    ]
    rng = torch.Generator().manual_seed(args.seed)
    stats: Dict[str, Dict[str, float]] = {spec.task: {"loss": 0.0, "count": 0.0} for spec in sources}

    for step in range(1, args.steps + 1):
        idx = _sample_source(rng, probs)
        spec = sources[idx]
        try:
            batch = next(iterators[idx])
        except StopIteration:
            iterators[idx] = _build_batch_iter(
                spec.path,
                batch_size=args.batch_size,
                horizon=args.horizon,
                gamma=args.gamma,
                task_id=task_id_map[spec.task],
                gae_lambda=args.gae_lambda,
            )
            batch = next(iterators[idx])

        obs = batch["obs"]
        if args.obs_noise_std > 0.0:
            obs = obs + torch.randn_like(obs) * args.obs_noise_std
        actions = batch["actions"]
        returns = batch["returns"]
        task_ids = batch.get("task_id")

        input_ids = actions.unsqueeze(1).clamp(max=cfg.vocab_size - 1)
        state = model.init_state(batch_size=obs.shape[0])
        autocast = torch.autocast("cpu", dtype=torch.bfloat16) if args.bf16 else torch.autocast("cpu", enabled=False)
        with autocast:
            out = model.forward(
                input_ids=input_ids,
                obs=obs,
                task_ids=task_ids,
                state=state,
                return_loss=False,
                return_hidden=args.rep_weight > 0.0,
            )

        logits = out["action_logits"]
        values = out["value"]
        if "advantages" in batch:
            advantages = batch["advantages"]
        else:
            advantages = returns - values.detach()
        adv_scaled = torch.clamp(advantages / max(args.awbc_temp, 1e-6), min=-20.0, max=20.0)
        weights = torch.exp(adv_scaled).clamp(max=args.weight_clip).squeeze(-1)

        ce = F.cross_entropy(logits, actions, reduction="none")
        policy_loss = torch.mean(ce * weights)
        value_loss = F.mse_loss(values, returns)
        loss = policy_loss + args.value_weight * value_loss

        if args.rep_weight > 0.0 and args.rep_noise_std > 0.0:
            from vagi_core.losses import representation_loss

            obs_noisy = obs + torch.randn_like(obs) * args.rep_noise_std
            with autocast:
            rep_out = model.forward(
                input_ids=input_ids,
                obs=obs_noisy,
                task_ids=task_ids,
                state=state,
                return_loss=False,
                return_hidden=True,
            )
            rep_loss = representation_loss(
                out["hidden"],
                rep_out["hidden"],
                method=args.rep_method,
            )
            loss = loss + args.rep_weight * rep_loss

        if anchor_state is not None and args.anchor_weight > 0.0:
            loss = loss + args.anchor_weight * _anchor_loss(model, anchor_state)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        stats[spec.task]["loss"] += float(loss.item())
        stats[spec.task]["count"] += 1.0

        if args.log_every and step % args.log_every == 0:
            per_task = {
                task: stats[task]["loss"] / max(stats[task]["count"], 1.0) for task in stats
            }
            record = {
                "step": step,
                "loss": float(loss.item()),
                "task": spec.task,
                "per_task_loss": per_task,
                "lr": get_lr(optimizer),
            }
            with metrics_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record) + "\n")
            print(
                "step={step} task={task} loss={loss:.6f} lr={lr:.6f}".format(
                    step=step, task=spec.task, loss=loss.item(), lr=get_lr(optimizer)
                )
            )

        if args.save_every and step % args.save_every == 0:
            save_checkpoint(
                model,
                optimizer,
                step=step,
                out_dir=out_dir,
                extra={"timestamp": time.time()},
            )

    save_checkpoint(
        model,
        optimizer,
        step=args.steps,
        out_dir=out_dir,
        extra={"timestamp": time.time()},
    )


if __name__ == "__main__":
    main()
