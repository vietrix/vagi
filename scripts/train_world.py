"""Train the world-model head from rollout data."""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import torch
from torch.utils.data import DataLoader, Dataset
from torch.nn import functional as F

from vagi_core import VAGIConfig, VAGICore

from scripts.utils import set_seed


@dataclass
class RolloutSample:
    obs: torch.Tensor
    action: int
    next_obs: torch.Tensor
    episode: int
    step: int
    done: bool


class RolloutDataset(Dataset):
    def __init__(self, samples: List[RolloutSample]) -> None:
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]
        return {
            "obs": sample.obs,
            "actions": torch.tensor(sample.action, dtype=torch.long),
            "next_obs": sample.next_obs,
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train vAGI world head from rollouts.")
    parser.add_argument("--data", type=str, default="logs/rollouts.jsonl")
    parser.add_argument("--out-dir", type=str, default="runs/world_model")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--vocab-size", type=int, default=128)
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--obs-dim", type=int, default=16)
    parser.add_argument("--obs-tokens", type=int, default=2)
    parser.add_argument("--action-dim", type=int, default=4)
    parser.add_argument("--memory-slots", type=int, default=4)
    parser.add_argument("--max-seq-len", type=int, default=0)
    parser.add_argument("--use-special-tokens", action="store_true", default=True)
    parser.add_argument("--no-special-tokens", action="store_false", dest="use_special_tokens")
    return parser.parse_args()


def load_rollouts(path: str | Path) -> List[RolloutSample]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Rollouts not found: {path}")
    samples: List[RolloutSample] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        samples.append(
            RolloutSample(
                obs=torch.tensor(record["obs"], dtype=torch.float32),
                action=int(record["action"]),
                next_obs=torch.tensor(record["next_obs"], dtype=torch.float32),
                episode=int(record["episode"]),
                step=int(record["step"]),
                done=bool(record["done"]),
            )
        )
    return samples


def compute_rollout_error(samples: List[RolloutSample], model: VAGICore, device: torch.device) -> float:
    by_episode: Dict[int, List[RolloutSample]] = {}
    for sample in samples:
        by_episode.setdefault(sample.episode, []).append(sample)

    total_error = 0.0
    total_count = 0
    model.eval()
    with torch.no_grad():
        for episode, steps in by_episode.items():
            steps = sorted(steps, key=lambda s: s.step)
            if not steps:
                continue
            state = model.init_state(batch_size=1, device=device)
            pred_obs = steps[0].obs.to(device)
            for sample in steps:
                input_ids = torch.tensor([[sample.action]], dtype=torch.long, device=device)
                out = model.step(input_ids=input_ids, obs=pred_obs.unsqueeze(0), state=state)
                world_pred = out["world_pred"]
                if world_pred is None:
                    continue
                target = sample.next_obs.to(device)
                mse = F.mse_loss(world_pred.squeeze(0), target, reduction="mean")
                total_error += float(mse.item())
                total_count += 1
                pred_obs = world_pred.squeeze(0)
                state = out["state"]
                if sample.done:
                    break
    return total_error / max(total_count, 1)


def train_world(
    *,
    data_path: str | Path,
    out_dir: str | Path,
    epochs: int,
    batch_size: int,
    lr: float,
    seed: int,
    vocab_size: int,
    hidden_size: int,
    layers: int,
    heads: int,
    obs_dim: int,
    obs_tokens: int,
    action_dim: int,
    memory_slots: int,
    max_seq_len: int,
    use_special_tokens: bool,
) -> List[Dict[str, float]]:
    set_seed(seed)
    device = torch.device("cpu")

    samples = load_rollouts(data_path)
    if not samples:
        raise ValueError("Rollout dataset is empty.")

    tokens_per_step = 1 + obs_tokens + (3 if use_special_tokens else 0)
    if max_seq_len <= 0:
        max_seq_len = max(8, tokens_per_step * 8)

    cfg = VAGIConfig(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        n_layers=layers,
        n_heads=heads,
        n_kv_heads=heads,
        mlp_ratio=2.0,
        max_seq_len=max_seq_len,
        obs_dim=obs_dim,
        obs_tokens=obs_tokens,
        action_dim=action_dim,
        memory_slots=memory_slots,
        dropout=0.0,
        use_world_pred=True,
        use_special_tokens=use_special_tokens,
    )
    model = VAGICore(cfg).to(device)
    model.train()

    dataset = RolloutDataset(samples)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = out_dir / "metrics.jsonl"

    metrics: List[Dict[str, float]] = []

    for epoch in range(epochs):
        total_loss = 0.0
        total_count = 0
        for batch in loader:
            obs = batch["obs"].to(device)
            actions = batch["actions"].to(device)
            next_obs = batch["next_obs"].to(device)
            input_ids = actions.unsqueeze(1)

            state = model.init_state(batch_size=obs.shape[0], device=device)
            out = model.forward(
                input_ids=input_ids,
                obs=obs,
                state=state,
                targets={"obs_next": next_obs},
                return_loss=True,
            )
            loss = out["loss"]
            if loss is None:
                raise ValueError("World model loss missing from output.")

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            total_loss += float(loss.item())
            total_count += 1

        mean_loss = total_loss / max(total_count, 1)
        rollout_error = compute_rollout_error(samples, model, device)
        record = {
            "epoch": float(epoch + 1),
            "world_loss": float(mean_loss),
            "rollout_error": float(rollout_error),
        }
        metrics.append(record)
        with metrics_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")

        print(
            "epoch={epoch} world_loss={loss:.6f} rollout_error={err:.6f}".format(
                epoch=epoch + 1,
                loss=mean_loss,
                err=rollout_error,
            )
        )

    meta = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "data_path": str(data_path),
        "epochs": epochs,
        "batch_size": batch_size,
        "lr": lr,
        "seed": seed,
        "model": {
            "vocab_size": vocab_size,
            "hidden_size": hidden_size,
            "layers": layers,
            "heads": heads,
            "obs_dim": obs_dim,
            "obs_tokens": obs_tokens,
            "action_dim": action_dim,
            "memory_slots": memory_slots,
            "max_seq_len": max_seq_len,
            "use_special_tokens": use_special_tokens,
        },
    }
    (out_dir / "metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return metrics


def main() -> None:
    args = parse_args()
    train_world(
        data_path=args.data,
        out_dir=args.out_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        seed=args.seed,
        vocab_size=args.vocab_size,
        hidden_size=args.hidden_size,
        layers=args.layers,
        heads=args.heads,
        obs_dim=args.obs_dim,
        obs_tokens=args.obs_tokens,
        action_dim=args.action_dim,
        memory_slots=args.memory_slots,
        max_seq_len=args.max_seq_len,
        use_special_tokens=args.use_special_tokens,
    )


if __name__ == "__main__":
    main()
