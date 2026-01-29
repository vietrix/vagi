"""Train a simple policy head offline from code-env rollouts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import torch
from torch.utils.data import DataLoader, Dataset

from vagi_core import VAGIConfig, VAGICore

from envs.code_env.actions import ACTION_DIM, action_type_id, parse_action
from scripts.utils import set_seed


class CodeRolloutDataset(Dataset):
    def __init__(self, records: List[Dict[str, object]]) -> None:
        self.records = records

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        record = self.records[idx]
        obs = torch.tensor(record["obs"], dtype=torch.float32)
        action = parse_action(str(record["action"]))
        action_id = action_type_id(action)
        return {
            "obs": obs,
            "action_id": torch.tensor(action_id, dtype=torch.long),
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train policy head from code-env rollouts.")
    parser.add_argument("--data", type=str, default="logs/code_rollouts.jsonl")
    parser.add_argument("--out-dir", type=str, default="runs/code_policy")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--obs-tokens", type=int, default=2)
    parser.add_argument("--memory-slots", type=int, default=4)
    return parser.parse_args()


def _load_records(path: str | Path) -> List[Dict[str, object]]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Missing rollouts: {path}")
    records: List[Dict[str, object]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        records.append(json.loads(line))
    return records


def train_policy(
    *,
    data_path: str | Path,
    out_dir: str | Path,
    epochs: int,
    batch_size: int,
    lr: float,
    seed: int,
    hidden_size: int,
    layers: int,
    heads: int,
    obs_tokens: int,
    memory_slots: int,
) -> List[Dict[str, float]]:
    set_seed(seed)
    records = _load_records(data_path)
    if not records:
        raise ValueError("No records found.")
    obs_dim = len(records[0]["obs"])
    dataset = CodeRolloutDataset(records)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    cfg = VAGIConfig(
        vocab_size=256,
        hidden_size=hidden_size,
        n_layers=layers,
        n_heads=heads,
        n_kv_heads=heads,
        mlp_ratio=2.0,
        max_seq_len=8,
        obs_dim=obs_dim,
        obs_tokens=obs_tokens,
        action_dim=ACTION_DIM,
        memory_slots=memory_slots,
        dropout=0.0,
        use_world_pred=False,
    )
    model = VAGICore(cfg)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics: List[Dict[str, float]] = []

    for epoch in range(epochs):
        total_loss = 0.0
        total = 0
        for batch in loader:
            obs = batch["obs"]
            actions = batch["action_id"]
            input_ids = torch.zeros((obs.shape[0], 1), dtype=torch.long)
            state = model.init_state(batch_size=obs.shape[0])

            out = model.forward(
                input_ids=input_ids,
                obs=obs,
                state=state,
                targets={"actions": actions},
                return_loss=True,
            )
            loss = out["loss"]
            if loss is None:
                raise ValueError("Missing policy loss.")

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            total_loss += float(loss.item())
            total += 1
        mean_loss = total_loss / max(total, 1)
        metrics.append({"epoch": float(epoch + 1), "policy_loss": float(mean_loss)})
        print(f"epoch={epoch + 1} policy_loss={mean_loss:.6f}")

    torch.save(model.state_dict(), out_dir / "policy_model.pt")
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics


def main() -> None:
    args = parse_args()
    train_policy(
        data_path=args.data,
        out_dir=args.out_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        seed=args.seed,
        hidden_size=args.hidden_size,
        layers=args.layers,
        heads=args.heads,
        obs_tokens=args.obs_tokens,
        memory_slots=args.memory_slots,
    )


if __name__ == "__main__":
    main()
