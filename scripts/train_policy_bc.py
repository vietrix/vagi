"""Behavior cloning for code-env action types."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import torch
from torch.utils.data import DataLoader, Dataset

from envs.code_env.actions import ACTION_DIM, action_type_id, parse_action
from vagi_core import VAGIConfig, VAGICore
from scripts.utils import set_deterministic


class RolloutDataset(Dataset):
    def __init__(self, records: List[Dict[str, object]]) -> None:
        self.records = records

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        record = self.records[idx]
        obs = torch.tensor(record["obs"], dtype=torch.float32)
        action = parse_action(str(record["action"]))
        action_id = action_type_id(action)
        return {"obs": obs, "action_id": torch.tensor(action_id, dtype=torch.long)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train policy head via behavior cloning.")
    parser.add_argument("--data", type=str, default="logs/code_rollouts.jsonl")
    parser.add_argument("--out-dir", type=str, default="runs/policy_bc")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--deterministic", action="store_true")
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


def main() -> None:
    args = parse_args()
    set_deterministic(args.seed, args.deterministic)

    records = _load_records(args.data)
    if not records:
        raise ValueError("No rollout records found.")
    obs_dim = len(records[0]["obs"])

    dataset = RolloutDataset(records)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    cfg = VAGIConfig(
        vocab_size=256,
        hidden_size=args.hidden_size,
        n_layers=args.layers,
        n_heads=args.heads,
        n_kv_heads=args.heads,
        mlp_ratio=2.0,
        max_seq_len=8,
        obs_dim=obs_dim,
        obs_tokens=args.obs_tokens,
        action_dim=ACTION_DIM,
        memory_slots=args.memory_slots,
        dropout=0.0,
        use_world_pred=False,
    )
    model = VAGICore(cfg)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    model.train()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(args.epochs):
        total_loss = 0.0
        total = 0
        correct = 0
        count = 0
        for batch in loader:
            obs = batch["obs"]
            action_ids = batch["action_id"]
            input_ids = torch.zeros((obs.shape[0], 1), dtype=torch.long)
            state = model.init_state(batch_size=obs.shape[0])

            out = model.forward(
                input_ids=input_ids,
                obs=obs,
                state=state,
                targets={"actions": action_ids},
                return_loss=True,
            )
            loss = out["loss"]
            if loss is None:
                raise ValueError("Missing policy loss.")
            logits = out["action_logits"]
            preds = torch.argmax(logits, dim=-1)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            total_loss += float(loss.item())
            total += 1
            correct += int((preds == action_ids).sum().item())
            count += int(action_ids.numel())

        mean_loss = total_loss / max(total, 1)
        acc = correct / max(count, 1)
        print(f"epoch={epoch + 1} loss={mean_loss:.6f} acc={acc:.3f}")

    torch.save(model.state_dict(), out_dir / "policy_bc.pt")


if __name__ == "__main__":
    main()
