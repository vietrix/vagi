"""Train a world model head to predict next observation."""

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


class WorldDataset(Dataset):
    def __init__(self, records: List[Dict[str, object]], horizon: int) -> None:
        self.records = records
        self.horizon = horizon

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        record = self.records[idx]
        obs = torch.tensor(record["obs"], dtype=torch.float32)
        obs_future = torch.tensor(record["obs_future"], dtype=torch.float32)
        action = parse_action(str(record["action"]))
        action_id = action_type_id(action)
        return {
            "obs": obs,
            "obs_future": obs_future,
            "action_id": torch.tensor(action_id, dtype=torch.long),
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train world model from rollouts.")
    parser.add_argument("--data", type=str, default="logs/code_rollouts.jsonl")
    parser.add_argument("--out-dir", type=str, default="runs/world")
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
    parser.add_argument("--horizon", type=int, default=1)
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


def _attach_future_obs(records: List[Dict[str, object]], horizon: int) -> List[Dict[str, object]]:
    if horizon <= 0:
        raise ValueError("horizon must be > 0")
    enriched: List[Dict[str, object]] = []
    episode: List[Dict[str, object]] = []
    for record in records:
        episode.append(record)
        if record.get("done", False):
            enriched.extend(_augment_episode(episode, horizon))
            episode = []
    if episode:
        enriched.extend(_augment_episode(episode, horizon))
    return enriched


def _augment_episode(episode: List[Dict[str, object]], horizon: int) -> List[Dict[str, object]]:
    out: List[Dict[str, object]] = []
    length = len(episode)
    for idx in range(length):
        if idx + horizon > length:
            break
        future = []
        for step in range(horizon):
            future.append(episode[idx + step]["obs_next"])
        item = dict(episode[idx])
        item["obs_future"] = future
        out.append(item)
    return out


def main() -> None:
    args = parse_args()
    set_deterministic(args.seed, args.deterministic)

    records = _load_records(args.data)
    if not records:
        raise ValueError("No rollout records found.")
    records = _attach_future_obs(records, args.horizon)
    obs_dim = len(records[0]["obs"])

    dataset = WorldDataset(records, horizon=args.horizon)
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
        use_world_pred=True,
        world_model_horizon=args.horizon,
    )
    model = VAGICore(cfg)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    model.train()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(args.epochs):
        total_loss = 0.0
        total = 0
        for batch in loader:
            obs = batch["obs"]
            obs_future = batch["obs_future"]
            action_ids = batch["action_id"]
            input_ids = action_ids.unsqueeze(-1)
            state = model.init_state(batch_size=obs.shape[0])

            out = model.forward(
                input_ids=input_ids,
                obs=obs,
                state=state,
                targets={"obs_future": obs_future},
                return_loss=True,
            )
            loss = out["loss"]
            if loss is None:
                raise ValueError("Missing world loss.")

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            total_loss += float(loss.item())
            total += 1
        mean_loss = total_loss / max(total, 1)
        print(f"epoch={epoch + 1} world_loss={mean_loss:.6f}")

    torch.save(model.state_dict(), out_dir / "world_model.pt")


if __name__ == "__main__":
    main()
