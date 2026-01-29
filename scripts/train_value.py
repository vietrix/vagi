"""Train value head on offline rollouts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import torch
from torch.utils.data import DataLoader, Dataset

from envs.code_env.actions import ACTION_DIM
from vagi_core import VAGIConfig, VAGICore
from scripts.utils import set_deterministic


class ValueDataset(Dataset):
    def __init__(self, records: List[Dict[str, object]]) -> None:
        self.records = records

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        record = self.records[idx]
        obs = torch.tensor(record["obs"], dtype=torch.float32)
        value = torch.tensor([record["return"]], dtype=torch.float32)
        return {"obs": obs, "value": value}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train value head from rollouts.")
    parser.add_argument("--data", type=str, default="logs/code_rollouts.jsonl")
    parser.add_argument("--out-dir", type=str, default="runs/value")
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
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--n-step", type=int, default=None)
    parser.add_argument("--lambda-return", type=float, default=0.9)
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


def _attach_returns(
    records: List[Dict[str, object]],
    gamma: float,
    n_step: int | None,
    lambda_return: float | None,
) -> List[Dict[str, object]]:
    enriched: List[Dict[str, object]] = []
    episode: List[Dict[str, object]] = []
    for record in records:
        episode.append(record)
        if record.get("done", False):
            _finalize_episode(episode, gamma, n_step, lambda_return, enriched)
            episode = []
    if episode:
        _finalize_episode(episode, gamma, n_step, lambda_return, enriched)
    return enriched


def _finalize_episode(
    episode: List[Dict[str, object]],
    gamma: float,
    n_step: int | None,
    lambda_return: float | None,
    out: List[Dict[str, object]],
) -> None:
    rewards = [float(record.get("reward", 0.0)) for record in episode]
    returns = _compute_returns(rewards, gamma, n_step, lambda_return)
    for record, ret in zip(episode, returns):
        item = dict(record)
        item["return"] = ret
        out.append(item)


def _compute_returns(
    rewards: List[float],
    gamma: float,
    n_step: int | None,
    lambda_return: float | None,
) -> List[float]:
    length = len(rewards)
    returns = [0.0 for _ in range(length)]
    for t in range(length):
        horizon = length - t
        if n_step is not None:
            horizon = min(horizon, max(n_step, 1))
        if lambda_return is None:
            returns[t] = _n_step_return(rewards, t, horizon, gamma)
        else:
            lam = max(0.0, min(1.0, float(lambda_return)))
            returns[t] = _lambda_return(rewards, t, horizon, gamma, lam)
    return returns


def _n_step_return(rewards: List[float], start: int, horizon: int, gamma: float) -> float:
    ret = 0.0
    for k in range(horizon):
        ret += (gamma ** k) * rewards[start + k]
    return ret


def _lambda_return(rewards: List[float], start: int, horizon: int, gamma: float, lam: float) -> float:
    ret = 0.0
    for n in range(1, horizon + 1):
        n_ret = _n_step_return(rewards, start, n, gamma)
        if n == horizon:
            weight = lam ** (n - 1)
        else:
            weight = (1.0 - lam) * (lam ** (n - 1))
        ret += weight * n_ret
    return ret


def main() -> None:
    args = parse_args()
    set_deterministic(args.seed, args.deterministic)

    records = _load_records(args.data)
    if not records:
        raise ValueError("No rollout records found.")
    records = _attach_returns(records, args.gamma, args.n_step, args.lambda_return)
    obs_dim = len(records[0]["obs"])

    dataset = ValueDataset(records)
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
        for batch in loader:
            obs = batch["obs"]
            values = batch["value"]
            input_ids = torch.zeros((obs.shape[0], 1), dtype=torch.long)
            state = model.init_state(batch_size=obs.shape[0])

            out = model.forward(
                input_ids=input_ids,
                obs=obs,
                state=state,
                targets={"values": values},
                return_loss=True,
            )
            loss = out["loss"]
            if loss is None:
                raise ValueError("Missing value loss.")

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            total_loss += float(loss.item())
            total += 1
        mean_loss = total_loss / max(total, 1)
        print(f"epoch={epoch + 1} value_loss={mean_loss:.6f}")

    torch.save(model.state_dict(), out_dir / "value_model.pt")


if __name__ == "__main__":
    main()
