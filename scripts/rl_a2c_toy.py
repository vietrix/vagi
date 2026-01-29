"""Run a minimal A2C-style loop on the toy environment."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Dict, List

import torch
import torch.nn.functional as F
from torch.distributions import Categorical

from vagi_core import RecurrentState, VAGIConfig, VAGICore

from scripts.toy_env import ToyEnv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a minimal A2C loop on the toy env.")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--episode-length", type=int, default=16)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--log-path", type=str, default="logs/rl_a2c_toy.jsonl")
    parser.add_argument("--vocab-size", type=int, default=128)
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--obs-dim", type=int, default=16)
    parser.add_argument("--obs-tokens", type=int, default=2)
    parser.add_argument("--action-dim", type=int, default=4)
    parser.add_argument("--memory-slots", type=int, default=4)
    parser.add_argument("--target", type=int, default=5)
    return parser.parse_args()


def _set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)


def _mean(values: List[float]) -> float:
    if not values:
        return 0.0
    return float(sum(values) / len(values))


def _detach_state(state: RecurrentState) -> RecurrentState:
    return RecurrentState(mem=state.mem.detach(), kv=state.kv, timestep=state.timestep)


def _step_with_grad(
    model: VAGICore, input_ids: torch.Tensor, obs: torch.Tensor, state: RecurrentState
) -> Dict[str, torch.Tensor]:
    # VAGICore.step is wrapped in no_grad. Use the undecorated function for training.
    step_fn = getattr(VAGICore.step, "__wrapped__", None)
    if step_fn is None:
        raise RuntimeError("VAGICore.step is missing __wrapped__ for grad-enabled use.")
    return step_fn(model, input_ids=input_ids, obs=obs, state=state)


def run_a2c(
    *,
    episodes: int = 10,
    episode_length: int = 16,
    gamma: float = 0.99,
    lr: float = 1e-3,
    seed: int = 0,
    log_path: str | Path = "logs/rl_a2c_toy.jsonl",
    vocab_size: int = 128,
    hidden_size: int = 64,
    layers: int = 2,
    heads: int = 4,
    obs_dim: int = 16,
    obs_tokens: int = 2,
    action_dim: int = 4,
    memory_slots: int = 4,
    target: int = 5,
) -> List[Dict[str, float]]:
    if episodes <= 0:
        raise ValueError("episodes must be > 0")
    if episode_length <= 0:
        raise ValueError("episode_length must be > 0")

    _set_seed(seed)
    device = torch.device("cpu")

    cfg = VAGIConfig(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        n_layers=layers,
        n_heads=heads,
        n_kv_heads=heads,
        mlp_ratio=2.0,
        max_seq_len=max(8, obs_tokens + 1),
        obs_dim=obs_dim,
        obs_tokens=obs_tokens,
        action_dim=action_dim,
        memory_slots=memory_slots,
        dropout=0.0,
        use_world_pred=False,
    )
    model = VAGICore(cfg).to(device)
    model.train()

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    env = ToyEnv(obs_dim=obs_dim, action_dim=action_dim, max_steps=episode_length, target=target)

    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    logs: List[Dict[str, float]] = []

    for ep_idx in range(episodes):
        obs = env.reset()
        state = model.init_state(batch_size=1, device=device)
        token_id = 0

        total_reward = 0.0
        policy_losses: List[float] = []
        value_losses: List[float] = []
        values: List[float] = []

        for _ in range(episode_length):
            input_ids = torch.tensor([[token_id % cfg.vocab_size]], dtype=torch.long, device=device)
            obs_tensor = obs.unsqueeze(0).to(device)

            out = _step_with_grad(model, input_ids=input_ids, obs=obs_tensor, state=state)
            action_logits = out["action_logits"]
            value = out["value"].squeeze(-1)

            dist = Categorical(logits=action_logits)
            action = dist.sample()
            log_prob = dist.log_prob(action)

            step_result = env.step(int(action.item()))
            next_obs = step_result.obs
            reward = step_result.reward
            done = step_result.done

            state = _detach_state(out["state"])

            with torch.no_grad():
                if done:
                    next_value = torch.zeros_like(value)
                else:
                    next_input_ids = torch.tensor(
                        [[int(action.item()) % cfg.vocab_size]],
                        dtype=torch.long,
                        device=device,
                    )
                    next_obs_tensor = next_obs.unsqueeze(0).to(device)
                    next_out = model.step(input_ids=next_input_ids, obs=next_obs_tensor, state=state)
                    next_value = next_out["value"].squeeze(-1)

            reward_tensor = torch.tensor([reward], dtype=value.dtype, device=device)
            target_value = reward_tensor + gamma * next_value
            advantage = target_value - value

            policy_loss = -(log_prob * advantage.detach()).mean()
            value_loss = F.mse_loss(value, target_value.detach())
            loss = policy_loss + value_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_reward += reward
            policy_losses.append(float(policy_loss.detach().item()))
            value_losses.append(float(value_loss.detach().item()))
            values.append(float(value.detach().mean().item()))

            obs = next_obs
            token_id = int(action.item())

            if done:
                break

        mean_policy_loss = _mean(policy_losses)
        mean_value_loss = _mean(value_losses)
        mean_value = _mean(values)

        record = {
            "episode": int(ep_idx),
            "total_reward": float(total_reward),
            "mean_value": float(mean_value),
            "policy_loss": float(mean_policy_loss),
            "value_loss": float(mean_value_loss),
        }
        logs.append(record)
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")

        print(
            "episode={ep} reward={reward:.3f} policy_loss={pl:.4f} value_loss={vl:.4f}".format(
                ep=ep_idx,
                reward=total_reward,
                pl=mean_policy_loss,
                vl=mean_value_loss,
            )
        )

    return logs


def main() -> None:
    args = parse_args()
    run_a2c(
        episodes=args.episodes,
        episode_length=args.episode_length,
        gamma=args.gamma,
        lr=args.lr,
        seed=args.seed,
        log_path=args.log_path,
        vocab_size=args.vocab_size,
        hidden_size=args.hidden_size,
        layers=args.layers,
        heads=args.heads,
        obs_dim=args.obs_dim,
        obs_tokens=args.obs_tokens,
        action_dim=args.action_dim,
        memory_slots=args.memory_slots,
        target=args.target,
    )


if __name__ == "__main__":
    main()
