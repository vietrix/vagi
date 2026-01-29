"""Plan actions with a learned world model using MPC-style rollouts."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import List, Tuple

import torch

from vagi_core import KVCache, RecurrentState, VAGIConfig, VAGICore

from scripts.toy_env import ToyEnv
from scripts.utils import set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plan in the toy env using a world model.")
    parser.add_argument("--model-path", type=str, default=None, help="Path to world model weights (.pt).")
    parser.add_argument("--meta-path", type=str, default=None, help="Path to metadata.json from train_world.")
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--episode-length", type=int, default=16)
    parser.add_argument("--horizon", type=int, default=5)
    parser.add_argument("--candidates", type=int, default=32)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--vocab-size", type=int, default=128)
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--obs-dim", type=int, default=16)
    parser.add_argument("--obs-tokens", type=int, default=2)
    parser.add_argument("--action-dim", type=int, default=4)
    parser.add_argument("--memory-slots", type=int, default=4)
    parser.add_argument("--target", type=int, default=5)
    parser.add_argument("--use-special-tokens", action="store_true", default=True)
    parser.add_argument("--no-special-tokens", action="store_false", dest="use_special_tokens")
    return parser.parse_args()


def _clone_state(state: RecurrentState) -> RecurrentState:
    keys = None
    values = None
    if state.kv.keys is not None:
        keys = [k.clone() if k is not None else None for k in state.kv.keys]
    if state.kv.values is not None:
        values = [v.clone() if v is not None else None for v in state.kv.values]
    kv = KVCache(keys=keys, values=values)
    return RecurrentState(mem=state.mem.clone(), kv=kv, timestep=state.timestep)


def _reward_from_obs(obs: torch.Tensor) -> Tuple[float, bool]:
    pos = int(torch.round(obs[0]).item())
    target = int(torch.round(obs[1]).item())
    done = pos == target
    reward = 1.0 if done else -0.01
    return reward, done


def _load_meta(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _build_config(args: argparse.Namespace) -> VAGIConfig:
    if args.meta_path:
        meta = _load_meta(args.meta_path)
        model_meta = meta.get("model", {})
        return VAGIConfig(
            vocab_size=int(model_meta.get("vocab_size", args.vocab_size)),
            hidden_size=int(model_meta.get("hidden_size", args.hidden_size)),
            n_layers=int(model_meta.get("layers", args.layers)),
            n_heads=int(model_meta.get("heads", args.heads)),
            n_kv_heads=int(model_meta.get("heads", args.heads)),
            mlp_ratio=2.0,
            max_seq_len=int(model_meta.get("max_seq_len", max(8, args.episode_length))),
            obs_dim=int(model_meta.get("obs_dim", args.obs_dim)),
            obs_tokens=int(model_meta.get("obs_tokens", args.obs_tokens)),
            action_dim=int(model_meta.get("action_dim", args.action_dim)),
            memory_slots=int(model_meta.get("memory_slots", args.memory_slots)),
            dropout=0.0,
            use_world_pred=True,
            use_special_tokens=bool(model_meta.get("use_special_tokens", args.use_special_tokens)),
        )
    tokens_per_step = 1 + args.obs_tokens + (3 if args.use_special_tokens else 0)
    return VAGIConfig(
        vocab_size=args.vocab_size,
        hidden_size=args.hidden_size,
        n_layers=args.layers,
        n_heads=args.heads,
        n_kv_heads=args.heads,
        mlp_ratio=2.0,
        max_seq_len=max(8, tokens_per_step * args.episode_length),
        obs_dim=args.obs_dim,
        obs_tokens=args.obs_tokens,
        action_dim=args.action_dim,
        memory_slots=args.memory_slots,
        dropout=0.0,
        use_world_pred=True,
        use_special_tokens=args.use_special_tokens,
    )


def plan_action(
    *,
    model: VAGICore,
    obs: torch.Tensor,
    state: RecurrentState,
    action_dim: int,
    horizon: int,
    candidates: int,
    gamma: float,
    rng: random.Random,
) -> int:
    device = obs.device
    best_return = -1e9
    best_action = 0
    for _ in range(candidates):
        actions = [rng.randrange(action_dim) for _ in range(horizon)]
        sim_obs = obs
        sim_state = _clone_state(state)
        total_return = 0.0
        discount = 1.0
        for action in actions:
            input_ids = torch.tensor([[action]], dtype=torch.long, device=device)
            out = model.step(input_ids=input_ids, obs=sim_obs.unsqueeze(0), state=sim_state)
            world_pred = out["world_pred"]
            if world_pred is None:
                break
            if world_pred.ndim == 3:
                world_pred = world_pred[:, 0, :]
            sim_obs = world_pred.squeeze(0)
            sim_state = out["state"]
            reward, done = _reward_from_obs(sim_obs)
            total_return += discount * reward
            discount *= gamma
            if done:
                break
        if total_return > best_return:
            best_return = total_return
            best_action = actions[0]
    return best_action


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    rng = random.Random(args.seed)

    cfg = _build_config(args)
    model = VAGICore(cfg)
    model.eval()
    if args.model_path:
        state_dict = torch.load(args.model_path, map_location="cpu")
        model.load_state_dict(state_dict)

    env = ToyEnv(obs_dim=cfg.obs_dim, action_dim=cfg.action_dim, max_steps=args.episode_length, target=args.target)

    for ep_idx in range(args.episodes):
        obs = env.reset()
        state = model.init_state(batch_size=1, device="cpu")
        total_reward = 0.0
        for _ in range(args.episode_length):
            obs_tensor = obs.to(torch.float32)
            action = plan_action(
                model=model,
                obs=obs_tensor,
                state=state,
                action_dim=cfg.action_dim,
                horizon=args.horizon,
                candidates=args.candidates,
                gamma=args.gamma,
                rng=rng,
            )
            step_result = env.step(action)
            total_reward += float(step_result.reward)

            input_ids = torch.tensor([[action]], dtype=torch.long)
            obs_tensor = obs.unsqueeze(0)
            out = model.step(input_ids=input_ids, obs=obs_tensor, state=state)
            state = out["state"]
            obs = step_result.obs
            if step_result.done:
                break
        print(f"episode={ep_idx} total_reward={total_reward:.3f}")


if __name__ == "__main__":
    main()
