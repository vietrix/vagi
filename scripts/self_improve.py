"""Offline self-improvement loop using best-of rollouts."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from safetensors.torch import load_file
from torch.nn import functional as F

from envs.toy_env import ToyEnv
from io.checkpoint import save_checkpoint
from scripts.utils import get_lr, set_deterministic
from utils.data.pack import pack_batches
from utils.data.reader import read_jsonl
from vagi_core import VAGIConfig, VAGICore


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Offline self-improvement loop.")
    parser.add_argument("--out-dir", type=str, default="runs/self_improve")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--steps", type=int, default=16)
    parser.add_argument("--obs-dim", type=int, default=8)
    parser.add_argument("--action-dim", type=int, default=4)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--min-reward", type=float, default=1.0)
    parser.add_argument("--max-uncertainty", type=float, default=10.0)
    parser.add_argument("--top-fraction", type=float, default=0.5)
    parser.add_argument("--uncertainty-weight", type=float, default=1.0)
    parser.add_argument("--novelty-weight", type=float, default=0.0)
    parser.add_argument("--success-weight", type=float, default=0.0)
    parser.add_argument("--horizon", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--policy-weight", type=float, default=1.0)
    parser.add_argument("--value-weight", type=float, default=0.5)
    parser.add_argument("--world-weight", type=float, default=1.0)
    parser.add_argument("--weight-clip", type=float, default=20.0)
    parser.add_argument("--awbc-temp", type=float, default=0.5)
    parser.add_argument("--gae-lambda", type=float, default=None)
    parser.add_argument("--anchor", type=str, default=None)
    parser.add_argument("--anchor-weight", type=float, default=0.0)
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--grad-checkpoint", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--log-every", type=int, default=25)
    return parser.parse_args()


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


def _sample_action(logits: torch.Tensor, temperature: float) -> int:
    if temperature <= 0.0:
        return int(torch.argmax(logits, dim=-1).item())
    scaled = logits / max(temperature, 1e-6)
    probs = torch.softmax(scaled, dim=-1)
    return int(torch.multinomial(probs, num_samples=1).item())


def _uncertainty_from_output(out: Dict[str, torch.Tensor]) -> float:
    world_logvar = out.get("world_logvar")
    value_logvar = out.get("value_logvar")
    uncertainty = 0.0
    count = 0
    if world_logvar is not None:
        if world_logvar.ndim == 3:
            uncertainty += float(torch.exp(world_logvar[:, 0, :]).mean().item())
        else:
            uncertainty += float(torch.exp(world_logvar).mean().item())
        count += 1
    if value_logvar is not None:
        uncertainty += float(torch.exp(value_logvar).mean().item())
        count += 1
    if count == 0:
        return 0.0
    return uncertainty / count


def _rollout_episode(
    model: VAGICore,
    env: ToyEnv,
    episode_id: str,
    steps: int,
    temperature: float,
) -> Tuple[List[Dict[str, object]], float, float]:
    obs = env.reset()
    state = model.init_state(batch_size=1)
    records: List[Dict[str, object]] = []
    total_reward = 0.0
    total_uncertainty = 0.0
    count_uncertainty = 0
    novelty = 0.0
    prev_obs = None
    last_action = 0

    for t in range(steps):
        input_ids = torch.tensor([[last_action]], dtype=torch.long)
        out = model.step(input_ids=input_ids, obs=obs.unsqueeze(0), state=state)
        action = _sample_action(out["action_logits"], temperature=temperature)
        uncertainty = _uncertainty_from_output(out)
        next_obs, reward, done, _ = env.step(action)
        record = {
            "schema_version": 1,
            "episode_id": episode_id,
            "timestep": t,
            "obs": obs.tolist(),
            "action": action,
            "reward": float(reward),
            "done": bool(done),
            "obs_next": next_obs.tolist(),
            "value": float(out["value"].item()),
            "info": {"uncertainty": uncertainty},
        }
        records.append(record)
        total_reward += float(reward)
        total_uncertainty += uncertainty
        count_uncertainty += 1
        obs = next_obs
        if prev_obs is not None:
            novelty += float(torch.mean(torch.abs(obs - prev_obs)).item())
        prev_obs = obs
        state = out["state"]
        last_action = action
        if done:
            break

    mean_uncertainty = total_uncertainty / max(count_uncertainty, 1)
    mean_novelty = novelty / max(len(records), 1)
    return records, total_reward, mean_uncertainty, mean_novelty


def _write_jsonl(path: Path, records: List[Dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")


def _score_episode(
    reward: float,
    uncertainty: float,
    novelty: float,
    *,
    uncertainty_weight: float,
    novelty_weight: float,
    success_weight: float,
) -> float:
    success_bonus = success_weight if reward > 0.0 else 0.0
    return reward - uncertainty_weight * uncertainty + novelty_weight * novelty + success_bonus


def main() -> None:
    args = parse_args()
    set_deterministic(args.seed, args.deterministic)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = VAGIConfig(
        vocab_size=max(32, args.action_dim + 1),
        hidden_size=64,
        n_layers=2,
        n_heads=4,
        n_kv_heads=4,
        mlp_ratio=2.0,
        max_seq_len=8,
        obs_dim=args.obs_dim,
        obs_tokens=2,
        action_dim=args.action_dim,
        memory_slots=4,
        dropout=0.0,
        use_world_pred=True,
        world_model_horizon=args.horizon,
        use_uncertainty=True,
        use_grad_checkpoint=args.grad_checkpoint,
    )
    model = VAGICore(cfg)
    model.eval()

    all_records: List[Dict[str, object]] = []
    episode_scores: List[Tuple[str, float, float, float, float]] = []
    for ep in range(args.episodes):
        env = ToyEnv(obs_dim=args.obs_dim, action_dim=args.action_dim, max_steps=args.steps, seed=args.seed + ep)
        episode_id = f"ep-{ep:04d}"
        records, reward, uncertainty, novelty = _rollout_episode(
            model, env, episode_id=episode_id, steps=args.steps, temperature=args.temperature
        )
        score = _score_episode(
            reward,
            uncertainty,
            novelty,
            uncertainty_weight=args.uncertainty_weight,
            novelty_weight=args.novelty_weight,
            success_weight=args.success_weight,
        )
        episode_scores.append((episode_id, reward, uncertainty, novelty, score))
        all_records.extend(records)

    rollouts_path = out_dir / "rollouts.jsonl"
    _write_jsonl(rollouts_path, all_records)

    episode_scores.sort(key=lambda item: item[4], reverse=True)
    top_count = max(1, int(len(episode_scores) * args.top_fraction))
    accepted = [
        ep for ep in episode_scores[:top_count] if ep[1] >= args.min_reward and ep[2] <= args.max_uncertainty
    ]
    if not accepted:
        raise ValueError("No episodes passed quality gate.")
    accepted_ids = {ep[0] for ep in accepted}

    best_records = [record for record in all_records if record["episode_id"] in accepted_ids]
    best_path = out_dir / "rollouts_best.jsonl"
    _write_jsonl(best_path, best_records)

    anchor_state = _load_anchor(args.anchor)
    model_train = VAGICore(cfg)
    optimizer = torch.optim.AdamW(model_train.parameters(), lr=args.lr)
    model_train.train()

    step = 0
    for epoch in range(args.epochs):
        batch_iter = pack_batches(
            read_jsonl(best_path),
            batch_size=args.batch_size,
            horizon=args.horizon,
            gamma=args.gamma,
            gae_lambda=args.gae_lambda,
        )
        for batch in batch_iter:
            obs = batch["obs"]
            actions = batch["actions"]
            returns = batch["returns"]
            obs_future = batch["obs_future"]
            mask = batch["mask"]

            input_ids = actions.unsqueeze(1).clamp(max=cfg.vocab_size - 1)
            state = model_train.init_state(batch_size=obs.shape[0])

            autocast = torch.autocast("cpu", dtype=torch.bfloat16) if args.bf16 else torch.autocast("cpu", enabled=False)
            with autocast:
                out = model_train.forward(
                    input_ids=input_ids,
                    obs=obs,
                    state=state,
                    return_loss=False,
                )
            logits = out["action_logits"]
            values = out["value"]
            mean = out["world_pred"]
            logvar = out["world_logvar"]
            if mean is None:
                raise ValueError("world_pred required for self-improvement distill")

            if "advantages" in batch:
                advantages = batch["advantages"]
            else:
                advantages = returns - values.detach()
            adv_scaled = torch.clamp(advantages / max(args.awbc_temp, 1e-6), min=-20.0, max=20.0)
            weights = torch.exp(adv_scaled).clamp(max=args.weight_clip).squeeze(-1)
            ce = F.cross_entropy(logits, actions, reduction="none")
            policy_loss = torch.mean(ce * weights)
            value_loss = F.mse_loss(values, returns)
            world_loss = _masked_nll(mean, logvar, obs_future, mask)

            loss = (
                args.policy_weight * policy_loss
                + args.value_weight * value_loss
                + args.world_weight * world_loss
            )

            if anchor_state is not None and args.anchor_weight > 0.0:
                loss = loss + args.anchor_weight * _anchor_loss(model_train, anchor_state)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            step += 1
            if args.log_every and step % args.log_every == 0:
                print(
                    "step={step} loss={loss:.6f} lr={lr:.6f}".format(
                        step=step, loss=loss.item(), lr=get_lr(optimizer)
                    )
                )

    save_checkpoint(
        model_train,
        optimizer,
        step=step,
        out_dir=out_dir,
        extra={"epochs": args.epochs, "timestamp": time.time()},
    )


def _masked_nll(
    mean: torch.Tensor,
    logvar: Optional[torch.Tensor],
    target: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    mask = mask.unsqueeze(-1)
    if logvar is None:
        loss = (mean - target) ** 2
    else:
        logvar = torch.clamp(logvar, min=-10.0, max=10.0)
        inv_var = torch.exp(-logvar)
        loss = 0.5 * ((target - mean) ** 2 * inv_var + logvar)
    weighted = loss * mask
    denom = torch.clamp(mask.sum(), min=1.0)
    return weighted.sum() / denom


if __name__ == "__main__":
    main()
