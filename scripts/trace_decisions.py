"""Generate decision traces without revealing chain-of-thought."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from envs.toy_env import ToyEnv
from runtime.logging import JsonlWriter
from scripts.utils import set_deterministic
from vagi_core import VAGIConfig, VAGICore


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Log decision traces for vAGI planning.")
    parser.add_argument("--out", type=str, default="logs/decision_trace.jsonl")
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--obs-dim", type=int, default=8)
    parser.add_argument("--action-dim", type=int, default=4)
    parser.add_argument("--horizon", type=int, default=3)
    parser.add_argument("--num-candidates", type=int, default=4)
    parser.add_argument("--risk-penalty", type=float, default=1.0)
    parser.add_argument("--min-confidence", type=float, default=0.0)
    parser.add_argument("--policy-only", action="store_true")
    parser.add_argument("--deterministic", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_deterministic(args.seed, args.deterministic)
    device = torch.device("cpu")

    cfg = VAGIConfig(
        vocab_size=32,
        hidden_size=32,
        n_layers=1,
        n_heads=4,
        n_kv_heads=4,
        mlp_ratio=2.0,
        max_seq_len=8,
        obs_dim=args.obs_dim,
        obs_tokens=1,
        action_dim=args.action_dim,
        memory_slots=2,
        dropout=0.0,
        use_world_pred=True,
        world_model_horizon=1,
        use_uncertainty=True,
    )
    model = VAGICore(cfg).to(device)
    model.eval()

    env = ToyEnv(obs_dim=args.obs_dim, action_dim=args.action_dim, max_steps=args.steps, seed=args.seed)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with JsonlWriter(out_path) as writer:
        for episode in range(args.episodes):
            obs = env.reset()
            state = model.init_state(batch_size=1, device=device)
            token_id = 0
            for step in range(args.steps):
                input_ids = torch.tensor([[token_id]], dtype=torch.long, device=device)
                plan = model.plan_step(
                    input_ids=input_ids,
                    obs=obs.unsqueeze(0),
                    state=state,
                    num_candidates=args.num_candidates,
                    horizon=args.horizon,
                    risk_penalty=args.risk_penalty,
                    min_confidence_to_act=args.min_confidence,
                    policy_only=args.policy_only,
                    trace=True,
                )
                action = int(plan["action"].item())
                writer.write(
                    {
                        "episode": episode,
                        "timestep": step,
                        "action": action,
                        "mode": plan.get("mode"),
                        "stopReason": plan.get("stopReason"),
                        "value": float(plan.get("candidate_values")[0][0])
                        if plan.get("candidate_values") is not None
                        else None,
                        "uncertainty": float(plan.get("uncertainty")[0][0].item()),
                        "confidence": float(plan.get("confidence")[0][0].item()),
                        "horizon": args.horizon,
                        "numCandidates": args.num_candidates,
                        "trace": plan.get("trace"),
                    }
                )

                obs, _reward, done, _info = env.step(action)
                step_out = model.step(
                    input_ids=torch.tensor([[action]], dtype=torch.long, device=device),
                    obs=obs.unsqueeze(0),
                    state=state,
                )
                state = step_out["state"]
                token_id = action
                if done:
                    break

    print(f"Saved decision trace to {out_path.as_posix()}")


if __name__ == "__main__":
    main()
