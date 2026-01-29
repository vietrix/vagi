"""Agent loop for CodeEnv with value-guided self-check."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Optional

import torch

from envs.code_env.actions import (
    ACTION_DIM,
    ApplyPatchAction,
    PlanLocateSourceAction,
    PlanPatchAction,
    PlanReadErrorsAction,
    PlanVerifyAction,
    ReadFileAction,
    RunTestsAction,
    serialize_action,
)
from envs.code_env.code_env import CodeEnv
from runtime.logging import JsonlWriter
from runtime.privacy import apply_retention, delete_logs
from vagi_core import VAGIConfig, VAGICore
from scripts.utils import set_deterministic


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run CodeEnv agent loop.")
    parser.add_argument("--task", type=str, default="envs/code_env/fixtures/mini_repo")
    parser.add_argument("--patch", type=str, default=None, help="Optional patch file for ApplyPatchAction")
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--obs-dim", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--log", type=str, default="runs/code_env_agent.jsonl")
    parser.add_argument("--value-threshold", type=float, default=0.0)
    parser.add_argument("--privacy-opt-in", action="store_true")
    parser.add_argument("--retain-days", type=int, default=7)
    parser.add_argument("--delete-logs", action="store_true")
    return parser.parse_args()


def _load_patch(path: Optional[str]) -> str:
    if not path:
        return ""
    return Path(path).read_text(encoding="utf-8")


def main() -> None:
    args = parse_args()
    set_deterministic(args.seed, args.deterministic)

    env = CodeEnv(obs_dim=args.obs_dim, max_steps=args.steps, seed=args.seed, repo_path=args.task)
    obs = env.reset()
    patch_text = _load_patch(args.patch)

    cfg = VAGIConfig(
        vocab_size=256,
        hidden_size=64,
        n_layers=2,
        n_heads=4,
        n_kv_heads=4,
        mlp_ratio=2.0,
        max_seq_len=8,
        obs_dim=args.obs_dim,
        obs_tokens=2,
        action_dim=ACTION_DIM,
        memory_slots=4,
        dropout=0.0,
        use_world_pred=True,
    )
    model = VAGICore(cfg)
    model.eval()
    state = model.init_state(batch_size=1)

    log_path = Path(args.log)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    if args.delete_logs:
        delete_logs(log_path.parent)
    apply_retention(log_path.parent, args.retain_days)
    writer = JsonlWriter(log_path, scrub_pii=True, privacy_opt_in=args.privacy_opt_in)
    try:
        for t in range(args.steps):
            input_ids = torch.zeros((1, 1), dtype=torch.long)
            out = model.step(input_ids=input_ids, obs=obs.unsqueeze(0), state=state)
            value_now = float(out["value"].item())

            if t == 0:
                action = serialize_action(PlanReadErrorsAction())
            elif t == 1:
                action = serialize_action(PlanLocateSourceAction())
            elif t == 2:
                action = serialize_action(ReadFileAction(path="src/buggy.py"))
            elif t == 3:
                action = serialize_action(PlanPatchAction())
            elif t == 4 and patch_text:
                # Value-guided check using world prediction
                imagined = model.forward(
                    input_ids=torch.zeros((1, 1), dtype=torch.long),
                    obs=obs.unsqueeze(0),
                    state=state,
                    return_loss=False,
                )
                world_pred = imagined["world_pred"]
                if world_pred is not None:
                    value_imagined = model.forward(
                        input_ids=torch.zeros((1, 1), dtype=torch.long),
                        obs=world_pred.detach(),
                        state=state,
                        return_loss=False,
                    )["value"]
                    if float(value_imagined.item()) < value_now - args.value_threshold:
                        action = serialize_action(PlanLocateSourceAction())
                    else:
                        action = serialize_action(ApplyPatchAction(path="src/buggy.py", diff=patch_text))
                else:
                    action = serialize_action(ApplyPatchAction(path="src/buggy.py", diff=patch_text))
            elif t == 5:
                action = serialize_action(PlanVerifyAction())
            else:
                action = serialize_action(RunTestsAction())

            obs, reward, done, info = env.step(action)
            if action.startswith("APPLY_PATCH") and patch_text:
                value_after = model.forward(
                    input_ids=torch.zeros((1, 1), dtype=torch.long),
                    obs=obs.unsqueeze(0),
                    state=state,
                    return_loss=False,
                )["value"]
                if float(value_after.item()) < value_now - args.value_threshold:
                    if env.rollback_last_patch():
                        obs = env.refresh_obs()
            record = {
                "timestep": t,
                "action": action,
                "reward": reward,
                "value": value_now,
                "fail_count": info.get("fail_count"),
                "top_error_type": info.get("top_error_type"),
            }
            writer.write(record)
            state = out["state"]
            if done:
                break
    finally:
        writer.close()


if __name__ == "__main__":
    main()
