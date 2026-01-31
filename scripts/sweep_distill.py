"""Run a distillation sweep and emit a Pareto report."""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Tuple

import torch

from envs.toy_env import ToyEnv
from vagi_core import VAGIConfig, VAGICore, save_vagi_lite_config

from scripts.distill_student import parse_args as distill_parse_args
from scripts.distill_student import train_student
from scripts.checkpoint import load_config_from_checkpoint


def _parse_int_list(text: str) -> List[int]:
    values = []
    for chunk in text.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        values.append(int(chunk))
    if not values:
        raise ValueError("Empty list value.")
    return values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sweep distillation configs and report Pareto front.")
    parser.add_argument("--teacher-checkpoint", type=str, default=None)
    parser.add_argument("--out-dir", type=str, default="runs/distill_sweep")
    parser.add_argument("--device", type=str, default="cpu")

    parser.add_argument("--hidden-sizes", type=str, default="32,48,64")
    parser.add_argument("--layers", type=str, default="1,2,3")
    parser.add_argument("--heads", type=str, default="2,4")

    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--qat-epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--seq-len", type=int, default=16)
    parser.add_argument("--num-samples", type=int, default=256)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--data", type=str, default=None)
    parser.add_argument("--with-obs", action="store_true")
    parser.add_argument("--with-world", action="store_true")
    parser.add_argument("--with-supervised", action="store_true")

    parser.add_argument("--temperature", type=float, default=2.0)
    parser.add_argument("--trace-parts", type=str, default="h_last,h_act,mem")
    parser.add_argument("--uncertainty-source", type=str, default="policy", choices=["policy", "text"])
    parser.add_argument("--qat-mode", type=str, default="int8+bf16", choices=["none", "int8", "bf16", "int8+bf16"])
    parser.add_argument("--qat-quantize-text", action="store_true", default=False)

    parser.add_argument("--w-text", type=float, default=1.0)
    parser.add_argument("--w-policy", type=float, default=1.0)
    parser.add_argument("--w-value", type=float, default=0.5)
    parser.add_argument("--w-world", type=float, default=0.5)
    parser.add_argument("--w-uncertainty", type=float, default=0.25)
    parser.add_argument("--w-trace", type=float, default=0.5)

    parser.add_argument("--teacher-vocab-size", type=int, default=256)
    parser.add_argument("--teacher-hidden-size", type=int, default=128)
    parser.add_argument("--teacher-layers", type=int, default=4)
    parser.add_argument("--teacher-heads", type=int, default=8)
    parser.add_argument("--teacher-mlp-ratio", type=float, default=4.0)
    parser.add_argument("--teacher-max-seq-len", type=int, default=128)
    parser.add_argument("--teacher-obs-dim", type=int, default=16)
    parser.add_argument("--teacher-obs-tokens", type=int, default=2)
    parser.add_argument("--teacher-action-dim", type=int, default=8)
    parser.add_argument("--teacher-memory-slots", type=int, default=4)
    parser.add_argument("--teacher-use-world", action="store_true", default=False)
    parser.add_argument("--teacher-use-special-tokens", action="store_true", default=True)
    parser.add_argument("--teacher-no-special-tokens", action="store_false", dest="teacher_use_special_tokens")

    parser.add_argument("--student-mlp-ratio", type=float, default=2.0)
    parser.add_argument("--student-memory-slots", type=int, default=None)

    parser.add_argument("--eval-episodes", type=int, default=5)
    parser.add_argument("--eval-steps", type=int, default=16)
    parser.add_argument("--latency-steps", type=int, default=50)
    parser.add_argument("--latency-warmup", type=int, default=5)
    parser.add_argument("--select-default", type=str, default="balanced", choices=["balanced", "best_score", "fastest"])
    parser.add_argument("--no-write-default", action="store_true", default=False)
    parser.add_argument("--allow-larger", action="store_true", default=False)
    return parser.parse_args()


def _count_params(model: VAGICore) -> int:
    return int(sum(p.numel() for p in model.parameters()))


def _model_size_mb(model: VAGICore) -> float:
    total_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    return float(total_bytes / (1024 * 1024))


def _measure_latency_ms(model: VAGICore, steps: int, warmup: int, device: torch.device) -> float:
    model.eval()
    obs = torch.randn(1, model.cfg.obs_dim, device=device)
    state = model.init_state(batch_size=1, device=device)
    token = torch.zeros((1, 1), dtype=torch.long, device=device)

    for _ in range(max(warmup, 0)):
        out = model.step(input_ids=token, obs=obs, state=state)
        token = torch.argmax(out["action_logits"], dim=-1).view(1, 1)
        state = out["state"]

    if device.type == "cuda":
        torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(max(steps, 1)):
        out = model.step(input_ids=token, obs=obs, state=state)
        token = torch.argmax(out["action_logits"], dim=-1).view(1, 1)
        state = out["state"]
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    return float((elapsed / max(steps, 1)) * 1000.0)


def _eval_pass_rate(model: VAGICore, episodes: int, steps: int, seed: int, device: torch.device) -> float:
    model.eval()
    env = ToyEnv(obs_dim=model.cfg.obs_dim, action_dim=model.cfg.action_dim, max_steps=steps, seed=seed)
    total = 0
    correct = 0
    for _ in range(episodes):
        obs = env.reset()
        state = model.init_state(batch_size=1, device=device)
        token_id = 0
        for _ in range(steps):
            input_ids = torch.tensor([[token_id]], dtype=torch.long, device=device)
            obs_tensor = obs.unsqueeze(0).to(device)
            out = model.step(input_ids=input_ids, obs=obs_tensor, state=state)
            action = int(torch.argmax(out["action_logits"], dim=-1).item())
            obs, reward, done, _info = env.step(action)
            correct += int(reward > 0.0)
            total += 1
            state = out["state"]
            token_id = action
            if done:
                break
    return float(correct / max(total, 1))


def _pareto_front(rows: List[Dict[str, object]]) -> List[Dict[str, object]]:
    front = []
    for candidate in rows:
        dominated = False
        cand_score = float(candidate["pass_rate"])
        cand_latency = float(candidate["latency_ms"])
        for other in rows:
            other_score = float(other["pass_rate"])
            other_latency = float(other["latency_ms"])
            if (other_score >= cand_score and other_latency <= cand_latency) and (
                other_score > cand_score or other_latency < cand_latency
            ):
                dominated = True
                break
        if not dominated:
            front.append(candidate)
    return front


def _select_default(rows: List[Dict[str, object]], mode: str) -> Dict[str, object]:
    if not rows:
        raise ValueError("No rows to select from.")
    if mode == "best_score":
        return max(rows, key=lambda r: float(r["pass_rate"]))
    if mode == "fastest":
        return min(rows, key=lambda r: float(r["latency_ms"]))
    scores = [float(r["pass_rate"]) for r in rows]
    latencies = [float(r["latency_ms"]) for r in rows]
    s_min, s_max = min(scores), max(scores)
    l_min, l_max = min(latencies), max(latencies)
    best = rows[0]
    best_val = -1e9
    for row in rows:
        score = float(row["pass_rate"])
        latency = float(row["latency_ms"])
        s_norm = 0.0 if s_max == s_min else (score - s_min) / (s_max - s_min)
        l_norm = 0.0 if l_max == l_min else (latency - l_min) / (l_max - l_min)
        value = s_norm - l_norm
        if value > best_val:
            best_val = value
            best = row
    return best


def _write_report(
    out_dir: Path,
    results: List[Dict[str, object]],
    pareto: List[Dict[str, object]],
    selected: Dict[str, object],
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "results": results,
        "pareto": pareto,
        "selected": selected,
    }
    (out_dir / "pareto_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    lines = [
        "# Pareto Report",
        "",
        "| label | params | latency_ms | pass_rate | memory_mb | pareto | default |",
        "| --- | ---: | ---: | ---: | ---: | :---: | :---: |",
    ]
    pareto_labels = {row["label"] for row in pareto}
    default_label = selected["label"]
    for row in results:
        label = row["label"]
        lines.append(
            "| {label} | {params} | {latency:.3f} | {pass_rate:.3f} | {memory:.3f} | {pareto} | {default} |".format(
                label=label,
                params=int(row["params"]),
                latency=float(row["latency_ms"]),
                pass_rate=float(row["pass_rate"]),
                memory=float(row["memory_mb"]),
                pareto="✓" if label in pareto_labels else "",
                default="✓" if label == default_label else "",
            )
        )
    (out_dir / "pareto_report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    hidden_sizes = _parse_int_list(args.hidden_sizes)
    layers = _parse_int_list(args.layers)
    heads = _parse_int_list(args.heads)

    base = distill_parse_args([])
    base.teacher_checkpoint = args.teacher_checkpoint
    base.data = args.data
    base.epochs = args.epochs
    base.qat_epochs = args.qat_epochs
    base.batch_size = args.batch_size
    base.lr = args.lr
    base.seed = args.seed
    base.seq_len = args.seq_len
    base.num_samples = args.num_samples
    base.max_steps = args.max_steps
    base.device = args.device
    base.temperature = args.temperature
    base.with_obs = args.with_obs
    base.with_world = args.with_world
    base.with_supervised = args.with_supervised

    base.teacher_vocab_size = args.teacher_vocab_size
    base.teacher_hidden_size = args.teacher_hidden_size
    base.teacher_layers = args.teacher_layers
    base.teacher_heads = args.teacher_heads
    base.teacher_mlp_ratio = args.teacher_mlp_ratio
    base.teacher_max_seq_len = args.teacher_max_seq_len
    base.teacher_obs_dim = args.teacher_obs_dim
    base.teacher_obs_tokens = args.teacher_obs_tokens
    base.teacher_action_dim = args.teacher_action_dim
    base.teacher_memory_slots = args.teacher_memory_slots
    base.teacher_use_world = args.teacher_use_world
    base.teacher_use_special_tokens = args.teacher_use_special_tokens

    base.student_mlp_ratio = args.student_mlp_ratio
    base.student_memory_slots = args.student_memory_slots

    base.w_text = args.w_text
    base.w_policy = args.w_policy
    base.w_value = args.w_value
    base.w_world = args.w_world
    base.w_uncertainty = args.w_uncertainty
    base.w_trace = args.w_trace
    base.trace_parts = args.trace_parts
    base.uncertainty_source = args.uncertainty_source
    base.qat_mode = args.qat_mode
    base.qat_quantize_text = args.qat_quantize_text

    device = torch.device(args.device)
    results: List[Dict[str, object]] = []

    teacher_cfg = load_config_from_checkpoint(args.teacher_checkpoint) if args.teacher_checkpoint else None
    if teacher_cfg is None:
        teacher_cfg = VAGIConfig(
            vocab_size=args.teacher_vocab_size,
            hidden_size=args.teacher_hidden_size,
            n_layers=args.teacher_layers,
            n_heads=args.teacher_heads,
            n_kv_heads=args.teacher_heads,
            mlp_ratio=args.teacher_mlp_ratio,
            max_seq_len=max(args.teacher_max_seq_len, 8),
            obs_dim=max(args.teacher_obs_dim, 1),
            obs_tokens=args.teacher_obs_tokens,
            action_dim=args.teacher_action_dim,
            memory_slots=args.teacher_memory_slots,
            dropout=0.0,
            use_world_pred=args.teacher_use_world or args.with_world,
            use_special_tokens=args.teacher_use_special_tokens,
        )

    for hidden_size in hidden_sizes:
        for layer_count in layers:
            for head_count in heads:
                if hidden_size % head_count != 0:
                    continue
                if not args.allow_larger:
                    if hidden_size >= teacher_cfg.hidden_size or layer_count >= teacher_cfg.n_layers:
                        continue
                label = f"h{hidden_size}_l{layer_count}_h{head_count}"
                run_dir = out_dir / label
                run_args = distill_parse_args([])
                run_args.__dict__.update(base.__dict__)
                run_args.student_hidden_size = hidden_size
                run_args.student_layers = layer_count
                run_args.student_heads = head_count
                run_args.out_dir = str(run_dir)

                print(f"==> distill {label}")
                student, train_metrics = train_student(run_args)

                params = _count_params(student)
                memory_mb = _model_size_mb(student)
                latency_ms = _measure_latency_ms(student, args.latency_steps, args.latency_warmup, device)
                pass_rate = _eval_pass_rate(student, args.eval_episodes, args.eval_steps, args.seed, device)

                results.append(
                    {
                        "label": label,
                        "params": params,
                        "latency_ms": latency_ms,
                        "pass_rate": pass_rate,
                        "memory_mb": memory_mb,
                        "train_metrics": train_metrics,
                        "config": asdict(student.cfg),
                    }
                )

    pareto = _pareto_front(results)
    selected = _select_default(pareto, args.select_default)
    _write_report(out_dir, results, pareto, selected)

    if not args.no_write_default:
        cfg = VAGIConfig(**selected["config"])
        save_vagi_lite_config(cfg)
        (out_dir / "selected_vagi_lite.json").write_text(
            json.dumps(selected["config"], indent=2), encoding="utf-8"
        )


if __name__ == "__main__":
    main()
