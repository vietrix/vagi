"""Summarize benchmark results with stats and win rates."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path
from typing import Dict, List, Tuple


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize benchmark results.")
    parser.add_argument("--run-dir", type=str, default=None, help="Path to results/run_<timestamp>")
    parser.add_argument("--input", type=str, default=None, help="Path to results.json")
    parser.add_argument("--out", type=str, default=None, help="Optional output summary markdown path")
    parser.add_argument("--min-pass-rate", type=float, default=None)
    parser.add_argument("--max-reward-drop", type=float, default=None)
    return parser.parse_args()


def _load_records(path: Path) -> Tuple[dict, List[dict]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    records = data.get("records", [])
    if not isinstance(records, list):
        raise ValueError("results.json missing records list")
    return data, records


def _mean_std(values: List[float]) -> Tuple[float, float]:
    if not values:
        return 0.0, 0.0
    if len(values) == 1:
        return values[0], 0.0
    return statistics.mean(values), statistics.pstdev(values)


def _aggregate(records: List[dict]) -> Dict[str, float]:
    total = len(records)
    if total == 0:
        return {
            "pass_rate": 0.0,
            "mean_reward": 0.0,
            "std_reward": 0.0,
            "mean_steps": 0.0,
            "std_steps": 0.0,
            "mean_latency_s": 0.0,
            "std_latency_s": 0.0,
            "episodes": 0,
        }
    rewards = [float(r["total_reward"]) for r in records]
    steps = [float(r["steps"]) for r in records]
    latency = [float(r["latency_s"]) for r in records]
    pass_rate = sum(1 for r in records if r.get("success")) / total
    mean_reward, std_reward = _mean_std(rewards)
    mean_steps, std_steps = _mean_std(steps)
    mean_latency, std_latency = _mean_std(latency)
    return {
        "pass_rate": pass_rate,
        "mean_reward": mean_reward,
        "std_reward": std_reward,
        "mean_steps": mean_steps,
        "std_steps": std_steps,
        "mean_latency_s": mean_latency,
        "std_latency_s": std_latency,
        "episodes": total,
    }


def _group(records: List[dict]) -> Dict[str, List[dict]]:
    grouped: Dict[str, List[dict]] = {}
    for record in records:
        agent = str(record.get("agent", "unknown"))
        grouped.setdefault(agent, []).append(record)
    return grouped


def _group_task_agent(records: List[dict]) -> Dict[str, Dict[str, List[dict]]]:
    grouped: Dict[str, Dict[str, List[dict]]] = {}
    for record in records:
        task = str(record.get("task", "unknown"))
        agent = str(record.get("agent", "unknown"))
        grouped.setdefault(task, {}).setdefault(agent, []).append(record)
    return grouped


def _choose_winner(task_agents: Dict[str, List[dict]]) -> str:
    best_agent = ""
    best_score = (-1.0, -math.inf, math.inf)
    for agent, recs in task_agents.items():
        metrics = _aggregate(recs)
        score = (metrics["pass_rate"], metrics["mean_reward"], -metrics["mean_steps"])
        if score > best_score:
            best_score = score
            best_agent = agent
    return best_agent


def _steps_saved(per_agent: Dict[str, Dict[str, float]]) -> Dict[str, float]:
    random_steps = per_agent.get("random", {}).get("mean_steps", 0.0)
    saved = {}
    for agent, metrics in per_agent.items():
        saved[agent] = random_steps - metrics.get("mean_steps", 0.0)
    return saved


def _render_markdown(summary: dict, win_rates: Dict[str, float], steps_saved: Dict[str, float]) -> str:
    lines = []
    lines.append("# Benchmark Summary")
    lines.append("")
    lines.append("## Aggregate metrics")
    lines.append("| agent | pass_rate | mean_reward | std_reward | mean_steps | std_steps | mean_latency_s | std_latency_s |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for agent, metrics in summary["agents"].items():
        lines.append(
            f"| {agent} | {metrics['pass_rate']:.3f} | {metrics['mean_reward']:.3f} | {metrics['std_reward']:.3f} | "
            f"{metrics['mean_steps']:.3f} | {metrics['std_steps']:.3f} | {metrics['mean_latency_s']:.3f} | {metrics['std_latency_s']:.3f} |"
        )
    lines.append("")
    lines.append("## Win rate per task")
    lines.append("| agent | win_rate |")
    lines.append("|---|---|")
    for agent, rate in win_rates.items():
        lines.append(f"| {agent} | {rate:.3f} |")
    lines.append("")
    lines.append("## Steps saved vs random")
    lines.append("| agent | steps_saved |")
    lines.append("|---|---|")
    for agent, saved in steps_saved.items():
        lines.append(f"| {agent} | {saved:.3f} |")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    if args.input:
        results_path = Path(args.input)
        run_dir = results_path.parent
    elif args.run_dir:
        run_dir = Path(args.run_dir)
        results_path = run_dir / "results.json"
    else:
        raise SystemExit("Provide --run-dir or --input")

    data, records = _load_records(results_path)
    grouped = _group(records)
    per_agent = {agent: _aggregate(recs) for agent, recs in grouped.items()}

    by_task = _group_task_agent(records)
    wins: Dict[str, int] = {}
    for task, agents in by_task.items():
        winner = _choose_winner(agents)
        wins[winner] = wins.get(winner, 0) + 1
    total_tasks = len(by_task) if by_task else 1
    win_rates = {agent: wins.get(agent, 0) / total_tasks for agent in per_agent.keys()}

    steps_saved = _steps_saved(per_agent)

    summary = {
        "config": data.get("config", {}),
        "agents": per_agent,
        "win_rates": win_rates,
        "steps_saved_vs_random": steps_saved,
    }
    summary_path = run_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    md = _render_markdown(summary, win_rates, steps_saved)
    out_path = Path(args.out) if args.out else (run_dir / "summary.md")
    out_path.write_text(md, encoding="utf-8")

    if args.min_pass_rate is not None:
        vagi_rate = per_agent.get("vagi", {}).get("pass_rate", 0.0)
        if vagi_rate < args.min_pass_rate:
            raise SystemExit(f"Regression gate: vagi pass_rate {vagi_rate:.3f} < {args.min_pass_rate:.3f}")
    if args.max_reward_drop is not None:
        vagi_reward = per_agent.get("vagi", {}).get("mean_reward", 0.0)
        baseline_best = max(
            per_agent.get("random", {}).get("mean_reward", 0.0),
            per_agent.get("heuristic", {}).get("mean_reward", 0.0),
        )
        if vagi_reward < baseline_best - args.max_reward_drop:
            raise SystemExit(
                f"Regression gate: vagi mean_reward {vagi_reward:.3f} < baseline {baseline_best:.3f} - {args.max_reward_drop:.3f}"
            )


if __name__ == "__main__":
    main()
