"""Generate curriculum splits from JSONL rollouts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

from utils.data.reader import read_jsonl
from utils.data.schema import RolloutRecord


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate curriculum splits from rollouts.")
    parser.add_argument("--data", type=str, default="logs/rollouts.jsonl")
    parser.add_argument("--out-dir", type=str, default="runs/curriculum")
    parser.add_argument("--steps-weight", type=float, default=1.0)
    parser.add_argument("--uncertainty-weight", type=float, default=1.0)
    parser.add_argument("--fail-weight", type=float, default=1.0)
    return parser.parse_args()


def _episode_metrics(episode: List[RolloutRecord]) -> Dict[str, float]:
    steps = len(episode)
    total_uncertainty = 0.0
    count_uncertainty = 0
    max_fail = 0.0
    for record in episode:
        if record.info:
            if "uncertainty" in record.info:
                total_uncertainty += float(record.info["uncertainty"])
                count_uncertainty += 1
            if "fail_count" in record.info:
                max_fail = max(max_fail, float(record.info["fail_count"]))
    mean_uncertainty = total_uncertainty / max(count_uncertainty, 1)
    return {
        "steps": float(steps),
        "mean_uncertainty": float(mean_uncertainty),
        "fail_count": float(max_fail),
    }


def _collect_episode_metrics(path: str) -> Dict[str, Dict[str, float]]:
    metrics: Dict[str, Dict[str, float]] = {}
    episode: List[RolloutRecord] = []
    current_id: str | None = None
    for record in read_jsonl(path):
        if current_id is None:
            current_id = record.episode_id
        if record.episode_id != current_id and episode:
            metrics[current_id] = _episode_metrics(episode)
            episode = []
            current_id = record.episode_id
        episode.append(record)
        if record.done:
            metrics[current_id] = _episode_metrics(episode)
            episode = []
            current_id = None
    if episode and current_id is not None:
        metrics[current_id] = _episode_metrics(episode)
    return metrics


def _score_episode(metrics: Dict[str, float], weights: Tuple[float, float, float]) -> float:
    steps_w, unc_w, fail_w = weights
    return (
        steps_w * metrics["steps"]
        + unc_w * metrics["mean_uncertainty"]
        + fail_w * metrics["fail_count"]
    )


def _assign_levels(metrics: Dict[str, Dict[str, float]], weights: Tuple[float, float, float]) -> Dict[str, int]:
    scores = [(episode_id, _score_episode(data, weights)) for episode_id, data in metrics.items()]
    scores.sort(key=lambda item: item[1])
    total = len(scores)
    if total == 0:
        return {}
    first_cut = max(1, total // 3)
    second_cut = max(first_cut + 1, (2 * total) // 3)
    levels: Dict[str, int] = {}
    for idx, (episode_id, _) in enumerate(scores):
        if idx < first_cut:
            level = 1
        elif idx < second_cut:
            level = 2
        else:
            level = 3
        levels[episode_id] = level
    return levels


def _record_to_json(record: RolloutRecord) -> Dict[str, object]:
    payload: Dict[str, object] = {
        "schema_version": record.schema_version,
        "episode_id": record.episode_id,
        "timestep": record.timestep,
        "obs": record.obs,
        "action": record.action,
        "reward": record.reward,
        "done": record.done,
    }
    if record.obs_next is not None:
        payload["obs_next"] = record.obs_next
    if record.return_ is not None:
        payload["return"] = record.return_
    if record.value is not None:
        payload["value"] = record.value
    if record.task is not None:
        payload["task"] = record.task
    if record.success is not None:
        payload["success"] = record.success
    if record.info is not None:
        payload["info"] = record.info
    return payload


def main() -> None:
    args = parse_args()
    weights = (args.steps_weight, args.uncertainty_weight, args.fail_weight)
    metrics = _collect_episode_metrics(args.data)
    levels = _assign_levels(metrics, weights)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    level_paths = {
        1: out_dir / "level_1.jsonl",
        2: out_dir / "level_2.jsonl",
        3: out_dir / "level_3.jsonl",
    }
    handles = {lvl: path.open("w", encoding="utf-8") for lvl, path in level_paths.items()}

    try:
        for record in read_jsonl(args.data):
            level = levels.get(record.episode_id, 2)
            payload = _record_to_json(record)
            handles[level].write(json.dumps(payload) + "\n")
    finally:
        for handle in handles.values():
            handle.close()

    summary = []
    for episode_id, data in metrics.items():
        summary.append(
            {
                "episode_id": episode_id,
                "level": levels.get(episode_id, 2),
                "steps": data["steps"],
                "mean_uncertainty": data["mean_uncertainty"],
                "fail_count": data["fail_count"],
                "score": _score_episode(data, weights),
            }
        )
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
