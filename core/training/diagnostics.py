"""Forgetting diagnostics utilities."""

from __future__ import annotations

from typing import Dict, Iterable


def aggregate_metrics(records: Iterable[Dict[str, float]], key: str) -> float:
    values = [float(item[key]) for item in records if key in item]
    if not values:
        return 0.0
    return sum(values) / len(values)


def compute_drop(baseline: Dict[str, float], current: Dict[str, float], key: str) -> float:
    if key not in baseline or key not in current:
        raise KeyError(f"Missing key '{key}' in metrics")
    return float(baseline[key]) - float(current[key])


def should_rollback(drop: float, threshold: float) -> bool:
    return float(drop) > float(threshold)
