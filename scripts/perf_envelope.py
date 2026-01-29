"""Generate performance envelope metrics and charts."""

from __future__ import annotations

import argparse
import csv
import math
import time
from pathlib import Path
from typing import List, Tuple

import torch

from vagi_core import VAGIConfig, VAGICore


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate performance envelope metrics.")
    parser.add_argument("--out-dir", type=str, default="results/perf")
    parser.add_argument("--batches", type=str, default="1,2,4")
    parser.add_argument("--kv-lens", type=str, default="0,16,32,64")
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=5)
    return parser.parse_args()


def _estimate_kv_bytes(cfg: VAGIConfig, batch: int, kv_len: int) -> int:
    if kv_len <= 0:
        return 0
    n_kv = cfg.n_kv_heads if cfg.use_gqa else cfg.n_heads
    head_dim = cfg.head_dim
    bytes_per = 4  # float32
    per_layer = batch * n_kv * kv_len * head_dim * 2 * bytes_per
    return per_layer * cfg.n_layers


def _benchmark_act(model: VAGICore, batch: int, steps: int, warmup: int) -> float:
    obs = torch.zeros((batch, model.cfg.obs_dim))
    state = model.init_state(batch)
    input_ids = torch.zeros((batch, 1), dtype=torch.long)
    for _ in range(warmup):
        _ = model.act(input_ids=input_ids, obs=obs, state=state)
    start = time.perf_counter()
    for _ in range(steps):
        _ = model.act(input_ids=input_ids, obs=obs, state=state)
    return (time.perf_counter() - start) * 1000.0 / max(steps, 1)


def _benchmark_think(model: VAGICore, batch: int, steps: int, warmup: int) -> float:
    obs = torch.zeros((batch, model.cfg.obs_dim))
    state = model.init_state(batch)
    input_ids = torch.zeros((batch, 1), dtype=torch.long)
    for _ in range(warmup):
        _ = model.think_then_act(input_ids=input_ids, obs=obs, state=state, horizon=2, num_candidates=4)
    start = time.perf_counter()
    for _ in range(steps):
        _ = model.think_then_act(input_ids=input_ids, obs=obs, state=state, horizon=2, num_candidates=4)
    return (time.perf_counter() - start) * 1000.0 / max(steps, 1)


def _svg_line_chart(points: List[Tuple[float, float]], title: str, x_label: str, y_label: str) -> str:
    width, height = 640, 360
    padding = 50
    if not points:
        return ""
    xs, ys = zip(*points)
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    if math.isclose(min_x, max_x):
        max_x += 1.0
    if math.isclose(min_y, max_y):
        max_y += 1.0

    def _scale_x(x: float) -> float:
        return padding + (x - min_x) / (max_x - min_x) * (width - 2 * padding)

    def _scale_y(y: float) -> float:
        return height - padding - (y - min_y) / (max_y - min_y) * (height - 2 * padding)

    path = " ".join(
        f"L{_scale_x(x):.2f},{_scale_y(y):.2f}" if idx else f"M{_scale_x(x):.2f},{_scale_y(y):.2f}"
        for idx, (x, y) in enumerate(points)
    )
    svg = f"""
<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">
  <rect width="100%" height="100%" fill="white"/>
  <text x="{width/2}" y="24" text-anchor="middle" font-size="14">{title}</text>
  <text x="{width/2}" y="{height-10}" text-anchor="middle" font-size="12">{x_label}</text>
  <text x="14" y="{height/2}" text-anchor="middle" font-size="12" transform="rotate(-90 14 {height/2})">{y_label}</text>
  <path d="{path}" fill="none" stroke="#1f77b4" stroke-width="2"/>
</svg>
"""
    return svg.strip()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = VAGIConfig(
        vocab_size=32,
        hidden_size=64,
        n_layers=2,
        n_heads=4,
        n_kv_heads=4,
        mlp_ratio=2.0,
        max_seq_len=8,
        obs_dim=16,
        obs_tokens=2,
        action_dim=8,
        memory_slots=4,
        dropout=0.0,
        use_world_pred=True,
        world_model_horizon=2,
        use_uncertainty=True,
        use_budget_head=True,
    )
    model = VAGICore(cfg)
    model.eval()

    batches = [int(x) for x in args.batches.split(",") if x.strip()]
    kv_lens = [int(x) for x in args.kv_lens.split(",") if x.strip()]

    rows = [("scenario", "batch", "kv_len", "latency_ms", "kv_mem_kb")]
    for batch in batches:
        latency_act = _benchmark_act(model, batch, args.steps, args.warmup)
        latency_think = _benchmark_think(model, batch, args.steps, args.warmup)
        rows.append(("act", batch, 0, f"{latency_act:.3f}", "0"))
        rows.append(("think", batch, 0, f"{latency_think:.3f}", "0"))

    for kv_len in kv_lens:
        kv_bytes = _estimate_kv_bytes(cfg, batch=1, kv_len=kv_len)
        rows.append(("kv_mem", 1, kv_len, "0.0", f"{kv_bytes / 1024:.2f}"))

    csv_path = out_dir / "perf.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        for row in rows:
            writer.writerow(row)

    act_points = [(float(row[1]), float(row[3])) for row in rows if row[0] == "act"]
    think_points = [(float(row[1]), float(row[3])) for row in rows if row[0] == "think"]
    kv_points = [(float(row[2]), float(row[4])) for row in rows if row[0] == "kv_mem"]

    (out_dir / "latency_act.svg").write_text(
        _svg_line_chart(act_points, "Latency vs Batch (act)", "batch", "ms"), encoding="utf-8"
    )
    (out_dir / "latency_think.svg").write_text(
        _svg_line_chart(think_points, "Latency vs Batch (think)", "batch", "ms"), encoding="utf-8"
    )
    (out_dir / "kv_memory.svg").write_text(
        _svg_line_chart(kv_points, "KV memory vs kv_len", "kv_len", "KB"), encoding="utf-8"
    )
    print(f"Wrote {csv_path}")


if __name__ == "__main__":
    main()
