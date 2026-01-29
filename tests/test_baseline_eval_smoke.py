from __future__ import annotations

import json
import sys
from pathlib import Path

import scripts.eval_baselines as eval_baselines


def test_eval_baselines_smoke(tmp_path) -> None:
    out_path = tmp_path / "baselines.json"
    task_path = Path("envs/code_env/fixtures/mini_repo").as_posix()
    argv = [
        "eval_baselines",
        "--episodes",
        "1",
        "--task",
        task_path,
        "--out",
        out_path.as_posix(),
        "--max-steps",
        "2",
        "--max-run-tests",
        "1",
    ]
    old_argv = sys.argv
    try:
        sys.argv = argv
        eval_baselines.main()
    finally:
        sys.argv = old_argv

    assert out_path.exists()
    data = json.loads(out_path.read_text(encoding="utf-8"))
    assert "agents" in data
    for agent in ["vagi", "random", "heuristic"]:
        metrics = data["agents"][agent]["metrics"]
        assert isinstance(metrics["pass_rate"], float)
        assert isinstance(metrics["mean_reward"], float)
        assert isinstance(metrics["mean_steps"], float)
        assert isinstance(metrics["mean_latency_s"], float)
