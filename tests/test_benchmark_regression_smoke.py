from __future__ import annotations

import json
import sys
from pathlib import Path

import scripts.run_all_benchmarks as run_all_benchmarks


def test_benchmark_regression_smoke(tmp_path) -> None:
    out_dir = tmp_path / "results"
    tasks_dir = Path("envs/code_env/fixtures/benchmarks").as_posix()
    argv = [
        "run_all_benchmarks",
        "--tasks-dir",
        tasks_dir,
        "--out-dir",
        out_dir.as_posix(),
        "--limit-tasks",
        "1",
        "--seeds",
        "0",
        "--episodes-per-task",
        "1",
        "--max-steps",
        "2",
        "--max-run-tests",
        "1",
    ]
    old_argv = sys.argv
    try:
        sys.argv = argv
        run_all_benchmarks.main()
    finally:
        sys.argv = old_argv

    run_dirs = sorted(out_dir.glob("run_*"))
    assert run_dirs, "No run directory created"
    run_dir = run_dirs[0]
    results_path = run_dir / "results.json"
    csv_path = run_dir / "results.csv"
    info_path = run_dir / "system_info.json"
    assert results_path.exists()
    assert csv_path.exists()
    assert info_path.exists()

    data = json.loads(results_path.read_text(encoding="utf-8"))
    assert "config" in data
    assert "summary" in data
    assert "records" in data
    summary = data["summary"]
    for agent in ["vagi", "random", "heuristic"]:
        assert agent in summary
