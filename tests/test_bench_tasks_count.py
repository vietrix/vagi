from pathlib import Path

from scripts.bench_utils import collect_tasks


def test_benchmark_tasks_count() -> None:
    tasks_root = Path("envs/code_env/fixtures/benchmarks")
    tasks = collect_tasks(tasks_root)
    assert len(tasks) >= 20
