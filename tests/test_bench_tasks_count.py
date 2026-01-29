from pathlib import Path


def test_benchmark_tasks_count() -> None:
    tasks_root = Path("envs/code_env/fixtures/benchmarks")
    tasks = [p for p in tasks_root.iterdir() if p.is_dir()]
    assert len(tasks) >= 20
