import json
from pathlib import Path

from scripts.golden_runs import run_golden


def test_golden_runs_match() -> None:
    base = Path("tests/fixtures/golden")
    assert base.exists()
    for path in sorted(base.glob("seed_*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        seed = int(payload["seed"])
        expected = run_golden(seed, episodes=int(payload["episodes"]), steps=12)
        for key, value in expected.items():
            assert key in payload
            assert abs(float(payload[key]) - float(value)) < 1e-6
