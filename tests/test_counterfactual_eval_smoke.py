import json
import sys
from pathlib import Path

import scripts.counterfactual_eval as counterfactual_eval


def test_counterfactual_eval_smoke(tmp_path: Path) -> None:
    out_path = tmp_path / "counterfactual.jsonl"
    argv = [
        "counterfactual_eval",
        "--out",
        out_path.as_posix(),
        "--episodes",
        "1",
        "--steps",
        "2",
    ]
    old_argv = sys.argv
    try:
        sys.argv = argv
        counterfactual_eval.main()
    finally:
        sys.argv = old_argv

    assert out_path.exists()
    line = out_path.read_text(encoding="utf-8").splitlines()[0]
    payload = json.loads(line)
    for key in [
        "uncertainty",
        "confidence",
        "value_spread",
        "delta_reward",
        "delta_latency",
    ]:
        assert isinstance(payload[key], float)
