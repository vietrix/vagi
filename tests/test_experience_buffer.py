import json
from pathlib import Path

from vagi_core.experience import ExperienceBuffer, QualityGate


def test_experience_quality_gate(tmp_path: Path) -> None:
    gate = QualityGate(min_reward=1.0, max_uncertainty=0.5, min_validity=0.8, require_metrics=True)
    buffer = ExperienceBuffer(max_size=10, gate=gate)

    good = {"reward": 1.0, "info": {"uncertainty": 0.2, "validity": 0.9}}
    bad_reward = {"reward": 0.0, "info": {"uncertainty": 0.1, "validity": 0.9}}
    bad_uncert = {"reward": 1.0, "info": {"uncertainty": 0.9, "validity": 0.9}}
    bad_valid = {"reward": 1.0, "info": {"uncertainty": 0.1, "validity": 0.1}}
    missing = {"reward": 1.0}

    assert buffer.add(good) is True
    assert buffer.add(bad_reward) is False
    assert buffer.add(bad_uncert) is False
    assert buffer.add(bad_valid) is False
    assert buffer.add(missing) is False
    assert len(buffer) == 1

    input_path = tmp_path / "input.jsonl"
    records = [
        {"episode_id": "a", "reward": 1.0, "info": {"uncertainty": 0.1, "validity": 0.9}},
        {"episode_id": "a", "reward": 1.0, "info": {"uncertainty": 0.1, "validity": 0.9}},
        {"episode_id": "b", "reward": 0.0, "info": {"uncertainty": 0.1, "validity": 0.9}},
    ]
    input_path.write_text("\n".join(json.dumps(r) for r in records) + "\n", encoding="utf-8")
    output_path = tmp_path / "output.jsonl"
    stats = buffer.filter_jsonl(input_path, output_path)
    assert stats["episodes_kept"] == 1
    assert stats["records_kept"] == 2
