from utils.data.schema import RolloutRecord, validate_record


def test_schema_accepts_version_zero() -> None:
    raw = {
        "schema_version": 0,
        "episode_id": "ep-0",
        "timestep": 0,
        "obs": [0.1, 0.2],
        "action": 1,
        "reward": 1.0,
        "done": False,
    }
    record = validate_record(raw)
    assert isinstance(record, RolloutRecord)
    assert record.schema_version == 1


def test_schema_rejects_unknown_version() -> None:
    raw = {
        "schema_version": 3,
        "episode_id": "ep-0",
        "timestep": 0,
        "obs": [0.1, 0.2],
        "action": 1,
        "reward": 1.0,
        "done": False,
    }
    try:
        validate_record(raw)
    except ValueError as exc:
        assert "schema_version" in str(exc)
    else:
        raise AssertionError("Expected schema version check to fail")
