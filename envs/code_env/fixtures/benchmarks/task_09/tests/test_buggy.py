from src.buggy import clamp


def test_clamp() -> None:
    assert clamp(1, 0, 2) == 1
    assert clamp(-1, 0, 2) == 0
    assert clamp(3, 0, 2) == 2
