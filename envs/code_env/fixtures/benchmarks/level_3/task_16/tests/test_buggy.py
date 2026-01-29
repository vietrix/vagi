from src.buggy import normalize


def test_normalize() -> None:
    assert normalize([2, 4]) == [0.5, 1.0]
    assert normalize([1, 3]) == [0.3333333333333333, 1.0]
