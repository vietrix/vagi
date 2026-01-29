from src.buggy import mean


def test_mean() -> None:
    assert mean([2, 4]) == 3
    assert mean([1, 2, 3]) == 2
