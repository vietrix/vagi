from src.buggy import dot


def test_dot() -> None:
    assert dot([1, 2], [3, 4]) == 11
