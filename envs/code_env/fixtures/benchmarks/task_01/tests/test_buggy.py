from src.buggy import add


def test_add() -> None:
    assert add(2, 3) == 5
    assert add(-1, 1) == 0
