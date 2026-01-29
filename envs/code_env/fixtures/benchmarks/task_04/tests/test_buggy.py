from src.buggy import divide


def test_divide() -> None:
    assert divide(6, 2) == 3
    assert divide(9, 3) == 3
