from src.buggy import factorial


def test_factorial() -> None:
    assert factorial(1) == 1
    assert factorial(4) == 24
