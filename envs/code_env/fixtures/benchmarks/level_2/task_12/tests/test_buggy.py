from src.buggy import fib


def test_fib() -> None:
    assert fib(2) == 1
    assert fib(5) == 5
