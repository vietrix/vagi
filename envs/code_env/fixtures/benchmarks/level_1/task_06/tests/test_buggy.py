from src.buggy import is_even


def test_is_even() -> None:
    assert is_even(2) is True
    assert is_even(3) is False
