from src.buggy import is_palindrome


def test_is_palindrome() -> None:
    assert is_palindrome("Level") is True
    assert is_palindrome("vagi") is False
