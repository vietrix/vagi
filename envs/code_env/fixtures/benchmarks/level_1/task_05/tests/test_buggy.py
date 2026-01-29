from src.buggy import reverse_string


def test_reverse_string() -> None:
    assert reverse_string("abc") == "cba"
    assert reverse_string("vagi") == "igav"
