from src.buggy import last_char


def test_last_char() -> None:
    assert last_char("vagi") == "i"
    assert last_char("abc") == "c"
