from src.buggy import shout


def test_shout() -> None:
    assert shout("vagi") == "VAGI"
    assert shout("Hello") == "HELLO"
