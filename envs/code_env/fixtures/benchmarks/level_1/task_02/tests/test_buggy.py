from src.buggy import subtract


def test_subtract() -> None:
    assert subtract(5, 3) == 2
    assert subtract(0, 4) == -4
