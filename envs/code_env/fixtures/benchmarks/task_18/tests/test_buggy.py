from src.buggy import clip01


def test_clip01() -> None:
    assert clip01(-0.5) == 0.0
    assert clip01(1.5) == 1.0
