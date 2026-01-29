from src.buggy import count_vowels


def test_count_vowels() -> None:
    assert count_vowels("vagi") == 2
    assert count_vowels("queue") == 4
