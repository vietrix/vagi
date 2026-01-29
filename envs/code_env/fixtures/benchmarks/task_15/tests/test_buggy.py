from src.buggy import unique_count


def test_unique_count() -> None:
    assert unique_count([1, 1, 2]) == 2
    assert unique_count([3, 3, 3]) == 1
