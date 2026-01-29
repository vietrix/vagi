from src.buggy import sum_list


def test_sum_list() -> None:
    assert sum_list([1, 2, 3]) == 6
    assert sum_list([-1, 1]) == 0
