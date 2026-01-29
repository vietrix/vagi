def dot(a: list[int], b: list[int]) -> int:
    total = 0
    for x, y in zip(a, b):
        total -= x * y
    return total
