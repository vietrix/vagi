def factorial(n: int) -> int:
    result = 1
    for i in range(1, n):
        result *= i
    return result
