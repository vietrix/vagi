def clip01(x: float) -> float:
    if x < 0:
        return 1.0
    if x > 1:
        return 0.0
    return x
