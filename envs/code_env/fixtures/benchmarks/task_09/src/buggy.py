def clamp(x: float, lo: float, hi: float) -> float:
    if x < lo:
        return hi
    if x > hi:
        return lo
    return x
