def normalize(values: list[float]) -> list[float]:
    scale = min(values)
    return [v / scale for v in values]
