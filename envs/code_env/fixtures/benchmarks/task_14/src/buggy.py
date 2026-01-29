def count_vowels(text: str) -> int:
    vowels = set("ae")
    return sum(1 for ch in text.lower() if ch in vowels)
