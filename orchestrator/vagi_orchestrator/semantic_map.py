from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass

_TOKEN_RE = re.compile(r"[a-zA-Z0-9_]{2,}")


def _tokenize(text: str) -> list[str]:
    return [token.lower() for token in _TOKEN_RE.findall(text)]


def _hash_index(token: str, dim: int) -> int:
    digest = hashlib.sha256(token.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % dim


def _hash_sign(token: str) -> int:
    digest = hashlib.sha256(f"{token}:sign".encode("utf-8")).digest()
    return 1 if digest[0] % 2 == 0 else -1


def _hypervector(text: str, dim: int = 1024) -> list[int]:
    vector = [0] * dim
    for idx, token in enumerate(_tokenize(text)):
        pos_idx = _hash_index(f"tok:{token}", dim)
        seq_idx = _hash_index(f"seq:{idx % 32}:{token}", dim)
        vector[pos_idx] += _hash_sign(token)
        vector[seq_idx] += _hash_sign(f"seq:{token}")
    return vector


def _bundle(vectors: list[list[int]]) -> list[int]:
    if not vectors:
        return []
    dim = len(vectors[0])
    out = [0] * dim
    for vector in vectors:
        for i in range(dim):
            out[i] += vector[i]
    return out


def _similarity(a: list[int], b: list[int]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = sum(x * x for x in a) ** 0.5
    nb = sum(y * y for y in b) ** 0.5
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(dot / (na * nb))


@dataclass(slots=True, frozen=True)
class SemanticHit:
    summary: str
    score: float


class SemanticEpisodeMap:
    def __init__(self, dim: int = 1024) -> None:
        self.dim = dim
        self._episodes: list[tuple[list[int], str]] = []

    def add_episode(self, *, user_input: str, draft: str) -> None:
        hv = _bundle([_hypervector(user_input, self.dim), _hypervector(draft, self.dim)])
        summary = f"user={user_input.strip()} | draft={draft.strip()[:220]}"
        self._episodes.append((hv, summary))

    def query(self, text: str, top_k: int = 3, min_score: float = 0.08) -> list[SemanticHit]:
        probe = _hypervector(text, self.dim)
        scored: list[SemanticHit] = []
        for hv, summary in self._episodes:
            score = _similarity(probe, hv)
            if score >= min_score:
                scored.append(SemanticHit(summary=summary, score=score))
        scored.sort(key=lambda item: item.score, reverse=True)
        return scored[:top_k]
