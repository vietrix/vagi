from __future__ import annotations

from vagi_orchestrator.semantic_map import SemanticEpisodeMap


def test_semantic_episode_map_returns_structural_hits() -> None:
    semantic = SemanticEpisodeMap(dim=512)
    semantic.add_episode(
        user_input="design secure login with token refresh",
        draft="validate nonce and rotate refresh token with timeout",
    )
    semantic.add_episode(
        user_input="build ingestion pipeline",
        draft="stream chunks and checkpoint offsets",
    )

    hits = semantic.query("secure auth login nonce", top_k=2, min_score=0.05)
    assert hits
    assert "secure login" in hits[0].summary
