from __future__ import annotations

import asyncio
from pathlib import Path

from vagi_orchestrator.dream import DreamService
from vagi_orchestrator.store import EpisodeStore


def test_dream_promotes_when_gate_passes(tmp_path: Path) -> None:
    store = EpisodeStore(
        db_path=tmp_path / "episodes.db",
        long_term_path=tmp_path / "memory.jsonl",
    )
    for idx in range(5):
        store.record_episode(
            session_id=f"sid-{idx}",
            user_input="input",
            draft="draft",
            verifier_pass=True,
            risk_score=0.2,
            trust_score=0.9,
            violations=[],
            source="chat",
        )
    dream = DreamService(store=store, promotion_threshold=0.82, minimum_pass_rate=0.95)
    report = asyncio.run(dream.run_once(source="manual"))
    assert report["promoted_count"] == 5
    assert store.metrics()["promoted_episodes"] == 5
    store.close()


def test_dream_blocks_promotion_on_regression(tmp_path: Path) -> None:
    store = EpisodeStore(
        db_path=tmp_path / "episodes.db",
        long_term_path=tmp_path / "memory.jsonl",
    )
    store.record_episode(
        session_id="sid-good",
        user_input="input",
        draft="draft",
        verifier_pass=True,
        risk_score=0.2,
        trust_score=0.9,
        violations=[],
        source="chat",
    )
    store.record_episode(
        session_id="sid-bad",
        user_input="input",
        draft="drop everything",
        verifier_pass=False,
        risk_score=0.9,
        trust_score=0.4,
        violations=["unsafe"],
        source="chat",
    )
    dream = DreamService(store=store, promotion_threshold=0.82, minimum_pass_rate=0.95)
    report = asyncio.run(dream.run_once(source="manual"))
    assert report["promoted_count"] == 0
    store.close()


def test_dream_self_correction_creates_repaired_episode(tmp_path: Path) -> None:
    store = EpisodeStore(
        db_path=tmp_path / "episodes.db",
        long_term_path=tmp_path / "memory.jsonl",
    )
    store.record_episode(
        session_id="sid-bad",
        user_input="input",
        draft="unsafe rm -rf",
        verifier_pass=False,
        risk_score=0.9,
        trust_score=0.3,
        violations=["unsafe"],
        source="chat",
    )
    dream = DreamService(
        store=store,
        promotion_threshold=0.82,
        minimum_pass_rate=0.95,
        enable_self_correction=True,
    )
    report = asyncio.run(dream.run_once(source="manual"))
    assert report["self_corrected"] >= 1
    assert store.metrics()["total_episodes"] >= 2
    store.close()
