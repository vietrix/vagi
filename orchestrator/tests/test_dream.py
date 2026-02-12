from __future__ import annotations

import asyncio
from pathlib import Path

from vagi_orchestrator.dream import DreamService
from vagi_orchestrator.store import EpisodeStore


class FakeKernelMutation:
    async def evolve_templates(self, **kwargs) -> dict:
        return {
            "base_template_id": "python_secure_v1",
            "promoted_template_id": "python_secure_v1_mut_abcd1234",
            "generations_run": kwargs.get("generations", 3),
            "population_size": kwargs.get("population_size", 8),
            "survivors": kwargs.get("survivors", 2),
        }


class FakeKernelMutationError:
    async def evolve_templates(self, **kwargs) -> dict:
        raise RuntimeError("mutation boom")


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


def test_dream_triggers_mutation_cycle_when_kernel_available(tmp_path: Path) -> None:
    store = EpisodeStore(
        db_path=tmp_path / "episodes.db",
        long_term_path=tmp_path / "memory.jsonl",
    )
    store.record_episode(
        session_id="sid-1",
        user_input="write secure python login",
        draft="add 5\nmul 2\nxor 3",
        verifier_pass=True,
        risk_score=0.2,
        trust_score=0.9,
        violations=[],
        source="chat",
    )
    dream = DreamService(
        store=store,
        kernel=FakeKernelMutation(),  # type: ignore[arg-type]
        mutation_enabled=True,
        mutation_generations=2,
        mutation_population_size=4,
        mutation_survivors=2,
    )
    report = asyncio.run(dream.run_once(source="manual"))
    assert report["mutation_runs"] == 1
    assert report["mutation_promotions"] == 1
    assert report["mutation_report"]["promoted_template_id"] is not None
    store.close()


def test_dream_reports_traceback_when_mutation_cycle_fails(tmp_path: Path) -> None:
    store = EpisodeStore(
        db_path=tmp_path / "episodes.db",
        long_term_path=tmp_path / "memory.jsonl",
    )
    dream = DreamService(
        store=store,
        kernel=FakeKernelMutationError(),  # type: ignore[arg-type]
        mutation_enabled=True,
    )
    report = asyncio.run(dream.run_once(source="manual"))
    assert report["mutation_runs"] == 0
    assert report["mutation_promotions"] == 0
    assert report["mutation_errors"]
    assert "RuntimeError: mutation boom" in report["mutation_errors"][0]
    store.close()
