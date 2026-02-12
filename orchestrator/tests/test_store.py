from __future__ import annotations

import sqlite3
from pathlib import Path

from vagi_orchestrator.store import EpisodeStore


def test_store_migrates_policy_columns_and_updates_decision(tmp_path: Path) -> None:
    db_path = tmp_path / "episodes.db"
    memory_path = tmp_path / "memory.jsonl"
    store = EpisodeStore(db_path=db_path, long_term_path=memory_path)

    episode_id = store.record_episode(
        session_id="sid-1",
        user_input="test input",
        draft="test draft",
        verifier_pass=True,
        risk_score=0.2,
        trust_score=0.9,
        violations=[],
        source="chat",
    )
    store.attach_policy_decision(
        episode_id=episode_id,
        policy_pass=False,
        policy_violations=[
            {
                "code": "policy_verifier_required",
                "message": "verifier gate failed",
                "stage": "decide",
                "severity": "high",
            }
        ],
        verifier_required=True,
        verifier_pass=False,
        ooda_trace={
            "observe_ok": True,
            "orient_ok": True,
            "decide_ok": False,
            "act_ok": False,
        },
    )
    metrics = store.metrics()
    assert metrics["total_episodes"] == 1
    assert metrics["policy_fail_episodes"] == 1
    store.close()

    conn = sqlite3.connect(db_path)
    cols = {
        row[1]
        for row in conn.execute("PRAGMA table_info(episodes);").fetchall()
    }
    conn.close()
    assert "policy_pass" in cols
    assert "policy_violations" in cols
    assert "ooda_trace" in cols
    assert "verifier_required" in cols
    assert "verifier_gate_pass" in cols

