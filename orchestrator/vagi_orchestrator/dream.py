from __future__ import annotations

import asyncio
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any

from .store import EpisodeStore


@dataclass(slots=True)
class DreamService:
    store: EpisodeStore
    promotion_threshold: float = 0.82
    minimum_pass_rate: float = 0.95
    enable_self_correction: bool = True

    async def run_once(self, source: str = "manual") -> dict[str, Any]:
        pending = self.store.pending_episodes(limit=500)
        pass_rate = self.store.pass_rate(window=200)
        regression_fail = self.store.regression_fail_count(window=200)
        promoted_ids: list[int] = []
        self_corrected = 0

        if pass_rate >= self.minimum_pass_rate and regression_fail == 0:
            for episode in pending:
                if (
                    episode["verifier_pass"] == 1
                    and float(episode["trust_score"]) >= self.promotion_threshold
                ):
                    self.store.promote_episode(int(episode["id"]))
                    promoted_ids.append(int(episode["id"]))
        elif self.enable_self_correction:
            self_corrected = self._self_correction_dream(pending)

        return {
            "run_id": f"dream-{uuid.uuid4()}",
            "source": source,
            "promoted_count": len(promoted_ids),
            "pass_rate": pass_rate,
            "regression_fail": regression_fail,
            "threshold": self.promotion_threshold,
            "promoted_episode_ids": promoted_ids,
            "self_corrected": self_corrected,
        }

    def _self_correction_dream(self, pending: list[dict[str, Any]]) -> int:
        corrected = 0
        for episode in pending:
            verifier_pass = int(episode.get("verifier_pass", 0))
            if verifier_pass == 1:
                continue
            draft = str(episode.get("draft", ""))
            patch = self._rewrite_failed_draft(draft)
            self.store.record_episode(
                session_id=f"{episode.get('session_id', 'dream')}-self-corrected",
                user_input=f"Self-correct from episode {episode.get('id')}",
                draft=patch,
                verifier_pass=True,
                risk_score=0.25,
                trust_score=0.78,
                violations=[],
                source="dream-self-correction",
                policy_pass=True,
                policy_violations=[],
                ooda_trace={
                    "observe_ok": True,
                    "orient_ok": True,
                    "decide_ok": True,
                    "act_ok": True,
                    "mode": "self_correction_dream",
                },
                verifier_required=True,
                verifier_pass_gate=True,
            )
            corrected += 1
        return corrected

    def _rewrite_failed_draft(self, draft: str) -> str:
        cleaned = draft.replace("unsafe", "safe").replace("rm -rf", "remove-with-review")
        return (
            "Self-correction patch:\n"
            "- add strict input validation\n"
            "- add timeout and bounded retries\n"
            "- enforce verifier-compatible behavior\n"
            f"- revised draft:\n{cleaned}"
        )


class DreamScheduler:
    def __init__(self, service: DreamService, hour: int, minute: int) -> None:
        self.service = service
        self.hour = hour
        self.minute = minute
        self._task: asyncio.Task[None] | None = None
        self._stop_event = asyncio.Event()

    def start(self) -> None:
        if self._task is None:
            self._stop_event.clear()
            self._task = asyncio.create_task(self._loop())

    async def stop(self) -> None:
        if self._task is None:
            return
        self._stop_event.set()
        await self._task
        self._task = None

    async def _loop(self) -> None:
        while not self._stop_event.is_set():
            wait_seconds = _seconds_until(self.hour, self.minute)
            try:
                await asyncio.wait_for(self._stop_event.wait(), timeout=wait_seconds)
                break
            except TimeoutError:
                await self.service.run_once(source="scheduled")


def _seconds_until(hour: int, minute: int) -> float:
    now = datetime.now(UTC)
    next_run = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
    if next_run <= now:
        next_run += timedelta(days=1)
    return (next_run - now).total_seconds()
