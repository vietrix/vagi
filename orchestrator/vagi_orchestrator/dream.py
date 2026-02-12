from __future__ import annotations

import asyncio
import logging
import traceback
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any

from .kernel_client import KernelClient
from .store import EpisodeStore

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class DreamService:
    store: EpisodeStore
    kernel: KernelClient | None = None
    promotion_threshold: float = 0.82
    minimum_pass_rate: float = 0.95
    mutation_enabled: bool = True
    mutation_generations: int = 3
    mutation_population_size: int = 8
    mutation_survivors: int = 2
    mutation_risk_threshold: float = 0.65
    mutation_promote: bool = True

    async def run_once(self, source: str = "manual") -> dict[str, Any]:
        pending = self.store.pending_episodes(limit=500)
        pass_rate = self.store.pass_rate(window=200)
        regression_fail = self.store.regression_fail_count(window=200)
        promoted_ids: list[int] = []
        mutation_runs = 0
        mutation_promotions = 0
        mutation_errors: list[str] = []

        if pass_rate >= self.minimum_pass_rate and regression_fail == 0:
            for episode in pending:
                if (
                    episode["verifier_pass"] == 1
                    and float(episode["trust_score"]) >= self.promotion_threshold
                ):
                    self.store.promote_episode(int(episode["id"]))
                    promoted_ids.append(int(episode["id"]))

        mutation_report, mutation_error = await self._run_mutation_cycle(source=source)
        if mutation_report is not None:
            mutation_runs = 1
            if mutation_report.get("promoted_template_id"):
                mutation_promotions = 1
        if mutation_error:
            mutation_errors.append(mutation_error)

        return {
            "run_id": f"dream-{uuid.uuid4()}",
            "source": source,
            "promoted_count": len(promoted_ids),
            "pass_rate": pass_rate,
            "regression_fail": regression_fail,
            "threshold": self.promotion_threshold,
            "promoted_episode_ids": promoted_ids,
            "mutation_runs": mutation_runs,
            "mutation_promotions": mutation_promotions,
            "mutation_errors": mutation_errors,
            "mutation_report": mutation_report,
        }

    async def _run_mutation_cycle(self, source: str) -> tuple[dict[str, Any] | None, str | None]:
        if not self.mutation_enabled or self.kernel is None:
            return None, None
        evolve_fn = getattr(self.kernel, "evolve_templates", None)
        if evolve_fn is None:
            return None, "kernel_missing_evolve_templates"

        pending = self.store.pending_episodes(limit=200)
        if pending:
            latest = pending[-1]
            query = str(latest.get("user_input", "secure patch")).strip() or "secure patch"
            seed_input = len(str(latest.get("draft", ""))) or 13
        else:
            query = f"{source} secure patch optimization"
            seed_input = 13

        try:
            report = await evolve_fn(
                query=query,
                generations=self.mutation_generations,
                population_size=self.mutation_population_size,
                survivors=self.mutation_survivors,
                risk_threshold=self.mutation_risk_threshold,
                seed_input=max(1, int(seed_input)),
                promote=self.mutation_promote,
            )
            if report is None:
                return None, "mutation_cycle_no_report"
            return report, None
        except Exception:
            trace = traceback.format_exc()
            logger.exception("dream mutation cycle failed")
            return None, trace


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
