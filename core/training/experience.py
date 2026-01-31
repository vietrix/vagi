"""Experience buffer with quality gating."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


@dataclass
class QualityGate:
    min_reward: float = 0.0
    max_uncertainty: float = float("inf")
    min_validity: float = 0.0
    require_metrics: bool = True

    def accept(self, reward: Optional[float], uncertainty: Optional[float], validity: Optional[float]) -> bool:
        if reward is None:
            return False
        if self.require_metrics and (uncertainty is None or validity is None):
            return False
        uncertainty = 0.0 if uncertainty is None else float(uncertainty)
        validity = 1.0 if validity is None else float(validity)
        return (
            float(reward) >= self.min_reward
            and float(uncertainty) <= self.max_uncertainty
            and float(validity) >= self.min_validity
        )


@dataclass
class ExperienceRecord:
    data: Dict[str, Any]
    reward: float
    uncertainty: Optional[float]
    validity: Optional[float]


@dataclass
class ExperienceBuffer:
    max_size: int = 10000
    gate: QualityGate = field(default_factory=QualityGate)
    records: List[ExperienceRecord] = field(default_factory=list)
    accepted: int = 0
    rejected: int = 0

    def __len__(self) -> int:
        return len(self.records)

    def _extract_metrics(self, record: Dict[str, Any]) -> Tuple[Optional[float], Optional[float], Optional[float]]:
        reward = record.get("reward")
        info = record.get("info") or {}
        uncertainty = record.get("uncertainty", info.get("uncertainty"))
        validity = record.get("validity", info.get("validity"))
        return reward, uncertainty, validity

    def add(self, record: Dict[str, Any]) -> bool:
        reward, uncertainty, validity = self._extract_metrics(record)
        if not self.gate.accept(reward, uncertainty, validity):
            self.rejected += 1
            return False
        exp = ExperienceRecord(
            data=record,
            reward=float(reward),
            uncertainty=None if uncertainty is None else float(uncertainty),
            validity=None if validity is None else float(validity),
        )
        self.records.append(exp)
        if len(self.records) > self.max_size:
            self.records.pop(0)
        self.accepted += 1
        return True

    def add_many(self, records: Iterable[Dict[str, Any]]) -> int:
        kept = 0
        for record in records:
            if self.add(record):
                kept += 1
        return kept

    def sample(self, count: int, *, seed: Optional[int] = None) -> List[ExperienceRecord]:
        if count <= 0:
            return []
        if count >= len(self.records):
            return list(self.records)
        rng = None
        if seed is not None:
            import random

            rng = random.Random(seed)
            indices = rng.sample(range(len(self.records)), count)
        else:
            import random

            indices = random.sample(range(len(self.records)), count)
        return [self.records[i] for i in indices]

    def to_jsonl(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            for record in self.records:
                handle.write(json.dumps(record.data) + "\n")

    def filter_jsonl(self, input_path: str | Path, output_path: str | Path) -> Dict[str, int]:
        input_path = Path(input_path)
        if not input_path.exists():
            raise FileNotFoundError(f"Missing file: {input_path}")
        episodes: Dict[str, List[Dict[str, Any]]] = {}
        for line in input_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            raw = json.loads(line)
            episode_id = str(raw.get("episode_id", f"single-{len(episodes)}-{len(episodes)}"))
            episodes.setdefault(episode_id, []).append(raw)

        accepted = 0
        rejected = 0
        filtered: List[Dict[str, Any]] = []
        for records in episodes.values():
            rewards = []
            uncertainties = []
            validities = []
            for record in records:
                reward, uncertainty, validity = self._extract_metrics(record)
                rewards.append(reward)
                uncertainties.append(uncertainty)
                validities.append(validity)
            avg_reward = None if any(r is None for r in rewards) else float(sum(rewards) / max(len(rewards), 1))
            avg_uncert = None
            if not any(u is None for u in uncertainties):
                avg_uncert = float(sum(uncertainties) / max(len(uncertainties), 1))
            avg_valid = None
            if not any(v is None for v in validities):
                avg_valid = float(sum(validities) / max(len(validities), 1))
            if self.gate.accept(avg_reward, avg_uncert, avg_valid):
                filtered.extend(records)
                accepted += 1
            else:
                rejected += 1

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as handle:
            for record in filtered:
                handle.write(json.dumps(record) + "\n")
        return {"episodes_kept": accepted, "episodes_rejected": rejected, "records_kept": len(filtered)}
