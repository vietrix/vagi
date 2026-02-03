"""Experience buffer with quality gating."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


@dataclass
class QualityGate:
    """Quality gate for filtering experiences using weighted scoring (Issue 7.9).

    Instead of using all() with hard thresholds, this uses weighted scoring
    to compute a quality score that must exceed a threshold. This allows
    experiences that are excellent in some dimensions but borderline in
    others to still pass through.

    Attributes:
        min_reward: Minimum reward threshold (used in scoring).
        max_uncertainty: Maximum uncertainty threshold (used in scoring).
        min_validity: Minimum validity threshold (used in scoring).
        require_metrics: If True, missing metrics result in rejection.
        use_weighted_scoring: If True, use weighted score; if False, use legacy all().
        quality_threshold: Minimum quality score to accept (0-1).
        reward_weight: Weight for reward component in quality score.
        uncertainty_weight: Weight for uncertainty component in quality score.
        validity_weight: Weight for validity component in quality score.
    """
    min_reward: float = 0.0
    max_uncertainty: float = float("inf")
    min_validity: float = 0.0
    require_metrics: bool = True
    # Issue 7.9: Weighted scoring configuration
    use_weighted_scoring: bool = True
    quality_threshold: float = 0.5
    reward_weight: float = 0.4
    uncertainty_weight: float = 0.3
    validity_weight: float = 0.3

    def _compute_quality_score(
        self,
        reward: float,
        uncertainty: float,
        validity: float
    ) -> float:
        """Compute weighted quality score (Issue 7.9).

        Each component is normalized to [0, 1] and combined with weights.
        This replaces the hard all() logic with soft scoring.

        Args:
            reward: Reward value.
            uncertainty: Uncertainty value (lower is better).
            validity: Validity value (higher is better).

        Returns:
            Quality score in [0, 1].
        """
        # Normalize reward: score is 1.0 if reward >= min_reward, scaling down below
        if self.min_reward > 0:
            reward_score = min(1.0, max(0.0, reward / self.min_reward))
        else:
            # If min_reward is 0 or negative, any positive reward is good
            reward_score = 1.0 if reward >= self.min_reward else 0.5 + 0.5 * (reward / abs(self.min_reward + 1e-8))
            reward_score = max(0.0, min(1.0, reward_score))

        # Normalize uncertainty: score is 1.0 if uncertainty <= max_uncertainty
        # Linear decay from 1.0 to 0.0 as uncertainty goes from max to 2*max
        if self.max_uncertainty < float("inf"):
            if uncertainty <= self.max_uncertainty:
                uncertainty_score = 1.0
            else:
                overshoot = (uncertainty - self.max_uncertainty) / (self.max_uncertainty + 1e-8)
                uncertainty_score = max(0.0, 1.0 - overshoot)
        else:
            # No uncertainty constraint
            uncertainty_score = 1.0

        # Normalize validity: score is 1.0 if validity >= min_validity
        if self.min_validity > 0:
            if validity >= self.min_validity:
                validity_score = 1.0
            else:
                validity_score = validity / self.min_validity
        else:
            validity_score = min(1.0, validity)

        # Weighted combination
        total_weight = self.reward_weight + self.uncertainty_weight + self.validity_weight
        if total_weight == 0:
            return 0.5  # Default if all weights are zero

        quality = (
            self.reward_weight * reward_score +
            self.uncertainty_weight * uncertainty_score +
            self.validity_weight * validity_score
        ) / total_weight

        return quality

    def accept(
        self,
        reward: Optional[float],
        uncertainty: Optional[float],
        validity: Optional[float]
    ) -> bool:
        """Determine if experience should be accepted.

        Issue 7.9: Uses weighted scoring instead of all() when use_weighted_scoring=True.

        Args:
            reward: Reward value (required).
            uncertainty: Uncertainty value (optional).
            validity: Validity value (optional).

        Returns:
            True if experience passes quality gate.
        """
        if reward is None:
            return False
        if self.require_metrics and (uncertainty is None or validity is None):
            return False

        uncertainty = 0.0 if uncertainty is None else float(uncertainty)
        validity = 1.0 if validity is None else float(validity)
        reward = float(reward)

        if self.use_weighted_scoring:
            # Issue 7.9: Use weighted scoring
            quality_score = self._compute_quality_score(reward, uncertainty, validity)
            return quality_score >= self.quality_threshold
        else:
            # Legacy all() behavior
            return (
                reward >= self.min_reward
                and uncertainty <= self.max_uncertainty
                and validity >= self.min_validity
            )

    def get_quality_score(
        self,
        reward: Optional[float],
        uncertainty: Optional[float],
        validity: Optional[float]
    ) -> float:
        """Get the quality score for an experience without accepting/rejecting.

        Useful for debugging and analysis.
        """
        if reward is None:
            return 0.0
        uncertainty = 0.0 if uncertainty is None else float(uncertainty)
        validity = 1.0 if validity is None else float(validity)
        return self._compute_quality_score(float(reward), uncertainty, validity)


@dataclass
class ExperienceRecord:
    data: Dict[str, Any]
    reward: float
    uncertainty: Optional[float]
    validity: Optional[float]


@dataclass
class StratifiedSamplingConfig:
    """Configuration for stratified sampling (Issue 7.8).

    Attributes:
        num_reward_strata: Number of reward bins for stratification.
        num_novelty_strata: Number of novelty bins for stratification.
        reward_weight: Weight for reward-based sampling (0-1).
        novelty_weight: Weight for novelty-based sampling (0-1).
        uniform_weight: Weight for uniform sampling (computed as 1 - reward - novelty).
        min_per_stratum: Minimum samples to draw from each non-empty stratum.
    """
    num_reward_strata: int = 5
    num_novelty_strata: int = 3
    reward_weight: float = 0.4
    novelty_weight: float = 0.3
    min_per_stratum: int = 1


@dataclass
class ExperienceBuffer:
    """Experience buffer with quality gating and stratified sampling (Issue 7.8).

    Supports stratified sampling by reward and novelty to ensure diverse
    experience replay that covers both high-reward and novel experiences.
    """
    max_size: int = 10000
    gate: QualityGate = field(default_factory=QualityGate)
    records: List[ExperienceRecord] = field(default_factory=list)
    accepted: int = 0
    rejected: int = 0
    stratified_config: StratifiedSamplingConfig = field(default_factory=StratifiedSamplingConfig)

    def __len__(self) -> int:
        return len(self.records)

    def _extract_metrics(self, record: Dict[str, Any]) -> Tuple[Optional[float], Optional[float], Optional[float]]:
        reward = record.get("reward")
        info = record.get("info") or {}
        uncertainty = record.get("uncertainty", info.get("uncertainty"))
        validity = record.get("validity", info.get("validity"))
        return reward, uncertainty, validity

    def _compute_novelty(self, record: ExperienceRecord) -> float:
        """Compute novelty score for a record.

        Novelty is approximated by uncertainty (higher uncertainty = more novel).
        Can also incorporate state-space distance metrics.
        """
        if record.uncertainty is not None:
            # Higher uncertainty indicates more novel state
            return min(1.0, record.uncertainty)
        # Default novelty based on reward magnitude (unusual rewards = novel)
        if record.reward != 0:
            return min(1.0, abs(record.reward) / (abs(record.reward) + 1.0))
        return 0.5  # Default novelty

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

    def _build_strata(self) -> Tuple[Dict[int, List[int]], Dict[int, List[int]]]:
        """Build stratification indices for reward and novelty (Issue 7.8).

        Returns:
            reward_strata: Dict mapping reward bin -> list of record indices
            novelty_strata: Dict mapping novelty bin -> list of record indices
        """
        import random

        cfg = self.stratified_config

        # Get reward and novelty ranges
        rewards = [r.reward for r in self.records]
        novelties = [self._compute_novelty(r) for r in self.records]

        min_reward, max_reward = min(rewards), max(rewards)
        min_novelty, max_novelty = min(novelties), max(novelties)

        # Avoid division by zero
        reward_range = max(max_reward - min_reward, 1e-8)
        novelty_range = max(max_novelty - min_novelty, 1e-8)

        # Build strata
        reward_strata: Dict[int, List[int]] = {i: [] for i in range(cfg.num_reward_strata)}
        novelty_strata: Dict[int, List[int]] = {i: [] for i in range(cfg.num_novelty_strata)}

        for idx, (reward, novelty) in enumerate(zip(rewards, novelties)):
            # Assign to reward stratum
            reward_bin = min(
                int((reward - min_reward) / reward_range * cfg.num_reward_strata),
                cfg.num_reward_strata - 1
            )
            reward_strata[reward_bin].append(idx)

            # Assign to novelty stratum
            novelty_bin = min(
                int((novelty - min_novelty) / novelty_range * cfg.num_novelty_strata),
                cfg.num_novelty_strata - 1
            )
            novelty_strata[novelty_bin].append(idx)

        return reward_strata, novelty_strata

    def sample(self, count: int, *, seed: Optional[int] = None) -> List[ExperienceRecord]:
        """Sample records uniformly."""
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

    def sample_stratified(
        self,
        count: int,
        *,
        seed: Optional[int] = None,
        by_reward: bool = True,
        by_novelty: bool = True
    ) -> List[ExperienceRecord]:
        """Sample records using stratified sampling by reward and/or novelty (Issue 7.8).

        This ensures diverse experience replay by drawing samples from different
        reward and novelty strata, preventing oversampling of common experiences.

        Args:
            count: Total number of records to sample.
            seed: Optional random seed for reproducibility.
            by_reward: Enable reward-based stratification.
            by_novelty: Enable novelty-based stratification.

        Returns:
            List of sampled experience records with diverse reward/novelty coverage.
        """
        if count <= 0:
            return []
        if count >= len(self.records):
            return list(self.records)

        import random
        rng = random.Random(seed) if seed is not None else random

        cfg = self.stratified_config
        reward_strata, novelty_strata = self._build_strata()

        # Compute allocation based on weights
        uniform_weight = max(0.0, 1.0 - cfg.reward_weight - cfg.novelty_weight)
        reward_count = int(count * cfg.reward_weight) if by_reward else 0
        novelty_count = int(count * cfg.novelty_weight) if by_novelty else 0
        uniform_count = count - reward_count - novelty_count

        sampled_indices: set = set()

        # Sample from reward strata (Issue 7.8)
        if reward_count > 0 and by_reward:
            # Distribute samples across non-empty strata
            non_empty_strata = [s for s in reward_strata.values() if len(s) > 0]
            if non_empty_strata:
                per_stratum = max(cfg.min_per_stratum, reward_count // len(non_empty_strata))
                remaining = reward_count

                for stratum_indices in non_empty_strata:
                    n_sample = min(per_stratum, len(stratum_indices), remaining)
                    if n_sample > 0:
                        chosen = rng.sample(stratum_indices, n_sample)
                        sampled_indices.update(chosen)
                        remaining -= n_sample
                    if remaining <= 0:
                        break

        # Sample from novelty strata (Issue 7.8)
        if novelty_count > 0 and by_novelty:
            non_empty_strata = [s for s in novelty_strata.values() if len(s) > 0]
            if non_empty_strata:
                # Bias toward high-novelty strata
                weights = [1.0 + 0.5 * i for i in range(len(non_empty_strata))]
                total_weight = sum(weights)

                remaining = novelty_count
                for stratum_indices, weight in zip(non_empty_strata, weights):
                    n_sample = int(remaining * weight / total_weight)
                    n_sample = min(n_sample, len(stratum_indices))
                    available = [i for i in stratum_indices if i not in sampled_indices]
                    if available and n_sample > 0:
                        n_sample = min(n_sample, len(available))
                        chosen = rng.sample(available, n_sample)
                        sampled_indices.update(chosen)

        # Fill remaining with uniform sampling
        remaining = count - len(sampled_indices)
        if remaining > 0:
            available = [i for i in range(len(self.records)) if i not in sampled_indices]
            if available:
                n_sample = min(remaining, len(available))
                chosen = rng.sample(available, n_sample)
                sampled_indices.update(chosen)

        return [self.records[i] for i in sampled_indices]

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
