from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(slots=True)
class Settings:
    kernel_url: str
    host: str
    port: int
    runtime_dir: Path
    dream_hour: int
    dream_minute: int
    max_decide_iters: int
    risk_threshold: float
    reasoner_mode: str
    weaver_top_k: int
    mutation_enabled: bool
    mutation_generations: int
    mutation_population_size: int
    mutation_survivors: int
    mutation_risk_threshold: float
    mutation_promote: bool


def load_settings() -> Settings:
    runtime_dir = Path(os.getenv("VAGI_RUNTIME_DIR", "runtime"))
    runtime_dir.mkdir(parents=True, exist_ok=True)
    return Settings(
        kernel_url=os.getenv("VAGI_KERNEL_URL", "http://127.0.0.1:7070"),
        host=os.getenv("VAGI_ORCH_HOST", "127.0.0.1"),
        port=int(os.getenv("VAGI_ORCH_PORT", "8080")),
        runtime_dir=runtime_dir,
        dream_hour=int(os.getenv("VAGI_DREAM_HOUR", "2")),
        dream_minute=int(os.getenv("VAGI_DREAM_MINUTE", "0")),
        max_decide_iters=int(os.getenv("VAGI_MAX_DECIDE_ITERS", "12")),
        risk_threshold=float(os.getenv("VAGI_RISK_THRESHOLD", "0.65")),
        reasoner_mode=os.getenv("VAGI_REASONER_MODE", "classic").strip().lower(),
        weaver_top_k=max(1, min(int(os.getenv("VAGI_WEAVER_TOP_K", "3")), 10)),
        mutation_enabled=os.getenv("VAGI_MUTATION_ENABLED", "true").strip().lower()
        in {"1", "true", "yes", "on"},
        mutation_generations=max(
            1, min(int(os.getenv("VAGI_MUTATION_GENERATIONS", "3")), 20)
        ),
        mutation_population_size=max(
            2, min(int(os.getenv("VAGI_MUTATION_POPULATION", "8")), 64)
        ),
        mutation_survivors=max(
            1, min(int(os.getenv("VAGI_MUTATION_SURVIVORS", "2")), 16)
        ),
        mutation_risk_threshold=float(
            os.getenv("VAGI_MUTATION_RISK_THRESHOLD", "0.65")
        ),
        mutation_promote=os.getenv("VAGI_MUTATION_PROMOTE", "true").strip().lower()
        in {"1", "true", "yes", "on"},
    )
