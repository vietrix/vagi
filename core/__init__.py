"""vAGI public API."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


def _enable_io_checkpoint() -> None:
    """Expose io.checkpoint by attaching a package path to stdlib io."""
    try:
        import io as stdlib_io
    except Exception:
        return

    io_dir = Path(__file__).resolve().parent.parent / "io"
    checkpoint_path = io_dir / "checkpoint.py"
    if not checkpoint_path.exists():
        return

    if "io.checkpoint" in sys.modules:
        return

    stdlib_io.__path__ = [str(io_dir)]
    spec = importlib.util.spec_from_file_location("io.checkpoint", checkpoint_path)
    if spec is None or spec.loader is None:
        return
    module = importlib.util.module_from_spec(spec)
    sys.modules["io.checkpoint"] = module
    spec.loader.exec_module(module)
    setattr(stdlib_io, "checkpoint", module)


_enable_io_checkpoint()

from .budget import BudgetController, BudgetDecision, CounterfactualRecord
from .diagnostics import aggregate_metrics, compute_drop, should_rollback
from .experience import ExperienceBuffer, ExperienceRecord, QualityGate
from .config import VAGIConfig
from .dyna import RolloutBatch, dyna_update, imagine_rollouts, mix_rollouts, policy_value_losses
from .memory import KVCache, RecurrentState
from .model import VAGICore
from .presets import load_vagi_lite_config, save_vagi_lite_config
from .returns import compute_gae, td_lambda_returns
from .vision import ImageObsEncoder

__all__ = [
    "VAGIConfig",
    "VAGICore",
    "RecurrentState",
    "KVCache",
    "ImageObsEncoder",
    "BudgetController",
    "BudgetDecision",
    "CounterfactualRecord",
    "ExperienceBuffer",
    "ExperienceRecord",
    "QualityGate",
    "aggregate_metrics",
    "compute_drop",
    "should_rollback",
    "compute_gae",
    "td_lambda_returns",
    "RolloutBatch",
    "imagine_rollouts",
    "mix_rollouts",
    "policy_value_losses",
    "dyna_update",
    "load_vagi_lite_config",
    "save_vagi_lite_config",
]
