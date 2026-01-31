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

from .base import (
    VAGIConfig,
    VAGICore,
    RecurrentState,
    KVCache,
    load_vagi_lite_config,
    save_vagi_lite_config,
)

from .planning import (
    BudgetController,
    BudgetDecision,
    CounterfactualRecord,
    RolloutBatch,
    imagine_rollouts,
    mix_rollouts,
    policy_value_losses,
    dyna_update,
)

from .training import (
    ExperienceBuffer,
    ExperienceRecord,
    QualityGate,
    aggregate_metrics,
    compute_drop,
    should_rollback,
    compute_gae,
    td_lambda_returns,
)

from .perception import ImageObsEncoder

try:
    from .agi import AGIConfig, AGIModel, AGIExecutor, load_agi_config, load_agi_small_config
    from .nlp import BytePairTokenizer, TextEmbedding, NextTokenPredictor, MaskedLanguageModel
    from .knowledge import HierarchicalMemory, KnowledgeGraph, SemanticMemory, EpisodicMemory, ConceptEncoder
    from .reasoning import AbstractReasoner, RelationalReasoning, CausalGraphLearner, AnalogyMaker, CounterfactualReasoner
    from .learning import TaskEmbedding, CurriculumScheduler, TransferLearner, FewShotLearner
    from .interaction import ToolRegistry, ToolUseController, ToolSelector
    from .perception import VisionTransformerEncoder, MultiModalEncoder, ImageTextAligner
    
    AGI_AVAILABLE = True
except ImportError as e:
    AGI_AVAILABLE = False
    import warnings
    warnings.warn(f"AGI components not available: {e}")

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

if AGI_AVAILABLE:
    __all__.extend([
        "AGIConfig",
        "AGIModel",
        "AGIExecutor",
        "load_agi_config",
        "load_agi_small_config",
        "BytePairTokenizer",
        "TextEmbedding",
        "NextTokenPredictor",
        "MaskedLanguageModel",
        "HierarchicalMemory",
        "KnowledgeGraph",
        "SemanticMemory",
        "EpisodicMemory",
        "ConceptEncoder",
        "AbstractReasoner",
        "RelationalReasoning",
        "CausalGraphLearner",
        "AnalogyMaker",
        "CounterfactualReasoner",
        "TaskEmbedding",
        "CurriculumScheduler",
        "TransferLearner",
        "FewShotLearner",
        "ToolRegistry",
        "ToolUseController",
        "ToolSelector",
    ])
