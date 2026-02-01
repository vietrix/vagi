"""Training utilities and loss functions."""

from .experience import ExperienceBuffer, ExperienceRecord, QualityGate
from .losses import (
    budget_loss,
    imagination_consistency_loss,
    language_loss,
    policy_loss,
    reflection_loss,
    total_loss,
    value_loss,
    world_loss,
    scene_graph_loss,
    program_synthesis_loss,
    grounded_language_loss,
    intrinsic_reward_loss,
    meta_cognition_loss,
)
from .returns import compute_gae, td_lambda_returns
from .calibration import ConfidenceCalibrator
from .diagnostics import aggregate_metrics, compute_drop, should_rollback
from .continuous_learner import (
    ContinuousLearner,
    ContinuousLearningConfig,
    SelfSupervisedLabeler,
    ExperienceReplay,
)

__all__ = [
    "ExperienceBuffer",
    "ExperienceRecord",
    "QualityGate",
    "budget_loss",
    "imagination_consistency_loss",
    "language_loss",
    "policy_loss",
    "reflection_loss",
    "total_loss",
    "value_loss",
    "world_loss",
    "compute_gae",
    "td_lambda_returns",
    "ConfidenceCalibrator", 
    "aggregate_metrics",
    "compute_drop",
    "should_rollback",
    "ContinuousLearner",
    "ContinuousLearningConfig",
    "SelfSupervisedLabeler",
    "ExperienceReplay",
    "scene_graph_loss",
    "program_synthesis_loss",
    "grounded_language_loss",
    "intrinsic_reward_loss",
    "meta_cognition_loss",
]
