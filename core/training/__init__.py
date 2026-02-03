"""Training utilities and loss functions."""

from .experience import ExperienceBuffer, ExperienceRecord, QualityGate
from .losses import (
    action_validity_loss,
    budget_loss,
    imagination_consistency_loss,
    language_loss,
    policy_loss,
    reflection_loss,
    temporal_consistency_loss,
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
from .online_learner import (
    OnlineLearner,
    OnlineLearningConfig,
    ConfidenceGate,
    OnlineExperienceBuffer,
)
from .gradient_safety import (
    GradientSafetyConfig,
    GradientSafetyManager,
    GradientMonitor,
    GradientClipper,
    LossScaler,
    safe_backward,
    check_model_health,
)

__all__ = [
    "ExperienceBuffer",
    "ExperienceRecord",
    "QualityGate",
    "action_validity_loss",
    "budget_loss",
    "imagination_consistency_loss",
    "language_loss",
    "policy_loss",
    "reflection_loss",
    "temporal_consistency_loss",
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
    # Online Learning (new)
    "OnlineLearner",
    "OnlineLearningConfig",
    "ConfidenceGate",
    "OnlineExperienceBuffer",
    # Gradient Safety (new)
    "GradientSafetyConfig",
    "GradientSafetyManager",
    "GradientMonitor",
    "GradientClipper",
    "LossScaler",
    "safe_backward",
    "check_model_health",
]
