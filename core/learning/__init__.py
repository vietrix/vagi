"""Meta-learning and transfer learning."""

from .meta import (
    TaskEmbedding,
    MAMLAdapter,
    CurriculumScheduler,
    TransferLearner,
    FewShotLearner,
)
from .metacognition import (
    MetaCognition,
    SelfModel,
    ThinkingMonitor,
    UncertaintyCalibrator,
    ThinkingState,
    ThoughtTrace,
)

__all__ = [
    "TaskEmbedding",
    "MAMLAdapter",
    "CurriculumScheduler",
    "TransferLearner",
    "FewShotLearner",
    "MetaCognition",
    "SelfModel",
    "ThinkingMonitor",
    "UncertaintyCalibrator",
    "ThinkingState",
    "ThoughtTrace",
]
