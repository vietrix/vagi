"""Abstract reasoning and causal inference.

This module provides comprehensive reasoning capabilities:

Program Synthesis (program_synthesis.py):
- Type-guided beam search with pruning (5.2)
- MDL-based program scoring (5.3)
- Behavioral specs and noisy example handling (5.4)
- Program complexity with MDL regularization (5.11)
- Sandboxed execution with error logging (5.12)

Abstract Reasoning (abstract.py):
- Explicit causal/analogical reasoning modes (5.6)
- Graph Attention Networks with type constraints (5.7)
- Analogy matching with dimension checking (5.8)
- Temporal ordering with Granger causality (5.9)
- Counterfactual plausibility scoring (5.10)
"""

from .abstract import (
    # Core reasoning modules
    AbstractReasoner,
    RelationalReasoning,
    CausalGraphLearner,
    AnalogyMaker,
    CounterfactualReasoner,
    # New: Graph Attention with types (5.7)
    TypedGraphAttention,
    # New: Granger causality (5.9)
    GrangerCausalityTest,
    # New: Type enums (5.6, 5.7)
    ReasoningMode,
    NodeType,
    EdgeType,
)
from .program_synthesis import (
    # Core synthesis
    ProgramSynthesizer,
    DomainSpecificLanguage,
    NeuralProgramEncoder,
    NeuralProgramSampler,
    ProgramVerifier,
    SymbolicReasoner,
    NeuroSymbolicIntegration,
    Program,
    PrimitiveOp,
    # New: Beam search (5.2)
    TypeGuidedBeamSearch,
    BeamCandidate,
    # New: MDL scoring (5.3)
    MDLProgramScorer,
    # New: Behavioral specs (5.4)
    BehavioralSpec,
    NoisyExampleHandler,
    # New: Complexity metrics (5.11)
    ProgramComplexity,
    # New: Execution safety (5.12)
    ExecutionContext,
    ExecutionError,
    TimeoutError,
    ResourceLimitError,
    TypeMismatchError,
)

__all__ = [
    # Abstract reasoning
    "AbstractReasoner",
    "RelationalReasoning",
    "CausalGraphLearner",
    "AnalogyMaker",
    "CounterfactualReasoner",
    "TypedGraphAttention",
    "GrangerCausalityTest",
    "ReasoningMode",
    "NodeType",
    "EdgeType",
    # Program synthesis
    "ProgramSynthesizer",
    "DomainSpecificLanguage",
    "NeuralProgramEncoder",
    "NeuralProgramSampler",
    "ProgramVerifier",
    "SymbolicReasoner",
    "NeuroSymbolicIntegration",
    "Program",
    "PrimitiveOp",
    "TypeGuidedBeamSearch",
    "BeamCandidate",
    "MDLProgramScorer",
    "BehavioralSpec",
    "NoisyExampleHandler",
    "ProgramComplexity",
    "ExecutionContext",
    "ExecutionError",
    "TimeoutError",
    "ResourceLimitError",
    "TypeMismatchError",
]
