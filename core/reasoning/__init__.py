"""Abstract reasoning, causal inference, and System 2 thinking.

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

MCTS Engine (mcts_engine.py):
- Monte Carlo Tree Search for System 2 reasoning
- UCT-based tree navigation
- Policy/Value model integration
- Parallel and beam search variants

Code Interpreter (code_interpreter.py):
- Secure sandboxed Python execution
- Code-as-reasoning for verification
- ReAct-style agent loops
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
from .mcts_engine import (
    # MCTS Core
    MCTSConfig,
    MCTSNode,
    MCTSEngine,
    # Policy/Value interfaces
    PolicyModel,
    ValueModel,
    LLMPolicyModel,
    VerifierValueModel,
    SelfEvalValueModel,
    # Variants
    BeamSearchMCTS,
    ParallelMCTS,
)
from .code_interpreter import (
    # Config and results
    CodeInterpreterConfig,
    ExecutionResult,
    # Core components
    CodeParser,
    SecurityChecker,
    CodeInterpreter,
    # Executors
    RestrictedExecutor,
    SubprocessExecutor,
    # Agents
    ReActCodeInterpreter,
    TorchCodeInterpreter,
)
from .reflexion import (
    ReflexionAgent,
    ReflexionConfig,
    ReflexionMemory,
    CodeExecutor,
)
from .verifier_hook import (
    VerifierHook,
    VerifierState,
    TokenResult,
    VerificationResult,
    StreamingVerifier,
    create_verifier_hook,
    process_model_output,
)
from .curiosity import (
    CuriosityGate,
    QuestionGenerator,
    PerplexityEstimator,
    CuriosityDecision,
    PerplexityResult,
)
from .planner import (
    PlannerAgent,
    PlannerConfig,
    Task,
    TaskQueue,
    TaskStatus,
)
from .simulator import (
    CounterfactualSimulator,
    SimulatorConfig,
    SafetyFlag,
    NegativeOutcome,
    SimulationResult,
)
from .orchestrator import (
    Orchestrator,
    OrchestratorConfig,
    OrchestratorResult,
    ExecutionResult,
    Executor,
)
from .verifiers import (
    PythonExecutor,
    ExecutionBackend,
    execute_python,
    verify_code,
    check_code_security,
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
    # MCTS Engine (System 2 Reasoning)
    "MCTSConfig",
    "MCTSNode",
    "MCTSEngine",
    "PolicyModel",
    "ValueModel",
    "LLMPolicyModel",
    "VerifierValueModel",
    "SelfEvalValueModel",
    "BeamSearchMCTS",
    "ParallelMCTS",
    # Code Interpreter
    "CodeInterpreterConfig",
    "ExecutionResult",
    "CodeParser",
    "SecurityChecker",
    "CodeInterpreter",
    "RestrictedExecutor",
    "SubprocessExecutor",
    "ReActCodeInterpreter",
    "TorchCodeInterpreter",
    # Reflexion
    "ReflexionAgent",
    "ReflexionConfig",
    "ReflexionMemory",
    "CodeExecutor",
    # Verifier Hook
    "VerifierHook",
    "VerifierState",
    "TokenResult",
    "VerificationResult",
    "StreamingVerifier",
    "create_verifier_hook",
    "process_model_output",
    # Curiosity / Active inquiry
    "CuriosityGate",
    "QuestionGenerator",
    "PerplexityEstimator",
    "CuriosityDecision",
    "PerplexityResult",
    # Planning + Simulation
    "PlannerAgent",
    "PlannerConfig",
    "Task",
    "TaskQueue",
    "TaskStatus",
    "CounterfactualSimulator",
    "SimulatorConfig",
    "SafetyFlag",
    "NegativeOutcome",
    "SimulationResult",
    "Orchestrator",
    "OrchestratorConfig",
    "OrchestratorResult",
    "ExecutionResult",
    "Executor",
    # Python Executor
    "PythonExecutor",
    "ExecutionBackend",
    "execute_python",
    "verify_code",
    "check_code_security",
]
