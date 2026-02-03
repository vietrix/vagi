"""Program synthesis and compositional reasoning for AGI.

This module provides:
- Domain-specific language (DSL) with 40+ primitive operations
- Type-guided beam search for efficient program synthesis (5.2)
- MDL-based program scoring (5.3)
- Behavioral specifications and noisy example handling (5.4)
- Program complexity metrics with MDL regularization (5.11)
- Sandboxed execution with proper error logging (5.12)
"""

from __future__ import annotations

import logging
import math
import time
import signal
import traceback
from contextlib import contextmanager
from typing import Any, Callable, Dict, List, Optional, Tuple, Union, Set
from dataclasses import dataclass, field
from enum import Enum
from functools import wraps

import torch
from torch import nn
from torch.nn import functional as F

# Configure module logger
logger = logging.getLogger(__name__)


# ============================================================================
# 5.12 Execution Safety - Error Categories and Sandboxing
# ============================================================================

class ExecutionError(Exception):
    """Base class for execution errors with categorization."""

    def __init__(self, message: str, category: str = "unknown", operation: str = ""):
        super().__init__(message)
        self.category = category
        self.operation = operation
        self.timestamp = time.time()


class TimeoutError(ExecutionError):
    """Raised when execution exceeds time limit."""

    def __init__(self, message: str, timeout: float, operation: str = ""):
        super().__init__(message, category="timeout", operation=operation)
        self.timeout = timeout


class ResourceLimitError(ExecutionError):
    """Raised when resource limits are exceeded."""

    def __init__(self, message: str, resource_type: str, limit: int, operation: str = ""):
        super().__init__(message, category="resource_limit", operation=operation)
        self.resource_type = resource_type
        self.limit = limit


class TypeMismatchError(ExecutionError):
    """Raised when type constraints are violated."""

    def __init__(self, message: str, expected_type: str, actual_type: str, operation: str = ""):
        super().__init__(message, category="type_mismatch", operation=operation)
        self.expected_type = expected_type
        self.actual_type = actual_type


@dataclass
class ExecutionContext:
    """Context for sandboxed execution with resource limits."""
    max_execution_time: float = 5.0  # seconds
    max_iterations: int = 10000
    max_list_size: int = 100000
    max_recursion_depth: int = 100
    current_depth: int = 0
    iteration_count: int = 0
    start_time: Optional[float] = None
    errors: List[ExecutionError] = field(default_factory=list)

    def check_time_limit(self, operation: str = "") -> None:
        """Check if execution time limit is exceeded."""
        if self.start_time is not None:
            elapsed = time.time() - self.start_time
            if elapsed > self.max_execution_time:
                error = TimeoutError(
                    f"Execution timeout after {elapsed:.2f}s",
                    timeout=self.max_execution_time,
                    operation=operation
                )
                self.errors.append(error)
                raise error

    def check_iteration_limit(self, operation: str = "") -> None:
        """Check if iteration limit is exceeded."""
        self.iteration_count += 1
        if self.iteration_count > self.max_iterations:
            error = ResourceLimitError(
                f"Iteration limit exceeded: {self.iteration_count}",
                resource_type="iterations",
                limit=self.max_iterations,
                operation=operation
            )
            self.errors.append(error)
            raise error

    def check_list_size(self, size: int, operation: str = "") -> None:
        """Check if list size limit is exceeded."""
        if size > self.max_list_size:
            error = ResourceLimitError(
                f"List size limit exceeded: {size}",
                resource_type="list_size",
                limit=self.max_list_size,
                operation=operation
            )
            self.errors.append(error)
            raise error

    def enter_recursion(self, operation: str = "") -> None:
        """Track recursion depth."""
        self.current_depth += 1
        if self.current_depth > self.max_recursion_depth:
            error = ResourceLimitError(
                f"Recursion depth limit exceeded: {self.current_depth}",
                resource_type="recursion_depth",
                limit=self.max_recursion_depth,
                operation=operation
            )
            self.errors.append(error)
            raise error

    def exit_recursion(self) -> None:
        """Exit recursion level."""
        self.current_depth = max(0, self.current_depth - 1)

    def log_error(self, error: ExecutionError) -> None:
        """Log an execution error."""
        self.errors.append(error)
        logger.warning(
            f"Execution error [{error.category}] in operation '{error.operation}': {error}"
        )


class PrimitiveOp(Enum):
    """Basic operations in the DSL - expanded from 9 to 30+ primitives."""
    # Stop token
    STOP = "stop"

    # Higher-order operations (core)
    MAP = "map"
    FILTER = "filter"
    REDUCE = "reduce"
    COMPOSE = "compose"
    FOLD = "fold"
    SCAN = "scan"
    UNFOLD = "unfold"

    # List operations
    REVERSE = "reverse"
    SORT = "sort"
    ZIP = "zip"
    TAKE = "take"
    DROP = "drop"
    CONCAT = "concat"
    HEAD = "head"
    TAIL = "tail"
    LENGTH = "length"
    INDEX = "index"

    # Arithmetic operations
    ADD = "add"
    SUBTRACT = "subtract"
    MULTIPLY = "multiply"
    DIVIDE = "divide"
    MODULO = "modulo"
    ABS = "abs"
    MAX = "max"
    MIN = "min"
    NEGATE = "negate"

    # Comparison/Logic operations
    GREATER = "greater"
    LESS = "less"
    EQUAL = "equal"
    NOT_EQUAL = "not_equal"
    AND = "and"
    OR = "or"
    NOT = "not"
    XOR = "xor"

    # Control flow
    IF = "if"
    CASE = "case"
    LOOP = "loop"
    RECURSE = "recurse"

    # Constants
    CONST_ZERO = "const_zero"
    CONST_ONE = "const_one"
    CONST_EMPTY = "const_empty"
    IDENTITY = "identity"


@dataclass
class Program:
    """Represents a synthesized program.

    Attributes:
        operations: List of (operation, parameters) tuples
        score: Program score (higher is better)
        complexity: Program complexity for MDL scoring
        metadata: Additional metadata (e.g., synthesis trace)
    """
    operations: List[Tuple[PrimitiveOp, Any]]
    score: float = 0.0
    complexity: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def execute(
        self,
        input_data: Any,
        context: Optional[ExecutionContext] = None
    ) -> Any:
        """Execute the program on input data with sandboxed execution (5.12).

        Args:
            input_data: Input to the program
            context: Execution context with resource limits. If None, creates default.

        Returns:
            Result of program execution

        Raises:
            ExecutionError: On timeout, resource limit, or type errors
        """
        if context is None:
            context = ExecutionContext()

        context.start_time = time.time()
        result = input_data

        for op, params in self.operations:
            try:
                context.check_time_limit(operation=op.value)
                result = self._execute_op(op, params, result, context)
            except ExecutionError:
                # Re-raise execution errors for proper handling
                raise
            except Exception as e:
                # Log unexpected errors and continue
                error = ExecutionError(
                    str(e),
                    category="runtime_error",
                    operation=op.value
                )
                context.log_error(error)
                logger.debug(f"Operation {op.value} failed: {traceback.format_exc()}")
                break

        return result

    def execute_safe(
        self,
        input_data: Any,
        context: Optional[ExecutionContext] = None
    ) -> Tuple[Any, List[ExecutionError]]:
        """Execute program safely, returning result and any errors.

        This is a non-throwing version that captures all errors.

        Args:
            input_data: Input to the program
            context: Execution context with resource limits

        Returns:
            Tuple of (result, list of errors encountered)
        """
        if context is None:
            context = ExecutionContext()

        try:
            result = self.execute(input_data, context)
            return result, context.errors
        except ExecutionError as e:
            context.log_error(e)
            return input_data, context.errors
        except Exception as e:
            error = ExecutionError(str(e), category="unexpected")
            context.log_error(error)
            return input_data, context.errors

    def _execute_op(
        self,
        op: PrimitiveOp,
        params: Any,
        data: Any,
        context: Optional[ExecutionContext] = None
    ) -> Any:
        """Execute a single operation with resource limit checking (5.12).

        Args:
            op: The operation to execute
            params: Parameters for the operation
            data: Input data
            context: Execution context for resource tracking

        Returns:
            Result of the operation
        """
        if context is None:
            context = ExecutionContext()

        # Higher-order operations with resource checking
        if op == PrimitiveOp.MAP:
            if isinstance(data, (list, tuple)):
                context.check_list_size(len(data), op.value)
                result = []
                for x in data:
                    context.check_iteration_limit(op.value)
                    result.append(params(x))
                return result
            return [params(x) for x in data]
        elif op == PrimitiveOp.FILTER:
            if isinstance(data, (list, tuple)):
                context.check_list_size(len(data), op.value)
                result = []
                for x in data:
                    context.check_iteration_limit(op.value)
                    if params(x):
                        result.append(x)
                return result
            return [x for x in data if params(x)]
        elif op == PrimitiveOp.REDUCE:
            if not data:
                return data
            result = data[0]
            for x in data[1:]:
                context.check_iteration_limit(op.value)
                result = params(result, x)
            return result
        elif op == PrimitiveOp.FOLD:
            init, func = params
            result = init
            for x in data:
                context.check_iteration_limit(op.value)
                result = func(result, x)
            return result
        elif op == PrimitiveOp.SCAN:
            init, func = params
            results = []
            acc = init
            for x in data:
                context.check_iteration_limit(op.value)
                acc = func(acc, x)
                results.append(acc)
            context.check_list_size(len(results), op.value)
            return results
        elif op == PrimitiveOp.COMPOSE:
            f, g = params
            return lambda x: f(g(x))

        # List operations
        elif op == PrimitiveOp.REVERSE:
            return list(reversed(data))
        elif op == PrimitiveOp.SORT:
            return sorted(data)
        elif op == PrimitiveOp.ZIP:
            return list(zip(data, params))
        elif op == PrimitiveOp.TAKE:
            return data[:params]
        elif op == PrimitiveOp.DROP:
            return data[params:]
        elif op == PrimitiveOp.CONCAT:
            return data + params
        elif op == PrimitiveOp.HEAD:
            return data[0] if data else None
        elif op == PrimitiveOp.TAIL:
            return data[1:] if data else []
        elif op == PrimitiveOp.LENGTH:
            return len(data)
        elif op == PrimitiveOp.INDEX:
            return data[params] if 0 <= params < len(data) else None

        # Arithmetic operations
        elif op == PrimitiveOp.ADD:
            return data + params
        elif op == PrimitiveOp.SUBTRACT:
            return data - params
        elif op == PrimitiveOp.MULTIPLY:
            return data * params
        elif op == PrimitiveOp.DIVIDE:
            return data / params if params != 0 else data
        elif op == PrimitiveOp.MODULO:
            return data % params if params != 0 else data
        elif op == PrimitiveOp.ABS:
            return abs(data)
        elif op == PrimitiveOp.MAX:
            return max(data, params)
        elif op == PrimitiveOp.MIN:
            return min(data, params)
        elif op == PrimitiveOp.NEGATE:
            return -data

        # Comparison operations
        elif op == PrimitiveOp.GREATER:
            return data > params
        elif op == PrimitiveOp.LESS:
            return data < params
        elif op == PrimitiveOp.EQUAL:
            return data == params
        elif op == PrimitiveOp.NOT_EQUAL:
            return data != params

        # Logic operations
        elif op == PrimitiveOp.AND:
            return data and params
        elif op == PrimitiveOp.OR:
            return data or params
        elif op == PrimitiveOp.NOT:
            return not data
        elif op == PrimitiveOp.XOR:
            return bool(data) != bool(params)

        # Control flow
        elif op == PrimitiveOp.IF:
            cond, then_val, else_val = params
            return then_val if cond else else_val
        elif op == PrimitiveOp.CASE:
            idx, cases = params
            return cases[idx] if 0 <= idx < len(cases) else cases[-1]
        elif op == PrimitiveOp.LOOP:
            # LOOP: (condition_func, body_func, max_iter) -> repeatedly apply body while condition
            cond_func, body_func, max_iter = params if len(params) == 3 else (params[0], params[1], 100)
            # Respect context limits
            max_iter = min(max_iter, context.max_iterations - context.iteration_count)
            result = data
            for _ in range(max_iter):
                context.check_iteration_limit(op.value)
                context.check_time_limit(op.value)
                if not cond_func(result):
                    break
                result = body_func(result)
            return result
        elif op == PrimitiveOp.RECURSE:
            # RECURSE: (base_case_check, base_val, recursive_func, max_depth) -> recursive computation
            base_check, base_val, rec_func = params[:3]
            max_depth = min(
                params[3] if len(params) > 3 else 50,
                context.max_recursion_depth - context.current_depth
            )

            def recurse_helper(val, depth):
                context.enter_recursion(op.value)
                try:
                    context.check_time_limit(op.value)
                    if depth >= max_depth or base_check(val):
                        return base_val(val) if callable(base_val) else base_val
                    return rec_func(val, lambda v: recurse_helper(v, depth + 1))
                finally:
                    context.exit_recursion()

            return recurse_helper(data, 0)
        elif op == PrimitiveOp.UNFOLD:
            # UNFOLD: (seed, condition, generator) -> generate list from seed
            seed, cond_func, gen_func = params
            result = []
            current = seed if seed is not None else data
            max_iter = min(1000, context.max_iterations - context.iteration_count)
            for _ in range(max_iter):
                context.check_iteration_limit(op.value)
                context.check_time_limit(op.value)
                if not cond_func(current):
                    break
                value, next_state = gen_func(current)
                result.append(value)
                current = next_state
                context.check_list_size(len(result), op.value)
            return result

        # Constants
        elif op == PrimitiveOp.CONST_ZERO:
            return 0
        elif op == PrimitiveOp.CONST_ONE:
            return 1
        elif op == PrimitiveOp.CONST_EMPTY:
            return []
        elif op == PrimitiveOp.IDENTITY:
            return data

        # Default: return unchanged
        return data


class DomainSpecificLanguage:
    """Define a domain-specific language for program synthesis - expanded."""

    def __init__(self):
        # All primitives mapping
        self.primitives = {
            # Stop token
            'stop': PrimitiveOp.STOP,
            # Higher-order
            'map': PrimitiveOp.MAP,
            'filter': PrimitiveOp.FILTER,
            'reduce': PrimitiveOp.REDUCE,
            'compose': PrimitiveOp.COMPOSE,
            'fold': PrimitiveOp.FOLD,
            'scan': PrimitiveOp.SCAN,
            'unfold': PrimitiveOp.UNFOLD,
            # List ops
            'reverse': PrimitiveOp.REVERSE,
            'sort': PrimitiveOp.SORT,
            'zip': PrimitiveOp.ZIP,
            'take': PrimitiveOp.TAKE,
            'drop': PrimitiveOp.DROP,
            'concat': PrimitiveOp.CONCAT,
            'head': PrimitiveOp.HEAD,
            'tail': PrimitiveOp.TAIL,
            'length': PrimitiveOp.LENGTH,
            'index': PrimitiveOp.INDEX,
            # Arithmetic
            'add': PrimitiveOp.ADD,
            'sub': PrimitiveOp.SUBTRACT,
            'mul': PrimitiveOp.MULTIPLY,
            'div': PrimitiveOp.DIVIDE,
            'mod': PrimitiveOp.MODULO,
            'abs': PrimitiveOp.ABS,
            'max': PrimitiveOp.MAX,
            'min': PrimitiveOp.MIN,
            'neg': PrimitiveOp.NEGATE,
            # Comparison/Logic
            'gt': PrimitiveOp.GREATER,
            'lt': PrimitiveOp.LESS,
            'eq': PrimitiveOp.EQUAL,
            'neq': PrimitiveOp.NOT_EQUAL,
            'and': PrimitiveOp.AND,
            'or': PrimitiveOp.OR,
            'not': PrimitiveOp.NOT,
            'xor': PrimitiveOp.XOR,
            # Control
            'if': PrimitiveOp.IF,
            'case': PrimitiveOp.CASE,
            'loop': PrimitiveOp.LOOP,
            'recurse': PrimitiveOp.RECURSE,
            # Constants
            'zero': PrimitiveOp.CONST_ZERO,
            'one': PrimitiveOp.CONST_ONE,
            'empty': PrimitiveOp.CONST_EMPTY,
            'id': PrimitiveOp.IDENTITY,
        }

        # Type signatures for type-guided search
        self.type_signatures = {
            # Higher-order with functions
            PrimitiveOp.MAP: ('List[A]', 'Func[A->B]', 'List[B]'),
            PrimitiveOp.FILTER: ('List[A]', 'Func[A->Bool]', 'List[A]'),
            PrimitiveOp.REDUCE: ('List[A]', 'Func[A,A->A]', 'A'),
            PrimitiveOp.FOLD: ('List[A]', 'B', 'Func[B,A->B]', 'B'),
            PrimitiveOp.SCAN: ('List[A]', 'B', 'Func[B,A->B]', 'List[B]'),
            PrimitiveOp.UNFOLD: ('A', 'Func[A->Maybe[Tuple[B,A]]]', 'List[B]'),
            # List -> List
            PrimitiveOp.REVERSE: ('List[A]', 'List[A]'),
            PrimitiveOp.SORT: ('List[A]', 'List[A]'),
            PrimitiveOp.TAIL: ('List[A]', 'List[A]'),
            PrimitiveOp.TAKE: ('List[A]', 'Int', 'List[A]'),
            PrimitiveOp.DROP: ('List[A]', 'Int', 'List[A]'),
            PrimitiveOp.CONCAT: ('List[A]', 'List[A]', 'List[A]'),
            # List -> Element
            PrimitiveOp.HEAD: ('List[A]', 'A'),
            PrimitiveOp.LENGTH: ('List[A]', 'Int'),
            PrimitiveOp.INDEX: ('List[A]', 'Int', 'A'),
            # Two lists
            PrimitiveOp.ZIP: ('List[A]', 'List[B]', 'List[Tuple[A,B]]'),
            # Arithmetic (Num -> Num)
            PrimitiveOp.ADD: ('Num', 'Num', 'Num'),
            PrimitiveOp.SUBTRACT: ('Num', 'Num', 'Num'),
            PrimitiveOp.MULTIPLY: ('Num', 'Num', 'Num'),
            PrimitiveOp.DIVIDE: ('Num', 'Num', 'Num'),
            PrimitiveOp.MODULO: ('Int', 'Int', 'Int'),
            PrimitiveOp.ABS: ('Num', 'Num'),
            PrimitiveOp.MAX: ('Num', 'Num', 'Num'),
            PrimitiveOp.MIN: ('Num', 'Num', 'Num'),
            PrimitiveOp.NEGATE: ('Num', 'Num'),
            # Comparison (A, A -> Bool)
            PrimitiveOp.GREATER: ('Num', 'Num', 'Bool'),
            PrimitiveOp.LESS: ('Num', 'Num', 'Bool'),
            PrimitiveOp.EQUAL: ('A', 'A', 'Bool'),
            PrimitiveOp.NOT_EQUAL: ('A', 'A', 'Bool'),
            # Logic (Bool -> Bool)
            PrimitiveOp.AND: ('Bool', 'Bool', 'Bool'),
            PrimitiveOp.OR: ('Bool', 'Bool', 'Bool'),
            PrimitiveOp.NOT: ('Bool', 'Bool'),
            PrimitiveOp.XOR: ('Bool', 'Bool', 'Bool'),
            # Control
            PrimitiveOp.IF: ('Bool', 'A', 'A', 'A'),
            PrimitiveOp.CASE: ('Int', 'List[A]', 'A'),
            # Constants
            PrimitiveOp.CONST_ZERO: ('Int',),
            PrimitiveOp.CONST_ONE: ('Int',),
            PrimitiveOp.CONST_EMPTY: ('List[A]',),
            PrimitiveOp.IDENTITY: ('A', 'A'),
        }

        # Arity of each operation (number of arguments)
        self.arity = {
            PrimitiveOp.STOP: 0,
            PrimitiveOp.MAP: 2, PrimitiveOp.FILTER: 2, PrimitiveOp.REDUCE: 2,
            PrimitiveOp.COMPOSE: 2, PrimitiveOp.FOLD: 3, PrimitiveOp.SCAN: 3,
            PrimitiveOp.UNFOLD: 2,
            PrimitiveOp.REVERSE: 1, PrimitiveOp.SORT: 1, PrimitiveOp.ZIP: 2,
            PrimitiveOp.TAKE: 2, PrimitiveOp.DROP: 2, PrimitiveOp.CONCAT: 2,
            PrimitiveOp.HEAD: 1, PrimitiveOp.TAIL: 1, PrimitiveOp.LENGTH: 1,
            PrimitiveOp.INDEX: 2,
            PrimitiveOp.ADD: 2, PrimitiveOp.SUBTRACT: 2, PrimitiveOp.MULTIPLY: 2,
            PrimitiveOp.DIVIDE: 2, PrimitiveOp.MODULO: 2, PrimitiveOp.ABS: 1,
            PrimitiveOp.MAX: 2, PrimitiveOp.MIN: 2, PrimitiveOp.NEGATE: 1,
            PrimitiveOp.GREATER: 2, PrimitiveOp.LESS: 2, PrimitiveOp.EQUAL: 2,
            PrimitiveOp.NOT_EQUAL: 2, PrimitiveOp.AND: 2, PrimitiveOp.OR: 2,
            PrimitiveOp.NOT: 1, PrimitiveOp.XOR: 2,
            PrimitiveOp.IF: 3, PrimitiveOp.CASE: 2, PrimitiveOp.LOOP: 2,
            PrimitiveOp.RECURSE: 1,
            PrimitiveOp.CONST_ZERO: 0, PrimitiveOp.CONST_ONE: 0,
            PrimitiveOp.CONST_EMPTY: 0, PrimitiveOp.IDENTITY: 1,
        }

    def get_primitives(self) -> List[PrimitiveOp]:
        """Get all available primitives."""
        return list(self.primitives.values())

    def get_arity(self, op: PrimitiveOp) -> int:
        """Get number of arguments for operation."""
        return self.arity.get(op, 0)

    def can_apply(self, op: PrimitiveOp, state_type: str) -> bool:
        """Check if operation can be applied to given state type."""
        if op not in self.type_signatures:
            return True  # Default: allow

        input_type = self.type_signatures[op][0]
        # Type matching logic
        if input_type == 'A':  # Polymorphic - matches anything
            return True
        if input_type == 'Num' and state_type in ('Int', 'Float', 'Num'):
            return True
        if input_type == 'Bool' and state_type == 'Bool':
            return True
        if input_type.startswith('List') and state_type.startswith('List'):
            return True
        return input_type in state_type

    def get_output_type(self, op: PrimitiveOp, input_types: List[str]) -> str:
        """Infer output type from operation and input types."""
        if op not in self.type_signatures:
            return 'Any'
        sig = self.type_signatures[op]
        return sig[-1]  # Last element is output type


# ============================================================================
# 5.2 Type-Guided Beam Search with Pruning
# ============================================================================

@dataclass
class BeamCandidate:
    """Candidate program in beam search."""
    operations: List[Tuple[PrimitiveOp, Any]]
    score: float
    current_type: str
    params: List[Any] = field(default_factory=list)

    def __lt__(self, other: "BeamCandidate") -> bool:
        """Comparison for heap operations (min-heap, so reverse for max)."""
        return self.score > other.score  # Higher score = better


class TypeGuidedBeamSearch:
    """Type-guided beam search for program synthesis (5.2).

    Uses type information to prune invalid candidates and guide search
    towards type-consistent programs.

    Attributes:
        dsl: Domain-specific language with type signatures
        beam_width: Maximum number of candidates to keep at each step
        max_length: Maximum program length
        score_threshold: Minimum score to keep a candidate
        type_bonus: Score bonus for type-consistent operations
    """

    def __init__(
        self,
        dsl: DomainSpecificLanguage,
        beam_width: int = 10,
        max_length: int = 15,
        score_threshold: float = -10.0,
        type_bonus: float = 0.5,
    ) -> None:
        self.dsl = dsl
        self.beam_width = beam_width
        self.max_length = max_length
        self.score_threshold = score_threshold
        self.type_bonus = type_bonus
        logger.debug(f"TypeGuidedBeamSearch initialized with beam_width={beam_width}")

    def infer_type(self, value: Any) -> str:
        """Infer the type of a value."""
        if isinstance(value, bool):
            return 'Bool'
        elif isinstance(value, int):
            return 'Int'
        elif isinstance(value, float):
            return 'Float'
        elif isinstance(value, list):
            if not value:
                return 'List[Any]'
            elem_type = self.infer_type(value[0])
            return f'List[{elem_type}]'
        elif isinstance(value, tuple):
            if len(value) == 2:
                return f'Tuple[{self.infer_type(value[0])},{self.infer_type(value[1])}]'
            return 'Tuple'
        return 'Any'

    def type_compatible(self, op: PrimitiveOp, current_type: str) -> bool:
        """Check if operation is type-compatible with current state.

        Args:
            op: Operation to check
            current_type: Current type of the program state

        Returns:
            True if operation can be applied
        """
        if op == PrimitiveOp.STOP:
            return True

        if op not in self.dsl.type_signatures:
            return True  # Unknown operations are allowed

        expected_input = self.dsl.type_signatures[op][0]

        # Polymorphic type matches anything
        if expected_input == 'A':
            return True

        # Numeric types
        if expected_input == 'Num':
            return current_type in ('Int', 'Float', 'Num', 'Any')

        # List types
        if expected_input.startswith('List'):
            if current_type.startswith('List') or current_type == 'Any':
                return True
            return False

        # Boolean
        if expected_input == 'Bool':
            return current_type in ('Bool', 'Any')

        # Integer
        if expected_input == 'Int':
            return current_type in ('Int', 'Num', 'Any')

        return current_type == expected_input or current_type == 'Any'

    def get_output_type(self, op: PrimitiveOp, current_type: str) -> str:
        """Get output type after applying operation.

        Args:
            op: Operation being applied
            current_type: Current type before operation

        Returns:
            Expected output type
        """
        if op not in self.dsl.type_signatures:
            return current_type

        output_type = self.dsl.type_signatures[op][-1]

        # Handle polymorphic types
        if output_type == 'A':
            # For List[A] -> A operations, extract element type
            if current_type.startswith('List[') and current_type.endswith(']'):
                return current_type[5:-1]
            return current_type

        # Handle List[A] output
        if output_type.startswith('List[') and 'A' in output_type:
            if current_type.startswith('List['):
                return current_type
            return f'List[{current_type}]'

        return output_type

    def search(
        self,
        examples: List[Tuple[Any, Any]],
        op_scorer: Callable[[PrimitiveOp, str, List[Tuple[Any, Any]]], float],
        param_generator: Optional[Callable[[PrimitiveOp, torch.Tensor], Any]] = None,
        context: Optional[torch.Tensor] = None,
    ) -> List[Program]:
        """Perform type-guided beam search for programs.

        Args:
            examples: Input-output examples to guide search
            op_scorer: Function that scores operations given current state type
            param_generator: Optional function to generate parameters
            context: Optional context tensor for parameter generation

        Returns:
            List of top-k programs found
        """
        if not examples:
            logger.warning("No examples provided for beam search")
            return []

        # Infer initial type from first input
        initial_type = self.infer_type(examples[0][0])
        logger.debug(f"Starting beam search with initial type: {initial_type}")

        # Initialize beam with empty program
        beam: List[BeamCandidate] = [
            BeamCandidate(
                operations=[],
                score=0.0,
                current_type=initial_type,
                params=[]
            )
        ]

        # Beam search loop
        for step in range(self.max_length):
            all_candidates: List[BeamCandidate] = []

            for candidate in beam:
                # Generate candidates by extending with each valid operation
                for op in self.dsl.get_primitives():
                    if op == PrimitiveOp.STOP:
                        # STOP completes the program
                        all_candidates.append(BeamCandidate(
                            operations=candidate.operations.copy(),
                            score=candidate.score,
                            current_type=candidate.current_type,
                            params=candidate.params.copy()
                        ))
                        continue

                    # Type-guided pruning
                    if not self.type_compatible(op, candidate.current_type):
                        continue

                    # Score the operation
                    op_score = op_scorer(op, candidate.current_type, examples)

                    # Add type consistency bonus
                    if self.type_compatible(op, candidate.current_type):
                        op_score += self.type_bonus

                    new_score = candidate.score + op_score

                    # Prune low-scoring candidates
                    if new_score < self.score_threshold:
                        continue

                    # Generate parameters if generator provided
                    params = None
                    if param_generator is not None and context is not None:
                        params = param_generator(op, context)

                    # Create new candidate
                    new_ops = candidate.operations.copy()
                    new_ops.append((op, params))

                    new_type = self.get_output_type(op, candidate.current_type)

                    all_candidates.append(BeamCandidate(
                        operations=new_ops,
                        score=new_score,
                        current_type=new_type,
                        params=candidate.params + [params]
                    ))

            # Keep top-k candidates
            all_candidates.sort(key=lambda c: c.score, reverse=True)
            beam = all_candidates[:self.beam_width]

            if not beam:
                logger.debug(f"Beam search terminated early at step {step}")
                break

        # Convert to Programs
        programs = []
        for candidate in beam:
            program = Program(
                operations=candidate.operations,
                score=candidate.score,
                metadata={"final_type": candidate.current_type}
            )
            programs.append(program)

        logger.debug(f"Beam search found {len(programs)} programs")
        return programs


# ============================================================================
# 5.3 MDL-Based Program Scoring (Minimum Description Length)
# ============================================================================

class MDLProgramScorer:
    """MDL-based program scorer (5.3).

    Implements Minimum Description Length principle for program scoring:
        MDL(P, D) = -log P(D|P) + L(P)

    Where:
        - P(D|P) is the likelihood of data D given program P
        - L(P) is the description length (complexity) of program P

    Attributes:
        dsl: Domain-specific language for operation costs
        op_costs: Dictionary mapping operations to their description costs
        base_cost: Base cost for any program (header overhead)
        data_noise: Assumed noise level for likelihood calculation
    """

    def __init__(
        self,
        dsl: DomainSpecificLanguage,
        base_cost: float = 1.0,
        data_noise: float = 0.1,
        length_penalty: float = 0.5,
    ) -> None:
        self.dsl = dsl
        self.base_cost = base_cost
        self.data_noise = data_noise
        self.length_penalty = length_penalty

        # Assign costs based on operation complexity
        self.op_costs = self._initialize_op_costs()
        logger.debug("MDLProgramScorer initialized")

    def _initialize_op_costs(self) -> Dict[PrimitiveOp, float]:
        """Initialize operation costs based on complexity.

        Higher-order operations cost more because they're more complex
        to describe (need function arguments, etc.)
        """
        costs = {}

        for op in PrimitiveOp:
            arity = self.dsl.arity.get(op, 0)

            # Base cost depends on arity
            base = math.log2(len(PrimitiveOp))  # bits to encode operation

            # Additional cost for parameters
            if op in (PrimitiveOp.MAP, PrimitiveOp.FILTER, PrimitiveOp.REDUCE):
                # Higher-order: need to describe function
                costs[op] = base + 3.0
            elif op in (PrimitiveOp.FOLD, PrimitiveOp.SCAN, PrimitiveOp.UNFOLD):
                # Even more complex higher-order
                costs[op] = base + 4.0
            elif op in (PrimitiveOp.LOOP, PrimitiveOp.RECURSE):
                # Control flow is expensive
                costs[op] = base + 5.0
            elif arity >= 2:
                # Binary operations need parameter
                costs[op] = base + 1.5
            elif arity == 1:
                # Unary operations are simple
                costs[op] = base + 0.5
            else:
                # Constants/nullary
                costs[op] = base

        return costs

    def program_length(self, program: Program) -> float:
        """Calculate description length of program (5.11).

        L(P) = base_cost + sum(op_cost for each op) + length_penalty * len

        Args:
            program: Program to measure

        Returns:
            Description length in bits
        """
        if not program.operations:
            return self.base_cost

        total_cost = self.base_cost

        for op, params in program.operations:
            # Operation cost
            op_cost = self.op_costs.get(op, 1.0)
            total_cost += op_cost

            # Parameter cost (if present)
            if params is not None:
                total_cost += self._param_cost(params)

        # Length penalty (5.11)
        total_cost += self.length_penalty * len(program.operations)

        return total_cost

    def _param_cost(self, params: Any) -> float:
        """Calculate cost of encoding parameters."""
        if params is None:
            return 0.0
        elif callable(params):
            # Functions are expensive to describe
            return 3.0
        elif isinstance(params, (int, float)):
            # Numbers cost ~log2(|value|) bits
            if params == 0:
                return 1.0
            return max(1.0, math.log2(abs(params) + 1))
        elif isinstance(params, (list, tuple)):
            # Collections: cost per element
            return sum(self._param_cost(p) for p in params) + 1.0
        return 1.0

    def data_likelihood(
        self,
        program: Program,
        examples: List[Tuple[Any, Any]],
        tolerance: float = 1e-6
    ) -> float:
        """Calculate negative log-likelihood of data given program.

        -log P(D|P) approximated as sum of squared errors + noise term

        Args:
            program: Program to evaluate
            examples: Input-output examples
            tolerance: Tolerance for floating point comparison

        Returns:
            Negative log-likelihood (lower is better fit)
        """
        if not examples:
            return 0.0

        total_nll = 0.0
        execution_context = ExecutionContext()

        for inp, expected in examples:
            try:
                actual = program.execute(inp, execution_context)

                # Calculate error
                error = self._compute_error(actual, expected, tolerance)

                # Convert to negative log-likelihood
                # Using Gaussian noise model: -log P(y|x,P) ~ error^2 / (2*sigma^2)
                nll = (error ** 2) / (2 * self.data_noise ** 2)
                total_nll += nll

            except Exception as e:
                # Execution failure = very high penalty
                logger.debug(f"Execution failed: {e}")
                total_nll += 100.0  # Large penalty

        return total_nll

    def _compute_error(self, actual: Any, expected: Any, tolerance: float) -> float:
        """Compute error between actual and expected output."""
        if actual is None:
            return 10.0  # Penalty for None output

        # Numeric comparison
        if isinstance(expected, (int, float)) and isinstance(actual, (int, float)):
            return abs(actual - expected)

        # List comparison
        if isinstance(expected, list) and isinstance(actual, list):
            if len(expected) != len(actual):
                return abs(len(expected) - len(actual)) * 2.0

            total_error = 0.0
            for a, e in zip(actual, expected):
                total_error += self._compute_error(a, e, tolerance)
            return total_error

        # Boolean comparison
        if isinstance(expected, bool) and isinstance(actual, bool):
            return 0.0 if actual == expected else 1.0

        # Generic equality
        if actual == expected:
            return 0.0
        return 5.0  # Mismatch penalty

    def score(
        self,
        program: Program,
        examples: List[Tuple[Any, Any]]
    ) -> float:
        """Calculate MDL score for program (5.3).

        MDL = -log P(D|P) + L(P)

        Lower MDL is better (more compressed representation).

        Args:
            program: Program to score
            examples: Input-output examples

        Returns:
            MDL score (lower is better)
        """
        # Data likelihood term
        data_term = self.data_likelihood(program, examples)

        # Program complexity term
        complexity_term = self.program_length(program)

        mdl = data_term + complexity_term

        logger.debug(
            f"MDL score: {mdl:.3f} (data: {data_term:.3f}, complexity: {complexity_term:.3f})"
        )

        return mdl

    def score_normalized(
        self,
        program: Program,
        examples: List[Tuple[Any, Any]]
    ) -> float:
        """Calculate normalized score (higher is better).

        Converts MDL to a score in [0, 1] range suitable for
        comparison with other scoring methods.

        Args:
            program: Program to score
            examples: Input-output examples

        Returns:
            Normalized score in [0, 1] (higher is better)
        """
        mdl = self.score(program, examples)
        # Convert: lower MDL -> higher score
        # Using sigmoid-like transformation
        return 1.0 / (1.0 + mdl / 10.0)


# ============================================================================
# 5.11 Program Complexity Metrics
# ============================================================================

class ProgramComplexity:
    """Program complexity measurement with MDL regularization (5.11).

    Provides various complexity metrics for regularizing program synthesis
    to prefer simpler programs.

    Attributes:
        mdl_scorer: MDL scorer for complexity calculation
        length_weight: Weight for length penalty
        depth_weight: Weight for nesting depth penalty
        uniqueness_weight: Weight for operation uniqueness bonus
    """

    def __init__(
        self,
        dsl: DomainSpecificLanguage,
        length_weight: float = 1.0,
        depth_weight: float = 0.5,
        uniqueness_weight: float = 0.2,
    ) -> None:
        self.dsl = dsl
        self.mdl_scorer = MDLProgramScorer(dsl)
        self.length_weight = length_weight
        self.depth_weight = depth_weight
        self.uniqueness_weight = uniqueness_weight

    def length_penalty(self, program: Program) -> float:
        """Calculate length penalty (5.11).

        Penalizes longer programs to encourage conciseness.

        Args:
            program: Program to measure

        Returns:
            Length penalty (higher = more complex)
        """
        n = len(program.operations)
        # Quadratic penalty to strongly discourage very long programs
        return self.length_weight * (n + 0.1 * n * n)

    def nesting_depth(self, program: Program) -> int:
        """Calculate maximum nesting depth of program.

        Higher-order operations like MAP and FOLD increase nesting.

        Args:
            program: Program to analyze

        Returns:
            Maximum nesting depth
        """
        nesting_ops = {
            PrimitiveOp.MAP, PrimitiveOp.FILTER, PrimitiveOp.REDUCE,
            PrimitiveOp.FOLD, PrimitiveOp.SCAN, PrimitiveOp.UNFOLD,
            PrimitiveOp.LOOP, PrimitiveOp.RECURSE
        }

        depth = 0
        for op, _ in program.operations:
            if op in nesting_ops:
                depth += 1

        return depth

    def operation_diversity(self, program: Program) -> float:
        """Calculate operation diversity score.

        Diverse programs (using many different operations) may be
        more expressive but also more complex.

        Args:
            program: Program to analyze

        Returns:
            Diversity score in [0, 1]
        """
        if not program.operations:
            return 0.0

        unique_ops = set(op for op, _ in program.operations)
        return len(unique_ops) / len(program.operations)

    def total_complexity(self, program: Program) -> float:
        """Calculate total complexity score (5.11).

        Combines multiple complexity metrics:
        - Length penalty
        - Nesting depth penalty
        - MDL-based complexity

        Args:
            program: Program to measure

        Returns:
            Total complexity score
        """
        length_term = self.length_penalty(program)
        depth_term = self.depth_weight * self.nesting_depth(program)
        mdl_term = self.mdl_scorer.program_length(program)

        # Diversity can reduce complexity (reusing operations is simpler)
        diversity_bonus = self.uniqueness_weight * (1.0 - self.operation_diversity(program))

        total = length_term + depth_term + mdl_term - diversity_bonus

        return max(0.0, total)

    def regularization_loss(
        self,
        program_scores: torch.Tensor,
        program_lengths: torch.Tensor,
        lambda_mdl: float = 0.1
    ) -> torch.Tensor:
        """Calculate MDL regularization loss for training (5.11).

        Encourages learning to generate shorter programs.

        Args:
            program_scores: Batch of program scores [B]
            program_lengths: Batch of program lengths [B]
            lambda_mdl: Regularization strength

        Returns:
            Regularization loss term
        """
        # Length penalty: quadratic in program length
        length_penalty = program_lengths.float() ** 2

        # MDL regularization: penalize long programs
        mdl_reg = lambda_mdl * length_penalty.mean()

        return mdl_reg


# ============================================================================
# 5.4 Behavioral Specifications and Noisy Examples
# ============================================================================

@dataclass
class BehavioralSpec:
    """Behavioral specification for program synthesis (5.4).

    Defines pre/post conditions and invariants that a synthesized
    program must satisfy.

    Attributes:
        preconditions: List of (name, predicate) for input validation
        postconditions: List of (name, predicate) for output validation
        invariants: List of (name, predicate) that must hold throughout
        tolerance: Tolerance for numeric comparisons
        noise_model: Type of noise model ('gaussian', 'uniform', 'outlier')
    """
    preconditions: List[Tuple[str, Callable[[Any], bool]]] = field(default_factory=list)
    postconditions: List[Tuple[str, Callable[[Any, Any], bool]]] = field(default_factory=list)
    invariants: List[Tuple[str, Callable[[Any], bool]]] = field(default_factory=list)
    tolerance: float = 1e-3
    noise_model: str = "gaussian"

    def check_preconditions(self, input_data: Any) -> Tuple[bool, List[str]]:
        """Check all preconditions on input.

        Args:
            input_data: Input to validate

        Returns:
            Tuple of (all_passed, list of failed condition names)
        """
        failures = []
        for name, pred in self.preconditions:
            try:
                if not pred(input_data):
                    failures.append(name)
            except Exception as e:
                logger.debug(f"Precondition {name} raised: {e}")
                failures.append(name)
        return len(failures) == 0, failures

    def check_postconditions(self, input_data: Any, output_data: Any) -> Tuple[bool, List[str]]:
        """Check all postconditions on output.

        Args:
            input_data: Original input
            output_data: Program output

        Returns:
            Tuple of (all_passed, list of failed condition names)
        """
        failures = []
        for name, pred in self.postconditions:
            try:
                if not pred(input_data, output_data):
                    failures.append(name)
            except Exception as e:
                logger.debug(f"Postcondition {name} raised: {e}")
                failures.append(name)
        return len(failures) == 0, failures


class NoisyExampleHandler:
    """Handler for noisy examples in program synthesis (5.4).

    Provides robust synthesis by:
    - Detecting and filtering outliers
    - Weighted example scoring based on confidence
    - Tolerance-based matching for numeric outputs

    Attributes:
        tolerance: Base tolerance for numeric matching
        outlier_threshold: Z-score threshold for outlier detection
        min_confidence: Minimum confidence to include example
    """

    def __init__(
        self,
        tolerance: float = 0.1,
        outlier_threshold: float = 2.5,
        min_confidence: float = 0.1,
    ) -> None:
        self.tolerance = tolerance
        self.outlier_threshold = outlier_threshold
        self.min_confidence = min_confidence
        logger.debug("NoisyExampleHandler initialized")

    def compute_confidences(
        self,
        examples: List[Tuple[Any, Any]]
    ) -> List[float]:
        """Compute confidence scores for each example.

        Uses variance analysis to detect potential outliers.

        Args:
            examples: List of (input, output) pairs

        Returns:
            List of confidence scores in [0, 1]
        """
        if len(examples) < 3:
            # Not enough data for outlier detection
            return [1.0] * len(examples)

        # Extract numeric features for outlier detection
        try:
            features = self._extract_features(examples)
            if features is None:
                return [1.0] * len(examples)

            # Compute z-scores
            mean = sum(features) / len(features)
            variance = sum((f - mean) ** 2 for f in features) / len(features)
            std = math.sqrt(variance + 1e-8)

            confidences = []
            for f in features:
                z_score = abs(f - mean) / std
                # Convert z-score to confidence
                if z_score > self.outlier_threshold:
                    conf = max(self.min_confidence, 1.0 - (z_score - self.outlier_threshold) / 2.0)
                else:
                    conf = 1.0
                confidences.append(conf)

            return confidences

        except Exception as e:
            logger.debug(f"Confidence computation failed: {e}")
            return [1.0] * len(examples)

    def _extract_features(self, examples: List[Tuple[Any, Any]]) -> Optional[List[float]]:
        """Extract numeric features from examples for outlier detection."""
        features = []
        for _, output in examples:
            if isinstance(output, (int, float)):
                features.append(float(output))
            elif isinstance(output, list) and output:
                if isinstance(output[0], (int, float)):
                    features.append(sum(output) / len(output))
                else:
                    features.append(float(len(output)))
            else:
                return None  # Can't extract numeric features
        return features

    def filter_outliers(
        self,
        examples: List[Tuple[Any, Any]]
    ) -> List[Tuple[Any, Any]]:
        """Filter out likely outlier examples.

        Args:
            examples: Original examples

        Returns:
            Filtered examples with outliers removed
        """
        confidences = self.compute_confidences(examples)

        filtered = [
            ex for ex, conf in zip(examples, confidences)
            if conf >= self.min_confidence
        ]

        n_filtered = len(examples) - len(filtered)
        if n_filtered > 0:
            logger.debug(f"Filtered {n_filtered} outlier examples")

        return filtered if filtered else examples  # Return original if all filtered

    def weighted_match(
        self,
        actual: Any,
        expected: Any,
        confidence: float = 1.0
    ) -> float:
        """Calculate weighted match score between actual and expected.

        Args:
            actual: Actual program output
            expected: Expected output
            confidence: Confidence in this example

        Returns:
            Match score in [0, 1]
        """
        if actual is None:
            return 0.0

        # Numeric matching with tolerance
        if isinstance(expected, (int, float)) and isinstance(actual, (int, float)):
            error = abs(actual - expected)
            max_error = max(self.tolerance, abs(expected) * self.tolerance)
            if error <= max_error:
                return confidence
            return confidence * max(0.0, 1.0 - error / (max_error * 10))

        # List matching
        if isinstance(expected, list) and isinstance(actual, list):
            if len(expected) != len(actual):
                # Length mismatch penalty
                len_penalty = abs(len(expected) - len(actual)) / max(len(expected), 1)
                return confidence * max(0.0, 1.0 - len_penalty)

            if not expected:
                return confidence

            element_scores = [
                self.weighted_match(a, e, 1.0)
                for a, e in zip(actual, expected)
            ]
            return confidence * (sum(element_scores) / len(element_scores))

        # Exact match for other types
        return confidence if actual == expected else 0.0


class NeuralProgramEncoder(nn.Module):
    """Encode programs as vectors."""
    
    def __init__(
        self,
        num_ops: int,
        embedding_dim: int = 64,
        hidden_size: int = 128,
    ):
        super().__init__()
        
        # Operation embeddings
        self.op_embeddings = nn.Embedding(num_ops, embedding_dim)
        
        # Program encoder (LSTM over operations)
        self.program_encoder = nn.LSTM(
            embedding_dim,
            hidden_size,
            batch_first=True
        )
        
    def forward(self, program_ops: torch.Tensor) -> torch.Tensor:
        """Encode program to vector.
        
        Args:
            program_ops: [B, max_length] tensor of operation indices
            
        Returns:
            program_embedding: [B, hidden_size]
        """
        # Embed operations
        op_embeddings = self.op_embeddings(program_ops)
        
        # Encode with LSTM
        _, (h_n, _) = self.program_encoder(op_embeddings)
        
        return h_n.squeeze(0)


class NeuralProgramSampler(nn.Module):
    """Sample programs guided by neural network."""
    
    def __init__(
        self,
        example_dim: int,
        num_ops: int,
        max_program_length: int = 10,
        hidden_size: int = 256,
    ):
        super().__init__()
        self.max_program_length = max_program_length
        self.num_ops = num_ops
        
        # Example encoder
        self.example_encoder = nn.Sequential(
            nn.Linear(example_dim * 2, hidden_size),  # input + output
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size)
        )
        
        # Program generator (autoregressive)
        self.program_generator = nn.LSTM(
            hidden_size,
            hidden_size,
            batch_first=True
        )
        
        # Operation predictor
        self.op_predictor = nn.Linear(hidden_size, num_ops)
        
    def forward(
        self,
        examples: List[Tuple[torch.Tensor, torch.Tensor]],
        temperature: float = 1.0
    ) -> torch.Tensor:
        """Sample program conditioned on examples.
        
        Args:
            examples: List of (input, output) pairs
            temperature: Sampling temperature
            
        Returns:
            program_ops: [max_length] tensor of operation indices
        """
        # Encode examples
        example_encodings = []
        for input_ex, output_ex in examples:
            combined = torch.cat([input_ex, output_ex], dim=-1)
            encoded = self.example_encoder(combined)
            example_encodings.append(encoded)
        
        # Aggregate examples
        example_context = torch.stack(example_encodings).mean(dim=0, keepdim=True)
        
        # Generate program autoregressively
        program_ops = []
        hidden = None
        
        for _ in range(self.max_program_length):
            # Forward through LSTM
            if hidden is None:
                output, hidden = self.program_generator(example_context)
            else:
                output, hidden = self.program_generator(example_context, hidden)
            
            # Predict next operation
            op_logits = self.op_predictor(output.squeeze(0))
            op_probs = F.softmax(op_logits / temperature, dim=-1)
            
            # Sample operation
            op_idx = torch.multinomial(op_probs, 1).item()
            program_ops.append(op_idx)
            
            # Stop token (assume op 0 is stop)
            if op_idx == 0:
                break
        
        return torch.tensor(program_ops)


class ProgramVerifier:
    """Verify programs by execution."""
    
    def __init__(self, dsl: DomainSpecificLanguage):
        self.dsl = dsl
        
    def verify(
        self,
        program: Program,
        examples: List[Tuple[Any, Any]],
        tolerance: float = 1e-3
    ) -> bool:
        """Verify program correctness on examples."""
        for input_ex, expected_output in examples:
            try:
                actual_output = program.execute(input_ex)
                
                # Check if outputs match
                if isinstance(expected_output, (int, float)):
                    if abs(actual_output - expected_output) > tolerance:
                        return False
                elif isinstance(expected_output, list):
                    if len(actual_output) != len(expected_output):
                        return False
                    for a, e in zip(actual_output, expected_output):
                        if abs(a - e) > tolerance:
                            return False
                else:
                    if actual_output != expected_output:
                        return False
            except Exception:
                return False
        
        return True
    
    def score_program(
        self,
        program: Program,
        examples: List[Tuple[Any, Any]]
    ) -> float:
        """Score program based on accuracy and complexity."""
        # Accuracy score
        correct = 0
        total = len(examples)
        
        for input_ex, expected_output in examples:
            try:
                actual_output = program.execute(input_ex)
                if actual_output == expected_output:
                    correct += 1
            except Exception:
                pass
        
        accuracy = correct / max(total, 1)
        
        # Simplicity bonus (shorter programs are better)
        simplicity = 1.0 / (1.0 + len(program.operations))
        
        # Combined score
        score = accuracy + 0.1 * simplicity
        
        return score


class ParameterSynthesizer(nn.Module):
    """Synthesize parameters for DSL operations instead of using identity functions.

    This module learns to generate appropriate parameters for each operation type:
    - Numeric parameters (for ADD, MULTIPLY, TAKE, etc.)
    - Function parameters (for MAP, FILTER, REDUCE)
    - Constants (for initializers)
    """

    def __init__(
        self,
        context_dim: int,
        hidden_size: int = 128,
        num_numeric_buckets: int = 20,  # Discretized numeric values
    ):
        super().__init__()
        self.context_dim = context_dim
        self.hidden_size = hidden_size
        self.num_numeric_buckets = num_numeric_buckets

        # Numeric parameter predictor (for ADD, MULTIPLY, TAKE, DROP, INDEX)
        self.numeric_predictor = nn.Sequential(
            nn.Linear(context_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_numeric_buckets)
        )

        # Numeric value mapping: bucket -> actual value
        # Range: [-10, 10] discretized into buckets
        self.register_buffer(
            'numeric_values',
            torch.linspace(-10, 10, num_numeric_buckets)
        )

        # Function body generator for MAP/FILTER/REDUCE
        # Generates coefficients for simple linear functions: f(x) = ax + b
        self.function_generator = nn.Sequential(
            nn.Linear(context_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 2)  # [a, b] for ax + b
        )

        # Predicate generator for FILTER (outputs threshold for comparison)
        self.predicate_generator = nn.Sequential(
            nn.Linear(context_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 2)  # [threshold, comparison_type]
        )

        # Binary function generator for REDUCE/FOLD
        self.binary_fn_generator = nn.Sequential(
            nn.Linear(context_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 4)  # Operation type selector
        )

        # Operation type to parameter type mapping
        self.param_types = {
            # Numeric parameters
            PrimitiveOp.ADD: 'numeric',
            PrimitiveOp.SUBTRACT: 'numeric',
            PrimitiveOp.MULTIPLY: 'numeric',
            PrimitiveOp.DIVIDE: 'numeric',
            PrimitiveOp.MODULO: 'numeric',
            PrimitiveOp.TAKE: 'numeric_int',
            PrimitiveOp.DROP: 'numeric_int',
            PrimitiveOp.INDEX: 'numeric_int',
            PrimitiveOp.MAX: 'numeric',
            PrimitiveOp.MIN: 'numeric',
            PrimitiveOp.GREATER: 'numeric',
            PrimitiveOp.LESS: 'numeric',
            # Function parameters
            PrimitiveOp.MAP: 'unary_fn',
            PrimitiveOp.FILTER: 'predicate',
            PrimitiveOp.REDUCE: 'binary_fn',
            PrimitiveOp.FOLD: 'binary_fn',
            PrimitiveOp.SCAN: 'binary_fn',
            # List parameters
            PrimitiveOp.ZIP: 'list',
            PrimitiveOp.CONCAT: 'list',
            # No parameters needed
            PrimitiveOp.REVERSE: 'none',
            PrimitiveOp.SORT: 'none',
            PrimitiveOp.HEAD: 'none',
            PrimitiveOp.TAIL: 'none',
            PrimitiveOp.LENGTH: 'none',
            PrimitiveOp.ABS: 'none',
            PrimitiveOp.NEGATE: 'none',
            PrimitiveOp.NOT: 'none',
            PrimitiveOp.IDENTITY: 'none',
        }

    def synthesize(
        self,
        op: PrimitiveOp,
        context: torch.Tensor,
        examples: Optional[List[Tuple[Any, Any]]] = None
    ) -> Any:
        """Synthesize parameter for given operation based on context.

        Args:
            op: The operation that needs a parameter
            context: Context embedding from examples [hidden_size]
            examples: Optional I/O examples for refinement

        Returns:
            Appropriate parameter for the operation
        """
        param_type = self.param_types.get(op, 'none')

        if param_type == 'none':
            return None

        elif param_type == 'numeric':
            return self._synthesize_numeric(context)

        elif param_type == 'numeric_int':
            return self._synthesize_numeric_int(context)

        elif param_type == 'unary_fn':
            return self._synthesize_unary_function(context)

        elif param_type == 'predicate':
            return self._synthesize_predicate(context)

        elif param_type == 'binary_fn':
            return self._synthesize_binary_function(context)

        elif param_type == 'list':
            # For list params, try to infer from examples
            if examples and len(examples) > 0:
                return self._infer_list_param(examples)
            return []

        return None

    def _synthesize_numeric(self, context: torch.Tensor) -> float:
        """Generate a numeric parameter."""
        logits = self.numeric_predictor(context)
        probs = F.softmax(logits, dim=-1)
        bucket_idx = torch.multinomial(probs, 1).item()
        return self.numeric_values[bucket_idx].item()

    def _synthesize_numeric_int(self, context: torch.Tensor) -> int:
        """Generate an integer parameter (0-10 range)."""
        logits = self.numeric_predictor(context)
        probs = F.softmax(logits[:self.num_numeric_buckets // 2], dim=-1)
        bucket_idx = torch.multinomial(probs, 1).item()
        return max(0, int(bucket_idx))

    def _synthesize_unary_function(self, context: torch.Tensor) -> Callable:
        """Generate a unary function f(x) = ax + b."""
        params = self.function_generator(context)
        a, b = params[0].item(), params[1].item()

        # Create function with captured parameters
        def generated_fn(x):
            if isinstance(x, (int, float)):
                return a * x + b
            return x
        return generated_fn

    def _synthesize_predicate(self, context: torch.Tensor) -> Callable:
        """Generate a predicate function for filtering."""
        params = self.predicate_generator(context)
        threshold = params[0].item()
        comparison_type = int(torch.sigmoid(params[1]).item() > 0.5)

        def generated_predicate(x):
            if isinstance(x, (int, float)):
                if comparison_type == 0:
                    return x > threshold
                else:
                    return x < threshold
            return True
        return generated_predicate

    def _synthesize_binary_function(self, context: torch.Tensor) -> Callable:
        """Generate a binary function for reduce/fold operations."""
        logits = self.binary_fn_generator(context)
        op_idx = torch.argmax(logits).item()

        # Different binary operations
        binary_ops = [
            lambda a, b: a + b,  # Sum
            lambda a, b: a * b,  # Product
            lambda a, b: max(a, b) if isinstance(a, (int, float)) and isinstance(b, (int, float)) else a,  # Max
            lambda a, b: min(a, b) if isinstance(a, (int, float)) and isinstance(b, (int, float)) else a,  # Min
        ]
        return binary_ops[op_idx % len(binary_ops)]

    def _infer_list_param(self, examples: List[Tuple[Any, Any]]) -> List:
        """Try to infer a list parameter from examples."""
        # Simple heuristic: if output is longer than input, try to find what was added
        for inp, out in examples:
            if isinstance(inp, list) and isinstance(out, list):
                if len(out) > len(inp):
                    # Return the difference
                    return out[len(inp):]
        return []


class ProgramSynthesizer(nn.Module):
    """Synthesize programs from input-output examples with actual parameter synthesis.

    Enhanced with:
    - Type-guided beam search (5.2)
    - MDL-based scoring (5.3)
    - Behavioral specs and noisy example handling (5.4)
    - Program complexity regularization (5.11)
    - Safe sandboxed execution (5.12)

    Attributes:
        example_dim: Dimension of example encodings
        max_program_length: Maximum program length
        num_candidates: Number of candidate programs to consider
        use_beam_search: Whether to use type-guided beam search
        use_mdl_scoring: Whether to use MDL-based scoring
    """

    def __init__(
        self,
        example_dim: int,
        max_program_length: int = 10,
        num_candidates: int = 10,
        beam_width: int = 10,
        use_beam_search: bool = True,
        use_mdl_scoring: bool = True,
    ):
        super().__init__()
        self.example_dim = example_dim
        self.max_program_length = max_program_length
        self.num_candidates = num_candidates
        self.use_beam_search = use_beam_search
        self.use_mdl_scoring = use_mdl_scoring

        # DSL
        self.dsl = DomainSpecificLanguage()
        self.num_ops = len(self.dsl.get_primitives())

        # Neural sampler
        self.sampler = NeuralProgramSampler(
            example_dim=example_dim,
            num_ops=self.num_ops,
            max_program_length=max_program_length
        )

        # Parameter synthesizer (NEW - replaces identity functions)
        self.param_synthesizer = ParameterSynthesizer(
            context_dim=example_dim,
            hidden_size=256
        )

        # Verifier
        self.verifier = ProgramVerifier(self.dsl)

        # Context encoder for examples
        self.context_encoder = nn.Sequential(
            nn.Linear(example_dim * 2, example_dim),
            nn.ReLU(),
            nn.Linear(example_dim, example_dim)
        )

        # Type-guided beam search (5.2)
        self.beam_search = TypeGuidedBeamSearch(
            dsl=self.dsl,
            beam_width=beam_width,
            max_length=max_program_length,
        )

        # MDL scorer (5.3)
        self.mdl_scorer = MDLProgramScorer(dsl=self.dsl)

        # Program complexity metrics (5.11)
        self.complexity_analyzer = ProgramComplexity(dsl=self.dsl)

        # Noisy example handler (5.4)
        self.noise_handler = NoisyExampleHandler()

        # Operation scorer network for beam search
        self.op_scorer_net = nn.Sequential(
            nn.Linear(example_dim + self.num_ops, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

        logger.debug("ProgramSynthesizer initialized with enhanced features")

    def synthesize_from_examples(
        self,
        examples: List[Tuple[torch.Tensor, torch.Tensor]],
        num_iterations: int = 100,
        behavioral_spec: Optional[BehavioralSpec] = None,
        filter_noisy: bool = True,
    ) -> Optional[Program]:
        """Synthesize program from examples with enhanced synthesis (5.2-5.4).

        Args:
            examples: List of (input, output) tensor pairs
            num_iterations: Number of synthesis attempts
            behavioral_spec: Optional behavioral specification (5.4)
            filter_noisy: Whether to filter noisy examples (5.4)

        Returns:
            Best program found, or None
        """
        if not examples:
            logger.warning("No examples provided for synthesis")
            return None

        # Convert to data examples
        data_examples = self._tensors_to_data(examples)

        # Filter noisy examples (5.4)
        if filter_noisy:
            data_examples = self.noise_handler.filter_outliers(data_examples)

        # Compute example confidences for weighted scoring (5.4)
        confidences = self.noise_handler.compute_confidences(data_examples)

        # Encode examples into context for parameter synthesis
        context = self._encode_examples(examples)

        best_program = None
        best_score = float('inf') if self.use_mdl_scoring else -float('inf')

        if self.use_beam_search:
            # Use type-guided beam search (5.2)
            programs = self.beam_search.search(
                examples=data_examples,
                op_scorer=lambda op, t, ex: self._score_operation(op, t, context),
                param_generator=self.param_synthesizer.synthesize,
                context=context,
            )

            # Score all programs and find best
            for program in programs:
                score = self._score_program(program, data_examples, confidences, behavioral_spec)

                if self.use_mdl_scoring:
                    # MDL: lower is better
                    if score < best_score:
                        best_score = score
                        best_program = program
                        program.score = -score  # Store as higher-is-better for compatibility
                else:
                    # Standard: higher is better
                    if score > best_score:
                        best_score = score
                        best_program = program
                        program.score = score
        else:
            # Fallback to sampling-based synthesis
            for _ in range(num_iterations):
                # Sample candidate program operations
                program_ops = self.sampler(examples, temperature=1.0)

                # Convert to Program with REAL parameters (not identity)
                program = self._ops_to_program(program_ops, context, examples)

                # Score program
                score = self._score_program(program, data_examples, confidences, behavioral_spec)

                if self.use_mdl_scoring:
                    if score < best_score:
                        best_score = score
                        best_program = program
                        program.score = -score
                else:
                    if score > best_score:
                        best_score = score
                        best_program = program
                        program.score = score

                # Early stopping
                if self.use_mdl_scoring and score < 1.0:
                    break
                elif not self.use_mdl_scoring and score >= 0.99:
                    break

        # Add complexity info to best program
        if best_program is not None:
            best_program.complexity = self.complexity_analyzer.total_complexity(best_program)
            logger.debug(f"Best program complexity: {best_program.complexity:.3f}")

        return best_program

    def _score_operation(
        self,
        op: PrimitiveOp,
        current_type: str,
        context: torch.Tensor
    ) -> float:
        """Score an operation for beam search.

        Args:
            op: Operation to score
            current_type: Current type state
            context: Encoded example context

        Returns:
            Score for this operation
        """
        # Create one-hot encoding for operation
        op_idx = list(self.dsl.primitives.values()).index(op)
        op_onehot = torch.zeros(self.num_ops)
        op_onehot[op_idx] = 1.0

        # Concatenate with context
        with torch.no_grad():
            combined = torch.cat([context, op_onehot])
            score = self.op_scorer_net(combined).item()

        return score

    def _score_program(
        self,
        program: Program,
        examples: List[Tuple[Any, Any]],
        confidences: List[float],
        behavioral_spec: Optional[BehavioralSpec] = None,
    ) -> float:
        """Score a program using MDL or standard scoring.

        Args:
            program: Program to score
            examples: Data examples
            confidences: Confidence weights for examples (5.4)
            behavioral_spec: Optional behavioral specification (5.4)

        Returns:
            Program score (interpretation depends on use_mdl_scoring)
        """
        if self.use_mdl_scoring:
            # MDL scoring (5.3) - lower is better
            base_score = self.mdl_scorer.score(program, examples)

            # Behavioral spec penalty (5.4)
            if behavioral_spec is not None:
                spec_penalty = self._check_behavioral_spec(program, examples, behavioral_spec)
                base_score += spec_penalty * 10.0  # Weight for spec violations

            return base_score
        else:
            # Standard accuracy-based scoring
            total_score = 0.0
            total_weight = 0.0

            execution_context = ExecutionContext()

            for (inp, expected), conf in zip(examples, confidences):
                try:
                    # Check preconditions (5.4)
                    if behavioral_spec:
                        passed, _ = behavioral_spec.check_preconditions(inp)
                        if not passed:
                            continue

                    actual = program.execute(inp, execution_context)

                    # Weighted match score (5.4)
                    match_score = self.noise_handler.weighted_match(actual, expected, conf)

                    # Check postconditions (5.4)
                    if behavioral_spec:
                        passed, _ = behavioral_spec.check_postconditions(inp, actual)
                        if not passed:
                            match_score *= 0.5  # Penalty for postcondition failure

                    total_score += match_score
                    total_weight += conf

                except Exception:
                    pass

            if total_weight == 0:
                return 0.0

            accuracy = total_score / total_weight

            # Simplicity bonus (5.11)
            complexity_penalty = self.complexity_analyzer.total_complexity(program) / 100.0
            return accuracy - complexity_penalty * 0.1

    def _check_behavioral_spec(
        self,
        program: Program,
        examples: List[Tuple[Any, Any]],
        spec: BehavioralSpec
    ) -> float:
        """Check behavioral specification compliance.

        Args:
            program: Program to check
            examples: Data examples
            spec: Behavioral specification

        Returns:
            Violation count (0 = fully compliant)
        """
        violations = 0
        execution_context = ExecutionContext()

        for inp, expected in examples:
            # Check preconditions
            pre_passed, _ = spec.check_preconditions(inp)
            if not pre_passed:
                continue  # Skip invalid inputs

            try:
                actual = program.execute(inp, execution_context)

                # Check postconditions
                post_passed, failures = spec.check_postconditions(inp, actual)
                if not post_passed:
                    violations += len(failures)

            except Exception:
                violations += 1

        return float(violations)

    def _encode_examples(
        self,
        examples: List[Tuple[torch.Tensor, torch.Tensor]]
    ) -> torch.Tensor:
        """Encode I/O examples into context vector for parameter synthesis."""
        if not examples:
            return torch.zeros(self.sampler.hidden_size if hasattr(self.sampler, 'hidden_size') else 256)

        encodings = []
        for inp, out in examples:
            # Flatten and pad/truncate to fixed size
            inp_flat = inp.flatten()
            out_flat = out.flatten()

            # Pad to example_dim
            target_dim = self.context_encoder[0].in_features // 2
            inp_padded = F.pad(inp_flat, (0, max(0, target_dim - len(inp_flat))))[:target_dim]
            out_padded = F.pad(out_flat, (0, max(0, target_dim - len(out_flat))))[:target_dim]

            combined = torch.cat([inp_padded, out_padded])
            encodings.append(combined)

        # Aggregate examples
        stacked = torch.stack(encodings)
        aggregated = stacked.mean(dim=0)

        # Encode
        context = self.context_encoder(aggregated)
        return context

    def _ops_to_program(
        self,
        ops_tensor: torch.Tensor,
        context: torch.Tensor,
        examples: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None
    ) -> Program:
        """Convert operation indices to Program with synthesized parameters."""
        operations = []
        data_examples = self._tensors_to_data(examples) if examples else None

        for op_idx in ops_tensor.tolist():
            if op_idx == 0:  # Stop token
                break

            # Map index to operation
            primitives_list = list(self.dsl.primitives.keys())
            op_name = primitives_list[op_idx % len(primitives_list)]
            op_enum = self.dsl.primitives[op_name]

            # SYNTHESIZE ACTUAL PARAMETERS (not identity function!)
            params = self.param_synthesizer.synthesize(
                op=op_enum,
                context=context,
                examples=data_examples
            )

            operations.append((op_enum, params))

        return Program(operations)
    
    def _tensors_to_data(self, examples: List[Tuple[torch.Tensor, torch.Tensor]]) -> List[Tuple[Any, Any]]:
        """Convert tensor examples to executable data."""
        return [
            (inp.tolist() if inp.dim() > 0 else inp.item(),
             out.tolist() if out.dim() > 0 else out.item())
            for inp, out in examples
        ]


class SymbolicReasoner(nn.Module):
    """Symbolic reasoning with logic rules."""
    
    def __init__(
        self,
        num_predicates: int = 50,
        num_entities: int = 100,
        embedding_dim: int = 64,
    ):
        super().__init__()
        
        # Entity embeddings
        self.entity_embeddings = nn.Embedding(num_entities, embedding_dim)
        
        # Predicate embeddings
        self.predicate_embeddings = nn.Embedding(num_predicates, embedding_dim)
        
        # Rule encoder
        self.rule_encoder = nn.Sequential(
            nn.Linear(embedding_dim * 3, embedding_dim * 2),
            nn.ReLU(),
            nn.Linear(embedding_dim * 2, 1),
            nn.Sigmoid()
        )
        
    def evaluate_rule(
        self,
        predicate: torch.Tensor,
        subject: torch.Tensor,
        object: torch.Tensor
    ) -> torch.Tensor:
        """Evaluate truth value of rule P(S, O)."""
        # Embed components
        p_embed = self.predicate_embeddings(predicate)
        s_embed = self.entity_embeddings(subject)
        o_embed = self.entity_embeddings(object)
        
        # Concatenate and evaluate
        combined = torch.cat([p_embed, s_embed, o_embed], dim=-1)
        truth_value = self.rule_encoder(combined)
        
        return truth_value
    
    def forward_chain(
        self,
        facts: List[Tuple[int, int, int]],
        rules: List[Tuple[int, int, int]],
        max_iterations: int = 10
    ) -> List[Tuple[int, int, int]]:
        """Forward chaining inference."""
        derived_facts = set(facts)
        
        for _ in range(max_iterations):
            new_facts = False
            
            for rule_p, rule_s, rule_o in rules:
                # Try to apply rule
                for fact_p, fact_s, fact_o in list(derived_facts):
                    # Simple pattern matching (can be extended)
                    if fact_p == rule_s:  # Fact matches rule premise
                        new_fact = (rule_p, fact_s, rule_o)
                        if new_fact not in derived_facts:
                            derived_facts.add(new_fact)
                            new_facts = True
            
            if not new_facts:
                break
        
        return list(derived_facts)


class NeuroSymbolicIntegration(nn.Module):
    """Integrate neural and symbolic reasoning."""
    
    def __init__(
        self,
        hidden_size: int,
        num_predicates: int = 50,
        num_entities: int = 100,
    ):
        super().__init__()
        
        # Neural component
        self.neural_encoder = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size)
        )
        
        # Symbolic component
        self.symbolic_reasoner = SymbolicReasoner(
            num_predicates=num_predicates,
            num_entities=num_entities
        )
        
        # Integration layer
        self.integrator = nn.Sequential(
            nn.Linear(hidden_size + 1, hidden_size),  # +1 for symbolic truth value
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size)
        )
        
    def forward(
        self,
        neural_input: torch.Tensor,
        symbolic_facts: Optional[List[Tuple[int, int, int]]] = None
    ) -> torch.Tensor:
        """Combine neural and symbolic reasoning."""
        # Neural processing
        neural_output = self.neural_encoder(neural_input)
        
        # Symbolic reasoning (if facts provided)
        if symbolic_facts:
            # Extract symbolic features
            symbolic_scores = []
            for predicate, subject, obj in symbolic_facts:
                p = torch.tensor([predicate], device=neural_input.device)
                s = torch.tensor([subject], device=neural_input.device)
                o = torch.tensor([obj], device=neural_input.device)
                
                score = self.symbolic_reasoner.evaluate_rule(p, s, o)
                symbolic_scores.append(score)
            
            # Aggregate symbolic scores
            symbolic_output = torch.stack(symbolic_scores).mean().unsqueeze(0).expand(neural_output.size(0), 1)
        else:
            symbolic_output = torch.zeros(neural_output.size(0), 1, device=neural_input.device)
        
        # Integrate
        combined = torch.cat([neural_output, symbolic_output], dim=-1)
        integrated_output = self.integrator(combined)
        
        return integrated_output
