"""Program synthesis and compositional reasoning for AGI."""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

import torch
from torch import nn
from torch.nn import functional as F


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
    """Represents a synthesized program."""
    operations: List[Tuple[PrimitiveOp, Any]]
    score: float = 0.0

    def execute(self, input_data: Any) -> Any:
        """Execute the program on input data with expanded DSL support."""
        result = input_data

        for op, params in self.operations:
            try:
                result = self._execute_op(op, params, result)
            except Exception:
                # On error, return current result
                break

        return result

    def _execute_op(self, op: PrimitiveOp, params: Any, data: Any) -> Any:
        """Execute a single operation."""
        # Higher-order operations
        if op == PrimitiveOp.MAP:
            return [params(x) for x in data]
        elif op == PrimitiveOp.FILTER:
            return [x for x in data if params(x)]
        elif op == PrimitiveOp.REDUCE:
            if not data:
                return data
            result = data[0]
            for x in data[1:]:
                result = params(result, x)
            return result
        elif op == PrimitiveOp.FOLD:
            init, func = params
            result = init
            for x in data:
                result = func(result, x)
            return result
        elif op == PrimitiveOp.SCAN:
            init, func = params
            results = []
            acc = init
            for x in data:
                acc = func(acc, x)
                results.append(acc)
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
            result = data
            for _ in range(max_iter):
                if not cond_func(result):
                    break
                result = body_func(result)
            return result
        elif op == PrimitiveOp.RECURSE:
            # RECURSE: (base_case_check, base_val, recursive_func, max_depth) -> recursive computation
            base_check, base_val, rec_func = params[:3]
            max_depth = params[3] if len(params) > 3 else 50
            def recurse_helper(val, depth):
                if depth >= max_depth or base_check(val):
                    return base_val(val) if callable(base_val) else base_val
                return rec_func(val, lambda v: recurse_helper(v, depth + 1))
            return recurse_helper(data, 0)
        elif op == PrimitiveOp.UNFOLD:
            # UNFOLD: (seed, condition, generator) -> generate list from seed
            seed, cond_func, gen_func = params
            result = []
            current = seed if seed is not None else data
            max_iter = 1000  # Safety limit
            for _ in range(max_iter):
                if not cond_func(current):
                    break
                value, next_state = gen_func(current)
                result.append(value)
                current = next_state
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
    """Synthesize programs from input-output examples with actual parameter synthesis."""

    def __init__(
        self,
        example_dim: int,
        max_program_length: int = 10,
        num_candidates: int = 10,
    ):
        super().__init__()
        self.max_program_length = max_program_length
        self.num_candidates = num_candidates

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
        
    def synthesize_from_examples(
        self,
        examples: List[Tuple[torch.Tensor, torch.Tensor]],
        num_iterations: int = 100
    ) -> Optional[Program]:
        """Synthesize program from examples with actual parameter synthesis.

        Args:
            examples: List of (input, output) tensor pairs
            num_iterations: Number of synthesis attempts

        Returns:
            Best program found, or None
        """
        best_program = None
        best_score = -float('inf')

        # Encode examples into context for parameter synthesis
        context = self._encode_examples(examples)

        for _ in range(num_iterations):
            # Sample candidate program operations
            program_ops = self.sampler(examples, temperature=1.0)

            # Convert to Program with REAL parameters (not identity)
            program = self._ops_to_program(program_ops, context, examples)

            # Score program on actual execution
            data_examples = self._tensors_to_data(examples)
            score = self.verifier.score_program(program, data_examples)

            if score > best_score:
                best_score = score
                best_program = program
                program.score = score

                # Early stopping if perfect
                if score >= 0.99:
                    break

        return best_program

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
