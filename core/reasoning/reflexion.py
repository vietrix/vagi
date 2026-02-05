#!/usr/bin/env python3
"""
Reflexion: Self-Correcting Reasoning Loop for vAGI.

Implements the Reflexion paradigm from "Reflexion: Language Agents with
Verbal Reinforcement Learning" (Shinn et al., 2023).

Reflexion Loop:
    1. GENERATE: Produce initial output (code, answer, plan)
    2. EVALUATE: Test output against external signal (unit test, compiler, critic)
    3. REFLECT: If failed, generate textual analysis of WHY it failed
    4. RETRY: Regenerate with reflection appended to context

Key Insight: Instead of updating model weights, Reflexion uses verbal
self-reflection stored in memory. This allows learning from mistakes
without fine-tuning.

FlashAttention Integration:
    For maximum inference speed, install FlashAttention:

    ```bash
    # CUDA 11.8+
    pip install flash-attn --no-build-isolation

    # Or build from source for custom CUDA
    git clone https://github.com/Dao-AILab/flash-attention.git
    cd flash-attention
    python setup.py install
    ```

    Then enable in model config:
    ```python
    config.use_flash_attn = True
    ```

Usage:
    from core.reasoning import ReflexionAgent, ReflexionConfig

    agent = ReflexionAgent(
        model=vagi_model,
        tokenizer=tokenizer,
        config=ReflexionConfig(max_retries=3)
    )

    # For coding tasks
    result = agent.solve_with_tests(
        task="Write a function to find prime numbers",
        test_code="assert is_prime(17) == True"
    )

    # For general tasks with critic
    result = agent.solve_with_critic(
        task="Explain quantum entanglement",
        critic_prompt="Is this explanation accurate and complete?"
    )
"""

from __future__ import annotations

import logging
import re
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
from torch import nn

logger = logging.getLogger(__name__)


# ============================================================================
# Configuration
# ============================================================================

class EvaluationType(Enum):
    """Types of evaluation for reflexion loop."""
    UNIT_TEST = "unit_test"         # Run code against unit tests
    COMPILER = "compiler"            # Check if code compiles/runs
    CRITIC = "critic"               # Use another prompt/model as critic
    CUSTOM = "custom"               # User-provided evaluation function
    HUMAN = "human"                 # Human-in-the-loop evaluation


@dataclass
class ReflexionConfig:
    """Configuration for Reflexion agent."""
    # Retry limits (to prevent infinite loops)
    max_retries: int = 3

    # Generation settings
    max_tokens: int = 2048
    temperature: float = 0.7
    top_p: float = 0.95

    # Evaluation settings
    evaluation_type: str = "unit_test"
    test_timeout: float = 10.0  # seconds

    # Reflection settings
    reflection_temperature: float = 0.3  # Lower temp for focused reflection
    include_error_traceback: bool = True
    include_previous_attempts: bool = True

    # Memory settings
    max_memory_entries: int = 10  # Max reflections to keep
    memory_decay: float = 0.9    # Older reflections get lower weight

    # FlashAttention (see docstring for installation)
    use_flash_attention: bool = False


@dataclass
class EvaluationResult:
    """Result of evaluating an output."""
    success: bool
    score: float = 0.0  # 0-1 score
    error_message: Optional[str] = None
    error_type: Optional[str] = None
    traceback: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ReflexionState:
    """State of a reflexion loop iteration."""
    attempt: int
    output: str
    evaluation: EvaluationResult
    reflection: Optional[str] = None
    timestamp: float = field(default_factory=time.time)


# ============================================================================
# Memory Container
# ============================================================================

@dataclass
class ReflexionMemory:
    """Lightweight container for reflexion states."""
    entries: List[ReflexionState] = field(default_factory=list)
    max_entries: int = 10
    decay: float = 0.9

    def add(self, state: ReflexionState) -> None:
        self.entries.append(state)
        if len(self.entries) > self.max_entries:
            self.entries = self.entries[-self.max_entries:]

    def recent(self, n: int = 3) -> List[ReflexionState]:
        if n <= 0:
            return []
        return list(self.entries[-n:])

    def summary(self) -> str:
        if not self.entries:
            return "No reflections in memory."
        lines = []
        for idx, state in enumerate(self.entries, start=1):
            if state.reflection:
                lines.append(f"{idx}. {state.reflection}")
        return "\n".join(lines) if lines else "No reflections in memory."


# ============================================================================
# Prompt Templates
# ============================================================================

REFLECTION_PROMPT = """You are analyzing why a solution failed. Based on the error, provide a concise reflection.

## Task
{task}

## Previous Attempt
{output}

## Error
Type: {error_type}
Message: {error_message}
{traceback_section}

## Previous Reflections
{previous_reflections}

## Instructions
Write a brief reflection (2-4 sentences) analyzing:
1. What specific error occurred
2. Why the previous solution caused this error
3. What approach should be tried next

<reflect>
"""

RETRY_PROMPT = """You are solving a task. Learn from previous failed attempts.

## Task
{task}

## Previous Attempts and Reflections
{attempts_summary}

## Important Learnings
{reflections}

## Instructions
Based on the reflections above, generate an improved solution.
Avoid the mistakes identified in previous attempts.

<think>
"""

CRITIC_PROMPT = """Evaluate the following response for accuracy and completeness.

## Task
{task}

## Response
{output}

## Evaluation Criteria
- Is the response factually accurate?
- Is it complete and addresses all aspects of the task?
- Are there any errors or misconceptions?

Respond with:
- "PASS" if the response is acceptable
- "FAIL: [reason]" if there are issues

Evaluation:
"""


# ============================================================================
# Code Executor
# ============================================================================

class CodeExecutor:
    """
    Safe code execution for evaluation.

    Executes Python code in a subprocess with timeout and captures output.
    """

    def __init__(self, timeout: float = 10.0):
        self.timeout = timeout

    def execute(
        self,
        code: str,
        test_code: Optional[str] = None,
    ) -> EvaluationResult:
        """
        Execute code and optional tests.

        Args:
            code: Code to execute
            test_code: Optional test code to run after main code

        Returns:
            EvaluationResult with success/failure details
        """
        # Combine code and tests
        full_code = code
        if test_code:
            full_code += "\n\n# Tests\n" + test_code

        # Write to temp file
        with tempfile.NamedTemporaryFile(
            mode='w',
            suffix='.py',
            delete=False,
            encoding='utf-8'
        ) as f:
            f.write(full_code)
            temp_path = f.name

        try:
            # Execute in subprocess
            result = subprocess.run(
                [sys.executable, temp_path],
                capture_output=True,
                text=True,
                timeout=self.timeout,
            )

            if result.returncode == 0:
                return EvaluationResult(
                    success=True,
                    score=1.0,
                    metadata={"stdout": result.stdout}
                )
            else:
                # Parse error
                error_type, error_msg = self._parse_error(result.stderr)
                return EvaluationResult(
                    success=False,
                    score=0.0,
                    error_type=error_type,
                    error_message=error_msg,
                    traceback=result.stderr,
                )

        except subprocess.TimeoutExpired:
            return EvaluationResult(
                success=False,
                score=0.0,
                error_type="TimeoutError",
                error_message=f"Execution timed out after {self.timeout}s",
            )
        except Exception as e:
            return EvaluationResult(
                success=False,
                score=0.0,
                error_type=type(e).__name__,
                error_message=str(e),
            )
        finally:
            # Cleanup
            try:
                Path(temp_path).unlink()
            except OSError:
                pass

    def _parse_error(self, stderr: str) -> Tuple[str, str]:
        """Parse error type and message from traceback."""
        lines = stderr.strip().split('\n')

        # Find the last line with error info
        for line in reversed(lines):
            # Match patterns like "ValueError: invalid literal"
            match = re.match(r'^(\w+Error|\w+Exception): (.+)$', line)
            if match:
                return match.group(1), match.group(2)

            # Match assertion errors
            if line.startswith('AssertionError'):
                return 'AssertionError', line.replace('AssertionError:', '').strip() or 'Assertion failed'

        return 'ExecutionError', lines[-1] if lines else 'Unknown error'


# ============================================================================
# Reflexion Agent
# ============================================================================

class ReflexionAgent:
    """
    Reflexion-based self-correcting agent.

    Wraps a language model and implements the generate -> evaluate -> reflect
    -> retry loop for improved task completion.

    The key innovation is using verbal reflection (stored as text in memory)
    rather than gradient updates. This allows the model to learn from mistakes
    without fine-tuning.

    Attributes:
        model: The underlying language model
        tokenizer: Tokenizer for the model
        config: ReflexionConfig settings
        memory: List of past reflections for learning
    """

    def __init__(
        self,
        model: nn.Module,
        tokenizer: Any,
        config: Optional[ReflexionConfig] = None,
        device: Optional[torch.device] = None,
    ):
        """
        Initialize Reflexion agent.

        Args:
            model: Language model for generation
            tokenizer: Tokenizer for encoding/decoding
            config: Reflexion configuration
            device: Torch device for inference
        """
        self.model = model
        self.tokenizer = tokenizer
        self.config = config or ReflexionConfig()
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Move model to device
        self.model = self.model.to(self.device)
        self.model.eval()

        # Memory for reflections
        self.memory: List[ReflexionState] = []

        # Code executor for unit tests
        self.executor = CodeExecutor(timeout=self.config.test_timeout)

        # FlashAttention check
        if self.config.use_flash_attention:
            self._setup_flash_attention()

    def _setup_flash_attention(self):
        """
        Setup FlashAttention for faster inference.

        Installation (requires CUDA 11.8+):
            pip install flash-attn --no-build-isolation

        For custom CUDA versions, build from source:
            git clone https://github.com/Dao-AILab/flash-attention.git
            cd flash-attention
            MAX_JOBS=4 pip install .

        Environment variables:
            - CUDA_HOME: Path to CUDA toolkit
            - TORCH_CUDA_ARCH_LIST: Target GPU architectures (e.g., "8.0;8.6;9.0")
        """
        try:
            from flash_attn import flash_attn_func
            logger.info("FlashAttention enabled for faster inference")

            # Enable in model if supported
            if hasattr(self.model, 'enable_flash_attention'):
                self.model.enable_flash_attention()
            elif hasattr(self.model, 'config'):
                self.model.config.use_flash_attention_2 = True

        except ImportError:
            logger.warning(
                "FlashAttention not available. Install with:\n"
                "  pip install flash-attn --no-build-isolation\n"
                "Requires CUDA 11.8+ and PyTorch 2.0+"
            )
            self.config.use_flash_attention = False

    # ========================================================================
    # Core Generation Methods
    # ========================================================================

    def generate(
        self,
        prompt: str,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        stop_sequences: Optional[List[str]] = None,
    ) -> str:
        """
        Generate text from the model.

        Args:
            prompt: Input prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            stop_sequences: Sequences to stop generation

        Returns:
            Generated text
        """
        max_tokens = max_tokens or self.config.max_tokens
        temperature = temperature or self.config.temperature

        # Encode prompt
        if hasattr(self.tokenizer, 'encode'):
            input_ids = self.tokenizer.encode(prompt)
            if hasattr(input_ids, 'ids'):
                input_ids = input_ids.ids
        else:
            input_ids = self.tokenizer(prompt, return_tensors='pt')['input_ids'][0].tolist()

        input_ids = torch.tensor([input_ids], dtype=torch.long, device=self.device)

        # Generate
        with torch.no_grad():
            # Try HuggingFace generate
            if hasattr(self.model, 'generate'):
                output_ids = self.model.generate(
                    input_ids,
                    max_new_tokens=max_tokens,
                    temperature=temperature,
                    top_p=self.config.top_p,
                    do_sample=temperature > 0,
                    pad_token_id=getattr(self.tokenizer, 'pad_token_id', 0),
                    eos_token_id=getattr(self.tokenizer, 'eos_token_id', None),
                )
            else:
                # Manual generation for custom models
                output_ids = self._manual_generate(
                    input_ids, max_tokens, temperature, stop_sequences
                )

        # Decode
        if hasattr(self.tokenizer, 'decode'):
            output = self.tokenizer.decode(output_ids[0].tolist())
        else:
            output = self.tokenizer.decode(output_ids[0])

        # Remove prompt from output
        if output.startswith(prompt):
            output = output[len(prompt):]

        # Apply stop sequences
        if stop_sequences:
            for stop in stop_sequences:
                if stop in output:
                    output = output[:output.index(stop)]

        return output.strip()

    def _manual_generate(
        self,
        input_ids: torch.Tensor,
        max_tokens: int,
        temperature: float,
        stop_sequences: Optional[List[str]],
    ) -> torch.Tensor:
        """Manual token-by-token generation for custom models."""
        generated = input_ids.clone()

        for _ in range(max_tokens):
            # Forward pass
            outputs = self.model(generated)

            if hasattr(outputs, 'logits'):
                logits = outputs.logits
            elif isinstance(outputs, dict):
                logits = outputs.get('logits', outputs.get('text_logits'))
            else:
                logits = outputs

            # Get next token logits
            next_logits = logits[:, -1, :]

            # Apply temperature
            if temperature > 0:
                next_logits = next_logits / temperature
                probs = torch.softmax(next_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = next_logits.argmax(dim=-1, keepdim=True)

            # Append to sequence
            generated = torch.cat([generated, next_token], dim=-1)

            # Check for EOS
            eos_id = getattr(self.tokenizer, 'eos_token_id', None)
            if eos_id is not None and next_token.item() == eos_id:
                break

        return generated

    # ========================================================================
    # Evaluation Methods
    # ========================================================================

    def evaluate(
        self,
        output: str,
        task: str,
        test_code: Optional[str] = None,
        critic_prompt: Optional[str] = None,
        custom_evaluator: Optional[Callable[[str, str], EvaluationResult]] = None,
    ) -> EvaluationResult:
        """
        Evaluate the generated output.

        Args:
            output: Generated output to evaluate
            task: Original task description
            test_code: Unit test code (for UNIT_TEST evaluation)
            critic_prompt: Prompt for critic model (for CRITIC evaluation)
            custom_evaluator: Custom evaluation function

        Returns:
            EvaluationResult with success/failure details
        """
        eval_type = EvaluationType(self.config.evaluation_type)

        if eval_type == EvaluationType.UNIT_TEST:
            # Extract code from output
            code = self._extract_code(output)
            if not code:
                return EvaluationResult(
                    success=False,
                    error_type="NoCodeFound",
                    error_message="No code block found in output"
                )
            return self.executor.execute(code, test_code)

        elif eval_type == EvaluationType.COMPILER:
            # Just check if code runs without errors
            code = self._extract_code(output)
            if not code:
                return EvaluationResult(success=True, score=0.5)  # No code to check
            return self.executor.execute(code)

        elif eval_type == EvaluationType.CRITIC:
            return self._evaluate_with_critic(output, task, critic_prompt)

        elif eval_type == EvaluationType.CUSTOM:
            if custom_evaluator is None:
                raise ValueError("custom_evaluator required for CUSTOM evaluation")
            return custom_evaluator(output, task)

        else:
            raise ValueError(f"Unsupported evaluation type: {eval_type}")

    def _evaluate_with_critic(
        self,
        output: str,
        task: str,
        critic_prompt: Optional[str] = None,
    ) -> EvaluationResult:
        """Use the model as its own critic to evaluate output."""
        prompt = (critic_prompt or CRITIC_PROMPT).format(task=task, output=output)

        # Generate critique with low temperature for consistency
        critique = self.generate(
            prompt,
            max_tokens=200,
            temperature=0.1,
            stop_sequences=["\n\n"]
        )

        # Parse critique
        critique_lower = critique.lower()
        if 'pass' in critique_lower and 'fail' not in critique_lower:
            return EvaluationResult(success=True, score=1.0, metadata={"critique": critique})
        else:
            # Extract reason
            reason = critique
            if 'fail:' in critique_lower:
                reason = critique.split(':', 1)[1].strip() if ':' in critique else critique

            return EvaluationResult(
                success=False,
                score=0.3,
                error_type="CriticRejection",
                error_message=reason,
                metadata={"critique": critique}
            )

    def _extract_code(self, text: str) -> Optional[str]:
        """Extract code blocks from text."""
        # Try markdown code blocks
        patterns = [
            r'```python\n(.*?)```',
            r'```py\n(.*?)```',
            r'```\n(.*?)```',
            r'<code_start>\n?(.*?)<code_end>',
        ]

        for pattern in patterns:
            match = re.search(pattern, text, re.DOTALL)
            if match:
                return match.group(1).strip()

        # Try to find indented code after common markers
        markers = ['def ', 'class ', 'import ', 'from ']
        for marker in markers:
            if marker in text:
                # Extract from marker to end or next non-code section
                idx = text.index(marker)
                code_section = text[idx:]
                # Stop at common non-code markers
                for stop in ['\n\n#', '\n\nThe ', '\n\nThis ', '\n\nOutput:']:
                    if stop in code_section:
                        code_section = code_section[:code_section.index(stop)]
                return code_section.strip()

        return None

    # ========================================================================
    # Reflection Methods
    # ========================================================================

    def reflect(
        self,
        task: str,
        output: str,
        evaluation: EvaluationResult,
    ) -> str:
        """
        Generate reflection on why the output failed.

        This is the key component of Reflexion - verbal self-reflection
        that can be stored and used for future attempts.

        Args:
            task: Original task
            output: Failed output
            evaluation: Evaluation result with error details

        Returns:
            Textual reflection analyzing the failure
        """
        # Format previous reflections
        previous = ""
        if self.config.include_previous_attempts and self.memory:
            recent = self.memory[-3:]  # Last 3 reflections
            for i, state in enumerate(recent):
                if state.reflection:
                    previous += f"\nAttempt {i+1}: {state.reflection}"

        # Format traceback
        traceback_section = ""
        if self.config.include_error_traceback and evaluation.traceback:
            # Limit traceback length
            tb = evaluation.traceback
            if len(tb) > 500:
                tb = tb[-500:]  # Keep last 500 chars
            traceback_section = f"\nTraceback:\n{tb}"

        # Build reflection prompt
        prompt = REFLECTION_PROMPT.format(
            task=task,
            output=output[:1000],  # Limit output length
            error_type=evaluation.error_type or "Unknown",
            error_message=evaluation.error_message or "No message",
            traceback_section=traceback_section,
            previous_reflections=previous or "None",
        )

        # Generate reflection with lower temperature for focused analysis
        reflection = self.generate(
            prompt,
            max_tokens=300,
            temperature=self.config.reflection_temperature,
            stop_sequences=["</reflect>", "\n\n##"]
        )

        # Clean up reflection
        reflection = reflection.replace("<reflect>", "").replace("</reflect>", "").strip()

        return reflection

    # ========================================================================
    # Retry Methods
    # ========================================================================

    def retry(
        self,
        task: str,
        reflections: List[str],
        attempts: List[ReflexionState],
    ) -> str:
        """
        Generate improved output using reflections.

        Args:
            task: Original task
            reflections: List of reflection texts
            attempts: Previous attempt states

        Returns:
            New generated output
        """
        # Format attempts summary
        attempts_summary = ""
        for i, state in enumerate(attempts[-3:]):  # Last 3 attempts
            attempts_summary += f"\n### Attempt {i+1}\n"
            attempts_summary += f"Output (truncated): {state.output[:300]}...\n"
            attempts_summary += f"Result: {'Success' if state.evaluation.success else 'Failed'}\n"
            if not state.evaluation.success:
                attempts_summary += f"Error: {state.evaluation.error_message}\n"

        # Format reflections
        reflections_text = "\n".join(f"- {r}" for r in reflections if r)

        # Build retry prompt
        prompt = RETRY_PROMPT.format(
            task=task,
            attempts_summary=attempts_summary,
            reflections=reflections_text or "No specific learnings yet.",
        )

        # Generate with slightly higher temperature for exploration
        output = self.generate(
            prompt,
            temperature=self.config.temperature * 1.1,
        )

        return output

    # ========================================================================
    # Main Solving Methods
    # ========================================================================

    def solve(
        self,
        task: str,
        test_code: Optional[str] = None,
        critic_prompt: Optional[str] = None,
        custom_evaluator: Optional[Callable] = None,
    ) -> Tuple[str, List[ReflexionState]]:
        """
        Solve a task using the reflexion loop.

        Args:
            task: Task description
            test_code: Unit test code for evaluation
            critic_prompt: Prompt for critic evaluation
            custom_evaluator: Custom evaluation function

        Returns:
            Tuple of (final_output, history_of_attempts)
        """
        attempts: List[ReflexionState] = []
        reflections: List[str] = []

        for attempt_num in range(self.config.max_retries + 1):
            logger.info(f"Attempt {attempt_num + 1}/{self.config.max_retries + 1}")

            # Generate output
            if attempt_num == 0:
                # Initial generation
                output = self.generate(f"Task: {task}\n\n<think>")
            else:
                # Retry with reflections
                output = self.retry(task, reflections, attempts)

            # Evaluate
            evaluation = self.evaluate(
                output, task, test_code, critic_prompt, custom_evaluator
            )

            # Record state
            state = ReflexionState(
                attempt=attempt_num,
                output=output,
                evaluation=evaluation,
            )

            if evaluation.success:
                logger.info(f"Success on attempt {attempt_num + 1}!")
                attempts.append(state)
                break

            # Reflect on failure
            if attempt_num < self.config.max_retries:
                reflection = self.reflect(task, output, evaluation)
                state.reflection = reflection
                reflections.append(reflection)
                logger.info(f"Reflection: {reflection[:100]}...")

            attempts.append(state)

            # Add to memory
            self._update_memory(state)

        return attempts[-1].output, attempts

    def solve_with_tests(
        self,
        task: str,
        test_code: str,
    ) -> Tuple[str, List[ReflexionState]]:
        """
        Solve a coding task with unit tests.

        Args:
            task: Coding task description
            test_code: Unit test code

        Returns:
            Tuple of (solution_code, attempt_history)
        """
        self.config.evaluation_type = "unit_test"
        return self.solve(task, test_code=test_code)

    def solve_with_critic(
        self,
        task: str,
        critic_prompt: Optional[str] = None,
    ) -> Tuple[str, List[ReflexionState]]:
        """
        Solve a task with critic-based evaluation.

        Args:
            task: Task description
            critic_prompt: Optional custom critic prompt

        Returns:
            Tuple of (solution, attempt_history)
        """
        self.config.evaluation_type = "critic"
        return self.solve(task, critic_prompt=critic_prompt)

    # ========================================================================
    # Memory Management
    # ========================================================================

    def _update_memory(self, state: ReflexionState) -> None:
        """Update memory with new state, applying decay to old entries."""
        self.memory.append(state)

        # Limit memory size
        if len(self.memory) > self.config.max_memory_entries:
            self.memory = self.memory[-self.config.max_memory_entries:]

    def clear_memory(self) -> None:
        """Clear reflection memory."""
        self.memory.clear()

    def get_memory_summary(self) -> str:
        """Get summary of learned reflections."""
        if not self.memory:
            return "No reflections in memory."

        summary = "Learned reflections:\n"
        for i, state in enumerate(self.memory):
            if state.reflection:
                summary += f"{i+1}. {state.reflection}\n"

        return summary


# ============================================================================
# Example Usage
# ============================================================================

def demo_reflexion():
    """Demonstrate Reflexion agent."""

    # Create dummy model for testing
    class DummyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.embed = nn.Embedding(1000, 64)
            self.linear = nn.Linear(64, 1000)

        def forward(self, x):
            return {"logits": self.linear(self.embed(x))}

    class DummyTokenizer:
        def __init__(self):
            self.eos_token_id = 0
            self.pad_token_id = 0

        def encode(self, text):
            return [ord(c) % 1000 for c in text[:100]]

        def decode(self, ids):
            return "".join(chr(i % 128 + 32) for i in ids if i > 0)

    print("=== Reflexion Agent Demo ===\n")

    # Create agent
    model = DummyModel()
    tokenizer = DummyTokenizer()
    config = ReflexionConfig(max_retries=2)

    agent = ReflexionAgent(model, tokenizer, config)

    # Test code execution
    print("Testing code executor...")
    executor = CodeExecutor()
    result = executor.execute("x = 1 + 1\nassert x == 2")
    print(f"  Simple test: {'PASS' if result.success else 'FAIL'}")

    result = executor.execute("x = 1 + 1\nassert x == 3")
    print(f"  Failing test: {'PASS' if result.success else 'FAIL'} - {result.error_message}")

    print("\nReflexion agent initialized successfully!")


if __name__ == "__main__":
    demo_reflexion()
