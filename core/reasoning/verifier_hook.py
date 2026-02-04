"""
Verifier Hook for vAGI Code Execution.

Monitors model output for <verify_code> tags and executes code to provide
observations back to the model.

Flow:
1. Stream model output
2. Detect <verify_code language="python">...</verify_code>
3. Pause generation
4. Execute code in sandbox
5. Inject <observation>Result: ...</observation>
6. Resume generation

Usage:
    hook = VerifierHook()

    # With streaming generator
    for token in model.generate_stream(prompt):
        result = hook.process_token(token)
        if result.should_pause:
            # Execute code and inject observation
            observation = hook.execute_and_observe()
            # Add observation to context
            prompt += observation
        yield result.token

    # Or wrap entire generation
    output = hook.wrap_generation(model, prompt)
"""

import re
import asyncio
from typing import Optional, List, Dict, Any, Generator, AsyncGenerator, Callable
from dataclasses import dataclass, field
from enum import Enum

from .verifiers.python_executor import PythonExecutor, ExecutionResult


class VerifierState(Enum):
    """State machine for verifier hook."""
    STREAMING = "streaming"       # Normal token streaming
    IN_VERIFY_TAG = "in_verify"  # Inside <verify_code> tag
    AWAITING_RESULT = "awaiting" # Code executed, waiting to inject result


@dataclass
class TokenResult:
    """Result of processing a token."""
    token: str
    should_pause: bool = False
    should_inject: bool = False
    injection_content: Optional[str] = None
    code_detected: Optional[str] = None
    language: Optional[str] = None


@dataclass
class VerificationResult:
    """Result of a code verification."""
    code: str
    language: str
    execution_result: ExecutionResult
    observation: str
    success: bool


class VerifierHook:
    """
    Hook for intercepting and verifying code in model output.

    Detects patterns:
    - <verify_code language="python">...code...</verify_code>
    - <code_verify>...code...</code_verify>
    - ```python\n# verify\n...code...```

    Injects:
    - <observation>Result: output</observation>
    - <observation>Error: error message</observation>
    """

    # Patterns to detect
    VERIFY_TAG_PATTERN = re.compile(
        r'<verify_code\s+language=["\'](\w+)["\']>(.*?)</verify_code>',
        re.DOTALL
    )
    VERIFY_TAG_START = re.compile(r'<verify_code\s+language=["\'](\w+)["\']>')
    VERIFY_TAG_END = '</verify_code>'

    # Alternative patterns
    CODE_VERIFY_PATTERN = re.compile(r'<code_verify>(.*?)</code_verify>', re.DOTALL)
    MARKDOWN_VERIFY_PATTERN = re.compile(r'```python\n# verify\n(.*?)```', re.DOTALL)

    def __init__(
        self,
        executor: Optional[PythonExecutor] = None,
        timeout: float = 5.0,
        memory_mb: int = 128,
        max_verifications: int = 10,
        auto_inject: bool = True,
    ):
        """
        Initialize verifier hook.

        Args:
            executor: Python executor instance (created if None)
            timeout: Code execution timeout
            memory_mb: Memory limit for execution
            max_verifications: Max code blocks to verify per generation
            auto_inject: Automatically inject observations
        """
        self.executor = executor or PythonExecutor(timeout=timeout, memory_mb=memory_mb)
        self.max_verifications = max_verifications
        self.auto_inject = auto_inject

        # State
        self.state = VerifierState.STREAMING
        self.buffer = ""
        self.current_language = None
        self.verification_count = 0
        self.verifications: List[VerificationResult] = []

    def reset(self):
        """Reset hook state for new generation."""
        self.state = VerifierState.STREAMING
        self.buffer = ""
        self.current_language = None
        self.verification_count = 0
        self.verifications = []

    def process_token(self, token: str) -> TokenResult:
        """
        Process a single token from model output.

        Args:
            token: Generated token

        Returns:
            TokenResult with processing info
        """
        self.buffer += token

        # Check if we've exceeded max verifications
        if self.verification_count >= self.max_verifications:
            return TokenResult(token=token)

        # State machine
        if self.state == VerifierState.STREAMING:
            return self._process_streaming(token)
        elif self.state == VerifierState.IN_VERIFY_TAG:
            return self._process_in_tag(token)
        else:
            return TokenResult(token=token)

    def _process_streaming(self, token: str) -> TokenResult:
        """Process token in streaming state."""
        # Check for tag start
        start_match = self.VERIFY_TAG_START.search(self.buffer)
        if start_match:
            self.state = VerifierState.IN_VERIFY_TAG
            self.current_language = start_match.group(1)
            return TokenResult(token=token)

        # Check for alternative patterns in buffer
        # (Only check when we have enough content)
        if len(self.buffer) > 50:
            # Markdown verify pattern
            md_match = self.MARKDOWN_VERIFY_PATTERN.search(self.buffer)
            if md_match:
                code = md_match.group(1)
                return self._trigger_verification(code, "python", token)

        return TokenResult(token=token)

    def _process_in_tag(self, token: str) -> TokenResult:
        """Process token while inside verify tag."""
        # Check for closing tag
        if self.VERIFY_TAG_END in self.buffer:
            # Extract code
            full_match = self.VERIFY_TAG_PATTERN.search(self.buffer)
            if full_match:
                language = full_match.group(1)
                code = full_match.group(2).strip()
                return self._trigger_verification(code, language, token)

            # Fallback: extract between tags
            start_match = self.VERIFY_TAG_START.search(self.buffer)
            if start_match:
                start_idx = start_match.end()
                end_idx = self.buffer.find(self.VERIFY_TAG_END)
                if end_idx > start_idx:
                    code = self.buffer[start_idx:end_idx].strip()
                    return self._trigger_verification(code, self.current_language or "python", token)

        return TokenResult(token=token)

    def _trigger_verification(self, code: str, language: str, token: str) -> TokenResult:
        """Trigger code verification."""
        self.state = VerifierState.STREAMING
        self.verification_count += 1

        # Only support Python for now
        if language.lower() != "python":
            observation = f"<observation>Error: Language '{language}' not supported. Only Python is available.</observation>"
            return TokenResult(
                token=token,
                should_pause=True,
                should_inject=True,
                injection_content=observation,
                code_detected=code,
                language=language,
            )

        return TokenResult(
            token=token,
            should_pause=True,
            should_inject=self.auto_inject,
            code_detected=code,
            language=language,
        )

    def execute_and_observe(self, code: Optional[str] = None, language: str = "python") -> str:
        """
        Execute code and return observation string.

        Args:
            code: Code to execute (uses detected code if None)
            language: Programming language

        Returns:
            Observation string to inject
        """
        if code is None:
            # Extract from buffer
            match = self.VERIFY_TAG_PATTERN.search(self.buffer)
            if match:
                code = match.group(2).strip()
            else:
                return "<observation>Error: No code found to verify</observation>"

        # Execute code
        result = self.executor.execute(code)

        # Store verification
        observation = result.to_observation()
        self.verifications.append(VerificationResult(
            code=code,
            language=language,
            execution_result=result,
            observation=observation,
            success=result.success,
        ))

        # Clear buffer after tag
        if self.VERIFY_TAG_END in self.buffer:
            end_idx = self.buffer.find(self.VERIFY_TAG_END) + len(self.VERIFY_TAG_END)
            self.buffer = self.buffer[end_idx:]

        return observation

    def process_text(self, text: str) -> str:
        """
        Process complete text and inject all observations.

        Args:
            text: Complete model output

        Returns:
            Text with observations injected after verify tags
        """
        result = text
        offset = 0

        # Find all verify tags
        for match in self.VERIFY_TAG_PATTERN.finditer(text):
            if self.verification_count >= self.max_verifications:
                break

            language = match.group(1)
            code = match.group(2).strip()

            # Execute and get observation
            if language.lower() == "python":
                exec_result = self.executor.execute(code)
                observation = exec_result.to_observation()
            else:
                observation = f"<observation>Error: Language '{language}' not supported</observation>"

            # Store verification
            self.verifications.append(VerificationResult(
                code=code,
                language=language,
                execution_result=exec_result if language.lower() == "python" else None,
                observation=observation,
                success=exec_result.success if language.lower() == "python" else False,
            ))
            self.verification_count += 1

            # Insert observation after tag
            insert_pos = match.end() + offset
            result = result[:insert_pos] + "\n" + observation + "\n" + result[insert_pos:]
            offset += len(observation) + 2

        return result

    def wrap_generation(
        self,
        generate_fn: Callable[[str], Generator[str, None, None]],
        prompt: str,
    ) -> Generator[str, None, None]:
        """
        Wrap a generation function with verification.

        Args:
            generate_fn: Token generation function
            prompt: Initial prompt

        Yields:
            Tokens with observations injected
        """
        self.reset()
        current_prompt = prompt

        for token in generate_fn(current_prompt):
            result = self.process_token(token)

            if result.should_pause and result.code_detected:
                # Execute code
                observation = self.execute_and_observe(result.code_detected, result.language)

                if result.should_inject:
                    # Yield observation tokens
                    yield "\n"
                    yield observation
                    yield "\n"

                    # Update prompt for continued generation
                    current_prompt += self.buffer + "\n" + observation + "\n"

            yield result.token

    async def wrap_async_generation(
        self,
        generate_fn: Callable[[str], AsyncGenerator[str, None]],
        prompt: str,
    ) -> AsyncGenerator[str, None]:
        """
        Wrap an async generation function with verification.

        Args:
            generate_fn: Async token generation function
            prompt: Initial prompt

        Yields:
            Tokens with observations injected
        """
        self.reset()
        current_prompt = prompt

        async for token in generate_fn(current_prompt):
            result = self.process_token(token)

            if result.should_pause and result.code_detected:
                # Execute code (run in thread pool for blocking executor)
                loop = asyncio.get_event_loop()
                observation = await loop.run_in_executor(
                    None,
                    self.execute_and_observe,
                    result.code_detected,
                    result.language,
                )

                if result.should_inject:
                    yield "\n"
                    yield observation
                    yield "\n"

                    current_prompt += self.buffer + "\n" + observation + "\n"

            yield result.token

    def get_verification_summary(self) -> Dict[str, Any]:
        """Get summary of all verifications."""
        return {
            "total": len(self.verifications),
            "successful": sum(1 for v in self.verifications if v.success),
            "failed": sum(1 for v in self.verifications if not v.success),
            "verifications": [
                {
                    "code": v.code[:100] + "..." if len(v.code) > 100 else v.code,
                    "language": v.language,
                    "success": v.success,
                    "observation": v.observation,
                }
                for v in self.verifications
            ],
        }


class StreamingVerifier:
    """
    Higher-level wrapper for streaming verification.

    Integrates with model generation and handles the full loop:
    1. Generate until verify tag detected
    2. Execute code
    3. Inject observation
    4. Continue generation
    """

    def __init__(
        self,
        executor: Optional[PythonExecutor] = None,
        max_iterations: int = 5,
        max_code_blocks: int = 10,
    ):
        self.hook = VerifierHook(executor=executor, max_verifications=max_code_blocks)
        self.max_iterations = max_iterations

    def verify_and_continue(
        self,
        model,
        tokenizer,
        prompt: str,
        max_new_tokens: int = 2048,
    ) -> str:
        """
        Generate with verification loop.

        Args:
            model: Language model
            tokenizer: Tokenizer
            prompt: Initial prompt
            max_new_tokens: Max tokens to generate

        Returns:
            Complete output with verifications
        """
        self.hook.reset()
        current_prompt = prompt
        full_output = ""
        iterations = 0

        while iterations < self.max_iterations:
            iterations += 1

            # Generate
            inputs = tokenizer(current_prompt, return_tensors="pt")
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.7,
                pad_token_id=tokenizer.pad_token_id,
            )

            generated = tokenizer.decode(
                outputs[0][inputs['input_ids'].shape[1]:],
                skip_special_tokens=True,
            )

            # Process for verify tags
            processed = self.hook.process_text(generated)

            # Check if any new verifications happened
            if self.hook.verification_count == 0 or '</verify_code>' not in generated:
                full_output += processed
                break

            # Add to output and continue
            full_output += processed
            current_prompt += processed

        return full_output


# =============================================================================
# Convenience Functions
# =============================================================================

def create_verifier_hook(**kwargs) -> VerifierHook:
    """Create a verifier hook with default settings."""
    return VerifierHook(**kwargs)


def process_model_output(text: str, executor: Optional[PythonExecutor] = None) -> str:
    """
    Process model output and inject all observations.

    Args:
        text: Model output text
        executor: Python executor (created if None)

    Returns:
        Text with observations injected
    """
    hook = VerifierHook(executor=executor)
    return hook.process_text(text)


# =============================================================================
# Example Integration
# =============================================================================

def example_integration():
    """
    Example of how to integrate the verifier hook with a model.

    This shows the recommended pattern for using the hook.
    """
    print("Verifier Hook Integration Example")
    print("=" * 50)

    # Create hook
    hook = VerifierHook(timeout=5.0, memory_mb=128)

    # Simulate model output with verify tag
    model_output = """
Let me think through this problem step by step.

<think>
I need to calculate the sum of squares from 1 to 10.
Let me verify my logic with code:
</think>

<verify_code language="python">
result = sum(x**2 for x in range(1, 11))
print(f"Sum of squares from 1 to 10: {result}")
</verify_code>

Based on the verification, the sum of squares is 385.
"""

    print("Input:")
    print(model_output)
    print("-" * 50)

    # Process and inject observations
    processed = hook.process_text(model_output)

    print("Output with observations:")
    print(processed)
    print("-" * 50)

    print("Verification Summary:")
    print(hook.get_verification_summary())


if __name__ == "__main__":
    example_integration()
