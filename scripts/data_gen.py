#!/usr/bin/env python3
"""
Reasoning Trace Generator for vAGI Training Data.

Generates synthetic training data by prompting a teacher LLM to solve problems
step-by-step using <think> tags, then verifies and saves valid reasoning traces.

Usage:
    # Using OpenAI API
    python scripts/data_gen.py --provider openai --model gpt-4o \
        --input data/questions.jsonl --output data/reasoning_traces.jsonl

    # Using Anthropic API
    python scripts/data_gen.py --provider anthropic --model claude-sonnet-4-20250514 \
        --input data/questions.jsonl --output data/reasoning_traces.jsonl

    # With custom settings
    python scripts/data_gen.py --provider openai --model gpt-4o \
        --input data/questions.jsonl --output data/traces.jsonl \
        --max-retries 5 --rate-limit 10 --verify-code

Environment Variables:
    OPENAI_API_KEY: API key for OpenAI
    ANTHROPIC_API_KEY: API key for Anthropic

Input Format (JSONL):
    {"id": "q1", "type": "math", "question": "What is 2+2?", "answer": "4"}
    {"id": "q2", "type": "coding", "question": "Write a function...", "expected_output": "..."}
    {"id": "q3", "type": "riddle", "question": "I have cities but no houses..."}

Output Format (JSONL):
    {"id": "q1", "question": "...", "reasoning": "<think>...</think>", "answer": "...", "verified": true}
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import re
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)


# ============================================================================
# Configuration
# ============================================================================

@dataclass
class GeneratorConfig:
    """Configuration for the reasoning trace generator."""
    provider: str = "openai"  # "openai" or "anthropic"
    model: str = "gpt-4o"
    max_retries: int = 3
    retry_delay: float = 1.0
    rate_limit_rpm: int = 20  # requests per minute
    timeout: float = 120.0
    verify_code: bool = True
    max_tokens: int = 4096
    temperature: float = 0.7


# ============================================================================
# Rate Limiter
# ============================================================================

class RateLimiter:
    """Token bucket rate limiter for API calls."""

    def __init__(self, requests_per_minute: int):
        self.rpm = requests_per_minute
        self.interval = 60.0 / requests_per_minute
        self.last_request = 0.0
        self._lock = asyncio.Lock()

    async def acquire(self):
        """Wait until a request can be made."""
        async with self._lock:
            now = time.time()
            wait_time = self.last_request + self.interval - now
            if wait_time > 0:
                logger.debug(f"Rate limiting: waiting {wait_time:.2f}s")
                await asyncio.sleep(wait_time)
            self.last_request = time.time()


# ============================================================================
# LLM Client Abstraction
# ============================================================================

class LLMClient:
    """Abstract base for LLM API clients."""

    async def generate(self, prompt: str, config: GeneratorConfig) -> str:
        raise NotImplementedError


class OpenAIClient(LLMClient):
    """OpenAI API client with retry logic."""

    def __init__(self):
        try:
            from openai import AsyncOpenAI, RateLimitError, APIError
            self.client = AsyncOpenAI()
            self.RateLimitError = RateLimitError
            self.APIError = APIError
        except ImportError:
            raise ImportError("Please install openai: pip install openai")

    async def generate(self, prompt: str, config: GeneratorConfig) -> str:
        """Generate response with retry logic for rate limits."""
        last_error = None

        for attempt in range(config.max_retries):
            try:
                response = await asyncio.wait_for(
                    self.client.chat.completions.create(
                        model=config.model,
                        messages=[
                            {"role": "system", "content": SYSTEM_PROMPT},
                            {"role": "user", "content": prompt}
                        ],
                        max_tokens=config.max_tokens,
                        temperature=config.temperature,
                    ),
                    timeout=config.timeout
                )
                return response.choices[0].message.content

            except self.RateLimitError as e:
                last_error = e
                wait_time = config.retry_delay * (2 ** attempt)
                logger.warning(f"Rate limit hit, waiting {wait_time}s (attempt {attempt + 1}/{config.max_retries})")
                await asyncio.sleep(wait_time)

            except self.APIError as e:
                last_error = e
                if attempt < config.max_retries - 1:
                    wait_time = config.retry_delay * (2 ** attempt)
                    logger.warning(f"API error: {e}, retrying in {wait_time}s")
                    await asyncio.sleep(wait_time)

            except asyncio.TimeoutError:
                last_error = TimeoutError(f"Request timed out after {config.timeout}s")
                logger.warning(f"Timeout on attempt {attempt + 1}/{config.max_retries}")

        raise RuntimeError(f"Failed after {config.max_retries} attempts: {last_error}")


class AnthropicClient(LLMClient):
    """Anthropic API client with retry logic."""

    def __init__(self):
        try:
            import anthropic
            self.client = anthropic.AsyncAnthropic()
            self.RateLimitError = anthropic.RateLimitError
            self.APIError = anthropic.APIError
        except ImportError:
            raise ImportError("Please install anthropic: pip install anthropic")

    async def generate(self, prompt: str, config: GeneratorConfig) -> str:
        """Generate response with retry logic for rate limits."""
        last_error = None

        for attempt in range(config.max_retries):
            try:
                response = await asyncio.wait_for(
                    self.client.messages.create(
                        model=config.model,
                        max_tokens=config.max_tokens,
                        system=SYSTEM_PROMPT,
                        messages=[{"role": "user", "content": prompt}]
                    ),
                    timeout=config.timeout
                )
                return response.content[0].text

            except self.RateLimitError as e:
                last_error = e
                wait_time = config.retry_delay * (2 ** attempt)
                logger.warning(f"Rate limit hit, waiting {wait_time}s (attempt {attempt + 1}/{config.max_retries})")
                await asyncio.sleep(wait_time)

            except self.APIError as e:
                last_error = e
                if attempt < config.max_retries - 1:
                    wait_time = config.retry_delay * (2 ** attempt)
                    logger.warning(f"API error: {e}, retrying in {wait_time}s")
                    await asyncio.sleep(wait_time)

            except asyncio.TimeoutError:
                last_error = TimeoutError(f"Request timed out after {config.timeout}s")
                logger.warning(f"Timeout on attempt {attempt + 1}/{config.max_retries}")

        raise RuntimeError(f"Failed after {config.max_retries} attempts: {last_error}")


def get_client(provider: str) -> LLMClient:
    """Factory function to get appropriate LLM client."""
    if provider == "openai":
        return OpenAIClient()
    elif provider == "anthropic":
        return AnthropicClient()
    else:
        raise ValueError(f"Unknown provider: {provider}. Use 'openai' or 'anthropic'")


# ============================================================================
# Prompts
# ============================================================================

SYSTEM_PROMPT = """You are an expert reasoning assistant fluent in both English and Vietnamese.
When solving problems, you MUST:

1. Think step-by-step inside <think> tags
2. Number each step clearly: [Step 1], [Step 2], etc. (or [Bước 1], [Bước 2] for Vietnamese)
3. Show all intermediate calculations and reasoning
4. After </think>, provide the final answer

Format your response EXACTLY like this:

<think>
[Step 1] First, I analyze the problem...
[Step 2] Next, I consider...
[Step 3] Therefore...
</think>

Final Answer: [your answer here]

For Vietnamese questions, respond in Vietnamese:

<think>
[Bước 1] Đầu tiên, tôi phân tích đề bài...
[Bước 2] Tiếp theo, tôi xem xét...
[Bước 3] Vì vậy...
</think>

Đáp án: [câu trả lời]

For coding problems, include the code inside the thinking process and provide the final working code after </think>.
Use Vietnamese variable names for Vietnamese coding questions (e.g., tinh_tong, dem_so, etc.)."""

QUESTION_PROMPT_TEMPLATE = """Solve this {question_type} problem step-by-step:
Giải bài toán {question_type} sau theo từng bước:

{question}

Remember to / Hãy nhớ:
1. Use <think> tags with numbered steps / Sử dụng thẻ <think> với các bước được đánh số
2. Show ALL your reasoning / Trình bày TẤT CẢ lập luận
3. Provide the final answer after </think> / Đưa ra đáp án sau </think>
4. If the question is in Vietnamese, respond in Vietnamese / Nếu câu hỏi bằng tiếng Việt, trả lời bằng tiếng Việt"""


# ============================================================================
# Code Verification
# ============================================================================

class CodeVerifier:
    """Verifies code outputs by executing them in a sandbox."""

    @staticmethod
    def extract_python_code(text: str) -> Optional[str]:
        """Extract Python code from markdown code blocks or plain code."""
        # Try markdown code blocks first
        patterns = [
            r"```python\n(.*?)```",
            r"```py\n(.*?)```",
            r"```\n(.*?)```",
        ]
        for pattern in patterns:
            match = re.search(pattern, text, re.DOTALL)
            if match:
                return match.group(1).strip()

        # Try to find code after "Final Answer:" or similar
        code_markers = ["Final Answer:", "Solution:", "Code:"]
        for marker in code_markers:
            if marker in text:
                code_section = text.split(marker)[-1].strip()
                # Remove any trailing explanation
                lines = []
                for line in code_section.split('\n'):
                    if line.strip() and not line.strip().startswith('#'):
                        lines.append(line)
                    elif line.strip().startswith('#'):
                        lines.append(line)
                    elif not line.strip() and lines:
                        lines.append(line)
                if lines:
                    return '\n'.join(lines)

        return None

    @staticmethod
    async def verify_code(code: str, expected_output: Optional[str] = None, timeout: float = 10.0) -> dict:
        """
        Execute Python code and verify the output.

        Args:
            code: Python code to execute
            expected_output: Expected output (if any)
            timeout: Execution timeout in seconds

        Returns:
            Dict with 'success', 'output', and 'error' keys
        """
        result = {
            "success": False,
            "output": None,
            "error": None,
            "matched_expected": None
        }

        # Create temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False, encoding='utf-8') as f:
            f.write(code)
            temp_path = f.name

        try:
            # Run code in subprocess with timeout
            process = await asyncio.create_subprocess_exec(
                sys.executable, temp_path,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )

            try:
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(),
                    timeout=timeout
                )
            except asyncio.TimeoutError:
                process.kill()
                await process.communicate()
                result["error"] = f"Execution timed out after {timeout}s"
                return result

            if process.returncode == 0:
                result["success"] = True
                result["output"] = stdout.decode('utf-8', errors='replace').strip()

                # Check expected output if provided
                if expected_output is not None:
                    result["matched_expected"] = (
                        result["output"].strip() == expected_output.strip()
                    )
            else:
                result["error"] = stderr.decode('utf-8', errors='replace').strip()

        except Exception as e:
            result["error"] = str(e)

        finally:
            # Clean up temp file
            try:
                os.unlink(temp_path)
            except OSError:
                pass

        return result


# ============================================================================
# Reasoning Trace Parser
# ============================================================================

class ReasoningParser:
    """Parse and validate reasoning traces."""

    @staticmethod
    def extract_thinking(response: str) -> Optional[str]:
        """Extract content within <think> tags."""
        match = re.search(r"<think>(.*?)</think>", response, re.DOTALL)
        if match:
            return match.group(1).strip()
        return None

    @staticmethod
    def extract_final_answer(response: str) -> Optional[str]:
        """Extract the final answer after </think> (supports English and Vietnamese)."""
        if "</think>" in response:
            after_think = response.split("</think>")[-1].strip()
            # Try to find answer prefixes (English and Vietnamese)
            answer_prefixes = [
                "Final Answer:",
                "Đáp án:",
                "Kết quả:",
                "Trả lời:",
                "Answer:",
            ]
            for prefix in answer_prefixes:
                if prefix in after_think:
                    return after_think.split(prefix)[-1].strip()
            return after_think if after_think else None
        return None

    @staticmethod
    def count_steps(thinking: str) -> int:
        """Count numbered steps in thinking (supports English and Vietnamese)."""
        # Match both [Step N] and [Bước N] formats
        english_steps = re.findall(r"\[Step \d+\]", thinking, re.IGNORECASE)
        vietnamese_steps = re.findall(r"\[Bước \d+\]", thinking, re.IGNORECASE)
        return len(english_steps) + len(vietnamese_steps)

    @staticmethod
    def validate_response(response: str) -> dict:
        """
        Validate that response has proper format.

        Returns dict with:
            - valid: bool
            - thinking: str or None
            - answer: str or None
            - step_count: int
            - issues: list of validation issues
        """
        result = {
            "valid": False,
            "thinking": None,
            "answer": None,
            "step_count": 0,
            "issues": []
        }

        # Check for thinking tags
        thinking = ReasoningParser.extract_thinking(response)
        if not thinking:
            result["issues"].append("Missing <think>...</think> tags")
            return result

        result["thinking"] = thinking

        # Check for steps
        step_count = ReasoningParser.count_steps(thinking)
        result["step_count"] = step_count
        if step_count == 0:
            result["issues"].append("No numbered steps found (expected [Step N] format)")

        # Check for final answer
        answer = ReasoningParser.extract_final_answer(response)
        if not answer:
            result["issues"].append("No final answer found after </think>")
        else:
            result["answer"] = answer

        # Validation passes if we have thinking with steps and an answer
        result["valid"] = (thinking is not None and
                          step_count > 0 and
                          answer is not None)

        return result


# ============================================================================
# Main Generator
# ============================================================================

@dataclass
class GenerationResult:
    """Result of generating a reasoning trace."""
    question_id: str
    question: str
    question_type: str
    success: bool
    reasoning: Optional[str] = None
    answer: Optional[str] = None
    full_response: Optional[str] = None
    verified: Optional[bool] = None
    verification_output: Optional[str] = None
    error: Optional[str] = None
    step_count: int = 0


class ReasoningTraceGenerator:
    """Generate and verify reasoning traces for training data."""

    def __init__(self, config: GeneratorConfig):
        self.config = config
        self.client = get_client(config.provider)
        self.rate_limiter = RateLimiter(config.rate_limit_rpm)
        self.parser = ReasoningParser()
        self.verifier = CodeVerifier()

    async def generate_trace(self, question_data: dict) -> GenerationResult:
        """Generate a reasoning trace for a single question."""
        question_id = question_data.get("id", "unknown")
        question = question_data.get("question", "")
        question_type = question_data.get("type", "general")

        result = GenerationResult(
            question_id=question_id,
            question=question,
            question_type=question_type,
            success=False
        )

        try:
            # Rate limiting
            await self.rate_limiter.acquire()

            # Generate prompt
            prompt = QUESTION_PROMPT_TEMPLATE.format(
                question_type=question_type,
                question=question
            )

            # Call LLM
            logger.info(f"Generating trace for question: {question_id}")
            response = await self.client.generate(prompt, self.config)
            result.full_response = response

            # Parse response
            validation = self.parser.validate_response(response)

            if not validation["valid"]:
                result.error = f"Invalid response format: {', '.join(validation['issues'])}"
                logger.warning(f"Question {question_id}: {result.error}")
                return result

            result.reasoning = f"<think>{validation['thinking']}</think>"
            result.answer = validation["answer"]
            result.step_count = validation["step_count"]

            # Verify code if applicable
            if question_type == "coding" and self.config.verify_code:
                code = self.verifier.extract_python_code(response)
                if code:
                    expected = question_data.get("expected_output")
                    verification = await self.verifier.verify_code(code, expected)
                    result.verified = verification["success"]
                    result.verification_output = verification.get("output")
                    if not verification["success"]:
                        logger.warning(
                            f"Question {question_id}: Code verification failed - {verification.get('error')}"
                        )
                else:
                    result.verified = None  # Could not extract code

            result.success = True
            logger.info(f"Successfully generated trace for {question_id} ({result.step_count} steps)")

        except Exception as e:
            result.error = str(e)
            logger.error(f"Error generating trace for {question_id}: {e}")

        return result

    async def process_batch(
        self,
        questions: list[dict],
        output_path: Path,
        append: bool = False
    ) -> dict:
        """Process a batch of questions and save results."""
        stats = {
            "total": len(questions),
            "success": 0,
            "failed": 0,
            "verified": 0,
            "verification_failed": 0
        }

        mode = 'a' if append else 'w'
        with open(output_path, mode, encoding='utf-8') as f:
            for i, question in enumerate(questions, 1):
                logger.info(f"Processing {i}/{len(questions)}: {question.get('id', 'unknown')}")

                result = await self.generate_trace(question)

                if result.success:
                    stats["success"] += 1

                    # Build output record
                    record = {
                        "id": result.question_id,
                        "type": result.question_type,
                        "question": result.question,
                        "reasoning": result.reasoning,
                        "answer": result.answer,
                        "step_count": result.step_count,
                        "verified": result.verified
                    }

                    if result.verification_output:
                        record["verification_output"] = result.verification_output

                    f.write(json.dumps(record, ensure_ascii=False) + '\n')
                    f.flush()

                    if result.verified is True:
                        stats["verified"] += 1
                    elif result.verified is False:
                        stats["verification_failed"] += 1
                else:
                    stats["failed"] += 1
                    logger.warning(f"Failed: {result.question_id} - {result.error}")

        return stats


# ============================================================================
# Sample Questions Generator
# ============================================================================

def generate_sample_questions() -> list[dict]:
    """Generate sample questions for testing (English and Vietnamese with full diacritics)."""
    return [
        # English Math
        {
            "id": "math_001",
            "type": "math",
            "question": "If a train travels 120 km in 2 hours, and then 180 km in 3 hours, what is its average speed for the entire journey?",
            "answer": "60 km/h"
        },
        {
            "id": "math_002",
            "type": "math",
            "question": "A store offers a 20% discount on a $150 item. If there's an additional 10% off for members, what is the final price for a member?",
            "answer": "$108"
        },
        # Vietnamese Math (Toán tiếng Việt)
        {
            "id": "math_003_vi",
            "type": "math",
            "question": "Một cửa hàng bán 3 loại trái cây: táo giá 25.000đ/kg, cam giá 30.000đ/kg, và nho giá 45.000đ/kg. Nếu mua 2kg táo, 1.5kg cam và 1kg nho, tổng tiền phải trả là bao nhiêu?",
            "answer": "140.000đ"
        },
        {
            "id": "math_004_vi",
            "type": "math",
            "question": "Một hình chữ nhật có chiều dài gấp đôi chiều rộng. Nếu chu vi là 36cm, hãy tính diện tích hình chữ nhật.",
            "answer": "72 cm²"
        },
        {
            "id": "math_005_vi",
            "type": "math",
            "question": "Bạn An có 150.000đ. An mua một quyển sách giá 45.000đ và một cây bút giá 12.000đ. Hỏi An còn lại bao nhiêu tiền?",
            "answer": "93.000đ"
        },
        # English Coding
        {
            "id": "coding_001",
            "type": "coding",
            "question": "Write a Python function `is_prime(n)` that returns True if n is a prime number and False otherwise. Test it with n=17 and print the result.",
            "expected_output": "True"
        },
        {
            "id": "coding_002",
            "type": "coding",
            "question": "Write a Python function `fibonacci(n)` that returns the nth Fibonacci number (0-indexed, so fibonacci(0)=0, fibonacci(1)=1). Print fibonacci(10).",
            "expected_output": "55"
        },
        # Vietnamese Coding (Lập trình tiếng Việt)
        {
            "id": "coding_003_vi",
            "type": "coding",
            "question": "Viết hàm Python `tinh_giai_thua(n)` để tính n! (giai thừa của n). In kết quả của tinh_giai_thua(5).",
            "expected_output": "120"
        },
        {
            "id": "coding_004_vi",
            "type": "coding",
            "question": "Viết hàm Python `dem_nguyen_am(chuoi)` đếm số nguyên âm (a, ă, â, e, ê, i, o, ô, ơ, u, ư, y) trong một chuỗi tiếng Việt (không phân biệt hoa thường). Kiểm tra với chuỗi 'Xin chào Việt Nam' và in kết quả.",
            "expected_output": "6"
        },
        # English Riddle
        {
            "id": "riddle_001",
            "type": "riddle",
            "question": "I have cities, but no houses live in them. I have mountains, but no trees grow on them. I have water, but no fish swim in it. I have roads, but no cars drive on them. What am I?",
            "answer": "A map"
        },
        # Vietnamese Riddles (Câu đố tiếng Việt)
        {
            "id": "riddle_002_vi",
            "type": "riddle",
            "question": "Cái gì đi bằng bốn chân vào buổi sáng, hai chân vào buổi trưa, và ba chân vào buổi chiều?",
            "answer": "Con người (bò khi còn bé, đi hai chân khi trưởng thành, chống gậy khi già)"
        },
        {
            "id": "riddle_003_vi",
            "type": "riddle",
            "question": "Cái gì càng khô càng ướt?",
            "answer": "Khăn tắm"
        },
        # English Logic
        {
            "id": "logic_001",
            "type": "logic",
            "question": "Three friends (Alice, Bob, Carol) each have a different pet (cat, dog, bird). Alice doesn't have a dog. The person with the bird is not Carol. Bob doesn't have a cat. Who has which pet?",
            "answer": "Alice has cat, Bob has bird, Carol has dog"
        },
        # Vietnamese Logic (Logic tiếng Việt)
        {
            "id": "logic_002_vi",
            "type": "logic",
            "question": "Ba người bạn An, Bình, Cường mỗi người thích một môn thể thao khác nhau: bóng đá, bóng rổ, cầu lông. An không thích bóng đá. Người thích cầu lông không phải Cường. Bình không thích bóng rổ. Hỏi mỗi người thích môn gì?",
            "answer": "An thích cầu lông, Bình thích bóng đá, Cường thích bóng rổ"
        },
        {
            "id": "logic_003_vi",
            "type": "logic",
            "question": "Nếu hôm nay là thứ Tư, thì 100 ngày sau là thứ mấy?",
            "answer": "Thứ Sáu"
        },
    ]


# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Generate reasoning traces using LLM APIs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Generate traces using OpenAI
    python scripts/data_gen.py --provider openai --model gpt-4o \\
        --input questions.jsonl --output traces.jsonl

    # Generate sample questions file
    python scripts/data_gen.py --generate-samples --output data/sample_questions.jsonl

    # Process with Anthropic and verify code
    python scripts/data_gen.py --provider anthropic --model claude-sonnet-4-20250514 \\
        --input questions.jsonl --output traces.jsonl --verify-code
        """
    )

    parser.add_argument("--provider", choices=["openai", "anthropic"], default="openai",
                        help="LLM provider to use (default: openai)")
    parser.add_argument("--model", type=str, default="gpt-4o",
                        help="Model name (default: gpt-4o)")
    parser.add_argument("--input", type=str, default="data/questions.jsonl",
                        help="Input JSONL file with questions")
    parser.add_argument("--output", type=str, default="data/reasoning_traces.jsonl",
                        help="Output JSONL file for traces")
    parser.add_argument("--max-retries", type=int, default=3,
                        help="Max retries per request (default: 3)")
    parser.add_argument("--rate-limit", type=int, default=20,
                        help="Rate limit in requests per minute (default: 20)")
    parser.add_argument("--verify-code", action="store_true",
                        help="Verify code outputs by executing them")
    parser.add_argument("--temperature", type=float, default=0.7,
                        help="Sampling temperature (default: 0.7)")
    parser.add_argument("--max-tokens", type=int, default=4096,
                        help="Max tokens in response (default: 4096)")
    parser.add_argument("--append", action="store_true",
                        help="Append to output file instead of overwriting")
    parser.add_argument("--generate-samples", action="store_true",
                        help="Generate sample questions file and exit")

    args = parser.parse_args()

    # Generate sample questions if requested
    if args.generate_samples:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        samples = generate_sample_questions()
        with open(output_path, 'w', encoding='utf-8') as f:
            for sample in samples:
                f.write(json.dumps(sample, ensure_ascii=False) + '\n')

        logger.info(f"Generated {len(samples)} sample questions in {output_path}")
        return

    # Validate input file
    input_path = Path(args.input)
    if not input_path.exists():
        logger.error(f"Input file not found: {input_path}")
        logger.info("Use --generate-samples to create a sample questions file")
        sys.exit(1)

    # Load questions
    questions = []
    with open(input_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            if line.strip():
                try:
                    questions.append(json.loads(line))
                except json.JSONDecodeError as e:
                    logger.warning(f"Skipping invalid JSON on line {line_num}: {e}")

    if not questions:
        logger.error("No valid questions found in input file")
        sys.exit(1)

    logger.info(f"Loaded {len(questions)} questions from {input_path}")

    # Create config
    config = GeneratorConfig(
        provider=args.provider,
        model=args.model,
        max_retries=args.max_retries,
        rate_limit_rpm=args.rate_limit,
        verify_code=args.verify_code,
        temperature=args.temperature,
        max_tokens=args.max_tokens
    )

    # Create output directory
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Run generator
    generator = ReasoningTraceGenerator(config)

    async def run():
        stats = await generator.process_batch(questions, output_path, append=args.append)
        return stats

    stats = asyncio.run(run())

    # Print summary
    logger.info("\n" + "=" * 50)
    logger.info("Generation Complete!")
    logger.info("=" * 50)
    logger.info(f"Total questions:     {stats['total']}")
    logger.info(f"Successful:          {stats['success']}")
    logger.info(f"Failed:              {stats['failed']}")
    if config.verify_code:
        logger.info(f"Code verified:       {stats['verified']}")
        logger.info(f"Verification failed: {stats['verification_failed']}")
    logger.info(f"Output saved to:     {output_path}")


if __name__ == "__main__":
    main()
