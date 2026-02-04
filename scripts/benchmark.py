#!/usr/bin/env python3
"""
Benchmark Runner for vAGI Reasoning Evaluation.

Evaluates the model on:
1. GSM8K (Grade School Math) - Mathematical reasoning
2. HumanEval (Coding) - Code generation and execution
3. Internal benchmarks - Speed, memory, throughput

Features:
- Chain-of-Thought reasoning with <think> tags
- Sandboxed code execution with timeouts
- Detailed metrics: accuracy, thinking time, pass@k

Usage:
    # Run all reasoning benchmarks
    python scripts/benchmark.py --benchmark all --num-samples 100

    # Run specific benchmark
    python scripts/benchmark.py --benchmark gsm8k --model ./models/vagi-7b
    python scripts/benchmark.py --benchmark humaneval --model ./models/vagi-7b

    # Run internal speed benchmark
    python scripts/benchmark.py --benchmark internal

Requirements:
    pip install datasets transformers torch
"""

import argparse
import json
import os
import re
import sys
import time
import traceback
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Tuple
import multiprocessing

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

# Try importing optional dependencies
try:
    from datasets import load_dataset
    DATASETS_AVAILABLE = True
except ImportError:
    DATASETS_AVAILABLE = False

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

# Import internal modules
try:
    from core.agi import AGIModel
    from core.agi.config import load_agi_small_config, load_agi_tiny_config
    from core.nlp import BytePairTokenizer
    INTERNAL_AVAILABLE = True
except ImportError:
    INTERNAL_AVAILABLE = False


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class BenchmarkResult:
    """Result for a single benchmark problem."""
    problem_id: str
    correct: bool
    predicted_answer: Optional[str]
    expected_answer: Optional[str]
    thinking_time_ms: float
    generation_time_ms: float
    error: Optional[str] = None
    reasoning: Optional[str] = None


@dataclass
class BenchmarkReport:
    """Aggregate benchmark report."""
    benchmark_name: str
    total_problems: int
    correct: int
    accuracy: float
    avg_thinking_time_ms: float
    avg_generation_time_ms: float
    total_time_seconds: float
    results: List[BenchmarkResult] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "benchmark": self.benchmark_name,
            "total": self.total_problems,
            "correct": self.correct,
            "accuracy": f"{self.accuracy:.2%}",
            "avg_thinking_time_ms": f"{self.avg_thinking_time_ms:.2f}",
            "avg_generation_time_ms": f"{self.avg_generation_time_ms:.2f}",
            "total_time_seconds": f"{self.total_time_seconds:.2f}",
        }


# =============================================================================
# Code Sandbox
# =============================================================================

class CodeSandbox:
    """
    Sandboxed code execution environment.

    Security measures:
    - Execution timeout
    - Restricted builtins
    - Process isolation via multiprocessing
    """

    RESTRICTED_BUILTINS = {
        'abs', 'all', 'any', 'bin', 'bool', 'bytearray', 'bytes',
        'chr', 'dict', 'divmod', 'enumerate', 'filter', 'float',
        'format', 'frozenset', 'hasattr', 'hash', 'hex', 'int',
        'isinstance', 'issubclass', 'iter', 'len', 'list', 'map',
        'max', 'min', 'next', 'oct', 'ord', 'pow', 'print', 'range',
        'repr', 'reversed', 'round', 'set', 'slice', 'sorted', 'str',
        'sum', 'tuple', 'type', 'zip', 'True', 'False', 'None',
    }

    def __init__(self, timeout_seconds: float = 5.0):
        self.timeout = timeout_seconds

    def _execute_code_process(
        self,
        code: str,
        test_code: str,
        result_queue: multiprocessing.Queue,
    ):
        """Execute code in isolated process."""
        try:
            import math
            import itertools
            import collections
            import functools
            import string

            exec_globals = {
                '__builtins__': {},
                'math': math,
                'itertools': itertools,
                'collections': collections,
                'functools': functools,
                'string': string,
                'len': len, 'range': range, 'int': int, 'float': float,
                'str': str, 'list': list, 'dict': dict, 'set': set,
                'tuple': tuple, 'bool': bool, 'print': print,
                'sum': sum, 'min': min, 'max': max, 'abs': abs,
                'sorted': sorted, 'reversed': reversed, 'enumerate': enumerate,
                'zip': zip, 'map': map, 'filter': filter, 'any': any, 'all': all,
                'True': True, 'False': False, 'None': None,
                'isinstance': isinstance, 'type': type,
            }

            exec(code, exec_globals)
            exec(test_code, exec_globals)
            result_queue.put({"success": True, "error": None})

        except Exception as e:
            result_queue.put({
                "success": False,
                "error": f"{type(e).__name__}: {str(e)}"
            })

    def execute(self, code: str, test_code: str) -> Tuple[bool, Optional[str]]:
        """Execute code with tests in sandbox."""
        result_queue = multiprocessing.Queue()
        process = multiprocessing.Process(
            target=self._execute_code_process,
            args=(code, test_code, result_queue),
        )

        process.start()
        process.join(timeout=self.timeout)

        if process.is_alive():
            process.terminate()
            process.join()
            return False, "Timeout: execution exceeded time limit"

        if result_queue.empty():
            return False, "Unknown error: no result returned"

        result = result_queue.get()
        return result["success"], result["error"]


# =============================================================================
# Solution Generator
# =============================================================================

class SolutionGenerator:
    """Generates solutions using Chain-of-Thought reasoning."""

    def __init__(
        self,
        model_path: Optional[str] = None,
        device: str = "auto",
        max_new_tokens: int = 2048,
        temperature: float = 0.0,
    ):
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.model = None
        self.tokenizer = None
        self.is_vagi = False

        if device == "auto":
            if torch.cuda.is_available():
                device = "cuda"
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        self.device = torch.device(device)

        if model_path:
            self._load_model(model_path)

    def _load_model(self, model_path: str):
        """Load model and tokenizer."""
        print(f"Loading model from {model_path}...")

        # Check if it's a vAGI checkpoint
        if model_path.endswith('.pt') and INTERNAL_AVAILABLE:
            self._load_vagi_model(model_path)
        elif TRANSFORMERS_AVAILABLE:
            self._load_hf_model(model_path)
        else:
            print("Warning: No model loading backend available")

    def _load_vagi_model(self, model_path: str):
        """Load vAGI internal model."""
        ckpt = torch.load(model_path, map_location=self.device, weights_only=False)
        config = ckpt.get('config', load_agi_tiny_config())
        self.model = AGIModel(config)
        self.model.load_state_dict(ckpt['model_state_dict'])
        self.model = self.model.to(self.device).eval()

        self.tokenizer = BytePairTokenizer(vocab_size=config.vocab_size)
        if 'tokenizer_vocab' in ckpt:
            self.tokenizer.vocab = ckpt['tokenizer_vocab']
            self.tokenizer.merges = [tuple(m) for m in ckpt.get('tokenizer_merges', [])]
            self.tokenizer.inverse_vocab = {v: k for k, v in self.tokenizer.vocab.items()}

        self.is_vagi = True
        print(f"vAGI model loaded on {self.device}")

    def _load_hf_model(self, model_path: str):
        """Load HuggingFace model."""
        dtype = torch.float16 if self.device.type == "cuda" else torch.float32

        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=dtype,
            device_map=str(self.device) if self.device.type == "cuda" else None,
            trust_remote_code=True,
        )
        if self.device.type != "cuda":
            self.model = self.model.to(self.device)
        self.model.eval()
        print(f"HuggingFace model loaded on {self.device}")

    def _build_cot_prompt(self, problem: str, task_type: str) -> str:
        """Build Chain-of-Thought prompt."""
        if task_type == "math":
            return f"""Solve this math problem step by step.
Use <think> tags to show your reasoning, then give the final answer.

Problem: {problem}

<think>
Let me work through this step by step:
"""
        else:
            return f"""Complete the following Python function.
Use <think> tags to plan your solution, then write the code.

{problem}

<think>
Let me analyze what this function needs to do:
"""

    def generate_solution(
        self,
        problem: str,
        task_type: str = "math",
    ) -> Tuple[str, str, float, float]:
        """
        Generate solution with Chain-of-Thought.

        Returns:
            (full_response, answer_only, thinking_time_ms, generation_time_ms)
        """
        if self.model is None:
            return self._mock_generate(problem, task_type)

        prompt = self._build_cot_prompt(problem, task_type)
        start_time = time.perf_counter()

        if self.is_vagi:
            full_response = self._generate_vagi(prompt)
        else:
            full_response = self._generate_hf(prompt)

        generation_time = (time.perf_counter() - start_time) * 1000
        thinking_time = self._estimate_thinking_time(full_response, generation_time)
        answer = self._extract_answer(full_response, task_type)

        return full_response, answer, thinking_time, generation_time

    def _generate_vagi(self, prompt: str) -> str:
        """Generate with vAGI model."""
        ids = self.tokenizer.encode(prompt, max_length=512)
        generated = ids.copy()

        with torch.no_grad():
            for _ in range(self.max_new_tokens):
                x = torch.tensor([generated[-512:]], dtype=torch.long, device=self.device)
                out = self.model(input_ids=x, mode='inference')
                logits = out.get('text_logits')
                if logits is None:
                    break
                next_tok = logits[0, -1].argmax().item()
                if next_tok == 0:
                    break
                generated.append(next_tok)

        return self.tokenizer.decode(generated[len(ids):])

    def _generate_hf(self, prompt: str) -> str:
        """Generate with HuggingFace model."""
        inputs = self.tokenizer(prompt, return_tensors="pt", padding=True).to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature if self.temperature > 0 else None,
                do_sample=self.temperature > 0,
                pad_token_id=self.tokenizer.pad_token_id,
            )

        return self.tokenizer.decode(
            outputs[0][inputs['input_ids'].shape[1]:],
            skip_special_tokens=True,
        )

    def _mock_generate(self, problem: str, task_type: str) -> Tuple[str, str, float, float]:
        """Mock generation for testing."""
        time.sleep(0.05)
        if task_type == "math":
            numbers = re.findall(r'\d+', problem)
            answer = str(int(numbers[0]) + int(numbers[1])) if len(numbers) >= 2 else "42"
            response = f"</think>\nThe answer is {answer}.\n\n#### {answer}"
        else:
            response = "</think>\n```python\ndef solution(x):\n    return x\n```"
            answer = "def solution(x):\n    return x"
        return response, answer, 25.0, 50.0

    def _estimate_thinking_time(self, response: str, total_time: float) -> float:
        """Estimate thinking time from <think> tags."""
        think_match = re.search(r'<think>(.*?)</think>', response, re.DOTALL)
        if think_match:
            return total_time * len(think_match.group(1)) / max(len(response), 1)
        return total_time * 0.5

    def _extract_answer(self, response: str, task_type: str) -> str:
        """Extract final answer from response."""
        if task_type == "math":
            match = re.search(r'####\s*(-?\d+(?:\.\d+)?)', response)
            if match:
                return match.group(1)
            numbers = re.findall(r'-?\d+(?:\.\d+)?', response)
            return numbers[-1] if numbers else ""
        else:
            code_match = re.search(r'```python\n(.*?)```', response, re.DOTALL)
            if code_match:
                return code_match.group(1).strip()
            after_think = response.split('</think>')[-1] if '</think>' in response else response
            return after_think.strip()


# =============================================================================
# GSM8K Benchmark
# =============================================================================

class GSM8KBenchmark:
    """GSM8K Math Benchmark Runner."""

    def __init__(self, generator: SolutionGenerator, num_samples: Optional[int] = None):
        self.generator = generator
        self.num_samples = num_samples

    def _extract_gsm8k_answer(self, answer_text: str) -> str:
        match = re.search(r'####\s*(-?\d+(?:,\d+)*(?:\.\d+)?)', answer_text)
        return match.group(1).replace(',', '') if match else answer_text.strip()

    def _compare_answers(self, predicted: str, expected: str) -> bool:
        try:
            return abs(float(predicted.replace(',', '')) - float(expected.replace(',', ''))) < 1e-6
        except (ValueError, AttributeError):
            return predicted.strip() == expected.strip()

    def run(self) -> BenchmarkReport:
        if not DATASETS_AVAILABLE:
            raise ImportError("datasets library required: pip install datasets")

        print("Loading GSM8K dataset...")
        dataset = load_dataset("gsm8k", "main", split="test")
        if self.num_samples:
            dataset = dataset.select(range(min(self.num_samples, len(dataset))))

        results = []
        start_time = time.perf_counter()

        for i, example in enumerate(dataset):
            print(f"\r[GSM8K] Processing {i+1}/{len(dataset)}", end="", flush=True)

            try:
                full_response, predicted, think_time, gen_time = self.generator.generate_solution(
                    example['question'], task_type="math"
                )
                expected = self._extract_gsm8k_answer(example['answer'])
                correct = self._compare_answers(predicted, expected)

                results.append(BenchmarkResult(
                    problem_id=f"gsm8k_{i}",
                    correct=correct,
                    predicted_answer=predicted,
                    expected_answer=expected,
                    thinking_time_ms=think_time,
                    generation_time_ms=gen_time,
                    reasoning=full_response,
                ))
            except Exception as e:
                results.append(BenchmarkResult(
                    problem_id=f"gsm8k_{i}",
                    correct=False,
                    predicted_answer=None,
                    expected_answer=None,
                    thinking_time_ms=0,
                    generation_time_ms=0,
                    error=str(e),
                ))

        print()
        total_time = time.perf_counter() - start_time
        correct_count = sum(1 for r in results if r.correct)

        return BenchmarkReport(
            benchmark_name="GSM8K",
            total_problems=len(results),
            correct=correct_count,
            accuracy=correct_count / len(results) if results else 0,
            avg_thinking_time_ms=sum(r.thinking_time_ms for r in results) / len(results) if results else 0,
            avg_generation_time_ms=sum(r.generation_time_ms for r in results) / len(results) if results else 0,
            total_time_seconds=total_time,
            results=results,
        )


# =============================================================================
# HumanEval Benchmark
# =============================================================================

class HumanEvalBenchmark:
    """HumanEval Coding Benchmark Runner."""

    def __init__(self, generator: SolutionGenerator, sandbox: CodeSandbox, num_samples: Optional[int] = None):
        self.generator = generator
        self.sandbox = sandbox
        self.num_samples = num_samples

    def run(self) -> BenchmarkReport:
        if not DATASETS_AVAILABLE:
            raise ImportError("datasets library required: pip install datasets")

        print("Loading HumanEval dataset...")
        dataset = load_dataset("openai_humaneval", split="test")
        if self.num_samples:
            dataset = dataset.select(range(min(self.num_samples, len(dataset))))

        results = []
        start_time = time.perf_counter()

        for i, example in enumerate(dataset):
            task_id = example['task_id']
            print(f"\r[HumanEval] Processing {i+1}/{len(dataset)}: {task_id}", end="", flush=True)

            try:
                full_response, code, think_time, gen_time = self.generator.generate_solution(
                    example['prompt'], task_type="code"
                )
                full_code = example['prompt'] + code
                test_code = f"{example['test']}\ncheck({example['entry_point']})"
                passed, error = self.sandbox.execute(full_code, test_code)

                results.append(BenchmarkResult(
                    problem_id=task_id,
                    correct=passed,
                    predicted_answer=code[:200] + "..." if len(code) > 200 else code,
                    expected_answer=None,
                    thinking_time_ms=think_time,
                    generation_time_ms=gen_time,
                    error=error,
                    reasoning=full_response,
                ))
            except Exception as e:
                results.append(BenchmarkResult(
                    problem_id=task_id,
                    correct=False,
                    predicted_answer=None,
                    expected_answer=None,
                    thinking_time_ms=0,
                    generation_time_ms=0,
                    error=str(e),
                ))

        print()
        total_time = time.perf_counter() - start_time
        correct_count = sum(1 for r in results if r.correct)

        return BenchmarkReport(
            benchmark_name="HumanEval",
            total_problems=len(results),
            correct=correct_count,
            accuracy=correct_count / len(results) if results else 0,
            avg_thinking_time_ms=sum(r.thinking_time_ms for r in results) / len(results) if results else 0,
            avg_generation_time_ms=sum(r.generation_time_ms for r in results) / len(results) if results else 0,
            total_time_seconds=total_time,
            results=results,
        )


# =============================================================================
# Internal Benchmarks
# =============================================================================

def benchmark_internal(model_path: str, device: torch.device):
    """Run internal speed and memory benchmarks."""
    if not INTERNAL_AVAILABLE:
        print("Internal benchmarks require vAGI core modules")
        return {}

    print("Running internal benchmarks...")

    if os.path.exists(model_path):
        ckpt = torch.load(model_path, map_location=device, weights_only=False)
        config = ckpt.get('config', load_agi_tiny_config())
        model = AGIModel(config)
        model.load_state_dict(ckpt['model_state_dict'])
        tokenizer = BytePairTokenizer(vocab_size=config.vocab_size)
        if 'tokenizer_vocab' in ckpt:
            tokenizer.vocab = ckpt['tokenizer_vocab']
            tokenizer.merges = [tuple(m) for m in ckpt.get('tokenizer_merges', [])]
            tokenizer.inverse_vocab = {v: k for k, v in tokenizer.vocab.items()}
    else:
        config = load_agi_tiny_config()
        model = AGIModel(config)
        tokenizer = BytePairTokenizer(vocab_size=config.vocab_size)

    model = model.to(device).eval()
    results = {'device': str(device), 'pytorch_version': torch.__version__}

    # Parameter count
    total_params = sum(p.numel() for p in model.parameters())
    results['parameters'] = {'total': total_params, 'formatted': f'{total_params:,}'}
    print(f"  Parameters: {total_params:,}")

    # Forward pass latency
    print("  Testing forward pass latency...")
    with torch.no_grad():
        for batch_size in [1, 4]:
            x = torch.randint(0, 1000, (batch_size, 128), device=device)
            for _ in range(3):  # Warmup
                model(input_ids=x, mode='inference')

            times = []
            for _ in range(10):
                x = torch.randint(0, 1000, (batch_size, 128), device=device)
                start = time.time()
                model(input_ids=x, mode='inference')
                times.append(time.time() - start)

            avg = sum(times) / len(times)
            results[f'batch_{batch_size}_latency_ms'] = round(avg * 1000, 2)
            print(f"    Batch {batch_size}: {avg*1000:.2f}ms")

    # Memory (GPU only)
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats()
        x = torch.randint(0, 1000, (1, 256), device=device)
        model(input_ids=x, mode='inference')
        results['peak_memory_mb'] = round(torch.cuda.max_memory_allocated() / 1024 / 1024, 2)
        print(f"  Peak memory: {results['peak_memory_mb']} MB")

    return results


# =============================================================================
# Report Printing
# =============================================================================

def print_report(report: BenchmarkReport, verbose: bool = False):
    """Print benchmark report."""
    print("\n" + "=" * 60)
    print(f"BENCHMARK REPORT: {report.benchmark_name}")
    print("=" * 60)
    print(f"Total Problems:     {report.total_problems}")
    print(f"Correct:            {report.correct}")
    print(f"Accuracy:           {report.accuracy:.2%}")
    print(f"Avg Thinking Time:  {report.avg_thinking_time_ms:.2f} ms")
    print(f"Avg Generation Time:{report.avg_generation_time_ms:.2f} ms")
    print(f"Total Time:         {report.total_time_seconds:.2f} seconds")
    print("=" * 60)

    if verbose:
        print("\nDetailed Results:")
        for result in report.results[:20]:  # Limit output
            status = "PASS" if result.correct else "FAIL"
            print(f"  [{status}] {result.problem_id}")
            if result.error:
                print(f"         Error: {result.error}")


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Benchmark vAGI on GSM8K, HumanEval, and internal tests")
    parser.add_argument("--model", type=str, default="checkpoints/model.pt", help="Path to model")
    parser.add_argument("--benchmark", type=str, choices=["gsm8k", "humaneval", "all", "internal"],
                        default="internal", help="Which benchmark to run")
    parser.add_argument("--num-samples", type=int, default=None, help="Number of samples")
    parser.add_argument("--device", type=str, default="auto", help="Device (auto, cuda, cpu, mps)")
    parser.add_argument("--output", type=str, default="benchmark_results.json", help="Output JSON file")
    parser.add_argument("--verbose", action="store_true", help="Print detailed results")
    parser.add_argument("--timeout", type=float, default=5.0, help="Code execution timeout")

    args = parser.parse_args()

    # Device setup
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    print(f"Device: {device}")
    print(f"PyTorch: {torch.__version__}")
    print()

    reports = []
    all_results = {"config": {"model": args.model, "device": str(device)}}

    # Run reasoning benchmarks
    if args.benchmark in ["gsm8k", "humaneval", "all"]:
        generator = SolutionGenerator(model_path=args.model, device=str(device))
        sandbox = CodeSandbox(timeout_seconds=args.timeout)

        if args.benchmark in ["gsm8k", "all"]:
            gsm8k = GSM8KBenchmark(generator, num_samples=args.num_samples)
            report = gsm8k.run()
            print_report(report, verbose=args.verbose)
            reports.append(report)

        if args.benchmark in ["humaneval", "all"]:
            humaneval = HumanEvalBenchmark(generator, sandbox, num_samples=args.num_samples)
            report = humaneval.run()
            print_report(report, verbose=args.verbose)
            reports.append(report)

    # Run internal benchmarks
    if args.benchmark == "internal":
        internal_results = benchmark_internal(args.model, device)
        all_results["internal"] = internal_results

    # Compile results
    if reports:
        all_results["benchmarks"] = [r.to_dict() for r in reports]

    # Save results
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to: {args.output}")


if __name__ == "__main__":
    main()
