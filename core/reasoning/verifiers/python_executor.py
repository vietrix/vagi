"""
Secure Python Code Executor for vAGI Truth Engine.

Provides sandboxed code execution for verifying model-generated logic.

Security Features:
- Execution timeout (default: 5s)
- Memory limit (default: 128MB)
- Blocked dangerous modules (os, subprocess, socket, etc.)
- No filesystem write access
- No network access
- Restricted builtins

Execution Backends:
1. Docker (recommended) - Full isolation via containers
2. RestrictedPython - AST-level restrictions
3. Subprocess with resource limits - OS-level restrictions

Usage:
    executor = PythonExecutor(timeout=5, memory_mb=128)
    result = executor.execute("print(2 + 2)")
    print(result.stdout)  # "4"

    # With error
    result = executor.execute("x = 1/0")
    print(result.error_message)  # "Error on line 1: ZeroDivisionError: division by zero"
"""

import ast
import sys
import io
import traceback
import resource
import signal
import tempfile
import textwrap
from pathlib import Path
from typing import Optional, Dict, Any, List, Set
from dataclasses import dataclass, field
from enum import Enum
import multiprocessing
from contextlib import contextmanager
import re

# Try importing Docker SDK
try:
    import docker
    DOCKER_AVAILABLE = True
except ImportError:
    DOCKER_AVAILABLE = False

# Try importing RestrictedPython
try:
    from RestrictedPython import compile_restricted, safe_builtins
    from RestrictedPython.Guards import safe_builtins as rp_safe_builtins
    from RestrictedPython.Eval import default_guarded_getiter, default_guarded_getitem
    RESTRICTED_PYTHON_AVAILABLE = True
except ImportError:
    RESTRICTED_PYTHON_AVAILABLE = False


class ExecutionBackend(Enum):
    """Available execution backends."""
    DOCKER = "docker"
    RESTRICTED = "restricted"
    SUBPROCESS = "subprocess"
    INLINE = "inline"  # Most dangerous, only for testing


@dataclass
class ExecutionResult:
    """Result of code execution."""
    success: bool
    stdout: str = ""
    stderr: str = ""
    return_value: Any = None
    execution_time_ms: float = 0.0
    error_message: Optional[str] = None  # LLM-friendly error description
    error_line: Optional[int] = None
    error_type: Optional[str] = None

    def to_observation(self) -> str:
        """Format result as an observation for the LLM."""
        if self.success:
            output = self.stdout.strip() if self.stdout else str(self.return_value)
            return f"<observation>Result: {output}</observation>"
        else:
            return f"<observation>Error: {self.error_message}</observation>"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "stdout": self.stdout,
            "stderr": self.stderr,
            "return_value": str(self.return_value) if self.return_value is not None else None,
            "execution_time_ms": self.execution_time_ms,
            "error_message": self.error_message,
            "error_line": self.error_line,
            "error_type": self.error_type,
        }


# =============================================================================
# Security: Blocked Modules and Builtins
# =============================================================================

BLOCKED_MODULES: Set[str] = {
    # System access
    "os", "sys", "subprocess", "shutil", "pathlib",
    "platform", "ctypes", "multiprocessing", "threading",

    # Network access
    "socket", "http", "urllib", "requests", "httplib",
    "ftplib", "smtplib", "telnetlib", "ssl", "asyncio",

    # File system
    "io", "tempfile", "glob", "fnmatch",

    # Code execution
    "exec", "eval", "compile", "code", "codeop",
    "importlib", "__import__", "builtins",

    # Dangerous
    "pickle", "shelve", "marshal", "dbm",
    "sqlite3", "mysql", "psycopg2",

    # Introspection (can be used to escape sandbox)
    "inspect", "dis", "gc", "weakref",
}

BLOCKED_BUILTINS: Set[str] = {
    "exec", "eval", "compile", "__import__", "open",
    "input", "breakpoint", "help", "license", "credits",
    "quit", "exit", "globals", "locals", "vars",
    "dir", "getattr", "setattr", "delattr", "hasattr",
    "memoryview", "bytearray",
}

SAFE_BUILTINS: Dict[str, Any] = {
    # Types
    "bool": bool, "int": int, "float": float, "str": str,
    "list": list, "dict": dict, "tuple": tuple, "set": set,
    "frozenset": frozenset, "bytes": bytes,

    # Functions
    "abs": abs, "all": all, "any": any, "ascii": ascii,
    "bin": bin, "callable": callable, "chr": chr,
    "divmod": divmod, "enumerate": enumerate, "filter": filter,
    "format": format, "hash": hash, "hex": hex, "id": id,
    "isinstance": isinstance, "issubclass": issubclass,
    "iter": iter, "len": len, "map": map, "max": max, "min": min,
    "next": next, "oct": oct, "ord": ord, "pow": pow,
    "print": print, "range": range, "repr": repr, "reversed": reversed,
    "round": round, "slice": slice, "sorted": sorted, "sum": sum,
    "type": type, "zip": zip,

    # Constants
    "True": True, "False": False, "None": None,

    # Exceptions (for error handling)
    "Exception": Exception, "ValueError": ValueError,
    "TypeError": TypeError, "KeyError": KeyError,
    "IndexError": IndexError, "ZeroDivisionError": ZeroDivisionError,
    "RuntimeError": RuntimeError, "StopIteration": StopIteration,
}

SAFE_MODULES: Dict[str, Any] = {}

# Lazily import safe modules
def _get_safe_modules() -> Dict[str, Any]:
    global SAFE_MODULES
    if not SAFE_MODULES:
        import math
        import random
        import statistics
        import collections
        import itertools
        import functools
        import operator
        import string
        import re as re_module
        import json
        import datetime
        import decimal
        import fractions

        SAFE_MODULES = {
            "math": math,
            "random": random,
            "statistics": statistics,
            "collections": collections,
            "itertools": itertools,
            "functools": functools,
            "operator": operator,
            "string": string,
            "re": re_module,
            "json": json,
            "datetime": datetime,
            "decimal": decimal,
            "fractions": fractions,
        }
    return SAFE_MODULES


# =============================================================================
# AST Security Checker
# =============================================================================

class SecurityChecker(ast.NodeVisitor):
    """
    AST visitor that checks for dangerous operations.

    Blocks:
    - Import of dangerous modules
    - Access to dangerous attributes (e.g., __class__, __bases__)
    - Dangerous function calls
    """

    DANGEROUS_ATTRIBUTES: Set[str] = {
        "__class__", "__bases__", "__mro__", "__subclasses__",
        "__globals__", "__code__", "__closure__", "__func__",
        "__self__", "__dict__", "__module__", "__name__",
        "__qualname__", "__annotations__", "__builtins__",
        "__import__", "__loader__", "__spec__", "__cached__",
        "__file__", "__path__", "__package__",
        "gi_frame", "gi_code", "f_globals", "f_locals", "f_builtins",
    }

    def __init__(self):
        self.errors: List[str] = []
        self.imports: Set[str] = set()

    def visit_Import(self, node: ast.Import):
        for alias in node.names:
            module_name = alias.name.split('.')[0]
            if module_name in BLOCKED_MODULES:
                self.errors.append(
                    f"Line {node.lineno}: Import of '{module_name}' is blocked for security"
                )
            self.imports.add(module_name)
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom):
        if node.module:
            module_name = node.module.split('.')[0]
            if module_name in BLOCKED_MODULES:
                self.errors.append(
                    f"Line {node.lineno}: Import from '{module_name}' is blocked for security"
                )
            self.imports.add(module_name)
        self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute):
        if node.attr in self.DANGEROUS_ATTRIBUTES:
            self.errors.append(
                f"Line {node.lineno}: Access to '{node.attr}' is blocked for security"
            )
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call):
        # Check for dangerous function calls
        if isinstance(node.func, ast.Name):
            if node.func.id in BLOCKED_BUILTINS:
                self.errors.append(
                    f"Line {node.lineno}: Call to '{node.func.id}' is blocked for security"
                )
        self.generic_visit(node)

    def visit_Subscript(self, node: ast.Subscript):
        # Block __getitem__ access to dangerous keys
        if isinstance(node.slice, ast.Constant):
            if isinstance(node.slice.value, str) and node.slice.value.startswith('__'):
                self.errors.append(
                    f"Line {node.lineno}: Access to '{node.slice.value}' is blocked"
                )
        self.generic_visit(node)


def check_code_security(code: str) -> List[str]:
    """
    Check code for security violations.

    Returns:
        List of error messages (empty if code is safe)
    """
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return [f"Syntax error on line {e.lineno}: {e.msg}"]

    checker = SecurityChecker()
    checker.visit(tree)
    return checker.errors


# =============================================================================
# Error Parser
# =============================================================================

def parse_error_for_llm(code: str, error: Exception, tb_string: str) -> ExecutionResult:
    """
    Parse an exception into an LLM-friendly error message.

    Extracts:
    - Error type (e.g., ZeroDivisionError)
    - Error line number
    - Relevant code context
    - Clear explanation
    """
    error_type = type(error).__name__
    error_msg = str(error)

    # Try to extract line number from traceback
    line_match = re.search(r'line (\d+)', tb_string)
    error_line = int(line_match.group(1)) if line_match else None

    # Get the offending line of code
    code_lines = code.split('\n')
    offending_line = ""
    if error_line and 0 < error_line <= len(code_lines):
        offending_line = code_lines[error_line - 1].strip()

    # Build LLM-friendly message
    if offending_line:
        message = f"Error on line {error_line}: {error_type}: {error_msg}\n"
        message += f"  Code: `{offending_line}`"
    else:
        message = f"{error_type}: {error_msg}"

    # Add hints for common errors
    hints = {
        "ZeroDivisionError": "Check for division by zero in your calculations.",
        "IndexError": "The list index is out of bounds. Check the list length.",
        "KeyError": "The dictionary key does not exist. Check available keys.",
        "TypeError": "Check the types of your variables and function arguments.",
        "NameError": "A variable is used before being defined.",
        "AttributeError": "The object doesn't have that attribute or method.",
        "ValueError": "The value is invalid for the operation.",
    }

    if error_type in hints:
        message += f"\nHint: {hints[error_type]}"

    return ExecutionResult(
        success=False,
        stderr=tb_string,
        error_message=message,
        error_line=error_line,
        error_type=error_type,
    )


# =============================================================================
# Execution Backends
# =============================================================================

class InlineExecutor:
    """
    Execute code inline with restricted builtins.

    WARNING: This is the least secure option. Use only for testing.
    """

    def __init__(self, timeout: float = 5.0):
        self.timeout = timeout

    def execute(self, code: str) -> ExecutionResult:
        import time
        start_time = time.perf_counter()

        # Security check first
        security_errors = check_code_security(code)
        if security_errors:
            return ExecutionResult(
                success=False,
                error_message="Security violation: " + "; ".join(security_errors),
            )

        # Prepare restricted environment
        safe_globals = {
            "__builtins__": SAFE_BUILTINS.copy(),
        }
        safe_globals.update(_get_safe_modules())

        # Capture output
        stdout_capture = io.StringIO()
        stderr_capture = io.StringIO()
        old_stdout, old_stderr = sys.stdout, sys.stderr

        result_value = None

        try:
            sys.stdout = stdout_capture
            sys.stderr = stderr_capture

            # Compile and execute
            compiled = compile(code, "<sandbox>", "exec")
            exec(compiled, safe_globals)

            # Try to get last expression value
            try:
                tree = ast.parse(code)
                if tree.body and isinstance(tree.body[-1], ast.Expr):
                    last_expr = ast.Expression(body=tree.body[-1].value)
                    compiled_expr = compile(last_expr, "<sandbox>", "eval")
                    result_value = eval(compiled_expr, safe_globals)
            except:
                pass

            execution_time = (time.perf_counter() - start_time) * 1000

            return ExecutionResult(
                success=True,
                stdout=stdout_capture.getvalue(),
                stderr=stderr_capture.getvalue(),
                return_value=result_value,
                execution_time_ms=execution_time,
            )

        except Exception as e:
            execution_time = (time.perf_counter() - start_time) * 1000
            tb_string = traceback.format_exc()

            result = parse_error_for_llm(code, e, tb_string)
            result.execution_time_ms = execution_time
            return result

        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr


class SubprocessExecutor:
    """
    Execute code in a subprocess with resource limits.

    Uses multiprocessing for isolation and resource control.
    """

    def __init__(self, timeout: float = 5.0, memory_mb: int = 128):
        self.timeout = timeout
        self.memory_mb = memory_mb

    def _execute_in_process(
        self,
        code: str,
        result_queue: multiprocessing.Queue,
    ):
        """Execute code in isolated process."""
        import time
        start_time = time.perf_counter()

        # Set resource limits (Unix only)
        try:
            # Memory limit
            memory_bytes = self.memory_mb * 1024 * 1024
            resource.setrlimit(resource.RLIMIT_AS, (memory_bytes, memory_bytes))

            # CPU time limit
            resource.setrlimit(resource.RLIMIT_CPU, (int(self.timeout) + 1, int(self.timeout) + 1))
        except (AttributeError, ValueError):
            pass  # Windows doesn't support resource limits

        # Security check
        security_errors = check_code_security(code)
        if security_errors:
            result_queue.put(ExecutionResult(
                success=False,
                error_message="Security violation: " + "; ".join(security_errors),
            ))
            return

        # Prepare environment
        safe_globals = {"__builtins__": SAFE_BUILTINS.copy()}
        safe_globals.update(_get_safe_modules())

        stdout_capture = io.StringIO()
        stderr_capture = io.StringIO()
        old_stdout, old_stderr = sys.stdout, sys.stderr

        try:
            sys.stdout = stdout_capture
            sys.stderr = stderr_capture

            compiled = compile(code, "<sandbox>", "exec")
            exec(compiled, safe_globals)

            execution_time = (time.perf_counter() - start_time) * 1000

            result_queue.put(ExecutionResult(
                success=True,
                stdout=stdout_capture.getvalue(),
                stderr=stderr_capture.getvalue(),
                execution_time_ms=execution_time,
            ))

        except Exception as e:
            execution_time = (time.perf_counter() - start_time) * 1000
            tb_string = traceback.format_exc()

            result = parse_error_for_llm(code, e, tb_string)
            result.execution_time_ms = execution_time
            result_queue.put(result)

        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr

    def execute(self, code: str) -> ExecutionResult:
        result_queue = multiprocessing.Queue()

        process = multiprocessing.Process(
            target=self._execute_in_process,
            args=(code, result_queue),
        )

        process.start()
        process.join(timeout=self.timeout)

        if process.is_alive():
            process.terminate()
            process.join()
            return ExecutionResult(
                success=False,
                error_message=f"Timeout: Code execution exceeded {self.timeout}s limit",
                error_type="TimeoutError",
            )

        if result_queue.empty():
            return ExecutionResult(
                success=False,
                error_message="Unknown error: Process terminated without result",
                error_type="RuntimeError",
            )

        return result_queue.get()


class DockerExecutor:
    """
    Execute code in a Docker container.

    Most secure option - full isolation with network disabled.
    """

    DOCKER_IMAGE = "python:3.11-slim"

    def __init__(self, timeout: float = 5.0, memory_mb: int = 128):
        self.timeout = timeout
        self.memory_mb = memory_mb
        self.client = None

        if DOCKER_AVAILABLE:
            try:
                self.client = docker.from_env()
            except docker.errors.DockerException:
                self.client = None

    def execute(self, code: str) -> ExecutionResult:
        if not self.client:
            return ExecutionResult(
                success=False,
                error_message="Docker is not available. Install docker-py: pip install docker",
                error_type="RuntimeError",
            )

        # Security check first (even though Docker is isolated)
        security_errors = check_code_security(code)
        if security_errors:
            return ExecutionResult(
                success=False,
                error_message="Security violation: " + "; ".join(security_errors),
            )

        # Wrap code in a runner script
        runner_code = f'''
import sys
import io
import traceback

code = {repr(code)}

# Restricted builtins
safe_builtins = {{
    "bool": bool, "int": int, "float": float, "str": str,
    "list": list, "dict": dict, "tuple": tuple, "set": set,
    "frozenset": frozenset, "bytes": bytes,
    "abs": abs, "all": all, "any": any, "bin": bin,
    "chr": chr, "divmod": divmod, "enumerate": enumerate,
    "filter": filter, "format": format, "hash": hash, "hex": hex,
    "isinstance": isinstance, "issubclass": issubclass,
    "iter": iter, "len": len, "map": map, "max": max, "min": min,
    "next": next, "oct": oct, "ord": ord, "pow": pow,
    "print": print, "range": range, "repr": repr, "reversed": reversed,
    "round": round, "slice": slice, "sorted": sorted, "sum": sum,
    "type": type, "zip": zip,
    "True": True, "False": False, "None": None,
    "Exception": Exception, "ValueError": ValueError,
    "TypeError": TypeError, "KeyError": KeyError,
    "IndexError": IndexError, "ZeroDivisionError": ZeroDivisionError,
}}

import math, random, statistics, collections, itertools, functools
import operator, string, re, json, datetime, decimal, fractions

safe_globals = {{"__builtins__": safe_builtins}}
safe_globals.update({{
    "math": math, "random": random, "statistics": statistics,
    "collections": collections, "itertools": itertools,
    "functools": functools, "operator": operator, "string": string,
    "re": re, "json": json, "datetime": datetime,
    "decimal": decimal, "fractions": fractions,
}})

try:
    exec(compile(code, "<sandbox>", "exec"), safe_globals)
except Exception as e:
    tb = traceback.format_exc()
    print(f"ERROR:{{type(e).__name__}}:{{str(e)}}", file=sys.stderr)
    print(tb, file=sys.stderr)
'''

        import time
        start_time = time.perf_counter()

        try:
            container = self.client.containers.run(
                self.DOCKER_IMAGE,
                command=["python", "-c", runner_code],
                detach=True,
                mem_limit=f"{self.memory_mb}m",
                network_disabled=True,
                read_only=True,
                remove=False,
            )

            # Wait for completion
            result = container.wait(timeout=self.timeout)

            stdout = container.logs(stdout=True, stderr=False).decode('utf-8')
            stderr = container.logs(stdout=False, stderr=True).decode('utf-8')

            container.remove()

            execution_time = (time.perf_counter() - start_time) * 1000

            if result['StatusCode'] != 0 or stderr:
                # Parse error from stderr
                error_match = re.search(r'ERROR:(\w+):(.+)', stderr)
                if error_match:
                    error_type = error_match.group(1)
                    error_msg = error_match.group(2)

                    # Extract line number
                    line_match = re.search(r'line (\d+)', stderr)
                    error_line = int(line_match.group(1)) if line_match else None

                    return ExecutionResult(
                        success=False,
                        stderr=stderr,
                        error_message=f"Error on line {error_line}: {error_type}: {error_msg}" if error_line else f"{error_type}: {error_msg}",
                        error_line=error_line,
                        error_type=error_type,
                        execution_time_ms=execution_time,
                    )

                return ExecutionResult(
                    success=False,
                    stderr=stderr,
                    error_message=stderr.strip(),
                    execution_time_ms=execution_time,
                )

            return ExecutionResult(
                success=True,
                stdout=stdout,
                execution_time_ms=execution_time,
            )

        except docker.errors.ContainerError as e:
            return ExecutionResult(
                success=False,
                error_message=f"Container error: {str(e)}",
                error_type="ContainerError",
            )
        except Exception as e:
            if "timeout" in str(e).lower() or "read timed out" in str(e).lower():
                return ExecutionResult(
                    success=False,
                    error_message=f"Timeout: Code execution exceeded {self.timeout}s limit",
                    error_type="TimeoutError",
                )
            return ExecutionResult(
                success=False,
                error_message=f"Execution error: {str(e)}",
                error_type=type(e).__name__,
            )


# =============================================================================
# Main Executor Class
# =============================================================================

class PythonExecutor:
    """
    Secure Python code executor with automatic backend selection.

    Backends (in order of security):
    1. Docker - Full container isolation (recommended)
    2. Subprocess - Process isolation with resource limits
    3. Inline - Restricted builtins only (least secure)

    Usage:
        executor = PythonExecutor()
        result = executor.execute("print(2 + 2)")

        if result.success:
            print(result.stdout)  # "4"
        else:
            print(result.error_message)  # LLM-friendly error
    """

    def __init__(
        self,
        timeout: float = 5.0,
        memory_mb: int = 128,
        backend: Optional[ExecutionBackend] = None,
    ):
        """
        Initialize executor.

        Args:
            timeout: Maximum execution time in seconds
            memory_mb: Maximum memory usage in MB
            backend: Execution backend (auto-selected if None)
        """
        self.timeout = timeout
        self.memory_mb = memory_mb

        # Auto-select backend
        if backend is None:
            backend = self._select_backend()

        self.backend = backend
        self._executor = self._create_executor()

    def _select_backend(self) -> ExecutionBackend:
        """Select the most secure available backend."""
        # Try Docker first
        if DOCKER_AVAILABLE:
            try:
                client = docker.from_env()
                client.ping()
                return ExecutionBackend.DOCKER
            except:
                pass

        # Fall back to subprocess (Unix with resource limits)
        if hasattr(resource, 'setrlimit'):
            return ExecutionBackend.SUBPROCESS

        # Last resort: inline with restricted builtins
        return ExecutionBackend.INLINE

    def _create_executor(self):
        """Create the appropriate executor instance."""
        if self.backend == ExecutionBackend.DOCKER:
            return DockerExecutor(self.timeout, self.memory_mb)
        elif self.backend == ExecutionBackend.SUBPROCESS:
            return SubprocessExecutor(self.timeout, self.memory_mb)
        else:
            return InlineExecutor(self.timeout)

    def execute(self, code: str) -> ExecutionResult:
        """
        Execute Python code securely.

        Args:
            code: Python code string to execute

        Returns:
            ExecutionResult with stdout, stderr, and error info
        """
        # Basic validation
        if not code or not code.strip():
            return ExecutionResult(
                success=False,
                error_message="Empty code provided",
                error_type="ValueError",
            )

        # Execute
        return self._executor.execute(code)

    def execute_for_llm(self, code: str) -> str:
        """
        Execute code and return LLM-formatted observation.

        Args:
            code: Python code to execute

        Returns:
            Observation string: <observation>Result: ...</observation>
        """
        result = self.execute(code)
        return result.to_observation()

    @property
    def backend_info(self) -> Dict[str, Any]:
        """Get information about the current backend."""
        return {
            "backend": self.backend.value,
            "timeout": self.timeout,
            "memory_mb": self.memory_mb,
            "docker_available": DOCKER_AVAILABLE,
            "restricted_python_available": RESTRICTED_PYTHON_AVAILABLE,
        }


# =============================================================================
# Convenience Functions
# =============================================================================

def execute_python(code: str, timeout: float = 5.0, memory_mb: int = 128) -> ExecutionResult:
    """
    Convenience function to execute Python code.

    Args:
        code: Python code to execute
        timeout: Maximum execution time
        memory_mb: Maximum memory

    Returns:
        ExecutionResult
    """
    executor = PythonExecutor(timeout=timeout, memory_mb=memory_mb)
    return executor.execute(code)


def verify_code(code: str) -> str:
    """
    Verify code and return observation for LLM.

    Args:
        code: Python code to verify

    Returns:
        Observation string
    """
    executor = PythonExecutor()
    return executor.execute_for_llm(code)


# =============================================================================
# Main (for testing)
# =============================================================================

if __name__ == "__main__":
    executor = PythonExecutor()
    print(f"Backend: {executor.backend.value}")
    print(f"Info: {executor.backend_info}")
    print()

    # Test cases
    tests = [
        ("Basic arithmetic", "print(2 + 2)"),
        ("Variables", "x = 10\ny = 20\nprint(x + y)"),
        ("List comprehension", "print([x**2 for x in range(5)])"),
        ("Math module", "import math\nprint(math.sqrt(16))"),
        ("Division by zero", "x = 1/0"),
        ("Name error", "print(undefined_var)"),
        ("Index error", "lst = [1,2,3]\nprint(lst[10])"),
        ("Security: os import", "import os\nos.system('ls')"),
        ("Security: __class__", "print(''.__class__)"),
        ("Timeout test", "while True: pass"),
    ]

    for name, code in tests:
        print(f"Test: {name}")
        print(f"Code: {code}")
        result = executor.execute(code)
        if result.success:
            print(f"Success: {result.stdout.strip()}")
        else:
            print(f"Error: {result.error_message}")
        print(f"Time: {result.execution_time_ms:.2f}ms")
        print("-" * 40)
