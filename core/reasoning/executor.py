"""Intelligent executor bridging plans to tools."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import json
import re
import shlex
import subprocess
from typing import Callable, Dict, List, Optional, Protocol, Tuple

from .planner import Task
from .verifiers import PythonExecutor


class Tool(Protocol):
    name: str

    def run(self, payload: Dict[str, str]) -> "ExecutionResult":
        ...


@dataclass
class ExecutionResult:
    status: str
    output: str = ""
    error: Optional[str] = None


@dataclass
class ExecutorConfig:
    timeout_seconds: int = 30
    root_dir: Optional[Path] = None
    allow_all_commands: bool = False
    allowed_commands: List[str] = field(
        default_factory=lambda: [
            "python",
            "python3",
            "pytest",
            "git",
            "rg",
            "ls",
            "dir",
        ]
    )


class Executor:
    """Execute tasks using a registry of safe tools."""

    def __init__(self, *, config: Optional[ExecutorConfig] = None) -> None:
        self.config = config or ExecutorConfig()
        self._python_executor = PythonExecutor(timeout=self.config.timeout_seconds)
        self._tool_registry: Dict[str, Callable[[Dict[str, str]], ExecutionResult]] = {
            "read_file": self._handle_read_file,
            "write_file": self._handle_write_file,
            "run_shell_command": self._handle_run_shell_command,
            "run_python_code": self._handle_run_python_code,
        }

    def execute_task(self, task: Task) -> ExecutionResult:
        task_type, payload = self._resolve_task(task)
        handler = self._tool_registry.get(task_type)
        if handler is None:
            return ExecutionResult(
                status="failed",
                error=f"UnknownTaskType: {task_type}",
            )
        try:
            return handler(payload)
        except FileNotFoundError as exc:
            return ExecutionResult(status="failed", error=f"FileNotFound: {exc}")
        except PermissionError as exc:
            return ExecutionResult(status="failed", error=f"PermissionDenied: {exc}")
        except subprocess.TimeoutExpired:
            return ExecutionResult(status="failed", error="TimeoutError: command exceeded time limit")
        except Exception as exc:  # noqa: BLE001
            return ExecutionResult(status="failed", error=f"ExecutionError: {exc}")

    def execute(self, task: Task):
        result = self.execute_task(task)
        return _to_orchestrator_result(task, result)

    def _resolve_task(self, task: Task) -> Tuple[str, Dict[str, str]]:
        payload = self._parse_payload(task)
        task_type = payload.get("task_type") or task.metadata.get("task_type")
        if task_type:
            return task_type, payload
        match = re.match(r"^\s*(\w+)\s*[:\-]", task.description)
        if match:
            task_type = match.group(1)
            payload.setdefault("raw", task.description)
            return task_type, payload
        return "run_shell_command", payload

    def _parse_payload(self, task: Task) -> Dict[str, str]:
        payload: Dict[str, str] = {}
        payload.update({str(k): str(v) for k, v in task.metadata.items()})
        raw_payload = task.metadata.get("payload")
        if isinstance(raw_payload, str):
            try:
                decoded = json.loads(raw_payload)
                if isinstance(decoded, dict):
                    payload.update({str(k): str(v) for k, v in decoded.items()})
            except json.JSONDecodeError:
                payload["raw"] = raw_payload
        return payload

    def _handle_read_file(self, payload: Dict[str, str]) -> ExecutionResult:
        path = payload.get("path") or payload.get("file")
        if not path:
            return ExecutionResult(status="failed", error="MissingArgument: path")
        resolved = self._resolve_path(path)
        content = resolved.read_text(encoding="utf-8")
        return ExecutionResult(status="success", output=content)

    def _handle_write_file(self, payload: Dict[str, str]) -> ExecutionResult:
        path = payload.get("path") or payload.get("file")
        content = payload.get("content")
        if not path:
            return ExecutionResult(status="failed", error="MissingArgument: path")
        if content is None:
            return ExecutionResult(status="failed", error="MissingArgument: content")
        resolved = self._resolve_path(path)
        resolved.parent.mkdir(parents=True, exist_ok=True)
        resolved.write_text(content, encoding="utf-8")
        return ExecutionResult(status="success", output=f"Wrote {resolved}")

    def _handle_run_shell_command(self, payload: Dict[str, str]) -> ExecutionResult:
        command = payload.get("command") or payload.get("cmd") or payload.get("raw")
        if not command:
            return ExecutionResult(status="failed", error="MissingArgument: command")
        if not self.config.allow_all_commands and self._is_command_blocked(command):
            return ExecutionResult(status="failed", error="ShellBlocked: command contains unsafe tokens")
        args = shlex.split(command)
        if not args:
            return ExecutionResult(status="failed", error="InvalidCommand: empty")
        if not self.config.allow_all_commands:
            allowed = {cmd.lower() for cmd in self.config.allowed_commands}
            if args[0].lower() not in allowed:
                return ExecutionResult(
                    status="failed",
                    error=f"CommandNotAllowed: {args[0]}",
                )
        completed = subprocess.run(
            args,
            capture_output=True,
            text=True,
            timeout=self.config.timeout_seconds,
            cwd=str(self.config.root_dir) if self.config.root_dir else None,
            check=False,
        )
        if completed.returncode != 0:
            error = completed.stderr.strip() or completed.stdout.strip()
            return ExecutionResult(
                status="failed",
                output=completed.stdout.strip(),
                error=f"ShellError({completed.returncode}): {error}",
            )
        return ExecutionResult(status="success", output=completed.stdout.strip())

    def _handle_run_python_code(self, payload: Dict[str, str]) -> ExecutionResult:
        code = payload.get("code") or payload.get("script") or payload.get("raw")
        if not code:
            return ExecutionResult(status="failed", error="MissingArgument: code")
        result = self._python_executor.execute(code)
        if result.success:
            output = result.stdout.strip() if result.stdout else str(result.return_value or "")
            return ExecutionResult(status="success", output=output)
        error = result.error_message or result.stderr or "PythonError"
        return ExecutionResult(status="failed", error=error)

    def _resolve_path(self, path: str) -> Path:
        resolved = Path(path).expanduser()
        if not resolved.is_absolute() and self.config.root_dir is not None:
            resolved = self.config.root_dir / resolved
        resolved = resolved.resolve()
        if self.config.root_dir is not None:
            root = self.config.root_dir.resolve()
            if root not in resolved.parents and resolved != root:
                raise PermissionError(f"Path outside root: {resolved}")
        return resolved

    @staticmethod
    def _is_command_blocked(command: str) -> bool:
        return bool(re.search(r"[|;&><]", command))


def _to_orchestrator_result(task: Task, result: ExecutionResult):
    from .orchestrator import ExecutionResult as OrchestratorExecutionResult

    success = result.status == "success"
    return OrchestratorExecutionResult(
        task_id=task.task_id,
        success=success,
        output=result.output if success else None,
        error=None if success else result.error,
    )
