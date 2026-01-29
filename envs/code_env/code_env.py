"""Minimal code-editing environment for vAGI."""

from __future__ import annotations

import hashlib
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Dict, List, Tuple

import torch

from .actions import (
    ACTION_DIM,
    ApplyPatchAction,
    ListDirAction,
    PlanLocateSourceAction,
    PlanPatchAction,
    PlanReadErrorsAction,
    PlanVerifyAction,
    ReadFileAction,
    RunTestsAction,
    SearchAction,
    action_group,
    is_info_action,
    action_type_id,
    parse_action,
    serialize_action,
)
from .obs import text_to_obs


PATCH_SEPARATOR = "\n---\n"


class CodeEnv:
    """Tiny code environment that runs pytest on a fixture repo."""

    def __init__(
        self,
        obs_dim: int = 64,
        max_steps: int = 8,
        max_run_tests: int = 3,
        max_patch_chars: int = 2000,
        max_output_chars: int = 2000,
        seed: int = 0,
        repo_path: str | Path | None = None,
        copy_repo: bool = True,
        require_read_before_write: bool = True,
        info_reward_bonus: float = 0.5,
    ) -> None:
        if obs_dim <= 0:
            raise ValueError("obs_dim must be > 0")
        if max_steps <= 0:
            raise ValueError("max_steps must be > 0")
        if max_run_tests <= 0:
            raise ValueError("max_run_tests must be > 0")
        if max_patch_chars <= 0:
            raise ValueError("max_patch_chars must be > 0")
        if max_output_chars <= 0:
            raise ValueError("max_output_chars must be > 0")
        self.obs_dim = obs_dim
        self.max_steps = max_steps
        self.max_run_tests = max_run_tests
        self.max_patch_chars = max_patch_chars
        self.max_output_chars = max_output_chars
        self.seed_value = seed
        self.copy_repo = copy_repo
        self.require_read_before_write = require_read_before_write
        self.info_reward_bonus = info_reward_bonus
        self.workspace_root = Path(__file__).resolve().parent / "fixtures"
        self.fixture_root = Path(repo_path) if repo_path is not None else self._fixture_path()
        if not self.fixture_root.resolve().is_relative_to(self.workspace_root):
            raise ValueError("repo_path must be inside fixtures workspace")
        self._work_dir: Path | None = None
        self.last_output = ""
        self.last_fail_count = 0
        self.last_failing_tests: List[str] = []
        self.last_error_type = ""
        self.last_action_id = -1
        self.last_tool_ok = False
        self.last_changed_files = 0
        self.last_patch_applied = False
        self.last_output_len = 0
        self.last_info_action = False
        self.prev_info_action = False
        self.run_tests_count = 0
        self.step_count = 0
        self.initial_fail_count = 0
        self.plan_state = 0
        self._read_files: set[str] = set()
        self._last_digest: Dict[str, Tuple[int, str]] = {}
        self._total_files = 0
        self._last_patch_backup: Tuple[str, str] | None = None

    def seed(self, seed: int) -> int:
        self.seed_value = seed
        return seed

    def reset(self) -> torch.Tensor:
        self._reset_repo()
        self.step_count = 0
        self.run_tests_count = 0
        self._read_files = set()
        self.last_fail_count, self.last_output, self.last_failing_tests, self.last_error_type = self._run_tests()
        self.initial_fail_count = self.last_fail_count
        self.last_action_id = -1
        self.last_tool_ok = True
        self.last_changed_files = 0
        self.last_patch_applied = False
        self.last_output_len = min(len(self.last_output), self.max_output_chars)
        self.last_info_action = False
        self.prev_info_action = False
        _digest_text, digest = _tree_digest(self.repo_path)
        self._last_digest = digest
        self._total_files = len(digest)
        self.plan_state = 0
        return self._build_obs()

    def step(self, action: str) -> Tuple[torch.Tensor, float, bool, dict]:
        parsed = parse_action(action)
        reward = -0.1
        done = False
        info = {"action": serialize_action(parsed)}
        self.last_patch_applied = False
        self.last_tool_ok = False
        self.last_action_id = action_type_id(parsed)
        self.prev_info_action = self.last_info_action
        self.last_info_action = is_info_action(parsed)

        if isinstance(parsed, ReadFileAction):
            self.last_output = self._read_file(parsed.path)
            self.last_tool_ok = True
        elif isinstance(parsed, ListDirAction):
            self.last_output = self._list_dir(parsed.path)
            self.last_tool_ok = True
        elif isinstance(parsed, SearchAction):
            self.last_output = self._search(parsed.pattern)
            self.last_tool_ok = True
        elif isinstance(parsed, PlanReadErrorsAction):
            self.plan_state = 1
            self.last_output = "PLAN_READ_ERRORS"
            self.last_tool_ok = True
        elif isinstance(parsed, PlanLocateSourceAction):
            self.plan_state = 2
            self.last_output = "PLAN_LOCATE_SOURCE"
            self.last_tool_ok = True
        elif isinstance(parsed, PlanPatchAction):
            self.plan_state = 3
            self.last_output = "PLAN_PATCH"
            self.last_tool_ok = True
        elif isinstance(parsed, PlanVerifyAction):
            self.plan_state = 4
            self.last_output = "PLAN_VERIFY"
            self.last_tool_ok = True
        elif isinstance(parsed, ApplyPatchAction):
            applied, reason = self._apply_patch_action(parsed.path, parsed.diff)
            self.last_patch_applied = applied
            self.last_tool_ok = applied
            self.last_output = reason
            if applied:
                reward += 0.2
        elif isinstance(parsed, RunTestsAction):
            if self.run_tests_count >= self.max_run_tests:
                self.last_output = "ERROR: run_tests_limit_exceeded"
                self.last_tool_ok = False
                reward -= 0.2
                done = True
            else:
                prev_fail = self.last_fail_count
                self.last_fail_count, self.last_output, self.last_failing_tests, self.last_error_type = self._run_tests()
                self.run_tests_count += 1
                self.last_tool_ok = True
                if self.last_fail_count < prev_fail:
                    reward += 1.0
                    if self.prev_info_action:
                        reward += self.info_reward_bonus
                if self.last_fail_count == 0:
                    reward += 10.0
                    done = True
        else:
            raise ValueError("Unsupported action")

        self.step_count += 1
        if self.step_count >= self.max_steps:
            done = True

        self.last_output_len = min(len(self.last_output), self.max_output_chars)
        obs = self._build_obs()
        info.update(
            {
                "fail_count": self.last_fail_count,
                "failing_tests": self.last_failing_tests,
                "top_error_type": self.last_error_type,
                "timestep": self.step_count,
                "action_id": self.last_action_id,
                "tool_ok": self.last_tool_ok,
                "patch_applied": self.last_patch_applied,
                "run_tests_count": self.run_tests_count,
                "changed_files": self.last_changed_files,
                "plan_state": self.plan_state,
                "info_action": self.last_info_action,
                "action_group": action_group(parsed),
            }
        )
        return obs, reward, done, info

    def _fixture_path(self) -> Path:
        return Path(__file__).resolve().parent / "fixtures" / "mini_repo"

    def _reset_repo(self) -> None:
        if self._work_dir and self._work_dir.exists():
            shutil.rmtree(self._work_dir)
        if self.copy_repo:
            temp_dir = Path(tempfile.mkdtemp(prefix="code_env_"))
            shutil.copytree(self.fixture_root, temp_dir, dirs_exist_ok=True)
            self._work_dir = temp_dir
        else:
            self._work_dir = self.fixture_root

    @property
    def repo_path(self) -> Path:
        if self._work_dir is None:
            raise RuntimeError("Environment not reset.")
        return self._work_dir

    def _run_tests(self) -> Tuple[int, str, List[str], str]:
        result = subprocess.run(
            [sys.executable, "-m", "pytest", "-q"],
            cwd=self.repo_path,
            capture_output=True,
            text=True,
            check=False,
        )
        output = (result.stdout or "") + (result.stderr or "")
        fail_count, failing_tests, error_type = _parse_test_output(output, result.returncode)
        return fail_count, output.strip(), failing_tests, error_type

    def _read_file(self, rel_path: str) -> str:
        path = self._safe_path(rel_path)
        text = path.read_text(encoding="utf-8")
        self._read_files.add(path.as_posix())
        return _truncate_output(text, self.max_output_chars)

    def _list_dir(self, rel_path: str) -> str:
        path = self._safe_path(rel_path, expect_dir=True)
        entries = []
        for child in sorted(path.iterdir()):
            if child.name.startswith("."):
                continue
            suffix = "/" if child.is_dir() else ""
            entries.append(f"{child.name}{suffix}")
        text = "\n".join(entries)
        return _truncate_output(text, self.max_output_chars)

    def _search(self, pattern: str) -> str:
        results = []
        for path in sorted(self.repo_path.rglob("*")):
            if path.is_dir():
                continue
            rel = path.relative_to(self.repo_path).as_posix()
            if rel.startswith(".") or "__pycache__" in rel or ".pytest_cache" in rel:
                continue
            text = path.read_text(encoding="utf-8")
            for idx, line in enumerate(text.splitlines(), start=1):
                if pattern in line:
                    results.append(f"{rel}:{idx}:{line.strip()}")
                if len(results) >= 25:
                    break
            if len(results) >= 25:
                break
        return _truncate_output("\n".join(results), self.max_output_chars)

    def _apply_patch_action(self, rel_path: str, diff: str) -> Tuple[bool, str]:
        if len(diff) > self.max_patch_chars:
            return False, "ERROR: patch_too_large"
        path = self._safe_path(rel_path)
        path_key = path.as_posix()
        if self.require_read_before_write and path_key not in self._read_files:
            return False, "ERROR: read_required_before_write"
        text = path.read_text(encoding="utf-8")
        updated, applied = _apply_patch(text, diff)
        if not applied:
            return False, "ERROR: patch_not_applied"
        path.write_text(updated, encoding="utf-8")
        self._last_patch_backup = (path_key, text)
        return True, "OK: patch_applied"

    def rollback_last_patch(self) -> bool:
        if self._last_patch_backup is None:
            return False
        path_key, text = self._last_patch_backup
        path = self._safe_path(path_key)
        path.write_text(text, encoding="utf-8")
        self._last_patch_backup = None
        return True

    def refresh_obs(self) -> torch.Tensor:
        """Rebuild observation without advancing the environment."""
        return self._build_obs()

    def _safe_path(self, rel_path: str, expect_dir: bool = False) -> Path:
        path = (self.repo_path / rel_path).resolve()
        if not path.exists():
            raise FileNotFoundError(f"Missing path: {rel_path}")
        if not path.is_relative_to(self.repo_path):
            raise ValueError("Path escapes sandbox")
        if expect_dir and not path.is_dir():
            raise NotADirectoryError(rel_path)
        if not expect_dir and not path.is_file():
            raise FileNotFoundError(f"Missing file: {rel_path}")
        return path

    def _build_obs(self) -> torch.Tensor:
        digest_text, digest = _tree_digest(self.repo_path)
        changed_files = _count_changed(self._last_digest, digest)
        self._last_digest = digest
        self.last_changed_files = changed_files
        text = f"{self.last_output}\n{digest_text}\n{' '.join(self.last_failing_tests)}\n{self.last_error_type}"
        features = _build_features(
            fail_count=self.last_fail_count,
            initial_fail=max(self.initial_fail_count, 1),
            step=self.step_count,
            max_steps=self.max_steps,
            run_tests=self.run_tests_count,
            max_run_tests=self.max_run_tests,
            changed_files=changed_files,
            total_files=max(self._total_files, 1),
            action_id=self.last_action_id,
            action_space=ACTION_DIM,
            plan_state=self.plan_state,
            plan_space=4,
            failing_tests_count=len(self.last_failing_tests),
            tool_ok=self.last_tool_ok,
            patch_applied=self.last_patch_applied,
            output_len=self.last_output_len,
            max_output=self.max_output_chars,
        )
        feature_slots = min(len(features), self.obs_dim)
        return text_to_obs(text, self.obs_dim, features=features[:feature_slots], feature_slots=feature_slots)


def _apply_patch(text: str, patch: str) -> Tuple[str, bool]:
    if PATCH_SEPARATOR not in patch:
        return text, False
    old, new = patch.split(PATCH_SEPARATOR, 1)
    if old not in text:
        return text, False
    return text.replace(old, new, 1), True


def _tree_digest(root: Path) -> Tuple[str, Dict[str, Tuple[int, str]]]:
    entries: List[str] = []
    digest_map: Dict[str, Tuple[int, str]] = {}
    for path in sorted(root.rglob("*")):
        if path.is_dir():
            continue
        rel = path.relative_to(root).as_posix()
        if rel.startswith(".") or "__pycache__" in rel or ".pytest_cache" in rel:
            continue
        data = path.read_bytes()
        size = len(data)
        digest = hashlib.sha256(data).hexdigest()[:8]
        entries.append(f"{rel}:{size}:{digest}")
        digest_map[rel] = (size, digest)
    return "\n".join(entries), digest_map


def _parse_test_output(output: str, returncode: int) -> Tuple[int, List[str], str]:
    failing_tests: List[str] = []
    error_type = ""
    for line in output.splitlines():
        if "::" in line and ("FAILED" in line or "ERROR" in line):
            failing_tests.append(line.split(" ")[0])
        if line.lstrip().startswith("E   ") and not error_type:
            error_type = line.strip().split(":", 1)[0].replace("E", "").strip()
    fail_count = _parse_fail_count(output, returncode)
    if fail_count == 0 and failing_tests:
        fail_count = len(failing_tests)
    return fail_count, failing_tests[:10], error_type


def _parse_fail_count(output: str, returncode: int) -> int:
    for token in output.split():
        if token.endswith("failed"):
            try:
                return int(token.split("failed")[0])
            except ValueError:
                continue
    summary = re.search(r"=+\\s*(\\d+) failed", output)
    if summary:
        try:
            return int(summary.group(1))
        except ValueError:
            pass
    return 0 if returncode == 0 else 1


def _truncate_output(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return text[:max_chars]


def _count_changed(prev: Dict[str, Tuple[int, str]], current: Dict[str, Tuple[int, str]]) -> int:
    changed = 0
    for path, digest in current.items():
        if path not in prev or prev[path] != digest:
            changed += 1
    return changed


def _build_features(
    *,
    fail_count: int,
    initial_fail: int,
    step: int,
    max_steps: int,
    run_tests: int,
    max_run_tests: int,
    changed_files: int,
    total_files: int,
    action_id: int,
    action_space: int,
    plan_state: int,
    plan_space: int,
    failing_tests_count: int,
    tool_ok: bool,
    patch_applied: bool,
    output_len: int,
    max_output: int,
) -> List[float]:
    return [
        _norm(fail_count, initial_fail),
        _norm(run_tests, max_run_tests),
        _norm(step, max_steps),
        _norm(changed_files, total_files),
        _norm(max(action_id, 0), max(action_space - 1, 1)),
        _norm(failing_tests_count, max(initial_fail, 1)),
        _norm(plan_state, max(plan_space, 1)),
        1.0 if tool_ok else 0.0,
        1.0 if patch_applied else 0.0,
        _norm(output_len, max_output),
    ]


def _norm(value: int, scale: int) -> float:
    if scale <= 0:
        return float(value)
    return min(1.0, float(value) / float(scale))
