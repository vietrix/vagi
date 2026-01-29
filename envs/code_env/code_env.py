"""Minimal code-editing environment for vAGI."""

from __future__ import annotations

import hashlib
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Tuple

import torch

from .actions import EditAction, RunTestsAction, action_type_id, parse_action, serialize_action
from .obs import text_to_obs


PATCH_SEPARATOR = "\n---\n"


class CodeEnv:
    """Tiny code environment that runs pytest on a fixture repo."""

    def __init__(
        self,
        obs_dim: int = 64,
        max_steps: int = 8,
        seed: int = 0,
        repo_path: str | Path | None = None,
        copy_repo: bool = True,
    ) -> None:
        if obs_dim <= 0:
            raise ValueError("obs_dim must be > 0")
        if max_steps <= 0:
            raise ValueError("max_steps must be > 0")
        self.obs_dim = obs_dim
        self.max_steps = max_steps
        self.seed_value = seed
        self.copy_repo = copy_repo
        self.fixture_root = Path(repo_path) if repo_path is not None else self._fixture_path()
        self._work_dir: Path | None = None
        self.last_output = ""
        self.last_fail_count = 0
        self.step_count = 0

    def seed(self, seed: int) -> int:
        self.seed_value = seed
        return seed

    def reset(self) -> torch.Tensor:
        self._reset_repo()
        self.step_count = 0
        self.last_fail_count, self.last_output = self._run_tests()
        return self._build_obs()

    def step(self, action: str) -> Tuple[torch.Tensor, float, bool, dict]:
        parsed = parse_action(action)
        reward = -0.1
        done = False
        info = {"action": serialize_action(parsed)}

        if isinstance(parsed, EditAction):
            applied = self._apply_edit(parsed.path, parsed.patch)
            info["patch_applied"] = applied
        elif isinstance(parsed, RunTestsAction):
            prev_fail = self.last_fail_count
            self.last_fail_count, self.last_output = self._run_tests()
            if self.last_fail_count < prev_fail:
                reward += 1.0
            if self.last_fail_count == 0:
                reward += 10.0
                done = True
        else:
            raise ValueError("Unsupported action")

        self.step_count += 1
        if self.step_count >= self.max_steps:
            done = True

        obs = self._build_obs()
        info.update({"fail_count": self.last_fail_count, "timestep": self.step_count, "action_id": action_type_id(parsed)})
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

    def _run_tests(self) -> Tuple[int, str]:
        result = subprocess.run(
            [sys.executable, "-m", "pytest", "-q"],
            cwd=self.repo_path,
            capture_output=True,
            text=True,
            check=False,
        )
        output = (result.stdout or "") + (result.stderr or "")
        fail_count = _parse_fail_count(output, result.returncode)
        return fail_count, output.strip()

    def _apply_edit(self, rel_path: str, patch: str) -> bool:
        path = (self.repo_path / rel_path).resolve()
        if not path.exists() or not path.is_file():
            raise FileNotFoundError(f"Missing file: {rel_path}")
        text = path.read_text(encoding="utf-8")
        updated, applied = _apply_patch(text, patch)
        if not applied:
            return False
        path.write_text(updated, encoding="utf-8")
        return True

    def _build_obs(self) -> torch.Tensor:
        digest = _tree_digest(self.repo_path)
        text = f"{self.last_output}\n{digest}"
        return text_to_obs(text, self.obs_dim, extra=[float(self.last_fail_count), float(self.step_count)])


def _apply_patch(text: str, patch: str) -> Tuple[str, bool]:
    if PATCH_SEPARATOR not in patch:
        return text, False
    old, new = patch.split(PATCH_SEPARATOR, 1)
    if old not in text:
        return text, False
    return text.replace(old, new, 1), True


def _tree_digest(root: Path) -> str:
    entries = []
    for path in sorted(root.rglob("*")):
        if path.is_dir():
            continue
        rel = path.relative_to(root).as_posix()
        if rel.startswith(".") or "__pycache__" in rel or ".pytest_cache" in rel:
            continue
        digest = hashlib.sha256(path.read_bytes()).hexdigest()[:8]
        entries.append(f"{rel}:{digest}")
    return "\n".join(entries)


def _parse_fail_count(output: str, returncode: int) -> int:
    for token in output.split():
        if token.endswith("failed"):
            try:
                return int(token.split("failed")[0])
            except ValueError:
                continue
    for line in output.splitlines():
        if "failed" in line:
            for part in line.split(","):
                if "failed" in part:
                    try:
                        return int(part.strip().split(" ")[0])
                    except ValueError:
                        continue
    return 0 if returncode == 0 else 1
