"""Hierarchical planner for decomposing goals into dependent tasks."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
import json
import re
from typing import Callable, Dict, List, Optional, Sequence


def _utc_now() -> datetime:
    return datetime.now(tz=timezone.utc)


class TaskStatus(str, Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    DONE = "done"
    FAILED = "failed"
    BLOCKED = "blocked"


@dataclass
class Task:
    task_id: str
    description: str
    dependencies: List[str] = field(default_factory=list)
    status: TaskStatus = TaskStatus.PENDING
    metadata: Dict[str, str] = field(default_factory=dict)


@dataclass
class TaskQueue:
    goal: str
    tasks: List[Task]
    created_at: datetime = field(default_factory=_utc_now)

    def to_json(self) -> Dict[str, object]:
        return {
            "goal": self.goal,
            "created_at": self.created_at.isoformat(),
            "tasks": [
                {
                    "id": task.task_id,
                    "description": task.description,
                    "dependencies": task.dependencies,
                    "status": task.status.value,
                    "metadata": task.metadata,
                }
                for task in self.tasks
            ],
        }

    def remaining_tasks(self) -> List[Task]:
        return [task for task in self.tasks if task.status not in {TaskStatus.DONE}]


@dataclass
class PlannerConfig:
    max_tasks: int = 12
    prompt_template: str = (
        "You are a planning agent. Decompose the goal into a list of tasks.\n"
        "Identify dependencies explicitly. Output JSON only in this schema:\n"
        "{ \"tasks\": [ {\"id\": \"T1\", \"description\": \"...\", \"dependencies\": [\"T0\"]} ] }\n\n"
        "Goal: {goal}\n"
    )
    replan_template: str = (
        "A task failed. Update the remaining plan to reach the goal.\n"
        "Provide JSON only in the same schema as before.\n\n"
        "Goal: {goal}\n"
        "Error: {error}\n"
        "Remaining Tasks: {remaining}\n"
    )


class PlannerAgent:
    def __init__(self, *, config: Optional[PlannerConfig] = None) -> None:
        self.config = config or PlannerConfig()

    def plan(self, goal: str, llm_fn: Optional[Callable[[str], str]] = None) -> TaskQueue:
        if llm_fn is None:
            return TaskQueue(goal=goal, tasks=[Task(task_id="T1", description=goal)])
        prompt = self.config.prompt_template.format(goal=goal)
        response = llm_fn(prompt)
        tasks = self._parse_tasks(response)
        if not tasks:
            tasks = [Task(task_id="T1", description=goal)]
        return TaskQueue(goal=goal, tasks=tasks[: self.config.max_tasks])

    def replan(
        self,
        error: str,
        current_queue: TaskQueue,
        llm_fn: Optional[Callable[[str], str]] = None,
    ) -> TaskQueue:
        remaining = [task for task in current_queue.tasks if task.status != TaskStatus.DONE]
        if llm_fn is None:
            fallback = Task(
                task_id=f"T{len(remaining) + 1}",
                description=f"Investigate alternative for failure: {error}",
                dependencies=[task.task_id for task in remaining if task.status == TaskStatus.FAILED],
            )
            return TaskQueue(goal=current_queue.goal, tasks=remaining + [fallback])
        remaining_json = json.dumps(
            [self._task_to_dict(task) for task in remaining],
            ensure_ascii=False,
        )
        prompt = self.config.replan_template.format(
            goal=current_queue.goal,
            error=error,
            remaining=remaining_json,
        )
        response = llm_fn(prompt)
        tasks = self._parse_tasks(response)
        if not tasks:
            tasks = remaining
        return TaskQueue(goal=current_queue.goal, tasks=tasks[: self.config.max_tasks])

    def _parse_tasks(self, response: str) -> List[Task]:
        if not response:
            return []
        payload = self._extract_json(response)
        if isinstance(payload, dict) and "tasks" in payload:
            raw_tasks = payload.get("tasks") or []
            return self._normalize_tasks(raw_tasks)
        return self._fallback_parse(response)

    @staticmethod
    def _extract_json(text: str) -> Optional[Dict[str, object]]:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not match:
            return None
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            return None

    @staticmethod
    def _normalize_tasks(raw_tasks: Sequence[object]) -> List[Task]:
        tasks: List[Task] = []
        for idx, raw in enumerate(raw_tasks, start=1):
            if not isinstance(raw, dict):
                continue
            task_id = str(raw.get("id") or raw.get("task_id") or f"T{idx}")
            desc = str(raw.get("description") or raw.get("task") or "").strip()
            if not desc:
                continue
            deps_raw = raw.get("dependencies") or raw.get("deps") or []
            dependencies = [str(dep) for dep in deps_raw] if isinstance(deps_raw, list) else []
            tasks.append(Task(task_id=task_id, description=desc, dependencies=dependencies))
        return tasks

    @staticmethod
    def _fallback_parse(text: str) -> List[Task]:
        tasks: List[Task] = []
        for idx, line in enumerate(text.splitlines(), start=1):
            line = line.strip()
            if not line:
                continue
            line = re.sub(r"^[-*]\s+", "", line)
            line = re.sub(r"^\d+[.)]\s+", "", line)
            if line:
                tasks.append(Task(task_id=f"T{idx}", description=line))
        return tasks

    @staticmethod
    def _task_to_dict(task: Task) -> Dict[str, object]:
        return {
            "id": task.task_id,
            "description": task.description,
            "dependencies": task.dependencies,
            "status": task.status.value,
            "metadata": task.metadata,
        }
