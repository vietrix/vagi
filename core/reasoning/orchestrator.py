"""Orchestrator for hierarchical planning, simulation, and execution."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, List, Optional, Protocol, Sequence

from .planner import PlannerAgent, Task, TaskQueue, TaskStatus
from .simulator import CounterfactualSimulator, SafetyFlag, SimulationResult


@dataclass
class ExecutionResult:
    task_id: str
    success: bool
    output: Optional[str] = None
    error: Optional[str] = None
    safety_flag: Optional[str] = None
    simulation: Optional[SimulationResult] = None


class Executor(Protocol):
    def execute(self, task: Task) -> ExecutionResult:
        ...


@dataclass
class OrchestratorConfig:
    max_replans: int = 3


@dataclass
class OrchestratorResult:
    plans: List[TaskQueue] = field(default_factory=list)
    results: List[ExecutionResult] = field(default_factory=list)


class Orchestrator:
    def __init__(
        self,
        planner: PlannerAgent,
        simulator: CounterfactualSimulator,
        executor: Executor,
        *,
        config: Optional[OrchestratorConfig] = None,
    ) -> None:
        self.planner = planner
        self.simulator = simulator
        self.executor = executor
        self.config = config or OrchestratorConfig()

    def run(
        self,
        user_request: str,
        *,
        plan_llm_fn: Optional[Callable[[str], str]] = None,
        sim_llm_fn: Optional[Callable[[str], str]] = None,
    ) -> OrchestratorResult:
        result = OrchestratorResult()
        plan = self.planner.plan(user_request, llm_fn=plan_llm_fn)
        result.plans.append(plan)

        replans = 0
        while replans <= self.config.max_replans:
            progress = False
            for task in plan.tasks:
                if task.status in {TaskStatus.DONE, TaskStatus.FAILED, TaskStatus.BLOCKED}:
                    continue
                if not self._dependencies_satisfied(task, plan.tasks):
                    task.status = TaskStatus.BLOCKED
                    continue

                task.status = TaskStatus.IN_PROGRESS
                try:
                    simulation = self.simulator.simulate_outcome(
                        task.description,
                        context=user_request,
                        llm_fn=sim_llm_fn,
                    )
                except SafetyFlag as flag:
                    task.status = TaskStatus.FAILED
                    result.results.append(
                        ExecutionResult(
                            task_id=task.task_id,
                            success=False,
                            error=str(flag),
                            safety_flag=str(flag),
                        )
                    )
                    plan = self.planner.replan(str(flag), plan, llm_fn=plan_llm_fn)
                    result.plans.append(plan)
                    replans += 1
                    progress = True
                    break

                exec_result = self.executor.execute(task)
                exec_result.simulation = simulation
                result.results.append(exec_result)
                if exec_result.success:
                    task.status = TaskStatus.DONE
                else:
                    task.status = TaskStatus.FAILED
                    plan = self.planner.replan(
                        exec_result.error or "Unknown execution error",
                        plan,
                        llm_fn=plan_llm_fn,
                    )
                    result.plans.append(plan)
                    replans += 1
                    progress = True
                    break
                progress = True
            if not progress:
                break

        return result

    @staticmethod
    def _dependencies_satisfied(task: Task, tasks: Sequence[Task]) -> bool:
        if not task.dependencies:
            return True
        status_map = {t.task_id: t.status for t in tasks}
        return all(status_map.get(dep) == TaskStatus.DONE for dep in task.dependencies)
