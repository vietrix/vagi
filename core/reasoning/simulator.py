"""Counterfactual simulator for pre-execution safety checks."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import re
from typing import Callable, List, Optional, Sequence


class SafetyFlag(RuntimeError):
    def __init__(self, message: str, outcomes: Sequence["NegativeOutcome"]) -> None:
        super().__init__(message)
        self.outcomes = list(outcomes)


@dataclass
class NegativeOutcome:
    label: str
    probability: float


@dataclass
class SimulationResult:
    action: str
    context: str
    outcomes: List[NegativeOutcome] = field(default_factory=list)
    safe: bool = True


@dataclass
class SimulatorConfig:
    risk_threshold: float = 0.7
    prompt_template: str = (
        "You are a safety simulation model.\n"
        "If I perform {action} in {context}, what are 3 possible negative outcomes?\n"
        "Return JSON only in this schema:\n"
        "{ \"outcomes\": [ {\"label\": \"Data Loss\", \"probability\": 0.8} ] }\n"
    )


class CounterfactualSimulator:
    def __init__(self, *, config: Optional[SimulatorConfig] = None) -> None:
        self.config = config or SimulatorConfig()

    def simulate_outcome(
        self,
        action: str,
        context: str,
        llm_fn: Optional[Callable[[str], str]] = None,
    ) -> SimulationResult:
        if llm_fn is None:
            return SimulationResult(action=action, context=context, safe=True)
        prompt = self.config.prompt_template.format(action=action, context=context)
        response = llm_fn(prompt)
        outcomes = self._parse_outcomes(response)
        unsafe = [o for o in outcomes if o.probability >= self.config.risk_threshold]
        if unsafe:
            raise SafetyFlag(self._format_flag(unsafe), outcomes)
        return SimulationResult(action=action, context=context, outcomes=outcomes, safe=True)

    def _parse_outcomes(self, response: str) -> List[NegativeOutcome]:
        if not response:
            return []
        payload = self._extract_json(response)
        outcomes: List[NegativeOutcome] = []
        if isinstance(payload, dict):
            raw = payload.get("outcomes") or []
            if isinstance(raw, list):
                outcomes.extend(self._normalize_outcomes(raw))
        if not outcomes:
            outcomes = self._fallback_parse(response)
        return outcomes

    @staticmethod
    def _extract_json(text: str) -> Optional[dict]:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not match:
            return None
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            return None

    @staticmethod
    def _normalize_outcomes(raw: Sequence[object]) -> List[NegativeOutcome]:
        outcomes: List[NegativeOutcome] = []
        for item in raw:
            if not isinstance(item, dict):
                continue
            label = str(item.get("label") or item.get("outcome") or "").strip()
            if not label:
                continue
            probability = float(item.get("probability", 0.0))
            probability = max(0.0, min(1.0, probability))
            outcomes.append(NegativeOutcome(label=label, probability=probability))
        return outcomes

    @staticmethod
    def _fallback_parse(text: str) -> List[NegativeOutcome]:
        outcomes: List[NegativeOutcome] = []
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            line = re.sub(r"^[-*]\s+", "", line)
            match = re.search(r"([A-Za-z0-9 _-]+)\s*[:\-]\s*([01](?:\.\d+)?)", line)
            if match:
                label = match.group(1).strip()
                probability = float(match.group(2))
                probability = max(0.0, min(1.0, probability))
                outcomes.append(NegativeOutcome(label=label, probability=probability))
        return outcomes

    @staticmethod
    def _format_flag(outcomes: Sequence[NegativeOutcome]) -> str:
        summary = ", ".join(f"{o.label} ({o.probability:.2f})" for o in outcomes)
        return f"SafetyFlag: high-risk outcomes predicted: {summary}"
