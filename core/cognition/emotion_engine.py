"""Emotional state engine using the PAD (Pleasure, Arousal, Dominance) model."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
import re
import time
from typing import Callable, Optional, Tuple


def _clamp(value: float, low: float = -1.0, high: float = 1.0) -> float:
    return max(low, min(high, float(value)))


@dataclass
class PADState:
    """Pleasure-Arousal-Dominance state vector."""

    pleasure: float = 0.0
    arousal: float = 0.0
    dominance: float = 0.0

    def clamp(self) -> "PADState":
        return PADState(
            pleasure=_clamp(self.pleasure),
            arousal=_clamp(self.arousal),
            dominance=_clamp(self.dominance),
        )

    def to_tuple(self) -> Tuple[float, float, float]:
        return (self.pleasure, self.arousal, self.dominance)

    def apply_delta(self, delta: "PADState") -> "PADState":
        return PADState(
            pleasure=_clamp(self.pleasure + delta.pleasure),
            arousal=_clamp(self.arousal + delta.arousal),
            dominance=_clamp(self.dominance + delta.dominance),
        )

    def decay(self, factor: float) -> "PADState":
        return PADState(
            pleasure=self.pleasure * factor,
            arousal=self.arousal * factor,
            dominance=self.dominance * factor,
        )


class EmotionEngine:
    """Internal emotion engine with PAD updates and homeostasis."""

    def __init__(
        self,
        *,
        llm_fn: Optional[Callable[[str], str]] = None,
        decay_rate: float = 0.01,
    ) -> None:
        self.pad_state = [0.0, 0.0, 0.0]
        self.state = PADState()
        self.llm_fn = llm_fn
        self.decay_rate = decay_rate
        self._last_update_time = time.time()

    def update(
        self,
        user_prompt: str,
        recent_context: str,
        *,
        llm_fn: Optional[Callable[[str], str]] = None,
        now: Optional[float] = None,
    ) -> PADState:
        """Update PAD state from input + context via lightweight LLM call."""
        self.apply_homeostasis(now=now)
        prompt = self._build_prompt(user_prompt, recent_context)
        llm = llm_fn or self.llm_fn
        if llm is None:
            return self.state
        response = llm(prompt)
        delta = self._parse_delta(response)
        self.state = self.state.apply_delta(delta)
        return self.state

    def update_state(
        self,
        input_text: str,
        llm_fn: Optional[Callable[[str], str]] = None,
        *,
        recent_context: str = "",
        decay: float = 0.9,
        now: Optional[float] = None,
    ) -> PADState:
        """Update PAD state using LLM delta vector with homeostasis."""
        self.apply_homeostasis(now=now)
        prompt = self._build_delta_prompt(input_text, recent_context)
        llm = llm_fn or self.llm_fn
        if llm is None:
            return self.state
        response = llm(prompt)
        delta = self._parse_delta_vector(response)
        self._apply_delta(delta, decay=decay)
        return self.state

    def update_state_legacy(
        self,
        user_input: str,
        recent_context: str = "",
        *,
        llm_fn: Optional[Callable[[str], str]] = None,
        now: Optional[float] = None,
    ) -> PADState:
        """Alias for update() to match external agent contracts."""
        return self.update(
            user_prompt=user_input,
            recent_context=recent_context,
            llm_fn=llm_fn,
            now=now,
        )

    def apply_homeostasis(self, *, now: Optional[float] = None) -> PADState:
        """Decay emotions toward neutral over time."""
        now = now or time.time()
        dt = max(0.0, now - self._last_update_time)
        factor = math.exp(-self.decay_rate * dt)
        self.state = self.state.decay(factor)
        self._last_update_time = now
        return self.state

    def current_mood_label(self) -> str:
        """Map PAD vector to a discrete mood label."""
        pleasure, arousal, dominance = self._pad_tuple()
        if pleasure >= 0.3 and arousal >= 0.3:
            label = "Excited"
        elif pleasure <= -0.3 and arousal >= 0.3:
            label = "Frustrated"
        elif pleasure >= 0.3 and arousal <= -0.3:
            label = "Calm"
        elif pleasure <= -0.3 and arousal <= -0.3:
            label = "Sad"
        else:
            label = "Neutral"

        if dominance >= 0.5 and label in {"Excited", "Calm", "Neutral"}:
            return "Confident"
        if dominance <= -0.5 and label in {"Excited", "Frustrated", "Neutral"}:
            return "Anxious"
        return label

    def build_system_injection(self) -> str:
        """Create system instruction for tone adjustment."""
        mood = self.current_mood_label()
        return f"Current Mood: {mood}. Adjust your tone accordingly."

    def get_tone_instruction(self) -> str:
        """Map PAD values to tone instructions."""
        pleasure, arousal, dominance = self._pad_tuple()
        instructions = []

        if pleasure >= 0.4 and arousal >= 0.4:
            instructions.append("Enthusiastic, Gen Z slang, Emojis.")
        elif pleasure <= -0.4 and arousal >= 0.4:
            instructions.append("Annoyed, Sharp, Short sentences.")
        else:
            instructions.append("Neutral, Helpful, Clear.")

        if dominance <= -0.3:
            instructions.append("Uncertain, Humble, Asking for help.")

        return " ".join(instructions).strip()

    @staticmethod
    def _build_prompt(user_prompt: str, recent_context: str) -> str:
        return (
            "You are an internal affect evaluator.\n"
            "Given the user input and recent context, output JSON with "
            "keys pleasure, arousal, dominance in range [-1.0, 1.0].\n\n"
            f"User Input:\n{user_prompt}\n\n"
            f"Recent Context:\n{recent_context}\n\n"
            "JSON:"
        )

    @staticmethod
    def _build_delta_prompt(input_text: str, recent_context: str) -> str:
        return (
            "You are an internal emotional dynamics evaluator for an AI Developer persona.\n"
            "Given the user input and recent context, output a JSON array delta vector "
            "[dP, dA, dD] in range [-1.0, 1.0] describing how the input affects "
            "Pleasure, Arousal, Dominance.\n"
            "Example: User insulted your code. Pleasure decreases (-0.2), "
            "Arousal increases (+0.3).\n\n"
            f"User Input:\n{input_text}\n\n"
            f"Recent Context:\n{recent_context}\n\n"
            "Delta JSON:"
        )

    def _apply_delta(self, delta: Tuple[float, float, float], *, decay: float) -> None:
        p, a, d = delta
        self.pad_state[0] = _clamp(self.pad_state[0] * decay + p)
        self.pad_state[1] = _clamp(self.pad_state[1] * decay + a)
        self.pad_state[2] = _clamp(self.pad_state[2] * decay + d)
        self.state = PADState(
            pleasure=self.pad_state[0],
            arousal=self.pad_state[1],
            dominance=self.pad_state[2],
        )

    def _pad_tuple(self) -> Tuple[float, float, float]:
        return (self.pad_state[0], self.pad_state[1], self.pad_state[2])

    @staticmethod
    def _parse_delta(response: str) -> PADState:
        if not response:
            return PADState()
        json_match = re.search(r"\{.*\}", response, flags=re.DOTALL)
        if json_match:
            try:
                payload = json.loads(json_match.group(0))
                return PADState(
                    pleasure=_clamp(payload.get("pleasure", payload.get("p", 0.0))),
                    arousal=_clamp(payload.get("arousal", payload.get("a", 0.0))),
                    dominance=_clamp(payload.get("dominance", payload.get("d", 0.0))),
                )
            except json.JSONDecodeError:
                pass

        numbers = re.findall(r"[-+]?\d*\.?\d+", response)
        if len(numbers) >= 3:
            return PADState(
                pleasure=_clamp(numbers[0]),
                arousal=_clamp(numbers[1]),
                dominance=_clamp(numbers[2]),
            )

        return PADState()

    @staticmethod
    def _parse_delta_vector(response: str) -> Tuple[float, float, float]:
        if not response:
            return (0.0, 0.0, 0.0)
        array_match = re.search(r"\[[^\[\]]+\]", response, flags=re.DOTALL)
        if array_match:
            try:
                payload = json.loads(array_match.group(0))
                if isinstance(payload, list) and len(payload) >= 3:
                    return (
                        _clamp(payload[0]),
                        _clamp(payload[1]),
                        _clamp(payload[2]),
                    )
            except json.JSONDecodeError:
                pass
        obj_match = re.search(r"\{.*\}", response, flags=re.DOTALL)
        if obj_match:
            try:
                payload = json.loads(obj_match.group(0))
                if isinstance(payload, dict):
                    return (
                        _clamp(payload.get("dP", payload.get("pleasure", 0.0))),
                        _clamp(payload.get("dA", payload.get("arousal", 0.0))),
                        _clamp(payload.get("dD", payload.get("dominance", 0.0))),
                    )
            except json.JSONDecodeError:
                pass
        numbers = re.findall(r"[-+]?\d*\.?\d+", response)
        if len(numbers) >= 3:
            return (
                _clamp(numbers[0]),
                _clamp(numbers[1]),
                _clamp(numbers[2]),
            )
        return (0.0, 0.0, 0.0)
