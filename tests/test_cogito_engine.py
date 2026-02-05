from __future__ import annotations

import torch

from core.reasoning.engine import CogitoEngine


class DummyTokenizer:
    def __init__(self) -> None:
        self.pad_token_id = 0
        self.eos_token_id = 0

    def __call__(self, text: str, return_tensors: str = "pt") -> dict:
        ids = [ord(ch) % 255 + 1 for ch in text]
        return {"input_ids": torch.tensor([ids], dtype=torch.long)}

    def decode(self, ids, skip_special_tokens: bool = True) -> str:
        return "".join(chr((i - 1) % 255) for i in ids if i != 0)


class DummyModel:
    def __init__(self, outputs: list[str]) -> None:
        self.outputs = outputs
        self.calls = 0
        self.device = torch.device("cpu")

    def generate(self, input_ids=None, **kwargs):
        output = self.outputs[self.calls]
        self.calls += 1
        ids = [ord(ch) % 255 + 1 for ch in output]
        return torch.tensor([ids], dtype=torch.long)


def test_cogito_branching_on_uncertainty() -> None:
    outputs = [
        "<think>maybe error</think>",
        "<think>clear approach</think>",
        "<think>clear approach</think> Final Answer: ok",
    ]
    engine = CogitoEngine(DummyModel(outputs), DummyTokenizer())
    result = engine.think("Test?", max_depth=2)
    assert result["iterations"] == 2
    assert result["was_uncertain"] is True
    assert "clear approach" in result["thought_process"]
    assert "Final Answer" in result["final_answer"]


def test_cogito_parse_fallback() -> None:
    outputs = ["no tags here", "Final Answer: done"]
    engine = CogitoEngine(DummyModel(outputs), DummyTokenizer())
    result = engine.think("Test?", max_depth=1)
    assert result["thought_process"] == "no tags here"
    assert "Final Answer" in result["final_answer"]


def test_stream_thought_emits_thinking_and_answer() -> None:
    outputs = [
        "<think>plan</think>",
        "<think>plan</think> Final Answer: ok",
    ]
    engine = CogitoEngine(DummyModel(outputs), DummyTokenizer())
    events = list(engine.stream_thought("Test?", max_depth=1))
    assert events
    assert events[0]["type"] == "thinking"
    assert events[-1]["type"] == "answer"
