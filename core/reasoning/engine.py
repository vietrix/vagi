"""Cogito Engine: System 2 controller for deliberate reasoning."""

from __future__ import annotations

import inspect
import re
import threading
from dataclasses import dataclass
from typing import Generator, Iterable, Optional

import torch

try:
    from transformers import TextIteratorStreamer
except ImportError:  # pragma: no cover - optional dependency
    TextIteratorStreamer = None


THINK_TAG_PATTERN = re.compile(r"<think>(.*?)</think>", re.DOTALL | re.IGNORECASE)
UNCERTAIN_PATTERN = re.compile(
    r"\b(uncertain|maybe|not sure|unsure|error|confused|mismatch|doubt)\b"
    r"|"
    r"\b(không chắc|có thể|sai|lỗi|mơ hồ|nghi ngờ)\b",
    re.IGNORECASE,
)


@dataclass
class CogitoResult:
    thought_process: str
    final_answer: str
    iterations: int
    was_uncertain: bool


class CogitoEngine:
    """System 2 thinking controller for vAGI."""

    def __init__(self, model: object, tokenizer: object):
        self.model = model
        self.tokenizer = tokenizer
        self.device = getattr(model, "device", torch.device("cpu"))

    def think(self, query: str, max_depth: int = 3) -> dict:
        if max_depth <= 0:
            raise ValueError("max_depth must be > 0")
        thought = ""
        iterations = 0
        was_uncertain = False

        for depth in range(max_depth):
            iterations += 1
            raw_thought = self._generate_thought(query, thought if depth > 0 else None)
            extracted_thought, _ = self._extract_think(raw_thought)
            thought = extracted_thought or raw_thought.strip()
            if self._is_uncertain(thought):
                was_uncertain = True
                if depth < max_depth - 1:
                    continue
            break

        final_raw = self._generate_answer(query, thought)
        final_thought, final_answer = self._extract_think(final_raw)
        if not final_thought:
            final_thought = thought
            final_answer = final_raw.strip()

        result = CogitoResult(
            thought_process=final_thought.strip(),
            final_answer=final_answer.strip(),
            iterations=iterations,
            was_uncertain=was_uncertain,
        )
        return result.__dict__

    def stream_thought(self, query: str, max_depth: int = 3) -> Generator[dict, None, None]:
        if max_depth <= 0:
            raise ValueError("max_depth must be > 0")
        thought = ""
        was_uncertain = False

        for depth in range(max_depth):
            prompt = self._build_thought_prompt(query, thought if depth > 0 else None)
            chunks = []
            for chunk in self._stream_generate(prompt, max_new_tokens=512):
                chunks.append(chunk)
                yield {"type": "thinking", "content": chunk}
            raw_thought = "".join(chunks)
            extracted_thought, _ = self._extract_think(raw_thought)
            thought = extracted_thought or raw_thought.strip()
            if self._is_uncertain(thought):
                was_uncertain = True
                if depth < max_depth - 1:
                    continue
            break

        answer_prompt = self._build_answer_prompt(query, thought)
        for chunk in self._stream_generate(answer_prompt, max_new_tokens=512):
            yield {"type": "answer", "content": chunk}

        if was_uncertain:
            return

    def _build_thought_prompt(self, query: str, previous_thought: Optional[str]) -> str:
        system = (
            "Bạn là Cogito Engine của vAGI. Luôn suy nghĩ kỹ trước khi trả lời."
        )
        if previous_thought:
            instruction = (
                "Trước đó bạn không chắc chắn. Hãy thử một cách tiếp cận khác, "
                "không lặp lại ý cũ. Chỉ xuất ra <think>...</think> và KHÔNG trả lời cuối.\n\n"
                f"Câu hỏi: {query}\n"
                f"Suy nghĩ trước: {previous_thought}"
            )
        else:
            instruction = (
                "Hãy suy nghĩ trong thẻ <think>...</think> và KHÔNG trả lời cuối.\n\n"
                f"Câu hỏi: {query}"
            )
        return self._format_prompt(system, instruction)

    def _build_answer_prompt(self, query: str, thought: str) -> str:
        system = "Bạn là Cogito Engine của vAGI. Trả lời rõ ràng, nhất quán."
        instruction = (
            "Dựa trên suy nghĩ đã chọn, hãy trả lời theo định dạng:\n"
            "<think>...</think>\n"
            "Final Answer: ...\n\n"
            f"Câu hỏi: {query}\n"
            f"Suy nghĩ đã chọn:\n<think>{thought}</think>"
        )
        return self._format_prompt(system, instruction)

    def _format_prompt(self, system: str, instruction: str) -> str:
        if hasattr(self.tokenizer, "apply_chat_template"):
            messages = [
                {"role": "system", "content": system},
                {"role": "user", "content": instruction},
            ]
            return self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        return f"{system}\n\nUser: {instruction}\nAssistant:"

    def _generate_thought(self, query: str, previous_thought: Optional[str]) -> str:
        prompt = self._build_thought_prompt(query, previous_thought)
        return self._generate(prompt, max_new_tokens=512)

    def _generate_answer(self, query: str, thought: str) -> str:
        prompt = self._build_answer_prompt(query, thought)
        return self._generate(prompt, max_new_tokens=512)

    def _generate(self, prompt: str, max_new_tokens: int) -> str:
        input_ids = self._encode_prompt(prompt)
        generate_fn = getattr(self.model, "generate", None)
        if generate_fn is None:
            raise AttributeError("Model does not implement generate()")
        kwargs = self._filter_generate_kwargs(generate_fn, max_new_tokens=max_new_tokens)
        output_ids = generate_fn(input_ids=input_ids, **kwargs)
        text = self._decode_output(output_ids, prompt)
        return text.strip()

    def _stream_generate(self, prompt: str, max_new_tokens: int) -> Iterable[str]:
        generate_fn = getattr(self.model, "generate", None)
        if generate_fn is None:
            raise AttributeError("Model does not implement generate()")

        if TextIteratorStreamer is not None and self._supports_streamer(generate_fn):
            streamer = TextIteratorStreamer(
                self.tokenizer, skip_prompt=True, skip_special_tokens=True
            )
            input_ids = self._encode_prompt(prompt)
            kwargs = self._filter_generate_kwargs(
                generate_fn, max_new_tokens=max_new_tokens, streamer=streamer
            )
            thread = threading.Thread(
                target=generate_fn, kwargs={"input_ids": input_ids, **kwargs}, daemon=True
            )
            thread.start()
            for text in streamer:
                yield text
            thread.join()
        else:
            text = self._generate(prompt, max_new_tokens=max_new_tokens)
            for chunk in self._chunk_text(text, 64):
                yield chunk

    def _encode_prompt(self, prompt: str) -> torch.Tensor:
        if hasattr(self.tokenizer, "__call__"):
            encoded = self.tokenizer(prompt, return_tensors="pt")
            input_ids = encoded["input_ids"]
        else:
            raise AttributeError("Tokenizer does not support call()")
        return input_ids.to(self.device)

    def _decode_output(self, output_ids: torch.Tensor, prompt: str) -> str:
        if hasattr(self.tokenizer, "decode"):
            text = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
        else:
            raise AttributeError("Tokenizer does not implement decode()")
        if text.startswith(prompt):
            text = text[len(prompt):]
        return text

    def _filter_generate_kwargs(self, generate_fn, **kwargs: object) -> dict:
        sig = inspect.signature(generate_fn)
        allowed = set(sig.parameters.keys())
        filtered = {}
        for key, value in kwargs.items():
            if key in allowed:
                filtered[key] = value
        if "do_sample" in allowed and "do_sample" not in filtered:
            filtered["do_sample"] = False
        if "temperature" in allowed and "temperature" not in filtered:
            filtered["temperature"] = 0.0
        return filtered

    def _supports_streamer(self, generate_fn) -> bool:
        sig = inspect.signature(generate_fn)
        return "streamer" in sig.parameters

    def _extract_think(self, text: str) -> tuple[str, str]:
        match = THINK_TAG_PATTERN.search(text)
        if not match:
            return "", text.strip()
        thought = match.group(1).strip()
        remainder = text[match.end():].strip()
        return thought, remainder

    def _is_uncertain(self, thought: str) -> bool:
        return bool(UNCERTAIN_PATTERN.search(thought))

    @staticmethod
    def _chunk_text(text: str, size: int) -> Iterable[str]:
        if size <= 0:
            yield text
            return
        for idx in range(0, len(text), size):
            yield text[idx:idx + size]
