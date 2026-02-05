"""Active inquiry module to ask clarifying questions under high uncertainty."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Callable, Optional, Sequence

try:
    import torch
    from torch.nn import functional as F
except Exception:  # pragma: no cover - optional for lightweight usage
    torch = None
    F = None


@dataclass
class PerplexityResult:
    perplexity: float
    uncertainty: float
    token_count: int
    source: str = "logprobs"


class PerplexityEstimator:
    """Compute perplexity and normalized uncertainty from token logprobs."""

    def __init__(self, *, perplexity_scale: float = 5.0) -> None:
        self.perplexity_scale = perplexity_scale

    def from_logprobs(self, token_logprobs: Sequence[float]) -> PerplexityResult:
        if not token_logprobs:
            return PerplexityResult(perplexity=1.0, uncertainty=0.0, token_count=0)
        mean_nll = -sum(float(lp) for lp in token_logprobs) / len(token_logprobs)
        perplexity = math.exp(min(20.0, mean_nll))
        uncertainty = 1.0 - math.exp(-perplexity / self.perplexity_scale)
        return PerplexityResult(
            perplexity=perplexity,
            uncertainty=uncertainty,
            token_count=len(token_logprobs),
        )

    def from_logits(self, logits, target_ids) -> PerplexityResult:
        if torch is None or F is None:
            raise RuntimeError("torch is required to compute perplexity from logits")
        if logits.ndim != 3:
            raise ValueError("logits must be [batch, seq, vocab]")
        if target_ids.ndim != 2:
            raise ValueError("target_ids must be [batch, seq]")
        log_probs = F.log_softmax(logits, dim=-1)
        gathered = log_probs.gather(-1, target_ids.unsqueeze(-1)).squeeze(-1)
        token_logprobs = gathered.flatten().tolist()
        return self.from_logprobs(token_logprobs)


@dataclass
class CuriosityDecision:
    should_ask: bool
    question: Optional[str]
    perplexity: float
    uncertainty: float
    source: str


class QuestionGenerator:
    """Generate a clarifying question to reduce ambiguity."""

    def __init__(
        self,
        *,
        llm_fn: Optional[Callable[[str], str]] = None,
        fallback_question: str = "Bạn có thể làm rõ mục tiêu hoặc ràng buộc chính không?",
    ) -> None:
        self.llm_fn = llm_fn
        self.fallback_question = fallback_question

    def generate(self, draft: str, context: Optional[str] = None) -> str:
        if self.llm_fn is None:
            return self.fallback_question
        prompt = self._build_prompt(draft, context)
        response = self.llm_fn(prompt)
        return response.strip() if response else self.fallback_question

    @staticmethod
    def _build_prompt(draft: str, context: Optional[str]) -> str:
        return (
            "You are a clarification agent.\n"
            "Given the draft answer and context, ask one concise question that "
            "would resolve the main ambiguity.\n\n"
            f"Draft:\n{draft}\n\n"
            f"Context:\n{context or ''}\n\n"
            "Question:"
        )


class CuriosityGate:
    """Gate to stop generation and ask a question under high uncertainty."""

    def __init__(
        self,
        *,
        threshold: float = 0.7,
        perplexity_scale: float = 5.0,
        question_generator: Optional[QuestionGenerator] = None,
    ) -> None:
        self.threshold = threshold
        self.estimator = PerplexityEstimator(perplexity_scale=perplexity_scale)
        self.question_generator = question_generator or QuestionGenerator()

    def evaluate(
        self,
        draft: str,
        *,
        token_logprobs: Optional[Sequence[float]] = None,
        explicit_uncertainty: Optional[float] = None,
        context: Optional[str] = None,
    ) -> CuriosityDecision:
        if explicit_uncertainty is not None:
            uncertainty = float(explicit_uncertainty)
            perplexity = max(1.0, uncertainty * 10.0)
            source = "explicit"
        elif token_logprobs is not None:
            result = self.estimator.from_logprobs(token_logprobs)
            uncertainty = result.uncertainty
            perplexity = result.perplexity
            source = result.source
        else:
            uncertainty = 0.0
            perplexity = 1.0
            source = "fallback"

        should_ask = uncertainty > self.threshold
        question = None
        if should_ask:
            question = self.question_generator.generate(draft, context=context)

        return CuriosityDecision(
            should_ask=should_ask,
            question=question,
            perplexity=perplexity,
            uncertainty=uncertainty,
            source=source,
        )

    def check(
        self,
        user_input: str,
        context: Optional[str] = None,
        *,
        draft: Optional[str] = None,
        token_logprobs: Optional[Sequence[float]] = None,
        explicit_uncertainty: Optional[float] = None,
    ) -> CuriosityDecision:
        """Compatibility wrapper for agent loops."""
        draft_text = draft or user_input
        return self.evaluate(
            draft_text,
            token_logprobs=token_logprobs,
            explicit_uncertainty=explicit_uncertainty,
            context=context,
        )
