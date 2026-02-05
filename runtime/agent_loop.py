"""Cognitive agent loop for vAGI."""

from __future__ import annotations

import argparse
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
import json
import re
from typing import Callable, List, Optional, Protocol, Sequence

import torch

from core.cognition import EmotionEngine, PADState
from core.memory.generative_memory import (
    MemoryObject,
    MemoryStream,
    ReflectionLoop,
    ReflectionLoopConfig,
)
from core.memory.reflexion import ReflexionManager
from core.reasoning.curiosity import CuriosityDecision, CuriosityGate, QuestionGenerator
from envs.toy_env import ToyEnv
from runtime.logging import JsonlWriter
from runtime.privacy import apply_retention, delete_logs
from scripts.utils import set_deterministic
from vagi_core import VAGIConfig, VAGICore


QuickInferFn = Callable[[str], str]
GenerateFn = Callable[[str], str]


class ModelEngine(Protocol):
    def quick_infer(self, prompt: str) -> str:
        ...

    def generate(self, prompt: str) -> str:
        ...

    def deep_infer(self, prompt: str) -> str:
        ...


@dataclass
class CognitiveAgentConfig:
    retrieval_top_k: int = 5
    reflection_min_memories: int = 5
    system_prompt_template: str = (
        "You are a cognitive agent.\n"
        "Current PAD Mood: {pad}\n"
        "Tone Instruction: {tone}\n"
        "Retrieved Memories:\n{memories}\n"
    )
    draft_prompt_template: str = (
        "Draft a concise response based on the user input and context.\n\n"
        "User Input:\n{user_input}\n\n"
        "Context:\n{context}\n\n"
        "Draft:"
    )
    uncertainty_prompt_template: str = (
        "Estimate uncertainty for answering the user input with given context.\n"
        "Return a single float between 0.0 and 1.0.\n\n"
        "User Input:\n{user_input}\n\n"
        "Context:\n{context}\n\n"
        "Draft:\n{draft}\n\n"
        "Uncertainty:"
    )
    importance_prompt_template: str = (
        "Rate the importance of this interaction for long-term memory.\n"
        "Return a single float between 0.0 and 1.0.\n\n"
        "User Input:\n{user_input}\n\n"
        "Agent Response:\n{response}\n\n"
        "Importance:"
    )


@dataclass
class CognitiveActResult:
    text: str
    is_question: bool
    mood: PADState
    memories: List[MemoryObject] = field(default_factory=list)
    prompt: str = ""
    curiosity: Optional[CuriosityDecision] = None


class CognitiveAgent:
    def __init__(
        self,
        model_engine: ModelEngine,
        *,
        memory_store_path: Optional[Path] = None,
        memory_stream: Optional[MemoryStream] = None,
        reflection_config: Optional[ReflectionLoopConfig] = None,
        emotion_engine: Optional[EmotionEngine] = None,
        curiosity_gate: Optional[CuriosityGate] = None,
        quick_infer_fn: Optional[QuickInferFn] = None,
        generate_fn: Optional[GenerateFn] = None,
        deep_infer_fn: Optional[QuickInferFn] = None,
        executor: Optional[ThreadPoolExecutor] = None,
        config: Optional[CognitiveAgentConfig] = None,
    ) -> None:
        self.model_engine = model_engine
        self.memory_store_path = memory_store_path
        self.quick_infer: QuickInferFn = quick_infer_fn or model_engine.quick_infer
        self.generate_fn: GenerateFn = generate_fn or model_engine.generate
        self.deep_infer: QuickInferFn = deep_infer_fn or getattr(
            model_engine,
            "deep_infer",
            self.generate_fn,
        )
        self.generative_memory = memory_stream or MemoryStream()
        self.emotion_engine = emotion_engine or EmotionEngine(llm_fn=self.quick_infer)
        question_generator = QuestionGenerator(llm_fn=self.quick_infer)
        self.curiosity_gate = curiosity_gate or CuriosityGate(question_generator=question_generator)
        self.config = config or CognitiveAgentConfig()
        self._executor = executor or ThreadPoolExecutor(max_workers=2)
        self._pending_futures: List[Future[Optional[MemoryObject]]] = []
        self._recent_context: str = ""
        self._insights_context: str = ""
        self._insights_path = self._resolve_insights_path(memory_store_path)
        self.reflexion = ReflexionManager(memory_store=self.generative_memory)
        self._reflection_loop = ReflectionLoop(
            self.generative_memory,
            llm_fn=self.quick_infer,
            config=reflection_config or ReflectionLoopConfig(
                min_memories=self.config.reflection_min_memories,
            ),
        )

    def act(self, user_input: str) -> CognitiveActResult:
        # Step 1: Perception (Emotion Update)
        self.emotion_engine.update_state(
            user_input,
            self.quick_infer,
            recent_context=self._recent_context,
        )

        # Step 2: Retrieval (Context)
        memories = self.generative_memory.retrieve(
            user_input,
            top_k=self.config.retrieval_top_k,
        )
        context_text = MemoryStream.format_memories(memories)

        # Step 3: Curiosity Check
        draft = self.quick_infer(
            self.config.draft_prompt_template.format(
                user_input=user_input,
                context=context_text,
            )
        )
        curiosity = self._curiosity_check(user_input, context_text, draft)
        if curiosity.should_ask:
            self._schedule_memory_update(user_input, curiosity.question or "")
            return CognitiveActResult(
                text=curiosity.question or "",
                is_question=True,
                mood=self.emotion_engine.state,
                memories=memories,
                prompt="",
                curiosity=curiosity,
            )

        # Step 4: Prompt Construction
        system_prompt = self._build_system_prompt(memories)
        full_prompt = f"{system_prompt}\nUser: {user_input}\nAssistant:"

        # Step 5: Reasoning & Generation
        response = self.generate_fn(full_prompt)

        # Step 6: Reflection (async save + reflect)
        self._schedule_memory_update(user_input, response)
        self.reflexion.add_turn(user_input, response)
        if self.reflexion.should_reflect():
            # TODO: move reflection into async background processing.
            insights = self.reflexion.reflect(llm_fn=self.deep_infer)
            for insight in insights:
                print(f"[System] Insight derived: {insight.content}")
            self._persist_insights(insights)
            self._insights_context = self._format_insights(insights)
        self._recent_context = context_text

        return CognitiveActResult(
            text=response,
            is_question=False,
            mood=self.emotion_engine.state,
            memories=memories,
            prompt=full_prompt,
            curiosity=curiosity,
        )

    def _curiosity_check(self, user_input: str, context: str, draft: str) -> CuriosityDecision:
        if not context.strip():
            question = self.curiosity_gate.question_generator.generate(draft, context=context)
            return CuriosityDecision(
                should_ask=True,
                question=question,
                perplexity=1.0,
                uncertainty=1.0,
                source="missing_context",
            )
        uncertainty = self._estimate_uncertainty(user_input, context, draft)
        return self.curiosity_gate.check(
            user_input,
            context,
            draft=draft,
            explicit_uncertainty=uncertainty,
        )

    def _build_system_prompt(self, memories: Sequence[MemoryObject]) -> str:
        pad = self.emotion_engine.state
        pad_str = f"[{pad.pleasure:.2f}, {pad.arousal:.2f}, {pad.dominance:.2f}]"
        tone = self._tone_instruction(self.emotion_engine.current_mood_label())
        memories_block = MemoryStream.format_memories(memories) or "- (none)"
        if self._insights_context:
            memories_block = f"{memories_block}\n\n{self._insights_context}"
        return self.config.system_prompt_template.format(
            pad=pad_str,
            tone=tone,
            memories=memories_block,
        )

    def _tone_instruction(self, mood_label: str) -> str:
        label = mood_label.strip().lower()
        if label == "frustrated":
            return "Be frustrated but helpful."
        if label == "excited":
            return "Be excited and helpful."
        if label == "calm":
            return "Be calm and helpful."
        if label == "sad":
            return "Be gentle and supportive."
        if label == "confident":
            return "Be confident and helpful."
        if label == "anxious":
            return "Be cautious but helpful."
        return "Be neutral and helpful."

    def _estimate_uncertainty(self, user_input: str, context: str, draft: str) -> float:
        prompt = self.config.uncertainty_prompt_template.format(
            user_input=user_input,
            context=context,
            draft=draft,
        )
        response = self.quick_infer(prompt)
        return _parse_scalar(response, default=0.0)

    def _estimate_importance(self, user_input: str, response: str) -> float:
        prompt = self.config.importance_prompt_template.format(
            user_input=user_input,
            response=response,
        )
        estimate = self.quick_infer(prompt)
        return _parse_scalar(estimate, default=0.5)

    def _schedule_memory_update(self, user_input: str, response: str) -> None:
        future = self._executor.submit(self._save_and_reflect, user_input, response)
        self._pending_futures.append(future)

    def _save_and_reflect(self, user_input: str, response: str) -> Optional[MemoryObject]:
        importance = self._estimate_importance(user_input, response)
        memory = self.generative_memory.add_memory(
            content=f"User: {user_input}\nAssistant: {response}",
            importance_score=importance,
        )
        if len(self.generative_memory.memories) >= self.config.reflection_min_memories:
            self._reflection_loop.maybe_reflect(step=len(self.generative_memory.memories))
        return memory

    def _resolve_insights_path(self, memory_store_path: Optional[Path]) -> Optional[Path]:
        if memory_store_path is None:
            return Path("insights.json")
        if memory_store_path.suffix.lower() == ".json":
            return memory_store_path
        return memory_store_path / "insights.json"

    def _persist_insights(self, insights: Sequence[MemoryObject]) -> None:
        if not insights or self._insights_path is None:
            return
        payload = []
        if self._insights_path.exists():
            try:
                payload = json.loads(self._insights_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                payload = []
        for insight in insights:
            payload.append(
                {
                    "timestamp": insight.timestamp.isoformat(),
                    "content": insight.content,
                    "importance_score": insight.importance_score,
                }
            )
        self._insights_path.parent.mkdir(parents=True, exist_ok=True)
        self._insights_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    @staticmethod
    def _format_insights(insights: Sequence[MemoryObject]) -> str:
        if not insights:
            return ""
        lines = ["[Insights]"]
        for insight in insights:
            lines.append(f"- {insight.content}")
        return "\n".join(lines)


def _parse_scalar(text: str, *, default: float) -> float:
    match = re.search(r"[-+]?\d*\.?\d+", text)
    if not match:
        return default
    try:
        value = float(match.group(0))
    except ValueError:
        return default
    return max(0.0, min(1.0, value))


def run_episode(
    model: VAGICore,
    env: ToyEnv,
    steps: int,
    log_path: Optional[str] = None,
    privacy_opt_in: bool = False,
    memory_stream: Optional[MemoryStream] = None,
    reflection_loop: Optional[ReflectionLoop] = None,
) -> int:
    model.eval()
    obs = env.reset()
    state = model.init_state(batch_size=1)
    token_id = 0
    writer = JsonlWriter(log_path, scrub_pii=True, privacy_opt_in=privacy_opt_in) if log_path else None

    try:
        for t in range(steps):
            input_ids = torch.tensor([[token_id]], dtype=torch.long)
            out = model.step(input_ids=input_ids, obs=obs.unsqueeze(0), state=state)
            action = int(torch.argmax(out["action_logits"], dim=-1).item())
            value = float(out["value"].item())
            uncertainty = float(out["uncertainty"].mean().item()) if out.get("uncertainty") is not None else None
            validity = None
            if out.get("action_valid") is not None:
                validity = float(out["action_valid"].squeeze(0)[action].item())
            next_obs, reward, done, info = env.step(action)

            if writer is not None:
                writer.write(
                    {
                        "timestep": t,
                        "obs": obs.tolist(),
                        "action": action,
                        "reward": float(reward),
                        "value": value,
                        "uncertainty": uncertainty,
                        "validity": validity,
                    }
                )

            if memory_stream is not None:
                importance = min(1.0, max(0.0, abs(float(reward))))
                memory_stream.add_memory(
                    content=(
                        f"t={t} obs={obs.tolist()} action={action} "
                        f"reward={float(reward):.4f}"
                    ),
                    importance_score=importance,
                    related_nodes=[f"action:{action}"],
                )

            if reflection_loop is not None:
                insight = reflection_loop.maybe_reflect(step=t)
                if insight is not None and writer is not None:
                    writer.write(
                        {
                            "timestep": t,
                            "reflection": insight.content,
                            "reflection_importance": insight.importance_score,
                        }
                    )

            state = out["state"]
            obs = next_obs
            token_id = action
            if done:
                return t + 1
    finally:
        if writer is not None:
            writer.close()

    return steps


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a minimal vAGI agent loop.")
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--log", type=str, default="runs/agent/transitions.jsonl")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--obs-dim", type=int, default=8)
    parser.add_argument("--action-dim", type=int, default=4)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--privacy-opt-in", action="store_true")
    parser.add_argument("--retain-days", type=int, default=7)
    parser.add_argument("--delete-logs", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_deterministic(args.seed, args.deterministic)

    env = ToyEnv(obs_dim=args.obs_dim, action_dim=args.action_dim, max_steps=args.steps, seed=args.seed)
    cfg = VAGIConfig(
        vocab_size=32,
        hidden_size=32,
        n_layers=1,
        n_heads=4,
        n_kv_heads=4,
        mlp_ratio=2.0,
        max_seq_len=8,
        obs_dim=args.obs_dim,
        obs_tokens=1,
        action_dim=args.action_dim,
        memory_slots=2,
        dropout=0.0,
        use_world_pred=False,
    )
    model = VAGICore(cfg)

    log_path = Path(args.log)
    if args.delete_logs:
        delete_logs(log_path.parent)
    apply_retention(log_path.parent, args.retain_days)
    steps = run_episode(
        model,
        env,
        steps=args.steps,
        log_path=args.log,
        privacy_opt_in=args.privacy_opt_in,
    )
    print(f"Completed {steps} steps. Logs at {args.log}")


if __name__ == "__main__":
    main()
