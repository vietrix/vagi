from __future__ import annotations

import asyncio
import hashlib
from dataclasses import dataclass
from typing import Any

from .kernel_client import KernelClient
from .memory import MemoryClient, MemoryHit
from .store import EpisodeStore


def compute_trust_score(
    source: str, verifier_pass: bool, risk_score: float, confidence: float
) -> float:
    source_base = {
        "trusted": 0.85,
        "manual": 0.75,
        "chat": 0.65,
        "unknown": 0.55,
    }.get(source, 0.55)
    score = source_base
    score += 0.2 if verifier_pass else -0.25
    score += (confidence - 0.5) * 0.2
    score -= risk_score * 0.25
    return max(0.0, min(1.0, score))


def build_session_id(seed: str) -> str:
    digest = hashlib.sha256(seed.encode("utf-8")).hexdigest()
    return f"sid-{digest[:16]}"


FALLBACK_RESPONSE = "Insufficient data in kernel state."


@dataclass(slots=True)
class Reasoner:
    kernel: KernelClient
    store: EpisodeStore
    max_decide_iters: int = 12
    risk_threshold: float = 0.65
    memory_client: MemoryClient | None = None
    memory_top_k: int = 3
    memory_min_score: float = 0.2

    async def run_chat(
        self, *, session_id: str, messages: list[dict[str, str]]
    ) -> dict[str, Any]:
        prompt = "\n".join(
            f"{msg.get('role', 'user')}: {msg.get('content', '')}"
            for msg in messages
            if msg.get("content")
        )
        if not prompt.strip():
            raise ValueError("messages do not contain valid content")
        user_input = self._extract_user_input(messages=messages, fallback_prompt=prompt)

        # OBSERVE
        retrieval_ctx = await self._retrieve_memory(user_input=user_input)
        infer_prompt = self._build_infer_prompt(
            prompt=prompt,
            retrieved_hits=retrieval_ctx["scored_hits"],
        )
        observe_ctx = self._observe(
            messages=messages,
            prompt=prompt,
            retrieval_ctx=retrieval_ctx,
        )
        await self.kernel.init_state(session_id=session_id)
        await self.kernel.update_state(session_id=session_id, input_text=prompt)
        model_runtime_meta, model_seed = await self._fast_system_seed(infer_prompt)
        force_insufficient_response = (
            model_runtime_meta.get("fallback_reason") == "garbage_detected"
        )

        # ORIENT
        orient_ctx = await self._orient(
            prompt=prompt,
            observe_ctx=observe_ctx,
            session_id=session_id,
            model_seed=model_seed,
            mcts_meta=model_runtime_meta.get("mcts_meta"),
        )
        draft = orient_ctx["selected_candidate"]["draft"]

        ooda_trace: dict[str, Any] = {
            "observe_ok": True,
            "orient_ok": True,
            "decide_ok": False,
            "act_ok": False,
            "iterations": 0,
            "candidate_count": len(orient_ctx["candidates"]),
        }
        final_sim: dict[str, Any] | None = None
        final_verifier: dict[str, Any] | None = None

        # DECIDE
        for attempt in range(1, self.max_decide_iters + 1):
            sim = await self.kernel.simulate_world(draft, session_id=session_id)
            verifier = await self.kernel.verify(
                patch_ir=draft,
                max_loop_iters=2048,
                side_effect_budget=3,
                timeout_ms=80,
            )
            final_sim = sim
            final_verifier = verifier
            ooda_trace["iterations"] = attempt
            if verifier.get("pass", False) and sim.get("risk_score", 1.0) <= self.risk_threshold:
                ooda_trace["decide_ok"] = True
                break
            draft = self._revise_draft(draft, sim=sim, verifier=verifier, attempt=attempt)

        assert final_sim is not None
        assert final_verifier is not None

        # ACT
        ooda_trace["act_ok"] = bool(ooda_trace["decide_ok"])
        act_response = self._act(
            draft=draft,
            observe_ctx=observe_ctx,
            orient_ctx=orient_ctx,
            sim=final_sim,
            verifier=final_verifier,
            ooda_trace=ooda_trace,
            model_runtime_meta=model_runtime_meta,
            force_insufficient_response=force_insufficient_response,
        )
        trust_score = compute_trust_score(
            source="chat",
            verifier_pass=bool(final_verifier.get("pass", False)),
            risk_score=float(final_sim.get("risk_score", 1.0)),
            confidence=float(final_sim.get("confidence", 0.0)),
        )
        episode_id = self.store.record_episode(
            session_id=session_id,
            user_input=prompt,
            draft=draft,
            verifier_pass=bool(final_verifier.get("pass", False)),
            risk_score=float(final_sim.get("risk_score", 1.0)),
            trust_score=trust_score,
            violations=list(final_verifier.get("violations", [])),
            source="chat",
            ooda_trace=ooda_trace,
            verifier_required=True,
            verifier_pass_gate=bool(final_verifier.get("pass", False)),
        )
        act_response["episode_id"] = episode_id
        act_response["trust_score"] = trust_score
        return act_response

    def _observe(
        self,
        *,
        messages: list[dict[str, str]],
        prompt: str,
        retrieval_ctx: dict[str, Any],
    ) -> dict[str, Any]:
        token_count = sum(len(msg.get("content", "").split()) for msg in messages)
        max_line_len = max((len(line) for line in prompt.splitlines()), default=0)
        return {
            "intent": "engineering_request",
            "constraints": [
                "require verifier pass",
                "respect risk threshold",
                "avoid unsafe execution primitives",
            ],
            "security_flags": {
                "contains_eval": "eval(" in prompt.lower() or "exec(" in prompt.lower(),
                "contains_delete": "drop" in prompt.lower() or "rm -rf" in prompt.lower(),
            },
            "hardware_limits": {
                "profile": "cpu-first",
                "token_count": token_count,
                "max_line_length": max_line_len,
            },
            "retrieval": {
                "enabled": self.memory_client is not None,
                "used": bool(retrieval_ctx.get("used", False)),
                "hits_count": len(retrieval_ctx.get("hits", [])),
                "hits": retrieval_ctx.get("hits", []),
                "scored_hits": retrieval_ctx.get("scored_hits", []),
                "error": retrieval_ctx.get("error"),
            },
        }

    def _extract_user_input(
        self, *, messages: list[dict[str, str]], fallback_prompt: str
    ) -> str:
        for msg in reversed(messages):
            role = str(msg.get("role", "")).lower()
            content = str(msg.get("content", "")).strip()
            if role == "user" and content:
                return content
        return fallback_prompt.strip()

    async def _retrieve_memory(self, *, user_input: str) -> dict[str, Any]:
        if self.memory_client is None:
            return {"used": False, "hits": [], "scored_hits": [], "error": None}

        top_k = max(1, int(self.memory_top_k))
        try:
            hits = await asyncio.to_thread(self.memory_client.retrieve_hits, user_input, top_k)
            filtered_hits = [
                hit for hit in hits if isinstance(hit, MemoryHit) and hit.score >= self.memory_min_score
            ]
            normalized_hits = [
                hit.text.strip()
                for hit in filtered_hits
                if isinstance(hit.text, str) and hit.text.strip()
            ]
            scored_hits = [
                {"text": hit.text.strip(), "score": round(float(hit.score), 6)}
                for hit in filtered_hits
                if isinstance(hit.text, str) and hit.text.strip()
            ]
            return {
                "used": bool(normalized_hits),
                "hits": normalized_hits[:top_k],
                "scored_hits": scored_hits[:top_k],
                "error": None,
            }
        except Exception as exc:
            return {
                "used": False,
                "hits": [],
                "scored_hits": [],
                "error": f"{type(exc).__name__}: {exc}",
            }

    def _build_infer_prompt(self, *, prompt: str, retrieved_hits: list[dict[str, Any]]) -> str:
        if not retrieved_hits:
            return prompt
        context = "\n".join(
            f"{idx}. (score={float(item.get('score', 0.0)):.3f}) {item.get('text', '')}"
            for idx, item in enumerate(retrieved_hits, start=1)
        )
        return (
            "Retrieved memory context:\n"
            f"{context}\n\n"
            "User conversation:\n"
            f"{prompt}"
        )

    async def _orient(
        self,
        *,
        prompt: str,
        observe_ctx: dict[str, Any],
        session_id: str,
        model_seed: str | None,
        mcts_meta: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        lower = prompt.lower()
        if "speed" in lower or "latency" in lower:
            priority = "speed"
        elif "memory" in lower or "ram" in lower:
            priority = "memory"
        else:
            priority = "correctness"

        candidates = self._build_candidates(
            prompt=prompt,
            priority=priority,
            model_seed=model_seed,
            mcts_meta=mcts_meta,
        )
        simulations: list[dict[str, Any]] = []
        for candidate in candidates:
            sim = await self.kernel.simulate_world(candidate["draft"], session_id=session_id)
            simulations.append(
                {
                    "candidate_id": candidate["id"],
                    "risk_score": float(sim.get("risk_score", 1.0)),
                    "confidence": float(sim.get("confidence", 0.0)),
                    "summary": sim,
                }
            )
        simulations.sort(key=lambda item: (item["risk_score"], -item["confidence"]))
        selected_id = simulations[0]["candidate_id"]
        selected_candidate = next(
            candidate for candidate in candidates if candidate["id"] == selected_id
        )

        return {
            "objective": "Solve engineering task via OODA with verifier gate",
            "constraints": observe_ctx["constraints"],
            "priority": priority,
            "acceptance_rules": {
                "verifier_pass": True,
                "max_risk_score": self.risk_threshold,
            },
            "candidates": candidates,
            "candidate_simulations": simulations,
            "selected_candidate": selected_candidate,
        }

    def _build_candidates(
        self, *, prompt: str, priority: str, model_seed: str | None,
        mcts_meta: dict[str, Any] | None = None,
    ) -> list[dict[str, str]]:
        candidates = [
            {
                "id": "secure-minimal",
                "draft": (
                    "Implement secure minimal patch:\n"
                    "- validate input strictly\n"
                    "- hash secret with safe primitive\n"
                    "- add timeout and rate limit\n"
                    f"- priority: {priority}\n"
                    f"- request context:\n{prompt}"
                ),
            },
            {
                "id": "balanced-throughput",
                "draft": (
                    "Implement balanced throughput patch:\n"
                    "- keep interface stable\n"
                    "- reduce extra allocations\n"
                    "- preserve audit log and safety checks\n"
                    f"- priority: {priority}\n"
                    f"- request context:\n{prompt}"
                ),
            },
            {
                "id": "memory-conservative",
                "draft": (
                    "Implement memory-conservative patch:\n"
                    "- avoid retaining large temporary buffers\n"
                    "- stream processing with deterministic guards\n"
                    "- enforce verifier compatibility\n"
                    f"- priority: {priority}\n"
                    f"- request context:\n{prompt}"
                ),
            },
        ]
        if model_seed:
            mcts_info = ""
            if mcts_meta:
                branches = mcts_meta.get("branches_explored", 0)
                reward = mcts_meta.get("best_branch_reward", 0.0)
                mcts_info = f"\n- MCTS: {branches} branches explored, best reward={reward:.3f}"
            candidates.insert(
                0,
                {
                    "id": "model-seed",
                    "draft": (
                        "Model-seeded draft:\n"
                        f"{model_seed}\n"
                        f"- Enforce deterministic safeguards before final act.{mcts_info}"
                    ),
                },
            )
        return candidates

    async def _fast_system_seed(self, prompt: str) -> tuple[dict[str, Any], str | None]:
        try:
            status = await self.kernel.model_status()
            if not bool(status.get("loaded", False)):
                return (
                    {
                        "used": False,
                        "model_id": None,
                        "fallback_reason": "model_not_loaded",
                        "latency_ms": 0,
                        "mcts_meta": None,
                    },
                    None,
                )

            # Try MCTS-based inference first for multi-branch reasoning.
            mcts_meta: dict[str, Any] | None = None
            infer: dict[str, Any] | None = None
            try:
                mcts_result = await self.kernel.model_infer_mcts(
                    prompt=prompt, max_new_tokens=64, num_branches=3
                )
                text = str(mcts_result.get("text", "")).strip()
                mcts_meta = {
                    "branches_explored": mcts_result.get("branches_explored", 0),
                    "best_branch_reward": float(mcts_result.get("best_branch_reward", 0.0)),
                }
                infer = mcts_result
            except Exception:
                # Fallback to greedy inference if MCTS fails.
                infer = await self.kernel.model_infer(prompt=prompt, max_new_tokens=96)
                text = str(infer.get("text", "")).strip()

            model_id = infer.get("model_id") or status.get("model_id")
            latency_ms = int(infer.get("latency_ms") or 0)
            if self._is_garbage_model_output(text):
                return (
                    {
                        "used": False,
                        "model_id": model_id,
                        "fallback_reason": "garbage_detected",
                        "latency_ms": latency_ms,
                        "mcts_meta": mcts_meta,
                    },
                    None,
                )
            return (
                {
                    "used": True,
                    "model_id": model_id,
                    "fallback_reason": None,
                    "latency_ms": latency_ms,
                    "mcts_meta": mcts_meta,
                },
                text,
            )
        except Exception as exc:
            return (
                {
                    "used": False,
                    "model_id": None,
                    "fallback_reason": f"kernel_model_infer_error:{type(exc).__name__}",
                    "latency_ms": 0,
                    "mcts_meta": None,
                },
                None,
            )

    def _is_garbage_model_output(self, text: str) -> bool:
        clean = text.strip()
        if not clean or len(clean) < 8:
            return True
        if clean.lower() in {"assistant:", "user:", "nan", "null"}:
            return True
        words = clean.split()
        if len(words) < 2:
            return True
        if max((len(set(chunk)) for chunk in clean.split()), default=0) <= 1:
            return True
        repeated_pairs = sum(
            1 for idx in range(1, len(clean)) if clean[idx] == clean[idx - 1]
        )
        if repeated_pairs / max(1, len(clean) - 1) > 0.35:
            return True
        word_counts: dict[str, int] = {}
        for word in words:
            normalized = word.lower()
            word_counts[normalized] = word_counts.get(normalized, 0) + 1
        dominant_ratio = max(word_counts.values()) / len(words)
        if len(words) >= 5 and dominant_ratio > 0.6:
            return True
        return False

    def _revise_draft(
        self,
        draft: str,
        *,
        sim: dict[str, Any],
        verifier: dict[str, Any],
        attempt: int,
    ) -> str:
        violations = ", ".join(verifier.get("violations", [])) or "none"
        risk = sim.get("risk_score", 1.0)
        sanitized = (
            draft.replace("drop", "soft-delete")
            .replace("rm -rf", "remove-with-review")
            .replace("unsafe", "safe")
            .replace("eval(", "safe_eval(")
        )
        return (
            f"{sanitized}\n"
            f"- Revise attempt #{attempt}: reduce risk below {self.risk_threshold}.\n"
            f"- Current risk: {risk:.2f}. Violations: {violations}.\n"
            "- Add guardrails: input validation, timeout, rate limit, audit log."
        )

    def _act(
        self,
        *,
        draft: str,
        observe_ctx: dict[str, Any],
        orient_ctx: dict[str, Any],
        sim: dict[str, Any],
        verifier: dict[str, Any],
        ooda_trace: dict[str, Any],
        model_runtime_meta: dict[str, Any],
        force_insufficient_response: bool,
    ) -> dict[str, Any]:
        if force_insufficient_response:
            content = FALLBACK_RESPONSE
        else:
            content = (
                "Proposed engineering artifact:\n"
                f"{draft}\n\n"
                f"Risk score: {sim['risk_score']:.2f} | Verifier pass: {verifier['pass']}"
            )
        return {
            "content": content,
            "metadata": {
                "observe": observe_ctx,
                "orient": orient_ctx,
                "simulation": sim,
                "verifier": verifier,
                "ooda_trace": ooda_trace,
                "verifier_required": True,
                "model_runtime": model_runtime_meta,
            },
        }
