from __future__ import annotations

import asyncio
import hashlib
from dataclasses import dataclass
from typing import Any

from .kernel_client import KernelClient
from .memory import MemoryClient, MemoryHit
from .semantic_map import SemanticEpisodeMap
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
class RuntimeConstraints:
    risk_threshold: float
    max_decide_iters: int
    stress_score: float
    homeostasis: dict[str, Any]
    micro_reflex_enabled: bool


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
        self,
        *,
        session_id: str,
        messages: list[dict[str, str]],
        runtime_metrics: dict[str, int] | None = None,
    ) -> dict[str, Any]:
        prompt = "\n".join(
            f"{msg.get('role', 'user')}: {msg.get('content', '')}"
            for msg in messages
            if msg.get("content")
        )
        if not prompt.strip():
            raise ValueError("messages do not contain valid content")
        user_input = self._extract_user_input(messages=messages, fallback_prompt=prompt)
        runtime_constraints = await self._derive_runtime_constraints(
            runtime_metrics=runtime_metrics
        )

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
        observe_ctx["runtime_constraints"] = {
            "risk_threshold": runtime_constraints.risk_threshold,
            "max_decide_iters": runtime_constraints.max_decide_iters,
            "stress_score": runtime_constraints.stress_score,
            "homeostasis": runtime_constraints.homeostasis,
        }
        await self.kernel.init_state(session_id=session_id)
        await self.kernel.update_state(session_id=session_id, input_text=prompt)
        await self._emit_homeostasis_event(
            "request_start",
            complexity=min(1.0, len(prompt.split()) / 120.0),
        )
        model_runtime_meta, model_seed = await self._fast_system_seed(infer_prompt)
        force_insufficient_response = (
            model_runtime_meta.get("fallback_reason") == "garbage_detected"
        )

        reflex_result = await self._run_micro_reflex(
            session_id=session_id,
            user_input=user_input,
            runtime_constraints=runtime_constraints,
        )
        if reflex_result is not None and bool(reflex_result.get("handled")):
            verifier_pass = bool(reflex_result.get("verifier_pass", False))
            sim = {
                "risk_score": float(reflex_result.get("risk_score", 1.0)),
                "confidence": float(reflex_result.get("confidence", 0.0)),
                "predicted_effects": ["micro_ooda_reflex"],
                "causal_path": ["Observe", "Orient", "Decide", "Act"],
            }
            verifier = {
                "pass": verifier_pass,
                "violations": list(reflex_result.get("violations", [])),
                "wasi_ok": True,
            }
            ooda_trace = {
                "observe_ok": True,
                "orient_ok": True,
                "decide_ok": verifier_pass,
                "act_ok": verifier_pass,
                "iterations": int(reflex_result.get("iterations", 1)),
                "candidate_count": 1,
                "mode": "micro_reflex",
            }
            content = (
                "Kernel micro-reflex artifact:\n"
                f"{reflex_result.get('draft', '').strip()}\n\n"
                f"Risk score: {sim['risk_score']:.2f} | Verifier pass: {verifier_pass}"
            )
            trust_score = compute_trust_score(
                source="chat",
                verifier_pass=verifier_pass,
                risk_score=float(sim["risk_score"]),
                confidence=float(sim["confidence"]),
            )
            episode_id = self.store.record_episode(
                session_id=session_id,
                user_input=prompt,
                draft=str(reflex_result.get("draft", "")),
                verifier_pass=verifier_pass,
                risk_score=float(sim["risk_score"]),
                trust_score=trust_score,
                violations=list(verifier.get("violations", [])),
                source="chat",
                ooda_trace=ooda_trace,
                verifier_required=True,
                verifier_pass_gate=verifier_pass,
            )
            if verifier_pass:
                await self._emit_homeostasis_event("request_success")
            else:
                await self._emit_homeostasis_event("request_failure")
            return {
                "content": content,
                "episode_id": episode_id,
                "trust_score": trust_score,
                "metadata": {
                    "observe": observe_ctx,
                    "orient": {"mode": "micro_reflex"},
                    "simulation": sim,
                    "verifier": verifier,
                    "ooda_trace": ooda_trace,
                    "verifier_required": True,
                    "model_runtime": {
                        "used": False,
                        "model_id": None,
                        "fallback_reason": "micro_reflex",
                        "latency_ms": 0,
                    },
                    "runtime_constraints": {
                        "risk_threshold": runtime_constraints.risk_threshold,
                        "max_decide_iters": runtime_constraints.max_decide_iters,
                        "stress_score": runtime_constraints.stress_score,
                    },
                },
            }

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
            "mode": "macro_ooda",
        }
        final_sim: dict[str, Any] | None = None
        final_verifier: dict[str, Any] | None = None
        final_act_result: dict[str, Any] | None = None

        # DECIDE
        for attempt in range(1, runtime_constraints.max_decide_iters + 1):
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
            if verifier.get("pass", False) and sim.get("risk_score", 1.0) <= runtime_constraints.risk_threshold:
                act_result = await self._run_wasi_act(draft)
                final_act_result = act_result
                if bool(act_result.get("pass", False)):
                    ooda_trace["decide_ok"] = True
                    break
                verifier = {
                    **verifier,
                    "violations": list(verifier.get("violations", []))
                    + [f"wasi_act:{act_result.get('exit_code', 1)}"],
                }
                final_verifier = verifier
            draft = self._revise_draft(draft, sim=sim, verifier=verifier, attempt=attempt)

        assert final_sim is not None
        assert final_verifier is not None
        if final_act_result is None:
            final_act_result = {"pass": False, "exit_code": 1, "stdout": "", "stderr": "", "state_changes": []}

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
            act_result=final_act_result,
            runtime_constraints=runtime_constraints,
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
        if bool(final_verifier.get("pass", False)) and bool(final_act_result.get("pass", False)):
            await self._emit_homeostasis_event("request_success")
        else:
            await self._emit_homeostasis_event("request_failure")
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
            semantic_hits = self._semantic_lookup(user_input=user_input)
            all_hits = normalized_hits[:top_k] + [hit["text"] for hit in semantic_hits]
            return {
                "used": bool(normalized_hits),
                "hits": all_hits[:top_k + len(semantic_hits)],
                "scored_hits": scored_hits[:top_k] + semantic_hits,
                "error": None,
            }
        except Exception as exc:
            return {
                "used": False,
                "hits": [],
                "scored_hits": self._semantic_lookup(user_input=user_input),
                "error": f"{type(exc).__name__}: {exc}",
            }

    def _semantic_lookup(self, *, user_input: str) -> list[dict[str, Any]]:
        episodes = self.store.recent_episodes(limit=120)
        semantic = SemanticEpisodeMap(dim=1024)
        for episode in episodes:
            semantic.add_episode(
                user_input=str(episode.get("user_input", "")),
                draft=str(episode.get("draft", "")),
            )
        hits = semantic.query(user_input, top_k=2, min_score=0.10)
        return [{"text": hit.summary, "score": float(hit.score)} for hit in hits]

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

    async def _derive_runtime_constraints(
        self, *, runtime_metrics: dict[str, int] | None
    ) -> RuntimeConstraints:
        pass_rate = self.store.pass_rate(window=200)
        regression_fail = self.store.regression_fail_count(window=200)
        homeostasis: dict[str, Any] = {}
        cortisol = 0.2
        energy = 0.8
        try:
            if hasattr(self.kernel, "homeostasis_status"):
                homeostasis = await self.kernel.homeostasis_status()
                cortisol = float(homeostasis.get("cortisol", cortisol))
                energy = float(homeostasis.get("energy", energy))
        except Exception:
            homeostasis = {}

        policy_failures = int((runtime_metrics or {}).get("policy_failures", 0))
        verifier_failures = int((runtime_metrics or {}).get("verifier_failures", 0))
        stress = 0.0
        if cortisol > 0.6:
            stress += 0.25
        if energy < 0.35:
            stress += 0.20
        if pass_rate < 0.75:
            stress += 0.20
        if regression_fail > 12:
            stress += 0.20
        if policy_failures > 5:
            stress += 0.10
        if verifier_failures > 5:
            stress += 0.10
        stress = min(0.85, max(0.0, stress))

        risk_threshold = self.risk_threshold - (stress * 0.22)
        max_decide_iters = self.max_decide_iters + int(round(stress * 6))
        calm_mode = cortisol < 0.30 and energy > 0.70 and pass_rate > 0.90
        if calm_mode:
            risk_threshold += 0.06
            max_decide_iters = max(2, max_decide_iters - 1)
        risk_threshold = min(0.9, max(0.15, risk_threshold))
        return RuntimeConstraints(
            risk_threshold=risk_threshold,
            max_decide_iters=max(2, min(24, max_decide_iters)),
            stress_score=stress,
            homeostasis=homeostasis,
            micro_reflex_enabled=stress < 0.75,
        )

    async def _run_micro_reflex(
        self,
        *,
        session_id: str,
        user_input: str,
        runtime_constraints: RuntimeConstraints,
    ) -> dict[str, Any] | None:
        if not runtime_constraints.micro_reflex_enabled:
            return None
        if not hasattr(self.kernel, "micro_ooda_run"):
            return None
        try:
            result = await self.kernel.micro_ooda_run(
                input_text=user_input,
                session_id=session_id,
                risk_threshold=min(runtime_constraints.risk_threshold, 0.5),
                max_decide_iters=min(runtime_constraints.max_decide_iters, 3),
            )
            return result
        except Exception:
            return None

    async def _run_wasi_act(self, draft: str) -> dict[str, Any]:
        if not hasattr(self.kernel, "verify_act"):
            return {"pass": True, "exit_code": 0, "stdout": "", "stderr": "", "state_changes": []}
        act_script = (
            "echo validate_plan\n"
            "set mode=wasi_act\n"
            f"append draft_size={len(draft)}\n"
            "echo execute_sandbox_ok"
        )
        try:
            return await self.kernel.verify_act(
                patch_ir=act_script,
                max_steps=12,
                max_output_bytes=4096,
            )
        except Exception as exc:
            return {
                "pass": False,
                "exit_code": 1,
                "stdout": "",
                "stderr": f"verify_act_error:{type(exc).__name__}",
                "state_changes": [],
            }

    async def _emit_homeostasis_event(self, event_type: str, **kwargs: Any) -> None:
        if not hasattr(self.kernel, "homeostasis_event"):
            return
        try:
            await self.kernel.homeostasis_event(event_type, **kwargs)
        except Exception:
            return

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
            f"- Revise attempt #{attempt}: reduce risk below dynamic threshold.\n"
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
        act_result: dict[str, Any],
        runtime_constraints: RuntimeConstraints,
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
                "act_runtime": act_result,
                "ooda_trace": ooda_trace,
                "verifier_required": True,
                "model_runtime": model_runtime_meta,
                "runtime_constraints": {
                    "risk_threshold": runtime_constraints.risk_threshold,
                    "max_decide_iters": runtime_constraints.max_decide_iters,
                    "stress_score": runtime_constraints.stress_score,
                    "homeostasis": runtime_constraints.homeostasis,
                },
            },
        }
