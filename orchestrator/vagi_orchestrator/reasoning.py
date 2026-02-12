from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from typing import Any

from .kernel_client import KernelClient
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


@dataclass(slots=True)
class Reasoner:
    kernel: KernelClient
    store: EpisodeStore
    max_decide_iters: int = 12
    risk_threshold: float = 0.65
    reasoner_mode: str = "classic"
    weaver_top_k: int = 3

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

        # OBSERVE
        observe_ctx = self._observe(messages=messages, prompt=prompt)
        await self.kernel.init_state(session_id=session_id)
        await self.kernel.update_state(session_id=session_id, input_text=prompt)

        # ORIENT
        orient_ctx = await self._orient(
            prompt=prompt, observe_ctx=observe_ctx, session_id=session_id
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

    def _observe(self, *, messages: list[dict[str, str]], prompt: str) -> dict[str, Any]:
        token_count = sum(len(msg.get("content", "").split()) for msg in messages)
        max_line_len = max((len(line) for line in prompt.splitlines()), default=0)
        domain = self._infer_domain(prompt)
        return {
            "intent": "engineering_request",
            "domain": domain,
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
        }

    async def _orient(
        self, *, prompt: str, observe_ctx: dict[str, Any], session_id: str
    ) -> dict[str, Any]:
        lower = prompt.lower()
        if "speed" in lower or "latency" in lower:
            priority = "speed"
        elif "memory" in lower or "ram" in lower:
            priority = "memory"
        else:
            priority = "correctness"

        mode = self.reasoner_mode if self.reasoner_mode in {"classic", "hybrid", "weaver"} else "classic"
        classic_candidates = self._build_candidates(
            prompt=prompt,
            priority=priority,
            domain=str(observe_ctx.get("domain", "general")),
        )
        weaver_candidates = []
        if mode in {"hybrid", "weaver"}:
            weaver_candidates = await self._build_weaver_candidates(
                prompt=prompt,
                priority=priority,
                session_id=session_id,
            )

        if mode == "weaver":
            candidates = weaver_candidates or classic_candidates
        elif mode == "hybrid":
            candidates = weaver_candidates + classic_candidates
        else:
            candidates = classic_candidates

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
            "reasoner_mode": mode,
            "acceptance_rules": {
                "verifier_pass": True,
                "max_risk_score": self.risk_threshold,
            },
            "candidates": candidates,
            "candidate_simulations": simulations,
            "selected_candidate": selected_candidate,
        }

    def _build_candidates(self, *, prompt: str, priority: str, domain: str) -> list[dict[str, str]]:
        return [
            {
                "id": "secure-minimal",
                "source": "classic",
                "draft": (
                    "Implement secure minimal patch:\n"
                    "- validate input strictly\n"
                    "- hash secret with safe primitive\n"
                    "- add timeout and rate limit\n"
                    f"- domain: {domain}\n"
                    f"- priority: {priority}\n"
                    f"- request context:\n{prompt}"
                ),
            },
            {
                "id": "balanced-throughput",
                "source": "classic",
                "draft": (
                    "Implement balanced throughput patch:\n"
                    "- keep interface stable\n"
                    "- reduce extra allocations\n"
                    "- preserve audit log and safety checks\n"
                    f"- domain: {domain}\n"
                    f"- priority: {priority}\n"
                    f"- request context:\n{prompt}"
                ),
            },
            {
                "id": "memory-conservative",
                "source": "classic",
                "draft": (
                    "Implement memory-conservative patch:\n"
                    "- avoid retaining large temporary buffers\n"
                    "- stream processing with deterministic guards\n"
                    "- enforce verifier compatibility\n"
                    f"- domain: {domain}\n"
                    f"- priority: {priority}\n"
                    f"- request context:\n{prompt}"
                ),
            },
        ]

    async def _build_weaver_candidates(
        self,
        *,
        prompt: str,
        priority: str,
        session_id: str,
    ) -> list[dict[str, str]]:
        weave_plan = getattr(self.kernel, "weave_plan", None)
        if weave_plan is None:
            return []

        bindings = self._derive_bindings_from_prompt(prompt)
        input_value = max(1, len(prompt.split()))
        plan = await weave_plan(
            query=prompt,
            input_value=input_value,
            bindings=bindings,
            top_k=self.weaver_top_k,
            risk_threshold=self.risk_threshold,
            verifier_required=True,
            session_id=session_id,
        )
        if not plan:
            return []

        candidates_payload = list(plan.get("candidates", []))
        if not candidates_payload:
            return []
        accepted = [entry for entry in candidates_payload if bool(entry.get("accepted", False))]
        if not accepted:
            return []
        selected_pool = accepted[: self.weaver_top_k]

        candidates: list[dict[str, str]] = []
        for idx, entry in enumerate(selected_pool):
            template_id = str(entry.get("template_id", f"template-{idx}"))
            bound_logic = str(entry.get("bound_logic", "")).strip()
            similarity = float(entry.get("similarity", 0.0))
            risk_score = float(entry.get("risk_score", 1.0))
            verifier_pass = bool(entry.get("verifier_pass", False))
            output_value = int(entry.get("output", 0))
            draft = (
                f"Implement via Weaver template `{template_id}`:\n"
                f"- source: hdc-weaver\n"
                f"- priority: {priority}\n"
                f"- similarity: {similarity:.3f}\n"
                f"- risk_preview: {risk_score:.2f}\n"
                f"- verifier_preview: {verifier_pass}\n"
                f"- jit_preview_output: {output_value}\n"
                f"- bound_logic:\n{bound_logic}\n"
                f"- request context:\n{prompt}"
            )
            candidates.append(
                {
                    "id": f"weaver-{template_id}-{idx}",
                    "source": "weaver",
                    "draft": draft,
                }
            )
        return candidates

    def _derive_bindings_from_prompt(self, prompt: str) -> dict[str, str]:
        values = re.findall(r"-?\d+", prompt)
        bindings = {
            "delta": values[0] if values else "5",
            "mask": values[1] if len(values) >= 2 else "3",
        }
        return bindings

    def _infer_domain(self, prompt: str) -> str:
        lower = prompt.lower()
        mapping = [
            ("python", "python"),
            ("rust", "rust"),
            ("javascript", "javascript"),
            ("typescript", "typescript"),
            ("sql", "database"),
            ("database", "database"),
            ("security", "security"),
            ("auth", "security"),
        ]
        for needle, domain in mapping:
            if needle in lower:
                return domain
        return "general"

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
    ) -> dict[str, Any]:
        return {
            "content": (
                "Proposed engineering artifact:\n"
                f"{draft}\n\n"
                f"Risk score: {sim['risk_score']:.2f} | Verifier pass: {verifier['pass']}"
            ),
            "metadata": {
                "observe": observe_ctx,
                "orient": orient_ctx,
                "simulation": sim,
                "verifier": verifier,
                "ooda_trace": ooda_trace,
                "verifier_required": True,
            },
        }
