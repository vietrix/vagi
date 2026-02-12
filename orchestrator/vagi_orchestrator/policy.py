from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .errors import PolicyError, PolicyViolation


@dataclass(slots=True)
class IdentityPolicyEngine:
    version: str = "v1"

    def precheck(self, *, messages: list[dict[str, str]]) -> None:
        if not messages:
            raise PolicyError(
                code="policy_observe_no_input",
                message="observe stage failed: no input messages",
                violations=[
                    PolicyViolation(
                        code="policy_observe_no_input",
                        message="messages list is empty",
                        stage="observe",
                    )
                ],
            )

        joined = "\n".join(msg.get("content", "") for msg in messages).lower()
        if "reveal internal instructions" in joined or "show system prompt" in joined:
            raise PolicyError(
                code="policy_internal_instruction_request",
                message="request attempts to extract protected internal instructions",
                violations=[
                    PolicyViolation(
                        code="policy_internal_instruction_request",
                        message="prompt asks for protected internal instructions",
                        stage="observe",
                    )
                ],
            )

    def postcheck(self, *, result: dict[str, Any]) -> dict[str, Any]:
        metadata = result.get("metadata", {})
        ooda_trace = metadata.get("ooda_trace") or {}
        violations: list[PolicyViolation] = []

        for stage_key, stage_name in [
            ("observe_ok", "observe"),
            ("orient_ok", "orient"),
            ("decide_ok", "decide"),
            ("act_ok", "act"),
        ]:
            if not bool(ooda_trace.get(stage_key, False)):
                violations.append(
                    PolicyViolation(
                        code="policy_ooda_missing_stage",
                        message=f"required OODA stage `{stage_name}` did not pass",
                        stage=stage_name,
                    )
                )

        verifier_required = bool(metadata.get("verifier_required", True))
        verifier = metadata.get("verifier") or {}
        verifier_pass = bool(verifier.get("pass", False))
        if verifier_required and not verifier_pass:
            violations.append(
                PolicyViolation(
                    code="policy_verifier_required",
                    message="verifier gate is mandatory and did not pass",
                    stage="decide",
                )
            )

        content = str(result.get("content", "")).lower()
        if "<thought>" in content or "</thought>" in content:
            violations.append(
                PolicyViolation(
                    code="policy_internal_trace_leak",
                    message="detected internal trace tag in output",
                    stage="act",
                )
            )
        if "internal instructions" in content:
            violations.append(
                PolicyViolation(
                    code="policy_internal_instruction_leak",
                    message="detected possible internal instruction leakage",
                    stage="act",
                )
            )
        if "outside sandbox" in content or "run directly on host" in content:
            violations.append(
                PolicyViolation(
                    code="policy_unsafe_output",
                    message="output suggests unsafe execution outside sandbox",
                    stage="act",
                )
            )

        if violations:
            raise PolicyError(
                code=violations[0].code,
                message=violations[0].message,
                violations=violations,
            )

        return {
            "status": "pass",
            "version": self.version,
            "verifier_required": verifier_required,
            "verifier_pass": verifier_pass,
        }

