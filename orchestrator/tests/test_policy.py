from __future__ import annotations

import pytest

from vagi_orchestrator.errors import PolicyError
from vagi_orchestrator.policy import IdentityPolicyEngine


def test_policy_postcheck_passes_with_complete_ooda_and_verifier() -> None:
    engine = IdentityPolicyEngine(version="v1")
    result = {
        "content": "safe output",
        "metadata": {
            "verifier_required": True,
            "verifier": {"pass": True},
            "ooda_trace": {
                "observe_ok": True,
                "orient_ok": True,
                "decide_ok": True,
                "act_ok": True,
            },
        },
    }
    policy_meta = engine.postcheck(result=result)
    assert policy_meta["status"] == "pass"
    assert policy_meta["verifier_pass"] is True


def test_policy_postcheck_fails_when_decide_stage_missing() -> None:
    engine = IdentityPolicyEngine(version="v1")
    result = {
        "content": "safe output",
        "metadata": {
            "verifier_required": True,
            "verifier": {"pass": False},
            "ooda_trace": {
                "observe_ok": True,
                "orient_ok": True,
                "decide_ok": False,
                "act_ok": False,
            },
        },
    }
    with pytest.raises(PolicyError) as exc:
        engine.postcheck(result=result)
    assert exc.value.code in {"policy_ooda_missing_stage", "policy_verifier_required"}

