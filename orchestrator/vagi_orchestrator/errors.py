from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class PolicyViolation:
    code: str
    message: str
    stage: str
    severity: str = "high"

    def to_dict(self) -> dict[str, str]:
        return {
            "code": self.code,
            "message": self.message,
            "stage": self.stage,
            "severity": self.severity,
        }


@dataclass(slots=True)
class PolicyError(Exception):
    code: str
    message: str
    violations: list[PolicyViolation] = field(default_factory=list)

    def to_response(self) -> dict[str, Any]:
        return {
            "error": {
                "code": self.code,
                "message": self.message,
                "details": [v.to_dict() for v in self.violations],
            }
        }

