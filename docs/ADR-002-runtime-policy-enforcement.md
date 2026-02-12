# ADR-002: Runtime Policy Enforcement for Chat Completions

## Status
Accepted

## Context
- vAGI requires deterministic OODA execution before answer output.
- Verifier gate is mandatory for safety.
- User-facing responses must not leak internal reasoning traces.

## Decision
- Introduce `IdentityPolicyEngine` in orchestrator runtime.
- Apply hard-enforcement only to `POST /v1/chat/completions` for this phase.
- Enforce:
  - required OODA stages (`observe`, `orient`, `decide`, `act`)
  - verifier gate pass when `verifier_required=true`
  - output leak prevention for internal instruction/trace strings
- On policy failure return `HTTP 422` with structured error payload.
- Persist policy audit fields in `episodes` table.

## Consequences
- Pros:
  - deterministic safety gate before response generation
  - explicit, machine-readable failure reasons
  - auditable history of policy decisions
- Trade-offs:
  - stricter gating may increase 422 rate during prompt edge cases
  - additional complexity in orchestrator response path

