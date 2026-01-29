# vAGI Model Card (Core)

## Overview

vAGI is a **model‑centric** architecture: one core model supports both fast actions and
deliberate planning. It is not an agent framework; tool use and runtime orchestration
are layered on top of the same neural core.

## Capabilities

- Unified policy/value/world modeling
- Reflection heads for error awareness and information gain
- Risk‑aware planning (CEM / tree search)
- Adaptive compute via budget controller
- Long‑horizon memory with stabilization

## Intended Use

Research and prototyping of unified decision‑making. The core is designed for offline
training and can be distilled to smaller models for efficiency.

## Limitations

- Planning quality depends on world model fidelity.
- Reflection heads require curated targets to be reliable.
- No external tool or environment guarantees are implied.

## Model‑centric AGI rationale

vAGI keeps intelligence **inside a single model** with:

1. Shared representations for language, actions, value, and world prediction.
2. Two compute modes (act vs think) without model switching.
3. Self‑assessment heads to control uncertainty and exploration.

This keeps the system modular while preserving a unified "brain" for learning and
adaptation.
