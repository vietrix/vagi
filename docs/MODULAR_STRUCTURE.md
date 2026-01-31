# vAGI - Modular Core Structure

## New File Organization

The `core` directory has been completely restructured into logical modules.

### Structure

```
core/
├── __init__.py                    # Main exports
│
├── agi/                           # AGI Integration
│   ├── __init__.py
│   ├── config.py                  # AGIConfig
│   ├── model.py                   # AGIModel
│   └── executor.py                # AGIExecutor
│
├── base/                          # Core RL Architecture (Backbone)
│   ├── __init__.py
│   ├── config.py                  # VAGIConfig
│   ├── model.py                   # VAGICore
│   ├── backbone.py                # Transformer Backbone
│   ├── heads.py                   # Prediction Heads
│   ├── memory.py                  # RecurrentState, KVCache
│   ├── tokenizer.py               # TokenizerWrapper
│   ├── utils.py                   # Utilities
│   └── presets.py                 # Lite Configs
│
├── nlp/                           # Natural Language Processing
│   ├── __init__.py
│   └── language.py                # NLP Components
│
├── knowledge/                     # Knowledge & Memory
│   ├── __init__.py
│   └── memory.py                  # KG, Hierarchical Memory
│
├── reasoning/                     # Abstract Reasoning
│   ├── __init__.py
│   └── abstract.py                # Reasoning Engines
│
├── learning/                      # Meta-Learning
│   ├── __init__.py
│   └── meta.py                    # MAML, Curriculum
│
├── planning/                      # Planning & Search
│   ├── __init__.py
│   ├── budget.py                  # Compute Budget
│   └── dyna.py                    # Model-based Rollouts
│
├── training/                      # Training Utilities
│   ├── __init__.py
│   ├── experience.py              # Replay Buffer
│   ├── losses.py                  # Loss Functions
│   ├── returns.py                 # GAE, TD-Lambda
│   ├── calibration.py             # Confidence Calibration
│   └── diagnostics.py             # Metrics
│
├── interaction/                   # Tool Use
│   ├── __init__.py
│   └── tools.py                   # Tool Registry
│
└── perception/                    # Vision
    ├── __init__.py
    └── vision.py                  # Vision Encoders
```

## Module Descriptions

### `agi`
High-level AGI integration. Combines all other modules into a unified system.

### `base`
The foundational VAGI architecture (Transformer, Heads, Memory). This is equivalent to the old `core` logic minus the specialized components.

### `nlp`
Dedicated language processing capabilities (tokenization, embeddings, masking).

### `knowledge`
Long-term memory systems (Semantic, Episodic, Knowledge Graph).

### `reasoning`
Abstract reasoning modules (Causal, Relational, Counterfactual).

### `learning`
Meta-learning algorithms (MAML, Few-Shot, Curriculum).

### `planning`
Inference-time planning and model-based rollouts (Dyna-Q style).

### `training`
Utilities for the training loop (Experience Replay, Losses, GAE).

### `interaction`
External tool usage and API calls.

### `perception`
Visual processing and multi-modal alignment.

## Import Guide

**Standard Imports (Recommended):**
```python
from core.agi import AGIModel
from core.base import VAGICore, VAGIConfig
from core.training import ExperienceBuffer
```

**Backward Compatibility:**
Top-level imports still work for backward compatibility:
```python
from core import AGIModel  # Works
from core import VAGICore  # Works
from core import ExperienceBuffer  # Works
```

## Migration Table

| Old Location | New Location |
|--------------|--------------|
| `core/model.py` | `core/base/model.py` |
| `core/config.py` | `core/base/config.py` |
| `core/backbone.py` | `core/base/backbone.py` |
| `core/losses.py` | `core/training/losses.py` |
| `core/experience.py` | `core/training/experience.py` |
| `core/dyna.py` | `core/planning/dyna.py` |
| `core/budget.py` | `core/planning/budget.py` |
| `core/agi_*, *.py` | `core/[module]/*.py` |

This structure ensures clean separation of concerns and simpler navigation.
