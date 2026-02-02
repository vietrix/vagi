# vAGI

<div align="center">

![vAGI](https://img.shields.io/badge/vAGI-Artificial%20General%20Intelligence-blue?style=for-the-badge)
![Python](https://img.shields.io/badge/Python-3.10+-green?style=flat-square&logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red?style=flat-square&logo=pytorch)
![License](https://img.shields.io/badge/License-Apache%202.0-yellow?style=flat-square)

**A Research-Focused Artificial General Intelligence Implementation**

[Features](#features) | [Installation](#installation) | [Quick Start](#quick-start) | [Architecture](#architecture) | [Benchmarks](#benchmarks) | [Contributing](#contributing)

</div>

---

## Overview

vAGI is a comprehensive AGI research implementation combining modern transformer architectures with cognitive-inspired modules. Built for researchers and developers exploring the frontiers of artificial intelligence.

### Key Capabilities

- **Language Understanding** - Transformer-based text generation with BPE tokenization
- **Meta-Cognition** - Self-awareness module for confidence estimation
- **Online Learning** - Continuous learning during inference
- **Program Synthesis** - Neuro-symbolic compositional reasoning
- **Scene Understanding** - Object-centric perception with slot attention
- **Intrinsic Motivation** - Curiosity-driven exploration

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              vAGI Architecture                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐  │
│  │    Input    │    │  Tokenizer  │    │  Embedding  │    │   Rotary    │  │
│  │   (Text)    │───▶│    (BPE)    │───▶│   Layer     │───▶│  Position   │  │
│  └─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘  │
│                                                                 │           │
│                                                                 ▼           │
│  ┌───────────────────────────────────────────────────────────────────────┐ │
│  │                     Transformer Backbone (N Layers)                    │ │
│  │  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐        │ │
│  │  │  Multi-Head     │  │  Feed-Forward   │  │     Layer       │        │ │
│  │  │  Self-Attention │─▶│    Network      │─▶│  Normalization  │        │ │
│  │  │  (GQA/RoPE)     │  │  (SwiGLU)       │  │    + Residual   │        │ │
│  │  └─────────────────┘  └─────────────────┘  └─────────────────┘        │ │
│  └───────────────────────────────────────────────────────────────────────┘ │
│                                        │                                    │
│              ┌─────────────────────────┼─────────────────────────┐         │
│              │                         │                         │         │
│              ▼                         ▼                         ▼         │
│  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐        │
│  │   Meta-Cognition │    │   Language      │    │   Memory        │        │
│  │   (Confidence)   │    │   Head          │    │   System        │        │
│  │                  │    │   (Text Gen)    │    │   (KV Cache)    │        │
│  └─────────────────┘    └─────────────────┘    └─────────────────┘        │
│              │                         │                         │         │
│              ▼                         ▼                         ▼         │
│  ┌───────────────────────────────────────────────────────────────────────┐ │
│  │                           Output Integration                           │ │
│  │     • Text Logits  • Confidence  • Uncertainty  • Online Learning     │ │
│  └───────────────────────────────────────────────────────────────────────┘ │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘

                              Extended Modules
┌─────────────────────────────────────────────────────────────────────────────┐
│                                                                             │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐   │
│  │   Program    │  │   Scene      │  │   Intrinsic  │  │  Continuous  │   │
│  │   Synthesis  │  │   Graphs     │  │   Motivation │  │   Learning   │   │
│  └──────────────┘  └──────────────┘  └──────────────┘  └──────────────┘   │
│                                                                             │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐   │
│  │   Grounded   │  │   Knowledge  │  │   Abstract   │  │     Tool     │   │
│  │   Language   │  │   Graph      │  │   Reasoning  │  │     Use      │   │
│  └──────────────┘  └──────────────┘  └──────────────┘  └──────────────┘   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Features

| Feature | Description | Status |
|---------|-------------|--------|
| **Transformer Core** | Multi-head attention with GQA, RoPE | ✅ Complete |
| **Meta-Cognition** | Self-awareness, confidence estimation | ✅ Complete |
| **Online Learning** | Learn during inference | ✅ Complete |
| **BPE Tokenizer** | Vietnamese + English support | ✅ Complete |
| **Program Synthesis** | DSL-based compositional reasoning | ✅ Complete |
| **Scene Graphs** | Object-centric slot attention | ✅ Complete |
| **Memory System** | Hierarchical memory with KV cache | ✅ Complete |
| **Intrinsic Motivation** | Curiosity-driven exploration | ✅ Complete |

---

## Installation

### Requirements

- Python 3.10+
- PyTorch 2.0+
- CUDA 11.8+ (optional, for GPU)

### Install from source

```bash
# Clone repository
git clone https://github.com/yourusername/vagi.git
cd vagi

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/macOS
# or: venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

---

## Quick Start

### Training

```bash
# Train tiny model (fast, CPU-friendly)
python scripts/train.py --tiny --epochs 20

# Train small model (GPU recommended)
python scripts/train.py --small --epochs 50

# Train full model (requires GPU)
python scripts/train.py --epochs 100
```

### Chat with Model

```bash
# Interactive chat
python scripts/chat.py

# With custom checkpoint
python scripts/chat.py --model checkpoints/model.pt --temp 0.8
```

### Demo

```bash
# Run capability demo
python scripts/demo.py
```

---

## Model Configurations

| Config | Parameters | Hidden | Layers | Heads | Use Case |
|--------|------------|--------|--------|-------|----------|
| **Tiny** | ~10M | 128 | 4 | 4 | Fast CPU training |
| **Small** | ~165M | 512 | 12 | 8 | GPU training |
| **Default** | ~895M | 1024 | 24 | 16 | Full capability |
| **Large** | ~2B | 2048 | 32 | 32 | Research scale |

---

## Benchmarks

### Training Performance (Actual Results)

| Config | Parameters | Dataset | Epochs | Final Loss | Time (CPU) |
|--------|------------|---------|--------|------------|------------|
| **Tiny** | 3.4M | 450 samples | 20 | **2.19** | ~12 min |
| Small | 165M | 450 samples | 20 | ~1.5 | ~60 min |

### Inference Performance (CPU - PyTorch 2.10)

| Metric | Batch Size 1 | Batch Size 2 | Batch Size 4 |
|--------|--------------|--------------|--------------|
| Latency | 9.6ms | 13.1ms | 26.3ms |
| Throughput | 104 samples/s | 153 samples/s | 152 samples/s |

### Text Generation Speed

| Config | Tokens/sec | Avg Generation Time |
|--------|------------|---------------------|
| **Tiny (CPU)** | 174 | 0.17s (30 tokens) |

### Architecture Details

| Component | Tiny | Small | Default | Large |
|-----------|------|-------|---------|-------|
| Parameters | 3.4M | 165M | 895M | 2B |
| Hidden Size | 128 | 512 | 1024 | 2048 |
| Layers | 4 | 12 | 24 | 32 |
| Attention Heads | 4 | 8 | 16 | 32 |
| Vocab Size | 5K | 10K | 50K | 100K |

---

## Project Structure

```
vagi/
├── core/
│   ├── agi/
│   │   ├── model.py          # Main AGI model
│   │   ├── config.py         # Configuration classes
│   │   └── executor.py       # Inference executor
│   ├── nlp/
│   │   ├── language.py       # BPE tokenizer
│   │   └── grounded_language.py
│   ├── learning/
│   │   └── metacognition.py  # Meta-cognition module
│   ├── reasoning/
│   │   └── program_synthesis.py
│   └── training/
│       ├── continuous_learner.py
│       └── online_learner.py
├── scripts/
│   ├── train.py              # Training script
│   ├── chat.py               # Interactive chat
│   ├── demo.py               # Capability demo
│   └── eval.py               # Evaluation
├── data/
│   └── train_dataset.jsonl   # Training data
├── checkpoints/              # Saved models
└── docs/                     # Documentation
```

---

## Training Data Format

JSONL format with input-output pairs:

```jsonl
{"input": "Xin chào", "output": "Xin chào! Tôi là vAGI, trợ lý AI."}
{"input": "Viết hàm tính giai thừa", "output": "def factorial(n):\n    if n <= 1:\n        return 1\n    return n * factorial(n-1)"}
{"input": "Thủ đô Việt Nam?", "output": "Thủ đô của Việt Nam là Hà Nội."}
```

---

## Research References

vAGI implements concepts from:

- **Attention Is All You Need** (Vaswani et al., 2017)
- **RoFormer** (Su et al., 2021) - Rotary Position Embedding
- **GQA** (Ainslie et al., 2023) - Grouped Query Attention
- **World Models** (Ha & Schmidhuber, 2018)
- **MuZero** (Schrittwieser et al., 2020)
- **Slot Attention** (Locatello et al., 2020)
- **Intrinsic Curiosity** (Pathak et al., 2017)
- **DreamCoder** (Ellis et al., 2021) - Program Synthesis
- **MAML** (Finn et al., 2017) - Meta-Learning

---

## Contributing

Contributions welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## License

Apache 2.0 - See [LICENSE](LICENSE) for details.

---

## Acknowledgments

Built with PyTorch. Inspired by the collective research of the AI community.

---

<div align="center">

**vAGI** - Exploring the Frontier of Artificial General Intelligence

[Documentation](docs/) | [Issues](https://github.com/yourusername/vagi/issues) | [Discussions](https://github.com/yourusername/vagi/discussions)

</div>
