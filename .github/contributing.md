# Contributing to vAGI

First off, thank you for considering contributing to vAGI! It's people like you that make vAGI such a great tool for AGI research.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Project Architecture](#project-architecture)
- [How to Contribute](#how-to-contribute)
- [Pull Request Process](#pull-request-process)
- [Style Guidelines](#style-guidelines)
- [Community](#community)

## Code of Conduct

This project and everyone participating in it is governed by our [Code of Conduct](CODE_OF_CONDUCT.md). By participating, you are expected to uphold this code. Please report unacceptable behavior to zyntherdev@gmail.com.

## Getting Started

### Prerequisites

- Python 3.10 or higher
- PyTorch 2.0 or higher
- Git

### Development Setup

```bash
# Clone the repository
git clone https://github.com/vietrix/vagi.git
cd vagi

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/macOS
# or: venv\Scripts\activate  # Windows

# Install in development mode
pip install -e ".[dev]"

# Run tests to verify setup
python -m pytest tests/ -v
```

## Project Architecture

```
vagi/
├── core/
│   ├── agi/           # Main AGI model and config
│   ├── base/          # Core transformer backbone
│   ├── knowledge/     # Memory systems (working, semantic, episodic)
│   ├── learning/      # Meta-cognition, curriculum learning
│   ├── nlp/           # Language processing, tokenization
│   ├── perception/    # Vision encoders, multimodal fusion
│   ├── planning/      # Intrinsic motivation, goal generation
│   ├── reasoning/     # Abstract reasoning, program synthesis
│   └── training/      # Loss functions, experience replay
├── scripts/           # Training, evaluation, demo scripts
├── data/              # Training datasets
├── checkpoints/       # Saved model weights
└── docs/              # Documentation
```

### Key Components

| Component | Description | Location |
|-----------|-------------|----------|
| AGIModel | Main unified model | `core/agi/model.py` |
| VAGICore | Transformer backbone | `core/base/model.py` |
| MetaCognition | Self-awareness module | `core/learning/metacognition.py` |
| ProgramSynthesizer | Neuro-symbolic reasoning | `core/reasoning/program_synthesis.py` |
| HierarchicalMemory | Working/semantic/episodic | `core/knowledge/memory.py` |

## How to Contribute

### Types of Contributions

#### 1. Bug Reports

Found a bug? Please create an issue with:

- Clear, descriptive title
- Steps to reproduce
- Expected vs actual behavior
- Environment details (OS, Python version, PyTorch version)
- Error logs/stack traces

#### 2. Feature Requests

Have an idea? Open an issue describing:

- The problem you're trying to solve
- Your proposed solution
- Alternative approaches you've considered
- Relevant research papers (if applicable)

#### 3. Code Contributions

- **Bug fixes**: Always welcome!
- **New features**: Please discuss in an issue first
- **Documentation**: Improvements to docs, docstrings, examples
- **Tests**: Additional test coverage
- **Benchmarks**: Performance measurements and comparisons

#### 4. Research Contributions

- Novel architectures or modules
- Benchmark results and analysis
- Integration with new datasets
- Ablation studies

## Pull Request Process

### 1. Before You Start

- Check existing issues/PRs to avoid duplicates
- For significant changes, open an issue for discussion first
- Fork the repository and create a feature branch

### 2. Development

```bash
# Create feature branch
git checkout -b feature/your-feature-name

# Make your changes
# ... edit files ...

# Run tests
python -m pytest tests/ -v

# Run type checking (if applicable)
mypy core/

# Format code
black core/ scripts/
isort core/ scripts/
```

### 3. Commit Guidelines

We follow [Conventional Commits](https://www.conventionalcommits.org/):

```
<type>(<scope>): <description>

[optional body]

[optional footer]
```

**Types:**
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation only
- `style`: Formatting, no code change
- `refactor`: Code change that neither fixes nor adds
- `perf`: Performance improvement
- `test`: Adding tests
- `chore`: Maintenance tasks

**Examples:**
```
feat(reasoning): add counterfactual reasoning module
fix(memory): resolve episodic memory batching issue
docs(readme): update installation instructions
```

### 4. Pull Request Checklist

- [ ] Code follows project style guidelines
- [ ] Self-review completed
- [ ] Tests added/updated for changes
- [ ] Documentation updated if needed
- [ ] All tests passing
- [ ] CLA signed (first-time contributors)

### 5. Review Process

1. Submit PR with clear description
2. Automated checks run (tests, linting)
3. Maintainer reviews code
4. Address feedback if any
5. Maintainer merges when approved

## Style Guidelines

### Python Style

- Follow PEP 8
- Use type hints for function signatures
- Maximum line length: 100 characters
- Use docstrings for public functions/classes

```python
def compute_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    mask: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """Compute scaled dot-product attention.

    Args:
        query: Query tensor [batch, seq, dim]
        key: Key tensor [batch, seq, dim]
        value: Value tensor [batch, seq, dim]
        mask: Optional attention mask

    Returns:
        Attention output [batch, seq, dim]
    """
    ...
```

### Documentation Style

- Use clear, concise language
- Include code examples where helpful
- Keep README updated with major changes
- Add docstrings to all public APIs

## Community

### Getting Help

- **GitHub Issues**: Bug reports, feature requests
- **GitHub Discussions**: Questions, ideas, show-and-tell
- **Email**: zyntherdev@gmail.com for private matters

### Recognition

Contributors are recognized in:
- Release notes
- AUTHORS file
- Project documentation

---

**Thank you for contributing to vAGI!**

Your contributions help advance AGI research for everyone.
