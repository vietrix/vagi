# vAGI - AGI Extension

This extension adds full AGI (Artificial General Intelligence) capabilities to the vAGI framework.

## New Components

### 1. Language Understanding (`core/language.py`)
- **BytePairTokenizer**: Real tokenization for natural language
- **TextEmbedding**: Positional encoding and token embeddings
- **NextTokenPredictor**: Autoregressive language modeling
- **MaskedLanguageModel**: BERT-style pre-training
- **SentenceEncoder**: Sentence-level representations

### 2. Knowledge System (`core/knowledge.py`)
- **KnowledgeGraph**: Graph-structured factual knowledge
- **SemanticMemory**: Long-term semantic memory with retrieval
- **EpisodicMemory**: Event sequence storage
- **HierarchicalMemory**: Unified working/semantic/episodic memory
- **ConceptEncoder**: Disentangled concept representations

### 3. Abstract Reasoning (`core/reasoning.py`)
- **RelationalReasoning**: Reasoning over object relations
- **CausalGraphLearner**: Learn causal structure from data
- **AnalogyMaker**: Make analogies between concepts
- **AbstractReasoner**: Unified abstract reasoning module
- **CounterfactualReasoner**: What-if scenario analysis

### 4. Meta-Learning (`core/metalearning.py`)
- **TaskEmbedding**: Embed tasks from few-shot examples
- **MAMLAdapter**: Model-Agnostic Meta-Learning
- **CurriculumScheduler**: Automatic curriculum generation
- **TransferLearner**: Cross-domain transfer
- **FewShotLearner**: Prototypical networks

### 5. Tool Use (`core/tools.py`)
- **ToolRegistry**: Register and manage tools
- **ToolSelector**: Select appropriate tool for context
- **APICallGenerator**: Generate API calls from natural language
- **ParameterExtractor**: Extract parameters for tool calls
- **CodeExecutor**: Generate and execute code
- **ToolUseController**: High-level tool orchestration

### 6. Advanced Vision (`core/vision.py`)
- **VisionTransformerEncoder**: ViT-style image encoding
- **CrossModalAttention**: Attend to vision from language
- **ImageTextAligner**: CLIP-style alignment
- **VideoEncoder**: Temporal video understanding
- **MultiModalEncoder**: Unified multi-modal fusion

### 7. AGI Model (`core/agi_model.py`)
- **AGIModel**: Full integration of all components
- Unified forward pass for all modalities
- High-level reasoning and action methods
- Few-shot learning interface
- Tool registration and curriculum management

## Quick Start

### Training

```bash
# Train small AGI model
python scripts/train_agi.py --config small --epochs 5 --batch-size 8

# Train default AGI model
python scripts/train_agi.py --config default --epochs 10 --batch-size 16

# Resume from checkpoint
python scripts/train_agi.py --resume-from checkpoints/agi/checkpoint_epoch5_step1000.pt
```

### Inference

```python
from core.agi_config import load_agi_small_config
from core.agi_model import AGIModel
import torch

# Load model
config = load_agi_small_config()
model = AGIModel(config)

# Initialize state
state = model.init_state(batch_size=1)

# Text input
input_ids = torch.randint(0, config.vocab_size, (1, 10))
obs = torch.randn(1, config.obs_dim)

# Forward pass
outputs = model(
    input_ids=input_ids,
    obs=obs,
    state=state,
    mode="inference"
)

# Think and act
action_output = model.think_and_act(
    input_ids=input_ids,
    obs=obs,
    state=state,
    horizon=4,
    num_candidates=8
)

print(f"Selected action: {action_output['action']}")
print(f"Confidence: {action_output['confidence']}")
```

### Few-Shot Learning

```python
# Provide examples
examples = [
    (torch.randn(config.hidden_size), torch.tensor([0])),
    (torch.randn(config.hidden_size), torch.tensor([1])),
    (torch.randn(config.hidden_size), torch.tensor([0])),
]

# Query
query = torch.randn(1, config.hidden_size)

# Predict from examples
logits = model.learn_from_examples(examples, query)
prediction = torch.argmax(logits, dim=-1)
```

### Tool Use

```python
# Register a tool
def calculator(a: float, b: float, op: str) -> float:
    if op == "add":
        return a + b
    elif op == "multiply":
        return a * b
    return 0.0

model.register_tool(
    name="calculator",
    function=calculator,
    description="Perform arithmetic operations"
)

# Model will automatically use tools when needed
outputs = model(
    input_ids=input_ids,
    obs=obs,
    state=state,
    mode="inference"
)

if outputs["tool_use"]["should_use"]:
    tool_id = outputs["tool_use"]["tool_id"]
    print(f"Model wants to use tool {tool_id}")
```

### Curriculum Learning

```python
# Update curriculum based on performance
for epoch in range(100):
    task_id = model.get_next_task(student_state)
    
    # Train on task
    performance = train_on_task(task_id)
    
    # Update curriculum
    model.update_curriculum(task_id, performance)
```

## Configuration

### Model Sizes

- **Small**: 512 hidden, 12 layers, ~50M parameters
- **Default**: 1024 hidden, 24 layers, ~200M parameters  
- **Large**: 2048 hidden, 32 layers, ~1B parameters

### Key Features

Enable/disable features in config:

```python
from core.agi_config import AGIConfig

config = AGIConfig(
    use_language_modeling=True,     # Language understanding
    use_knowledge_graph=True,        # Factual knowledge
    use_semantic_memory=True,        # Long-term memory
    use_episodic_memory=True,        # Event memory
    use_abstract_reasoning=True,     # Abstract thinking
    use_meta_learning=True,          # Few-shot learning
    use_curriculum=True,             # Auto curriculum
    use_tool_use=True,               # External tools
    use_vision=True,                 # Vision encoder
    use_multimodal_fusion=True,      # Multi-modal
    use_transfer_learning=True,      # Transfer across domains
)
```

## Architecture Overview

```
Input (Text/Image/Obs)
    |
    v
[Language Embedding / Vision Encoder]
    |
    v
[Transformer Backbone] <--> [Hierarchical Memory]
    |                            |
    v                            v
[Knowledge Graph] <--> [Semantic Memory]
    |                            |
    v                            v
[Abstract Reasoner] <--> [Episodic Memory]
    |
    v
[Meta-Learning] --> [Task Adaptation]
    |
    v
[Tool Controller] --> [Action/Tool Selection]
    |
    v
Output (Action/Text/Tool Call)
```

## Key Differences from Base vAGI

| Feature | Base vAGI | AGI Extension |
|---------|-----------|---------------|
| Vocabulary | 128-256 tokens | 10K-100K tokens |
| Language | No real understanding | Full NLP capabilities |
| Memory | 4-8 slots | Hierarchical (semantic + episodic) |
| Knowledge | None | Knowledge graph + facts |
| Reasoning | Planning only | Abstract + causal + analogies |
| Learning | Task-specific | Meta-learning + few-shot |
| Vision | Toy 2-layer CNN | Vision Transformer |
| Tools | No support | Full tool use system |
| Scale | ~20K-100K params | 50M-1B+ params |

## Training Data Requirements

For full AGI capabilities:

1. **Language Corpus**: 
   - Books, Wikipedia, web text
   - Minimum: 1GB text
   - Recommended: 100GB+

2. **Visual Data**:
   - ImageNet, COCO, etc.
   - Image-text pairs for alignment

3. **Task Data**:
   - Multi-task datasets
   - Few-shot benchmarks

4. **Tool Use**:
   - Code execution traces
   - API call examples

## Performance Notes

- Small model: Can run on CPU
- Default model: Requires GPU (8GB+ VRAM)
- Large model: Requires 16GB+ VRAM or multi-GPU

## Limitations

- Still requires large-scale pre-training for full capabilities
- Not production-ready without proper data
- Scaling laws still apply (bigger is better)
- No guarantees on safety or alignment

## Future Work

- [ ] Add reinforcement learning from human feedback
- [ ] Implement constitutional AI
- [ ] Add retrieval-augmented generation
- [ ] Multi-GPU training support
- [ ] Quantization and compression
- [ ] Production deployment tools

## Citation

If you use this AGI extension, please cite:

```
@software{vagi_agi_extension,
  title={vAGI: Model-Centric AGI Framework},
  author={Your Name},
  year={2026},
  url={https://github.com/yourusername/vagi}
}
```

## License

Same as base vAGI project.
