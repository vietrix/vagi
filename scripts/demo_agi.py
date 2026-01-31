"""Demonstration of AGI capabilities."""

from __future__ import annotations

import torch
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))

from core.agi_config import load_agi_small_config
from core.agi_model import AGIModel 


def demo_language_understanding():
    """Demonstrate language understanding."""
    print("\n=== Language Understanding Demo ===")
    
    config = load_agi_small_config()
    model = AGIModel(config)
    
    input_ids = torch.randint(0, config.vocab_size, (1, 20))
    state = model.init_state(batch_size=1)
    
    outputs = model(input_ids=input_ids, state=state, mode="inference")
    
    print(f"Input shape: {input_ids.shape}")
    print(f"Output keys: {outputs.keys()}")
    print("Language understanding: OK")


def demo_few_shot_learning():
    """Demonstrate few-shot learning."""
    print("\n=== Few-Shot Learning Demo ===")
    
    config = load_agi_small_config()
    model = AGIModel(config)
    
    examples = [
        (torch.randn(config.hidden_size), torch.tensor([0])),
        (torch.randn(config.hidden_size), torch.tensor([1])),
        (torch.randn(config.hidden_size), torch.tensor([0])),
        (torch.randn(config.hidden_size), torch.tensor([1])),
    ]
    
    query = torch.randn(1, config.hidden_size)
    
    logits = model.learn_from_examples(examples, query)
    prediction = torch.argmax(logits, dim=-1)
    
    print(f"Number of examples: {len(examples)}")
    print(f"Query shape: {query.shape}")
    print(f"Prediction: {prediction.item()}")
    print("Few-shot learning: OK")


def demo_memory_retrieval():
    """Demonstrate hierarchical memory."""
    print("\n=== Memory Retrieval Demo ===")
    
    config = load_agi_small_config()
    model = AGIModel(config)
    
    query = torch.randn(1, config.hidden_size)
    
    if hasattr(model, "hierarchical_memory"):
        memory_output, memory_info = model.hierarchical_memory(query)
        
        print(f"Query shape: {query.shape}")
        print(f"Memory output shape: {memory_output.shape}")
        print(f"Routing weights: {memory_info['routing_weights']}")
        print("Memory retrieval: OK")
    else:
        print("Hierarchical memory not available")


def demo_abstract_reasoning():
    """Demonstrate abstract reasoning."""
    print("\n=== Abstract Reasoning Demo ===")
    
    config = load_agi_small_config()
    model = AGIModel(config)
    
    input_ids = torch.randint(0, config.vocab_size, (1, 10))
    obs = torch.randn(1, config.obs_dim)
    state = model.init_state(batch_size=1)
    
    outputs = model(
        input_ids=input_ids,
        obs=obs,
        state=state,
        mode="reasoning"
    )
    
    if "reasoning" in outputs:
        reasoning_outputs = outputs["reasoning"]
        print(f"Reasoning outputs: {reasoning_outputs.keys()}")
        print("Abstract reasoning: OK")
    else:
        print("Abstract reasoning not triggered")


def demo_tool_use():
    """Demonstrate tool use."""
    print("\n=== Tool Use Demo ===")
    
    config = load_agi_small_config()
    model = AGIModel(config)
    
    def simple_calculator(x: float, y: float) -> float:
        return x + y
    
    model.register_tool(
        name="calculator",
        function=simple_calculator,
        description="Add two numbers"
    )
    
    input_ids = torch.randint(0, config.vocab_size, (1, 15))
    obs = torch.randn(1, config.obs_dim)
    state = model.init_state(batch_size=1)
    
    outputs = model(
        input_ids=input_ids,
        obs=obs,
        state=state,
        mode="inference"
    )
    
    if "tool_use" in outputs:
        tool_info = outputs["tool_use"]
        print(f"Should use tool: {tool_info['should_use']}")
        if tool_info["should_use"]:
            print(f"Tool ID: {tool_info['tool_id']}")
        print("Tool use: OK")
    else:
        print("Tool use not triggered")


def demo_thinking_and_planning():
    """Demonstrate thinking and planning."""
    print("\n=== Thinking and Planning Demo ===")
    
    config = load_agi_small_config()
    model = AGIModel(config)
    
    input_ids = torch.randint(0, config.vocab_size, (1, 5))
    obs = torch.randn(1, config.obs_dim)
    state = model.init_state(batch_size=1)
    
    plan_output = model.think_and_act(
        input_ids=input_ids,
        obs=obs,
        state=state,
        horizon=3,
        num_candidates=4
    )
    
    print(f"Planning output keys: {plan_output.keys()}")
    print(f"Selected action: {plan_output.get('action', 'N/A')}")
    if "confidence" in plan_output:
        print(f"Confidence: {plan_output['confidence']:.4f}")
    print("Thinking and planning: OK")


def demo_curriculum_learning():
    """Demonstrate curriculum learning."""
    print("\n=== Curriculum Learning Demo ===")
    
    config = load_agi_small_config()
    model = AGIModel(config)
    
    student_state = torch.randn(config.hidden_size)
    
    for i in range(5):
        task_id = model.get_next_task(student_state)
        
        performance = 0.5 + i * 0.1
        
        model.update_curriculum(task_id, performance)
        
        print(f"Iteration {i+1}: Task {task_id}, Performance {performance:.2f}")
    
    print("Curriculum learning: OK")


def demo_multimodal():
    """Demonstrate multi-modal processing."""
    print("\n=== Multi-Modal Processing Demo ===")
    
    config = load_agi_small_config()
    model = AGIModel(config)
    
    image = torch.randn(1, 3, 224, 224)
    input_ids = torch.randint(0, config.vocab_size, (1, 10))
    state = model.init_state(batch_size=1)
    
    outputs = model(
        input_ids=input_ids,
        image=image,
        state=state,
        mode="inference"
    )
    
    if "vision_features" in outputs:
        print(f"Vision features shape: {outputs['vision_features'].shape}")
        print("Multi-modal processing: OK")
    else:
        print("Vision features not generated")


def demo_model_info():
    """Display model information."""
    print("\n=== Model Information ===")
    
    config = load_agi_small_config()
    model = AGIModel(config)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Configuration: Small")
    print(f"Vocabulary size: {config.vocab_size}")
    print(f"Hidden size: {config.hidden_size}")
    print(f"Number of layers: {config.n_layers}")
    print(f"Number of heads: {config.n_heads}")
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Memory slots: {config.memory_slots}")
    print(f"Semantic memory capacity: {config.semantic_capacity}")
    print(f"Episodic memory capacity: {config.episodic_capacity}")
    print(f"Number of tools: {config.num_tools}")
    
    print("\nEnabled features:")
    print(f"  Language modeling: {config.use_language_modeling}")
    print(f"  Knowledge graph: {config.use_knowledge_graph}")
    print(f"  Semantic memory: {config.use_semantic_memory}")
    print(f"  Episodic memory: {config.use_episodic_memory}")
    print(f"  Abstract reasoning: {config.use_abstract_reasoning}")
    print(f"  Meta-learning: {config.use_meta_learning}")
    print(f"  Curriculum: {config.use_curriculum}")
    print(f"  Tool use: {config.use_tool_use}")
    print(f"  Vision: {config.use_vision}")
    print(f"  Multi-modal fusion: {config.use_multimodal_fusion}")


def main():
    """Run all demonstrations."""
    print("=" * 60)
    print("vAGI - AGI Capabilities Demonstration")
    print("=" * 60)
    
    demo_model_info()
    demo_language_understanding()
    demo_few_shot_learning()
    demo_memory_retrieval()
    demo_abstract_reasoning()
    demo_thinking_and_planning()
    demo_tool_use()
    demo_curriculum_learning()
    demo_multimodal()
    
    print("\n" + "=" * 60)
    print("All demonstrations completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
