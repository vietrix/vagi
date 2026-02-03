"""Extended configuration for AGI model."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class AGIConfig:
    """Configuration for full AGI model."""
    
    vocab_size: int = 50000
    hidden_size: int = 1024
    n_layers: int = 24
    n_heads: int = 16
    n_kv_heads: int = 16
    mlp_ratio: float = 4.0
    max_seq_len: int = 2048
    obs_dim: int = 256
    obs_tokens: int = 8
    action_dim: int = 256
    memory_slots: int = 16
    dropout: float = 0.1
    
    use_rotary: bool = True
    use_gqa: bool = True
    use_flash_attn: bool = False
    use_grad_checkpoint: bool = True
    
    use_task_embedding: bool = True
    task_vocab_size: int = 1000
    
    use_reflection: bool = True
    error_type_dim: int = 32
    
    use_budget_head: bool = True
    budget_max_horizon: int = 8
    budget_max_candidates: int = 16
    
    use_vision: bool = True
    vision_image_size: int = 224
    vision_patch_size: int = 16
    vision_channels: int = 3
    vision_embed_dim: int = 768
    vision_depth: int = 12
    vision_num_heads: int = 12
    
    use_world_pred: bool = True
    world_model_horizon: int = 4
    
    use_confidence: bool = True
    use_uncertainty: bool = True
    uncertainty_obs_scale: float = 0.1
    
    use_action_validity: bool = True
    action_validity_threshold: float = 0.5
    action_rejection_threshold: float = 0.2  # Reject actions when confidence below this

    ood_uncertainty_threshold: float = 2.0
    ood_trace_threshold: float = 0.1
    ood_policy: str = "fallback"
    
    memory_decay: float = 0.99
    memory_protect: bool = True
    memory_consolidate_every: int = 100
    
    use_special_tokens: bool = True
    
    use_language_modeling: bool = True
    use_masked_lm: bool = True
    mask_prob: float = 0.15
    
    use_knowledge_graph: bool = True
    num_entities: int = 10000
    num_relations: int = 100
    entity_embed_dim: int = 256
    
    use_semantic_memory: bool = True
    semantic_capacity: int = 10000
    
    use_episodic_memory: bool = True
    episodic_capacity: int = 1000
    episodic_sequence_length: int = 32
    
    use_abstract_reasoning: bool = True
    num_reasoning_variables: int = 10
    num_relation_layers: int = 2
    
    use_meta_learning: bool = True
    inner_lr: float = 0.01
    num_inner_steps: int = 5
    num_few_shot_examples: int = 5
    
    use_curriculum: bool = True
    num_curriculum_tasks: int = 100
    
    use_tool_use: bool = True
    num_tools: int = 50
    max_code_length: int = 512
    
    use_multimodal_fusion: bool = True
    fusion_dim: int = 768
    
    use_transfer_learning: bool = True
    shared_transfer_dim: int = 512
    
    # Continuous Learning
    use_continuous_learning: bool = True
    continuous_learning_buffer_size: int = 100000
    continuous_learning_batch_size: int = 32
    continuous_learning_update_freq: int = 10
    continuous_learning_alpha: float = 0.6  # Priority exponent
    continuous_learning_beta: float = 0.4  # Importance sampling

    # Online Learning (Real-time learning during inference)
    online_learning_enabled: bool = True
    online_learning_confidence_threshold: float = 0.5  # Learn when confidence < this
    online_learning_min_experiences: int = 4  # Min experiences before online update
    emergency_learning_threshold: float = 0.3  # Trigger emergency learning below this
    emergency_learning_lr_multiplier: float = 2.0  # LR boost for emergency learning
    online_learning_max_grad_norm: float = 1.0  # Gradient clipping for stability
    
    # Object-Centric Perception (Scene Graphs)
    use_scene_graphs: bool = True
    num_object_slots: int = 10
    object_dim: int = 128
    scene_graph_iterations: int = 3
    
    # Intrinsic Motivation
    use_intrinsic_motivation: bool = True
    intrinsic_reward_weight: float = 0.1
    curiosity_weight: float = 0.5
    novelty_weight: float = 0.3
    empowerment_weight: float = 0.2
    intrinsic_hidden_size: int = 256
    novelty_buffer_size: int = 1000
    
    # Program Synthesis
    use_program_synthesis: bool = True
    num_primitives: int = 20
    max_program_length: int = 10
    program_hidden_size: int = 256
    num_program_samples: int = 5
    
    # Grounded Language Understanding
    use_grounded_language: bool = True  # ENABLED - connects language to perception/action
    use_vqa: bool = True
    use_instruction_following: bool = True
    grounded_lang_hidden_size: int = 512
    max_instruction_length: int = 50

    # Meta-Cognition
    use_metacognition: bool = True  # ENABLED - self-awareness and reasoning about reasoning
    metacog_hidden_size: int = 256
    metacog_task_embedding_dim: int = 128
    metacog_capability_dim: int = 64
    metacog_num_capability_types: int = 20
    
    def __post_init__(self) -> None:
        """Validate configuration."""
        if self.use_confidence and not self.use_uncertainty:
            self.use_uncertainty = True
        self.validate()

    @property
    def head_dim(self) -> int:
        """Compute head dimension."""
        return self.hidden_size // self.n_heads

    def validate(self) -> None:
        """Validate all configuration parameters."""
        if self.vocab_size <= 0:
            raise ValueError("vocab_size must be > 0")
        if self.hidden_size <= 0:
            raise ValueError("hidden_size must be > 0")
        if self.n_layers <= 0:
            raise ValueError("n_layers must be > 0")
        if self.n_heads <= 0:
            raise ValueError("n_heads must be > 0")
        if self.hidden_size % self.n_heads != 0:
            raise ValueError("hidden_size must be divisible by n_heads")
        if self.n_kv_heads <= 0:
            raise ValueError("n_kv_heads must be > 0")
        if self.use_gqa and (self.n_heads % self.n_kv_heads != 0):
            raise ValueError("n_heads must be divisible by n_kv_heads when use_gqa is True")
        if self.mlp_ratio <= 0:
            raise ValueError("mlp_ratio must be > 0")
        if self.max_seq_len <= 0:
            raise ValueError("max_seq_len must be > 0")
        if self.obs_dim <= 0:
            raise ValueError("obs_dim must be > 0")
        if self.obs_tokens < 0:
            raise ValueError("obs_tokens must be >= 0")
        if self.action_dim <= 0:
            raise ValueError("action_dim must be > 0")
        if self.memory_slots < 0:
            raise ValueError("memory_slots must be >= 0")
        if self.world_model_horizon <= 0:
            raise ValueError("world_model_horizon must be > 0")
        if self.task_vocab_size <= 0:
            raise ValueError("task_vocab_size must be > 0")
        if self.error_type_dim <= 0:
            raise ValueError("error_type_dim must be > 0")
        if self.budget_max_horizon <= 0:
            raise ValueError("budget_max_horizon must be > 0")
        if self.budget_max_candidates <= 0:
            raise ValueError("budget_max_candidates must be > 0")
        if not (0.0 <= self.memory_decay <= 1.0):
            raise ValueError("memory_decay must be in [0, 1]")
        if self.memory_consolidate_every < 0:
            raise ValueError("memory_consolidate_every must be >= 0")
        if self.uncertainty_obs_scale < 0.0:
            raise ValueError("uncertainty_obs_scale must be >= 0")
        if not (0.0 <= self.action_validity_threshold <= 1.0):
            raise ValueError("action_validity_threshold must be in [0, 1]")
        if self.ood_uncertainty_threshold < 0.0:
            raise ValueError("ood_uncertainty_threshold must be >= 0")
        if self.ood_trace_threshold < 0.0:
            raise ValueError("ood_trace_threshold must be >= 0")
        if not (0.0 <= self.dropout < 1.0):
            raise ValueError("dropout must be in [0, 1)")


def load_agi_config() -> AGIConfig:
    """Load default AGI configuration."""
    return AGIConfig()


def load_agi_large_config() -> AGIConfig:
    """Load large AGI configuration."""
    return AGIConfig(
        vocab_size=100000,
        hidden_size=2048,
        n_layers=32,
        n_heads=32,
        n_kv_heads=8,
        max_seq_len=4096,
        obs_dim=512,
        obs_tokens=16,
        action_dim=512,
        memory_slots=32,
        vision_embed_dim=1024,
        vision_depth=24,
        semantic_capacity=50000,
        episodic_capacity=5000,
        num_tools=100,
    )


def load_agi_small_config() -> AGIConfig:
    """Load small AGI configuration for testing."""
    return AGIConfig(
        vocab_size=10000,
        hidden_size=512,
        n_layers=12,
        n_heads=8,
        n_kv_heads=8,
        max_seq_len=1024,
        obs_dim=128,
        obs_tokens=4,
        action_dim=128,
        memory_slots=8,
        vision_embed_dim=384,
        vision_depth=6,
        semantic_capacity=1000,
        episodic_capacity=100,
        num_tools=20,
    )


def load_agi_tiny_config() -> AGIConfig:
    """Load tiny AGI configuration for fast CPU training."""
    return AGIConfig(
        vocab_size=5000,
        hidden_size=128,
        n_layers=4,
        n_heads=4,
        n_kv_heads=4,
        mlp_ratio=2.0,
        max_seq_len=256,
        obs_dim=64,
        obs_tokens=2,
        action_dim=64,
        memory_slots=4,
        dropout=0.1,
        use_rotary=True,
        use_gqa=False,
        use_flash_attn=False,
        use_grad_checkpoint=False,
        use_task_embedding=False,
        use_reflection=False,
        use_budget_head=False,
        use_vision=False,
        use_world_pred=False,
        use_confidence=True,
        use_uncertainty=True,
        use_action_validity=False,
        use_special_tokens=False,
        use_language_modeling=True,
        use_masked_lm=False,
        use_knowledge_graph=False,
        use_semantic_memory=False,
        use_episodic_memory=False,
        use_abstract_reasoning=False,
        use_meta_learning=False,
        use_curriculum=False,
        use_tool_use=False,
        use_multimodal_fusion=False,
        use_transfer_learning=False,
        use_continuous_learning=False,
        online_learning_enabled=True,
        use_scene_graphs=False,
        use_intrinsic_motivation=False,
        use_program_synthesis=False,
        use_grounded_language=False,
        use_metacognition=True,
        metacog_hidden_size=64,
        metacog_task_embedding_dim=32,
        metacog_capability_dim=32,
    )
