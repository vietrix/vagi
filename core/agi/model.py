"""Main AGI model integrating all components."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import torch
from torch import nn
from torch.nn import functional as F

from .config import AGIConfig
from ..base.model import VAGICore
from ..base.config import VAGIConfig
from ..nlp import BytePairTokenizer, TextEmbedding, NextTokenPredictor, MaskedLanguageModel
from ..knowledge import HierarchicalMemory, KnowledgeGraph, ConceptEncoder
from ..reasoning import AbstractReasoner, CounterfactualReasoner
from ..learning import TaskEmbedding, CurriculumScheduler, TransferLearner, FewShotLearner
from ..interaction import ToolRegistry, ToolUseController
from ..perception import VisionTransformerEncoder, MultiModalEncoder, ImageTextAligner

logger = logging.getLogger(__name__)


class AGIModel(nn.Module):
    """Full AGI model with all advanced capabilities."""

    def __init__(self, cfg: AGIConfig) -> None:
        super().__init__()
        self.cfg = cfg
        
        self.core = VAGICore(self._get_core_config(cfg))
        
        if cfg.use_language_modeling:
            self.text_embedding = TextEmbedding(
                vocab_size=cfg.vocab_size,
                hidden_size=cfg.hidden_size,
                max_seq_len=cfg.max_seq_len,
                dropout=cfg.dropout
            )
            self.language_head = NextTokenPredictor(
                hidden_size=cfg.hidden_size,
                vocab_size=cfg.vocab_size
            )
            
            if cfg.use_masked_lm:
                self.masked_lm = MaskedLanguageModel(
                    hidden_size=cfg.hidden_size,
                    vocab_size=cfg.vocab_size,
                    mask_prob=cfg.mask_prob
                )
        
        if cfg.use_knowledge_graph or cfg.use_semantic_memory or cfg.use_episodic_memory:
            self.hierarchical_memory = HierarchicalMemory(
                working_memory_slots=cfg.memory_slots,
                semantic_capacity=cfg.semantic_capacity if cfg.use_semantic_memory else 100,
                episodic_capacity=cfg.episodic_capacity if cfg.use_episodic_memory else 10,
                hidden_size=cfg.hidden_size
            )
        
        if cfg.use_knowledge_graph:
            self.knowledge_graph = KnowledgeGraph(
                num_entities=cfg.num_entities,
                num_relations=cfg.num_relations,
                embedding_dim=cfg.entity_embed_dim,
                hidden_size=cfg.hidden_size
            )
        
        if cfg.use_abstract_reasoning:
            self.abstract_reasoner = AbstractReasoner(
                hidden_size=cfg.hidden_size,
                num_variables=cfg.num_reasoning_variables,
                num_relation_layers=cfg.num_relation_layers
            )
            
            self.counterfactual_reasoner = CounterfactualReasoner(
                world_model=self.core,
                hidden_size=cfg.hidden_size
            )
        
        if cfg.use_meta_learning:
            self.task_embedding = TaskEmbedding(
                input_dim=cfg.obs_dim,
                output_dim=cfg.hidden_size,
                num_examples=cfg.num_few_shot_examples
            )
            
            self.few_shot_learner = FewShotLearner(
                input_dim=cfg.hidden_size,
                hidden_size=cfg.hidden_size,
                num_classes=cfg.action_dim
            )
        
        if cfg.use_curriculum:
            self.curriculum_scheduler = CurriculumScheduler(
                num_tasks=cfg.num_curriculum_tasks,
                hidden_size=cfg.hidden_size
            )
        
        if cfg.use_transfer_learning:
            self.transfer_learner = TransferLearner(
                source_dim=cfg.hidden_size,
                target_dim=cfg.hidden_size,
                shared_dim=cfg.shared_transfer_dim
            )
        
        if cfg.use_tool_use:
            self.tool_registry = ToolRegistry()
            self.tool_controller = ToolUseController(
                hidden_size=cfg.hidden_size,
                num_tools=cfg.num_tools,
                tool_registry=self.tool_registry
            )
        
        if cfg.use_vision:
            self.vision_encoder = VisionTransformerEncoder(
                image_size=cfg.vision_image_size,
                patch_size=cfg.vision_patch_size,
                in_channels=cfg.vision_channels,
                embed_dim=cfg.vision_embed_dim,
                depth=cfg.vision_depth,
                num_heads=cfg.vision_num_heads,
                dropout=cfg.dropout
            )
            
            if cfg.use_multimodal_fusion:
                self.image_text_aligner = ImageTextAligner(
                    vision_dim=cfg.vision_embed_dim,
                    text_dim=cfg.hidden_size,
                    shared_dim=cfg.fusion_dim
                )
                if cfg.use_language_modeling:
                    self.multimodal_encoder = MultiModalEncoder(
                        vision_encoder=self.vision_encoder,
                        text_encoder=self.text_embedding,
                        fusion_dim=cfg.fusion_dim,
                    )
        
        self.concept_encoder = ConceptEncoder(
            input_dim=cfg.hidden_size,
            concept_dim=cfg.hidden_size // 10,
            num_concepts=10
        )
        
        # 1.12 Memory Projection - Multi-layer with non-linearity (was simple Linear)
        self.memory_projection = nn.Sequential(
            nn.Linear(cfg.hidden_size, cfg.hidden_size * 2),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.hidden_size * 2, cfg.hidden_size),
            nn.LayerNorm(cfg.hidden_size)
        )
        self.kg_projection = nn.Linear(cfg.entity_embed_dim, cfg.hidden_size)
        self.reasoning_projection = nn.Linear(cfg.hidden_size, cfg.hidden_size)
        self.augmented_action_head = nn.Linear(cfg.hidden_size, cfg.action_dim)
        self.augmented_value_head = nn.Linear(cfg.hidden_size, 1)
        self.vision_fusion_weight = nn.Parameter(torch.tensor(0.5))
        self._tokenizer: Optional[BytePairTokenizer] = None

        # Learned Context Gates - dynamically weight context contributions
        # Each gate learns attention weights based on query (hidden_pooled)
        self.context_gate_query = nn.Linear(cfg.hidden_size, cfg.hidden_size // 4)
        self.context_gate_key = nn.Linear(cfg.hidden_size, cfg.hidden_size // 4)
        self.context_gate_temperature = nn.Parameter(torch.tensor(1.0))

        # Scene graph projection (pre-computed to avoid dynamic layer creation)
        scene_embed_dim = cfg.num_object_slots * cfg.object_dim
        self.scene_projection = nn.Linear(scene_embed_dim, cfg.hidden_size)

        # Learnable task embedding for metacognition (fixes gradient flow)
        self.task_embedding_layer = nn.Sequential(
            nn.Linear(cfg.hidden_size, cfg.metacog_task_embedding_dim),
            nn.Tanh()
        )

        # 1.10 Entity/Relation extractors with entropy-based confidence filtering
        if cfg.use_knowledge_graph:
            self.entity_extractor = nn.Linear(cfg.hidden_size, cfg.num_entities)
            self.relation_extractor = nn.Linear(cfg.hidden_size, cfg.num_relations)
            # Entropy threshold for filtering low-confidence extractions
            self.entity_entropy_threshold = nn.Parameter(torch.tensor(0.5), requires_grad=False)
            self.relation_entropy_threshold = nn.Parameter(torch.tensor(0.5), requires_grad=False)

        # 1.13 Tool Use Gradient Flow - Differentiable tool interface with straight-through estimator
        if cfg.use_tool_use:
            self.tool_embedding = nn.Embedding(cfg.num_tools, cfg.hidden_size)
            self.tool_param_encoder = nn.Linear(cfg.hidden_size, cfg.hidden_size)
            self.tool_result_decoder = nn.Linear(cfg.hidden_size, cfg.hidden_size)

        # 1.15 Unified initialization flag for continuous learning
        self._learners_initialized = False
        self._pending_optimizer = None

        # Setup extended AGI modules
        self.setup_extended_modules()

    def _initialize_learners(self, optimizer: torch.optim.Optimizer) -> None:
        """1.15 Unified initialization for all learners requiring optimizer."""
        if self._learners_initialized:
            return

        # Setup online learner
        if hasattr(self, '_online_learning_config') and self._online_learning_config is not None:
            from ..training.online_learner import OnlineLearner
            self._online_learner = OnlineLearner(
                model=self,
                optimizer=optimizer,
                config=self._online_learning_config
            )

        # Setup continuous learner
        if hasattr(self, 'continuous_learning_config') and self.continuous_learning_config is not None:
            from ..training import ContinuousLearner
            self._continuous_learner = ContinuousLearner(
                model=self,
                optimizer=optimizer,
                config=self.continuous_learning_config
            )

        self._learners_initialized = True
        logger.info("All learners initialized with optimizer")

    def setup_extended_modules(self) -> None:
        """Initialize all extended AGI modules."""
        cfg = self.cfg
        
        # Continuous Learning Setup
        if cfg.use_continuous_learning:
            from ..training import ContinuousLearningConfig
            cl_config = ContinuousLearningConfig(
                buffer_size=cfg.continuous_learning_buffer_size,
                batch_size=cfg.continuous_learning_batch_size,
                update_frequency=cfg.continuous_learning_update_freq
            )
            self.continuous_learning_config = cl_config
            self._continuous_learner = None  # Will be set when optimizer is available
        
        # Scene Graphs
        if cfg.use_scene_graphs:
            from ..perception import SceneGraphBuilder, GroundedWorldModel
            self.scene_graph_builder = SceneGraphBuilder(
                obs_dim=cfg.obs_dim,
                object_dim=cfg.object_dim,
                max_objects=cfg.num_object_slots
            )
            if cfg.use_world_pred:
                self.grounded_world_model = GroundedWorldModel(
                    obs_dim=cfg.obs_dim,
                    action_dim=cfg.action_dim,
                    object_dim=cfg.object_dim,
                    max_objects=cfg.num_object_slots
                )
        
        # Intrinsic Motivation 
        if cfg.use_intrinsic_motivation:
            from ..planning import IntrinsicMotivationSystem, IntrinsicRewardConfig
            intrinsic_config = IntrinsicRewardConfig(
                curiosity_weight=cfg.curiosity_weight,
                novelty_weight=cfg.novelty_weight,
                empowerment_weight=cfg.empowerment_weight
            )
            self.intrinsic_motivation = IntrinsicMotivationSystem(
                state_dim=cfg.obs_dim,
                action_dim=cfg.action_dim,
                config=intrinsic_config
            )
        
        # Program Synthesis
        if cfg.use_program_synthesis:
            from ..reasoning import ProgramSynthesizer
            # Create synthesizer (uses built-in DSL)
            self.program_synthesizer = ProgramSynthesizer(
                example_dim=cfg.hidden_size,
                max_program_length=cfg.max_program_length
            )
        
        # Grounded Language - connects language to perception and action
        if cfg.use_grounded_language and cfg.use_vision and cfg.use_language_modeling:
            from ..nlp.grounded_language import GroundedLanguageModel
            self.grounded_language = GroundedLanguageModel(
                vision_encoder=self.vision_encoder,
                text_encoder=self.text_embedding,
                action_space_dim=cfg.action_dim,
                vocab_size=cfg.vocab_size,
                hidden_size=cfg.grounded_lang_hidden_size
            )

        # Meta-Cognition - self-awareness and reasoning about reasoning
        if cfg.use_metacognition:
            from ..learning.metacognition import MetaCognition
            self.metacognition = MetaCognition(
                hidden_size=cfg.metacog_hidden_size,
                task_embedding_dim=cfg.metacog_task_embedding_dim
            )

        # Online Learning - real-time learning during inference
        if cfg.online_learning_enabled:
            from ..training.online_learner import OnlineLearner, OnlineLearningConfig
            online_config = OnlineLearningConfig(
                confidence_threshold=cfg.online_learning_confidence_threshold,
                emergency_threshold=cfg.emergency_learning_threshold,
                emergency_lr_multiplier=cfg.emergency_learning_lr_multiplier,
                max_grad_norm=cfg.online_learning_max_grad_norm,
                min_experiences=cfg.online_learning_min_experiences
            )
            self._online_learning_config = online_config
            self._online_learner = None  # Will be set when optimizer is available

    def setup_online_learner(self, optimizer: torch.optim.Optimizer) -> None:
        """Setup online learner with optimizer. Call after model initialization.

        1.8 Auto-setup: This is now also called automatically from train() when
        optimizer is first used.
        """
        # 1.15 Use unified initialization
        self._initialize_learners(optimizer)

    def setup_continuous_learner(self, optimizer: torch.optim.Optimizer) -> None:
        """Setup continuous learner with optimizer. Call after model initialization.

        1.8 Auto-setup: This is now also called automatically from train() when
        optimizer is first used.
        """
        # 1.15 Use unified initialization
        self._initialize_learners(optimizer)

    def train(self, mode: bool = True):
        """Override train() to auto-setup learners when optimizer becomes available.

        1.8 Online Learner Auto Setup - Auto-setup when optimizer available.
        """
        result = super().train(mode)

        # If we have a pending optimizer and learners aren't initialized, do it now
        if mode and self._pending_optimizer is not None and not self._learners_initialized:
            self._initialize_learners(self._pending_optimizer)
            self._pending_optimizer = None

        return result

    def set_optimizer(self, optimizer: torch.optim.Optimizer) -> None:
        """Set optimizer for auto-initialization of learners.

        1.8 Online Learner Auto Setup - Store optimizer for deferred initialization.
        """
        if not self._learners_initialized:
            self._pending_optimizer = optimizer
            # If already in training mode, initialize immediately
            if self.training:
                self._initialize_learners(optimizer)

    def _get_core_config(self, cfg: AGIConfig):
        """Extract VAGICore config from AGI config."""
        from ..base.config import VAGIConfig
        return VAGIConfig(
            vocab_size=cfg.vocab_size,
            hidden_size=cfg.hidden_size,
            n_layers=cfg.n_layers,
            n_heads=cfg.n_heads,
            n_kv_heads=cfg.n_kv_heads,
            mlp_ratio=cfg.mlp_ratio,
            max_seq_len=cfg.max_seq_len,
            obs_dim=cfg.obs_dim,
            obs_tokens=cfg.obs_tokens,
            action_dim=cfg.action_dim,
            memory_slots=cfg.memory_slots,
            dropout=cfg.dropout,
            use_rotary=cfg.use_rotary,
            use_gqa=cfg.use_gqa,
            use_flash_attn=cfg.use_flash_attn,
            use_grad_checkpoint=cfg.use_grad_checkpoint,
            use_task_embedding=cfg.use_task_embedding,
            task_vocab_size=cfg.task_vocab_size,
            use_reflection=cfg.use_reflection,
            error_type_dim=cfg.error_type_dim,
            use_budget_head=cfg.use_budget_head,
            budget_max_horizon=cfg.budget_max_horizon,
            budget_max_candidates=cfg.budget_max_candidates,
            use_world_pred=cfg.use_world_pred,
            world_model_horizon=cfg.world_model_horizon,
            use_confidence=cfg.use_confidence,
            use_uncertainty=cfg.use_uncertainty,
            uncertainty_obs_scale=cfg.uncertainty_obs_scale,
            use_action_validity=cfg.use_action_validity,
            action_validity_threshold=cfg.action_validity_threshold,
            ood_uncertainty_threshold=cfg.ood_uncertainty_threshold,
            ood_trace_threshold=cfg.ood_trace_threshold,
            ood_policy=cfg.ood_policy,
            memory_decay=cfg.memory_decay,
            memory_protect=cfg.memory_protect,
            memory_consolidate_every=cfg.memory_consolidate_every,
            use_special_tokens=cfg.use_special_tokens,
        )

    def _resolve_input_ids(
        self,
        input_ids: Optional[torch.Tensor],
        *,
        obs: Optional[torch.Tensor],
        image: Optional[torch.Tensor],
        text: Optional[str],
        state: Optional[Any],
    ) -> torch.Tensor:
        """Ensure input_ids exists for core forward calls."""
        if input_ids is not None:
            if input_ids.dtype != torch.long:
                input_ids = input_ids.long()
            return input_ids

        batch_size = None
        device = None
        if obs is not None:
            batch_size = obs.size(0)
            device = obs.device
        elif image is not None:
            batch_size = image.size(0)
            device = image.device
        elif state is not None and hasattr(state, "mem"):
            batch_size = state.mem.size(0)
            device = state.mem.device
        else:
            device = next(self.parameters()).device

        if text is not None:
            if self._tokenizer is None:
                self._tokenizer = BytePairTokenizer(vocab_size=self.cfg.vocab_size)
            token_ids = self._tokenizer.encode(text, max_length=self.cfg.max_seq_len)
            if not token_ids:
                token_ids = [0]
            ids = torch.tensor(token_ids, dtype=torch.long, device=device).unsqueeze(0)
            if batch_size is not None and batch_size > 1:
                ids = ids.repeat(batch_size, 1)
            return ids

        if batch_size is None:
            raise ValueError("input_ids, obs, image, state, or text is required")

        return torch.zeros((batch_size, 1), dtype=torch.long, device=device)

    # =========================================================================
    # 1.1 Forward Pass Refactored - Separate methods for each modality
    # =========================================================================

    def _process_language(
        self,
        input_ids: torch.Tensor,
        hidden: torch.Tensor,
        hidden_pooled: torch.Tensor,
        labels: Optional[torch.Tensor],
        mode: str,
        outputs: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Process language modality.

        1.1 Refactored from forward() for cleaner separation of concerns.
        """
        if not self.cfg.use_language_modeling:
            return outputs

        # Language head processing is done in loss computation section
        # This method handles language-specific feature extraction

        if hasattr(self, "masked_lm") and labels is not None and mode == "train":
            # MLM features are processed during loss computation
            outputs["mlm_enabled"] = True

        return outputs

    def _process_vision(
        self,
        image: Optional[torch.Tensor],
        obs: Optional[torch.Tensor],
        outputs: Dict[str, Any]
    ) -> Tuple[Optional[torch.Tensor], Dict[str, Any]]:
        """Process vision modality with fusion weight clamping.

        1.1 Refactored from forward() for cleaner separation.
        1.2 Vision Fusion Weight - Add constraint via torch.clamp.
        """
        if image is None or not self.cfg.use_vision:
            return obs, outputs

        vision_features = self.vision_encoder(image)
        outputs["vision_features"] = vision_features

        if self.cfg.use_multimodal_fusion and hasattr(self, "multimodal_encoder"):
            # Pass pre-computed vision_features to avoid redundant encoding
            fused_features = self.multimodal_encoder(
                image=None,  # Don't re-encode
                text=None,
                vision_features=vision_features  # Use already-encoded features
            )

            if fused_features.dim() == 3:
                obs_from_vision = fused_features.mean(dim=1)
            else:
                obs_from_vision = fused_features
        else:
            if vision_features.dim() == 3:
                obs_from_vision = vision_features[:, 0, :]
            else:
                obs_from_vision = vision_features

        if obs is None:
            obs = obs_from_vision
        elif obs_from_vision.size(-1) == obs.size(-1):
            # 1.2 Vision Fusion Weight - Clamp to [0, 1] for valid interpolation
            clamped_weight = torch.clamp(self.vision_fusion_weight, 0.0, 1.0)
            obs = obs * (1 - clamped_weight) + obs_from_vision * clamped_weight
            outputs["vision_fusion_weight_clamped"] = clamped_weight.item()

        return obs, outputs

    def _process_reasoning(
        self,
        hidden_pooled: torch.Tensor,
        hidden: torch.Tensor,
        outputs: Dict[str, Any],
        device: torch.device
    ) -> Tuple[Optional[torch.Tensor], Dict[str, Any]]:
        """Process abstract reasoning modality.

        1.1 Refactored from forward() for cleaner separation.
        """
        reasoning_contribution = None

        if not self.cfg.use_abstract_reasoning:
            return reasoning_contribution, outputs

        reasoning_outputs = self.abstract_reasoner(
            query=hidden_pooled,
            context=hidden if hidden.dim() == 3 else hidden.unsqueeze(1),
            mode="auto"
        )
        outputs["reasoning"] = reasoning_outputs

        if "relational" in reasoning_outputs:
            relational_features = reasoning_outputs["relational"]
            if relational_features.dim() == 3:
                relational_pooled = relational_features.mean(dim=1)
            else:
                relational_pooled = relational_features

            reasoning_contribution = self.reasoning_projection(relational_pooled)

        # Counterfactual reasoning: "what if we took different action?"
        if hasattr(self, "counterfactual_reasoner") and "action_logits" in outputs:
            action_probs = F.softmax(outputs["action_logits"], dim=-1)
            top_actions = action_probs.topk(min(3, action_probs.size(-1)), dim=-1)

            counterfactual_outcomes = []
            for k in range(top_actions.values.size(-1)):
                intervention = torch.zeros_like(action_probs)
                intervention.scatter_(-1, top_actions.indices[:, k:k+1], 1.0)

                cf_state = self.counterfactual_reasoner.generate_counterfactual(
                    factual_state=hidden_pooled,
                    intervention=intervention
                )

                comparison = self.counterfactual_reasoner.compare_outcomes(
                    factual=hidden_pooled,
                    counterfactual=cf_state
                )
                counterfactual_outcomes.append({
                    "action_idx": top_actions.indices[:, k],
                    "counterfactual_state": cf_state,
                    "outcome_diff": comparison
                })

            outputs["counterfactual_analysis"] = counterfactual_outcomes

            if counterfactual_outcomes:
                outcome_diffs = torch.stack([cf["outcome_diff"] for cf in counterfactual_outcomes], dim=-1)
                best_cf_idx = outcome_diffs.argmax(dim=-1)
                outputs["recommended_action"] = top_actions.indices.gather(-1, best_cf_idx.unsqueeze(-1))

        return reasoning_contribution, outputs

    def _process_memory(
        self,
        hidden_pooled: torch.Tensor,
        entities: Optional[torch.Tensor],
        relations: Optional[torch.Tensor],
        outputs: Dict[str, Any]
    ) -> Tuple[List[torch.Tensor], Dict[str, Any]]:
        """Process memory and knowledge graph modalities.

        1.1 Refactored from forward() for cleaner separation.
        1.10 Entity/Relation Extraction - Add entropy-based confidence filtering.
        """
        context_additions = []

        # Hierarchical memory
        if hasattr(self, "hierarchical_memory"):
            memory_output, memory_info = self.hierarchical_memory(hidden_pooled)
            outputs["memory_retrieval"] = memory_output
            outputs["memory_info"] = memory_info

            memory_contribution = self.memory_projection(memory_output)
            context_additions.append(memory_contribution)

        # Knowledge graph
        if self.cfg.use_knowledge_graph and hasattr(self, "knowledge_graph"):
            # 1.10 Extract entities with entropy-based confidence filtering
            if entities is None and hasattr(self, "entity_extractor"):
                entity_logits = self.entity_extractor(hidden_pooled)
                entity_probs = torch.softmax(entity_logits, dim=-1)

                # Compute entropy for confidence filtering
                entity_entropy = -torch.sum(entity_probs * torch.log(entity_probs + 1e-10), dim=-1)
                max_entropy = torch.log(torch.tensor(entity_probs.size(-1), dtype=torch.float, device=entity_probs.device))
                normalized_entropy = entity_entropy / max_entropy

                # Filter low-confidence extractions (high entropy = low confidence)
                confidence_mask = normalized_entropy < self.entity_entropy_threshold
                outputs["entity_confidence"] = 1.0 - normalized_entropy
                outputs["entity_confidence_mask"] = confidence_mask

                entities = entity_probs

            if relations is None and hasattr(self, "relation_extractor"):
                relation_logits = self.relation_extractor(hidden_pooled)
                relation_probs = torch.softmax(relation_logits, dim=-1)

                # Compute entropy for confidence filtering
                relation_entropy = -torch.sum(relation_probs * torch.log(relation_probs + 1e-10), dim=-1)
                max_entropy = torch.log(torch.tensor(relation_probs.size(-1), dtype=torch.float, device=relation_probs.device))
                normalized_entropy = relation_entropy / max_entropy

                # Filter low-confidence extractions
                confidence_mask = normalized_entropy < self.relation_entropy_threshold
                outputs["relation_confidence"] = 1.0 - normalized_entropy
                outputs["relation_confidence_mask"] = confidence_mask

                relations = relation_probs

            if entities is not None:
                # Handle different entity input formats
                if entities.dim() == 1:
                    kg_embeddings = self.knowledge_graph.entity_embeddings(entities)
                    kg_context = kg_embeddings
                elif entities.dim() == 2 and entities.size(-1) > self.cfg.entity_embed_dim:
                    top_k = min(5, entities.size(-1))
                    entity_scores, entity_indices = entities.topk(top_k, dim=-1)
                    kg_embeddings = self.knowledge_graph.entity_embeddings(entity_indices)
                    kg_context = (kg_embeddings * entity_scores.unsqueeze(-1)).sum(dim=1)
                else:
                    kg_context = entities

                kg_contribution = self.kg_projection(kg_context)
                context_additions.append(kg_contribution)

                outputs["knowledge_context"] = kg_context
                if entities.dim() == 1:
                    outputs["extracted_entities"] = entities
                outputs["extracted_relations"] = relations

        return context_additions, outputs

    def _process_action(
        self,
        hidden_pooled: torch.Tensor,
        augmented_hidden: Optional[torch.Tensor],
        context_additions: List[torch.Tensor],
        outputs: Dict[str, Any],
        mode: str
    ) -> Dict[str, Any]:
        """Process action/tool use modality.

        1.1 Refactored from forward() for cleaner separation.
        1.13 Tool Use Gradient Flow - Implement differentiable tool interface.
        """
        if not self.cfg.use_tool_use or mode not in ["train", "inference"]:
            return outputs

        context = augmented_hidden if augmented_hidden is not None else hidden_pooled

        should_use_tool, tool_id, tool_params = self.tool_controller(context)

        # 1.13 Differentiable tool interface with straight-through estimator
        if hasattr(self, 'tool_embedding'):
            # Get soft tool selection (for gradient flow)
            tool_logits = None
            if hasattr(self.tool_controller, 'tool_selector'):
                selector_output = self.tool_controller.tool_selector(context)
                # Handle case where selector returns tuple (logits, other)
                if isinstance(selector_output, tuple):
                    tool_logits = selector_output[0] if isinstance(selector_output[0], torch.Tensor) else None
                elif isinstance(selector_output, torch.Tensor):
                    tool_logits = selector_output

            # Only proceed if tool_logits is a float tensor (not indices)
            if (tool_logits is not None and
                isinstance(tool_logits, torch.Tensor) and
                tool_logits.dtype in (torch.float32, torch.float16, torch.bfloat16, torch.float64)):
                # Soft tool selection for gradient flow
                tool_probs = F.softmax(tool_logits, dim=-1)

                # Straight-through estimator: hard selection in forward, soft in backward
                if self.training:
                    # Hard selection
                    hard_tool_id = tool_probs.argmax(dim=-1)
                    # Straight-through: use hard in forward, but gradient flows through soft
                    tool_one_hot = F.one_hot(hard_tool_id, num_classes=tool_probs.size(-1)).float()
                    # Straight-through trick: detach hard, add soft gradient
                    tool_selection = tool_one_hot - tool_probs.detach() + tool_probs

                    # Get differentiable tool embedding
                    tool_embed = torch.matmul(tool_selection, self.tool_embedding.weight)
                else:
                    # Inference: just use hard selection
                    tool_embed = self.tool_embedding(tool_id if isinstance(tool_id, torch.Tensor) else torch.tensor([tool_id], device=context.device))
                    if tool_embed.dim() == 3:
                        tool_embed = tool_embed.squeeze(1)

                # Encode tool parameters for gradient flow
                param_encoding = self.tool_param_encoder(context)

                # Differentiable tool result (placeholder - actual tool execution is non-differentiable)
                # This provides a gradient pathway for learning tool selection
                differentiable_result = self.tool_result_decoder(tool_embed + param_encoding)

                outputs["tool_use"] = {
                    "should_use": should_use_tool,
                    "tool_id": tool_id,
                    "params": tool_params,
                    "context": context,
                    "tool_embedding": tool_embed,
                    "differentiable_result": differentiable_result,
                    "tool_probs": tool_probs if self.training else None
                }
            else:
                outputs["tool_use"] = {
                    "should_use": should_use_tool,
                    "tool_id": tool_id,
                    "params": tool_params,
                    "context": context
                }
        else:
            outputs["tool_use"] = {
                "should_use": should_use_tool,
                "tool_id": tool_id,
                "params": tool_params,
                "context": context
            }

        return outputs

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        obs: Optional[torch.Tensor] = None,
        image: Optional[torch.Tensor] = None,
        text: Optional[str] = None,
        state: Optional[Any] = None,
        task_ids: Optional[torch.Tensor] = None,
        mode: str = "train",
        entities: Optional[torch.Tensor] = None,
        relations: Optional[torch.Tensor] = None,
        return_loss: bool = False,
        labels: Optional[torch.Tensor] = None,
        targets: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Unified forward pass integrating ALL components with gradient flow."""
        input_ids = self._resolve_input_ids(
            input_ids,
            obs=obs,
            image=image,
            text=text,
            state=state,
        )
        batch_size = input_ids.size(0)
        device = input_ids.device
        
        outputs = {}
        augmented_hidden = None
        
        # 1.1 Vision processing via refactored helper method
        # 1.2 Vision fusion weight clamping is handled inside _process_vision
        obs, outputs = self._process_vision(image, obs, outputs)
        
        core_outputs = self.core.forward(
            input_ids=input_ids,
            obs=obs,
            state=state,
            task_ids=task_ids,
            image=image,
            return_hidden=True,
            labels=labels if not self.cfg.use_language_modeling else None,
            targets=targets,
            return_loss=return_loss and not self.cfg.use_language_modeling,
            **kwargs
        )
        outputs.update(core_outputs)
        
        hidden = core_outputs.get("hidden")
        if hidden is None:
            return outputs
        
        if hidden.dim() == 3:
            hidden_pooled = hidden[:, -1, :]
        else:
            hidden_pooled = hidden

        # 1.1 Language processing via refactored helper method
        outputs = self._process_language(input_ids, hidden, hidden_pooled, labels, mode, outputs)

        # 1.1 Memory processing via refactored helper method
        # 1.10 Entity/Relation extraction with entropy-based confidence filtering
        context_additions, outputs = self._process_memory(hidden_pooled, entities, relations, outputs)

        # 1.1 Reasoning processing via refactored helper method
        reasoning_contribution, outputs = self._process_reasoning(hidden_pooled, hidden, outputs, device)
        if reasoning_contribution is not None:
            context_additions.append(reasoning_contribution)

        # === META-LEARNING INTEGRATION ===
        # Use meta-learning modules when enabled

        if self.cfg.use_meta_learning and hasattr(self, "task_embedding"):
            # Generate task representation from observations
            if obs is not None:
                # TaskEmbedding expects examples: [batch, num_examples, obs_dim]
                # Use current obs as single "example" for task inference
                task_examples = obs.unsqueeze(1)  # [B, 1, obs_dim]
                task_repr = self.task_embedding(task_examples)  # [B, hidden]
                outputs["task_representation"] = task_repr

                # Few-shot learning: adapt action prediction using task context
                if hasattr(self, "few_shot_learner"):
                    # Combine hidden state with task representation
                    adapted_features = hidden_pooled + task_repr * 0.5

                    # FewShotLearner requires prototypes - use task_repr as proxy prototype
                    # Create pseudo-prototypes from task representation (one per action class)
                    task_repr_expanded = task_repr.unsqueeze(1)  # [B, 1, H]
                    num_actions = self.cfg.action_dim
                    proto_noise = torch.randn(
                        batch_size, num_actions, task_repr.size(-1),
                        device=device
                    ) * 0.1
                    pseudo_prototypes = task_repr_expanded + proto_noise

                    # Use encoder output as query
                    query_emb = self.few_shot_learner.encoder(adapted_features)  # [B, H]
                    # Compute distances to prototypes
                    few_shot_logits = -torch.cdist(
                        query_emb.unsqueeze(1),  # [B, 1, H]
                        pseudo_prototypes  # [B, num_actions, H]
                    ).squeeze(1)  # [B, num_actions]

                    outputs["few_shot_logits"] = few_shot_logits

                    # Blend with main action logits if available
                    if "action_logits" in outputs:
                        # Confidence-weighted blending
                        blend_weight = 0.3  # Can be learned later
                        outputs["action_logits"] = (
                            outputs["action_logits"] * (1 - blend_weight) +
                            few_shot_logits * blend_weight
                        )

        if self.cfg.use_transfer_learning and hasattr(self, "transfer_learner"):
            # Apply transfer learning to adapt features
            transfer_output = self.transfer_learner(hidden_pooled)
            # TransferLearner returns (features, domain_logits) tuple
            if isinstance(transfer_output, tuple):
                transferred_features, domain_logits = transfer_output
                outputs["domain_logits"] = domain_logits
            else:
                transferred_features = transfer_output
            outputs["transferred_features"] = transferred_features
            context_additions.append(transferred_features * 0.3)  # Weighted contribution

        # === NEW AGI MODULES IN FORWARD PASS ===

        # 1.4 Scene Graph Parsing - Batch-aware, no unnecessary flattening
        scene_graph = None
        if self.cfg.use_scene_graphs and hasattr(self, "scene_graph_builder") and obs is not None:
            scene_graph = self.scene_graph_builder(obs)
            outputs["scene_graph"] = scene_graph

            # Add scene graph embedding to context (batch-aware)
            if scene_graph is not None and hasattr(scene_graph, "objects"):
                scene_objects = scene_graph.objects

                # 1.4 Batch-aware scene graph handling
                if scene_objects.dim() == 2:
                    # [num_objects, object_dim] - single scene graph for all batch items
                    # Keep structure, just expand for batch
                    num_objects = scene_objects.size(0)
                    object_dim = scene_objects.size(1)

                    # Reshape preserving object structure: [1, num_objects * object_dim]
                    scene_flat = scene_objects.reshape(1, -1)
                    scene_embedding = scene_flat.expand(batch_size, -1)
                elif scene_objects.dim() == 3:
                    # [B, num_objects, object_dim] - already batch-aware
                    batch_size_scene = scene_objects.size(0)
                    scene_embedding = scene_objects.reshape(batch_size_scene, -1)

                    # Handle batch size mismatch
                    if batch_size_scene != batch_size:
                        if batch_size_scene == 1:
                            scene_embedding = scene_embedding.expand(batch_size, -1)
                        else:
                            # Truncate or pad to match
                            scene_embedding = scene_embedding[:batch_size] if batch_size_scene > batch_size else \
                                F.pad(scene_embedding, (0, 0, 0, batch_size - batch_size_scene))
                else:
                    # Fallback: flatten whatever we have
                    scene_embedding = scene_objects.reshape(-1).unsqueeze(0).expand(batch_size, -1)

                # Project to hidden size
                scene_contribution = self.scene_projection(scene_embedding)
                context_additions.append(scene_contribution)
                outputs["scene_embedding"] = scene_embedding  # For debugging
        
        # Intrinsic Motivation - compute curiosity/novelty/empowerment rewards
        if self.cfg.use_intrinsic_motivation and hasattr(self, "intrinsic_motivation"):
            if obs is not None:
                # Get predicted next state from world model if available
                world_pred = outputs.get("world_pred")
                if world_pred is not None:
                    # Handle multi-step world predictions [B, horizon, obs_dim]
                    if world_pred.dim() == 3:
                        next_state = world_pred[:, 0, :]  # Use first prediction step
                    else:
                        next_state = world_pred

                    # Ensure next_state matches obs dimensions
                    if next_state.size(-1) != obs.size(-1):
                        # Project to match obs_dim if needed
                        next_state = F.adaptive_avg_pool1d(
                            next_state.unsqueeze(1), obs.size(-1)
                        ).squeeze(1)

                    # Use action logits as soft action
                    action_logits = outputs.get("action_logits")
                    if action_logits is not None:
                        action_probs = F.softmax(action_logits, dim=-1)

                        try:
                            # Compute intrinsic rewards
                            intrinsic_rewards = self.intrinsic_motivation.compute_intrinsic_reward(
                                state=obs,
                                action=action_probs,
                                next_state=next_state
                            )
                            outputs["intrinsic_rewards"] = intrinsic_rewards

                            # Add total intrinsic reward to value estimate
                            if "intrinsic_reward" in intrinsic_rewards and "value" in outputs:
                                intrinsic_bonus = intrinsic_rewards["intrinsic_reward"].unsqueeze(-1)
                                outputs["value"] = outputs["value"] + intrinsic_bonus * 0.1
                        except RuntimeError:
                            # Dimension mismatch - skip intrinsic rewards for this batch
                            pass

                # Also compute novelty for current state (doesn't need next_state)
                try:
                    if hasattr(self.intrinsic_motivation, "novelty"):
                        novelty_score = self.intrinsic_motivation.novelty(obs)
                        outputs["novelty_score"] = novelty_score
                except RuntimeError:
                    # Skip on dimension errors
                    pass

                # Generate exploration goals
                if hasattr(self.intrinsic_motivation, "goal_generator"):
                    exploration_goal = self.intrinsic_motivation.goal_generator(obs)
                    outputs["exploration_goal"] = exploration_goal

        # 1.5 Program Synthesis - Type checking and proper error handling with logging
        if self.cfg.use_program_synthesis and hasattr(self, "program_synthesizer"):
            program_context = None
            if "reasoning" in outputs and "relational" in outputs["reasoning"]:
                program_context = outputs["reasoning"]["relational"]
                outputs["program_context"] = program_context

            # Attempt synthesis when we have context and low confidence
            confidence_val = outputs.get("confidence", torch.tensor(1.0))
            if isinstance(confidence_val, torch.Tensor):
                confidence_mean = confidence_val.mean().item()
            else:
                confidence_mean = float(confidence_val)

            should_synthesize = (
                program_context is not None and
                mode == "inference" and
                confidence_mean < 0.7
            )

            if should_synthesize:
                # Create pseudo-examples from context for synthesis
                # Input: current hidden state, Output: expected action pattern
                if obs is not None and "action_logits" in outputs:
                    # 1.5 Type checking for inputs
                    synth_input = hidden_pooled
                    action_logits = outputs["action_logits"]

                    # Validate tensor types
                    if not isinstance(synth_input, torch.Tensor):
                        logger.warning("Program synthesis skipped: synth_input is not a tensor")
                        outputs["program_synthesis_success"] = False
                        outputs["program_synthesis_error"] = "Invalid input type"
                    elif not isinstance(action_logits, torch.Tensor):
                        logger.warning("Program synthesis skipped: action_logits is not a tensor")
                        outputs["program_synthesis_success"] = False
                        outputs["program_synthesis_error"] = "Invalid action_logits type"
                    else:
                        synth_output = F.softmax(action_logits, dim=-1)

                        # Create example pairs for synthesis with type validation
                        examples = []
                        for i in range(min(batch_size, 3)):
                            inp = synth_input[i]
                            out = synth_output[i]
                            # Validate each example
                            if inp.dim() == 1 and out.dim() == 1:
                                examples.append((inp, out))
                            else:
                                logger.debug(f"Skipping example {i}: invalid dimensions inp={inp.dim()}, out={out.dim()}")

                        if not examples:
                            logger.warning("Program synthesis skipped: no valid examples")
                            outputs["program_synthesis_success"] = False
                            outputs["program_synthesis_error"] = "No valid examples"
                        else:
                            try:
                                # Synthesize program from examples
                                synthesized_program = self.program_synthesizer.synthesize_from_examples(
                                    examples=examples,
                                    num_iterations=10  # Limited iterations for speed
                                )

                                if synthesized_program is not None:
                                    outputs["synthesized_program"] = synthesized_program
                                    outputs["program_synthesis_success"] = True
                                    logger.debug(f"Program synthesized with score: {synthesized_program.score:.3f}")

                                    # Execute synthesized program on hidden state
                                    try:
                                        input_data = hidden_pooled.mean(dim=0).tolist()
                                        # Type check input data
                                        if not isinstance(input_data, list):
                                            raise TypeError(f"Expected list, got {type(input_data)}")

                                        program_output = synthesized_program.execute(input_data)
                                        outputs["program_output"] = program_output
                                        logger.debug(f"Program executed successfully, output type: {type(program_output)}")
                                    except TypeError as te:
                                        logger.warning(f"Program execution type error: {te}")
                                        outputs["program_execution_error"] = str(te)
                                    except Exception as e:
                                        logger.warning(f"Program execution failed: {e}")
                                        outputs["program_execution_error"] = str(e)
                                else:
                                    outputs["program_synthesis_success"] = False
                                    logger.debug("Program synthesis returned None")
                            except TypeError as te:
                                logger.error(f"Program synthesis type error: {te}")
                                outputs["program_synthesis_success"] = False
                                outputs["program_synthesis_error"] = f"Type error: {te}"
                            except ValueError as ve:
                                logger.error(f"Program synthesis value error: {ve}")
                                outputs["program_synthesis_success"] = False
                                outputs["program_synthesis_error"] = f"Value error: {ve}"
                            except Exception as e:
                                logger.error(f"Program synthesis unexpected error: {e}")
                                outputs["program_synthesis_success"] = False
                                outputs["program_synthesis_error"] = f"Unexpected: {e}"
        
        # 1.6 Grounded Language Understanding - Allow any modality (not strict AND)
        if self.cfg.use_grounded_language and hasattr(self, "grounded_language"):
            # Get available modalities
            vision_features = outputs.get("vision_features")
            has_vision = vision_features is not None
            has_text = hidden_pooled is not None  # Text features from core processing

            # 1.6 Allow grounding with ANY available modality, not requiring both
            if has_vision or has_text:
                grounded_output = self.grounded_language(
                    text_features=hidden_pooled if has_text else None,
                    vision_features=vision_features if has_vision else None,
                    image=image if not has_vision and image is not None else None,
                    mode=mode
                )
                outputs["grounded_language"] = grounded_output
                outputs["grounded_modalities"] = {
                    "vision": has_vision,
                    "text": has_text
                }

                # Add grounded understanding to context
                if "grounded_hidden" in grounded_output:
                    context_additions.append(grounded_output["grounded_hidden"])
        
        # Meta-Cognition (inference only to avoid training overhead)
        # Process each sample in batch for proper metacognition analysis
        if self.cfg.use_metacognition and hasattr(self, "metacognition") and mode == "inference":
            metacog_results = []
            batch_confidences = []

            for i in range(batch_size):
                # Extract single sample's hidden state
                hidden_for_metacog = hidden_pooled[i:i+1]

                # Learnable task embedding from hidden state (fixes gradient flow)
                task_emb = self.task_embedding_layer(hidden_for_metacog)

                # Thought sequence for this sample
                thought_sequence = [hidden_for_metacog]

                # Meta-cognitive analysis
                try:
                    metacog_output = self.metacognition(
                        task_embedding=task_emb,
                        current_thoughts=thought_sequence,
                        hidden_state=hidden_for_metacog
                    )
                    metacog_results.append(metacog_output)
                    batch_confidences.append(metacog_output.get('calibrated_confidence', 1.0))
                except Exception:
                    # Fallback for this sample
                    metacog_results.append({"confidence": 1.0, "should_attempt": True})
                    batch_confidences.append(1.0)

            # Aggregate batch results
            outputs["metacognition"] = {
                "batch_results": metacog_results,
                "mean_confidence": sum(batch_confidences) / len(batch_confidences),
                "min_confidence": min(batch_confidences),
                "confidences": torch.tensor(batch_confidences, device=device),
                # Use first sample's detailed output for backward compatibility
                **metacog_results[0]
            }
        
        # === ONLINE LEARNING INTEGRATION ===
        # Collect confidence metrics for online learning decision
        confidence_metrics = {}
        if "memory_info" in outputs:
            confidence_metrics['memory_confidence'] = outputs["memory_info"].get('confidence', 1.0)
        if "reasoning" in outputs and "confidence" in outputs["reasoning"]:
            confidence_metrics['reasoning_confidence'] = outputs["reasoning"]["confidence"]
        if "metacognition" in outputs:
            metacog = outputs["metacognition"]
            if isinstance(metacog, dict):
                confidence_metrics['action_confidence'] = metacog.get('confidence', 1.0)
        if "uncertainty" in outputs:
            confidence_metrics['uncertainty'] = outputs["uncertainty"].mean().item() if isinstance(outputs["uncertainty"], torch.Tensor) else outputs["uncertainty"]

        # Process through online learner if available and in inference mode
        if (
            mode == "inference"
            and hasattr(self, '_online_learner')
            and self._online_learner is not None
            and obs is not None
        ):
            state_dict = {'obs': obs, 'hidden': hidden_pooled}
            action = outputs.get('action_logits', torch.zeros(batch_size, self.cfg.action_dim, device=device))

            online_result = self._online_learner.process_inference_step(
                hidden_state=hidden_pooled,
                confidence_metrics=confidence_metrics,
                state=state_dict,
                action=action.argmax(dim=-1) if action.dim() > 1 else action,
                outputs=outputs
            )
            outputs["online_learning"] = online_result

        # Store confidence metrics for external use
        outputs["confidence_metrics"] = confidence_metrics

        # === END NEW MODULES ===

        if context_additions:
            # Learned context gating: compute attention weights for each context
            # Query from hidden_pooled, keys from each context addition
            query = self.context_gate_query(hidden_pooled)  # [B, D/4]

            # Stack context additions and compute keys
            context_stack = torch.stack(context_additions, dim=1)  # [B, num_ctx, D]
            keys = self.context_gate_key(context_stack)  # [B, num_ctx, D/4]

            # Scaled dot-product attention weights
            attn_scores = torch.bmm(keys, query.unsqueeze(-1)).squeeze(-1)  # [B, num_ctx]
            attn_scores = attn_scores / (self.context_gate_temperature * (query.size(-1) ** 0.5))
            attn_weights = F.softmax(attn_scores, dim=-1)  # [B, num_ctx]

            # Weighted sum of context contributions
            weighted_context = torch.bmm(
                attn_weights.unsqueeze(1),  # [B, 1, num_ctx]
                context_stack  # [B, num_ctx, D]
            ).squeeze(1)  # [B, D]

            augmented_hidden = hidden_pooled + weighted_context
            outputs["augmented_hidden"] = augmented_hidden
            outputs["context_attention_weights"] = attn_weights  # For interpretability

            augmented_action_logits = self.augmented_action_head(augmented_hidden)
            augmented_value = self.augmented_value_head(augmented_hidden)
            
            if "action_logits" in outputs:
                outputs["action_logits"] = outputs["action_logits"] + augmented_action_logits
            if "value" in outputs:
                outputs["value"] = outputs["value"] + augmented_value
        
        # 1.1 Action/Tool processing via refactored helper method
        # 1.13 Differentiable tool interface with straight-through estimator
        outputs = self._process_action(hidden_pooled, augmented_hidden, context_additions, outputs, mode)

        # === UNCERTAINTY-BASED ACTION REJECTION (Safety Feature) ===
        # Refuse to output confident actions when uncertainty is too high
        if mode == "inference" and "action_logits" in outputs:
            # Gather uncertainty indicators
            uncertainty = outputs.get("uncertainty", torch.zeros(batch_size, device=device))
            if isinstance(uncertainty, torch.Tensor) and uncertainty.numel() > 1:
                uncertainty = uncertainty.mean(dim=-1) if uncertainty.dim() > 1 else uncertainty

            confidence = outputs.get("confidence", torch.ones(batch_size, device=device))
            if isinstance(confidence, torch.Tensor) and confidence.numel() > 1:
                confidence = confidence.mean(dim=-1) if confidence.dim() > 1 else confidence

            # Also consider metacognition confidence
            metacog_conf = 1.0
            if "metacognition" in outputs:
                metacog = outputs["metacognition"]
                if isinstance(metacog, dict):
                    metacog_conf = metacog.get("mean_confidence", 1.0)

            # Combined confidence score
            combined_confidence = (confidence.mean().item() + metacog_conf) / 2

            # Rejection threshold (configurable via config)
            rejection_threshold = getattr(self.cfg, 'action_rejection_threshold', 0.2)

            # Mark actions as rejected if confidence too low
            should_reject = combined_confidence < rejection_threshold
            outputs["action_rejected"] = should_reject
            outputs["rejection_confidence"] = combined_confidence

            if should_reject:
                # Don't modify action_logits, but add a clear signal
                outputs["rejection_reason"] = f"Confidence too low: {combined_confidence:.3f} < {rejection_threshold}"
                # Flatten action distribution when rejecting (uniform = "I don't know")
                if "action_logits" in outputs:
                    uniform_logits = torch.zeros_like(outputs["action_logits"])
                    outputs["rejected_original_logits"] = outputs["action_logits"]
                    outputs["action_logits"] = uniform_logits  # Uniform distribution

        if return_loss and self.cfg.use_language_modeling:
            losses = outputs.get("losses_breakdown", {})

            if hasattr(self, "masked_lm") and labels is not None:
                # Align hidden and labels sequence lengths
                mlm_hidden = hidden
                if mlm_hidden.dim() == 3 and labels.dim() == 2:
                    label_seq_len = labels.size(1)
                    hidden_seq_len = mlm_hidden.size(1)
                    if hidden_seq_len > label_seq_len:
                        # Truncate hidden to match labels
                        mlm_hidden = mlm_hidden[:, :label_seq_len, :]
                    elif hidden_seq_len < label_seq_len:
                        # Truncate labels to match hidden
                        labels = labels[:, :hidden_seq_len]

                try:
                    mlm_outputs = self.masked_lm(mlm_hidden, labels)
                    if "loss" in mlm_outputs:
                        losses["masked_lm"] = mlm_outputs["loss"]
                except (ValueError, RuntimeError):
                    # Skip MLM loss if shapes still don't match
                    pass
            
            if hasattr(self, "knowledge_graph") and entities is not None and targets is not None:
                if "kg_triples" in targets:
                    kg_triples = targets["kg_triples"]
                    head, relation, tail = kg_triples[:, 0], kg_triples[:, 1], kg_triples[:, 2]
                    kg_scores = self.knowledge_graph.score_triple(head, relation, tail)
                    kg_labels = targets.get("kg_labels", torch.ones_like(kg_scores))
                    losses["knowledge_graph"] = F.binary_cross_entropy_with_logits(kg_scores, kg_labels)
            
            if augmented_hidden is not None and targets is not None:
                if "actions" in targets:
                    augmented_policy_loss = F.cross_entropy(
                        augmented_action_logits,
                        targets["actions"]
                    )
                    losses["augmented_policy"] = augmented_policy_loss
                
                if "values" in targets:
                    augmented_value_loss = F.mse_loss(
                        augmented_value.squeeze(-1),
                        targets["values"]
                    )
                    losses["augmented_value"] = augmented_value_loss

            # === COMPOUND LOSSES FOR ALL AGI MODULES ===

            # Intrinsic Motivation Loss - train curiosity/novelty predictors
            if "intrinsic_rewards" in outputs and targets is not None:
                from ..training.losses import intrinsic_reward_loss
                if "actual_rewards" in targets:
                    intrinsic_loss = intrinsic_reward_loss(
                        predicted_rewards=outputs["intrinsic_rewards"],
                        actual_rewards=targets["actual_rewards"],
                        states=obs if obs is not None else hidden_pooled,
                        next_states=outputs.get("world_pred", hidden_pooled)
                    )
                    losses["intrinsic_motivation"] = intrinsic_loss * 0.1

            # Meta-Cognition Loss - calibrate confidence predictions
            if "metacognition" in outputs and targets is not None:
                from ..training.losses import meta_cognition_loss
                metacog = outputs["metacognition"]
                if isinstance(metacog, dict) and "actual_success" in targets:
                    predicted_conf = metacog.get("mean_confidence", 0.5)
                    metacog_loss = meta_cognition_loss(
                        predicted_confidence=torch.tensor([predicted_conf], device=device),
                        actual_success=targets["actual_success"],
                        task_embedding=hidden_pooled.mean(dim=0, keepdim=True)
                    )
                    losses["metacognition"] = metacog_loss * 0.1

            # Program Synthesis Loss - train program generator
            if "synthesized_program" in outputs and targets is not None:
                from ..training.losses import program_synthesis_loss
                if "target_program" in targets:
                    prog_loss = program_synthesis_loss(
                        predicted_program=outputs.get("program_context", hidden_pooled),
                        target_program=targets["target_program"],
                        execution_results=outputs.get("program_output"),
                        expected_outputs=targets.get("expected_outputs")
                    )
                    losses["program_synthesis"] = prog_loss * 0.1

            # Transfer Learning Loss - align source/target domains
            if "transferred_features" in outputs and self.cfg.use_transfer_learning:
                # Domain alignment via feature consistency
                transfer_features = outputs["transferred_features"]
                transfer_loss = F.mse_loss(
                    transfer_features,
                    hidden_pooled.detach()  # Should be similar to original
                ) * 0.01
                losses["transfer_alignment"] = transfer_loss

            # Few-Shot Learning Loss
            if "few_shot_logits" in outputs and targets is not None and "actions" in targets:
                few_shot_loss = F.cross_entropy(
                    outputs["few_shot_logits"],
                    targets["actions"]
                )
                losses["few_shot"] = few_shot_loss * 0.3

            if losses:
                loss_weights = targets.get("loss_weights", {}) if targets else {}
                total = 0.0
                for key, value in losses.items():
                    weight = loss_weights.get(key, 1.0)
                    total = total + weight * value

                outputs["loss"] = total
                outputs["losses_breakdown"] = losses
        
        return outputs

    def think_and_act(
        self,
        input_ids: torch.Tensor,
        obs: Optional[torch.Tensor],
        state: Any,
        task_ids: Optional[torch.Tensor] = None,
        image: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """High-level reasoning and action selection."""
        return self.core.think_then_act(
            input_ids=input_ids,
            obs=obs,
            state=state,
            task_ids=task_ids,
            image=image,
            **kwargs
        )

    def learn_from_examples(
        self,
        examples: List[Tuple[torch.Tensor, torch.Tensor]],
        query: torch.Tensor,
        adaptation_steps: int = 5,
        inner_lr: float = 0.01
    ) -> torch.Tensor:
        """Few-shot learning with real MAML adaptation."""
        if not hasattr(self, "task_embedding"):
            raise ValueError("Meta-learning not enabled")
        
        support_x = torch.stack([ex[0] for ex in examples])
        support_y = torch.stack([ex[1] for ex in examples])
        
        if hasattr(self.core, 'pi'):
            adapted_params = {}
            for name, param in self.core.pi.named_parameters():
                adapted_params[name] = param.clone()
            
            for step in range(adaptation_steps):
                support_logits = []
                for x in support_x:
                    state = self.init_state(batch_size=1, device=x.device)
                    out = self(
                        input_ids=None,
                        obs=x.unsqueeze(0) if x.dim() == 1 else x,
                        state=state,
                        mode="inference"
                    )
                    support_logits.append(out["action_logits"].squeeze(0))
                
                support_logits_tensor = torch.stack(support_logits)
                support_y_flat = support_y.squeeze() if support_y.dim() > 1 else support_y
                
                adapt_loss = F.cross_entropy(support_logits_tensor, support_y_flat)
                
                grads = torch.autograd.grad(
                    adapt_loss,
                    [p for p in self.core.pi.parameters() if p.requires_grad],
                    create_graph=True,
                    allow_unused=True
                )
                
                param_list = [p for p in self.core.pi.parameters() if p.requires_grad]
                for param, grad in zip(param_list, grads):
                    if grad is not None:
                        param.data = param.data - inner_lr * grad
        
        query_state = self.init_state(batch_size=query.size(0) if query.dim() > 1 else 1, device=query.device)
        query_out = self(
            input_ids=None,
            obs=query if query.dim() > 1 else query.unsqueeze(0),
            state=query_state,
            mode="inference"
        )
        
        logits = query_out["action_logits"]
        return logits

    def register_tool(self, name: str, function: callable, description: str) -> None:
        """Register a new tool."""
        if not hasattr(self, "tool_registry"):
            raise ValueError("Tool use not enabled")
        
        self.tool_registry.register(name, function, description)

    def update_curriculum(self, task_id: int, performance: float) -> None:
        """Update curriculum based on performance."""
        if hasattr(self, "curriculum_scheduler"):
            self.curriculum_scheduler.update_performance(task_id, performance)

    def get_next_task(self, student_state: torch.Tensor) -> int:
        """Get next task from curriculum."""
        if not hasattr(self, "curriculum_scheduler"):
            raise ValueError("Curriculum learning not enabled")
        
        return self.curriculum_scheduler.get_next_task(student_state)

    def init_state(self, batch_size: int, device: Optional[torch.device] = None, **kwargs):
        """Initialize model state."""
        return self.core.init_state(batch_size, device, **kwargs)

    @property
    def dsl(self):
        """Access DSL from program synthesizer (if enabled)."""
        if hasattr(self, "program_synthesizer"):
            return self.program_synthesizer.dsl
        return None
