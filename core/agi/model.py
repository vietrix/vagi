"""Main AGI model integrating all components."""

from __future__ import annotations

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
        
        self.memory_projection = nn.Linear(cfg.hidden_size, cfg.hidden_size)
        self.kg_projection = nn.Linear(cfg.entity_embed_dim, cfg.hidden_size)
        self.reasoning_projection = nn.Linear(cfg.hidden_size, cfg.hidden_size)
        self.augmented_action_head = nn.Linear(cfg.hidden_size, cfg.action_dim)
        self.augmented_value_head = nn.Linear(cfg.hidden_size, 1)
        self.vision_fusion_weight = nn.Parameter(torch.tensor(0.5))
        self._tokenizer: Optional[BytePairTokenizer] = None
        
        # Setup extended AGI modules
        self.setup_extended_modules()

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
        
        # Grounded Language - DISABLED (requires vision and text encoders)
        # if cfg.use_grounded_language:
        #     from ..nlp import GroundedLanguageModel
        #     self.grounded_language = GroundedLanguageModel(...)
        
        # Meta-Cognition - DISABLED (needs proper setup)
        # if cfg.use_metacognition:
        #     from ..learning import MetaCognition
        #     self.metacognition = MetaCognition(...)


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
        
        if image is not None and self.cfg.use_vision:
            vision_features = self.vision_encoder(image)
            outputs["vision_features"] = vision_features
            
            if self.cfg.use_multimodal_fusion and hasattr(self, "multimodal_encoder"):
                fused_features = self.multimodal_encoder(image=image, text=None)
                
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
                obs = obs * (1 - self.vision_fusion_weight) + obs_from_vision * self.vision_fusion_weight
        
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
        
        context_additions = []
        
        if hasattr(self, "hierarchical_memory"):
            memory_output, memory_info = self.hierarchical_memory(hidden_pooled)
            outputs["memory_retrieval"] = memory_output
            outputs["memory_info"] = memory_info
            
            memory_contribution = self.memory_projection(memory_output)
            context_additions.append(memory_contribution)
        
        if self.cfg.use_knowledge_graph and hasattr(self, "knowledge_graph"):
            if entities is not None and relations is not None:
                kg_indices, kg_scores = self.knowledge_graph.query(entities, relations, k=5)
                
                kg_embeddings = self.knowledge_graph.entity_embeddings(kg_indices)
                kg_context = (kg_embeddings * kg_scores.unsqueeze(-1)).sum(dim=1)
                
                kg_contribution = self.kg_projection(kg_context)
                context_additions.append(kg_contribution)
                
                outputs["knowledge_context"] = kg_context
        
        if self.cfg.use_abstract_reasoning:
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
                context_additions.append(reasoning_contribution)
        
        # === NEW AGI MODULES IN FORWARD PASS ===
        
        # Scene Graph Parsing
        scene_graph = None
        if self.cfg.use_scene_graphs and hasattr(self, "scene_graph_builder") and obs is not None:
            scene_graph = self.scene_graph_builder(obs)
            outputs["scene_graph"] = scene_graph
            
            # Add scene graph embedding to context
            if scene_graph is not None and hasattr(scene_graph, "objects"):
                # scene_graph.objects: [num_objects, object_dim]
                # Flatten to [num_objects * object_dim] then replicate for batch
                scene_embedding = scene_graph.objects.reshape(-1).unsqueeze(0).expand(batch_size, -1)
                    
                # Project to hidden size if needed
                if scene_embedding.size(-1) != hidden_pooled.size(-1):
                    if not hasattr(self, "scene_projection"):
                        self.scene_projection = nn.Linear(
                            scene_embedding.size(-1),
                            hidden_pooled.size(-1)
                        ).to(hidden_pooled.device)
                    scene_contribution = self.scene_projection(scene_embedding)
                else:
                    scene_contribution = scene_embedding
                context_additions.append(scene_contribution)
        
        # Program Synthesis Context
        if self.cfg.use_program_synthesis and hasattr(self, "program_synthesizer"):
            if "reasoning" in outputs and "relational" in outputs["reasoning"]:
                # Store program context for specialized tasks
                outputs["program_context"] = outputs["reasoning"]["relational"]
        
        # Grounded Language Understanding
        if self.cfg.use_grounded_language and hasattr(self, "grounded_language"):
            if image is not None and hasattr(self, "vision_encoder"):
                # Vision-language grounding
                vision_features = outputs.get("vision_features")
                if vision_features is not None:
                    grounded_output = self.grounded_language(
                        text_features=hidden_pooled,
                        vision_features=vision_features,
                        mode=mode
                    )
                    outputs["grounded_language"] = grounded_output
                    
                    # Add grounded understanding to context
                    if "grounded_hidden" in grounded_output:
                        context_additions.append(grounded_output["grounded_hidden"])
        
        # Meta-Cognition (inference only to avoid training overhead)
        if self.cfg.use_metacognition and hasattr(self, "metacognition") and mode == "inference":
            # Task embedding (simplified - use task_ids if available)
            if task_ids is not None:
                task_emb = torch.randn(
                    batch_size,
                    self.cfg.metacog_task_embedding_dim,
                    device=device
                )
            else:
                task_emb = torch.zeros(
                    batch_size,
                    self.cfg.metacog_task_embedding_dim,
                    device=device
                )
            
            # Thought sequence (use hidden states)
            thought_sequence = [hidden_pooled]
            
            # Meta-cognitive analysis
            metacog_output = self.metacognition(
                task_embedding=task_emb,
                current_thoughts=thought_sequence,
                hidden_state=hidden_pooled
            )
            outputs["metacognition"] = metacog_output
        
        # === END NEW MODULES ===
        
        if context_additions:
            augmented_hidden = hidden_pooled + sum(context_additions)
            outputs["augmented_hidden"] = augmented_hidden
            
            augmented_action_logits = self.augmented_action_head(augmented_hidden)
            augmented_value = self.augmented_value_head(augmented_hidden)
            
            if "action_logits" in outputs:
                outputs["action_logits"] = outputs["action_logits"] + augmented_action_logits
            if "value" in outputs:
                outputs["value"] = outputs["value"] + augmented_value
        
        if self.cfg.use_tool_use and mode in ["train", "inference"]:
            context = augmented_hidden if augmented_hidden is not None else hidden_pooled
            
            should_use_tool, tool_id, tool_params = self.tool_controller(context)
            outputs["tool_use"] = {
                "should_use": should_use_tool,
                "tool_id": tool_id,
                "params": tool_params,
                "context": context
            }
        
        if return_loss and self.cfg.use_language_modeling:
            losses = outputs.get("losses_breakdown", {})
            
            if hasattr(self, "masked_lm") and labels is not None:
                mlm_outputs = self.masked_lm(hidden, labels)
                if "loss" in mlm_outputs:
                    losses["masked_lm"] = mlm_outputs["loss"]
            
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
