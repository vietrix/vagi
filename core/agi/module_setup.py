"""Helper method to set up continuous learner after optimizer is available."""

def setup_continuous_learner(model, optimizer):
    """Initialize continuous learner with optimizer."""
    if hasattr(model, 'continuous_learning_config') and model.cfg.use_continuous_learning:
        from core.training import ContinuousLearner
        
        model._continuous_learner = ContinuousLearner(
            model=model.core,
            optimizer=optimizer,
            config=model.continuous_learning_config
        )
        return model._continuous_learner
    return None

def add_agi_modules_to_model(model):
    """Add new AGI modules to existing AGIModel instance."""
    cfg = model.cfg
    
    # Scene Graph Builder
    if cfg.use_scene_graphs and not hasattr(model, 'scene_graph_builder'):
        from core.perception import SceneGraphBuilder, GroundedWorldModel
        model.scene_graph_builder = SceneGraphBuilder(
            obs_dim=cfg.obs_dim,
            num_slots=cfg.num_object_slots,
            object_dim=cfg.object_dim,
            hidden_size=cfg.hidden_size,
            num_iterations=cfg.scene_graph_iterations
        )
        if cfg.use_world_pred:
            model.grounded_world_model = GroundedWorldModel(
                scene_graph_dim=cfg.object_dim * cfg.num_object_slots,
                action_dim=cfg.action_dim,
                hidden_size=cfg.hidden_size,
                horizon=cfg.world_model_horizon
            )
    
    # Intrinsic Motivation
    if cfg.use_intrinsic_motivation and not hasattr(model, 'intrinsic_motivation'):
        from core.planning import IntrinsicMotivationSystem, IntrinsicRewardConfig
        intrinsic_config = IntrinsicRewardConfig(
            curiosity_weight=cfg.curiosity_weight,
            novelty_weight=cfg.novelty_weight,
            empowerment_weight=cfg.empowerment_weight
        )
        model.intrinsic_motivation = IntrinsicMotivationSystem(
            state_dim=cfg.obs_dim,
            action_dim=cfg.action_dim,
            hidden_size=cfg.intrinsic_hidden_size,
            config=intrinsic_config,
            novelty_buffer_size=cfg.novelty_buffer_size
        )
    
    # Program Synthesis
    if cfg.use_program_synthesis and not hasattr(model, 'program_synthesizer'):
        from core.reasoning import ProgramSynthesizer, DomainSpecificLanguage
        model.dsl = DomainSpecificLanguage(num_primitives=cfg.num_primitives)
        model.program_synthesizer = ProgramSynthesizer(
            dsl=model.dsl,
            hidden_size=cfg.program_hidden_size,
            max_length=cfg.max_program_length,
            num_samples=cfg.num_program_samples
        )
    
    # Grounded Language
    if cfg.use_grounded_language and not hasattr(model, 'grounded_language'):
        from core.nlp import GroundedLanguageModel
        model.grounded_language = GroundedLanguageModel(
            text_dim=cfg.hidden_size,
            vision_dim=cfg.vision_embed_dim if cfg.use_vision else cfg.hidden_size,
            hidden_size=cfg.grounded_lang_hidden_size,
            vocab_size=cfg.vocab_size
        )
    
    # Meta-Cognition
    if cfg.use_metacognition and not hasattr(model, 'metacognition'):
        from core.learning import MetaCognition
        model.metacognition = MetaCognition(
            hidden_size=cfg.metacog_hidden_size,
            task_embedding_dim=cfg.metacog_task_embedding_dim
        )
    
    return model
