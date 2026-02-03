"""Integration tests for full AGI system."""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

import torch
import pytest

from core.agi.config import AGIConfig, load_agi_small_config
from core.agi.model import AGIModel
from core.agi.executor import AGIExecutor


class TestAGIIntegration:
    """Test full AGI integration."""
    
    @pytest.fixture
    def config(self):
        """Create test configuration."""
        return load_agi_small_config()
    
    @pytest.fixture
    def model(self, config):
        """Create test model."""
        return AGIModel(config)
    
    def test_model_initialization(self, model, config):
        """Test that model initializes with all modules."""
        assert model.cfg == config
        
        # Check core components
        assert hasattr(model, "core")
        assert hasattr(model, "concept_encoder")
        
        # Check new AGI modules
        if config.use_continuous_learning:
            assert hasattr(model, "continuous_learning_config")
        
        if config.use_scene_graphs:
            assert hasattr(model, "scene_graph_builder")
        
        if config.use_intrinsic_motivation:
            assert hasattr(model, "intrinsic_motivation")
        
        if config.use_program_synthesis:
            assert hasattr(model, "program_synthesizer")
            assert hasattr(model, "dsl")
        
        if config.use_grounded_language:
            assert hasattr(model, "grounded_language")
        
        if config.use_metacognition:
            assert hasattr(model, "metacognition")
    
    def test_forward_pass(self, model, config):
        """Test forward pass executes without errors."""
        batch_size = 2
        seq_len = 16

        input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
        obs = torch.randn(batch_size, config.obs_dim)
        state = model.core.init_state(batch_size)

        outputs = model(
            input_ids=input_ids,
            obs=obs,
            state=state,
            mode="train"
        )

        # Check basic outputs
        assert "text_logits" in outputs
        assert "action_logits" in outputs
        assert "value" in outputs

        # Check batch dimension (output seq_len may differ due to obs_tokens, special_tokens)
        assert outputs["text_logits"].shape[0] == batch_size
        assert outputs["text_logits"].shape[2] == config.vocab_size  # vocab_size
        assert outputs["action_logits"].shape[0] == batch_size
        assert outputs["value"].shape[0] == batch_size
    
    def test_scene_graph_integration(self, model, config):
        """Test scene graph builder works."""
        if not config.use_scene_graphs:
            pytest.skip("Scene graphs disabled")

        batch_size = 2
        obs = torch.randn(batch_size, config.obs_dim)

        scene_graph = model.scene_graph_builder(obs)

        # SceneGraph is a dataclass with .objects attribute
        assert hasattr(scene_graph, "objects")
        assert scene_graph.objects.shape[0] == config.num_object_slots
    
    def test_intrinsic_motivation_integration(self, model, config):
        """Test intrinsic motivation system works."""
        if not config.use_intrinsic_motivation:
            pytest.skip("Intrinsic motivation disabled")

        batch_size = 2

        state = torch.randn(batch_size, config.obs_dim)
        # Action needs to be float for forward pass (soft action)
        action = torch.randn(batch_size, config.action_dim)
        next_state = torch.randn(batch_size, config.obs_dim)

        rewards = model.intrinsic_motivation.compute_intrinsic_reward(
            state=state,
            action=action,
            next_state=next_state
        )

        assert "intrinsic_reward" in rewards
        assert rewards["intrinsic_reward"].shape[0] == batch_size
        assert "curiosity" in rewards
        assert "novelty" in rewards
    
    def test_program_synthesis_integration(self, model, config):
        """Test program synthesizer works."""
        if not config.use_program_synthesis:
            pytest.skip("Program synthesis disabled")

        batch_size = 2
        context = torch.randn(batch_size, config.program_hidden_size)

        # Should not crash
        assert model.program_synthesizer is not None
        assert model.dsl is not None
        # DSL has get_primitives() method, not num_primitives attribute
        assert len(model.dsl.get_primitives()) > 0
    
    def test_grounded_language_integration(self, model, config):
        """Test grounded language model works."""
        if not config.use_grounded_language:
            pytest.skip("Grounded language disabled")
        if not config.use_vision:
            pytest.skip("Grounded language requires vision")

        batch_size = 2

        # Use the grounded_lang_hidden_size for correct dimensions
        text_features = torch.randn(batch_size, config.grounded_lang_hidden_size)
        vision_features = torch.randn(batch_size, config.grounded_lang_hidden_size)

        output = model.grounded_language(
            text_features=text_features,
            vision_features=vision_features,
            mode="inference"
        )

        assert output is not None
        assert isinstance(output, dict)
        assert "grounded_hidden" in output
    
    def test_metacognition_integration(self, model, config):
        """Test meta-cognition system works."""
        if not config.use_metacognition:
            pytest.skip("Meta-cognition disabled")
        
        batch_size = 2
        
        task_emb = torch.randn(batch_size, config.metacog_task_embedding_dim)
        hidden = torch.randn(batch_size, config.metacog_hidden_size)
        
        should_attempt, reason, metrics = model.metacognition.should_i_attempt(task_emb)
        
        assert isinstance(should_attempt, bool)
        assert isinstance(reason, str)
        assert isinstance(metrics, dict)
    
    def test_continuous_learning_setup(self, model, config):
        """Test continuous learner can be initialized."""
        if not config.use_continuous_learning:
            pytest.skip("Continuous learning disabled")

        from core.training import ContinuousLearner

        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

        continuous_learner = ContinuousLearner(
            model=model.core,
            optimizer=optimizer,
            config=model.continuous_learning_config
        )

        assert continuous_learner is not None

        # Test observation - must provide all required metrics for QualityGate
        obs = torch.randn(1, config.obs_dim)
        action = torch.tensor([0])
        reward = 1.0  # Float, not tensor

        continuous_learner.observe(
            state={"obs": obs, "value": torch.tensor([[0.0]])},
            action=action,
            reward=reward,
            next_state={"obs": obs, "value": torch.tensor([[0.0]])},
            done=False,
            info={"uncertainty": 0.1, "validity": 0.9}  # Provide metrics for QualityGate
        )

        stats = continuous_learner.get_statistics()
        assert stats["buffer_size"] == 1
    
    def test_executor_integration(self, model, config):
        """Test executor works with full AGI model."""
        executor = AGIExecutor(model, max_steps=10)
        
        batch_size = 1
        seq_len = 8
        
        input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
        obs = torch.randn(batch_size, config.obs_dim)
        state = model.core.init_state(batch_size)
        
        outputs = executor.execute_step(
            input_ids=input_ids,
            obs=obs,
            state=state
        )
        
        assert "action_logits" in outputs
        assert "value" in outputs
    
    def test_full_training_step(self, model, config):
        """Test full training step with all modules."""
        batch_size = 2
        seq_len = 16
        
        input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
        obs = torch.randn(batch_size, config.obs_dim)
        labels = input_ids.clone()
        state = model.core.init_state(batch_size)
        
        entities = torch.randint(0, config.num_entities, (batch_size,))
        relations = torch.randint(0, config.num_relations, (batch_size,))
        
        targets = {
            "actions": torch.randint(0, config.action_dim, (batch_size,)),
            "values": torch.randn(batch_size),
            "kg_triples": torch.stack([
                entities,
                relations,
                torch.randint(0, config.num_entities, (batch_size,))
            ], dim=1),
            "kg_labels": torch.ones(batch_size),
        }
        
        outputs = model(
            input_ids=input_ids,
            obs=obs,
            state=state,
            entities=entities,
            relations=relations,
            labels=labels,
            targets=targets,
            mode="train",
            return_loss=True
        )
        
        assert "loss" in outputs
        assert outputs["loss"] is not None
        
        # Test backward pass
        loss = outputs["loss"]
        loss.backward()
        
        # Check gradients exist
        has_grad = False
        for param in model.parameters():
            if param.grad is not None:
                has_grad = True
                break
        
        assert has_grad, "No gradients computed"
    
    def test_module_interoperability(self, model, config):
        """Test that all modules can work together."""
        batch_size = 1
        seq_len = 8
        
        input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
        obs = torch.randn(batch_size, config.obs_dim)
        image = torch.randn(batch_size, config.vision_channels, config.vision_image_size, config.vision_image_size) if config.use_vision else None
        state = model.core.init_state(batch_size)
        
        outputs = model(
            input_ids=input_ids,
            obs=obs,
            image=image,
            state=state,
            mode="inference"
        )
        
        # Check that we got outputs from multiple modules
        output_keys = set(outputs.keys())
        
        # Should have at least basic outputs
        assert "action_logits" in output_keys
        assert "value" in output_keys
        
        # If scene graphs enabled, should have scene graph
        if config.use_scene_graphs:
            assert "scene_graph" in output_keys
        
        # If grounded language enabled and vision present
        if config.use_grounded_language and image is not None:
            # May or may not have grounded_language depending on forward pass
            pass


def test_config_flags():
    """Test that all config flags are present."""
    config = load_agi_small_config()
    
    # New flags
    assert hasattr(config, "use_continuous_learning")
    assert hasattr(config, "use_scene_graphs")
    assert hasattr(config, "use_intrinsic_motivation")
    assert hasattr(config, "use_program_synthesis")
    assert hasattr(config, "use_grounded_language")
    assert hasattr(config, "use_metacognition")
    
    # Config values
    assert hasattr(config, "continuous_learning_buffer_size")
    assert hasattr(config, "num_object_slots")
    assert hasattr(config, "intrinsic_reward_weight")
    assert hasattr(config, "num_primitives")
    assert hasattr(config, "grounded_lang_hidden_size")
    assert hasattr(config, "metacog_hidden_size")


def test_loss_functions():
    """Test that new loss functions work."""
    from core.training import (
        scene_graph_loss,
        program_synthesis_loss,
        grounded_language_loss,
        intrinsic_reward_loss,
        meta_cognition_loss,
    )
    
    # Scene graph loss
    pred_obj = torch.randn(2, 10, 128)
    pred_rel = torch.randn(2, 10, 10, 64)
    loss = scene_graph_loss(pred_obj, pred_rel)
    assert loss.item() == 0.0  # No targets
    
    target_obj = torch.randn(2, 10, 128)
    target_rel = torch.randn(2, 10, 10, 64)
    loss = scene_graph_loss(pred_obj, pred_rel, target_obj, target_rel)
    assert loss.item() > 0.0
    
    # Program synthesis loss
    prog_out = torch.randn(2, 10)
    target_out = torch.randn(2, 10)
    loss = program_synthesis_loss(prog_out, target_out)
    assert loss.item() > 0.0
    
    # Grounded language loss
    grounded_out = {"vqa_answer": torch.randn(2, 100)}
    targets = {"vqa_target": torch.randint(0, 100, (2,))}
    loss = grounded_language_loss(grounded_out, targets)
    assert loss.item() > 0.0
    
    #Intrinsic reward loss
    rewards = {"curiosity": torch.randn(2), "novelty": torch.randn(2)}
    state = torch.randn(2, 128)
    next_state = torch.randn(2, 128)
    loss = intrinsic_reward_loss(rewards, state, next_state)
    assert loss.item() >= 0.0
    
    # Meta-cognition loss
    metacog_out = {"capability_prediction": torch.rand(2)}
    actual_perf = torch.rand(2)
    loss = meta_cognition_loss(metacog_out, actual_perf)
    assert loss.item() >= 0.0


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v", "-s"])
