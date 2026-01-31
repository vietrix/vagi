"""Grounded language understanding - connect language to perception and action."""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple, Any

import torch
from torch import nn
from torch.nn import functional as F


class VisionLanguageGrounder(nn.Module):
    """Ground language in visual perception."""
    
    def __init__(
        self,
        vision_dim: int,
        language_dim: int,
        grounded_dim: int = 256,
        num_heads: int = 8,
    ):
        super().__init__()
        
        # Cross-modal attention
        self.cross_attention = nn.MultiheadAttention(
            grounded_dim,
            num_heads,
            batch_first=True
        )
        
        # Project modalities to same space
        self.vision_proj = nn.Linear(vision_dim, grounded_dim)
        self.language_proj = nn.Linear(language_dim, grounded_dim)
        
        # Entity  attention (attend to specific objects)
        self.entity_attention = nn.Sequential(
            nn.Linear(grounded_dim * 2, grounded_dim),
            nn.ReLU(),
            nn.Linear(grounded_dim, 1)
        )
        
    def ground_nouns(
        self,
        text_features: torch.Tensor,
        vision_features: torch.Tensor,
        noun_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Ground noun phrases to visual entities.
        
        Args:
            text_features: [B, T, language_dim] text token features
            vision_features: [B, V, vision_dim] visual object features
            noun_mask: [B, T] mask indicating noun positions
            
        Returns:
            grounded_entities: [B, N, grounded_dim]
            attention_weights: [B, N, V]
        """
        batch_size = text_features.size(0)
        
        # Project to grounded space
        text_grounded = self.language_proj(text_features)
        vision_grounded = self.vision_proj(vision_features)
        
        # Cross-modal attention: text attends to vision
        grounded_entities, attention_weights = self.cross_attention(
            text_grounded,
            vision_grounded,
            vision_grounded
        )
        
        # Filter to nouns if mask provided
        if noun_mask is not None:
            grounded_entities = grounded_entities * noun_mask.unsqueeze(-1)
        
        return grounded_entities, attention_weights
    
    def attend(
        self,
        query: torch.Tensor,
        visual_features: torch.Tensor
    ) -> torch.Tensor:
        """Attend over visual features given textual query."""
        # Project query to grounded space
        query_grounded = self.language_proj(query)
        vision_grounded = self.vision_proj(visual_features)
        
        # Compute attention scores
        query_expanded = query_grounded.unsqueeze(1)  # [B, 1, D]
        vision_expanded = vision_grounded  # [B, V, D]
        
        # Concatenate for scoring
        combined = torch.cat([
            query_expanded.expand(-1, vision_expanded.size(1), -1),
            vision_expanded
        ], dim=-1)
        
        attention_scores = self.entity_attention(combined).squeeze(-1)
        attention_weights = F.softmax(attention_scores, dim=-1)
        
        # Weighted sum of visual features
        attended = torch.einsum('bv,bvd->bd', attention_weights, vision_expanded)
        
        return attended


class VisualQuestionAnswering(nn.Module):
    """Answer questions about images."""
    
    def __init__(
        self,
        vision_encoder: nn.Module,
        language_encoder: nn.Module,
        vocab_size: int,
        hidden_size: int = 512,
    ):
        super().__init__()
        
        self.vision_encoder = vision_encoder
        self.language_encoder = language_encoder
        
        # Vision-language grounder
        self.grounder = VisionLanguageGrounder(
            vision_dim=hidden_size,
            language_dim=hidden_size,
            grounded_dim=hidden_size
        )
        
        # Answer decoder
        self.answer_decoder = nn.LSTM(
            hidden_size,
            hidden_size,
            batch_first=True
        )
        
        self.output_layer = nn.Linear(hidden_size, vocab_size)
        
    def forward(
        self,
        image: torch.Tensor,
        question: torch.Tensor,
        max_answer_length: int = 20
    ) -> torch.Tensor:
        """Answer visual question.
        
        Args:
            image: [B, C, H, W]
            question: [B, Q] question token IDs
            max_answer_length: Maximum answer length
            
        Returns:
            answer_logits: [B, A, vocab_size]
        """
        # Encode vision
        vision_features = self.vision_encoder(image)  # [B, V, vision_dim]
        
        # Encode question
        question_features = self.language_encoder(question)  # [B, Q, lang_dim]
        if question_features.dim() == 2:
            question_features = question_features.unsqueeze(1)
        
        # Ground question in visual context
        grounded_features, _ = self.grounder.ground_nouns(
            question_features,
            vision_features
        )
        
        # Aggregate grounded features
        context = grounded_features.mean(dim=1, keepdim=True)  # [B, 1, D]
        
        # Decode answer
        answer_logits = []
        hidden = None
        
        for _ in range(max_answer_length):
            output, hidden = self.answer_decoder(context, hidden)
            logits = self.output_layer(output)
            answer_logits.append(logits)
        
        answer_logits = torch.cat(answer_logits, dim=1)
        
        return answer_logits


class InstructionParser(nn.Module):
    """Parse natural language instructions into action sequences."""
    
    def __init__(
        self,
        vocab_size: int,
        action_dim: int,
        hidden_size: int = 256,
    ):
        super().__init__()
        
        # Instruction encoder
        self.encoder = nn.LSTM(
            hidden_size,
            hidden_size,
            batch_first=True,
            bidirectional=True
        )
        
        # Token embeddings
        self.token_embeddings = nn.Embedding(vocab_size, hidden_size)
        
        # Action decoder
        self.action_decoder = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, action_dim)
        )
        
        # Temporal segmentation (identify action boundaries)
        self.segment_predictor = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
            nn.Sigmoid()
        )
        
    def forward(
        self,
        instruction: torch.Tensor,
        max_actions: int = 10
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Parse instruction into action sequence.
        
        Args:
            instruction: [B, T] token IDs
            max_actions: Maximum number of actions
            
        Returns:
            actions: [B, A, action_dim]
            segment_probs: [B, T] segmentation probabilities
        """
        # Embed tokens
        embedded = self.token_embeddings(instruction)
        
        # Encode instruction
        encoded, _ = self.encoder(embedded)  # [B, T, hidden*2]
        
        # Predict segmentation points
        segment_probs = self.segment_predictor(encoded).squeeze(-1)  # [B, T]
        
        # Extract action segments
        # Use segments with prob > 0.5
        actions = []
        
        for i in range(encoded.size(0)):  # For each batch
            segments_i = (segment_probs[i] > 0.5).nonzero(as_tuple=False).squeeze(-1)
            
            if len(segments_i) == 0:
                # Use whole sequence
                action = self.action_decoder(encoded[i].mean(dim=0))
                actions.append(action.unsqueeze(0))
            else:
                # Extract actions at segment points
                segment_actions = []
                for seg_idx in segments_i[:max_actions]:
                    action = self.action_decoder(encoded[i, seg_idx])
                    segment_actions.append(action)
                
                actions.append(torch.stack(segment_actions))
        
        # Pad to same length
        max_len = max(a.size(0) for a in actions)
        padded_actions = []
        
        for action_seq in actions:
            if action_seq.size(0) < max_len:
                padding = torch.zeros(
                    max_len - action_seq.size(0),
                    action_seq.size(-1),
                    device=action_seq.device
                )
                action_seq = torch.cat([action_seq, padding], dim=0)
            padded_actions.append(action_seq)
        
        actions_tensor = torch.stack(padded_actions)
        
        return actions_tensor, segment_probs


class GroundedLanguageModel(nn.Module):
    """Language model grounded in perception and action."""
    
    def __init__(
        self,
        vision_encoder: nn.Module,
        text_encoder: nn.Module,
        action_space_dim: int,
        vocab_size: int,
        hidden_size: int = 512,
    ):
        super().__init__()
        
        self.vision_encoder = vision_encoder
        self.text_encoder = text_encoder
        
        # Grounder
        self.grounder = VisionLanguageGrounder(
            vision_dim=hidden_size,
            language_dim=hidden_size,
            grounded_dim=hidden_size
        )
        
        # VQA module
        self.vqa = VisualQuestionAnswering(
            vision_encoder=vision_encoder,
            language_encoder=text_encoder,
            vocab_size=vocab_size,
            hidden_size=hidden_size
        )
        
        # Instruction parser
        self.instruction_parser = InstructionParser(
            vocab_size=vocab_size,
            action_dim=action_space_dim,
            hidden_size=hidden_size
        )
        
        # Action executor
        self.action_executor = nn.Sequential(
            nn.Linear(action_space_dim + hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size)
        )
        
    def answer_visual_question(
        self,
        image: torch.Tensor,
        question: torch.Tensor
    ) -> torch.Tensor:
        """Answer question about image."""
        return self.vqa(image, question)
    
    def execute_instruction(
        self,
        instruction: torch.Tensor,
        obs: torch.Tensor,
        max_steps: int = 10
    ) -> Dict[str, torch.Tensor]:
        """Execute natural language instruction.
        
        Args:
            instruction: [B, T] instruction tokens
            obs: [B, obs_dim] current observation
            max_steps: Maximum execution steps
            
        Returns:
            execution_trace: Dictionary with actions and observations
        """
        # Parse instruction to actions
        action_sequence, segment_probs = self.instruction_parser(instruction)
        
        # Execute actions
        observations = [obs]
        actions_taken = []
        
        current_obs = obs
        
        for step in range(min(max_steps, action_sequence.size(1))):
            action = action_sequence[:, step, :]
            
            # Execute action (placeholder - would interact with environment)
            # Here we just update observation using action
            action_effect = self.action_executor(
                torch.cat([action, current_obs], dim=-1)
            )
            
            current_obs = current_obs + action_effect
            
            observations.append(current_obs)
            actions_taken.append(action)
        
        return {
            'observations': torch.stack(observations, dim=1),
            'actions': torch.stack(actions_taken, dim=1),
            'segment_probs': segment_probs
        }
    
    def describe_scene(
        self,
        image: torch.Tensor,
        max_length: int = 50
    ) -> torch.Tensor:
        """Generate natural language description of scene."""
        # Encode image
        vision_features = self.vision_encoder(image)
        
        # Generate description (placeholder - would use language decoder)
        # For now, return dummy description
        batch_size = image.size(0)
        description = torch.zeros(batch_size, max_length, dtype=torch.long, device=image.device)
        
        return description


class EmbodiedLanguageLearner(nn.Module):
    """Learn language through embodied interaction."""
    
    def __init__(
        self,
        grounded_language_model: GroundedLanguageModel,
        learning_rate: float = 1e-4,
    ):
        super().__init__()
        
        self.language_model = grounded_language_model
        
        # Feedback integrator
        self.feedback_encoder = nn.Sequential(
            nn.Linear(1, 64),  # Scalar feedback
            nn.ReLU(),
            nn.Linear(64, 128)
        )
        
        # Language update network
        self.language_updater = nn.LSTM(128, 128, batch_first=True)
        
    def learn_from_feedback(
        self,
        instruction: torch.Tensor,
        execution_result: Dict[str, torch.Tensor],
        feedback: torch.Tensor
    ) -> Torch.Tensor:
        """Update language understanding from execution feedback.
        
        Args:
            instruction: [B, T] instruction tokens
            execution_result: Result from execute_instruction
            feedback: [B] scalar feedback (1=success, 0=failure)
            
        Returns:
            learning_loss: Loss for updating language model
        """
        # Encode feedback
        feedback_encoded = self.feedback_encoder(feedback.unsqueeze(-1))
        
        # Compute learning signal
        # If feedback is negative, instruction interpretation was wrong
        target_actions = execution_result['actions']
        
        # Re-parse instruction
        predicted_actions, _ = self.language_model.instruction_parser(instruction)
        
        # Compute loss weighted by feedback
        action_loss = F.mse_loss(
            predicted_actions,
            target_actions,
            reduction='none'
        ).mean(dim=-1)
        
        # Weight by feedback (learn more from failures)
        weighted_loss = action_loss * (1 - feedback).unsqueeze(-1)
        
        return weighted_loss.mean()
    
    def forward(
        self,
        instruction: torch.Tensor,
        obs: torch.Tensor,
        feedback: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """Full learning cycle."""
        # Execute instruction
        execution_result = self.language_model.execute_instruction(instruction, obs)
        
        # Learn from feedback
        learning_loss = self.learn_from_feedback(instruction, execution_result, feedback)
        
        return {
            **execution_result,
            'learning_loss': learning_loss
        }
