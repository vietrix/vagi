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
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size

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

        # Token embedding for decoder input (during generation)
        self.token_embedding = nn.Embedding(vocab_size, hidden_size)

    def _encode_context(
        self,
        image: torch.Tensor,
        question: torch.Tensor
    ) -> torch.Tensor:
        """Encode image and question into context vector.

        Args:
            image: [B, C, H, W]
            question: [B, Q] question token IDs

        Returns:
            context: [B, 1, hidden_size]
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
        return context

    def forward(
        self,
        image: torch.Tensor,
        question: torch.Tensor,
        max_answer_length: int = 20,
        decoding_strategy: str = "greedy",
        beam_width: int = 5,
        top_p: float = 0.9,
        temperature: float = 1.0,
        eos_token_id: Optional[int] = None,
        return_all_beams: bool = False,
    ) -> torch.Tensor:
        """Answer visual question with configurable decoding strategy.

        Args:
            image: [B, C, H, W]
            question: [B, Q] question token IDs
            max_answer_length: Maximum answer length
            decoding_strategy: One of "greedy", "beam_search", or "nucleus"
            beam_width: Beam width for beam search (default: 5)
            top_p: Nucleus sampling threshold (default: 0.9)
            temperature: Softmax temperature for sampling (default: 1.0)
            eos_token_id: End of sequence token ID for early stopping
            return_all_beams: If True, return all beam candidates (beam search only)

        Returns:
            answer_logits: [B, A, vocab_size] for greedy/nucleus
                          or [B, beam_width, A] token IDs for beam search
        """
        context = self._encode_context(image, question)

        if decoding_strategy == "beam_search":
            return self._beam_search_decode(
                context,
                max_answer_length,
                beam_width,
                eos_token_id,
                return_all_beams,
            )
        elif decoding_strategy == "nucleus":
            return self._nucleus_sampling_decode(
                context,
                max_answer_length,
                top_p,
                temperature,
                eos_token_id,
            )
        else:
            # Default greedy decoding
            return self._greedy_decode(context, max_answer_length)

    def _greedy_decode(
        self,
        context: torch.Tensor,
        max_length: int,
    ) -> torch.Tensor:
        """Greedy decoding - select highest probability token at each step.

        Args:
            context: [B, 1, hidden_size] encoded context
            max_length: Maximum sequence length

        Returns:
            answer_logits: [B, max_length, vocab_size]
        """
        answer_logits = []
        hidden = None
        current_input = context

        for _ in range(max_length):
            output, hidden = self.answer_decoder(current_input, hidden)
            logits = self.output_layer(output)
            answer_logits.append(logits)

            # Use argmax token embedding as next input
            next_token = logits.argmax(dim=-1)  # [B, 1]
            current_input = self.token_embedding(next_token)

        answer_logits = torch.cat(answer_logits, dim=1)
        return answer_logits

    def _beam_search_decode(
        self,
        context: torch.Tensor,
        max_length: int,
        beam_width: int,
        eos_token_id: Optional[int],
        return_all_beams: bool,
    ) -> torch.Tensor:
        """Beam search decoding with length normalization.

        Args:
            context: [B, 1, hidden_size] encoded context
            max_length: Maximum sequence length
            beam_width: Number of beams to maintain
            eos_token_id: End of sequence token for early stopping
            return_all_beams: Whether to return all beam candidates

        Returns:
            If return_all_beams: [B, beam_width, max_length] token IDs
            Else: [B, max_length, vocab_size] logits for best beam
        """
        batch_size = context.size(0)
        device = context.device

        # Initialize beams: (score, sequence, hidden_state)
        # Expand context for beam_width beams
        context_expanded = context.repeat(1, beam_width, 1).view(
            batch_size * beam_width, 1, -1
        )

        # Initialize scores and sequences
        beam_scores = torch.zeros(batch_size, beam_width, device=device)
        beam_sequences = torch.zeros(
            batch_size, beam_width, max_length, dtype=torch.long, device=device
        )

        # Track which beams are finished
        finished_beams = torch.zeros(batch_size, beam_width, dtype=torch.bool, device=device)

        # Initialize hidden state
        hidden = None
        current_input = context_expanded

        for step in range(max_length):
            # Reshape for LSTM
            current_input_reshaped = current_input.view(batch_size * beam_width, 1, -1)

            # Decode one step
            if hidden is not None:
                h, c = hidden
                h = h.view(1, batch_size * beam_width, -1)
                c = c.view(1, batch_size * beam_width, -1)
                hidden = (h, c)

            output, hidden = self.answer_decoder(current_input_reshaped, hidden)
            logits = self.output_layer(output)  # [B*beam, 1, vocab]
            log_probs = F.log_softmax(logits.squeeze(1), dim=-1)  # [B*beam, vocab]

            # Reshape for beam operations
            log_probs = log_probs.view(batch_size, beam_width, -1)  # [B, beam, vocab]

            if step == 0:
                # First step: only use first beam (all are identical)
                next_scores = beam_scores[:, :1].unsqueeze(-1) + log_probs[:, :1, :]
                next_scores = next_scores.expand(-1, beam_width, -1)
            else:
                # Add scores from all beams
                next_scores = beam_scores.unsqueeze(-1) + log_probs  # [B, beam, vocab]

            # Flatten beam and vocab dimensions
            next_scores = next_scores.view(batch_size, -1)  # [B, beam * vocab]

            # Select top-k scores
            top_scores, top_indices = next_scores.topk(beam_width, dim=-1)  # [B, beam]

            # Convert flat indices to beam and token indices
            beam_indices = top_indices // self.vocab_size  # Which beam
            token_indices = top_indices % self.vocab_size  # Which token

            # Update sequences
            new_sequences = beam_sequences.clone()
            for b in range(batch_size):
                for k in range(beam_width):
                    src_beam = beam_indices[b, k]
                    new_sequences[b, k, :step] = beam_sequences[b, src_beam, :step]
                    new_sequences[b, k, step] = token_indices[b, k]
            beam_sequences = new_sequences

            # Update scores with length normalization
            # Using Wu et al. (2016) length penalty: ((5 + length) / 6) ^ alpha
            length_penalty = ((5.0 + step + 1) / 6.0) ** 0.6
            beam_scores = top_scores / length_penalty

            # Check for EOS
            if eos_token_id is not None:
                finished_beams = finished_beams | (token_indices == eos_token_id)
                if finished_beams.all():
                    break

            # Prepare next input
            next_tokens = token_indices.view(batch_size * beam_width)
            current_input = self.token_embedding(next_tokens).unsqueeze(1)

            # Reorder hidden states according to selected beams
            h, c = hidden
            h = h.view(1, batch_size, beam_width, -1)
            c = c.view(1, batch_size, beam_width, -1)

            new_h = torch.zeros_like(h)
            new_c = torch.zeros_like(c)
            for b in range(batch_size):
                for k in range(beam_width):
                    src_beam = beam_indices[b, k]
                    new_h[0, b, k] = h[0, b, src_beam]
                    new_c[0, b, k] = c[0, b, src_beam]

            hidden = (
                new_h.view(1, batch_size * beam_width, -1),
                new_c.view(1, batch_size * beam_width, -1)
            )

        if return_all_beams:
            return beam_sequences  # [B, beam_width, max_length]

        # Return best beam as logits (one-hot encoded)
        best_sequences = beam_sequences[:, 0, :]  # [B, max_length]
        best_logits = F.one_hot(best_sequences, num_classes=self.vocab_size).float()
        return best_logits  # [B, max_length, vocab_size]

    def _nucleus_sampling_decode(
        self,
        context: torch.Tensor,
        max_length: int,
        top_p: float,
        temperature: float,
        eos_token_id: Optional[int],
    ) -> torch.Tensor:
        """Nucleus (top-p) sampling decoding.

        Args:
            context: [B, 1, hidden_size] encoded context
            max_length: Maximum sequence length
            top_p: Cumulative probability threshold for nucleus
            temperature: Softmax temperature (higher = more random)
            eos_token_id: End of sequence token for early stopping

        Returns:
            answer_logits: [B, max_length, vocab_size]
        """
        batch_size = context.size(0)
        device = context.device

        answer_logits = []
        hidden = None
        current_input = context
        finished = torch.zeros(batch_size, dtype=torch.bool, device=device)

        for step in range(max_length):
            output, hidden = self.answer_decoder(current_input, hidden)
            logits = self.output_layer(output)  # [B, 1, vocab]
            answer_logits.append(logits)

            # Apply temperature
            scaled_logits = logits.squeeze(1) / temperature  # [B, vocab]

            # Compute probabilities
            probs = F.softmax(scaled_logits, dim=-1)  # [B, vocab]

            # Sort probabilities in descending order
            sorted_probs, sorted_indices = torch.sort(probs, dim=-1, descending=True)

            # Compute cumulative probabilities
            cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

            # Create nucleus mask: keep tokens until cumulative prob exceeds top_p
            # Shift cumulative_probs right by 1 to include the token that crosses threshold
            shifted_cumulative = torch.cat([
                torch.zeros(batch_size, 1, device=device),
                cumulative_probs[:, :-1]
            ], dim=-1)
            nucleus_mask = shifted_cumulative < top_p

            # Always keep at least one token
            nucleus_mask[:, 0] = True

            # Zero out probabilities outside nucleus
            sorted_probs = sorted_probs * nucleus_mask.float()

            # Renormalize
            sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True)

            # Sample from nucleus
            sampled_sorted_idx = torch.multinomial(sorted_probs, num_samples=1)  # [B, 1]

            # Map back to original vocabulary indices
            next_token = torch.gather(sorted_indices, dim=-1, index=sampled_sorted_idx)  # [B, 1]

            # Check for EOS
            if eos_token_id is not None:
                finished = finished | (next_token.squeeze(-1) == eos_token_id)
                if finished.all():
                    break

            # Prepare next input
            current_input = self.token_embedding(next_token)

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

        self.hidden_size = hidden_size

        # Instruction encoder (bidirectional LSTM)
        self.encoder = nn.LSTM(
            hidden_size,
            hidden_size,
            batch_first=True,
            bidirectional=True
        )

        # Token embeddings
        self.token_embeddings = nn.Embedding(vocab_size, hidden_size)

        # Action decoder - takes properly pooled bidirectional output
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

        # Global context from bidirectional pooling
        self.global_context_proj = nn.Linear(hidden_size * 2, hidden_size * 2)

    def _bidirectional_pool(
        self,
        outputs: torch.Tensor,
        hidden: Tuple[torch.Tensor, torch.Tensor],
        sequence_lengths: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Properly pool bidirectional LSTM outputs.

        For bidirectional LSTMs, the output at each timestep is the concatenation
        of forward and backward hidden states. However, for the final representation:
        - Forward pass: use the hidden state at the last timestep
        - Backward pass: use the hidden state at the first timestep

        Args:
            outputs: [B, T, hidden*2] - concatenated forward/backward outputs
            hidden: Tuple of (h_n, c_n) from LSTM
                   h_n shape: [num_layers*2, B, hidden]
            sequence_lengths: Optional [B] tensor of actual sequence lengths

        Returns:
            pooled_output: [B, hidden*2] - properly pooled bidirectional representation
            timestep_outputs: [B, T, hidden*2] - per-timestep concatenated outputs
        """
        batch_size, seq_len, _ = outputs.shape

        # Split outputs into forward and backward components
        forward_outputs = outputs[:, :, :self.hidden_size]  # [B, T, hidden]
        backward_outputs = outputs[:, :, self.hidden_size:]  # [B, T, hidden]

        # Get final hidden states from the hidden tuple
        # h_n has shape [num_layers*num_directions, B, hidden]
        # For 1-layer bidirectional: [2, B, hidden]
        # Index 0: forward final state, Index 1: backward final state
        h_n, _ = hidden

        # Forward direction: last hidden state (index 0 for layer 0)
        forward_final = h_n[0]  # [B, hidden]

        # Backward direction: last hidden state (index 1 for layer 0)
        # This corresponds to processing from the end, so it's the state after seeing first token
        backward_final = h_n[1]  # [B, hidden]

        # If sequence lengths provided, get forward output at actual last position
        if sequence_lengths is not None:
            # Gather forward outputs at actual sequence end positions
            indices = (sequence_lengths - 1).view(batch_size, 1, 1).expand(-1, -1, self.hidden_size)
            forward_final = forward_outputs.gather(1, indices).squeeze(1)  # [B, hidden]
            # Backward final is already correct (first timestep output)
            backward_final = backward_outputs[:, 0, :]  # [B, hidden]

        # Concatenate forward and backward final states
        pooled_output = torch.cat([forward_final, backward_final], dim=-1)  # [B, hidden*2]

        return pooled_output, outputs

    def forward(
        self,
        instruction: torch.Tensor,
        max_actions: int = 10,
        sequence_lengths: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Parse instruction into action sequence.

        Args:
            instruction: [B, T] token IDs
            max_actions: Maximum number of actions
            sequence_lengths: Optional [B] tensor of actual sequence lengths
                             (for proper bidirectional pooling with padded sequences)

        Returns:
            actions: [B, A, action_dim]
            segment_probs: [B, T] segmentation probabilities
        """
        # Embed tokens
        embedded = self.token_embeddings(instruction)

        # Encode instruction with bidirectional LSTM
        encoded, hidden = self.encoder(embedded)  # [B, T, hidden*2], (h_n, c_n)

        # Properly pool bidirectional outputs
        global_context, timestep_outputs = self._bidirectional_pool(
            encoded, hidden, sequence_lengths
        )

        # Apply projection to global context for enhanced representation
        global_context = self.global_context_proj(global_context)  # [B, hidden*2]

        # Predict segmentation points using per-timestep bidirectional outputs
        segment_probs = self.segment_predictor(timestep_outputs).squeeze(-1)  # [B, T]

        # Extract action segments
        # Use segments with prob > 0.5
        actions = []

        for i in range(encoded.size(0)):  # For each batch
            segments_i = (segment_probs[i] > 0.5).nonzero(as_tuple=False).squeeze(-1)

            if segments_i.dim() == 0:
                segments_i = segments_i.unsqueeze(0) if segments_i.numel() > 0 else segments_i

            if segments_i.numel() == 0:
                # Use global bidirectional context (properly pooled)
                action = self.action_decoder(global_context[i])
                actions.append(action.unsqueeze(0))
            else:
                # Extract actions at segment points using local context
                segment_actions = []
                for seg_idx in segments_i[:max_actions]:
                    # Use the bidirectional output at segment point
                    local_context = timestep_outputs[i, seg_idx]
                    action = self.action_decoder(local_context)
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

    def get_instruction_embedding(
        self,
        instruction: torch.Tensor,
        sequence_lengths: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Get a single embedding for the entire instruction.

        Uses proper bidirectional pooling to create a representation
        that captures both forward and backward context.

        Args:
            instruction: [B, T] token IDs
            sequence_lengths: Optional [B] tensor of actual sequence lengths

        Returns:
            embedding: [B, hidden*2] instruction embedding
        """
        embedded = self.token_embeddings(instruction)
        encoded, hidden = self.encoder(embedded)
        global_context, _ = self._bidirectional_pool(encoded, hidden, sequence_lengths)
        return self.global_context_proj(global_context)


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
        self.hidden_size = hidden_size

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

        # Vision-text fusion for forward pass
        self.vision_text_fusion = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size)
        )

    def forward(
        self,
        text_features: Optional[torch.Tensor] = None,
        vision_features: Optional[torch.Tensor] = None,
        image: Optional[torch.Tensor] = None,
        text: Optional[torch.Tensor] = None,
        mode: str = "inference"
    ) -> Dict[str, torch.Tensor]:
        """Forward pass for grounded language understanding.

        Args:
            text_features: Pre-computed text features [B, hidden_size]
            vision_features: Pre-computed vision features [B, V, hidden_size] or [B, hidden_size]
            image: Raw image tensor (will be encoded if vision_features not provided)
            text: Raw text tokens (will be encoded if text_features not provided)
            mode: "train" or "inference"

        Returns:
            Dictionary with grounded representations
        """
        outputs = {}

        # Encode vision if needed
        if vision_features is None and image is not None:
            vision_features = self.vision_encoder(image)

        # Encode text if needed
        if text_features is None and text is not None:
            text_features = self.text_encoder(text)
            if text_features.dim() == 3:
                text_features = text_features.mean(dim=1)

        # Handle vision features shape
        if vision_features is not None:
            if vision_features.dim() == 3:
                # [B, V, D] -> [B, D] via mean pooling
                vision_pooled = vision_features.mean(dim=1)
            else:
                vision_pooled = vision_features
            outputs["vision_pooled"] = vision_pooled

        # Handle text features shape
        if text_features is not None:
            if text_features.dim() == 3:
                text_pooled = text_features.mean(dim=1)
            else:
                text_pooled = text_features
            outputs["text_pooled"] = text_pooled

        # Fuse vision and text if both available
        if vision_features is not None and text_features is not None:
            # Ensure same dimensions
            vision_for_fusion = vision_pooled if vision_pooled.size(-1) == self.hidden_size else vision_pooled
            text_for_fusion = text_pooled if text_pooled.size(-1) == self.hidden_size else text_pooled

            # Handle dimension mismatch gracefully
            if vision_for_fusion.size(-1) != self.hidden_size:
                vision_for_fusion = F.adaptive_avg_pool1d(
                    vision_for_fusion.unsqueeze(1), self.hidden_size
                ).squeeze(1)
            if text_for_fusion.size(-1) != self.hidden_size:
                text_for_fusion = F.adaptive_avg_pool1d(
                    text_for_fusion.unsqueeze(1), self.hidden_size
                ).squeeze(1)

            # Concatenate and fuse
            combined = torch.cat([text_for_fusion, vision_for_fusion], dim=-1)
            grounded_hidden = self.vision_text_fusion(combined)
            outputs["grounded_hidden"] = grounded_hidden

            # Ground nouns in visual context (only if dimensions match)
            if vision_features.dim() == 3 and text_features.dim() >= 2:
                text_seq = text_features if text_features.dim() == 3 else text_features.unsqueeze(1)
                # Check dimension compatibility before grounding
                try:
                    grounded_entities, attention = self.grounder.ground_nouns(
                        text_seq, vision_features
                    )
                    outputs["grounded_entities"] = grounded_entities
                    outputs["attention_weights"] = attention
                except RuntimeError:
                    # Skip if dimension mismatch
                    pass

        elif text_features is not None:
            outputs["grounded_hidden"] = text_pooled
        elif vision_features is not None:
            outputs["grounded_hidden"] = vision_pooled

        return outputs
        
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
    ) -> torch.Tensor:
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
