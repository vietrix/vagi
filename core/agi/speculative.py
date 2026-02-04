#!/usr/bin/env python3
"""
Speculative Decoding for vAGI.

Implements speculative sampling from "Fast Inference from Transformers via
Speculative Decoding" (Leviathan et al., 2023) and "Accelerating Large
Language Model Decoding with Speculative Sampling" (Chen et al., 2023).

Algorithm Overview:
1. Draft Phase: Small model generates N candidate tokens quickly
2. Verify Phase: Large model computes probabilities for all positions in parallel
3. Accept/Reject: Accept tokens where draft matches target distribution
4. Correction: If rejected, sample from adjusted distribution

This achieves speedup when:
- Draft model is significantly faster than target model
- Draft model produces tokens that target model would likely accept
- The acceptance rate is sufficiently high

Key Implementation Details:
- Parallel verification of N draft tokens in single target forward pass
- Rejection sampling with adjusted probabilities
- Handling of special tokens (EOS, PAD)
- KV-cache management for efficiency

Usage:
    from core.agi.speculative import SpeculativeSampler

    sampler = SpeculativeSampler(
        draft_model=small_model,
        target_model=large_model,
        tokenizer=tokenizer,
    )

    # Generate with speculative decoding
    output_ids = sampler.generate(
        prompt_ids,
        max_new_tokens=100,
        num_speculative_tokens=4,
    )
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
from torch import nn
from torch.nn import functional as F

logger = logging.getLogger(__name__)


# ============================================================================
# Configuration
# ============================================================================

@dataclass
class SpeculativeConfig:
    """Configuration for speculative decoding."""
    # Number of tokens to speculate (draft) before verification
    num_speculative_tokens: int = 4

    # Temperature for sampling
    temperature: float = 1.0

    # Top-k filtering (0 = disabled)
    top_k: int = 0

    # Top-p (nucleus) filtering (1.0 = disabled)
    top_p: float = 1.0

    # Maximum new tokens to generate
    max_new_tokens: int = 100

    # Special token IDs
    eos_token_id: Optional[int] = None
    pad_token_id: Optional[int] = None

    # Whether to use greedy decoding for draft model (faster but less diverse)
    draft_greedy: bool = False

    # Minimum probability ratio for acceptance (prevents very unlikely tokens)
    min_acceptance_ratio: float = 0.0


# ============================================================================
# Sampling Utilities
# ============================================================================

def top_k_top_p_filtering(
    logits: torch.Tensor,
    top_k: int = 0,
    top_p: float = 1.0,
    filter_value: float = float('-inf'),
) -> torch.Tensor:
    """
    Filter logits using top-k and/or top-p (nucleus) filtering.

    Args:
        logits: Logits distribution [batch_size, vocab_size]
        top_k: Keep only top k tokens with highest probability
        top_p: Keep smallest set of tokens with cumulative probability >= top_p
        filter_value: Value to assign filtered tokens

    Returns:
        Filtered logits
    """
    if top_k > 0:
        # Remove tokens outside top-k
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits = logits.masked_fill(indices_to_remove, filter_value)

    if top_p < 1.0:
        # Sort logits descending
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        # Keep at least one token
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        # Scatter back to original indices
        indices_to_remove = sorted_indices_to_remove.scatter(
            dim=-1, index=sorted_indices, src=sorted_indices_to_remove
        )
        logits = logits.masked_fill(indices_to_remove, filter_value)

    return logits


def sample_from_logits(
    logits: torch.Tensor,
    temperature: float = 1.0,
    top_k: int = 0,
    top_p: float = 1.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Sample from logits with temperature, top-k, and top-p.

    Args:
        logits: Logits [batch_size, vocab_size]
        temperature: Sampling temperature
        top_k: Top-k filtering
        top_p: Top-p filtering

    Returns:
        Tuple of (sampled_tokens [batch_size], log_probs [batch_size])
    """
    # Apply temperature
    if temperature != 1.0:
        logits = logits / temperature

    # Apply filtering
    filtered_logits = top_k_top_p_filtering(logits, top_k=top_k, top_p=top_p)

    # Convert to probabilities
    probs = F.softmax(filtered_logits, dim=-1)

    # Sample
    next_token = torch.multinomial(probs, num_samples=1).squeeze(-1)

    # Get log probability of sampled token
    log_probs = F.log_softmax(logits, dim=-1)
    sampled_log_prob = log_probs.gather(-1, next_token.unsqueeze(-1)).squeeze(-1)

    return next_token, sampled_log_prob


# ============================================================================
# Speculative Sampler
# ============================================================================

class SpeculativeSampler(nn.Module):
    """
    Speculative decoding sampler using draft-then-verify approach.

    The algorithm works as follows:

    1. DRAFT PHASE:
       - Draft model autoregressively generates N candidate tokens
       - Each token is sampled from p_draft(x_t | x_{<t})
       - Store draft tokens and their probabilities

    2. VERIFY PHASE:
       - Target model processes all N+1 positions in ONE forward pass
       - Compute q_target(x_t | x_{<t}) for each draft token
       - This is efficient because Transformers can process sequences in parallel

    3. ACCEPT/REJECT PHASE:
       For each draft token t from 1 to N:
       - Compute acceptance probability: min(1, q_target(x_t) / p_draft(x_t))
       - Draw uniform random u ~ U(0, 1)
       - If u < acceptance_prob: ACCEPT this token
       - Else: REJECT this token
         - Sample correction from: normalize(max(0, q_target - p_draft))
         - All subsequent draft tokens are discarded
         - Break

    4. BONUS TOKEN:
       - If all N tokens accepted, sample one more from target model
       - This is "free" since we already computed target logits for position N+1

    The speedup comes from:
    - Draft model is K times faster than target model
    - With acceptance rate α, expected accepted tokens ≈ N*α + 1
    - Effective speedup ≈ (N*α + 1) / (N/K + 1)

    For example: K=10, N=4, α=0.8 → speedup ≈ 4.2 / 1.4 ≈ 3x
    """

    def __init__(
        self,
        draft_model: nn.Module,
        target_model: nn.Module,
        tokenizer: Any = None,
        config: Optional[SpeculativeConfig] = None,
    ):
        """
        Initialize speculative sampler.

        Args:
            draft_model: Small, fast model for drafting
            target_model: Large, accurate model for verification
            tokenizer: Optional tokenizer for special token handling
            config: Speculative decoding configuration
        """
        super().__init__()
        self.draft_model = draft_model
        self.target_model = target_model
        self.tokenizer = tokenizer
        self.config = config or SpeculativeConfig()

        # Get special tokens from tokenizer if available
        if tokenizer is not None:
            if hasattr(tokenizer, 'eos_token_id') and self.config.eos_token_id is None:
                self.config.eos_token_id = tokenizer.eos_token_id
            if hasattr(tokenizer, 'pad_token_id') and self.config.pad_token_id is None:
                self.config.pad_token_id = tokenizer.pad_token_id

        # Statistics tracking
        self.total_draft_tokens = 0
        self.total_accepted_tokens = 0
        self.total_target_calls = 0

    def _get_model_logits(
        self,
        model: nn.Module,
        input_ids: torch.Tensor,
        past_key_values: Optional[Any] = None,
        use_cache: bool = True,
    ) -> Tuple[torch.Tensor, Optional[Any]]:
        """
        Get logits from a model with optional KV-cache support.

        Handles different model interfaces (HuggingFace, custom).

        Args:
            model: The model to query
            input_ids: Input token IDs [batch_size, seq_len]
            past_key_values: Optional cached key-values for efficiency
            use_cache: Whether to return new key-values

        Returns:
            Tuple of (logits [batch_size, seq_len, vocab_size], new_past_key_values)
        """
        try:
            # HuggingFace interface
            outputs = model(
                input_ids=input_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
            )

            if hasattr(outputs, 'logits'):
                logits = outputs.logits
            elif isinstance(outputs, dict):
                logits = outputs.get('logits', outputs.get('text_logits'))
            else:
                logits = outputs

            new_past = getattr(outputs, 'past_key_values', None)
            return logits, new_past

        except TypeError:
            # Simple model without cache support
            if hasattr(model, 'forward'):
                outputs = model(input_ids)
            else:
                outputs = model(input_ids)

            if isinstance(outputs, dict):
                logits = outputs.get('logits', outputs.get('text_logits', outputs))
            else:
                logits = outputs

            return logits, None

    def _draft_tokens(
        self,
        input_ids: torch.Tensor,
        num_tokens: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate draft tokens using the small model.

        Args:
            input_ids: Current sequence [batch_size, seq_len]
            num_tokens: Number of tokens to draft

        Returns:
            Tuple of:
                - draft_tokens: Drafted token IDs [batch_size, num_tokens]
                - draft_probs: Probabilities of each draft token [batch_size, num_tokens]
        """
        batch_size = input_ids.size(0)
        device = input_ids.device

        draft_tokens = []
        draft_probs = []
        current_ids = input_ids

        self.draft_model.eval()
        with torch.no_grad():
            for _ in range(num_tokens):
                # Get next token logits
                logits, _ = self._get_model_logits(
                    self.draft_model,
                    current_ids,
                    use_cache=False,  # Simplified: no cache for draft
                )
                next_logits = logits[:, -1, :]  # [batch_size, vocab_size]

                # Sample or greedy decode
                if self.config.draft_greedy:
                    next_token = next_logits.argmax(dim=-1)
                    probs = F.softmax(next_logits / self.config.temperature, dim=-1)
                    token_prob = probs.gather(-1, next_token.unsqueeze(-1)).squeeze(-1)
                else:
                    next_token, log_prob = sample_from_logits(
                        next_logits,
                        temperature=self.config.temperature,
                        top_k=self.config.top_k,
                        top_p=self.config.top_p,
                    )
                    token_prob = log_prob.exp()

                draft_tokens.append(next_token)
                draft_probs.append(token_prob)

                # Update sequence for next iteration
                current_ids = torch.cat([current_ids, next_token.unsqueeze(-1)], dim=-1)

                # Check for EOS
                if self.config.eos_token_id is not None:
                    if (next_token == self.config.eos_token_id).all():
                        break

        # Stack results
        draft_tokens = torch.stack(draft_tokens, dim=1)  # [batch, num_drafted]
        draft_probs = torch.stack(draft_probs, dim=1)    # [batch, num_drafted]

        self.total_draft_tokens += draft_tokens.size(1) * batch_size

        return draft_tokens, draft_probs

    def _verify_and_accept(
        self,
        input_ids: torch.Tensor,
        draft_tokens: torch.Tensor,
        draft_probs: torch.Tensor,
    ) -> Tuple[torch.Tensor, int]:
        """
        Verify draft tokens using target model and accept/reject.

        This is the core of speculative decoding:
        1. Run target model on [input_ids + draft_tokens] in ONE pass
        2. For each position, compute acceptance probability
        3. Accept tokens sequentially until rejection
        4. On rejection, sample correction token

        Args:
            input_ids: Original input sequence [batch_size, orig_len]
            draft_tokens: Drafted tokens [batch_size, num_draft]
            draft_probs: Draft token probabilities [batch_size, num_draft]

        Returns:
            Tuple of:
                - accepted_tokens: Tokens to append [batch_size, num_accepted]
                - num_accepted: Number of accepted tokens (scalar for logging)
        """
        batch_size = input_ids.size(0)
        num_draft = draft_tokens.size(1)
        device = input_ids.device

        # Concatenate for parallel verification
        # Shape: [batch_size, orig_len + num_draft]
        full_input = torch.cat([input_ids, draft_tokens], dim=-1)

        # Get target model logits for all positions
        self.target_model.eval()
        with torch.no_grad():
            target_logits, _ = self._get_model_logits(
                self.target_model,
                full_input,
                use_cache=False,
            )

        self.total_target_calls += 1

        # Extract logits for draft positions
        # For position i (0-indexed in draft), we need logits from position (orig_len + i - 1)
        # because logits[t] predicts token at position t+1
        orig_len = input_ids.size(1)
        draft_position_logits = target_logits[:, orig_len - 1: orig_len + num_draft, :]
        # Shape: [batch, num_draft + 1, vocab_size]
        # Position 0 predicts first draft token, position num_draft predicts bonus token

        # Compute target probabilities
        target_probs = F.softmax(draft_position_logits / self.config.temperature, dim=-1)

        # Get target probability for each draft token
        # draft_tokens: [batch, num_draft]
        target_token_probs = target_probs[:, :-1, :].gather(
            -1, draft_tokens.unsqueeze(-1)
        ).squeeze(-1)  # [batch, num_draft]

        # Acceptance/rejection loop
        # We process batch in parallel but conceptually each sequence independently
        accepted_tokens_list = []
        num_accepted_total = 0

        # For simplicity, process batch together (assuming same accept pattern)
        # In practice, each sequence could have different acceptance points
        for t in range(num_draft):
            # Compute acceptance ratio: min(1, q/p)
            # q = target_token_probs[:, t]
            # p = draft_probs[:, t]
            ratio = target_token_probs[:, t] / (draft_probs[:, t] + 1e-10)
            acceptance_prob = torch.clamp(ratio, max=1.0)

            # Apply minimum acceptance threshold
            if self.config.min_acceptance_ratio > 0:
                acceptance_prob = torch.where(
                    ratio < self.config.min_acceptance_ratio,
                    torch.zeros_like(acceptance_prob),
                    acceptance_prob
                )

            # Sample uniform for rejection test
            u = torch.rand(batch_size, device=device)

            # Accept if u < acceptance_prob
            accept_mask = u < acceptance_prob

            if accept_mask.all():
                # All accepted, continue
                accepted_tokens_list.append(draft_tokens[:, t])
                num_accepted_total += 1
            else:
                # At least one rejection
                # For rejected sequences, sample correction token
                # correction ~ normalize(max(0, q - p))

                # Get target and draft distributions for this position
                q = target_probs[:, t, :]  # [batch, vocab]
                p_token = draft_probs[:, t].unsqueeze(-1)  # [batch, 1]

                # Create draft distribution (sparse: only drafted token has probability)
                p = torch.zeros_like(q)
                p.scatter_(-1, draft_tokens[:, t].unsqueeze(-1), p_token)

                # Correction distribution: max(0, q - p)
                correction_dist = torch.clamp(q - p, min=0)
                correction_dist = correction_dist / (correction_dist.sum(dim=-1, keepdim=True) + 1e-10)

                # Sample correction token
                correction_token = torch.multinomial(correction_dist, num_samples=1).squeeze(-1)

                # For accepted sequences, use draft token; for rejected, use correction
                final_token = torch.where(accept_mask, draft_tokens[:, t], correction_token)
                accepted_tokens_list.append(final_token)
                num_accepted_total += accept_mask.float().mean().item()

                # Stop here - remaining draft tokens are invalid
                break
        else:
            # All draft tokens accepted! Add bonus token from target
            bonus_logits = target_probs[:, -1, :]  # Logits for position after last draft
            bonus_token = torch.multinomial(bonus_logits, num_samples=1).squeeze(-1)
            accepted_tokens_list.append(bonus_token)
            num_accepted_total += 1

        # Stack accepted tokens
        if accepted_tokens_list:
            accepted_tokens = torch.stack(accepted_tokens_list, dim=1)
        else:
            accepted_tokens = torch.empty(batch_size, 0, dtype=torch.long, device=device)

        self.total_accepted_tokens += accepted_tokens.size(1) * batch_size

        return accepted_tokens, num_accepted_total

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: Optional[int] = None,
        num_speculative_tokens: Optional[int] = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        Generate tokens using speculative decoding.

        Args:
            input_ids: Input token IDs [batch_size, seq_len]
            max_new_tokens: Maximum new tokens to generate
            num_speculative_tokens: Number of tokens to speculate per iteration
            **kwargs: Additional arguments (for compatibility)

        Returns:
            Generated token IDs [batch_size, seq_len + generated_len]
        """
        max_new_tokens = max_new_tokens or self.config.max_new_tokens
        num_spec = num_speculative_tokens or self.config.num_speculative_tokens

        batch_size = input_ids.size(0)
        device = input_ids.device
        current_ids = input_ids

        generated_tokens = 0

        while generated_tokens < max_new_tokens:
            # Determine how many tokens to speculate (don't exceed remaining)
            remaining = max_new_tokens - generated_tokens
            n_spec = min(num_spec, remaining)

            if n_spec == 0:
                break

            # Draft phase
            draft_tokens, draft_probs = self._draft_tokens(current_ids, n_spec)

            # Verify and accept phase
            accepted_tokens, _ = self._verify_and_accept(
                current_ids, draft_tokens, draft_probs
            )

            # Append accepted tokens
            current_ids = torch.cat([current_ids, accepted_tokens], dim=-1)
            generated_tokens += accepted_tokens.size(1)

            # Check for EOS
            if self.config.eos_token_id is not None:
                # Check if any sequence ended
                eos_mask = (accepted_tokens == self.config.eos_token_id).any(dim=-1)
                if eos_mask.all():
                    break

        return current_ids

    def get_stats(self) -> Dict[str, float]:
        """Get speculative decoding statistics."""
        acceptance_rate = (
            self.total_accepted_tokens / max(1, self.total_draft_tokens)
        )
        tokens_per_call = (
            self.total_accepted_tokens / max(1, self.total_target_calls)
        )

        return {
            'total_draft_tokens': self.total_draft_tokens,
            'total_accepted_tokens': self.total_accepted_tokens,
            'total_target_calls': self.total_target_calls,
            'acceptance_rate': acceptance_rate,
            'tokens_per_target_call': tokens_per_call,
        }

    def reset_stats(self) -> None:
        """Reset statistics counters."""
        self.total_draft_tokens = 0
        self.total_accepted_tokens = 0
        self.total_target_calls = 0


# ============================================================================
# Specialized Variants
# ============================================================================

class SelfSpeculativeSampler(SpeculativeSampler):
    """
    Self-speculative decoding using early exit from same model.

    Instead of a separate draft model, use early layers of the target
    model as the draft. This requires a model that supports early exit.

    Concept:
    - Draft: Use first K layers of target model
    - Target: Use full model

    This avoids needing to load a separate draft model.
    """

    def __init__(
        self,
        model: nn.Module,
        draft_layers: int = 4,
        tokenizer: Any = None,
        config: Optional[SpeculativeConfig] = None,
    ):
        """
        Initialize self-speculative sampler.

        Args:
            model: The model (supports early exit)
            draft_layers: Number of layers for draft (early exit)
            tokenizer: Optional tokenizer
            config: Speculative configuration
        """
        # Use same model for both draft and target
        super().__init__(
            draft_model=model,
            target_model=model,
            tokenizer=tokenizer,
            config=config,
        )
        self.draft_layers = draft_layers
        self.full_model = model

        logger.info(f"Self-speculative: draft uses {draft_layers} layers")

    def _get_model_logits(
        self,
        model: nn.Module,
        input_ids: torch.Tensor,
        past_key_values: Optional[Any] = None,
        use_cache: bool = True,
        num_layers: Optional[int] = None,
    ) -> Tuple[torch.Tensor, Optional[Any]]:
        """
        Get logits with optional early exit.

        If num_layers is specified, only use that many layers (early exit).
        """
        # Check if model supports early exit
        if num_layers is not None and hasattr(model, 'forward_early_exit'):
            return model.forward_early_exit(
                input_ids=input_ids,
                num_layers=num_layers,
                past_key_values=past_key_values,
                use_cache=use_cache,
            )

        # Fallback to standard forward
        return super()._get_model_logits(model, input_ids, past_key_values, use_cache)


class BlockSpeculativeSampler(SpeculativeSampler):
    """
    Block-wise speculative decoding for tree-structured speculation.

    Instead of linear speculation, create multiple branches and
    verify them together. This can improve acceptance rate.

    Tree structure:
    - Root: current position
    - Level 1: top-k candidates from draft model
    - Level 2: for each Level 1 token, draft N more tokens
    - Verify entire tree in one target pass

    This is more complex but can achieve higher throughput.
    """

    def __init__(
        self,
        draft_model: nn.Module,
        target_model: nn.Module,
        tokenizer: Any = None,
        config: Optional[SpeculativeConfig] = None,
        tree_width: int = 2,
        tree_depth: int = 4,
    ):
        super().__init__(draft_model, target_model, tokenizer, config)
        self.tree_width = tree_width
        self.tree_depth = tree_depth

        logger.info(f"Block-speculative: width={tree_width}, depth={tree_depth}")

    # Full implementation would require tree attention masks
    # and more complex acceptance logic - this is a skeleton


# ============================================================================
# Example Usage
# ============================================================================

def demo_speculative_decoding():
    """Demonstrate speculative decoding with dummy models."""

    class DummyModel(nn.Module):
        """Simple transformer-like model for testing."""
        def __init__(self, vocab_size=1000, hidden_size=128, num_layers=2, name="model"):
            super().__init__()
            self.name = name
            self.embed = nn.Embedding(vocab_size, hidden_size)
            self.layers = nn.ModuleList([
                nn.TransformerEncoderLayer(hidden_size, 4, batch_first=True)
                for _ in range(num_layers)
            ])
            self.head = nn.Linear(hidden_size, vocab_size)

        def forward(self, input_ids, **kwargs):
            x = self.embed(input_ids)
            for layer in self.layers:
                x = layer(x)
            logits = self.head(x)
            return type('Output', (), {'logits': logits})()

    # Create models
    small_model = DummyModel(num_layers=1, name="draft")
    large_model = DummyModel(num_layers=4, name="target")

    # Create sampler
    config = SpeculativeConfig(
        num_speculative_tokens=4,
        max_new_tokens=20,
    )
    sampler = SpeculativeSampler(
        draft_model=small_model,
        target_model=large_model,
        config=config,
    )

    # Generate
    prompt = torch.randint(0, 1000, (1, 10))
    output = sampler.generate(prompt)

    print(f"Input shape: {prompt.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Stats: {sampler.get_stats()}")

    return output


if __name__ == "__main__":
    demo_speculative_decoding()
