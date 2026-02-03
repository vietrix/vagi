"""Loss functions for vAGI."""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
from torch.nn import functional as F


def language_loss(
    text_logits: torch.Tensor,
    labels: torch.Tensor,
    k_prefix: int = 0,
    k_suffix: int = 0,
    ignore_index: int = -100,
) -> torch.Tensor:
    """Compute cross-entropy loss for language modeling.

    Args:
        text_logits: Model output logits of shape (batch, seq_len, vocab_size).
        labels: Target token indices of shape (batch, seq_len). Must be torch.long.
        k_prefix: Number of prefix tokens to exclude from loss computation (Issue 7.2).
            Use when model output includes special prefix tokens (e.g., BOS, task
            embeddings, or prompt tokens) that should not contribute to the loss.
            Example: k_prefix=1 skips the BOS token position in loss calculation.
        k_suffix: Number of suffix tokens to exclude from loss computation (Issue 7.2).
            Use when model output includes special suffix tokens (e.g., EOS padding,
            or auxiliary output positions) that should not contribute to the loss.
            Example: k_suffix=2 excludes the last 2 positions from loss calculation.
        ignore_index: Token index to ignore in loss (default: -100, standard for
            cross_entropy). Use for padding tokens or masked positions in labels.

    Returns:
        Scalar cross-entropy loss averaged over non-ignored positions.

    Raises:
        TypeError: If labels are not torch.long dtype.
        ValueError: If k_prefix or k_suffix are negative.
        ValueError: If sliced logits and labels have mismatched sequence lengths.

    Example:
        >>> # Skip BOS token (k_prefix=1) and EOS padding (k_suffix=1)
        >>> logits = model(input_ids)  # (B, seq_len+2, vocab)
        >>> loss = language_loss(logits, labels, k_prefix=1, k_suffix=1)
    """
    if labels.dtype != torch.long:
        raise TypeError("labels must be torch.long")
    if k_prefix < 0:
        raise ValueError("k_prefix must be >= 0")
    if k_suffix < 0:
        raise ValueError("k_suffix must be >= 0")
    end = text_logits.shape[1] - k_suffix if k_suffix else text_logits.shape[1]
    logits = text_logits[:, k_prefix:end, :]
    if logits.shape[1] != labels.shape[1]:
        raise ValueError("labels length must match text token length")
    vocab_size = logits.shape[-1]
    return F.cross_entropy(
        logits.reshape(-1, vocab_size),
        labels.reshape(-1),
        ignore_index=ignore_index,
    )


def policy_loss(action_logits: torch.Tensor, action_targets: torch.Tensor) -> torch.Tensor:
    if action_targets.dtype == torch.long and action_targets.ndim in (1, 2):
        if action_targets.ndim == 2:
            action_targets = action_targets.squeeze(-1)
        return F.cross_entropy(action_logits, action_targets)
    return F.mse_loss(action_logits, action_targets)


def gaussian_nll_loss(
    mean: torch.Tensor,
    logvar: torch.Tensor,
    target: torch.Tensor,
    clamp: Tuple[float, float] = (-10.0, 10.0),
) -> torch.Tensor:
    logvar = torch.clamp(logvar, min=clamp[0], max=clamp[1])
    inv_var = torch.exp(-logvar)
    return 0.5 * torch.mean((target - mean) ** 2 * inv_var + logvar)


def value_loss(value: torch.Tensor, value_targets: torch.Tensor, logvar: Optional[torch.Tensor] = None) -> torch.Tensor:
    if logvar is None:
        return F.mse_loss(value, value_targets)
    return gaussian_nll_loss(value, logvar, value_targets)


def world_loss(
    world_pred: torch.Tensor,
    obs_next: torch.Tensor,
    logvar: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if world_pred.ndim == 3 and obs_next.ndim == 2:
        mean = world_pred[:, 0, :]
    else:
        mean = world_pred
    if logvar is None:
        return F.mse_loss(mean, obs_next)
    if logvar.ndim == 3 and obs_next.ndim == 2:
        logvar = logvar[:, 0, :]
    return gaussian_nll_loss(mean, logvar, obs_next)


def imagination_consistency_loss(
    world_pred: torch.Tensor,
    logvar: Optional[torch.Tensor] = None,
    max_delta: float = 1.0,
    adaptive_threshold: bool = True,
    base_threshold_scale: float = 1.0,
    uncertainty_sensitivity: float = 2.0,
) -> torch.Tensor:
    """Penalize implausible drift across imagined world steps.

    Issue 7.3: Added uncertainty-based adaptive threshold. When logvar is provided
    and adaptive_threshold=True, the max_delta threshold is scaled based on model
    uncertainty - higher uncertainty allows larger deltas (more exploration),
    while lower uncertainty enforces stricter consistency.

    Args:
        world_pred: World model predictions of shape (B, H, O) where H is horizon.
        logvar: Optional log-variance for uncertainty weighting (B, H, O) or (B, 1, O).
        max_delta: Base maximum allowed delta before penalty applies.
        adaptive_threshold: If True and logvar provided, scale threshold by uncertainty.
        base_threshold_scale: Multiplier for the base threshold (default 1.0).
        uncertainty_sensitivity: How strongly uncertainty affects the threshold.
            Higher values = uncertainty has more effect on threshold scaling.

    Returns:
        Scalar consistency loss penalizing large prediction jumps.
    """
    if world_pred.ndim == 2:
        world_pred = world_pred.unsqueeze(1)
    if world_pred.ndim != 3:
        raise ValueError("world_pred must have shape (B, H, O)")
    if world_pred.shape[1] < 2:
        return torch.zeros((), device=world_pred.device)

    deltas = torch.abs(world_pred[:, 1:, :] - world_pred[:, :-1, :])

    # Issue 7.3: Uncertainty-based adaptive threshold
    if logvar is not None:
        if logvar.ndim == 2:
            logvar = logvar.unsqueeze(1)
        if logvar.ndim != 3:
            raise ValueError("logvar must have shape (B, H, O)")
        if logvar.shape[1] == 1:
            logvar = logvar.expand(-1, world_pred.shape[1], -1)

        if adaptive_threshold:
            # Compute per-step uncertainty from averaged neighboring logvars
            avg_logvar = 0.5 * (logvar[:, 1:, :] + logvar[:, :-1, :])
            # Convert logvar to std: std = exp(logvar/2)
            uncertainty = torch.exp(avg_logvar * 0.5)
            # Adaptive threshold: higher uncertainty -> higher threshold
            # threshold = base * (1 + sensitivity * uncertainty)
            adaptive_max_delta = max_delta * base_threshold_scale * (
                1.0 + uncertainty_sensitivity * uncertainty
            )
            penalty = torch.relu(deltas - adaptive_max_delta)
        else:
            penalty = torch.relu(deltas - max_delta * base_threshold_scale)

        # Weight by inverse variance (certain predictions penalized more)
        weight = torch.exp(-logvar)
        weight = 0.5 * (weight[:, 1:, :] + weight[:, :-1, :])
        penalty = penalty ** 2 * weight
    else:
        penalty = torch.relu(deltas - max_delta * base_threshold_scale)
        penalty = penalty ** 2

    return torch.mean(penalty)


def total_loss(losses: Dict[str, torch.Tensor], weights: Optional[Dict[str, float]] = None) -> torch.Tensor:
    weights = weights or {}
    total = None
    for name, loss in losses.items():
        weight = weights.get(name, 1.0)
        total = loss * weight if total is None else total + loss * weight
    if total is None:
        raise ValueError("No losses to combine")
    return total


def consistency_loss(values: torch.Tensor, anchor: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Penalize drift of predicted values across imagined steps."""
    if values.ndim == 2:
        values = values.unsqueeze(-1)
    if values.ndim != 3:
        raise ValueError("values must have shape (B, K, 1) or (B, K)")
    if anchor is None:
        anchor = values[:, 0:1, :]
    if anchor.ndim == 2:
        anchor = anchor.unsqueeze(1)
    if anchor.shape[0] != values.shape[0]:
        raise ValueError("anchor batch size mismatch")
    return F.mse_loss(values, anchor.expand_as(values))


def drift_loss(values: torch.Tensor, max_delta: float = 1.0) -> torch.Tensor:
    """Penalize large step-to-step drift in value predictions."""
    if values.ndim == 2:
        values = values.unsqueeze(-1)
    if values.ndim != 3:
        raise ValueError("values must have shape (B, K, 1) or (B, K)")
    deltas = torch.abs(values[:, 1:, :] - values[:, :-1, :])
    penalty = torch.relu(deltas - max_delta)
    return torch.mean(penalty ** 2)


def representation_loss(
    anchor: torch.Tensor,
    target: torch.Tensor,
    *,
    method: str = "mse",
    temperature: float = 0.1,
) -> torch.Tensor:
    """Auxiliary loss to keep representations consistent."""
    if method == "cosine":
        anchor_norm = torch.nn.functional.normalize(anchor, dim=-1)
        target_norm = torch.nn.functional.normalize(target, dim=-1)
        similarity = (anchor_norm * target_norm).sum(dim=-1)
        return torch.mean(1.0 - similarity)
    if method == "contrastive":
        anchor_norm = torch.nn.functional.normalize(anchor, dim=-1)
        target_norm = torch.nn.functional.normalize(target, dim=-1)
        logits = torch.matmul(anchor_norm, target_norm.transpose(0, 1)) / max(temperature, 1e-6)
        labels = torch.arange(logits.shape[0], device=logits.device)
        return torch.nn.functional.cross_entropy(logits, labels)
    return F.mse_loss(anchor, target)


def reflection_loss(
    error_logits: Optional[torch.Tensor],
    error_targets: Optional[torch.Tensor],
    info_gain: Optional[torch.Tensor],
    info_targets: Optional[torch.Tensor],
) -> Dict[str, torch.Tensor]:
    """Compute reflection losses for error type and info gain (Issue 2.7).

    Args:
        error_logits: Error type prediction logits (B, num_error_types)
        error_targets: Ground truth error type indices (B,) or (B, 1)
        info_gain: Predicted information gain scores (B, 1)
        info_targets: Target information gain values (B, 1)

    Returns:
        Dictionary of loss terms that can be added to total loss
    """
    losses: Dict[str, torch.Tensor] = {}
    if error_logits is not None and error_targets is not None:
        if error_targets.dtype != torch.long:
            raise TypeError("error_targets must be torch.long")
        if error_targets.ndim == 2 and error_targets.shape[-1] == 1:
            error_targets = error_targets.squeeze(-1)
        losses["error_type"] = F.cross_entropy(error_logits, error_targets)
    if info_gain is not None and info_targets is not None:
        losses["info_gain"] = F.mse_loss(info_gain, info_targets)
    return losses


def action_validity_loss(
    action_valid: Optional[torch.Tensor],
    action_valid_targets: Optional[torch.Tensor],
    action_mask: Optional[torch.Tensor] = None,
) -> Dict[str, torch.Tensor]:
    """Compute action validity prediction loss (Issue 2.6).

    Args:
        action_valid: Predicted action validity probabilities (B, action_dim)
        action_valid_targets: Ground truth validity (B, action_dim) in [0, 1] or binary
        action_mask: Optional mask for which actions to consider

    Returns:
        Dictionary with action_validity loss if inputs are provided
    """
    losses: Dict[str, torch.Tensor] = {}
    if action_valid is not None and action_valid_targets is not None:
        if action_mask is not None:
            # Only compute loss on masked positions
            valid = action_valid[action_mask]
            targets = action_valid_targets[action_mask]
        else:
            valid = action_valid
            targets = action_valid_targets
        losses["action_validity"] = F.binary_cross_entropy(valid, targets.float())
    return losses


def temporal_consistency_loss(
    world_pred: torch.Tensor,
    world_logvar: Optional[torch.Tensor] = None,
    max_delta: float = 1.0,
    smoothness_weight: float = 0.1,
) -> torch.Tensor:
    """Temporal consistency loss for world model predictions (Issue 2.11).

    Penalizes inconsistent predictions over time by enforcing smooth transitions
    and penalizing implausible jumps.

    Args:
        world_pred: World predictions (B, horizon, obs_dim)
        world_logvar: Optional log variance for uncertainty weighting
        max_delta: Maximum allowed delta before penalty kicks in
        smoothness_weight: Weight for smoothness term

    Returns:
        Temporal consistency loss scalar
    """
    if world_pred.ndim == 2:
        world_pred = world_pred.unsqueeze(1)
    if world_pred.ndim != 3:
        raise ValueError("world_pred must have shape (B, H, O)")
    if world_pred.shape[1] < 2:
        return torch.zeros((), device=world_pred.device)

    # Compute step-to-step deltas
    deltas = world_pred[:, 1:, :] - world_pred[:, :-1, :]

    # Hard constraint: penalize jumps larger than max_delta
    large_delta_penalty = torch.relu(torch.abs(deltas) - max_delta) ** 2

    # Soft constraint: encourage smooth transitions (minimize second derivative)
    if world_pred.shape[1] >= 3:
        second_derivative = world_pred[:, 2:, :] - 2 * world_pred[:, 1:-1, :] + world_pred[:, :-2, :]
        smoothness_penalty = second_derivative ** 2
    else:
        smoothness_penalty = torch.zeros_like(large_delta_penalty)

    # Weight by uncertainty if available
    if world_logvar is not None:
        if world_logvar.ndim == 2:
            world_logvar = world_logvar.unsqueeze(1)
        if world_logvar.ndim != 3:
            raise ValueError("world_logvar must have shape (B, H, O)")
        if world_logvar.shape[1] == 1:
            world_logvar = world_logvar.expand(-1, world_pred.shape[1], -1)
        # Use inverse variance as weight (more certain = higher weight)
        weight = torch.exp(-world_logvar)
        weight_delta = 0.5 * (weight[:, 1:, :] + weight[:, :-1, :])
        large_delta_penalty = large_delta_penalty * weight_delta
        if world_pred.shape[1] >= 3:
            weight_smooth = weight[:, 1:-1, :]
            smoothness_penalty = smoothness_penalty * weight_smooth

    loss = torch.mean(large_delta_penalty)
    if world_pred.shape[1] >= 3:
        loss = loss + smoothness_weight * torch.mean(smoothness_penalty)

    return loss


def budget_loss(
    mode_logits: Optional[torch.Tensor],
    horizon_logits: Optional[torch.Tensor],
    candidate_logits: Optional[torch.Tensor],
    targets: Optional[Dict[str, torch.Tensor]],
) -> Dict[str, torch.Tensor]:
    losses: Dict[str, torch.Tensor] = {}
    if targets is None:
        return losses
    if mode_logits is not None and "budget_mode" in targets:
        losses["budget_mode"] = F.cross_entropy(mode_logits, targets["budget_mode"])
    if horizon_logits is not None and "budget_horizon" in targets:
        losses["budget_horizon"] = F.cross_entropy(horizon_logits, targets["budget_horizon"])
    if candidate_logits is not None and "budget_candidates" in targets:
        losses["budget_candidates"] = F.cross_entropy(candidate_logits, targets["budget_candidates"])
    return losses


def scene_graph_loss(
    pred_objects: torch.Tensor,
    pred_relations: torch.Tensor,
    target_objects: Optional[torch.Tensor] = None,
    target_relations: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Loss for scene graph prediction.
    
    Args:
        pred_objects: Predicted object embeddings (batch, num_objects, object_dim)
        pred_relations: Predicted relations (batch, num_objects, num_objects, relation_dim)
        target_objects: Target object embeddings (optional)
        target_relations: Target relations (optional)
    
    Returns:
        Scene graph reconstruction loss
    """
    if target_objects is None or target_relations is None:
        return torch.tensor(0.0, device=pred_objects.device)
    
    obj_loss = F.mse_loss(pred_objects, target_objects)
    rel_loss = F.mse_loss(pred_relations, target_relations)
    
    return obj_loss + 0.5 * rel_loss


def program_synthesis_loss(
    program_output: torch.Tensor,
    target_output: torch.Tensor,
    program_logits: Optional[torch.Tensor] = None,
    target_program: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Loss for program synthesis.
    
    Args:
        program_output: Output of synthesized program
        target_output: Expected output
        program_logits: Logits for program structure (optional)
        target_program: Target program structure (optional)
    
    Returns:
        Program synthesis loss
    """
    output_loss = F.mse_loss(program_output, target_output)
    
    if program_logits is not None and target_program is not None:
        structure_loss = F.cross_entropy(
            program_logits.reshape(-1, program_logits.size(-1)),
            target_program.reshape(-1)
        )
        return output_loss + 0.1 * structure_loss
    
    return output_loss


def grounded_language_loss(
    grounded_output: Dict[str, torch.Tensor],
    targets: Dict[str, torch.Tensor],
) -> torch.Tensor:
    """Loss for grounded language understanding.
    
    Args:
        grounded_output: Dictionary with model outputs
        targets: Dictionary with target values
    
    Returns:
        Total grounded language loss
    """
    total_loss = torch.tensor(0.0)
    device = next(iter(grounded_output.values())).device if grounded_output else None
    
    if device is None:
        return total_loss
    
    total_loss = total_loss.to(device)
    
    # VQA loss
    if "vqa_answer" in grounded_output and "vqa_target" in targets:
        vqa_loss = F.cross_entropy(
            grounded_output["vqa_answer"],
            targets["vqa_target"]
        )
        total_loss = total_loss + vqa_loss
    
    # Vision-language grounding loss
    if "attention_weights" in grounded_output and "grounding_target" in targets:
        grounding_loss = F.mse_loss(
            grounded_output["attention_weights"],
            targets["grounding_target"]
        )
        total_loss = total_loss + 0.5 * grounding_loss
    
    # Instruction following loss
    if "instruction_embedding" in grounded_output and "instruction_target" in targets:
        instruction_loss = F.mse_loss(
            grounded_output["instruction_embedding"],
            targets["instruction_target"]
        )
        total_loss = total_loss + 0.3 * instruction_loss
    
    return total_loss


def intrinsic_reward_loss(
    intrinsic_rewards: Dict[str, torch.Tensor],
    state: torch.Tensor,
    next_state: torch.Tensor,
) -> torch.Tensor:
    """Loss for intrinsic motivation system (regularization).
    
    Args:
        intrinsic_rewards: Dictionary with intrinsic reward components
        state: Current state
        next_state: Next state
    
    Returns:
        Intrinsic reward regularization loss
    """
    total_loss = torch.tensor(0.0, device=state.device)
    
    # Curiosity regularization (prevent collapse)
    if "curiosity" in intrinsic_rewards:
        curiosity = intrinsic_rewards["curiosity"]
        # Encourage exploration but prevent infinite growth
        reg_loss = F.relu(curiosity.mean() - 5.0)  # Cap at 5.0
        total_loss = total_loss + 0.01 * reg_loss
    
    # Novelty regularization
    if "novelty" in intrinsic_rewards:
        novelty = intrinsic_rewards["novelty"]
        # Encourage novelty detection but prevent always-novel
        reg_loss = F.relu(novelty.mean() - 3.0)  # Cap at 3.0
        total_loss = total_loss + 0.01 * reg_loss
    
    return total_loss


def meta_cognition_loss(
    metacog_output: Dict[str, torch.Tensor],
    actual_performance: torch.Tensor,
) -> torch.Tensor:
    """Loss for meta-cognition system (capability prediction).
    
    Args:
        metacog_output: Meta-cognition outputs
        actual_performance: Actual task performance (0-1)
    
    Returns:
        Meta-cognition calibration loss
    """
    if "capability_prediction" not in metacog_output:
        return torch.tensor(0.0, device=actual_performance.device)
    
    predicted_capability = metacog_output["capability_prediction"]
    
    # Calibration loss (predict actual performance)
    calibration_loss = F.mse_loss(predicted_capability, actual_performance)
    
    # Confidence penalty (prevent overconfidence)
    if "confidence" in metacog_output:
        confidence = metacog_output["confidence"]
        overconfidence_penalty = F.relu(confidence - 0.95).mean()
        calibration_loss = calibration_loss + 0.1 * overconfidence_penalty
    
    return calibration_loss
