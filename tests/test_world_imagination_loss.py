import torch

from vagi_core.losses import imagination_consistency_loss


def test_imagination_loss_zero_for_single_step() -> None:
    world_pred = torch.zeros((2, 1, 4))
    loss = imagination_consistency_loss(world_pred)
    assert torch.isclose(loss, torch.tensor(0.0))


def test_imagination_loss_weighted_by_uncertainty() -> None:
    world_pred = torch.tensor([[[0.0], [2.0], [4.0]]])
    low_uncertainty = torch.zeros_like(world_pred)
    high_uncertainty = torch.full_like(world_pred, 4.0)

    base_loss = imagination_consistency_loss(world_pred, low_uncertainty, max_delta=1.0)
    weighted_loss = imagination_consistency_loss(world_pred, high_uncertainty, max_delta=1.0)

    assert weighted_loss < base_loss
