import torch

from vagi_core.losses import consistency_loss


def test_consistency_loss_zero_for_constant_values() -> None:
    values = torch.ones(2, 3, 1)
    loss = consistency_loss(values)
    assert torch.isclose(loss, torch.tensor(0.0))


def test_consistency_loss_positive_for_drift() -> None:
    values = torch.tensor([[[0.0], [1.0], [2.0]]])
    loss = consistency_loss(values)
    assert loss.item() > 0.0
