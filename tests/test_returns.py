import torch

from vagi_core.returns import compute_gae, td_lambda_returns


def test_compute_gae_simple_case() -> None:
    rewards = torch.tensor([[1.0, 1.0]])
    values = torch.tensor([[0.5, 0.5, 0.0]])
    dones = torch.tensor([[0.0, 1.0]])

    advantages, returns = compute_gae(rewards, values, dones, gamma=1.0, lam=1.0)

    expected_adv = torch.tensor([[1.5, 0.5]])
    expected_ret = torch.tensor([[2.0, 1.0]])
    assert torch.allclose(advantages, expected_adv)
    assert torch.allclose(returns, expected_ret)


def test_td_lambda_returns_matches_gae() -> None:
    rewards = torch.tensor([[0.2, 0.4, 0.6]])
    values = torch.tensor([[0.1, 0.2, 0.3, 0.0]])
    dones = torch.tensor([[0.0, 0.0, 1.0]])
    returns = td_lambda_returns(rewards, values, dones, gamma=0.9, lam=0.8)
    adv, gae_returns = compute_gae(rewards, values, dones, gamma=0.9, lam=0.8)
    assert torch.allclose(returns, gae_returns)
    assert torch.allclose(returns, adv + values[:, :-1])
