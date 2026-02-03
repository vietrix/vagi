from core.planning import BudgetController, CounterfactualRecord


def test_budget_controller_decision_bounds() -> None:
    controller = BudgetController(max_horizon=4, max_candidates=6, min_confidence_to_act=0.9)
    decision = controller.decide(uncertainty=5.0, value_spread=0.5, task_difficulty=0.2)
    assert decision.mode == "act"
    assert decision.reason in {"needsInfo", "lowGain", "policyOnly"}


def test_budget_controller_update_changes_weights() -> None:
    controller = BudgetController()
    records = [
        CounterfactualRecord(uncertainty=0.2, value_spread=0.1, task_difficulty=0.5, delta_reward=1.0, delta_latency=0.2),
        CounterfactualRecord(uncertainty=1.0, value_spread=0.8, task_difficulty=0.5, delta_reward=-0.5, delta_latency=0.3),
    ]
    before = controller.weights.clone()
    controller.update_from_counterfactuals(records, steps=50, lr=0.1)
    assert not (controller.weights == before).all()
