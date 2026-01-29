from envs.code_env.actions import (
    PlanLocateSourceAction,
    PlanPatchAction,
    PlanReadErrorsAction,
    PlanVerifyAction,
    parse_action,
    serialize_action,
)


def test_plan_actions_roundtrip() -> None:
    actions = [
        PlanReadErrorsAction(),
        PlanLocateSourceAction(),
        PlanPatchAction(),
        PlanVerifyAction(),
    ]
    for action in actions:
        text = serialize_action(action)
        parsed = parse_action(text)
        assert type(parsed) is type(action)
