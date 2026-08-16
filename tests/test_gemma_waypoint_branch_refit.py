from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from semantic_3d_chat.robot.gemma_waypoint_policy import ActualGemmaWaypointPolicy
from semantic_3d_chat.training.gemma_waypoint_branch_refit import (
    REPORT_SCHEMA,
    refit_waypoint_branch,
)


def _policy() -> ActualGemmaWaypointPolicy:
    return ActualGemmaWaypointPolicy(
        hidden_size=8,
        scene_token_count=5,
        robot_token_count=2,
        history_feature_dim=3,
        max_history_tokens=4,
        head_hidden_dim=6,
        max_waypoint_step_m=0.5,
        freeze_context_projection=True,
    )


def _case(policy: ActualGemmaWaypointPolicy):
    hidden = torch.randn(8, 8, generator=torch.Generator().manual_seed(44))
    with torch.no_grad():
        reference = policy.forward_heads_from_cached_gemma_hidden(
            hidden
        ).waypoint_delta_robot_m.detach()
    actions = (0, 0, 1, 2, 0, 0, 1, 2)
    targets = reference.clone()
    targets[4] = torch.tensor([0.16, -0.08])
    targets[5] = torch.tensor([-0.12, 0.18])
    samples = tuple(
        SimpleNamespace(
            action_index=action,
            waypoint_delta_robot_m=targets[index],
        )
        for index, action in enumerate(actions)
    )
    shared = torch.tensor((True, True, True, True, False, False, True, True))
    weights = torch.where(shared, torch.ones(8), torch.full((8,), 32.0))
    return hidden, samples, reference, shared, weights


def _state(module: nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: value.detach().cpu().clone()
        for name, value in module.state_dict().items()
    }


def test_refit_updates_only_waypoint_branch_with_shared_retention() -> None:
    policy = _policy()
    hidden, samples, reference, shared, weights = _case(policy)
    before = _state(policy)
    requires_grad_before = {
        name: parameter.requires_grad for name, parameter in policy.named_parameters()
    }

    report = refit_waypoint_branch(
        policy,
        hidden,
        samples,
        sample_weights=weights,
        reference_waypoints=reference,
        retention_mask=shared,
    )

    after = _state(policy)
    assert report["schema"] == REPORT_SCHEMA
    assert report["steps"] == 300
    assert report["deterministic_full_batch"] is True
    assert report["move_to_sample_count"] == 4
    assert report["new_move_to_sample_count"] == 2
    assert report["retention_sample_count"] == 6
    assert report["shared_move_to_sample_count"] == 2
    assert report["waypoint_branch_changed"] is True
    assert report["non_waypoint_controller_tensors_unchanged"] is True
    assert report["final_new_move_error_m"]["mean"] < report[
        "initial_new_move_error_m"
    ]["mean"]
    assert report["shared_waypoint_drift_m"]["max"] is not None
    for name, expected in before.items():
        if not name.startswith("numeric_heads.waypoint."):
            assert torch.equal(after[name], expected), name
    assert any(
        not torch.equal(after[name], before[name])
        for name in after
        if name.startswith("numeric_heads.waypoint.")
    )
    assert {
        name: parameter.requires_grad for name, parameter in policy.named_parameters()
    } == requires_grad_before


def test_refit_is_bit_deterministic_on_cpu() -> None:
    first = _policy()
    second = _policy()
    first_case = _case(first)
    second_case = _case(second)

    first_report = refit_waypoint_branch(
        first,
        first_case[0],
        first_case[1],
        steps=25,
        learning_rate=3e-4,
        sample_weights=first_case[4],
        reference_waypoints=first_case[2],
        retention_mask=first_case[3],
    )
    second_report = refit_waypoint_branch(
        second,
        second_case[0],
        second_case[1],
        steps=25,
        learning_rate=3e-4,
        sample_weights=second_case[4],
        reference_waypoints=second_case[2],
        retention_mask=second_case[3],
    )

    assert first_report == second_report
    assert all(
        torch.equal(first.state_dict()[name], second.state_dict()[name])
        for name in first.state_dict()
    )


@pytest.mark.parametrize(
    ("mutation", "match"),
    (
        ("hidden_shape", "hidden must be"),
        ("hidden_nan", "hidden must be"),
        ("weights_zero", "sample_weights"),
        ("reference_shape", "reference_waypoints"),
        ("mask_dtype", "retention_mask"),
        ("missing_reference", "supplied together"),
    ),
)
def test_refit_rejects_invalid_cached_inputs_without_mutation(
    mutation: str,
    match: str,
) -> None:
    policy = _policy()
    hidden, samples, reference, shared, weights = _case(policy)
    before = _state(policy)
    kwargs: dict[str, object] = {
        "sample_weights": weights,
        "reference_waypoints": reference,
        "retention_mask": shared,
    }
    if mutation == "hidden_shape":
        hidden = hidden[:-1]
    elif mutation == "hidden_nan":
        hidden = hidden.clone()
        hidden[0, 0] = torch.nan
    elif mutation == "weights_zero":
        kwargs["sample_weights"] = weights.clone()
        kwargs["sample_weights"][0] = 0.0  # type: ignore[index]
    elif mutation == "reference_shape":
        kwargs["reference_waypoints"] = reference[:-1]
    elif mutation == "mask_dtype":
        kwargs["retention_mask"] = shared.long()
    else:
        kwargs["reference_waypoints"] = None

    with pytest.raises((TypeError, ValueError), match=match):
        refit_waypoint_branch(policy, hidden, samples, **kwargs)
    assert all(
        torch.equal(policy.state_dict()[name], expected)
        for name, expected in before.items()
    )


class _FailsAfterOneUpdate(nn.Module):
    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.linear = nn.Linear(hidden_size, 2)
        self.calls = 0

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        self.calls += 1
        output = self.linear(value)
        return output if self.calls < 3 else torch.full_like(output, torch.nan)


def test_refit_rolls_back_after_post_update_nonfinite_failure() -> None:
    policy = _policy()
    hidden, samples, reference, shared, weights = _case(policy)
    policy.numeric_heads.waypoint = _FailsAfterOneUpdate(policy.hidden_size)
    before = _state(policy)
    requires_grad_before = {
        name: parameter.requires_grad for name, parameter in policy.named_parameters()
    }
    policy.train()

    with pytest.raises(FloatingPointError, match="produced NaN"):
        refit_waypoint_branch(
            policy,
            hidden,
            samples,
            steps=5,
            sample_weights=weights,
            reference_waypoints=reference,
            retention_mask=shared,
        )

    assert policy.training is True
    assert all(
        torch.equal(policy.state_dict()[name], expected)
        for name, expected in before.items()
    )
    assert {
        name: parameter.requires_grad for name, parameter in policy.named_parameters()
    } == requires_grad_before
