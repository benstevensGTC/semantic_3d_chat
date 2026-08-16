"""Deterministic, branch-isolated waypoint refitting over cached Gemma states.

This module is intentionally narrower than the ordinary joint waypoint-policy
trainer.  It receives already-authenticated cached Gemma decision states and an
already-warm-started controller, then updates only ``numeric_heads.waypoint``.
It never runs Gemma, reads scene metadata, chooses an action, or plans a route.

The caller owns checkpoint and row-authentication policy.  In particular, the
reference waypoint tensor and retention mask should come from the authenticated
warm-start checkpoint before this function is called.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any, Final

import torch
from torch import nn

from semantic_3d_chat.robot.gemma_waypoint_policy import ACTION_NAMES

REPORT_SCHEMA: Final[str] = "semantic_3d_chat.gemma_waypoint_branch_refit.v1"
MOVE_TO_ACTION_INDEX: Final[int] = ACTION_NAMES.index("move_to")
_WAYPOINT_STATE_PREFIX: Final[str] = "numeric_heads.waypoint."


def _finite_nonnegative(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise ValueError(f"{name} must be finite and nonnegative")
    return result


def _finite_positive(value: object, *, name: str) -> float:
    result = _finite_nonnegative(value, name=name)
    if result == 0.0:
        raise ValueError(f"{name} must be positive")
    return result


def _positive_integer(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _sample_targets(
    samples: Sequence[object],
    *,
    max_waypoint_step_m: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    if not samples:
        raise ValueError("waypoint branch refit requires at least one sample")
    action_indices: list[int] = []
    waypoint_targets: list[torch.Tensor] = []
    for index, sample in enumerate(samples):
        action = getattr(sample, "action_index", None)
        if (
            isinstance(action, bool)
            or not isinstance(action, int)
            or not 0 <= action < len(ACTION_NAMES)
        ):
            raise ValueError(f"sample {index} has an invalid action_index")
        waypoint = getattr(sample, "waypoint_delta_robot_m", None)
        if (
            not isinstance(waypoint, torch.Tensor)
            or tuple(waypoint.shape) != (2,)
            or not waypoint.is_floating_point()
            or not bool(torch.isfinite(waypoint).all())
        ):
            raise ValueError(f"sample {index} has an invalid waypoint target")
        canonical = waypoint.detach().float().cpu().contiguous()
        if action == MOVE_TO_ACTION_INDEX and float(torch.linalg.vector_norm(canonical)) > (
            max_waypoint_step_m + 1e-6
        ):
            raise ValueError(f"sample {index} waypoint target exceeds the model step bound")
        action_indices.append(action)
        waypoint_targets.append(canonical)
    actions = torch.tensor(action_indices, dtype=torch.long)
    targets = torch.stack(waypoint_targets)
    if not bool((actions == MOVE_TO_ACTION_INDEX).any()):
        raise ValueError("waypoint branch refit requires at least one MOVE_TO row")
    return actions, targets


def _validated_weights(value: torch.Tensor | None, *, rows: int) -> torch.Tensor:
    if value is None:
        return torch.ones(rows, dtype=torch.float32)
    if (
        not isinstance(value, torch.Tensor)
        or tuple(value.shape) != (rows,)
        or not value.is_floating_point()
        or not bool(torch.isfinite(value).all())
        or bool((value <= 0.0).any())
    ):
        raise ValueError("waypoint branch sample_weights must be finite and positive")
    return value.detach().float().cpu().contiguous()


def _validated_retention(
    reference_waypoints: torch.Tensor | None,
    retention_mask: torch.Tensor | None,
    *,
    rows: int,
    retention_weight: float,
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    if reference_waypoints is None and retention_mask is None:
        if retention_weight != 0.0:
            raise ValueError("positive waypoint retention_weight requires a reference and mask")
        return None, None
    if reference_waypoints is None or retention_mask is None:
        raise ValueError("waypoint retention reference and mask must be supplied together")
    if (
        not isinstance(reference_waypoints, torch.Tensor)
        or tuple(reference_waypoints.shape) != (rows, 2)
        or not reference_waypoints.is_floating_point()
        or not bool(torch.isfinite(reference_waypoints).all())
    ):
        raise ValueError("reference_waypoints must be a finite [N,2] floating tensor")
    if (
        not isinstance(retention_mask, torch.Tensor)
        or retention_mask.dtype is not torch.bool
        or tuple(retention_mask.shape) != (rows,)
        or not bool(retention_mask.any())
    ):
        raise ValueError("retention_mask must select at least one of N rows")
    return (
        reference_waypoints.detach().float().cpu().contiguous(),
        retention_mask.detach().cpu().contiguous(),
    )


def _metric(values: torch.Tensor) -> dict[str, float | None]:
    canonical = values.detach().float().cpu()
    if canonical.ndim != 1 or not bool(torch.isfinite(canonical).all()):
        raise ValueError("waypoint refit metric values are invalid")
    if canonical.numel() == 0:
        return {"mean": None, "p99": None, "max": None}
    return {
        "mean": float(canonical.mean()),
        "p99": float(torch.quantile(canonical, 0.99)),
        "max": float(canonical.max()),
    }


def _state_on_cpu(module: nn.Module) -> dict[str, torch.Tensor]:
    state = {
        name: value.detach().cpu().clone()
        for name, value in module.state_dict().items()
    }
    if not state:
        raise ValueError("waypoint controller has no checkpointable state")
    return state


def _states_equal(
    observed: Mapping[str, torch.Tensor],
    expected: Mapping[str, torch.Tensor],
) -> bool:
    return set(observed) == set(expected) and all(
        torch.equal(observed[name].detach().cpu(), value)
        for name, value in expected.items()
    )


def refit_waypoint_branch(
    controller: nn.Module,
    hidden: torch.Tensor,
    samples: Sequence[object],
    *,
    steps: int = 300,
    learning_rate: float = 1e-4,
    weight_decay: float = 1e-4,
    gradient_clip_norm: float = 1.0,
    sample_weights: torch.Tensor | None = None,
    reference_waypoints: torch.Tensor | None = None,
    retention_mask: torch.Tensor | None = None,
    retention_weight: float = 10.0,
    error_tolerance_m: float = 0.025,
    device: torch.device | str = "cpu",
) -> dict[str, Any]:
    """Refit only the numeric waypoint MLP over fixed cached Gemma states.

    The supervised term is row-weighted Smooth L1 over MOVE_TO targets after
    normalization by the controller's maximum waypoint step.  The optional
    retention term is Smooth L1 against warm-start outputs over every row in
    the caller-supplied mask.  Supplying an all-action shared mask is useful:
    it constrains latent waypoint behavior even on rows whose current action is
    FACE or STOP.

    Optimization is deterministic and full-batch: there is no shuffle,
    sampling, dropout, or model forward through Gemma.  On any exception after
    mutation begins, the complete controller state is restored transactionally.
    """

    if not isinstance(controller, nn.Module):
        raise TypeError("waypoint branch refit requires an nn.Module controller")
    step_count = _positive_integer(steps, name="steps")
    rate = _finite_positive(learning_rate, name="learning_rate")
    decay = _finite_nonnegative(weight_decay, name="weight_decay")
    clip_norm = _finite_positive(gradient_clip_norm, name="gradient_clip_norm")
    retained_weight = _finite_nonnegative(retention_weight, name="retention_weight")
    tolerance = _finite_positive(error_tolerance_m, name="error_tolerance_m")
    target_device = torch.device(device)

    numeric_heads = getattr(controller, "numeric_heads", None)
    input_norm = getattr(numeric_heads, "input_norm", None)
    waypoint = getattr(numeric_heads, "waypoint", None)
    hidden_size = getattr(numeric_heads, "hidden_size", None)
    max_waypoint_step_m = getattr(numeric_heads, "max_waypoint_step_m", None)
    if (
        not isinstance(numeric_heads, nn.Module)
        or not isinstance(input_norm, nn.LayerNorm)
        or not isinstance(waypoint, nn.Module)
        or isinstance(hidden_size, bool)
        or not isinstance(hidden_size, int)
    ):
        raise TypeError("waypoint branch refit requires the standard numeric heads")
    maximum_step = _finite_positive(
        max_waypoint_step_m,
        name="numeric_heads.max_waypoint_step_m",
    )
    rows = len(samples)
    if (
        not isinstance(hidden, torch.Tensor)
        or tuple(hidden.shape) != (rows, hidden_size)
        or not hidden.is_floating_point()
        or not bool(torch.isfinite(hidden).all())
    ):
        raise ValueError(f"hidden must be a finite floating tensor with shape [{rows},{hidden_size}]")
    actions, targets = _sample_targets(samples, max_waypoint_step_m=maximum_step)
    weights = _validated_weights(sample_weights, rows=rows)
    reference, shared_mask = _validated_retention(
        reference_waypoints,
        retention_mask,
        rows=rows,
        retention_weight=retained_weight,
    )

    move_mask = actions == MOVE_TO_ACTION_INDEX
    new_move_mask = move_mask if shared_mask is None else move_mask & ~shared_mask
    shared_move_mask = (
        torch.zeros(rows, dtype=torch.bool)
        if shared_mask is None
        else move_mask & shared_mask
    )
    initial_state = _state_on_cpu(controller)
    frozen_state = {
        name: value
        for name, value in initial_state.items()
        if not name.startswith(_WAYPOINT_STATE_PREFIX)
    }
    original_requires_grad = {
        name: parameter.requires_grad for name, parameter in controller.named_parameters()
    }
    original_training = controller.training

    def restore_parameter_flags() -> None:
        for name, parameter in controller.named_parameters():
            parameter.requires_grad_(original_requires_grad[name])

    try:
        controller.to(target_device)
        controller.eval()
        controller.requires_grad_(False)
        waypoint.requires_grad_(True)
        with torch.no_grad():
            features = input_norm(hidden.detach().float().to(target_device)).detach()
        if tuple(features.shape) != (rows, hidden_size) or not bool(
            torch.isfinite(features).all()
        ):
            raise ValueError("normalized cached Gemma features are invalid")

        targets_device = targets.to(target_device)
        weights_device = weights.to(target_device)
        move_device = move_mask.to(target_device)
        reference_device = None if reference is None else reference.to(target_device)
        shared_device = None if shared_mask is None else shared_mask.to(target_device)

        def predict() -> torch.Tensor:
            output = torch.tanh(waypoint(features).float()) * maximum_step
            if tuple(output.shape) != (rows, 2) or not bool(torch.isfinite(output).all()):
                raise FloatingPointError("waypoint branch produced NaN, infinity, or wrong shape")
            return output

        def losses(prediction: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            supervised_rows = torch.nn.functional.smooth_l1_loss(
                prediction[move_device] / maximum_step,
                targets_device[move_device] / maximum_step,
                reduction="none",
            ).mean(dim=-1)
            move_weights = weights_device[move_device]
            supervised = (supervised_rows * move_weights).sum() / move_weights.sum()
            if reference_device is None or shared_device is None:
                retention = prediction.sum() * 0.0
            else:
                retention = torch.nn.functional.smooth_l1_loss(
                    prediction[shared_device] / maximum_step,
                    reference_device[shared_device] / maximum_step,
                )
            objective = supervised + retained_weight * retention
            if not bool(torch.isfinite(objective)):
                raise FloatingPointError("waypoint branch objective became non-finite")
            return supervised, retention, objective

        with torch.no_grad():
            initial_prediction_device = predict()
            initial_supervised, initial_retention, initial_objective = losses(
                initial_prediction_device
            )
            initial_prediction = initial_prediction_device.detach().cpu()

        optimizer = torch.optim.AdamW(
            waypoint.parameters(),
            lr=rate,
            weight_decay=decay,
        )
        for _step in range(step_count):
            prediction = predict()
            _supervised, _retention, objective = losses(prediction)
            optimizer.zero_grad(set_to_none=True)
            objective.backward()
            gradients = [
                parameter.grad
                for parameter in waypoint.parameters()
                if parameter.grad is not None
            ]
            if not gradients or any(not bool(torch.isfinite(value).all()) for value in gradients):
                raise FloatingPointError("waypoint branch gradients are empty or non-finite")
            torch.nn.utils.clip_grad_norm_(waypoint.parameters(), clip_norm)
            optimizer.step()
            if any(
                not bool(torch.isfinite(parameter).all())
                for parameter in waypoint.parameters()
            ):
                raise FloatingPointError("waypoint branch parameters became non-finite")

        with torch.no_grad():
            final_prediction_device = predict()
            final_supervised, final_retention, final_objective = losses(
                final_prediction_device
            )
            final_prediction = final_prediction_device.detach().cpu()

        observed_state = _state_on_cpu(controller)
        observed_frozen = {
            name: value
            for name, value in observed_state.items()
            if not name.startswith(_WAYPOINT_STATE_PREFIX)
        }
        if not _states_equal(observed_frozen, frozen_state):
            raise RuntimeError("waypoint branch refit changed a frozen controller tensor")

        all_initial_error = torch.linalg.vector_norm(
            initial_prediction[move_mask] - targets[move_mask], dim=-1
        )
        all_final_error = torch.linalg.vector_norm(
            final_prediction[move_mask] - targets[move_mask], dim=-1
        )
        new_initial_error = torch.linalg.vector_norm(
            initial_prediction[new_move_mask] - targets[new_move_mask], dim=-1
        )
        new_final_error = torch.linalg.vector_norm(
            final_prediction[new_move_mask] - targets[new_move_mask], dim=-1
        )
        shared_drift = (
            torch.empty(0, dtype=torch.float32)
            if reference is None or shared_mask is None
            else torch.linalg.vector_norm(
                final_prediction[shared_mask] - reference[shared_mask], dim=-1
            )
        )
        shared_move_drift = (
            torch.empty(0, dtype=torch.float32)
            if reference is None
            else torch.linalg.vector_norm(
                final_prediction[shared_move_mask] - reference[shared_move_mask], dim=-1
            )
        )
        new_within_tolerance = (
            None
            if new_final_error.numel() == 0
            else float((new_final_error <= tolerance).float().mean())
        )
        waypoint_changed = any(
            not torch.equal(observed_state[name], initial_state[name])
            for name in observed_state
            if name.startswith(_WAYPOINT_STATE_PREFIX)
        )
        report = {
            "schema": REPORT_SCHEMA,
            "enabled": True,
            "optimizer": "adamw_full_batch_waypoint_only",
            "deterministic_full_batch": True,
            "training_rows_only": True,
            "steps": step_count,
            "learning_rate": rate,
            "weight_decay": decay,
            "gradient_clip_norm": clip_norm,
            "retention_weight": retained_weight,
            "error_tolerance_m": tolerance,
            "sample_count": rows,
            "move_to_sample_count": int(move_mask.sum()),
            "new_move_to_sample_count": int(new_move_mask.sum()),
            "retention_sample_count": 0 if shared_mask is None else int(shared_mask.sum()),
            "shared_move_to_sample_count": int(shared_move_mask.sum()),
            "sample_weighting_enabled": sample_weights is not None,
            "retention_enabled": reference is not None,
            "retention_scope_is_caller_supplied_mask": True,
            "initial_supervised_loss": float(initial_supervised),
            "final_supervised_loss": float(final_supervised),
            "initial_retention_loss": float(initial_retention),
            "final_retention_loss": float(final_retention),
            "initial_objective": float(initial_objective),
            "final_objective": float(final_objective),
            "initial_all_move_error_m": _metric(all_initial_error),
            "final_all_move_error_m": _metric(all_final_error),
            "initial_new_move_error_m": _metric(new_initial_error),
            "final_new_move_error_m": _metric(new_final_error),
            "shared_waypoint_drift_m": _metric(shared_drift),
            "shared_move_waypoint_drift_m": _metric(shared_move_drift),
            "new_move_within_tolerance_fraction": new_within_tolerance,
            "waypoint_branch_changed": waypoint_changed,
            "non_waypoint_controller_tensors_unchanged": True,
            "parameter_requires_grad_flags_restored": True,
        }
    except Exception:
        controller.load_state_dict(initial_state, strict=True)
        restore_parameter_flags()
        controller.train(original_training)
        raise
    restore_parameter_flags()
    controller.train(original_training)
    return report


__all__ = [
    "MOVE_TO_ACTION_INDEX",
    "REPORT_SCHEMA",
    "refit_waypoint_branch",
]
