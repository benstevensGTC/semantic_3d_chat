"""Training-only expansion of the aggregate candidate-logit gate.

This module serializes per-unit and per-side evidence without question text,
scene descriptions, object metadata, geometry, or decoded token strings.  It is
an offline training diagnostic and must not be imported by ``chat`` modules.

The training loop can integrate it after collecting gate batches while keeping
the units in the exact same order as their margins::

    ordered_units.extend(batch_units)
    all_margins.append(diagnostics["margins"].detach().float().cpu())
    detail = build_candidate_gate_detail(
        ordered_units,
        torch.cat(all_margins, dim=0),
        ranking_margin=ranking_margin,
    )

That immediate form records canonical supervised targets.  If the training
loop later surfaces each side's ``(own_token_id, alternate_token_id)`` pair,
pass those pairs through ``candidate_token_ids`` to omit target text entirely.
The expected token-ID shape is ``[unit_count][2 sides][2 candidates]``.
"""

from __future__ import annotations

import math
import operator
import re
from collections.abc import Sequence
from typing import Protocol

import torch

_SCENE_ID = re.compile(r"^scene_[0-9a-f]{6,64}$")
_QUESTION_ID = re.compile(r"^q_[0-9a-f]{6,64}$")


class _TrainingRecord(Protocol):
    scene_id: str
    question_id: str
    answer: str


class TrainingPairUnit(Protocol):
    """Structural subset of ``CounterfactualPairUnit`` used by this module."""

    @property
    def records(self) -> tuple[_TrainingRecord, _TrainingRecord]: ...


CandidateTokenIds = Sequence[Sequence[Sequence[int]]]


def _validate_opaque_id(value: object, *, kind: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{kind} ID must be a string")
    pattern = _SCENE_ID if kind == "scene" else _QUESTION_ID
    if pattern.fullmatch(value) is None:
        raise ValueError(
            f"{kind} ID must be an opaque {pattern.pattern!r} identifier, got {value!r}"
        )
    return value


def _margin_tensor(
    margins: torch.Tensor | Sequence[Sequence[float]],
    *,
    unit_count: int,
) -> torch.Tensor:
    if isinstance(margins, torch.Tensor):
        values = margins.detach().to(device="cpu", dtype=torch.float32)
    else:
        values = torch.tensor(margins, dtype=torch.float32)
    if values.ndim != 2 or tuple(values.shape) != (unit_count, 2):
        raise ValueError(
            "Candidate-gate detail margins must have shape "
            f"[{unit_count}, 2], got {list(values.shape)}"
        )
    if not torch.isfinite(values).all():
        raise ValueError("Candidate-gate detail margins contain NaN or infinity")
    return values


def _token_id(value: object, *, unit_index: int, side_index: int, candidate: str) -> int:
    if isinstance(value, bool):
        raise TypeError("Candidate token IDs must be integers, not booleans")
    try:
        token_id = operator.index(value)
    except TypeError as exc:
        raise TypeError(
            f"Unit {unit_index} side {side_index} {candidate} candidate token ID must be an integer"
        ) from exc
    if token_id < 0:
        raise ValueError("Candidate token IDs cannot be negative")
    return token_id


def _normalize_candidate_token_ids(
    candidate_token_ids: CandidateTokenIds,
    *,
    unit_count: int,
) -> list[tuple[tuple[int, int], tuple[int, int]]]:
    if isinstance(candidate_token_ids, (str, bytes)) or len(candidate_token_ids) != unit_count:
        raise ValueError(
            f"candidate_token_ids must have shape [{unit_count}][2 sides][2 candidates]"
        )
    normalized: list[tuple[tuple[int, int], tuple[int, int]]] = []
    for unit_index, raw_sides in enumerate(candidate_token_ids):
        if isinstance(raw_sides, (str, bytes)) or len(raw_sides) != 2:
            raise ValueError(
                f"candidate_token_ids must have shape [{unit_count}][2 sides][2 candidates]"
            )
        sides: list[tuple[int, int]] = []
        for side_index, raw_candidates in enumerate(raw_sides):
            if isinstance(raw_candidates, (str, bytes)) or len(raw_candidates) != 2:
                raise ValueError(
                    f"candidate_token_ids must have shape [{unit_count}][2 sides][2 candidates]"
                )
            own = _token_id(
                raw_candidates[0],
                unit_index=unit_index,
                side_index=side_index,
                candidate="own",
            )
            alternate = _token_id(
                raw_candidates[1],
                unit_index=unit_index,
                side_index=side_index,
                candidate="alternate",
            )
            if own == alternate:
                raise ValueError("Own and alternate candidate token IDs must differ")
            sides.append((own, alternate))
        first, second = sides
        if first != (second[1], second[0]):
            raise ValueError(
                f"Unit {unit_index} candidate token IDs must reverse across its two sides"
            )
        normalized.append((first, second))
    return normalized


def build_candidate_gate_detail(
    units: Sequence[TrainingPairUnit],
    margins: torch.Tensor | Sequence[Sequence[float]],
    *,
    ranking_margin: float = 0.5,
    candidate_token_ids: CandidateTokenIds | None = None,
    full_vocab_margins: torch.Tensor | Sequence[Sequence[float]] | None = None,
) -> dict[str, object]:
    """Build JSON-safe candidate-gate evidence for every unit and side.

    ``margins[i, j]`` must be the same own-minus-alternate candidate-logit
    margin consumed by the aggregate gate.  Positive margins pass the existing
    binary own-candidate preference rule; margins at or above
    ``ranking_margin`` additionally clear the configured hinge target.

    When ``candidate_token_ids`` is absent, the function reads only the
    canonical ``answer`` targets from each training unit.  When token IDs are
    provided, it never reads answer text and the returned artifact contains no
    target strings. ``full_vocab_margins`` optionally records the same first-token
    target-versus-best-other evidence used by the hardened gate. Neither mode
    reads question text or any oracle geometry.
    """

    if not units:
        raise ValueError("Candidate-gate detail requires at least one pair unit")
    if not math.isfinite(ranking_margin) or ranking_margin < 0:
        raise ValueError("ranking_margin must be finite and non-negative")

    values = _margin_tensor(margins, unit_count=len(units))
    full_vocab_values = (
        None
        if full_vocab_margins is None
        else _margin_tensor(full_vocab_margins, unit_count=len(units))
    )
    token_ids = (
        None
        if candidate_token_ids is None
        else _normalize_candidate_token_ids(candidate_token_ids, unit_count=len(units))
    )
    representation = "canonical_training_targets" if token_ids is None else "candidate_token_ids"

    detail_units: list[dict[str, object]] = []
    side_preference_pass_count = 0
    side_margin_pass_count = 0
    changed_unit_pass_count = 0
    prediction_flip_count = 0
    wrong_prefix_flip_count = 0
    full_vocab_top1_side_count = 0
    full_vocab_top1_unit_count = 0

    for unit_index, unit in enumerate(units):
        records = unit.records
        if not isinstance(records, tuple) or len(records) != 2:
            raise ValueError(f"Unit {unit_index} must expose exactly two ordered records")
        scene_ids = tuple(_validate_opaque_id(record.scene_id, kind="scene") for record in records)
        question_ids = tuple(
            _validate_opaque_id(record.question_id, kind="question") for record in records
        )
        if scene_ids[0] == scene_ids[1]:
            raise ValueError(f"Unit {unit_index} must use two different opaque scene IDs")

        targets: tuple[str, str] | None = None
        if token_ids is None:
            if not all(
                isinstance(record.answer, str) and record.answer.strip() for record in records
            ):
                raise ValueError("Canonical training targets must be non-empty strings")
            targets = (records[0].answer, records[1].answer)
            if targets[0].strip() == targets[1].strip():
                raise ValueError("A changed pair unit must have two different canonical targets")

        side_records: list[dict[str, object]] = []
        side_preference_passes: list[bool] = []
        side_configured_margin_passes: list[bool] = []
        full_vocab_top1_passes: list[bool] = []
        for side_index, (scene_id, question_id) in enumerate(
            zip(scene_ids, question_ids, strict=True)
        ):
            margin = float(values[unit_index, side_index].item())
            own_preference_passed = margin > 0.0
            configured_margin_passed = margin >= ranking_margin
            predicted_preference = "own" if own_preference_passed else "alternate"
            side: dict[str, object] = {
                "side_index": side_index,
                "scene_id": scene_id,
                "question_id": question_id,
                "own_vs_alternate_candidate_logit_margin": margin,
                "predicted_preference": predicted_preference,
                "own_preference_passed": own_preference_passed,
                "configured_margin_passed": configured_margin_passed,
            }
            if full_vocab_values is not None:
                full_vocab_margin = float(full_vocab_values[unit_index, side_index].item())
                full_vocab_top1_passed = full_vocab_margin > 0.0
                side.update(
                    {
                        "first_token_target_vs_best_other_logit_margin": full_vocab_margin,
                        "full_vocab_top1_passed": full_vocab_top1_passed,
                    }
                )
                full_vocab_top1_passes.append(full_vocab_top1_passed)
            if token_ids is None:
                assert targets is not None
                own_target = targets[side_index]
                alternate_target = targets[1 - side_index]
                side.update(
                    {
                        "own_canonical_target": own_target,
                        "alternate_canonical_target": alternate_target,
                        "predicted_canonical_target": (
                            own_target if own_preference_passed else alternate_target
                        ),
                    }
                )
            else:
                own_token_id, alternate_token_id = token_ids[unit_index][side_index]
                side.update(
                    {
                        "own_candidate_token_id": own_token_id,
                        "alternate_candidate_token_id": alternate_token_id,
                        "predicted_candidate_token_id": (
                            own_token_id if own_preference_passed else alternate_token_id
                        ),
                    }
                )
            side_records.append(side)
            side_preference_passes.append(own_preference_passed)
            side_configured_margin_passes.append(configured_margin_passed)

        changed_unit_passed = all(side_preference_passes)
        # This exactly mirrors pair_gate_metrics: with two reversed candidate
        # identities, equal own/alternate decisions imply different predictions.
        prediction_flip_passed = side_preference_passes[0] == side_preference_passes[1]
        wrong_prefix_flip_passed = changed_unit_passed
        unit_margin_passed = all(side_configured_margin_passes)
        full_vocab_top1_unit_passed = (
            None if full_vocab_values is None else all(full_vocab_top1_passes)
        )
        side_preference_pass_count += sum(side_preference_passes)
        side_margin_pass_count += sum(side_configured_margin_passes)
        changed_unit_pass_count += int(changed_unit_passed)
        prediction_flip_count += int(prediction_flip_passed)
        wrong_prefix_flip_count += int(wrong_prefix_flip_passed)
        if full_vocab_values is not None:
            full_vocab_top1_side_count += sum(full_vocab_top1_passes)
            full_vocab_top1_unit_count += int(bool(full_vocab_top1_unit_passed))
        detail_unit: dict[str, object] = {
            "unit_index": unit_index,
            "scene_ids": list(scene_ids),
            "question_ids": list(question_ids),
            "changed_unit_passed": changed_unit_passed,
            "prediction_flip_passed": prediction_flip_passed,
            "wrong_prefix_flip_passed": wrong_prefix_flip_passed,
            "configured_margin_passed": unit_margin_passed,
            "sides": side_records,
        }
        if full_vocab_top1_unit_passed is not None:
            detail_unit["full_vocab_top1_unit_passed"] = full_vocab_top1_unit_passed
        detail_units.append(detail_unit)

    summary_counts = {
        "changed_units_passed": changed_unit_pass_count,
        "side_preferences_passed": side_preference_pass_count,
        "prediction_flips_passed": prediction_flip_count,
        "wrong_prefix_flips_passed": wrong_prefix_flip_count,
        "sides_at_configured_margin": side_margin_pass_count,
    }
    if full_vocab_values is not None:
        summary_counts.update(
            {
                "full_vocab_top1_sides_passed": full_vocab_top1_side_count,
                "full_vocab_top1_units_passed": full_vocab_top1_unit_count,
            }
        )
    return {
        "schema_version": 1,
        "artifact": "training_candidate_gate_detail",
        "training_only": True,
        "free_generation_evaluated": False,
        "candidate_representation": representation,
        "contains_question_text": False,
        "contains_oracle_geometry": False,
        "contains_canonical_training_targets": token_ids is None,
        "full_vocab_first_token_evaluated": full_vocab_values is not None,
        "ranking_margin": float(ranking_margin),
        "binary_prediction_rule": "margin > 0 selects own; margin <= 0 selects alternate",
        "configured_margin_rule": "margin >= ranking_margin",
        "unit_count": len(units),
        "side_count": len(units) * 2,
        "summary_counts": summary_counts,
        "units": detail_units,
    }
