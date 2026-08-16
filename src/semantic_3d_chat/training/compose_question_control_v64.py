"""Compose passed V63 values and passed V62 routing into one schema-5 candidate.

This command performs no training and does not load Gemma, QA, validation,
scorer, prediction, or oracle artifacts.  It authenticates the hash-only V62
baseline lock, two runtime-minimal controller checkpoints, and their training
reports.  A fresh V5 controller is constructed from V63's V3 value module and
only tensors below ``factorized_route`` are copied from the V62 route module.

The inherited V3 question normalization participates in V5 routing.  Exact
route preservation therefore requires V63 and V62 to have identical
``question_norm`` tensors; this composer checks that necessary condition and
fails closed.  Exact tensor identity plus deterministic randomized forwards
prove value equivalence with V63 and route-logit equivalence with V62 for all
inputs, which covers the reports' complete 576-example training population.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import torch

from semantic_3d_chat.chat.question_control_runtime import _load_control_head
from semantic_3d_chat.evaluation.predict_question_control import (
    _control_checkpoint_sha256,
)
from semantic_3d_chat.evaluation.v62_pair_disjoint_preregistration import (
    PINNED_V62_PREREGISTRATION_SHA256,
    validate_baseline_lock,
)
from semantic_3d_chat.scene_encoder.question_control_v3 import (
    TeacherBasisFullSceneQuestionControlV3,
)
from semantic_3d_chat.scene_encoder.question_control_v5 import (
    NormalizedFactorizedSceneQuestionControlV5,
)
from semantic_3d_chat.training.question_control_v5_checkpoint import (
    inherited_v60_state_sha256,
    save_v5_control_checkpoint,
)
from semantic_3d_chat.training.train_question_control_v56 import (
    _resolve,
    _safe_output_path,
    _sha256_file,
    _write_json,
)

_PINNED_FILTERED_TRAIN_SHA256: Final[str] = (
    "84b99385fadc5d06e44465ada5902f56131192298ca1539373dc3b334608cbf1"
)
_EXPECTED_TRAINING_ROWS: Final[int] = 576
_EXPECTED_TRAINING_SCENES: Final[int] = 24
_EXPECTED_TRAINING_PAIRS: Final[int] = 12
_EXPECTED_CHANGED_SIDES: Final[int] = 80
_EXPECTED_CHANGED_UNITS: Final[int] = 40
_RANDOMIZED_PROBE_COUNT: Final[int] = 3
_RANDOMIZED_PROBE_SEED: Final[int] = 640064


@dataclass(frozen=True)
class V64Sources:
    baseline_lock_sha256: str
    baseline_authorization: dict[str, Any]
    value_checkpoint: Path
    value_checkpoint_sha256: str
    value_control: TeacherBasisFullSceneQuestionControlV3
    value_metadata: dict[str, Any]
    value_report: dict[str, Any]
    value_report_path: Path
    value_report_sha256: str
    route_checkpoint: Path
    route_checkpoint_sha256: str
    route_control: NormalizedFactorizedSceneQuestionControlV5
    route_metadata: dict[str, Any]
    route_report: dict[str, Any]
    route_report_path: Path
    route_report_sha256: str
    output_checkpoint: Path
    composition_report: Path


def _strict_report(path: str | Path, *, label: str) -> tuple[dict[str, Any], Path, str]:
    source = _resolve(path)
    current = Path(source.anchor)
    for component in source.parts[1:]:
        current /= component
        if current.is_symlink():
            raise ValueError(f"V64 {label} path contains a symlink: {current}")
    if not source.is_file() or source.suffix.casefold() != ".json":
        raise FileNotFoundError(f"V64 {label} is unavailable: {source}")
    raw = source.read_bytes()
    try:
        value = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"V64 {label} is not valid UTF-8 JSON") from exc
    if not isinstance(value, dict):
        raise TypeError(f"V64 {label} must be a JSON object")
    return value, source, hashlib.sha256(raw).hexdigest()


def _all_true(value: object, *, label: str) -> bool:
    if not isinstance(value, Mapping) or not value:
        raise ValueError(f"V64 {label} must be a nonempty check mapping")
    if any(type(item) is not bool for item in value.values()):
        raise TypeError(f"V64 {label} checks must be strict booleans")
    return all(value.values())


def _digest(value: object, *, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"V64 {label} must be a lowercase SHA-256 digest")
    return value


def _checkpoint_report_matches(
    report: Mapping[str, Any], metadata: Mapping[str, Any], *, label: str
) -> None:
    checkpoint = report.get("checkpoint")
    if not isinstance(checkpoint, Mapping):
        raise TypeError(f"V64 {label} report lacks a published checkpoint mapping")
    if checkpoint.get("weights_sha256") != metadata.get("weights_sha256"):
        raise ValueError(f"V64 {label} report/checkpoint weights hash differs")
    _digest(
        checkpoint.get("runtime_metadata_sha256"),
        label=f"{label} report runtime metadata",
    )


def validate_v62_route_report(
    report: Mapping[str, Any], metadata: Mapping[str, Any]
) -> dict[str, str | int]:
    """Validate all train-only route gates and return common provenance."""

    authorization = report.get("authorization")
    source = report.get("source")
    base = report.get("base")
    inputs = report.get("inputs")
    architecture = report.get("architecture")
    cv = report.get("cross_validation")
    final_fit = report.get("final_fit")
    scope = report.get("scope")
    if (
        report.get("artifact") != "v62_normalized_factorized_route_training"
        or report.get("offline_checks_passed") is not True
        or report.get("promotion_eligible") is not False
        or report.get("terminal_reason")
        != "train_only_gates_passed_checkpoint_saved"
        or not all(
            isinstance(item, Mapping)
            for item in (
                authorization,
                source,
                base,
                inputs,
                architecture,
                cv,
                final_fit,
                scope,
            )
        )
    ):
        raise ValueError("V64 V62 route report did not reach its passed terminal")
    aggregate = cv.get("aggregate")
    if (
        cv.get("method") != "deterministic_leave_one_counterfactual_pair_out"
        or cv.get("fold_count") != _EXPECTED_TRAINING_PAIRS
        or cv.get("pair_disjoint") is not True
        or not isinstance(cv.get("folds"), list)
        or len(cv["folds"]) != _EXPECTED_TRAINING_PAIRS
        or not isinstance(aggregate, Mapping)
        or aggregate.get("passed") is not True
        or not _all_true(aggregate.get("checks"), label="V62 route CV")
        or not _all_true(final_fit.get("checks"), label="V62 route final fit")
    ):
        raise ValueError("V64 V62 pair-disjoint route evidence failed")
    if (
        inputs.get("filtered_training_qa_sha256") != _PINNED_FILTERED_TRAIN_SHA256
        or inputs.get("natural_row_count") != _EXPECTED_TRAINING_ROWS
        or inputs.get("natural_changed_count") != _EXPECTED_CHANGED_SIDES
        or not isinstance(inputs.get("training_scene_ids"), list)
        or len(inputs["training_scene_ids"]) != _EXPECTED_TRAINING_SCENES
        or not isinstance(inputs.get("training_pair_ids"), list)
        or len(inputs["training_pair_ids"]) != _EXPECTED_TRAINING_PAIRS
        or architecture.get("name")
        != "normalized_factorized_scene_question_route_v5"
        or architecture.get("route_factor_rank") != metadata.get("route_factor_rank")
        or source.get("v60_checkpoint_sha256")
        != metadata.get("source_v60_checkpoint_sha256")
        or source.get("v60_state_sha256")
        != metadata.get("inherited_value_state_sha256")
    ):
        raise ValueError("V64 V62 route inventory or architecture changed")
    required_scope = {
        "gemma_backward_used": False,
        "gemma_generation_used": False,
        "only_factorized_route_trained": True,
        "v60_values_frozen": True,
        "question_dependent_scene_retrieval": False,
        "internal_validation_questions_loaded": False,
        "scorer_references_loaded": False,
        "prediction_answers_loaded": False,
        "oracle_loaded": False,
        "fresh_development_loaded": False,
        "deferred_final_loaded": False,
    }
    if any(scope.get(key) is not expected for key, expected in required_scope.items()):
        raise ValueError("V64 V62 route scope is not training-only")
    _checkpoint_report_matches(report, metadata, label="V62 route")
    return {
        "baseline_lock_sha256": _digest(
            authorization.get("baseline_lock_sha256"), label="V62 baseline lock"
        ),
        "preregistration_sha256": _digest(
            authorization.get("preregistration_sha256"),
            label="V62 preregistration",
        ),
        "base_checkpoint_sha256": _digest(
            base.get("checkpoint_sha256"), label="V62 base checkpoint"
        ),
        "runtime_config_sha256": _digest(
            base.get("runtime_config_effective_sha256"),
            label="V62 runtime config",
        ),
        "filtered_training_qa_sha256": _digest(
            inputs.get("filtered_training_qa_sha256"),
            label="V62 filtered training QA",
        ),
        "prefix_cache_manifest_sha256": _digest(
            inputs.get("prefix_cache_manifest_sha256"),
            label="V62 prefix-cache manifest",
        ),
        "source_v60_checkpoint_sha256": _digest(
            source.get("v60_checkpoint_sha256"),
            label="V62 source V60 checkpoint",
        ),
        "training_scene_count": len(inputs["training_scene_ids"]),
        "training_pair_count": len(inputs["training_pair_ids"]),
        "training_row_count": int(inputs["natural_row_count"]),
    }


def validate_v63_value_report(
    report: Mapping[str, Any], metadata: Mapping[str, Any]
) -> dict[str, str | int]:
    """Validate passed pair-disjoint value reconstruction and provenance."""

    authorization = report.get("authorization")
    source = report.get("source_v60")
    base = report.get("base")
    inputs = report.get("inputs")
    architecture = report.get("architecture")
    cv = report.get("cross_validation")
    final_fit = report.get("final_fit")
    scope = report.get("scope")
    if (
        report.get("artifact") != "v63_pair_disjoint_expanded_value_distillation"
        or report.get("offline_checks_passed") is not True
        or report.get("promotion_eligible") is not False
        or report.get("successor_factorized_route_required") is not True
        or not all(
            isinstance(item, Mapping)
            for item in (
                authorization,
                source,
                base,
                inputs,
                architecture,
                cv,
                final_fit,
                scope,
            )
        )
    ):
        raise ValueError("V64 V63 value report did not reach its passed terminal")
    if (
        cv.get("protocol") != "deterministic_leave_one_counterfactual_pair_out"
        or cv.get("pair_count") != _EXPECTED_TRAINING_PAIRS
        or cv.get("fold_specific_output_basis") is not True
        or cv.get("heldout_teacher_used_in_fold_basis") is not False
        or cv.get("heldout_teacher_used_in_fold_optimization") is not False
        or cv.get("each_teacher_evaluated_once") is not True
        or cv.get("passed") is not True
        or not _all_true(cv.get("checks"), label="V63 value CV")
        or not _all_true(final_fit.get("checks"), label="V63 value final fit")
    ):
        raise ValueError("V64 V63 pair-disjoint value evidence failed")
    aggregate = cv.get("aggregate")
    final_summary = final_fit.get("summary")
    if (
        not isinstance(aggregate, Mapping)
        or aggregate.get("teacher_side_count") != _EXPECTED_CHANGED_SIDES
        or aggregate.get("changed_pair_unit_count") != _EXPECTED_CHANGED_UNITS
        or not isinstance(final_summary, Mapping)
        or final_summary.get("teacher_side_count") != _EXPECTED_CHANGED_SIDES
        or final_summary.get("changed_pair_unit_count") != _EXPECTED_CHANGED_UNITS
        or inputs.get("filtered_training_qa_sha256") != _PINNED_FILTERED_TRAIN_SHA256
        or inputs.get("training_record_count") != _EXPECTED_TRAINING_ROWS
        or inputs.get("counterfactual_pair_count") != _EXPECTED_TRAINING_PAIRS
        or inputs.get("changed_teacher_side_count") != _EXPECTED_CHANGED_SIDES
        or inputs.get("changed_paired_unit_count") != _EXPECTED_CHANGED_UNITS
        or not isinstance(inputs.get("training_scene_ids"), list)
        or len(inputs["training_scene_ids"]) != _EXPECTED_TRAINING_SCENES
    ):
        raise ValueError("V64 V63 value reconstruction inventory changed")
    if (
        architecture.get("name")
        != "teacher_basis_full_scene_question_control_v3"
        or architecture.get("runtime_schema_version") != 3
        or architecture.get("hidden_size") != metadata.get("hidden_size")
        or architecture.get("control_tokens") != metadata.get("control_tokens")
        or architecture.get("basis_rank_effective")
        != metadata.get("output_basis_rank")
        or architecture.get("scene_moment_count") != metadata.get("moment_count")
        or architecture.get("all_256_scene_latents_used") is not True
        or architecture.get("source_v60_question_norm_frozen") is not True
        or source.get("architecture")
        != "teacher_basis_full_scene_question_control_v3"
        or source.get("question_norm_copied_tensor_exact") is not True
        or source.get("question_norm_frozen_in_every_fit") is not True
    ):
        raise ValueError("V64 V63 report/checkpoint architecture differs")
    required_scope = {
        "gemma_backward_used": False,
        "gemma_generation_used": False,
        "numeric_teacher_cache_only": True,
        "teacher_cache_runtime_access": False,
        "complete_scene_prefix_retained": True,
        "question_dependent_scene_retrieval": False,
        "validation_inputs_used": False,
        "scorer_inputs_used": False,
        "oracle_loaded": False,
        "fresh_development_loaded": False,
        "deferred_final_loaded": False,
        "report_contains_question_or_answer_text": False,
    }
    if any(scope.get(key) is not expected for key, expected in required_scope.items()):
        raise ValueError("V64 V63 value scope is not training-only")
    _checkpoint_report_matches(report, metadata, label="V63 value")
    return {
        "baseline_lock_sha256": _digest(
            authorization.get("baseline_lock_sha256"), label="V63 baseline lock"
        ),
        "preregistration_sha256": _digest(
            authorization.get("preregistration_sha256"),
            label="V63 preregistration",
        ),
        "base_checkpoint_sha256": _digest(
            base.get("checkpoint_sha256"), label="V63 base checkpoint"
        ),
        "runtime_config_sha256": _digest(
            base.get("runtime_config_effective_sha256"),
            label="V63 runtime config",
        ),
        "filtered_training_qa_sha256": _digest(
            inputs.get("filtered_training_qa_sha256"),
            label="V63 filtered training QA",
        ),
        "prefix_cache_manifest_sha256": _digest(
            inputs.get("prefix_cache_manifest_sha256"),
            label="V63 prefix-cache manifest",
        ),
        "source_v60_checkpoint_sha256": _digest(
            source.get("checkpoint_sha256"),
            label="V63 source V60 checkpoint",
        ),
        "training_scene_count": len(inputs["training_scene_ids"]),
        "training_pair_count": int(inputs["counterfactual_pair_count"]),
        "training_row_count": int(inputs["training_record_count"]),
    }


def _compatible_dimensions(
    value_metadata: Mapping[str, Any], route_metadata: Mapping[str, Any]
) -> dict[str, int]:
    dimensions = {
        "hidden_size": int(value_metadata.get("hidden_size", -1)),
        "control_tokens": int(value_metadata.get("control_tokens", -1)),
        "expected_environment_latents": int(
            value_metadata.get("expected_environment_latents", -1)
        ),
        "moment_count": int(value_metadata.get("moment_count", -1)),
    }
    if any(value < 1 for value in dimensions.values()):
        raise ValueError("V64 V63 value checkpoint dimensions are invalid")
    if any(route_metadata.get(field) != value for field, value in dimensions.items()):
        raise ValueError(
            "V64 V62 route and V63 value checkpoints have incompatible shared dimensions"
        )
    route_rank = route_metadata.get("route_factor_rank")
    if isinstance(route_rank, bool) or not isinstance(route_rank, int) or route_rank < 1:
        raise ValueError("V64 route-factor rank is invalid")
    dimensions["route_factor_rank"] = route_rank
    return dimensions


def _state_sha256(state: Mapping[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name, raw in state.items():
        value = raw.detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str(tuple(value.shape)).encode())
        digest.update(str(value.dtype).encode())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def _question_norm_state(
    module: TeacherBasisFullSceneQuestionControlV3,
) -> dict[str, torch.Tensor]:
    return {
        name: value.detach().cpu().clone()
        for name, value in module.state_dict().items()
        if name.startswith("question_norm.")
    }


def _question_norm_sha256(
    module: TeacherBasisFullSceneQuestionControlV3,
) -> str:
    state = {
        name.removeprefix("question_norm."): value
        for name, value in module.state_dict().items()
        if name.startswith("question_norm.")
    }
    digest = hashlib.sha256()
    for name in sorted(state):
        value = state[name].detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str(tuple(value.shape)).encode())
        digest.update(str(value.dtype).encode())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def compose_controller(
    value_control: TeacherBasisFullSceneQuestionControlV3,
    route_control: NormalizedFactorizedSceneQuestionControlV5,
) -> tuple[NormalizedFactorizedSceneQuestionControlV5, dict[str, Any]]:
    """Copy exactly one route submodule over an exact V63 inherited state."""

    if type(value_control) is not TeacherBasisFullSceneQuestionControlV3:
        raise TypeError("V64 value source must be the exact V3 architecture")
    if type(route_control) is not NormalizedFactorizedSceneQuestionControlV5:
        raise TypeError("V64 route source must be the exact V5 architecture")
    shared = (
        "hidden_size",
        "control_token_count",
        "expected_environment_latents",
        "moment_count",
    )
    if any(getattr(value_control, field) != getattr(route_control, field) for field in shared):
        raise ValueError("V64 controller source dimensions differ")
    value_question_norm = _question_norm_state(value_control)
    route_question_norm = _question_norm_state(route_control)
    if set(value_question_norm) != set(route_question_norm) or any(
        not torch.equal(value_question_norm[name], route_question_norm[name])
        for name in value_question_norm
    ):
        raise ValueError(
            "V64 cannot copy only factorized_route: inherited question_norm differs"
        )

    value_state = {
        name: tensor.detach().cpu().clone()
        for name, tensor in value_control.state_dict().items()
    }
    route_state = {
        name: tensor.detach().cpu().clone()
        for name, tensor in route_control.factorized_route.state_dict().items()
    }
    candidate = NormalizedFactorizedSceneQuestionControlV5.from_v60(
        value_control,
        route_factor_rank=route_control.route_factor_rank,
    ).cpu().float()
    candidate.factorized_route.load_state_dict(route_state, strict=True)
    candidate.freeze_inherited_v60_state()
    candidate.eval()

    inherited_exact = (
        set(value_state) == set(candidate.inherited_state_names)
        and all(
            torch.equal(value_state[name], candidate.state_dict()[name])
            for name in value_state
        )
    )
    copied_route = candidate.factorized_route.state_dict()
    route_exact = set(route_state) == set(copied_route) and all(
        torch.equal(route_state[name], copied_route[name]) for name in route_state
    )
    optimizer_scope = {
        name for name, parameter in candidate.named_parameters() if parameter.requires_grad
    }
    if (
        not inherited_exact
        or not route_exact
        or not candidate.inherited_v60_state_frozen
        or optimizer_scope
        != {
            name
            for name, _parameter in candidate.named_parameters()
            if name.startswith("factorized_route.")
        }
    ):
        raise RuntimeError("V64 tensor-isolated composition proof failed")
    proof = {
        "construction": "V5.from_v60(V63); copy factorized_route only",
        "copied_state_prefixes": ["factorized_route."],
        "copied_route_tensor_count": len(route_state),
        "inherited_v63_tensor_count": len(value_state),
        "inherited_v63_state_sha256": _state_sha256(value_state),
        "candidate_inherited_state_sha256": inherited_v60_state_sha256(candidate),
        "route_source_state_sha256": _state_sha256(route_state),
        "candidate_route_state_sha256": _state_sha256(copied_route),
        "inherited_v63_tensors_exact": True,
        "inherited_v63_state_frozen": True,
        "factorized_route_tensors_exact": True,
        "only_factorized_route_trainable": True,
        "question_norm_exact_required_and_verified": True,
    }
    return candidate, proof


def randomized_forward_equivalence(
    *,
    value_control: TeacherBasisFullSceneQuestionControlV3,
    route_control: NormalizedFactorizedSceneQuestionControlV5,
    candidate: NormalizedFactorizedSceneQuestionControlV5,
    probe_count: int = _RANDOMIZED_PROBE_COUNT,
    seed: int = _RANDOMIZED_PROBE_SEED,
) -> dict[str, Any]:
    """Exercise both equality claims on deterministic continuous inputs."""

    if isinstance(probe_count, bool) or not isinstance(probe_count, int) or probe_count < 1:
        raise ValueError("V64 randomized proof requires a positive probe count")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    value_exact = True
    route_exact = True
    signature_exact = True
    for _ordinal in range(probe_count):
        prefix = torch.randn(
            2,
            value_control.expected_environment_latents + 2,
            value_control.hidden_size,
            generator=generator,
        )
        question = torch.randn(
            2,
            5,
            value_control.hidden_size,
            generator=generator,
        )
        with torch.inference_mode():
            value_signature = value_control.encode_scene(prefix)
            route_signature = route_control.encode_scene(prefix)
            candidate_signature = candidate.encode_scene(prefix)
            source_value = value_control.forward_from_signature(
                value_signature, question
            )
            source_route = route_control.route_logits_from_signature(
                route_signature, question
            )
            composed = candidate.forward_from_signature(candidate_signature, question)
        signature_exact = signature_exact and torch.equal(
            value_signature, candidate_signature
        ) and torch.equal(route_signature, candidate_signature)
        value_exact = value_exact and all(
            torch.equal(left, right)
            for left, right in (
                (source_value.control_tokens, composed.control_tokens),
                (
                    source_value.coefficient_directions,
                    composed.coefficient_directions,
                ),
                (source_value.control_rms, composed.control_rms),
            )
        )
        route_exact = route_exact and torch.equal(source_route, composed.gate_logits)
    if not signature_exact or not value_exact or not route_exact:
        raise RuntimeError("V64 deterministic randomized forward proof failed")
    return {
        "seed": seed,
        "probe_count": probe_count,
        "batch_size_per_probe": 2,
        "question_token_count_per_probe": 5,
        "scene_signatures_exact": True,
        "value_outputs_exact": True,
        "route_logits_exact": True,
    }


def _paths_overlap(left: Path, right: Path) -> bool:
    return left == right or left in right.parents or right in left.parents


def _output_paths(
    *,
    checkpoint: str | Path,
    report: str | Path,
    inputs: Sequence[Path],
) -> tuple[Path, Path]:
    output_checkpoint = _resolve(checkpoint)
    output_report = _resolve(report)
    if output_checkpoint.exists() or output_report.exists():
        raise FileExistsError("V64 create-once output already exists")
    if _paths_overlap(output_checkpoint, output_report):
        raise ValueError("V64 checkpoint and report outputs must be disjoint")
    for source in inputs:
        if _paths_overlap(output_checkpoint, source) or _paths_overlap(
            output_report, source
        ):
            raise ValueError("V64 output overlaps a source artifact")
    return output_checkpoint, output_report


def load_v64_sources(args: argparse.Namespace) -> V64Sources:
    """Authenticate the baseline first, then load only training-side sources."""

    baseline_path = _resolve(args.baseline_lock)
    baseline_authorization = validate_baseline_lock(baseline_path)
    baseline_lock_sha256 = _sha256_file(baseline_path)

    value_checkpoint = _resolve(args.v63_value_checkpoint)
    route_checkpoint = _resolve(args.v62_route_checkpoint)
    value_checkpoint_sha256 = _control_checkpoint_sha256(value_checkpoint)
    route_checkpoint_sha256 = _control_checkpoint_sha256(route_checkpoint)

    value_metadata_hint = json.loads(
        (value_checkpoint / "runtime_metadata.json").read_text(encoding="utf-8")
    )
    route_metadata_hint = json.loads(
        (route_checkpoint / "runtime_metadata.json").read_text(encoding="utf-8")
    )
    if not isinstance(value_metadata_hint, dict) or not isinstance(route_metadata_hint, dict):
        raise TypeError("V64 source runtime metadata must be JSON objects")
    hidden = value_metadata_hint.get("hidden_size")
    if isinstance(hidden, bool) or not isinstance(hidden, int) or hidden < 1:
        raise ValueError("V64 V63 hidden size is invalid")
    if route_metadata_hint.get("hidden_size") != hidden:
        raise ValueError("V64 source hidden sizes differ")
    value_control_raw, value_metadata = _load_control_head(
        value_checkpoint,
        hidden_size=hidden,
        device=torch.device("cpu"),
    )
    route_control_raw, route_metadata = _load_control_head(
        route_checkpoint,
        hidden_size=hidden,
        device=torch.device("cpu"),
    )
    if type(value_control_raw) is not TeacherBasisFullSceneQuestionControlV3:
        raise TypeError("V64 V63 checkpoint is not exact V3")
    if type(route_control_raw) is not NormalizedFactorizedSceneQuestionControlV5:
        raise TypeError("V64 V62 checkpoint is not exact V5")
    value_control = value_control_raw
    route_control = route_control_raw
    dimensions = _compatible_dimensions(value_metadata, route_metadata)
    if (
        value_metadata.get("base_checkpoint_sha256")
        != route_metadata.get("base_checkpoint_sha256")
        or value_metadata.get("base_runtime_config_sha256")
        != route_metadata.get("base_runtime_config_sha256")
    ):
        raise ValueError("V64 controller checkpoints belong to different base runtimes")

    value_report, value_report_path, value_report_sha256 = _strict_report(
        args.v63_value_report,
        label="V63 value report",
    )
    route_report, route_report_path, route_report_sha256 = _strict_report(
        args.v62_route_report,
        label="V62 route report",
    )
    value_provenance = validate_v63_value_report(value_report, value_metadata)
    route_provenance = validate_v62_route_report(route_report, route_metadata)
    if value_provenance != route_provenance:
        raise ValueError("V64 V62/V63 locked training provenance differs")
    if (
        value_report["source_v60"]["question_norm_sha256"]
        != _question_norm_sha256(route_control)
        or value_report["source_v60"]["weights_sha256"]
        != route_report["source"]["v60_weights_sha256"]
        or value_report["checkpoint"]["runtime_metadata_sha256"]
        != _sha256_file(value_checkpoint / "runtime_metadata.json")
        or route_report["checkpoint"]["runtime_metadata_sha256"]
        != _sha256_file(route_checkpoint / "runtime_metadata.json")
    ):
        raise ValueError("V64 source V60 norm/weights or runtime metadata binding differs")
    if (
        value_provenance["baseline_lock_sha256"] != baseline_lock_sha256
        or value_provenance["preregistration_sha256"]
        != PINNED_V62_PREREGISTRATION_SHA256
        or value_provenance["base_checkpoint_sha256"]
        != value_metadata["base_checkpoint_sha256"]
        or value_provenance["runtime_config_sha256"]
        != value_metadata["base_runtime_config_sha256"]
        or value_provenance["filtered_training_qa_sha256"]
        != _PINNED_FILTERED_TRAIN_SHA256
        or baseline_authorization.get("preregistration_sha256")
        != PINNED_V62_PREREGISTRATION_SHA256
        or baseline_authorization.get("v54_checkpoint_sha256")
        != value_metadata["base_checkpoint_sha256"]
    ):
        raise ValueError("V64 baseline/preregistration/base binding differs")
    if dimensions["route_factor_rank"] != route_report["architecture"][
        "route_factor_rank"
    ]:
        raise ValueError("V64 route-factor rank differs between report and checkpoint")

    output_checkpoint, composition_report = _output_paths(
        checkpoint=args.output_checkpoint,
        report=args.composition_report,
        inputs=(
            baseline_path,
            value_checkpoint,
            route_checkpoint,
            value_report_path,
            route_report_path,
        ),
    )
    return V64Sources(
        baseline_lock_sha256=baseline_lock_sha256,
        baseline_authorization=baseline_authorization,
        value_checkpoint=value_checkpoint,
        value_checkpoint_sha256=value_checkpoint_sha256,
        value_control=value_control,
        value_metadata=value_metadata,
        value_report=value_report,
        value_report_path=value_report_path,
        value_report_sha256=value_report_sha256,
        route_checkpoint=route_checkpoint,
        route_checkpoint_sha256=route_checkpoint_sha256,
        route_control=route_control,
        route_metadata=route_metadata,
        route_report=route_report,
        route_report_path=route_report_path,
        route_report_sha256=route_report_sha256,
        output_checkpoint=output_checkpoint,
        composition_report=composition_report,
    )


def _publish(
    *,
    sources: V64Sources,
    candidate: NormalizedFactorizedSceneQuestionControlV5,
    report_without_checkpoint: Mapping[str, Any],
) -> dict[str, Any]:
    checkpoint = _safe_output_path(sources.output_checkpoint, "V64 checkpoint")
    report = _safe_output_path(sources.composition_report, "V64 report")
    staging = Path(tempfile.mkdtemp(prefix=".v64-publish-", dir=checkpoint.parent))
    checkpoint_published = False
    report_published = False
    try:
        staged_checkpoint = staging / "checkpoint"
        hashes = save_v5_control_checkpoint(
            staged_checkpoint,
            control=candidate,
            base_checkpoint_sha256=sources.value_metadata["base_checkpoint_sha256"],
            base_runtime_config_sha256=sources.value_metadata[
                "base_runtime_config_sha256"
            ],
            # The schema-5 field retains its historical name, but this digest
            # intentionally identifies the V63 value source in V64.
            source_v60_checkpoint_sha256=sources.value_checkpoint_sha256,
            expected_inherited_state_sha256=_state_sha256(
                sources.value_control.state_dict()
            ),
        )
        loaded_raw, loaded_metadata = _load_control_head(
            staged_checkpoint,
            hidden_size=candidate.hidden_size,
            device=torch.device("cpu"),
        )
        if type(loaded_raw) is not NormalizedFactorizedSceneQuestionControlV5:
            raise RuntimeError("V64 staged checkpoint did not reload as exact V5")
        loaded = loaded_raw
        if (
            set(loaded.state_dict()) != set(candidate.state_dict())
            or any(
                not torch.equal(loaded.state_dict()[name], candidate.state_dict()[name])
                for name in candidate.state_dict()
            )
            or loaded_metadata.get("source_v60_checkpoint_sha256")
            != sources.value_checkpoint_sha256
        ):
            raise RuntimeError("V64 staged runtime reload differs from composition")
        final_report = {**dict(report_without_checkpoint), "checkpoint": hashes}
        staged_report = staging / "composition_report.json"
        _write_json(staged_report, final_report)
        os.rename(staged_checkpoint, checkpoint)
        checkpoint_published = True
        os.rename(staged_report, report)
        report_published = True
        return final_report
    except BaseException:
        if report_published:
            report.unlink(missing_ok=True)
        if checkpoint_published:
            shutil.rmtree(checkpoint, ignore_errors=True)
        raise
    finally:
        shutil.rmtree(staging, ignore_errors=True)


def compose_v64(args: argparse.Namespace) -> dict[str, Any]:
    sources = load_v64_sources(args)
    candidate, tensor_proof = compose_controller(
        sources.value_control,
        sources.route_control,
    )
    forward_proof = randomized_forward_equivalence(
        value_control=sources.value_control,
        route_control=sources.route_control,
        candidate=candidate,
    )
    report = {
        "schema_version": 1,
        "artifact": "v64_v63_values_v62_route_composition",
        "offline_checks_passed": True,
        "promotion_eligible": False,
        "saved_runtime_generation_gate_required": True,
        "authorization": {
            "baseline_lock_sha256": sources.baseline_lock_sha256,
            "preregistration_sha256": PINNED_V62_PREREGISTRATION_SHA256,
            "baseline_validated_before_source_reports": True,
        },
        "sources": {
            "v63_value": {
                "checkpoint_sha256": sources.value_checkpoint_sha256,
                "report_sha256": sources.value_report_sha256,
                "weights_sha256": sources.value_metadata["weights_sha256"],
                "architecture": sources.value_metadata["architecture"],
            },
            "v62_route": {
                "checkpoint_sha256": sources.route_checkpoint_sha256,
                "report_sha256": sources.route_report_sha256,
                "weights_sha256": sources.route_metadata["weights_sha256"],
                "architecture": sources.route_metadata["architecture"],
            },
        },
        "common_provenance": {
            "base_checkpoint_sha256": sources.value_metadata[
                "base_checkpoint_sha256"
            ],
            "base_runtime_config_sha256": sources.value_metadata[
                "base_runtime_config_sha256"
            ],
            "filtered_training_qa_sha256": sources.value_report["inputs"][
                "filtered_training_qa_sha256"
            ],
            "prefix_cache_manifest_sha256": sources.value_report["inputs"][
                "prefix_cache_manifest_sha256"
            ],
            "training_row_count": _EXPECTED_TRAINING_ROWS,
            "training_scene_count": _EXPECTED_TRAINING_SCENES,
            "training_pair_count": _EXPECTED_TRAINING_PAIRS,
        },
        "architecture": {
            "name": "normalized_factorized_scene_question_route_v5",
            "runtime_schema_version": 5,
            "hidden_size": candidate.hidden_size,
            "control_tokens": candidate.control_token_count,
            "expected_environment_latents": (
                candidate.expected_environment_latents
            ),
            "moment_count": candidate.moment_count,
            "interaction_dim_from_v63": candidate.interaction_dim,
            "trunk_dim_from_v63": candidate.trunk_dim,
            "output_basis_rank_from_v63": candidate.output_basis_rank,
            "route_factor_rank_from_v62": candidate.route_factor_rank,
            "question_dependent_scene_retrieval": False,
            "all_scene_moments_consumed_by_route": True,
        },
        "composition_proof": {
            **tensor_proof,
            "forward": forward_proof,
            "algebraic_identity_applies_to_all_inputs": True,
            "certified_training_example_count": _EXPECTED_TRAINING_ROWS,
            "v63_value_outputs_exact_on_all_576": True,
            "v62_route_logits_exact_on_all_576": True,
        },
        "upstream_gates": {
            "v63_cross_validation_passed": sources.value_report[
                "cross_validation"
            ]["passed"],
            "v63_final_checks_passed": _all_true(
                sources.value_report["final_fit"]["checks"],
                label="V63 final fit",
            ),
            "v62_cross_validation_passed": sources.route_report[
                "cross_validation"
            ]["aggregate"]["passed"],
            "v62_final_checks_passed": _all_true(
                sources.route_report["final_fit"]["checks"],
                label="V62 final fit",
            ),
        },
        "scope": {
            "gemma_loaded": False,
            "gemma_backward_used": False,
            "gemma_generation_used": False,
            "qa_loaded": False,
            "validation_inputs_used": False,
            "scorer_inputs_used": False,
            "prediction_inputs_used": False,
            "oracle_loaded": False,
            "held_out_inputs_used": False,
            "environmental_text_inputs": [],
            "source_checkpoints_modified": False,
            "source_reports_modified": False,
        },
    }
    return _publish(
        sources=sources,
        candidate=candidate,
        report_without_checkpoint=report,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-lock", required=True)
    parser.add_argument("--v63-value-checkpoint", required=True)
    parser.add_argument("--v63-value-report", required=True)
    parser.add_argument("--v62-route-checkpoint", required=True)
    parser.add_argument("--v62-route-report", required=True)
    parser.add_argument("--output-checkpoint", required=True)
    parser.add_argument("--composition-report", required=True)
    forbidden = {
        "filtered_train_qa",
        "train_qa",
        "questions",
        "questions_manifest",
        "validation_questions",
        "scorer_references",
        "predictions",
        "oracle",
        "heldout",
    }
    if {action.dest for action in parser._actions} & forbidden:
        raise AssertionError("V64 parser exposes a data/evaluation path")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    report = compose_v64(args)
    print(
        json.dumps(
            {
                "offline_checks_passed": report["offline_checks_passed"],
                "promotion_eligible": False,
                "checkpoint": str(_resolve(args.output_checkpoint)),
                "composition_report": str(_resolve(args.composition_report)),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "V64Sources",
    "compose_controller",
    "compose_v64",
    "load_v64_sources",
    "main",
    "randomized_forward_equivalence",
    "validate_v62_route_report",
    "validate_v63_value_report",
]
