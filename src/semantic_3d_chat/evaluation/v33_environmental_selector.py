"""Independently select V33 environmental-only development checkpoints.

Every saved arm is rescored with one local Gemma load.  The selector proves
that only the eight audited dense-sidecar tensors changed, recomputes teacher
and greedy evidence on scenes 19--24, measures actual adapted continuous-prefix
separation, and keeps development progress separate from chat promotion.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import torch
from safetensors.torch import load_file

from semantic_3d_chat.config import config_hash, load_config
from semantic_3d_chat.evaluation.metrics import exact_normalized_match, normalize_answer
from semantic_3d_chat.evaluation.v27_sidecar_screen import _atomic_json
from semantic_3d_chat.evaluation.v28_stage_b_selector import _retention_control_config
from semantic_3d_chat.evaluation.v30_joint_pair_selector import (
    PairMarginEvidence,
    RuntimeArmEvidence,
    SelectionRequirements,
    _compare_pair_evidence,
    _metadata,
    _pair_margin_evidence,
    _RuntimeEvaluator,
    _selection_requirements,
    _sidecar_state,
    _source_v29_evidence,
    _validate_no_leakage_or_final_scenes,
    _validate_runtime_metadata,
    _validate_source_against_config,
    _validation_nll,
)
from semantic_3d_chat.language.lora import tensor_state_sha256
from semantic_3d_chat.training.checkpointing import (
    RUNTIME_METADATA_FILENAME,
    TRAINING_METADATA_FILENAME,
)
from semantic_3d_chat.training.train_environmental_sidecar_v33 import (
    _TRAINABLE_NAMES,
    _VALIDATION_FAMILY_PAIR_IDS,
    V33Contract,
    V33Settings,
    assert_v33_trainable_surface,
    freeze_for_v33,
    prefix_separation_diagnostics,
    prefix_separation_ratios,
    v32_rejection_status,
    v33_contract,
    v33_settings,
    validation_family_teacher_metrics,
)
from semantic_3d_chat.training.train_joint_pair_v30 import v30_contract

DEFAULT_CONFIG = Path("configs/experiments/gemma4_diverse28_environmental_sidecar_v33.yaml")
DEFAULT_CHECKPOINT_ROOT = Path("data_gemma4/checkpoints/gemma4_v33_diverse28_environmental_sidecar")
DEFAULT_OUTPUT = Path("reports/gemma4/metrics/v33_environmental_sidecar_selection.json")
_SHA256 = re.compile(r"[0-9a-f]{64}")
_SCENE_ID = re.compile(r"scene_[0-9]{6}")
_AUTHORIZED_TENSOR_NAMES = frozenset(f"dense_sidecar_adapter.{name}" for name in _TRAINABLE_NAMES)


def _mapping(value: Any, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{field} must be a mapping")
    return value


def _v33_frozen_tensor_sha256(tensors: Mapping[str, torch.Tensor]) -> str:
    frozen = {
        name: value for name, value in tensors.items() if name not in _AUTHORIZED_TENSOR_NAMES
    }
    if not frozen:
        raise ValueError("V33 checkpoint contains no inherited frozen tensors")
    return tensor_state_sha256(frozen)


def _validate_v33_surface(
    metadata: Mapping[str, Any], tensors: Mapping[str, torch.Tensor]
) -> dict[str, Any]:
    v33 = _mapping(metadata.get("v33_environmental"), "metadata.v33_environmental")
    surface = _mapping(
        _mapping(metadata.get("v30_joint_pair"), "metadata.v30_joint_pair").get(
            "trainable_surface"
        ),
        "metadata.v30_joint_pair.trainable_surface",
    )
    names = surface.get("parameter_names")
    if not isinstance(names, Sequence) or isinstance(names, (str, bytes)):
        raise TypeError("V33 trainable surface names must be a sequence")
    if frozenset(str(name) for name in names) != _AUTHORIZED_TENSOR_NAMES:
        raise ValueError("V33 authorized trainable tensor names changed")
    if surface.get("group_parameter_counts") != {
        "output": 198_144,
        "sidecar_hidden": 199_808,
        "position": 6_656,
    }:
        raise ValueError("V33 trainable group parameter counts changed")
    if surface.get("total_parameter_count") != 404_608:
        raise ValueError("V33 total trainable parameter count changed")
    for field in (
        "gemma_decoder_frozen",
        "all_lora_banks_frozen",
        "base_norm_and_projection_frozen",
    ):
        if surface.get(field) is not True or v33.get(field) is not True:
            raise ValueError(f"V33 checkpoint does not prove {field}")
    if surface.get("every_other_parameter_frozen") is not True:
        raise ValueError("V33 checkpoint does not prove every_other_parameter_frozen")
    missing = sorted(_AUTHORIZED_TENSOR_NAMES - set(tensors))
    if missing:
        raise ValueError(f"V33 checkpoint lacks environmental tensors: {missing}")
    count = sum(int(tensors[name].numel()) for name in _AUTHORIZED_TENSOR_NAMES)
    if count != 404_608:
        raise ValueError(f"V33 checkpoint environmental tensor count changed: {count}")
    sidecar = _sidecar_state(tensors)
    if tensor_state_sha256(sidecar) != metadata.get("dense_sidecar_adapter_state_sha256"):
        raise ValueError("V33 checkpoint sidecar hash differs from metadata")
    return {
        "parameter_names": sorted(_AUTHORIZED_TENSOR_NAMES),
        "parameter_count": count,
        "sidecar_state_sha256": tensor_state_sha256(sidecar),
        "frozen_tensor_sha256": _v33_frozen_tensor_sha256(tensors),
        "gemma_decoder_frozen": True,
        "all_lora_banks_frozen": True,
    }


def _validate_optimizer_state_step(path: Path, expected_step: int, settings: V33Settings) -> None:
    payload = torch.load(path / "optimizer.pt", map_location="cpu", weights_only=True)
    if not isinstance(payload, Mapping):
        raise TypeError(f"V33 optimizer state is not a mapping: {path.name}")
    groups = payload.get("param_groups")
    state = payload.get("state")
    if not isinstance(groups, list) or len(groups) != 3 or not isinstance(state, Mapping):
        raise ValueError("V33 optimizer must contain exactly three environmental groups")
    expected = (
        ("dense_sidecar_adapter.output", settings.output_learning_rate, 2),
        ("dense_sidecar_adapter.sidecar_hidden", settings.hidden_learning_rate, 4),
        ("dense_sidecar_adapter.position", settings.position_learning_rate, 2),
    )
    parameter_ids: list[Any] = []
    for index, (group, (name, learning_rate, tensor_count)) in enumerate(
        zip(groups, expected, strict=True)
    ):
        parsed = _mapping(group, f"optimizer.param_groups[{index}]")
        parameters = parsed.get("params")
        if (
            parsed.get("name") != name
            or float(parsed.get("lr", math.nan)) != learning_rate
            or float(parsed.get("weight_decay", math.nan)) != 0.0
            or not isinstance(parameters, list)
            or len(parameters) != tensor_count
        ):
            raise ValueError(f"V33 optimizer group {index} differs from its lock")
        parameter_ids.extend(parameters)
    if len(parameter_ids) != 8 or len(set(parameter_ids)) != 8 or set(state) != set(parameter_ids):
        raise ValueError("V33 Adam state must cover exactly eight tensors once")
    for parameter_id in parameter_ids:
        entry = _mapping(state[parameter_id], f"optimizer.state[{parameter_id}]")
        if set(entry) != {"step", "exp_avg", "exp_avg_sq"}:
            raise ValueError("V33 Adam state fields changed")
        for field in ("exp_avg", "exp_avg_sq"):
            value = entry[field]
            if not isinstance(value, torch.Tensor) or not torch.isfinite(value).all():
                raise ValueError(f"V33 Adam {field} contains an invalid tensor")
        step = entry["step"]
        step = step.item() if isinstance(step, torch.Tensor) and step.numel() == 1 else step
        if isinstance(step, bool) or not isinstance(step, (int, float)):
            raise TypeError("V33 Adam step is not numeric")
        if float(step) != expected_step:
            raise ValueError(f"V33 Adam state does not prove update {expected_step}")


def validate_v33_checkpoint_envelope(
    config: Mapping[str, Any], checkpoint_root: Path, contract: V33Contract
) -> tuple[Path, ...]:
    """Require exact saved arms and their strict causal envelopes."""

    paths = tuple(checkpoint_root / f"update_{step:03d}" for step in contract.saved_optimizer_steps)
    observed = sorted(path.name for path in checkpoint_root.glob("update_*") if path.is_dir())
    expected = [path.name for path in paths]
    if observed != expected:
        raise FileNotFoundError(f"V33 saved arms differ: observed={observed} expected={expected}")
    settings = v33_settings(config)
    expected_hash = config_hash(dict(config))
    provenance: tuple[str, str] | None = None
    for step, path in zip(contract.saved_optimizer_steps, paths, strict=True):
        if path.is_symlink():
            raise ValueError(f"V33 saved arm must not be a symlink: {path}")
        files = ["adapter.safetensors", TRAINING_METADATA_FILENAME, RUNTIME_METADATA_FILENAME]
        if step:
            files.append("optimizer.pt")
        missing = [name for name in files if not (path / name).is_file()]
        if missing:
            raise FileNotFoundError(f"V33 checkpoint {path.name} is incomplete: {missing}")
        if any((path / name).is_symlink() for name in files):
            raise ValueError(f"V33 checkpoint files must not be symlinks: {path.name}")
        metadata = _metadata(path)
        if metadata.get("optimizer_step") != step or metadata.get("config_hash") != expected_hash:
            raise ValueError(f"V33 checkpoint step/config mismatch: {path.name}")
        v30 = _mapping(metadata.get("v30_joint_pair"), "metadata.v30_joint_pair")
        v33 = _mapping(metadata.get("v33_environmental"), "metadata.v33_environmental")
        if (
            tuple(v30.get("train_scene_ids", ())) != contract.v31.train_scene_ids
            or tuple(v30.get("validation_scene_ids", ())) != contract.v31.validation_scene_ids
            or v30.get("final_test_scene_ids_loaded") != []
            or v30.get("oracle_environment_files_loaded") is not False
            or v33.get("deferred_final_scene_ids_loaded") != []
        ):
            raise ValueError(f"V33 checkpoint crossed its data boundary: {path.name}")
        condition = _mapping(v33.get("conditional_v32_rejection"), "v33.conditional_v32_rejection")
        observed_provenance = (str(condition.get("report")), str(condition.get("report_sha256")))
        if (
            condition.get("status") != "rejected"
            or condition.get("training_authorized") is not True
            or Path(observed_provenance[0]).resolve() != contract.v32_selection_report
            or observed_provenance[1] != contract.v32_selection_report_sha256
        ):
            raise ValueError(f"V33 checkpoint lacks pinned V32 rejection: {path.name}")
        if provenance is None:
            provenance = observed_provenance
        elif provenance != observed_provenance:
            raise ValueError("V33 V32 rejection provenance changed across arms")
        schedule = _mapping(v33.get("schedule"), "v33.schedule")
        if (
            schedule.get("optimizer_step_count") != 100
            or schedule.get("pair_unit_minimum_recurrence") != 4
            or schedule.get("pair_unit_maximum_recurrence") != 4
            or schedule.get("true_optimizer_step_per_schedule_row") is not True
            or schedule.get("pair_units_atomic") is not True
        ):
            raise ValueError(f"V33 schedule proof failed: {path.name}")
        history = metadata.get("history")
        if not isinstance(history, list) or len(history) != step + 1:
            raise ValueError(f"V33 history is not one row per true update: {path.name}")
        row = _mapping(history[-1], "history[-1]")
        if (
            row.get("optimizer_update") != step
            or row.get("validation_answer_token_nll") is None
            or row.get("validation_pair_metrics") is None
            or row.get("adapted_prefix_separation") is None
            or row.get("adapted_prefix_separation_ratios_from_update0") is None
        ):
            raise ValueError(f"V33 saved-arm diagnostics are incomplete: {path.name}")
        if step and row.get("separate_group_clipping") is not True:
            raise ValueError(f"V33 saved arm lacks separate clipping proof: {path.name}")
        if step:
            _validate_optimizer_state_step(path, step, settings)
    return paths


@dataclass(frozen=True)
class V33RuntimeEvidence:
    base: RuntimeArmEvidence
    greedy_complete_by_family: Mapping[str, int]
    greedy_prediction_changed_by_family: Mapping[str, int]
    prefix_diagnostics: Mapping[str, Any]


class V33ArmEvaluator(Protocol):
    validation_scene_ids: tuple[str, ...]

    def install(self, tensors: Mapping[str, torch.Tensor]) -> None: ...

    def evaluate_v33(self) -> V33RuntimeEvidence: ...

    def evaluate_aggregate_exact(self) -> tuple[int, int]: ...


class _V33RuntimeEvaluator(_RuntimeEvaluator):
    def __init__(
        self,
        config: dict[str, Any],
        control_config: dict[str, Any],
        checkpoint: Path,
        requirements: SelectionRequirements,
    ) -> None:
        super().__init__(config, control_config, checkpoint, requirements)
        freeze_for_v33(self.bundle)
        assert_v33_trainable_surface(self.bundle)

    def evaluate_v33(self) -> V33RuntimeEvidence:
        base = super().evaluate()
        prefixes = self._prefixes()
        changed, _broad = self._generation_rows(prefixes)
        grouped: defaultdict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
        for row in changed:
            grouped[(str(row["pair_id"]), str(row["question_key"]))].append(row)
        reverse = {pair_id: family for family, pair_id in _VALIDATION_FAMILY_PAIR_IDS.items()}
        exact = {family: 0 for family in reverse.values()}
        changed_counts = {family: 0 for family in reverse.values()}
        for (pair_id, _question_key), rows in grouped.items():
            if pair_id not in reverse or len(rows) != 2:
                raise ValueError("V33 greedy family evidence is incomplete")
            family = reverse[pair_id]
            exact[family] += int(
                all(exact_normalized_match(row["prediction"], row["target"]) for row in rows)
            )
            changed_counts[family] += int(
                normalize_answer(rows[0]["prediction"]) != normalize_answer(rows[1]["prediction"])
            )
        return V33RuntimeEvidence(
            base=base,
            greedy_complete_by_family=exact,
            greedy_prediction_changed_by_family=changed_counts,
            prefix_diagnostics=prefix_separation_diagnostics(
                units=self.validation_units,
                caches=self.validation_caches,
                bundle=self.bundle,
            ),
        )


def _family_teacher_deltas(
    current: Mapping[str, Any], baseline: Mapping[str, Any]
) -> dict[str, float]:
    return {
        family: float(_mapping(current[family], family)["mean_margin"])
        - float(_mapping(baseline[family], family)["mean_margin"])
        for family in _VALIDATION_FAMILY_PAIR_IDS
    }


def v33_chat_promotion_checks(
    selected: Mapping[str, Any],
    *,
    update0_aggregate: tuple[int, int],
    selected_aggregate: tuple[int, int],
) -> dict[str, bool]:
    """Apply the complete chat gate without conflating development progress."""

    if update0_aggregate[0] <= 0 or selected_aggregate[0] != update0_aggregate[0]:
        raise ValueError("V33 aggregate validation audits are misaligned")
    by_family = _mapping(selected.get("greedy_complete_units_by_family"), "selected families")
    new_negatives = selected.get("new_negative_sides")
    if not isinstance(new_negatives, Sequence) or isinstance(new_negatives, (str, bytes)):
        raise TypeError("V33 selected new-negative evidence must be a sequence")
    return {
        "development_checkpoint_selected": True,
        "changed_complete_pair_threshold_met": (
            int(selected["greedy_exact_complete_units_correct"]) >= 6
        ),
        "each_validation_family_demonstrated": all(
            int(by_family[family]) >= 1 for family in _VALIDATION_FAMILY_PAIR_IDS
        ),
        "old_color_12_sides_retained": int(selected["color_full_vocab_sides"]) >= 12,
        "old_mirror_10_sides_retained": int(selected["mirror_full_vocab_sides"]) >= 10,
        "old_controls_no_new_negatives": not new_negatives,
        "aggregate_validation_exact_accuracy_retained": (
            selected_aggregate[1] / selected_aggregate[0]
            >= update0_aggregate[1] / update0_aggregate[0]
        ),
    }


def select_v33(
    config_path: Path,
    checkpoint_root: Path,
    *,
    evaluator_factory: Callable[
        [dict[str, Any], dict[str, Any], Path, SelectionRequirements], V33ArmEvaluator
    ] = _V33RuntimeEvaluator,
) -> dict[str, Any]:
    config = load_config(config_path)
    contract = v33_contract(config)
    condition = v32_rejection_status(config)
    if condition.get("training_authorized") is not True:
        raise ValueError("V33 selection requires its pinned terminal V32 rejection")
    checkpoints = validate_v33_checkpoint_envelope(config, checkpoint_root, contract)
    requirements = _selection_requirements(config)
    control = _retention_control_config(config)
    source = _source_v29_evidence(_metadata(checkpoints[0]))
    _validate_source_against_config(source, config)
    evaluator = evaluator_factory(config, control, checkpoints[0], requirements)
    if tuple(evaluator.validation_scene_ids) != contract.v31.validation_scene_ids:
        raise ValueError("V33 evaluator must remain exactly on scenes 19--24")

    arms: list[dict[str, Any]] = []
    frozen_hash: str | None = None
    baseline_negatives: frozenset[tuple[str, str]] | None = None
    baseline_pair: PairMarginEvidence | None = None
    baseline_broad: float | None = None
    baseline_prefix: Mapping[str, Any] | None = None
    baseline_family_teacher: Mapping[str, Any] | None = None
    update0_aggregate: tuple[int, int] | None = None
    for arm_index, (step, checkpoint) in enumerate(
        zip(contract.saved_optimizer_steps, checkpoints, strict=True)
    ):
        metadata = _metadata(checkpoint)
        _validate_runtime_metadata(checkpoint, metadata)
        _validate_no_leakage_or_final_scenes(metadata)
        tensors = load_file(checkpoint / "adapter.safetensors", device="cpu")
        surface = _validate_v33_surface(metadata, tensors)
        observed_frozen = str(surface["frozen_tensor_sha256"])
        if (
            _mapping(metadata.get("v30_joint_pair"), "metadata.v30_joint_pair").get(
                "frozen_inherited_state_sha256"
            )
            != observed_frozen
        ):
            raise ValueError(f"V33 frozen-state metadata hash mismatch in {checkpoint.name}")
        if frozen_hash is None:
            frozen_hash = observed_frozen
        elif frozen_hash != observed_frozen:
            raise RuntimeError(f"V33 frozen tensors changed in {checkpoint.name}")
        if arm_index == 0:
            source_contract = v30_contract(config)
            if surface["sidecar_state_sha256"] != source_contract["source_sidecar_state_sha256"]:
                raise ValueError("V33 update zero is not the approved V29 sidecar state")
            tolerance = float(source_contract["update_zero_validation_nll_absolute_tolerance"])
            if (
                abs(_validation_nll(metadata) - float(source["validation_answer_token_nll"]))
                > tolerance
            ):
                raise ValueError("V33 update zero validation NLL differs from approved V29")
        evaluator.install(tensors)
        evidence = evaluator.evaluate_v33()
        pair = evidence.base.pair_margins
        recorded_pair = _pair_margin_evidence(
            metadata, expected_unit_count=requirements.validation_pair_unit_count
        )
        _compare_pair_evidence(recorded_pair, pair)
        recorded_prefix = _mapping(
            _mapping(metadata["v33_environmental"], "v33").get("adapted_prefix_separation"),
            "v33.adapted_prefix_separation",
        )
        for family, value in _mapping(
            recorded_prefix.get("rms_by_validation_family"), "recorded prefix families"
        ).items():
            observed = _mapping(
                evidence.prefix_diagnostics.get("rms_by_validation_family"),
                "observed prefix families",
            ).get(family)
            if not math.isclose(float(value), float(observed), rel_tol=0.0, abs_tol=1e-6):
                raise ValueError(f"V33 adapted-prefix RMS replay differs for {family}")
        family_teacher = validation_family_teacher_metrics(
            _mapping(
                _mapping(metadata["history"][-1], "history[-1]").get("validation_pair_metrics"),
                "history pair metrics",
            )
        )
        if baseline_negatives is None:
            baseline_negatives = evidence.base.negative_sides
            baseline_pair = pair
            baseline_broad = evidence.base.generation.broad_exact_accuracy
            baseline_prefix = evidence.prefix_diagnostics
            baseline_family_teacher = family_teacher
            update0_aggregate = evaluator.evaluate_aggregate_exact()
        assert baseline_pair is not None
        assert baseline_broad is not None
        assert baseline_prefix is not None
        assert baseline_family_teacher is not None
        ratios = prefix_separation_ratios(evidence.prefix_diagnostics, baseline_prefix)
        family_deltas = _family_teacher_deltas(family_teacher, baseline_family_teacher)
        nonmirror_teacher = sum(
            int(_mapping(family_teacher[family], family)["complete_units"])
            for family in ("book_support", "picture_support")
        )
        new_negatives = sorted(evidence.base.negative_sides - baseline_negatives)
        checks = {
            "old_color_retained": evidence.base.color_full_vocab_sides >= 12,
            "old_mirror_retained": evidence.base.mirror_full_vocab_sides >= 10,
            "no_new_negative_sides": not new_negatives,
            "teacher_pair_mean_improved": pair.mean_margin > baseline_pair.mean_margin,
            "teacher_pair_passed_units_not_lower": pair.passed_units >= baseline_pair.passed_units,
            "broad_retention_no_regression": (
                evidence.base.generation.broad_exact_accuracy >= baseline_broad
            ),
            "weak_pair_prefix_rms_improved_25pct": ratios["weak_pair_mean"] >= 1.25,
            "unrelated_prefix_rms_inflation_bounded": ratios["unrelated_mean"] <= 1.25,
            "greedy_development_unit_demonstrated": (
                evidence.base.generation.exact_complete_units_correct >= 1
            ),
        }
        eligible = step > 0 and all(checks.values())
        arms.append(
            {
                "checkpoint": str(checkpoint),
                "arm_index": arm_index,
                "optimizer_step": step,
                "update": step,
                "validation_answer_token_nll": _validation_nll(metadata),
                "validation_pair_passed_units": pair.passed_units,
                "validation_pair_mean_margin": pair.mean_margin,
                "validation_pair_minimum_margin": pair.minimum_margin,
                "validation_pair_mean_margin_delta_from_update0": (
                    pair.mean_margin - baseline_pair.mean_margin
                ),
                "validation_family_teacher": family_teacher,
                "validation_family_teacher_margin_delta_from_update0": family_deltas,
                "nonmirror_teacher_complete_units": nonmirror_teacher,
                "color_full_vocab_sides": evidence.base.color_full_vocab_sides,
                "mirror_full_vocab_sides": evidence.base.mirror_full_vocab_sides,
                "new_negative_sides": new_negatives,
                "greedy_exact_complete_units_correct": (
                    evidence.base.generation.exact_complete_units_correct
                ),
                "greedy_prediction_changed_units": (
                    evidence.base.generation.prediction_changed_units
                ),
                "greedy_complete_units_by_family": dict(evidence.greedy_complete_by_family),
                "greedy_prediction_changed_by_family": dict(
                    evidence.greedy_prediction_changed_by_family
                ),
                "broad_retention_exact_accuracy": (evidence.base.generation.broad_exact_accuracy),
                "adapted_prefix_separation": dict(evidence.prefix_diagnostics),
                "adapted_prefix_separation_ratios_from_update0": ratios,
                "frozen_tensor_sha256": observed_frozen,
                "surface": surface,
                "greedy_screen_designated": step in {32, 64, 100},
                "checks": checks,
                "eligible": eligible,
            }
        )

    update64 = next(arm for arm in arms if arm["optimizer_step"] == 64)
    update64_gate = {
        "nonmirror_teacher_complete": update64["nonmirror_teacher_complete_units"] >= 1,
        "book_advantage_positive": (
            update64["validation_family_teacher_margin_delta_from_update0"]["book_support"] > 0.0
        ),
        "picture_advantage_positive": (
            update64["validation_family_teacher_margin_delta_from_update0"]["picture_support"] > 0.0
        ),
    }
    update64_gate["passed"] = all(update64_gate.values())
    candidates = [arm for arm in arms if arm["eligible"] and update64_gate["passed"]]
    selected = min(
        candidates,
        key=lambda arm: (
            -int(arm["greedy_exact_complete_units_correct"]),
            -int(arm["validation_pair_passed_units"]),
            float(arm["validation_answer_token_nll"]),
            int(arm["optimizer_step"]),
        ),
        default=None,
    )

    promotion: dict[str, Any] = {
        "evaluated": selected is not None,
        "changed_complete_pair_threshold": 6,
        "requires_each_validation_family": True,
        "aggregate_validation_exact_accuracy_no_regression": True,
        "checks": {},
        "eligible": False,
    }
    if selected is not None:
        assert update0_aggregate is not None
        selected_path = checkpoint_root / f"update_{int(selected['optimizer_step']):03d}"
        evaluator.install(load_file(selected_path / "adapter.safetensors", device="cpu"))
        selected_aggregate = evaluator.evaluate_aggregate_exact()
        promotion_checks = v33_chat_promotion_checks(
            selected,
            update0_aggregate=update0_aggregate,
            selected_aggregate=selected_aggregate,
        )
        promotion.update(
            {
                "update0_aggregate_validation": {
                    "row_count": update0_aggregate[0],
                    "exact_correct": update0_aggregate[1],
                    "exact_accuracy": update0_aggregate[1] / update0_aggregate[0],
                },
                "selected_aggregate_validation": {
                    "row_count": selected_aggregate[0],
                    "exact_correct": selected_aggregate[1],
                    "exact_accuracy": selected_aggregate[1] / selected_aggregate[0],
                },
                "checks": promotion_checks,
                "eligible": all(promotion_checks.values()),
            }
        )

    return {
        "schema_version": 1,
        "artifact": "v33_environmental_sidecar_development_selection",
        "development_validation_model_selection_only": True,
        "training_evaluation_only": True,
        "final_test_scenes_touched": False,
        "deferred_final_scene_ids": list(contract.v31.deferred_final_scene_ids),
        "oracle_loaded": False,
        "question_text_serialized": False,
        "answer_text_serialized": False,
        "question_dependent_scene_processing": False,
        "question_dependent_retrieval": False,
        "all_scene_prefixes_built_before_questions": True,
        "gemma_decoder_frozen": True,
        "all_lora_banks_frozen": True,
        "exact_trainable_parameter_count": 404_608,
        "model_load_count": 1,
        "source_v29": source,
        "v32_rejection": condition,
        "train_scene_ids": list(contract.v31.train_scene_ids),
        "validation_scene_ids": list(evaluator.validation_scene_ids),
        "saved_optimizer_steps": list(contract.saved_optimizer_steps),
        "all_saved_arms_inspected": True,
        "optimizer_state_steps_verified": list(contract.saved_optimizer_steps[1:]),
        "greedy_screen_steps": [32, 64, 100],
        "update64_environmental_gate": update64_gate,
        "conditional_next_surface": {
            "enabled": False,
            "parameter_count": 199_808,
            "triggered": not update64_gate["passed"],
            "auto_enabled": False,
        },
        "frozen_tensor_sha256": frozen_hash,
        "arms": arms,
        "selected_checkpoint": None if selected is None else selected["checkpoint"],
        "selected_update": None if selected is None else selected["optimizer_step"],
        "selected_optimizer_step": None if selected is None else selected["optimizer_step"],
        "development_selection_passed": selected is not None,
        "chat_promotion": promotion,
        "chat_promotion_eligible": promotion["eligible"],
        "development_progress_is_not_chat_promotion": True,
        "passed": selected is not None,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--checkpoint-root", type=Path, default=DEFAULT_CHECKPOINT_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    report = select_v33(args.config, args.checkpoint_root)
    _atomic_json(args.output, report)
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
    return 0 if report["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "V33RuntimeEvidence",
    "select_v33",
    "v33_chat_promotion_checks",
    "validate_v33_checkpoint_envelope",
]
