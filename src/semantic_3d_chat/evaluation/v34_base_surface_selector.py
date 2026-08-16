"""Independently select V34 base-route development arms.

The selector loads Gemma once, replays every numbered arm, recomputes the
train-scene 8-vs-112 selectivity evidence, and evaluates validation exactly
once after the bounded run.  Approved V29 is the retention/aggregate baseline;
V33 update 64 is the improvement baseline.  Final scenes remain inaccessible.
"""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import torch
from safetensors.torch import load_file

from semantic_3d_chat.config import config_hash, load_config
from semantic_3d_chat.evaluation.v27_sidecar_screen import _atomic_json
from semantic_3d_chat.evaluation.v28_stage_b_selector import _retention_control_config
from semantic_3d_chat.evaluation.v30_joint_pair_selector import (
    PairMarginEvidence,
    SelectionRequirements,
    _compare_pair_evidence,
    _metadata,
    _pair_margin_evidence,
    _selection_requirements,
    _sidecar_state,
    _source_v29_evidence,
    _validate_no_leakage_or_final_scenes,
    _validate_runtime_metadata,
    _validate_source_against_config,
    _validation_nll,
)
from semantic_3d_chat.evaluation.v33_environmental_selector import (
    V33RuntimeEvidence,
    _V33RuntimeEvaluator,
)
from semantic_3d_chat.language.lora import tensor_state_sha256
from semantic_3d_chat.training.checkpointing import (
    RUNTIME_METADATA_FILENAME,
    TRAINING_METADATA_FILENAME,
    runtime_checkpoint_metadata,
    validate_runtime_checkpoint_metadata,
)
from semantic_3d_chat.training.pair_curriculum import build_exact_question_pair_units
from semantic_3d_chat.training.train_base_surface_v34 import (
    _TRAINABLE_NAMES,
    PrefixSeparationReference,
    V34Contract,
    _optimizer_step_audit,
    assert_v34_trainable_surface,
    build_prefix_separation_reference,
    freeze_for_v34,
    physical_pair_sets,
    require_v33_terminal_gate,
    training_separation_diagnostics,
    v34_contract,
    v34_early_training_gate,
    v34_settings,
)
from semantic_3d_chat.training.train_environmental_sidecar_v33 import (
    _VALIDATION_FAMILY_PAIR_IDS,
    prefix_separation_ratios,
    validation_family_teacher_metrics,
)
from semantic_3d_chat.training.train_joint_pair_v30 import (
    cache_pre_sidecar_scenes,
)
from semantic_3d_chat.training.train_joint_pair_v31 import load_v31_qa_records

DEFAULT_CONFIG = Path("configs/experiments/gemma4_diverse28_base_surface_v34.yaml")
DEFAULT_CHECKPOINT_ROOT = Path("data_gemma4/checkpoints/gemma4_v34_diverse28_base_surface")
DEFAULT_OUTPUT = Path("reports/gemma4/metrics/v34_base_surface_selection.json")
_AUTHORIZED = frozenset(f"dense_sidecar_adapter.{name}" for name in _TRAINABLE_NAMES)
_FRESH_BANK_PREFIX = "lora_banks.extension_v30_joint_pair_query.adapters."


def _mapping(value: object, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{field} must be a mapping")
    return value


def _frozen_tensor_sha256(tensors: Mapping[str, torch.Tensor]) -> str:
    frozen = {name: value for name, value in tensors.items() if name not in _AUTHORIZED}
    if not frozen:
        raise ValueError("V34 checkpoint contains no frozen inherited tensors")
    return tensor_state_sha256(frozen)


def _approved_v29_runtime_tensor_envelope(
    update0: Mapping[str, torch.Tensor], approved_v29: Mapping[str, torch.Tensor]
) -> dict[str, torch.Tensor]:
    """Add V30's exact-zero query bank to the narrower V29 checkpoint.

    The evaluator bundle is already initialized from the exact approved V29
    checkpoint. Its install API additionally requires the later V30 query-bank
    envelope. Only that eight-tensor bank may be absent from V29, and every B
    matrix must remain exact zero, so merging it cannot change V29 inference.
    """

    if not set(approved_v29).issubset(update0):
        raise ValueError("Approved V29 contains tensors absent from V34 update zero")
    for name, value in approved_v29.items():
        if tuple(value.shape) != tuple(update0[name].shape):
            raise ValueError(f"Approved V29 tensor shape changed: {name}")
    extra = sorted(set(update0) - set(approved_v29))
    if len(extra) != 8 or any(not name.startswith(_FRESH_BANK_PREFIX) for name in extra):
        raise ValueError(f"V34-only V29 retention tensors are not exactly the fresh bank: {extra}")
    b_tensors = [update0[name] for name in extra if name.endswith(".lora_b")]
    if len(b_tensors) != 4 or any(torch.count_nonzero(value).item() for value in b_tensors):
        raise ValueError("V34 V29 retention envelope fresh bank is not exact-zero output")
    merged = {name: value for name, value in update0.items()}
    merged.update(approved_v29)
    return merged


def _validate_surface(
    metadata: Mapping[str, Any], tensors: Mapping[str, torch.Tensor],
) -> dict[str, Any]:
    stage = _mapping(metadata.get("v34_base_surface"), "metadata.v34_base_surface")
    surface = _mapping(stage.get("trainable_surface"), "v34.trainable_surface")
    names = surface.get("parameter_names")
    if not isinstance(names, Sequence) or isinstance(names, (str, bytes)) or frozenset(
        str(name) for name in names
    ) != _AUTHORIZED:
        raise ValueError("V34 checkpoint trainable names differ from four base tensors")
    if surface.get("group_parameter_counts") != {
        "base_norm": 3_072,
        "base_projection": 196_736,
    } or surface.get("total_parameter_count") != 199_808:
        raise ValueError("V34 checkpoint trainable parameter counts changed")
    for field in (
        "gemma_decoder_frozen",
        "all_lora_banks_frozen",
        "all_v33_learned_tensors_frozen",
        "every_other_parameter_frozen",
    ):
        if surface.get(field) is not True:
            raise ValueError(f"V34 checkpoint does not prove {field}")
    missing = sorted(_AUTHORIZED - set(tensors))
    if missing:
        raise ValueError(f"V34 checkpoint lacks base tensors: {missing}")
    if sum(int(tensors[name].numel()) for name in _AUTHORIZED) != 199_808:
        raise ValueError("V34 checkpoint base tensor count changed")
    sidecar = _sidecar_state(tensors)
    if tensor_state_sha256(sidecar) != metadata.get("dense_sidecar_adapter_state_sha256"):
        raise ValueError("V34 sidecar hash differs from metadata")
    return {
        "parameter_names": sorted(_AUTHORIZED),
        "parameter_count": 199_808,
        "sidecar_state_sha256": tensor_state_sha256(sidecar),
        "frozen_tensor_sha256": _frozen_tensor_sha256(tensors),
    }


def validate_v34_checkpoint_envelope(
    config: Mapping[str, Any], checkpoint_root: Path, contract: V34Contract,
) -> tuple[Path, ...]:
    checkpoints = tuple(
        checkpoint_root / f"update_{step:03d}" for step in contract.saved_optimizer_steps
    )
    observed = sorted(path.name for path in checkpoint_root.glob("update_*") if path.is_dir())
    expected = [path.name for path in checkpoints]
    if observed != expected:
        raise FileNotFoundError(f"V34 saved arms differ: observed={observed} expected={expected}")
    terminal = require_v33_terminal_gate(config)
    settings = v34_settings(config)
    expected_config_hash = config_hash(dict(config))
    frozen_hash: str | None = None
    update0_tensors: Mapping[str, torch.Tensor] | None = None
    for step, checkpoint in zip(contract.saved_optimizer_steps, checkpoints, strict=True):
        if checkpoint.is_symlink():
            raise ValueError(f"V34 checkpoint must not be a symlink: {checkpoint}")
        required = ["adapter.safetensors", TRAINING_METADATA_FILENAME, RUNTIME_METADATA_FILENAME]
        if step:
            required.append("optimizer.pt")
        if any(not (checkpoint / name).is_file() or (checkpoint / name).is_symlink() for name in required):
            raise FileNotFoundError(f"V34 checkpoint is incomplete or aliased: {checkpoint}")
        metadata = _metadata(checkpoint)
        stage = _mapping(metadata.get("v34_base_surface"), "metadata.v34_base_surface")
        if metadata.get("optimizer_step") != step or metadata.get("config_hash") != expected_config_hash:
            raise ValueError(f"V34 checkpoint step/config mismatch: {checkpoint.name}")
        if stage.get("conditional_v33_terminal_gate") != {
            "path": terminal["path"],
            "sha256": terminal["sha256"],
        } or Path(str(stage.get("source_checkpoint"))).resolve() != contract.source_checkpoint:
            raise ValueError(f"V34 checkpoint terminal/source provenance changed: {checkpoint.name}")
        if stage.get("source_file_sha256") != dict(contract.source_file_sha256):
            raise ValueError(f"V34 checkpoint source hashes changed: {checkpoint.name}")
        if (
            stage.get("deferred_final_scene_ids_loaded") != []
            or stage.get("oracle_environment_files_loaded") is not False
            or stage.get("question_dependent_scene_processing") is not False
            or stage.get("question_dependent_retrieval") is not False
            or stage.get("separation_uses_training_scenes_only") is not True
            or stage.get("separation_uses_question_or_answer_text") is not False
            or stage.get("separation_uses_oracle_environment_inputs") is not False
        ):
            raise ValueError(f"V34 checkpoint crossed its data boundary: {checkpoint.name}")
        legacy = _mapping(metadata.get("v30_joint_pair"), "metadata.v30_joint_pair")
        cache = _mapping(legacy.get("scene_cache"), "v34 pre-sidecar cache")
        if not (
            cache.get("cache_boundary") == "complete_frozen_pre_sidecar_scene_stack"
            and cache.get("all_voxels_covered") is True
            and cache.get("question_inputs_to_scene_cache") is False
            and cache.get("question_dependent_scene_processing") is False
        ):
            raise ValueError(f"V34 cache boundary proof failed: {checkpoint.name}")
        schedule = _mapping(stage.get("schedule"), "v34.schedule")
        if not (
            schedule.get("optimizer_step_count") == 64
            and schedule.get("pair_unit_count") == 25
            and schedule.get("pair_unit_minimum_recurrence") == 2
            and schedule.get("pair_unit_maximum_recurrence") == 3
            and schedule.get("pair_units_with_third_recurrence") == 14
            and schedule.get("pair_units_atomic") is True
            and schedule.get("true_optimizer_step_per_schedule_row") is True
        ):
            raise ValueError(f"V34 schedule proof failed: {checkpoint.name}")
        history = metadata.get("history")
        if not isinstance(history, list) or len(history) != step + 1:
            raise ValueError(f"V34 history is not one row per true update: {checkpoint.name}")
        row = _mapping(history[-1], "history[-1]")
        if (
            row.get("optimizer_update") != step
            or row.get("validation_answer_token_nll") is None
            or row.get("validation_pair_metrics") is None
            or row.get("training_separation") is None
            or row.get("validation_adapted_prefix_separation") is None
        ):
            raise ValueError(f"V34 saved-arm diagnostics are incomplete: {checkpoint.name}")
        if step and row.get("separate_group_clipping") is not True:
            raise ValueError(f"V34 saved arm lacks separate clipping proof: {checkpoint.name}")
        runtime = json.loads((checkpoint / RUNTIME_METADATA_FILENAME).read_text(encoding="utf-8"))
        validate_runtime_checkpoint_metadata(runtime)
        if runtime != runtime_checkpoint_metadata(metadata):
            raise ValueError(f"V34 runtime metadata was not freshly sanitized: {checkpoint.name}")
        tensors = load_file(checkpoint / "adapter.safetensors", device="cpu")
        surface = _validate_surface(metadata, tensors)
        if frozen_hash is None:
            frozen_hash = str(surface["frozen_tensor_sha256"])
            update0_tensors = tensors
            source = load_file(contract.source_checkpoint / "adapter.safetensors", device="cpu")
            if set(source) != set(tensors) or any(not torch.equal(source[name], tensors[name]) for name in source):
                raise ValueError("V34 update zero is not tensor-bit-exact V33 update 64")
            equivalence = _mapping(
                legacy.get("update_zero_equivalence"), "v34 update-zero equivalence"
            )
            if not (
                equivalence.get("exact_v33_update64_source_tensors") is True
                and equivalence.get("exact_v33_update64_source_prefixes") is True
                and equivalence.get("exact_v33_update64_validation_nll") is True
                and equivalence.get("source_prefix_scene_count") == 22
                and equivalence.get("fresh_adam_state") is True
            ):
                raise ValueError("V34 update zero lacks exact tensor/prefix/NLL proof")
        else:
            if surface["frozen_tensor_sha256"] != frozen_hash:
                raise ValueError(f"V34 frozen tensors changed: {checkpoint.name}")
            assert update0_tensors is not None
            changed = {name for name in tensors if not torch.equal(tensors[name], update0_tensors[name])}
            if not changed or not changed.issubset(_AUTHORIZED):
                raise ValueError(f"V34 arm changed an unauthorized tensor: {checkpoint.name}")
        if stage.get("frozen_state_sha256") != frozen_hash:
            raise ValueError(f"V34 frozen-state metadata hash mismatch: {checkpoint.name}")
        if step:
            _optimizer_step_audit(checkpoint, step, settings)
        if step >= contract.early_gate_optimizer_step:
            gate = _mapping(stage.get("early_training_gate"), "v34 accepted early gate")
            if gate.get("passed") is not True or gate.get("training_scenes_only") is not True:
                raise ValueError(f"V34 arm lacks its accepted train-only gate: {checkpoint.name}")
    return checkpoints


@dataclass(frozen=True)
class V34RuntimeEvidence:
    validation: V33RuntimeEvidence
    training_separation: Mapping[str, Any]


class V34ArmEvaluator(Protocol):
    validation_scene_ids: tuple[str, ...]

    def install(self, tensors: Mapping[str, torch.Tensor]) -> None: ...

    def evaluate_v33(self) -> V33RuntimeEvidence: ...

    def evaluate_v34(self) -> V34RuntimeEvidence: ...

    def evaluate_aggregate_exact(self) -> tuple[int, int]: ...


class _V34RuntimeEvaluator(_V33RuntimeEvaluator):
    def __init__(
        self,
        config: dict[str, Any],
        control_config: dict[str, Any],
        checkpoint: Path,
        requirements: SelectionRequirements,
    ) -> None:
        super().__init__(config, control_config, checkpoint, requirements)
        freeze_for_v34(self.bundle)
        assert_v34_trainable_surface(self.bundle)
        train_records, _validation_records, _qa = load_v31_qa_records(config)
        self.train_units = build_exact_question_pair_units(train_records)
        changed, _unrelated = physical_pair_sets(self.train_units)
        self.train_scene_ids = tuple(sorted({scene for pair in changed.values() for scene in pair}))
        self.train_caches, audit = cache_pre_sidecar_scenes(
            self.bundle,
            self.train_scene_ids,
            allow_unpinned_source_scene_ids=tuple(
                f"scene_{index:06d}" for index in range(31, 39)
            ),
        )
        if not (
            audit.get("cache_boundary") == "complete_frozen_pre_sidecar_scene_stack"
            and audit.get("all_voxels_covered") is True
            and audit.get("question_inputs_to_scene_cache") is False
        ):
            raise ValueError("V34 selector training cache omitted voxels")
        update0 = load_file(checkpoint / "adapter.safetensors", device="cpu")
        self.install(update0)
        self.separation_reference: PrefixSeparationReference = build_prefix_separation_reference(
            self.train_units,
            self.train_caches,
            self.bundle,
            rms_floor=v34_settings(config).separation_rms_floor,
        )

    def evaluate_v34(self) -> V34RuntimeEvidence:
        validation = self.evaluate_v33()
        training = training_separation_diagnostics(
            reference=self.separation_reference,
            caches=self.train_caches,
            bundle=self.bundle,
            settings=v34_settings(self.config),
        )
        return V34RuntimeEvidence(validation=validation, training_separation=training)


def _close_mapping(recorded: Mapping[str, Any], observed: Mapping[str, Any]) -> None:
    fields = (
        "unique_changed_physical_pair_count",
        "all_nonchanged_train_scene_pair_count",
        "changed_ratio_mean",
        "changed_ratio_median",
        "changed_ratio_minimum",
        "changed_ratio_maximum",
        "unrelated_ratio_mean",
        "unrelated_ratio_median",
        "unrelated_ratio_p90",
        "unrelated_ratio_maximum",
        "unrelated_abs_log_ratio_p90",
        "unrelated_abs_log_ratio_maximum",
        "changed_selectivity_ratio_geometric_mean",
        "changed_selectivity_ratio_minimum",
        "changed_selectivity_over_1_02_count",
    )
    for field in fields:
        left, right = recorded.get(field), observed.get(field)
        if isinstance(left, bool) or isinstance(right, bool) or isinstance(left, int) and isinstance(right, int):
            if left != right:
                raise ValueError(f"V34 replay differs for {field}")
        elif not math.isclose(float(left), float(right), rel_tol=0.0, abs_tol=1e-6):
            raise ValueError(f"V34 replay differs for {field}: {left} != {right}")


def _promotion(
    selected: Mapping[str, Any] | None,
    *,
    approved_v29_aggregate: tuple[int, int],
    selected_aggregate: tuple[int, int] | None,
) -> dict[str, Any]:
    development_selected = selected is not None
    changed_threshold = development_selected and int(
        selected["greedy_exact_complete_units_correct"]
    ) >= 6
    aggregate_retained = (
        development_selected
        and selected_aggregate is not None
        and selected_aggregate[0] == approved_v29_aggregate[0]
        and selected_aggregate[1] / selected_aggregate[0]
        >= approved_v29_aggregate[1] / approved_v29_aggregate[0]
    )
    # Keep this exact three-key outward shape for final_once attestation.
    outward = {
        "development_checkpoint_selected": development_selected,
        "changed_complete_pair_threshold_met": changed_threshold,
        "aggregate_validation_exact_accuracy_retained": aggregate_retained,
    }
    internal = {
        "selected_checkpoint_is_numbered": development_selected and Path(
            str(selected["checkpoint"])
        ).name == f"update_{int(selected['optimizer_step']):03d}",
        "each_validation_family_demonstrated": development_selected and all(
            int(_mapping(selected["greedy_complete_units_by_family"], "families")[family]) >= 1
            for family in _VALIDATION_FAMILY_PAIR_IDS
        ),
        "old_color_12_sides_retained": development_selected
        and int(selected["color_full_vocab_sides"]) >= 12,
        "old_mirror_10_sides_retained": development_selected
        and int(selected["mirror_full_vocab_sides"]) >= 10,
        "approved_v29_controls_no_new_negatives": development_selected
        and not selected["new_negative_sides_vs_approved_v29"],
        "all_development_checks_passed": development_selected
        and all(_mapping(selected["checks"], "selected checks").values()),
    }
    eligible = all(outward.values()) and all(internal.values())
    return {
        "evaluated": development_selected,
        "checks": outward,
        "audited_internal_requirements": internal,
        "approved_v29_aggregate_validation": {
            "row_count": approved_v29_aggregate[0],
            "exact_correct": approved_v29_aggregate[1],
            "exact_accuracy": approved_v29_aggregate[1] / approved_v29_aggregate[0],
        },
        "selected_aggregate_validation": None
        if selected_aggregate is None
        else {
            "row_count": selected_aggregate[0],
            "exact_correct": selected_aggregate[1],
            "exact_accuracy": selected_aggregate[1] / selected_aggregate[0],
        },
        "eligible": eligible,
    }


def select_v34(
    config_path: Path,
    checkpoint_root: Path,
    *,
    evaluator_factory: Callable[
        [dict[str, Any], dict[str, Any], Path, SelectionRequirements], V34ArmEvaluator
    ] = _V34RuntimeEvaluator,
) -> dict[str, Any]:
    config = load_config(config_path)
    contract = v34_contract(config)
    terminal = require_v33_terminal_gate(config)
    checkpoints = validate_v34_checkpoint_envelope(config, checkpoint_root, contract)
    requirements = _selection_requirements(config)
    control = _retention_control_config(config)
    source_v29 = _source_v29_evidence(_metadata(checkpoints[0]))
    _validate_source_against_config(source_v29, config)
    evaluator = evaluator_factory(config, control, checkpoints[0], requirements)
    if tuple(evaluator.validation_scene_ids) != contract.v31.validation_scene_ids:
        raise ValueError("V34 evaluator must remain exactly on scenes 19--24")

    approved_v29_tensors = load_file(
        Path(str(source_v29["checkpoint"])) / "adapter.safetensors", device="cpu"
    )
    update0_tensors = load_file(checkpoints[0] / "adapter.safetensors", device="cpu")
    evaluator.install(
        _approved_v29_runtime_tensor_envelope(update0_tensors, approved_v29_tensors)
    )
    approved_v29_retention = evaluator.evaluate_v33()
    approved_v29_aggregate = evaluator.evaluate_aggregate_exact()

    arms: list[dict[str, Any]] = []
    baseline_pair: PairMarginEvidence | None = None
    baseline_prefix: Mapping[str, Any] | None = None
    baseline_family: Mapping[str, Any] | None = None
    update32_gate: Mapping[str, Any] | None = None
    for step, checkpoint in zip(contract.saved_optimizer_steps, checkpoints, strict=True):
        metadata = _metadata(checkpoint)
        _validate_runtime_metadata(checkpoint, metadata)
        _validate_no_leakage_or_final_scenes(metadata)
        tensors = load_file(checkpoint / "adapter.safetensors", device="cpu")
        evaluator.install(tensors)
        evidence = evaluator.evaluate_v34()
        base = evidence.validation.base
        pair = base.pair_margins
        recorded_pair = _pair_margin_evidence(
            metadata, expected_unit_count=requirements.validation_pair_unit_count
        )
        _compare_pair_evidence(recorded_pair, pair)
        stage = _mapping(metadata["v34_base_surface"], "v34 metadata")
        recorded_training = _mapping(stage.get("training_separation"), "recorded training separation")
        _close_mapping(recorded_training, evidence.training_separation)
        family = validation_family_teacher_metrics(
            _mapping(metadata["history"][-1]["validation_pair_metrics"], "pair metrics")
        )
        if baseline_pair is None:
            baseline_pair = pair
            baseline_prefix = evidence.validation.prefix_diagnostics
            baseline_family = family
        assert baseline_prefix is not None and baseline_pair is not None and baseline_family is not None
        ratios = prefix_separation_ratios(evidence.validation.prefix_diagnostics, baseline_prefix)
        nonmirror_teacher = sum(
            int(_mapping(family[name], name)["complete_units"])
            for name in ("book_support", "picture_support")
        )
        new_negatives = sorted(base.negative_sides - approved_v29_retention.base.negative_sides)
        training_checks = {
            "training_selectivity_geometric_mean_at_least_1_02": (
                float(evidence.training_separation["changed_selectivity_ratio_geometric_mean"])
                >= contract.development_training_changed_pair_selectivity_ratio_minimum
            ),
            "training_at_least_6_of_8_changed_pairs_over_1_02": (
                int(evidence.training_separation["changed_selectivity_over_1_02_count"])
                >= contract.development_training_changed_pair_coverage_minimum
            ),
        }
        validation_differential = ratios["weak_pair_mean"] - ratios["unrelated_mean"]
        checks = {
            **training_checks,
            "validation_nll_improved_from_v33_u64": _validation_nll(metadata)
            < _validation_nll(_metadata(checkpoints[0])),
            "teacher_pair_mean_improved_from_v33_u64": pair.mean_margin > baseline_pair.mean_margin,
            "teacher_pair_passed_units_not_lower": pair.passed_units >= baseline_pair.passed_units,
            "validation_weak_vs_unrelated_differential_met": validation_differential
            >= contract.development_validation_weak_minus_unrelated_ratio_minimum,
            "book_and_picture_each_exceed_unrelated": ratios["book_support"] > ratios["unrelated_mean"]
            and ratios["picture_support"] > ratios["unrelated_mean"],
            "validation_unrelated_ratio_two_sided_bounded": contract.development_unrelated_ratio_minimum
            <= ratios["unrelated_mean"]
            <= contract.development_unrelated_ratio_maximum,
            "nonmirror_teacher_complete": nonmirror_teacher >= 1,
            "greedy_development_unit_demonstrated": base.generation.exact_complete_units_correct >= 1,
            "old_color_retained_vs_approved_v29": base.color_full_vocab_sides >= 12,
            "old_mirror_retained_vs_approved_v29": base.mirror_full_vocab_sides >= 10,
            "no_new_negative_sides_vs_approved_v29": not new_negatives,
            "broad_retention_vs_approved_v29": base.generation.broad_exact_accuracy
            >= approved_v29_retention.base.generation.broad_exact_accuracy,
        }
        arm = {
            "checkpoint": str(checkpoint),
            "optimizer_step": step,
            "update": step,
            "validation_answer_token_nll": _validation_nll(metadata),
            "validation_pair_passed_units": pair.passed_units,
            "validation_pair_mean_margin": pair.mean_margin,
            "validation_pair_mean_margin_delta_from_v33_u64": pair.mean_margin
            - baseline_pair.mean_margin,
            "validation_family_teacher": family,
            "nonmirror_teacher_complete_units": nonmirror_teacher,
            "color_full_vocab_sides": base.color_full_vocab_sides,
            "mirror_full_vocab_sides": base.mirror_full_vocab_sides,
            "new_negative_sides_vs_approved_v29": new_negatives,
            "greedy_exact_complete_units_correct": base.generation.exact_complete_units_correct,
            "greedy_prediction_changed_units": base.generation.prediction_changed_units,
            "greedy_complete_units_by_family": dict(evidence.validation.greedy_complete_by_family),
            "greedy_prediction_changed_by_family": dict(
                evidence.validation.greedy_prediction_changed_by_family
            ),
            "broad_retention_exact_accuracy": base.generation.broad_exact_accuracy,
            "training_separation": dict(evidence.training_separation),
            "validation_adapted_prefix_separation": dict(evidence.validation.prefix_diagnostics),
            "validation_adapted_prefix_ratios_from_v33_u64": ratios,
            "validation_weak_minus_unrelated_ratio": validation_differential,
            "checks": checks,
            "eligible": False,
        }
        if step == contract.early_gate_optimizer_step:
            update32_gate = v34_early_training_gate(evidence.training_separation, contract)
            recorded_gate = _mapping(stage.get("early_training_gate"), "recorded early gate")
            if recorded_gate != update32_gate:
                raise ValueError("V34 independently replayed update-32 train-only gate differs")
        arms.append(arm)
    if update32_gate is None or update32_gate.get("passed") is not True:
        raise ValueError("A complete V34 run must contain a passing independent update-32 gate")
    for arm in arms:
        arm["eligible"] = (
            int(arm["optimizer_step"]) >= contract.early_gate_optimizer_step
            and all(_mapping(arm["checks"], "arm checks").values())
            and update32_gate["passed"] is True
        )
    candidates = [arm for arm in arms if arm["eligible"]]
    selected = min(
        candidates,
        key=lambda arm: (
            -int(arm["greedy_exact_complete_units_correct"]),
            -int(arm["nonmirror_teacher_complete_units"]),
            -float(arm["validation_weak_minus_unrelated_ratio"]),
            float(arm["validation_answer_token_nll"]),
            int(arm["optimizer_step"]),
        ),
        default=None,
    )
    selected_aggregate: tuple[int, int] | None = None
    if selected is not None:
        selected_path = checkpoint_root / f"update_{int(selected['optimizer_step']):03d}"
        evaluator.install(load_file(selected_path / "adapter.safetensors", device="cpu"))
        selected_aggregate = evaluator.evaluate_aggregate_exact()
    promotion = _promotion(
        selected,
        approved_v29_aggregate=approved_v29_aggregate,
        selected_aggregate=selected_aggregate,
    )
    return {
        "schema_version": 1,
        "artifact": "v34_base_surface_development_selection",
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
        "all_v33_learned_tensors_frozen": True,
        "exact_trainable_parameter_count": 199_808,
        "model_load_count": 1,
        "source_v29": source_v29,
        "retention_and_aggregate_baseline": "approved_v29",
        "improvement_baseline": "exact_v33_update_064",
        "v33_terminal_gate": {"path": terminal["path"], "sha256": terminal["sha256"]},
        "train_scene_ids": list(contract.v31.train_scene_ids),
        "validation_scene_ids": list(evaluator.validation_scene_ids),
        "saved_optimizer_steps": list(contract.saved_optimizer_steps),
        "all_saved_arms_inspected": True,
        "optimizer_state_steps_verified": list(contract.saved_optimizer_steps[1:]),
        "early_update32_training_only_gate": dict(update32_gate),
        "validation_used_for_training_continuation": False,
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
    report = select_v34(args.config, args.checkpoint_root)
    _atomic_json(args.output, report)
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
    return 0 if report["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "V34RuntimeEvidence",
    "select_v34",
    "validate_v34_checkpoint_envelope",
]
