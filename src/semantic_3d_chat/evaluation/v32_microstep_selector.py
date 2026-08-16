"""Independently inspect every saved V32 true-microstep development arm.

Only checkpoints at optimizer steps 0, 8, ..., 80 are saved.  This selector
recomputes all V30 tensor, leakage, teacher-margin, greedy counterfactual, and
retention evidence for every one of those arms while keeping validation fixed
to scenes 19--24.  Development progress (>=1/12 changed units) remains
strictly separate from chat promotion (>=6/12 plus aggregate non-regression).
"""

from __future__ import annotations

import argparse
import json
import re
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file

from semantic_3d_chat.config import config_hash, load_config
from semantic_3d_chat.evaluation.v27_sidecar_screen import _atomic_json
from semantic_3d_chat.evaluation.v28_stage_b_selector import _retention_control_config
from semantic_3d_chat.evaluation.v30_joint_pair_selector import (
    ArmEvaluator,
    PairMarginEvidence,
    SelectionRequirements,
    _compare_pair_evidence,
    _finite,
    _frozen_tensor_sha256,
    _metadata,
    _pair_margin_evidence,
    _RuntimeEvaluator,
    _select_eligible_arm,
    _selection_requirements,
    _source_v29_evidence,
    _validate_no_leakage_or_final_scenes,
    _validate_runtime_metadata,
    _validate_source_against_config,
    _validate_trainable_surface,
    _validate_update_zero,
    _validation_nll,
)
from semantic_3d_chat.training.checkpointing import (
    RUNTIME_METADATA_FILENAME,
    TRAINING_METADATA_FILENAME,
)
from semantic_3d_chat.training.train_microstep_v32 import (
    V32Contract,
    V32Settings,
    v31_rejection_status,
    v32_contract,
    v32_settings,
)

DEFAULT_CONFIG = Path("configs/experiments/gemma4_diverse28_microstep_v32.yaml")
DEFAULT_CHECKPOINT_ROOT = Path("data_gemma4/checkpoints/gemma4_v32_diverse28_microstep")
DEFAULT_OUTPUT = Path("reports/gemma4/metrics/v32_microstep_selection.json")
_SHA256 = re.compile(r"[0-9a-f]{64}")
_SCENE_ID = re.compile(r"scene_[0-9]{6}")


def _mapping(value: Any, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{field} must be a mapping")
    return value


def _validate_optimizer_state_step(
    path: Path, expected_step: int, settings: V32Settings
) -> None:
    """Prove that each saved nonzero arm contains an Adam state at that step."""

    optimizer_path = path / "optimizer.pt"
    if not optimizer_path.is_file():
        raise FileNotFoundError(f"V32 checkpoint lacks optimizer state: {optimizer_path}")
    payload = torch.load(optimizer_path, map_location="cpu", weights_only=True)
    if not isinstance(payload, Mapping):
        raise TypeError(f"V32 optimizer state is not a mapping: {path.name}")
    groups = payload.get("param_groups")
    state = payload.get("state")
    if not isinstance(groups, list) or len(groups) != 2 or not isinstance(state, Mapping):
        raise ValueError(
            f"V32 optimizer state must contain exactly two parameter groups: {path.name}"
        )
    parsed_groups = [
        _mapping(group, f"optimizer.param_groups[{index}]") for index, group in enumerate(groups)
    ]
    expected_groups = (
        ("dense_sidecar_adapter.output_surfaces", settings.sidecar_learning_rate, 2),
        (settings.trainable_bank, settings.decoder_learning_rate, 8),
    )
    for index, (group, (name, learning_rate, parameter_count)) in enumerate(
        zip(parsed_groups, expected_groups, strict=True)
    ):
        if (
            group.get("name") != name
            or float(group.get("lr", float("nan"))) != learning_rate
            or float(group.get("weight_decay", float("nan"))) != settings.weight_decay
            or len(group.get("params", ())) != parameter_count
        ):
            raise ValueError(
                f"V32 optimizer group {index} differs from the locked trainable surface: "
                f"{path.name}"
            )
    parameter_ids = [parameter for group in parsed_groups for parameter in group.get("params", ())]
    if not parameter_ids or len(parameter_ids) != len(set(parameter_ids)):
        raise ValueError(f"V32 optimizer parameter IDs are empty or repeated: {path.name}")
    if set(state) != set(parameter_ids):
        raise ValueError(f"V32 optimizer state does not cover every trainable tensor: {path.name}")
    observed_steps: list[int] = []
    for parameter_id in parameter_ids:
        entry = _mapping(state[parameter_id], f"optimizer.state[{parameter_id}]")
        if set(entry) != {"step", "exp_avg", "exp_avg_sq"}:
            raise ValueError(f"V32 Adam state fields changed: {path.name}")
        for moment_name in ("exp_avg", "exp_avg_sq"):
            moment = entry[moment_name]
            if not isinstance(moment, torch.Tensor) or not torch.isfinite(moment).all():
                raise ValueError(f"V32 Adam {moment_name} is invalid: {path.name}")
        raw_step = entry.get("step")
        if isinstance(raw_step, torch.Tensor):
            if raw_step.numel() != 1:
                raise ValueError(f"V32 Adam step is not scalar: {path.name}")
            raw_step = raw_step.item()
        if isinstance(raw_step, bool) or not isinstance(raw_step, (int, float)):
            raise TypeError(f"V32 Adam step is not numeric: {path.name}")
        parsed = int(raw_step)
        if float(raw_step) != parsed:
            raise ValueError(f"V32 Adam step is not integral: {path.name}")
        observed_steps.append(parsed)
    if set(observed_steps) != {expected_step}:
        raise ValueError(
            f"V32 optimizer state does not prove {expected_step} true updates: "
            f"observed={sorted(set(observed_steps))}"
        )


def validate_v32_checkpoint_envelope(
    config: Mapping[str, Any], checkpoint_root: Path, contract: V32Contract
) -> tuple[Path, ...]:
    """Require exactly update_000, update_008, ..., update_080."""

    paths = tuple(checkpoint_root / f"update_{step:03d}" for step in contract.saved_optimizer_steps)
    observed = sorted(path.name for path in checkpoint_root.glob("update_*") if path.is_dir())
    expected = [path.name for path in paths]
    if observed != expected:
        raise FileNotFoundError(
            f"V32 must expose every saved microstep arm: observed={observed} expected={expected}"
        )
    expected_config_hash = config_hash(dict(config))
    settings = v32_settings(config)
    required_files = {
        "adapter.safetensors",
        TRAINING_METADATA_FILENAME,
        RUNTIME_METADATA_FILENAME,
    }
    condition_provenance: tuple[str, str] | None = None
    for step, path in zip(contract.saved_optimizer_steps, paths, strict=True):
        missing = sorted(name for name in required_files if not (path / name).is_file())
        if missing:
            raise FileNotFoundError(f"V32 checkpoint {path.name} is incomplete: {missing}")
        metadata = json.loads((path / TRAINING_METADATA_FILENAME).read_text(encoding="utf-8"))
        if not isinstance(metadata, Mapping):
            raise TypeError(f"V32 checkpoint metadata is not a mapping: {path.name}")
        if metadata.get("optimizer_step") != step:
            raise ValueError(f"V32 checkpoint/update mismatch: {path.name}")
        if metadata.get("config_hash") != expected_config_hash:
            raise ValueError(f"V32 checkpoint config hash mismatch: {path.name}")
        v30 = _mapping(metadata.get("v30_joint_pair"), "metadata.v30_joint_pair")
        v32 = _mapping(metadata.get("v32_microstep"), "metadata.v32_microstep")
        if tuple(v30.get("train_scene_ids", ())) != contract.v31.train_scene_ids:
            raise ValueError(f"V32 checkpoint train split mismatch: {path.name}")
        if tuple(v30.get("validation_scene_ids", ())) != contract.v31.validation_scene_ids:
            raise ValueError(f"V32 checkpoint validation split mismatch: {path.name}")
        if v30.get("train_question_count") != contract.v31.train_question_count:
            raise ValueError(f"V32 checkpoint train QA count mismatch: {path.name}")
        if v30.get("validation_question_count") != contract.v31.validation_question_count:
            raise ValueError(f"V32 checkpoint validation QA count mismatch: {path.name}")
        if v30.get("final_test_scene_ids_loaded") != []:
            raise ValueError(f"V32 checkpoint touched deferred final scenes: {path.name}")
        cache = _mapping(v30.get("scene_cache"), "metadata.v30_joint_pair.scene_cache")
        expected_scene_ids = set(contract.v31.train_scene_ids) | set(
            contract.v31.validation_scene_ids
        )
        derived_scene_ids = {f"scene_{index:06d}" for index in range(31, 39)}
        pinned_scene_ids = expected_scene_ids - derived_scene_ids
        prefix_hashes = _mapping(
            cache.get("source_prefix_sha256_by_scene"),
            "metadata.v30_joint_pair.scene_cache.source_prefix_sha256_by_scene",
        )
        if (
            cache.get("scene_count") != len(expected_scene_ids)
            or cache.get("exact_source_scene_prefixes") is not True
            or cache.get("derived_source_prefixes_recomputed_bit_exact") is not True
            or set(cache.get("deterministically_derived_source_scene_ids", ()))
            != derived_scene_ids
            or set(cache.get("historically_pinned_source_scene_ids", ())) != pinned_scene_ids
            or set(prefix_hashes) != expected_scene_ids
            or any(_SHA256.fullmatch(str(value)) is None for value in prefix_hashes.values())
        ):
            raise ValueError(f"V32 exact-zero source-prefix provenance failed: {path.name}")
        loaded_environment_files = cache.get("loaded_environment_files")
        if not isinstance(loaded_environment_files, list):
            raise TypeError(f"V32 loaded-environment audit must be a list: {path.name}")
        loaded_scene_ids: list[str] = []
        for loaded_path in loaded_environment_files:
            matches = _SCENE_ID.findall(str(loaded_path))
            if len(matches) != 1:
                raise ValueError(f"V32 loaded-environment path is not scene-scoped: {path.name}")
            loaded_scene_ids.append(matches[0])
        if set(loaded_scene_ids) != expected_scene_ids or len(loaded_scene_ids) != len(
            expected_scene_ids
        ):
            raise ValueError(f"V32 loaded-environment coverage changed: {path.name}")
        if v32.get("optimizer_step") != step:
            raise ValueError(f"V32 nested optimizer-step mismatch: {path.name}")
        if v32.get("exact_trainable_parameter_count") != 329_216:
            raise ValueError(f"V32 checkpoint trainable surface changed: {path.name}")
        if tuple(v32.get("train_scene_ids", ())) != contract.v31.train_scene_ids:
            raise ValueError(f"V32 nested train split mismatch: {path.name}")
        if tuple(v32.get("validation_scene_ids", ())) != contract.v31.validation_scene_ids:
            raise ValueError(f"V32 nested validation split mismatch: {path.name}")
        if v32.get("deferred_final_scene_ids_loaded") != []:
            raise ValueError(f"V32 nested metadata loaded final scenes: {path.name}")
        if v32.get("source_is_approved_v29_update_004") is not True:
            raise ValueError(f"V32 checkpoint source is not approved V29: {path.name}")
        if v32.get("every_saved_arm_requires_independent_selection") is not True:
            raise ValueError(f"V32 checkpoint does not require independent selection: {path.name}")
        condition = _mapping(
            v32.get("conditional_v31_rejection"), "v32_microstep.conditional_v31_rejection"
        )
        report_path = str(condition.get("report", ""))
        report_sha = str(condition.get("report_sha256", ""))
        if (
            condition.get("status") != "rejected"
            or condition.get("training_authorized") is not True
            or Path(report_path).resolve() != contract.v31_selection_report
            or _SHA256.fullmatch(report_sha) is None
        ):
            raise ValueError(f"V32 checkpoint was not conditioned on V31 rejection: {path.name}")
        observed_condition = (report_path, report_sha)
        if condition_provenance is None:
            condition_provenance = observed_condition
        elif condition_provenance != observed_condition:
            raise ValueError(f"V32 V31-rejection provenance changed: {path.name}")
        schedule = _mapping(v32.get("schedule"), "v32_microstep.schedule")
        if (
            schedule.get("optimizer_step_count") != contract.optimizer_steps
            or schedule.get("true_optimizer_step_per_schedule_row") is not True
            or schedule.get("pair_units_atomic") is not True
            or schedule.get("every_pair_unit_recurred") is not True
            or schedule.get("pair_unit_minimum_recurrence", 0)
            < contract.minimum_pair_unit_recurrence
        ):
            raise ValueError(f"V32 checkpoint schedule contract failed: {path.name}")
        history = metadata.get("history")
        if not isinstance(history, list) or len(history) != step + 1:
            raise ValueError(f"V32 checkpoint lacks one history row per true update: {path.name}")
        if history[-1].get("optimizer_update") != step:
            raise ValueError(f"V32 history/update mismatch: {path.name}")
        if history[-1].get("validation_answer_token_nll") is None:
            raise ValueError(f"V32 saved arm lacks validation NLL: {path.name}")
        if history[-1].get("validation_pair_metrics") is None:
            raise ValueError(f"V32 saved arm lacks pair metrics: {path.name}")
        if step > 0:
            _validate_optimizer_state_step(path, step, settings)
    return paths


def select_v32(
    config_path: Path,
    checkpoint_root: Path,
    *,
    evaluator_factory: Callable[
        [dict[str, Any], dict[str, Any], Path, SelectionRequirements], ArmEvaluator
    ] = _RuntimeEvaluator,
) -> dict[str, Any]:
    """Recompute every saved V32 arm and apply development/promotion gates."""

    config = load_config(config_path)
    contract = v32_contract(config)
    checkpoints = validate_v32_checkpoint_envelope(config, checkpoint_root, contract)
    current_condition = v31_rejection_status(config)
    if current_condition.get("training_authorized") is not True:
        raise ValueError("V32 selection requires the audited V31 rejection to remain available")
    first_condition = _mapping(
        _mapping(_metadata(checkpoints[0]).get("v32_microstep"), "v32_microstep").get(
            "conditional_v31_rejection"
        ),
        "v32_microstep.conditional_v31_rejection",
    )
    if first_condition.get("report_sha256") != current_condition.get("report_sha256"):
        raise ValueError("V31 rejection report changed after V32 training")
    requirements = _selection_requirements(config)
    if (
        requirements.minimum_greedy_complete_units_correct
        != contract.development_changed_complete_pairs_minimum
        or requirements.promotion_changed_complete_pairs_minimum
        != contract.chat_promotion_changed_complete_pairs_minimum
    ):
        raise ValueError("V32 selector gates differ from the locked 1/12 and 6/12 requirements")
    control_config = _retention_control_config(config)
    first_metadata = _metadata(checkpoints[0])
    source = _source_v29_evidence(first_metadata)
    _validate_source_against_config(source, config)
    evaluator = evaluator_factory(config, control_config, checkpoints[0], requirements)
    if tuple(evaluator.validation_scene_ids) != contract.v31.validation_scene_ids:
        raise ValueError("V32 evaluator validation set must remain exactly scenes 19--24")

    arms: list[dict[str, Any]] = []
    frozen_hash: str | None = None
    baseline_negatives: frozenset[tuple[str, str]] | None = None
    baseline_pair: PairMarginEvidence | None = None
    baseline_broad_accuracy: float | None = None
    exact_source_contract: Mapping[str, Any] | None = None
    checkpoint_by_step = dict(zip(contract.saved_optimizer_steps, checkpoints, strict=True))
    for arm_index, (step, checkpoint) in enumerate(checkpoint_by_step.items()):
        metadata = _metadata(checkpoint)
        _validate_runtime_metadata(checkpoint, metadata)
        _validate_no_leakage_or_final_scenes(metadata)
        tensors = load_file(checkpoint / "adapter.safetensors", device="cpu")
        audit = _validate_trainable_surface(metadata, tensors)
        observed_frozen = _frozen_tensor_sha256(tensors)
        v30 = _mapping(metadata.get("v30_joint_pair"), "metadata.v30_joint_pair")
        if v30.get("frozen_inherited_state_sha256") != observed_frozen:
            raise ValueError(f"Frozen inherited metadata hash mismatch in {checkpoint.name}")
        if frozen_hash is None:
            frozen_hash = observed_frozen
        elif frozen_hash != observed_frozen:
            raise RuntimeError(f"Inherited frozen tensors changed in {checkpoint.name}")
        if arm_index == 0:
            tolerance = _finite(
                _mapping(config.get("v30_joint_pair"), "v30_joint_pair").get(
                    "update_zero_validation_nll_absolute_tolerance"
                ),
                "update_zero_validation_nll_absolute_tolerance",
            )
            _validate_update_zero(
                metadata,
                audit,
                source,
                expected_nll_tolerance=tolerance,
            )
            exact_source_contract = {
                key: v30.get(key)
                for key in (
                    "source_v29_checkpoint",
                    "source_v29_adapter_sha256",
                    "source_v29_runtime_metadata_sha256",
                    "source_v29_selection_report",
                    "source_v29_selection_report_sha256",
                    "source_v29_selected_update",
                )
            }
        else:
            assert exact_source_contract is not None
            if any(v30.get(key) != value for key, value in exact_source_contract.items()):
                raise ValueError(f"V29 source provenance changed in {checkpoint.name}")

        recorded_pair = _pair_margin_evidence(
            metadata, expected_unit_count=requirements.validation_pair_unit_count
        )
        evaluator.install(tensors)
        observed = evaluator.evaluate()
        _compare_pair_evidence(recorded_pair, observed.pair_margins)
        if baseline_negatives is None:
            baseline_negatives = observed.negative_sides
        if baseline_pair is None:
            baseline_pair = observed.pair_margins
        if baseline_broad_accuracy is None:
            baseline_broad_accuracy = observed.generation.broad_exact_accuracy
        new_negatives = sorted(observed.negative_sides - baseline_negatives)
        validation_nll = _validation_nll(metadata)
        mean_delta = observed.pair_margins.mean_margin - baseline_pair.mean_margin
        passed_delta = observed.pair_margins.passed_units - baseline_pair.passed_units
        checks = {
            "color_retained": (
                observed.color_full_vocab_sides >= requirements.color_full_vocab_sides
            ),
            "mirror_retained": (
                observed.mirror_full_vocab_sides >= requirements.mirror_full_vocab_sides
            ),
            "no_new_negative_sides": not new_negatives,
            "below_selected_v29_source_nll": (
                validation_nll < float(source["validation_answer_token_nll"])
            ),
            "pair_mean_margin_strictly_improved": (
                mean_delta > 0.0 and mean_delta >= requirements.minimum_mean_margin_improvement
            ),
            "pair_passed_units_improved": (
                passed_delta >= requirements.minimum_passed_unit_improvement
            ),
            "pair_minimum_margin_met": (
                observed.pair_margins.minimum_margin >= requirements.minimum_pair_margin
            ),
            "greedy_changed_units_demonstrated": (
                observed.generation.exact_complete_units_correct
                >= contract.development_changed_complete_pairs_minimum
            ),
            "broad_exact_accuracy_retained": (
                observed.generation.broad_exact_accuracy >= baseline_broad_accuracy
            ),
        }
        arms.append(
            {
                "checkpoint": str(checkpoint),
                "arm_index": arm_index,
                "optimizer_step": step,
                "update": step,
                "fresh_bank_state_sha256": audit["fresh_bank_state_sha256"],
                "frozen_inherited_state_sha256": observed_frozen,
                "validation_answer_token_nll": validation_nll,
                "validation_pair_passed_units": observed.pair_margins.passed_units,
                "validation_pair_passed_sides": observed.pair_margins.passed_sides,
                "validation_pair_mean_margin": observed.pair_margins.mean_margin,
                "validation_pair_minimum_margin": observed.pair_margins.minimum_margin,
                "validation_pair_mean_margin_delta_from_update0": mean_delta,
                "validation_pair_passed_unit_delta_from_update0": passed_delta,
                "color_full_vocab_sides": observed.color_full_vocab_sides,
                "color_full_vocab_units": observed.color_full_vocab_units,
                "mirror_full_vocab_sides": observed.mirror_full_vocab_sides,
                "mirror_full_vocab_units": observed.mirror_full_vocab_units,
                "new_negative_sides": new_negatives,
                "greedy_changed_row_count": observed.generation.changed_row_count,
                "greedy_changed_unit_count": observed.generation.changed_unit_count,
                "greedy_exact_correct_sides": observed.generation.exact_correct_sides,
                "greedy_exact_complete_units_correct": (
                    observed.generation.exact_complete_units_correct
                ),
                "greedy_prediction_changed_units": observed.generation.prediction_changed_units,
                "broad_retention_exact_correct": observed.generation.broad_exact_correct,
                "broad_retention_row_count": observed.generation.broad_row_count,
                "broad_retention_exact_accuracy": observed.generation.broad_exact_accuracy,
                "prefix_sha256_by_validation_scene": dict(
                    sorted(observed.prefix_sha256_by_scene.items())
                ),
                "checks": checks,
                "eligible": arm_index > 0 and all(checks.values()),
            }
        )

    selected = _select_eligible_arm(arms)
    promotion_evaluator = getattr(evaluator, "evaluate_aggregate_exact", None)
    promotion: dict[str, Any] = {
        "label": requirements.promotion_label,
        "evaluated": False,
        "validation_changed_complete_pairs_minimum": (
            contract.chat_promotion_changed_complete_pairs_minimum
        ),
        "aggregate_validation_exact_accuracy_no_regression": True,
        "update0_aggregate_validation": None,
        "selected_aggregate_validation": None,
        "checks": {
            "development_checkpoint_selected": selected is not None,
            "changed_complete_pair_threshold_met": False,
            "aggregate_validation_exact_accuracy_retained": False,
        },
        "eligible": False,
    }
    if selected is not None and callable(promotion_evaluator):
        evaluator.install(load_file(checkpoints[0] / "adapter.safetensors", device="cpu"))
        update0_count, update0_correct = promotion_evaluator()
        selected_step = int(selected["optimizer_step"])
        evaluator.install(
            load_file(checkpoint_by_step[selected_step] / "adapter.safetensors", device="cpu")
        )
        selected_count, selected_correct = promotion_evaluator()
        if update0_count <= 0 or selected_count != update0_count:
            raise ValueError("V32 aggregate validation promotion audits are misaligned")
        update0_accuracy = update0_correct / update0_count
        selected_accuracy = selected_correct / selected_count
        promotion_checks = {
            "development_checkpoint_selected": True,
            "changed_complete_pair_threshold_met": (
                int(selected["greedy_exact_complete_units_correct"])
                >= contract.chat_promotion_changed_complete_pairs_minimum
            ),
            "aggregate_validation_exact_accuracy_retained": selected_accuracy >= update0_accuracy,
        }
        promotion.update(
            {
                "evaluated": True,
                "update0_aggregate_validation": {
                    "row_count": update0_count,
                    "exact_correct": update0_correct,
                    "exact_accuracy": update0_accuracy,
                },
                "selected_aggregate_validation": {
                    "row_count": selected_count,
                    "exact_correct": selected_correct,
                    "exact_accuracy": selected_accuracy,
                },
                "checks": promotion_checks,
                "eligible": all(promotion_checks.values()),
            }
        )
    assert baseline_pair is not None
    assert baseline_broad_accuracy is not None
    return {
        "schema_version": 1,
        "artifact": "v32_true_microstep_development_selection",
        "development_validation_model_selection_only": True,
        "final_test_scenes_touched": False,
        "deferred_final_scene_ids": list(contract.v31.deferred_final_scene_ids),
        "training_evaluation_only": True,
        "question_text_serialized": False,
        "answer_text_serialized": False,
        "oracle_loaded": False,
        "question_dependent_scene_processing": False,
        "question_dependent_retrieval": False,
        "all_scene_prefixes_built_before_questions": True,
        "model_load_count": 1,
        "source_v29": source,
        "train_scene_ids": list(contract.v31.train_scene_ids),
        "validation_scene_ids": list(evaluator.validation_scene_ids),
        "frozen_inherited_state_sha256": frozen_hash,
        "saved_optimizer_steps": list(contract.saved_optimizer_steps),
        "all_saved_arms_inspected": True,
        "optimizer_state_steps_verified": list(contract.saved_optimizer_steps[1:]),
        "requirements": {
            "color_full_vocab_sides": requirements.color_full_vocab_sides,
            "mirror_full_vocab_sides": requirements.mirror_full_vocab_sides,
            "no_new_negative_sides": True,
            "selected_v29_source_nll_must_improve": True,
            "validation_pair_unit_count": requirements.validation_pair_unit_count,
            "minimum_pair_margin": requirements.minimum_pair_margin,
            "minimum_mean_margin_improvement": requirements.minimum_mean_margin_improvement,
            "minimum_passed_unit_improvement": requirements.minimum_passed_unit_improvement,
            "greedy_changed_row_count": requirements.greedy_changed_row_count,
            "minimum_greedy_complete_units_correct": (
                contract.development_changed_complete_pairs_minimum
            ),
            "broad_retention_subset_size": requirements.broad_retention_subset_size,
            "broad_exact_accuracy_no_regression": True,
            "chat_promotion_changed_complete_pairs_minimum": (
                contract.chat_promotion_changed_complete_pairs_minimum
            ),
            "chat_promotion_aggregate_validation_exact_accuracy_no_regression": True,
        },
        "update0_pair_mean_margin": baseline_pair.mean_margin,
        "update0_pair_passed_units": baseline_pair.passed_units,
        "update0_broad_exact_accuracy": baseline_broad_accuracy,
        "arms": arms,
        "selected_checkpoint": None if selected is None else selected["checkpoint"],
        # Promotion uses the stage-neutral selector field name.  Preserve the
        # optimizer-specific spelling as additional V32 diagnostic evidence.
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
    report = select_v32(args.config, args.checkpoint_root)
    _atomic_json(args.output, report)
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
    return 0 if report["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "select_v32",
    "validate_v32_checkpoint_envelope",
]
