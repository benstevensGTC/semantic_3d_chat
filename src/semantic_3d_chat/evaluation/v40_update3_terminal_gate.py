"""Seal V40's pre-update-3 guard failure and authorize exact V41 training.

The terminal audit reads only pinned source/config/code/report/checkpoint files.
It never imports Gemma and never opens QA, maps, validation, oracle, or final
scene inputs.  V40 updates one and two existed only in process memory; the raw
component guard rejected update three before clipping or an optimizer step, and
only the exact update-zero checkpoint was persisted.

The resulting authorization is intentionally narrow: V41 may restart from the
exact V40/V38 update-zero state and update the existing layer-14 LoRA-B tensor
with a deterministic CPU-float64 half-space projection and momentum-free SGD.
It authorizes neither validation nor runtime promotion.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import yaml

from semantic_3d_chat.config import PROJECT_ROOT

DEFAULT_CONFIG = Path("configs/experiments/gemma4_diverse28_cross_preserving_v40.yaml")
DEFAULT_TRAINER = Path(
    "src/semantic_3d_chat/training/train_cross_preserving_v40.py"
)
DEFAULT_CHECKPOINT_ROOT = Path(
    "data_gemma4/checkpoints/gemma4_v40_diverse28_cross_preserving_l14_query"
)
DEFAULT_FAILURE = DEFAULT_CHECKPOINT_ROOT / "guard_failure_update_003.json"
DEFAULT_OUTPUT = Path("reports/gemma4/metrics/v40_update3_terminal_gate.json")

_V39_TERMINAL = Path("reports/gemma4/metrics/v39_layer14_query_terminal_gate.json")
_PROTECTED = Path(
    "reports/gemma4/metrics/"
    "training_selection_gemma4_color_mirror_full_vocab_v11_resume36.json"
)
_TRAINING_TEST = Path("tests/test_v40_cross_preserving_training.py")
_SELECTOR_TEST = Path("tests/test_v40_cross_preserving_selector.py")

_LOCKED_SHA256 = {
    "v39_terminal": "fcf0494c18ed13c3f1fe54eb109a51391183a4eeb14abb9dbd2ad0ad0ca448c3",
    "v40_config": "5e7d67a91a10f65e44699a7af1644fffff481dcd21ce34a028cf371048f1c9bc",
    "v40_trainer": "0801d580e4903c82a481f81383a7319e883e431b211fb18761d4af2a72d1fdaa",
    "v40_training_test": "bf8e5f5acca9e77546680a3c84243e3ccb3b9fffc93da85f39604fa25a4fa2ac",
    "v40_selector_test": "ed6abefded33853a94236000bef6f44fa307e417e0c922f6c4d005c73df19e52",
    "guard_failure": "0136edab8669346e4c24163650659608a7bf728fb8244ad81366a8efb5fa1f61",
    "update0_adapter": "b0bdceb7699e9d97467915c69186f433d3b0fac2b09144d38c3078afe1f70cb0",
    "update0_metadata": "ee74b1572061cae09d20bdf2b07e5f94ce9ef5c3ebfb6908131448bf8e5b484d",
    "update0_runtime": "209858f923ffa0916484209aeefad6f56a2cb4902bbd0dacd29decc222245c49",
    "protected": "c0086f66edbb8854a7938e09c57535bfd47100adbaf3b3c95eeb4b08014ce2f8",
}

_FULL_STATE_SHA256 = (
    "7b951c6d7ae4f7b50603159f0bc4dfb4d50b5b40f9325134d78d1de1dae87fc0"
)
_SOURCE_TARGET_SHA256 = (
    "d0834cc588ee2a9edf08aabedfd01e0a6d2b01c6b6ae7e3a3d764eaddf58cc3e"
)
_TRANSIENT_PRE_UPDATE3_TARGET_SHA256 = (
    "8c50aa3a5975f450c3c95fb00dbf077a33285bf22ac3208f5d745cd617bd8d48"
)
_FROZEN_SHA256 = (
    "cec01bc088bb87c6bb44e0659eb03aa766f951ddeee706ca9a70edaa080dea5e"
)
_V28_BANK_SHA256 = (
    "cc9dfa838bb87f32e2922d675658af4a1085d53a84ccdca6d5bacc6f7097217b"
)
_V23_SHA256 = (
    "5c9233bd96b381e2f63443f8a739a868a21997b28061d90fb407a46d9de2d4cb"
)
_CORE_SHA256 = (
    "92652fd2dbde2406227503f50717b2031baa1bcbc050902a379ddb9ddb52764f"
)
_PAIR_SCHEDULE_SHA256 = (
    "4e26160c9a36e20ac894ff5e26604e6fad985ff5c84211282fd9807c9e738b3f"
)
_FULL_SCHEDULE_SHA256 = (
    "2e7a2a8136c968739a8aaaa1138be87bec322fefa0fb340e6fb9bc6e07278bea"
)
_CONFIG_HASH = "7f8de1306c3b"

_TARGET_NAME = "lora_banks.extension_v28_stage_b_query.adapters.1.lora_b"
_TARGET_SHAPE = (4096, 4)
_TARGET_PARAMETER_COUNT = 16_384
_TRAIN_SCENES = tuple(
    f"scene_{index:06d}" for index in (*range(11, 19), *range(31, 39))
)
_SAVED_STEPS = (0, 8, 16, 24, 32, 40, 41)
_DIAGNOSTIC_STEPS = (0, 8, 16, 41)
_UPDATE_PATTERN = re.compile(r"update_[0-9]{3}")

_GATES: Mapping[str, Mapping[str, Any]] = {
    "update8": {
        "optimizer_step": 8,
        "priority_side_deficit_minimum_improvement": 0.5,
        "complete_units_minimum": 9,
        "positive_sides_minimum": 34,
        "cross_prefix_complete_units_minimum": 17,
        "broad_nll_maximum_increase": 0.02,
    },
    "update16": {
        "optimizer_step": 16,
        "require_update8_passed": True,
        "priority_side_deficit_minimum_improvement": 3.12,
        "complete_units_minimum": 10,
        "positive_sides_minimum": 35,
        "complete_physical_pair_coverage_minimum": 5,
        "book_or_picture_complete_units_minimum": 1,
        "cross_prefix_complete_units_minimum": 17,
        "broad_nll_maximum_increase": 0.02,
    },
    "update41": {
        "optimizer_step": 41,
        "require_update16_passed": True,
        "priority_side_deficit_minimum_improvement": 6.24,
        "complete_units_minimum": 12,
        "positive_sides_minimum": 37,
        "complete_physical_pair_coverage_minimum": 6,
        "book_complete_units_minimum": 1,
        "picture_complete_units_minimum": 1,
        "mirror_complete_units_minimum": 2,
        "cross_prefix_complete_units_minimum": 18,
        "greedy_complete_units_minimum": 6,
        "require_one_greedy_complete_per_priority_family": True,
        "broad_greedy_accuracy_must_meet_source": True,
        "broad_nll_maximum_increase": 0.02,
    },
}


def _resolve(path: str | Path) -> Path:
    value = Path(path)
    return value.resolve() if value.is_absolute() else (PROJECT_ROOT / value).resolve()


def _relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path.resolve())


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _real_file(path: Path, field: str) -> None:
    if path.is_symlink() or not path.is_file():
        raise FileNotFoundError(f"{field} must be a real file: {path}")


def _locked_file(path: Path, expected: str, field: str) -> None:
    _real_file(path, field)
    observed = _sha256(path)
    if observed != expected:
        raise ValueError(f"{field} bytes changed: expected {expected}, observed {observed}")


def _mapping(value: object, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{field} must be a mapping")
    return value


def _sequence(value: object, field: str) -> Sequence[Any]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise TypeError(f"{field} must be a sequence")
    return value


def _read_json(path: Path, expected: str, field: str) -> Mapping[str, Any]:
    _locked_file(path, expected, field)
    with path.open("r", encoding="utf-8") as handle:
        return _mapping(json.load(handle), field)


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
        delete=False,
    ) as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
        temporary = Path(handle.name)
    os.replace(temporary, path)


def _validate_config(path: Path) -> dict[str, Any]:
    _locked_file(path, _LOCKED_SHA256["v40_config"], "V40 config")
    with path.open("r", encoding="utf-8") as handle:
        config = _mapping(yaml.safe_load(handle), "V40 config")
    if config.get("_base_") != "gemma4_diverse28_query_recovery_v38.yaml":
        raise ValueError("V40 config base changed")
    training = _mapping(config.get("training"), "V40 training config")
    objective = _mapping(
        training.get("v40_cross_preserving"), "V40 objective config"
    )
    expected_objective = {
        "enabled": True,
        "optimizer_steps": 41,
        "broad_nll_weight": 1.0,
        "pair_correct_nll_weight": 0.5,
        "side_hinge_weight": 8.0,
        "side_hinge_margin": 0.5,
        "cross_prefix_flip_weight": 56.0,
        "cross_prefix_flip_margin": 0.10,
        "residual_penalty_weight": 0.0,
        "residual_penalty_scale": 0.05,
        "learning_rate": 0.003,
        "weight_decay": 0.0,
        "gradient_clip_norm": 1.0,
    }
    if dict(objective) != expected_objective:
        raise ValueError("V40 objective/optimizer config changed")
    contract = _mapping(config.get("v40_cross_preserving"), "V40 contract config")
    expected_scalars = {
        "v39_terminal_report_sha256": _LOCKED_SHA256["v39_terminal"],
        "source_optimizer_step": 0,
        "source_full_tensor_state_sha256": _FULL_STATE_SHA256,
        "complete_v28_bank_state_sha256": _V28_BANK_SHA256,
        "frozen_excluding_target_state_sha256": _FROZEN_SHA256,
        "target_source_state_sha256": _SOURCE_TARGET_SHA256,
        "source_v23_state_sha256": _V23_SHA256,
        "source_block_core_state_sha256": _CORE_SHA256,
        "source_optimizer_state_loaded": False,
        "source_optimizer_file_opened": False,
        "validation_qa_loaded_during_training": False,
        "continuation_gates_use_training_only": True,
        "question_dependent_scene_processing": False,
        "question_dependent_retrieval": False,
        "pair_schedule_sha256": _PAIR_SCHEDULE_SHA256,
        "schedule_sha256": _FULL_SCHEDULE_SHA256,
        "saved_optimizer_steps": list(_SAVED_STEPS),
        "per_unit_nll_diagnostics_required_at_steps": list(_DIAGNOSTIC_STEPS),
        "final_test_deferred": True,
    }
    if any(contract.get(key) != value for key, value in expected_scalars.items()):
        raise ValueError("V40 pinned contract changed")
    if (
        contract.get("target_parameter_names") != [_TARGET_NAME]
        or contract.get("target_parameter_shapes") != [list(_TARGET_SHAPE)]
        or contract.get("target_parameter_count") != _TARGET_PARAMETER_COUNT
        or contract.get("target_tensor_count") != 1
    ):
        raise ValueError("V40 B-only target surface changed")
    return {
        "config_sha256": _LOCKED_SHA256["v40_config"],
        "normalized_checkpoint_config_hash": _CONFIG_HASH,
        "objective_exact": True,
        "b_only_surface_exact": True,
        "train_only_boundaries_exact": True,
    }


def _validate_update_zero(checkpoint: Path) -> tuple[dict[str, Any], list[Path]]:
    if checkpoint.is_symlink() or not checkpoint.is_dir():
        raise FileNotFoundError(f"V40 update zero must be a real directory: {checkpoint}")
    expected_names = {"adapter.safetensors", "metadata.json", "runtime_metadata.json"}
    observed_names = {path.name for path in checkpoint.iterdir()}
    if observed_names != expected_names:
        raise ValueError(f"V40 update-zero file set changed: {sorted(observed_names)}")
    files = [checkpoint / name for name in sorted(expected_names)]
    _locked_file(
        checkpoint / "adapter.safetensors",
        _LOCKED_SHA256["update0_adapter"],
        "V40 update-zero adapter",
    )
    metadata = _read_json(
        checkpoint / "metadata.json",
        _LOCKED_SHA256["update0_metadata"],
        "V40 update-zero metadata",
    )
    _locked_file(
        checkpoint / "runtime_metadata.json",
        _LOCKED_SHA256["update0_runtime"],
        "V40 update-zero runtime metadata",
    )
    if (
        metadata.get("schema_version") != 1
        or metadata.get("optimizer_step") != 0
        or metadata.get("config_hash") != _CONFIG_HASH
    ):
        raise ValueError("V40 update-zero metadata identity changed")
    history = _sequence(metadata.get("history"), "V40 update-zero history")
    if len(history) != 1:
        raise ValueError("V40 update zero contains trained history")
    row0 = _mapping(history[0], "V40 update-zero history row")
    if (
        row0.get("optimizer_update") != 0
        or row0.get("saved_checkpoint") is not True
        or row0.get("query_bank_state_sha256") != _SOURCE_TARGET_SHA256
        or row0.get("frozen_excluding_query_state_sha256") != _FROZEN_SHA256
        or row0.get("scene_prefix_and_residual_exact") is not True
        or row0.get("validation_qa_loaded") is not False
        or row0.get("oracle_environment_files_loaded") is not False
    ):
        raise ValueError("V40 update-zero history attestation changed")
    stage = _mapping(metadata.get("v40_cross_preserving"), "V40 stage metadata")
    if (
        stage.get("schema_version") != 1
        or stage.get("optimizer_step") != 0
        or stage.get("source_v38_u0_tensor_state_sha256") != _FULL_STATE_SHA256
        or stage.get("update_zero_tensor_state_sha256") != _FULL_STATE_SHA256
        or stage.get("source_target_lora_b_state_sha256") != _SOURCE_TARGET_SHA256
        or stage.get("target_lora_b_state_sha256") != _SOURCE_TARGET_SHA256
        or stage.get("frozen_excluding_query_state_sha256") != _FROZEN_SHA256
        or stage.get("complete_v28_bank_state_sha256") != _V28_BANK_SHA256
        or stage.get("hybrid_v23_state_sha256") != _V23_SHA256
        or stage.get("source_block_core_state_sha256") != _CORE_SHA256
        or stage.get("source_optimizer_files_opened") is not False
        or stage.get("source_optimizer_states_loaded") is not False
        or stage.get("validation_qa_loaded") is not False
        or stage.get("oracle_environment_files_loaded") is not False
        or stage.get("deferred_final_scene_ids_loaded") != []
        or stage.get("question_dependent_retrieval") is not False
        or stage.get("question_dependent_scene_processing") is not False
        or stage.get("update8_train_only_gate") is not None
        or stage.get("update16_train_only_gate") is not None
        or stage.get("update41_train_only_gate") is not None
    ):
        raise ValueError("V40 update-zero stage attestation changed")
    surface = _mapping(stage.get("trainable_surface"), "V40 trainable surface")
    if (
        surface.get("target_parameter_names") != [_TARGET_NAME]
        or surface.get("trainable_tensor_count") != 1
        or surface.get("trainable_parameter_count") != _TARGET_PARAMETER_COUNT
        or surface.get("every_other_tensor_and_buffer_frozen") is not True
    ):
        raise ValueError("V40 persisted B-only surface changed")
    qa = _mapping(stage.get("train_qa_dataset"), "V40 train QA attestation")
    cache = _mapping(stage.get("scene_cache"), "V40 scene-cache attestation")
    if (
        tuple(qa.get("train_scene_ids", ())) != _TRAIN_SCENES
        or qa.get("validation_qa_loaded") is not False
        or qa.get("deferred_final_qa_loaded") is not False
        or qa.get("oracle_environment_files_loaded") is not False
        or tuple(cache.get("scene_ids", ())) != _TRAIN_SCENES
        or cache.get("scene_scope") != "training_only"
        or cache.get("validation_qa_loaded") is not False
        or cache.get("validation_environment_maps_loaded") is not False
        or cache.get("validation_scene_ids_loaded") != []
        or cache.get("deferred_final_scene_ids_loaded") != []
        or cache.get("oracle_environment_files_loaded") is not False
        or cache.get("all_occupied_blocks_processed") is not True
        or cache.get("all_voxels_covered") is not True
    ):
        raise ValueError("V40 persisted train-only scene boundary changed")
    baseline = _mapping(
        _mapping(stage.get("update_zero_attestation"), "V40 update-zero attestation").get(
            "behavioral_baseline"
        ),
        "V40 behavioral baseline",
    )
    if (
        baseline.get("passed") is not True
        or baseline.get("recomputed_before_optimizer_step_1") is not True
        or baseline.get("training_scenes_only") is not True
        or baseline.get("validation_qa_loaded") is not False
    ):
        raise ValueError("V40 update-zero behavioral baseline changed")
    return {
        "optimizer_step": 0,
        "only_update_zero_history_persisted": True,
        "adapter_sha256": _LOCKED_SHA256["update0_adapter"],
        "metadata_sha256": _LOCKED_SHA256["update0_metadata"],
        "runtime_metadata_sha256": _LOCKED_SHA256["update0_runtime"],
        "full_tensor_state_sha256": _FULL_STATE_SHA256,
        "target_lora_b_state_sha256": _SOURCE_TARGET_SHA256,
        "frozen_excluding_b_state_sha256": _FROZEN_SHA256,
        "source_optimizer_or_checkpoint_optimizer_present": False,
        "training_scene_count": 16,
        "validation_loaded": False,
        "oracle_loaded": False,
        "final_test_loaded": False,
    }, files


def _validate_failure(path: Path) -> tuple[dict[str, Any], Mapping[str, Any]]:
    failure = _read_json(
        path, _LOCKED_SHA256["guard_failure"], "V40 update-three guard failure"
    )
    if (
        failure.get("schema_version") != 1
        or failure.get("artifact") != "v40_pre_step_gradient_guard_failure"
        or failure.get("optimizer_step_not_executed") != 3
        or failure.get("optimizer_step_executed") is not False
        or failure.get("checkpoint_written") is not False
        or failure.get("validation_qa_loaded") is not False
        or failure.get("oracle_environment_files_loaded") is not False
    ):
        raise ValueError("V40 update-three failure envelope changed")
    audit = _mapping(failure.get("audit"), "V40 failure audit")
    guard = _mapping(audit.get("component_gradient_guard"), "V40 component guard")
    if (
        audit.get("failed_guard_stage") != "raw_component_direction"
        or audit.get("clip_direction_attestation") is not None
        or audit.get("target_hash_before") != _TRANSIENT_PRE_UPDATE3_TARGET_SHA256
        or audit.get("target_hash_after") != _TRANSIENT_PRE_UPDATE3_TARGET_SHA256
        or audit.get("frozen_excluding_b_hash_before") != _FROZEN_SHA256
        or audit.get("frozen_excluding_b_hash_after") != _FROZEN_SHA256
        or guard.get("schema_version") != 1
        or guard.get("guard_stage") != "raw_component_direction"
        or guard.get("gradient_method")
        != "torch.autograd.grad_separate_components"
        or guard.get("component_order") != ["broad", "answer", "side", "cross"]
        or guard.get("raw_guard_passed") is not False
        or guard.get("guard_evaluated_before_clip_and_optimizer_step") is not True
    ):
        raise ValueError("V40 pre-step-three mutation/guard attestation changed")
    finite = _mapping(guard.get("component_finite"), "V40 component finite flags")
    if set(finite) != {"broad", "answer", "side", "cross", "scene", "total"} or not all(
        value is True for value in finite.values()
    ):
        raise ValueError("V40 component finite flags changed")
    directional = _mapping(guard.get("directional_checks"), "V40 directional checks")
    failed = sorted(
        name
        for name, value in directional.items()
        if _mapping(value, f"V40 {name} direction").get(
            "strictly_positive_if_nonzero"
        )
        is not True
    )
    if failed != ["broad"]:
        raise ValueError(f"V40 failed directional set changed: {failed}")
    broad = _mapping(directional.get("broad"), "V40 broad direction")
    if (
        broad.get("nonzero") is not True
        or broad.get("dot_with_total") != -0.06791611851052101
        or broad.get("cosine_with_total") != -0.038372349473092954
        or broad.get("strictly_positive_if_nonzero") is not False
    ):
        raise ValueError("V40 broad-conflict evidence changed")
    for name in ("answer", "scene", "cross"):
        row = _mapping(directional.get(name), f"V40 {name} direction")
        if (
            row.get("nonzero") is not True
            or row.get("dot_finite") is not True
            or row.get("cosine_finite") is not True
            or row.get("dot_with_total", 0.0) <= 0.0
            or row.get("cosine_with_total", 0.0) <= 0.0
            or row.get("strictly_positive_if_nonzero") is not True
        ):
            raise ValueError(f"V40 {name} direction changed")
    return {
        "failure_sha256": _LOCKED_SHA256["guard_failure"],
        "failed_before_optimizer_step": 3,
        "optimizer_step_three_executed": False,
        "clip_applied_for_update_three": False,
        "checkpoint_written_for_update_three": False,
        "target_bit_exact_across_failed_attempt": True,
        "frozen_surface_bit_exact_across_failed_attempt": True,
        "only_failed_direction": "broad",
        "broad_dot_with_raw_total": broad["dot_with_total"],
        "broad_cosine_with_raw_total": broad["cosine_with_total"],
        "transient_pre_update3_target_sha256": (
            _TRANSIENT_PRE_UPDATE3_TARGET_SHA256
        ),
        "validation_loaded": False,
        "oracle_loaded": False,
    }, failure


def _validate_envelope(root: Path) -> dict[str, Any]:
    if root.is_symlink() or not root.is_dir():
        raise FileNotFoundError(f"V40 checkpoint root must be a real directory: {root}")
    entries = sorted(path.name for path in root.iterdir())
    expected = ["guard_failure_update_003.json", "update_000"]
    if entries != expected:
        raise ValueError(f"V40 stopped envelope changed: {entries}")
    update_directories = sorted(name for name in entries if _UPDATE_PATTERN.fullmatch(name))
    if update_directories != ["update_000"]:
        raise ValueError(f"V40 persisted update directories changed: {update_directories}")
    return {
        "root_entries": entries,
        "persisted_update_directories": update_directories,
        "only_update_zero_persisted": True,
        "update_one_and_two_were_transient_process_memory_only": True,
        "no_update_one_or_two_checkpoint_written_by_schedule": True,
        "no_update_three_or_later_checkpoint": True,
        "no_trained_optimizer_file_persisted": True,
        "process_exit_discarded_transient_updates_one_and_two": True,
    }


def _v41_authorization() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "authorization_id": "v41_cpu_f64_halfspace_projected_l14_lora_b",
        "authorized": True,
        "only_exact_successor_authorized": "v41_train_only_projected_gradient_continuation",
        "source": {
            "checkpoint": str(DEFAULT_CHECKPOINT_ROOT / "update_000"),
            "optimizer_step": 0,
            "adapter_file_sha256": _LOCKED_SHA256["update0_adapter"],
            "metadata_file_sha256": _LOCKED_SHA256["update0_metadata"],
            "runtime_metadata_file_sha256": _LOCKED_SHA256["update0_runtime"],
            "full_tensor_state_sha256": _FULL_STATE_SHA256,
            "target_lora_b_state_sha256": _SOURCE_TARGET_SHA256,
            "frozen_excluding_b_state_sha256": _FROZEN_SHA256,
            "source_is_exact_v40_and_v38_update_zero": True,
            "v40_transient_pre_update3_state_is_not_a_source_checkpoint": True,
            "source_optimizer_access_authorized": False,
        },
        "target_surface": {
            "parameter_names": [_TARGET_NAME],
            "parameter_shapes": [list(_TARGET_SHAPE)],
            "tensor_count": 1,
            "parameter_count": _TARGET_PARAMETER_COUNT,
            "existing_bank": "extension_v28_stage_b_query",
            "existing_adapter_index": 1,
            "language_layer": 14,
            "module_path": "model.language_model.layers.14.self_attn.q_proj",
            "lora_b_only": True,
            "lora_a_frozen_learned_basis": True,
            "all_other_parameters_and_buffers_frozen": True,
            "new_bank_or_module_authorized": False,
        },
        "weighted_objective_components": {
            "broad": 1.0,
            "answer": 0.5,
            "side": 8.0,
            "cross": 56.0,
            "side_hinge_margin": 0.5,
            "cross_prefix_margin": 0.10,
            "scene_formula": "side + cross",
            "raw_total_formula": "broad + answer + side + cross",
        },
        "projected_gradient_solver": {
            "authorization_revision": 3,
            "required_before_every_optimizer_step": True,
            "component_gradient_api": "torch.autograd.grad",
            "component_order": ["broad", "answer", "side", "cross"],
            "constraint_direction_order": ["broad", "answer", "scene", "cross"],
            "require_all_raw_components_finite": True,
            "nonfinite_component_action": "fail_stop_before_mutation",
            "constraint_activity_norm_floor_inclusive": 0.0,
            "active_constraint_rule": "finite_l2_norm_strictly_greater_than_zero",
            "active_constraint_direction_order": (
                "stable_subsequence_of_broad_answer_scene_cross_with_positive_norm"
            ),
            "inactive_constraint_direction_order": (
                "stable_subsequence_of_broad_answer_scene_cross_with_exact_zero_norm"
            ),
            "all_constraint_directions_may_be_zero_and_inactive": True,
            "minimum_active_constraint_count": 1,
            "zero_constraint_policy": (
                "record_inactive_and_first_order_satisfied_without_normalization"
            ),
            "zero_constraint_first_order_rationale": (
                "a zero gradient cannot be worsened to first order by the update"
            ),
            "standalone_side_is_not_a_constraint": True,
            "standalone_side_may_be_zero": True,
            "scene_may_be_zero_and_explicitly_inactive": True,
            "cross_may_be_zero_and_explicitly_inactive": True,
            "inactive_constraints_must_be_persisted_with_norm_and_reason": True,
            "raw_total_must_be_finite": True,
            "raw_total_norm_minimum_exclusive": 1e-12,
            "raw_total_zero_or_nonfinite_action": "fail_stop_before_mutation",
            "solver_device": "cpu",
            "solver_dtype": "torch.float64",
            "normalize_constraint_directions": True,
            "normalized_direction_formula": "u_i = component_i / l2(component_i)",
            "beta_formula": "max(1e-12, 1e-4 * l2(g_raw))",
            "beta_absolute_floor": 1e-12,
            "beta_raw_norm_multiplier": 1e-4,
            "optimization_problem": (
                "minimize 0.5*l2(d-g_raw)^2 subject to u_i dot d >= beta "
                "for every active nonzero constraint direction in canonical "
                "filtered order"
            ),
            "active_set_enumeration": {
                "maximum_constraint_count": 4,
                "active_constraint_count_allowed": [1, 2, 3, 4],
                "mask_count_formula": "2 ** active_constraint_count",
                "mask_count_allowed": [2, 4, 8, 16],
                "mask_order": "ascending_integer_over_canonical_active_direction_order",
                "independent_active_subsets_only": True,
                "rank_absolute_tolerance": 1e-12,
                "rank_relative_tolerance": 1e-10,
                "active_gram_solve": "torch.linalg.solve_cpu_float64",
                "dual_lambda_lower_tolerance": -1e-10,
                "kkt_absolute_tolerance": 1e-10,
                "kkt_relative_tolerance": 1e-8,
                "require_primal_feasibility": True,
                "require_dual_feasibility": True,
                "require_active_equality_feasibility": True,
                "require_stationarity": True,
                "require_complementarity": True,
                "candidate_objective": "0.5*l2(d-g_raw)^2",
                "selection": "minimum_objective_then_lowest_mask",
                "objective_tie_relative_tolerance": 1e-12,
                "no_feasible_candidate_action": "fail_stop_before_mutation",
            },
            "determinism": {
                "solve_twice_from_independent_cpu_float64_clones": True,
                "selected_mask_must_match": True,
                "lambdas_must_be_bit_exact": True,
                "projected_direction_must_be_bit_exact": True,
                "projected_direction_sha256_must_match": True,
            },
            "cpu_solution_safety": {
                "all_active_constraint_dots_at_least_beta": True,
                "all_active_constraint_dots_and_cosines_finite_and_strictly_positive": True,
                "inactive_exact_zero_directions_recorded_satisfied": True,
                "projected_to_raw_cosine_minimum": 0.95,
                "correction_ratio_formula": "l2(d-g_raw)/l2(g_raw)",
                "correction_ratio_maximum": 0.25,
            },
            "device_cast_safety": {
                "cast_only_after_cpu_solution_passes": True,
                "reattest_on_cpu_float64_after_target_device_dtype_roundtrip": True,
                "normalized_constraint_margin_minimum": "beta/2",
                "all_active_dots_and_cosines_finite_and_strictly_positive": True,
                "inactive_exact_zero_directions_remain_recorded_satisfied": True,
                "failure_action": "fail_stop_before_clip_or_optimizer_step",
            },
            "scalar_clip_safety": {
                "clip_norm": 1.0,
                "single_global_scalar_over_lora_b": True,
                "clip_scalar_must_be_finite_and_strictly_positive": True,
                "projected_to_clipped_cosine_minimum": 0.9999999,
                "all_active_component_dots_and_cosines_remain_finite_and_strictly_positive": True,
                "inactive_exact_zero_directions_remain_recorded_satisfied": True,
                "failure_action": "fail_stop_before_optimizer_step",
            },
            "persist_every_microstep": [
                "weighted_component_norms_and_hashes",
                "normalized_constraint_gram_matrix",
                "raw_total_norm_and_hash",
                "beta",
                "all_2**active_constraint_count_candidate_feasibility_records",
                "selected_mask_and_active_constraints",
                "selected_lambdas",
                "primal_dual_equality_stationarity_complementarity_residuals",
                "projection_objective_and_correction_ratio",
                "projected_raw_cosine",
                "double_solve_bit_exact_replay",
                "cpu_projected_direction_sha256",
                "post_device_cast_directional_attestation",
                "post_scalar_clip_directional_attestation",
                "target_and_frozen_hashes_before_and_after",
            ],
        },
        "transient_replay_gate": {
            "required_before_optimizer_step_three": True,
            "exact_target_hash_after_replayed_steps_one_and_two": (
                _TRANSIENT_PRE_UPDATE3_TARGET_SHA256
            ),
            "source_failure_artifact_sha256": _LOCKED_SHA256["guard_failure"],
            "fail_before_step_three_if_hash_differs": True,
            "proves_exact_replay_of_two_transient_v40_updates": True,
        },
        "optimizer": {
            "implementation": "torch.optim.SGD",
            "fresh": True,
            "learning_rate": 0.003,
            "momentum": 0.0,
            "dampening": 0.0,
            "weight_decay": 0.0,
            "nesterov": False,
            "foreach": False,
            "fused": False,
            "adam_or_adamw_authorized": False,
            "resume_only_self_hashed_v41_sgd_state": True,
        },
        "schedule": {
            "maximum_optimizer_step": 41,
            "saved_optimizer_steps": list(_SAVED_STEPS),
            "diagnostic_steps": list(_DIAGNOSTIC_STEPS),
            "pair_schedule_sha256": _PAIR_SCHEDULE_SHA256,
            "full_schedule_sha256": _FULL_SCHEDULE_SHA256,
            "true_microsteps": True,
            "unchanged_from_v38_v40": True,
        },
        "hard_train_only_gates": {
            "unchanged_from_v38_v40": True,
            **{name: dict(value) for name, value in _GATES.items()},
        },
        "stop_and_isolation": {
            "authorized_output_root": (
                "data_gemma4/checkpoints/"
                "gemma4_v41_diverse28_projected_gradient_l14_query"
            ),
            "v40_checkpoint_root_write_authorized": False,
            "stop_before_mutation_on_projection_or_attestation_failure": True,
            "stop_at_failed_update8_gate": True,
            "stop_at_failed_update16_gate": True,
            "stop_at_failed_update41_gate": True,
            "no_gate_relaxation_authorized": True,
            "new_terminal_seal_required_after_training": True,
        },
        "data_scope": {
            "exact_training_scene_ids": list(_TRAIN_SCENES),
            "training_qa_and_maps_only": True,
            "all_occupied_blocks_must_be_processed": True,
            "question_dependent_retrieval": False,
            "validation_access_authorized": False,
            "oracle_access_authorized": False,
            "final_test_access_authorized": False,
        },
        "selector_execution_authorized": False,
        "chat_or_runtime_promotion_authorized": False,
        "embodied_agent_promotion_authorized": False,
    }


def audit_v40_update3_failure(
    config_path: Path = DEFAULT_CONFIG,
    trainer_path: Path = DEFAULT_TRAINER,
    checkpoint_root: Path = DEFAULT_CHECKPOINT_ROOT,
) -> dict[str, Any]:
    config_path = _resolve(config_path)
    trainer_path = _resolve(trainer_path)
    root = _resolve(checkpoint_root)
    update0 = root / "update_000"
    failure_path = root / "guard_failure_update_003.json"
    v39_path = _resolve(_V39_TERMINAL)
    training_test = _resolve(_TRAINING_TEST)
    selector_test = _resolve(_SELECTOR_TEST)
    protected = _resolve(_PROTECTED)

    _locked_file(v39_path, _LOCKED_SHA256["v39_terminal"], "V39 terminal seal")
    config_audit = _validate_config(config_path)
    _locked_file(trainer_path, _LOCKED_SHA256["v40_trainer"], "V40 trainer")
    _locked_file(
        training_test, _LOCKED_SHA256["v40_training_test"], "V40 training tests"
    )
    _locked_file(
        selector_test, _LOCKED_SHA256["v40_selector_test"], "V40 selector tests"
    )
    envelope = _validate_envelope(root)
    source, source_files = _validate_update_zero(update0)
    failure, _failure_payload = _validate_failure(failure_path)
    _locked_file(protected, _LOCKED_SHA256["protected"], "protected V29 artifact")

    if source["target_lora_b_state_sha256"] == failure[
        "transient_pre_update3_target_sha256"
    ]:
        raise ValueError("V40 failure does not contain two transient updates")

    loaded = [
        v39_path,
        config_path,
        trainer_path,
        training_test,
        selector_test,
        *source_files,
        failure_path,
        protected,
    ]
    report = {
        "schema_version": 1,
        "artifact": "v40_update3_terminal_gate",
        "seal_revision": 3,
        "supersedes_revision1_sha256": (
            "9ce66c309adeb9e81636dc45cc1237e20cb7f050ed6ba9fe0cfa762c6c33660e"
        ),
        "supersedes_revision2_sha256": (
            "63b22d16303018d0482710a34c1a848d131a0e65a34dd96735fcfb241deba844"
        ),
        "revision2_change_scope": (
            "finite_exact_zero_constraint_gradients_are_explicitly_inactive"
        ),
        "revision3_change_scope": (
            "dynamic_active_constraint_candidate_inventory_wording_only"
        ),
        "passed": True,
        "v40_training_completed": False,
        "v40_failure_kind": "pre_step_raw_component_direction_conflict",
        "input_integrity": {
            "v39_terminal": {
                "path": _relative(v39_path),
                "sha256": _LOCKED_SHA256["v39_terminal"],
            },
            "v40_config": {
                "path": _relative(config_path),
                "sha256": _LOCKED_SHA256["v40_config"],
            },
            "v40_trainer": {
                "path": _relative(trainer_path),
                "sha256": _LOCKED_SHA256["v40_trainer"],
            },
            "v40_training_tests": {
                "path": _relative(training_test),
                "sha256": _LOCKED_SHA256["v40_training_test"],
            },
            "v40_selector_tests": {
                "path": _relative(selector_test),
                "sha256": _LOCKED_SHA256["v40_selector_test"],
            },
            "guard_failure": {
                "path": _relative(failure_path),
                "sha256": _LOCKED_SHA256["guard_failure"],
            },
            "protected_artifact": {
                "path": _relative(protected),
                "sha256": _LOCKED_SHA256["protected"],
                "access": "bytes_hashed_only",
                "unchanged": True,
            },
        },
        "config_replay": config_audit,
        "persisted_update_zero": source,
        "failure_replay": failure,
        "stopped_envelope": envelope,
        "execution_conclusion": {
            "optimizer_steps_one_and_two_executed_in_memory": True,
            "optimizer_step_three_executed": False,
            "optimizer_step_three_failed_before_clip": True,
            "optimizer_step_three_target_state_unchanged": True,
            "only_update_zero_checkpoint_persisted": True,
            "no_trained_checkpoint_persisted": True,
            "no_optimizer_file_persisted": True,
            "transient_updates_discarded_on_process_exit": True,
            "v41_must_restart_exact_update_zero": True,
        },
        "conditional_successor_authorization": _v41_authorization(),
        "v41_train_only_projected_gradient_continuation_authorized": True,
        "only_exact_successor_authorized": (
            "v41_train_only_projected_gradient_continuation"
        ),
        "arbitrary_training_authorized": False,
        "validation_access_authorized": False,
        "oracle_access_authorized": False,
        "final_test_access_authorized": False,
        "selector_execution_authorized": False,
        "chat_or_runtime_promotion_authorized": False,
        "terminal_process_access_audit": {
            "gemma_loaded": False,
            "qa_loaded": False,
            "maps_loaded": False,
            "validation_loaded": False,
            "oracle_loaded": False,
            "final_test_loaded": False,
            "optimizer_deserialized": False,
            "adapter_access": "bytes_hashed_only",
            "loaded_file_count": len(loaded),
            "loaded_file_inventory": sorted(_relative(path) for path in loaded),
        },
    }
    return json.loads(json.dumps(report, allow_nan=False))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--trainer", type=Path, default=DEFAULT_TRAINER)
    parser.add_argument("--checkpoint-root", type=Path, default=DEFAULT_CHECKPOINT_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    report = audit_v40_update3_failure(args.config, args.trainer, args.checkpoint_root)
    _atomic_json(_resolve(args.output), report)
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["audit_v40_update3_failure"]
