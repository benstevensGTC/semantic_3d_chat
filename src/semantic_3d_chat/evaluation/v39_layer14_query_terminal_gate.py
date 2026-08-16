"""Seal the V39 layer-14 gradient screen and authorize one exact V40 run.

This audit is deliberately training-free.  It authenticates and replays the
persisted V39 diagnostic, checks its no-write and train-only boundaries, and
solves a gradient compatibility interval from the persisted cosine/Gram
evidence.  It never imports the Gemma backend and never opens a checkpoint,
optimizer, QA file, scene map, validation input, oracle file, or final scene.

The resulting seal authorizes only the existing V28 layer-14 ``q_proj`` LoRA-B
tensor at the exact V38 update-zero checkpoint; LoRA-A remains a frozen learned
basis.  The cross-prefix objective is fixed at weight 56, momentum-free SGD
preserves the raw direction after scalar clipping, and every microstep must pass
a fresh component-gradient direction guard before mutation.  No evaluation or
promotion is authorized by this artifact.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from semantic_3d_chat.config import PROJECT_ROOT

DEFAULT_SCREEN_REPORT = Path(
    "reports/gemma4/metrics/v39_layer14_query_gradient_screen.json"
)
DEFAULT_V38_TERMINAL = Path("reports/gemma4/metrics/v38_update8_terminal_gate.json")
DEFAULT_OUTPUT = Path(
    "reports/gemma4/metrics/v39_layer14_query_terminal_gate.json"
)

_SCREEN_REPORT_SHA256 = (
    "d10ee48738864bb8e1b8136d7f16a2176e6e46e8adea5a9faa05bf974eb4bdbe"
)
_SCREEN_MODULE = Path(
    "src/semantic_3d_chat/evaluation/v39_layer14_query_gradient_screen.py"
)
_SCREEN_MODULE_SHA256 = (
    "49477f22908c7b9230a4f8f0862824b427967c48dfd181a7dca1a812ad365c5c"
)
_SCREEN_TEST = Path("tests/test_v39_layer14_query_gradient_screen.py")
_SCREEN_TEST_SHA256 = (
    "bff3612e5433c39cb5429f6e13934291fe8080c62a7934ef4cb89c41b2633f6b"
)
_V38_TERMINAL_SHA256 = (
    "1015949e802abccd562f7762cc01111818646527f3366aeaf01de3854bbe164a"
)
_PROTECTED_ARTIFACT = Path(
    "reports/gemma4/metrics/"
    "training_selection_gemma4_color_mirror_full_vocab_v11_resume36.json"
)
_PROTECTED_SHA256 = (
    "c0086f66edbb8854a7938e09c57535bfd47100adbaf3b3c95eeb4b08014ce2f8"
)

_SOURCE_CHECKPOINT = Path(
    "data_gemma4/checkpoints/gemma4_v38_diverse28_query_recovery/update_000"
)
_SOURCE_FILE_SHA256 = {
    "adapter.safetensors": (
        "b0bdceb7699e9d97467915c69186f433d3b0fac2b09144d38c3078afe1f70cb0"
    ),
    "metadata.json": (
        "9a4b03e8fd7f8a6ef50b6d85ae6c07c602f353ecfe104dae28efaa239da5a0ed"
    ),
    "runtime_metadata.json": (
        "7ec71195b6187524b903f8955af4db375b109c890fbbda9986f179b97dc58d30"
    ),
}
_SOURCE_FULL_STATE_SHA256 = (
    "7b951c6d7ae4f7b50603159f0bc4dfb4d50b5b40f9325134d78d1de1dae87fc0"
)
_TARGET_STATE_SHA256 = (
    "9ff9d535a094f96328483c46ff8c8ea5fca30edc35878492976c35f8674a9f87"
)
_V28_BANK_STATE_SHA256 = (
    "cc9dfa838bb87f32e2922d675658af4a1085d53a84ccdca6d5bacc6f7097217b"
)
_FROZEN_EXCLUDING_TARGET_SHA256 = (
    "7f33e541d36de33b10ceeac25e5f40374bffd1cf4b234af7a6b6341198b85360"
)

_TARGET_BANK = "extension_v28_stage_b_query"
_TARGET_NAMES = (
    "lora_banks.extension_v28_stage_b_query.adapters.1.lora_a",
    "lora_banks.extension_v28_stage_b_query.adapters.1.lora_b",
)
_TARGET_SHAPES = ((4, 1536), (4096, 4))
_TARGET_PARAMETER_COUNT = 22_528
_V40_TARGET_NAME = _TARGET_NAMES[1]
_V40_FROZEN_BASIS_NAME = _TARGET_NAMES[0]
_V40_TARGET_SHAPE = _TARGET_SHAPES[1]
_V40_TARGET_PARAMETER_COUNT = 16_384
_V40_LOCAL_B_STATE_SHA256 = (
    "d0834cc588ee2a9edf08aabedfd01e0a6d2b01c6b6ae7e3a3d764eaddf58cc3e"
)
_V40_LOCAL_A_STATE_SHA256 = (
    "9f0ee5f9bbb9ec07bd42aaca1e0817be567a11c396c693e6412e5f2b08f37403"
)
_V40_FULL_KEY_B_STATE_SHA256 = (
    "1cdda782f0caf121c743d36d8b122e9480aa8300453c03872da37dfe81556799"
)
_V40_FROZEN_EXCLUDING_B_SHA256 = (
    "cec01bc088bb87c6bb44e0659eb03aa766f951ddeee706ca9a70edaa080dea5e"
)
_V40_FROZEN_EXCLUDING_B_TENSOR_COUNT = 178
_V40_FROZEN_EXCLUDING_B_PARAMETER_COUNT = 13_969_292

_PROPOSED = "proposed_training_aggregate"
_CROSS = "cross_prefix_maintenance_aggregate"
_DIRECTIONS = (
    "book_support_aggregate",
    "picture_support_aggregate",
    "book_scene_discriminative_aggregate",
    "picture_scene_discriminative_aggregate",
    "broad_retention_aggregate",
    "scene_discriminative_aggregate",
    _CROSS,
)
_CAUSAL_DIRECTIONS = (
    "book_scene_discriminative_aggregate",
    "picture_scene_discriminative_aggregate",
    "broad_retention_aggregate",
    "scene_discriminative_aggregate",
    _CROSS,
)
_SURFACES: Mapping[str, str | None] = {
    "global": None,
    "lora_a": _TARGET_NAMES[0],
    "lora_b": _TARGET_NAMES[1],
}

_EXPECTED_FAILED_DIRECTION = (
    "proposed_training_aggregate__cross_prefix_maintenance_aggregate"
)
_EXPECTED_FAILED_COSINE = -0.30298788564923285
_EXPECTED_FAILED_DOT = -0.013504734244905985

_EXPECTED_INTERVALS = {
    "global": (4.834377059531332, 29.880385532751976),
    "lora_a": (12.566257500163564, 13.859227571889285),
    "lora_b": (3.545042356673292, 37.89459979731636),
    "joint": (12.566257500163564, 13.859227571889285),
}
_AUTHORIZED_T = 13.0
_SCREEN_CROSS_WEIGHT = 4.0
_AUTHORIZED_CROSS_WEIGHT = 56.0

_TRAIN_SCENES = tuple(
    f"scene_{index:06d}" for index in (*range(11, 19), *range(31, 39))
)
_VALIDATION_SCENES = tuple(f"scene_{index:06d}" for index in range(19, 25))
_PAIR_SCHEDULE_SHA256 = (
    "4e26160c9a36e20ac894ff5e26604e6fad985ff5c84211282fd9807c9e738b3f"
)
_FULL_SCHEDULE_SHA256 = (
    "2e7a2a8136c968739a8aaaa1138be87bec322fefa0fb340e6fb9bc6e07278bea"
)
_SAVED_STEPS = (0, 8, 16, 24, 32, 40, 41)
_DIAGNOSTIC_STEPS = (0, 8, 16, 41)

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


def _canonical_sha256(value: object) -> str:
    raw = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _real_file(path: Path, field: str) -> None:
    if path.is_symlink() or not path.is_file():
        raise FileNotFoundError(f"{field} must be a real file: {path}")


def _locked_file(path: Path, expected_sha256: str, field: str) -> str:
    _real_file(path, field)
    observed = _sha256(path)
    if observed != expected_sha256:
        raise ValueError(
            f"{field} bytes changed: expected {expected_sha256}, observed {observed}"
        )
    return observed


def _mapping(value: object, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{field} must be a mapping")
    return value


def _sequence(value: object, field: str) -> Sequence[Any]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise TypeError(f"{field} must be a sequence")
    return value


def _load_locked_json(path: Path, expected_sha256: str, field: str) -> Mapping[str, Any]:
    _locked_file(path, expected_sha256, field)
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    return _mapping(value, field)


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
        delete=False,
    ) as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
        temporary = Path(handle.name)
    os.replace(temporary, path)


def _float(value: object, field: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{field} is not finite")
    return result


def _close(observed: float, expected: float, field: str, tolerance: float = 1e-12) -> None:
    if not math.isclose(observed, expected, rel_tol=0.0, abs_tol=tolerance):
        raise ValueError(f"{field} changed: expected {expected}, observed {observed}")


def _validate_source_and_surface(report: Mapping[str, Any]) -> dict[str, Any]:
    terminal = _mapping(report.get("terminal"), "V39 terminal authorization")
    if terminal != {
        "path": str(DEFAULT_V38_TERMINAL),
        "sha256": _V38_TERMINAL_SHA256,
        "exact_revision_2_authorization_verified": True,
    }:
        raise ValueError("V39 no longer binds the exact revision-2 V38 seal")

    source = _mapping(report.get("source"), "V39 source")
    expected_source = {
        "source_checkpoint": str(_SOURCE_CHECKPOINT),
        "source_file_sha256": _SOURCE_FILE_SHA256,
        "source_full_tensor_state_sha256": _SOURCE_FULL_STATE_SHA256,
        "target_source_state_sha256": _TARGET_STATE_SHA256,
        "complete_v28_bank_state_sha256": _V28_BANK_STATE_SHA256,
        "frozen_excluding_target_state_sha256": _FROZEN_EXCLUDING_TARGET_SHA256,
        "source_optimizer_file_opened": False,
        "source_optimizer_state_loaded": False,
        "update8_checkpoint_opened": False,
    }
    if dict(source) != expected_source:
        raise ValueError("V39 exact update-zero source attestation changed")

    surface = _mapping(report.get("target_surface"), "V39 target surface")
    expected_surface = {
        "target_parameter_names": list(_TARGET_NAMES),
        "target_parameter_shapes": [list(shape) for shape in _TARGET_SHAPES],
        "trainable_tensor_count": 2,
        "trainable_parameter_count": _TARGET_PARAMETER_COUNT,
        "all_other_checkpoint_parameters_frozen": True,
        "all_other_gemma_parameters_frozen": True,
        "optimizer_constructed": False,
    }
    if dict(surface) != expected_surface:
        raise ValueError("V39 exact layer-14 query surface changed")

    architecture = _mapping(
        report.get("loaded_gemma_architecture"), "V39 Gemma architecture"
    )
    expected_architecture = {
        "language_layer_count": 35,
        "num_kv_shared_layers": 20,
        "first_shared_kv_layer": 15,
        "layer_13_attention_type": "sliding_attention",
        "layer_14_attention_type": "full_attention",
        "layer_13_role": "last_nonshared_sliding_kv_producer",
        "layer_14_role": "last_nonshared_full_kv_producer",
        "layers_15_through_34_reuse_shared_kv_states": True,
    }
    if dict(architecture) != expected_architecture:
        raise ValueError("V39 authenticated Gemma architecture changed")

    objective = _mapping(report.get("objective"), "V39 objective")
    expected_objective_values = {
        "answer_nll_weight": 0.5,
        "side_hinge_weight": 8.0,
        "side_hinge_margin": 0.5,
        "cross_prefix_hinge_weight": _SCREEN_CROSS_WEIGHT,
        "cross_prefix_margin": 0.1,
        "broad_retention_nll_weight": 1.0,
        "gradient_accumulation_across_objectives": False,
    }
    if any(objective.get(key) != value for key, value in expected_objective_values.items()):
        raise ValueError("V39 diagnostic objective changed")
    return {
        "terminal_exact": True,
        "source_exact": True,
        "target_surface_exact": True,
        "gemma_architecture_exact": True,
        "diagnostic_objective_exact": True,
    }


def _validate_pass_failure(report: Mapping[str, Any]) -> dict[str, Any]:
    evaluation = _mapping(
        report.get("pass_contract_evaluation"), "V39 pass-contract evaluation"
    )
    checks = _mapping(evaluation.get("checks"), "V39 pass-contract checks")
    false_checks = sorted(key for key, value in checks.items() if value is not True)
    if false_checks != ["all_predeclared_directional_compatibility_checks_passed"]:
        raise ValueError(f"V39 pass-contract failures changed: {false_checks}")

    directional = _mapping(
        evaluation.get("directional_compatibility"), "V39 directional compatibility"
    )
    failed = sorted(
        key for key, value in directional.items() if _mapping(value, key).get("passed") is not True
    )
    if failed != [_EXPECTED_FAILED_DIRECTION]:
        raise ValueError(f"V39 directional failure set changed: {failed}")
    failed_row = _mapping(directional[_EXPECTED_FAILED_DIRECTION], "failed direction")
    _close(
        _float(failed_row.get("cosine"), "failed cosine"),
        _EXPECTED_FAILED_COSINE,
        "failed cosine",
    )
    _close(
        _float(failed_row.get("dot_product"), "failed dot"),
        _EXPECTED_FAILED_DOT,
        "failed dot",
    )
    if failed_row.get("positive_dot") is not False or failed_row.get(
        "cosine_at_least_zero"
    ) is not False:
        raise ValueError("V39 failed cross-prefix direction flags changed")
    for key, value in directional.items():
        if key == _EXPECTED_FAILED_DIRECTION:
            continue
        row = _mapping(value, f"direction {key}")
        if (
            row.get("passed") is not True
            or row.get("positive_dot") is not True
            or row.get("cosine_at_least_zero") is not True
        ):
            raise ValueError(f"V39 formerly passing direction changed: {key}")
    if (
        evaluation.get("passed") is not False
        or report.get("passed") is not False
        or report.get("diagnostic_completed") is not True
        or evaluation.get("passing_this_screen_authorizes_training") is not False
        or evaluation.get("passing_this_screen_authorizes_runtime_promotion")
        is not False
        or report.get("training_authorized") is not False
        or report.get("runtime_promotion_authorized") is not False
    ):
        raise ValueError("V39 failure/authorization state changed")
    return {
        "diagnostic_completed": True,
        "diagnostic_passed": False,
        "false_contract_checks": false_checks,
        "failed_directional_pairs": failed,
        "failed_cross_prefix_cosine": _EXPECTED_FAILED_COSINE,
        "failed_cross_prefix_dot_product": _EXPECTED_FAILED_DOT,
        "all_other_predeclared_directions_passed": True,
        "screen_itself_authorized_no_training_or_promotion": True,
    }


def _validate_mutation_and_data_scope(report: Mapping[str, Any]) -> dict[str, Any]:
    mutation = _mapping(report.get("mutation_audit"), "V39 mutation audit")
    required_true = (
        "autograd_grad_used",
        "checkpoint_state_bit_exact",
        "gradients_cleared_between_objectives",
        "model_version_counters_unchanged",
        "temporary_requires_grad_surface_restored_to_frozen",
    )
    required_false = (
        "backward_called",
        "checkpoint_written",
        "gradients_accumulated_in_parameter_grad",
        "optimizer_constructed",
        "optimizer_state_opened",
        "optimizer_step_called",
        "parameter_or_buffer_write_performed",
    )
    if any(mutation.get(key) is not True for key in required_true) or any(
        mutation.get(key) is not False for key in required_false
    ):
        raise ValueError("V39 no-write mutation attestation changed")
    before = _mapping(mutation.get("checkpoint_state_before"), "state before")
    after = _mapping(mutation.get("checkpoint_state_after"), "state after")
    expected_state = {
        "full": _SOURCE_FULL_STATE_SHA256,
        "target": _TARGET_STATE_SHA256,
        "v28_bank": _V28_BANK_STATE_SHA256,
        "frozen_excluding_target": _FROZEN_EXCLUDING_TARGET_SHA256,
    }
    if dict(before) != expected_state or dict(after) != expected_state:
        raise ValueError("V39 checkpoint state hashes changed during the screen")
    measurements = _sequence(mutation.get("objective_measurements"), "measurements")
    if len(measurements) != 32:
        raise ValueError("V39 objective-measurement inventory changed")
    for index, value in enumerate(measurements):
        row = _mapping(value, f"measurement {index}")
        if (
            row.get("state_bit_exact") is not True
            or row.get("no_accumulated_gradients") is not True
            or row.get("target_and_frozen_hashes_before") != expected_state
            or row.get("target_and_frozen_hashes_after") != expected_state
            or _mapping(row.get("gradient"), f"measurement {index} gradient").get(
                "all_finite"
            )
            is not True
        ):
            raise ValueError(f"V39 measurement {index} integrity changed")

    declared = _mapping(report.get("declared_data_reads"), "declared data reads")
    if (
        declared.get("optimizer_file_reads") != []
        or declared.get("validation_or_final_environment_reads") != []
        or declared.get("observed_map_reads_exact") is not True
        or declared.get("observed_train_qa_jsonl_reads_exact") is not True
        or len(_sequence(declared.get("train_map_files"), "train map files")) != 16
        or len(_sequence(declared.get("train_qa_jsonl_files"), "train QA files")) != 1
    ):
        raise ValueError("V39 declared train-only data boundary changed")

    qa = _mapping(report.get("qa_audit"), "V39 QA audit")
    if (
        tuple(qa.get("train_scene_ids", ())) != _TRAIN_SCENES
        or tuple(qa.get("validation_scene_ids_from_pinned_contract", ()))
        != _VALIDATION_SCENES
        or qa.get("train_question_count") != 384
        or qa.get("train_changed_pair_unit_count") != 25
        or qa.get("validation_qa_loaded") is not False
        or qa.get("deferred_final_qa_loaded") is not False
        or qa.get("oracle_environment_files_loaded") is not False
    ):
        raise ValueError("V39 QA boundary changed")

    scene = _mapping(report.get("scene_input_audit"), "V39 scene-input audit")
    if (
        tuple(scene.get("train_scene_ids", ())) != _TRAIN_SCENES
        or scene.get("train_scene_count") != 16
        or scene.get("validation_scene_ids_loaded") != []
        or scene.get("deferred_final_scene_ids_loaded") != []
        or scene.get("question_dependent_retrieval") is not False
        or scene.get("all_occupied_blocks_processed") is not True
        or scene.get("all_voxels_covered") is not True
        or scene.get("scene_prefixes_built_before_questions") is not True
        or scene.get("scene_prefixes_question_independent_and_exact") is not True
        or scene.get("scene_prefix_sha256_before")
        != scene.get("scene_prefix_sha256_after")
    ):
        raise ValueError("V39 scene-input boundary changed")

    loaded = _sequence(report.get("loaded_file_inventory"), "loaded files")
    environment = _sequence(
        report.get("loaded_environment_file_inventory"), "environment files"
    )
    forbidden = (
        "/data/oracle/",
        "/oracle/",
        "/qa/validation.jsonl",
        "/qa/test.jsonl",
        "/optimizer.pt",
        "/update_008/",
        "scene_000025",
        "scene_000026",
        "scene_000027",
        "scene_000028",
        "scene_000029",
        "scene_000030",
    )
    bad = sorted(
        str(path)
        for path in loaded
        if any(fragment in str(path).casefold() for fragment in forbidden)
    )
    if (
        len(loaded) != 151
        or len(environment) != 24
        or bad
        or report.get("forbidden_file_access_count") != 0
        or report.get("forbidden_file_accesses") != []
        or report.get("validation_qa_loaded") is not False
        or report.get("oracle_loaded") is not False
        or report.get("final_test_scenes_touched") is not False
        or report.get("safety_attestations_passed_before_report_write") is not True
    ):
        raise ValueError("V39 loaded-file safety boundary changed")
    return {
        "checkpoint_state_bit_exact": True,
        "objective_measurement_count": len(measurements),
        "optimizer_constructed": False,
        "optimizer_opened": False,
        "optimizer_step_called": False,
        "checkpoint_or_parameter_write_performed": False,
        "training_scene_count": 16,
        "training_question_count": 384,
        "validation_qa_loaded": False,
        "validation_scene_maps_loaded": False,
        "oracle_loaded": False,
        "final_test_scenes_touched": False,
        "question_dependent_retrieval": False,
        "all_occupied_blocks_processed": True,
        "screen_loaded_file_inventory_count": len(loaded),
        "screen_loaded_environment_file_inventory_count": len(environment),
        "screen_loaded_file_inventory_sha256": _canonical_sha256(loaded),
        "screen_loaded_environment_file_inventory_sha256": _canonical_sha256(
            environment
        ),
    }


def _matrix(report: Mapping[str, Any], tensor_name: str | None) -> Mapping[str, Any]:
    if tensor_name is None:
        return _mapping(report.get("cosine_conflict_matrix"), "global cosine matrix")
    matrices = _mapping(
        report.get("per_tensor_cosine_conflict_matrices"), "per-tensor matrices"
    )
    return _mapping(matrices.get(tensor_name), f"matrix {tensor_name}")


def _norm(
    report: Mapping[str, Any], vector_name: str, tensor_name: str | None
) -> float:
    aggregates = _mapping(report.get("aggregate_gradients"), "aggregate gradients")
    vector = _mapping(aggregates.get(vector_name), f"aggregate {vector_name}")
    if vector.get("all_finite") is not True or vector.get(
        "all_target_tensors_nonzero"
    ) is not True:
        raise ValueError(f"V39 aggregate {vector_name} is not finite/nonzero")
    if tensor_name is None:
        return _float(vector.get("total_l2"), f"norm {vector_name}")
    per_tensor = _mapping(vector.get("per_tensor"), f"per tensor {vector_name}")
    row = _mapping(per_tensor.get(tensor_name), f"{vector_name}/{tensor_name}")
    if row.get("finite") is not True or row.get("nonzero") is not True:
        raise ValueError(f"V39 aggregate subvector {vector_name}/{tensor_name} invalid")
    return _float(row.get("l2"), f"norm {vector_name}/{tensor_name}")


def _cosine(
    report: Mapping[str, Any], first: str, second: str, tensor_name: str | None
) -> float:
    matrix = _matrix(report, tensor_name)
    names = list(_sequence(matrix.get("names"), "matrix names"))
    if len(names) != len(set(names)):
        raise ValueError("V39 cosine matrix names are not unique")
    try:
        first_index = names.index(first)
        second_index = names.index(second)
    except ValueError as error:
        raise ValueError(f"V39 cosine matrix lacks {first} or {second}") from error
    rows = _sequence(matrix.get("cosine"), "cosine rows")
    if len(rows) != len(names) or any(
        len(_sequence(row, "cosine row")) != len(names) for row in rows
    ):
        raise ValueError("V39 cosine matrix shape changed")
    forward = _float(
        _sequence(rows[first_index], "forward row")[second_index], "forward cosine"
    )
    reverse = _float(
        _sequence(rows[second_index], "reverse row")[first_index], "reverse cosine"
    )
    _close(forward, reverse, "cosine matrix symmetry")
    if forward < -1.000000000001 or forward > 1.000000000001:
        raise ValueError("V39 cosine lies outside [-1, 1]")
    return forward


def _dot(
    report: Mapping[str, Any], first: str, second: str, tensor_name: str | None
) -> float:
    return (
        _cosine(report, first, second, tensor_name)
        * _norm(report, first, tensor_name)
        * _norm(report, second, tensor_name)
    )


def _solve_surface(
    report: Mapping[str, Any], surface: str, tensor_name: str | None
) -> dict[str, Any]:
    base_norm = _norm(report, _PROPOSED, tensor_name)
    cross_norm = _norm(report, _CROSS, tensor_name)
    base_cross_dot = _dot(report, _PROPOSED, _CROSS, tensor_name)
    lower = -math.inf
    upper = math.inf
    constraints: dict[str, dict[str, Any]] = {}
    for direction in _CAUSAL_DIRECTIONS:
        base_dot = _dot(report, _PROPOSED, direction, tensor_name)
        cross_dot = _dot(report, _CROSS, direction, tensor_name)
        if cross_dot == 0.0:
            if base_dot <= 0.0:
                raise ValueError(f"V39 {surface}/{direction} has no feasible t")
            relation = "always_positive"
            root = None
        else:
            root = -base_dot / cross_dot
            if cross_dot > 0.0:
                lower = max(lower, root)
                relation = "t_strictly_greater_than_root"
            else:
                upper = min(upper, root)
                relation = "t_strictly_less_than_root"
        constraints[direction] = {
            "base_dot": base_dot,
            "cross_dot": cross_dot,
            "zero_crossing_t": root,
            "feasible_relation": relation,
        }
    if not math.isfinite(lower) or not math.isfinite(upper) or lower >= upper:
        raise ValueError(f"V39 {surface} has no bounded nonempty feasible interval")

    authorized_norm_squared = (
        base_norm * base_norm
        + 2.0 * _AUTHORIZED_T * base_cross_dot
        + _AUTHORIZED_T * _AUTHORIZED_T * cross_norm * cross_norm
    )
    if authorized_norm_squared <= 0.0:
        raise ValueError(f"V39 {surface} produces a zero authorized gradient")
    authorized_norm = math.sqrt(authorized_norm_squared)
    authorized: dict[str, dict[str, Any]] = {}
    for direction in _DIRECTIONS:
        direction_norm = _norm(report, direction, tensor_name)
        dot = _dot(report, _PROPOSED, direction, tensor_name) + _AUTHORIZED_T * _dot(
            report, _CROSS, direction, tensor_name
        )
        cosine = dot / (authorized_norm * direction_norm)
        authorized[direction] = {
            "dot_product": dot,
            "cosine": cosine,
            "positive_dot": dot > 0.0,
            "positive_cosine": cosine > 0.0,
            "passed": dot > 0.0 and cosine > 0.0,
        }
    if not all(row["passed"] for row in authorized.values()):
        raise ValueError(f"V39 {surface} does not support fixed t=13")
    return {
        "surface": surface,
        "tensor_name": tensor_name,
        "strict_feasible_interval": {"lower_exclusive": lower, "upper_exclusive": upper},
        "constraints": constraints,
        "authorized_t": _AUTHORIZED_T,
        "authorized_gradient_norm": authorized_norm,
        "authorized_directional_compatibility": authorized,
        "all_authorized_dots_and_cosines_strictly_positive": True,
    }


def solve_cross_preserving_interval(report: Mapping[str, Any]) -> dict[str, Any]:
    """Replay the global and per-tensor ``g_total + t*g_cross`` feasibility."""

    surfaces = {
        name: _solve_surface(report, name, tensor_name)
        for name, tensor_name in _SURFACES.items()
    }
    for name, row in surfaces.items():
        interval = _mapping(row["strict_feasible_interval"], f"{name} interval")
        expected_lower, expected_upper = _EXPECTED_INTERVALS[name]
        _close(
            _float(interval["lower_exclusive"], f"{name} lower"),
            expected_lower,
            f"{name} lower",
        )
        _close(
            _float(interval["upper_exclusive"], f"{name} upper"),
            expected_upper,
            f"{name} upper",
        )
    joint_lower = max(
        _float(row["strict_feasible_interval"]["lower_exclusive"], "joint lower")
        for row in surfaces.values()
    )
    joint_upper = min(
        _float(row["strict_feasible_interval"]["upper_exclusive"], "joint upper")
        for row in surfaces.values()
    )
    _close(joint_lower, _EXPECTED_INTERVALS["joint"][0], "joint lower")
    _close(joint_upper, _EXPECTED_INTERVALS["joint"][1], "joint upper")
    if not joint_lower < _AUTHORIZED_T < joint_upper:
        raise ValueError("V40 fixed t=13 is not strictly inside joint feasibility")

    lora_a = _mapping(surfaces["lora_a"], "LoRA-A surface")
    lora_a_cross = _mapping(
        _mapping(lora_a["constraints"], "LoRA-A constraints")[_CROSS],
        "LoRA-A cross constraint",
    )
    rejected_t = 6.0
    rejected_dot = _float(lora_a_cross["base_dot"], "LoRA-A base cross dot") + (
        rejected_t * _float(lora_a_cross["cross_dot"], "LoRA-A cross slope")
    )
    if rejected_dot >= 0.0:
        raise ValueError("The rejected t=6 control unexpectedly became compatible")
    return {
        "schema_version": 1,
        "formula": "g(t) = proposed_training_aggregate + t * cross_prefix_maintenance_aggregate",
        "persisted_cross_gradient_already_includes_weight": _SCREEN_CROSS_WEIGHT,
        "effective_cross_weight_formula": "4 * (1 + t)",
        "directions_used_for_interval": list(_CAUSAL_DIRECTIONS),
        "directions_replayed_at_authorized_t": list(_DIRECTIONS),
        "surfaces": surfaces,
        "global_strict_feasible_interval": surfaces["global"][
            "strict_feasible_interval"
        ],
        "joint_global_and_per_tensor_strict_feasible_interval": {
            "lower_exclusive": joint_lower,
            "upper_exclusive": joint_upper,
        },
        "rejected_t6_control": {
            "t": rejected_t,
            "effective_cross_weight": 28.0,
            "global_interval_contains_t": True,
            "lora_a_interval_contains_t": False,
            "lora_a_cross_prefix_dot_product": rejected_dot,
            "rejected": True,
            "reason": "LoRA-A cross-prefix subvector remains directionally negative",
        },
        "selected_t": _AUTHORIZED_T,
        "selected_effective_cross_weight": _AUTHORIZED_CROSS_WEIGHT,
        "selected_t_strictly_inside_joint_interval": True,
        "global_and_both_tensor_subvectors_strictly_compatible": True,
        "raw_gradient_feasibility_does_not_certify_adam_preconditioning": True,
        "adam_preconditioning_not_certified_by_raw_gradient": True,
        "adam_rejected_for_v40": True,
        "direction_preserving_sgd_selected": True,
        "per_tensor_raw_gradient_subvectors_checked": True,
        "cross_hinge_can_shut_off_when_margin_is_satisfied": True,
        "passed": True,
    }


def _v40_authorization() -> dict[str, Any]:
    return {
        "authorization_schema_version": 1,
        "authorization_id": "v40_cross_preserving_l14_q_t13",
        "authorized": True,
        "only_exact_successor_authorized": "v40_cross_preserving_layer14_query_training",
        "source": {
            "checkpoint": str(_SOURCE_CHECKPOINT),
            "optimizer_step": 0,
            "adapter_file_sha256": _SOURCE_FILE_SHA256["adapter.safetensors"],
            "metadata_file_sha256": _SOURCE_FILE_SHA256["metadata.json"],
            "runtime_metadata_file_sha256": _SOURCE_FILE_SHA256[
                "runtime_metadata.json"
            ],
            "full_tensor_state_sha256": _SOURCE_FULL_STATE_SHA256,
            "target_state_sha256": _TARGET_STATE_SHA256,
            "complete_v28_bank_state_sha256": _V28_BANK_STATE_SHA256,
            "frozen_excluding_target_state_sha256": (
                _FROZEN_EXCLUDING_TARGET_SHA256
            ),
            "source_optimizer_access_authorized": False,
            "update8_checkpoint_access_authorized": False,
        },
        "target_surface": {
            "existing_bank": _TARGET_BANK,
            "existing_adapter_index": 1,
            "language_layer": 14,
            "module_path": "model.language_model.layers.14.self_attn.q_proj",
            "parameter_names": [_V40_TARGET_NAME],
            "parameter_shapes": [list(_V40_TARGET_SHAPE)],
            "tensor_count": 1,
            "parameter_count": _V40_TARGET_PARAMETER_COUNT,
            "rank": 4,
            "alpha": 8.0,
            "dropout": 0.0,
            "lora_b_only": True,
            "lora_a_learned_basis_frozen": True,
            "frozen_lora_a_name": _V40_FROZEN_BASIS_NAME,
            "frozen_lora_a_shape": list(_TARGET_SHAPES[0]),
            "lora_a_write_authorized": False,
            "source_a_plus_b_state_sha256": _TARGET_STATE_SHA256,
            "source_local_b_state_sha256": _V40_LOCAL_B_STATE_SHA256,
            "source_local_a_state_sha256": _V40_LOCAL_A_STATE_SHA256,
            "source_full_checkpoint_key_b_state_sha256": (
                _V40_FULL_KEY_B_STATE_SHA256
            ),
            "source_frozen_excluding_b_state_sha256": (
                _V40_FROZEN_EXCLUDING_B_SHA256
            ),
            "source_frozen_excluding_b_tensor_count": (
                _V40_FROZEN_EXCLUDING_B_TENSOR_COUNT
            ),
            "source_frozen_excluding_b_parameter_count": (
                _V40_FROZEN_EXCLUDING_B_PARAMETER_COUNT
            ),
            "source_b_only_hash_must_be_verified_before_first_step": True,
            "frozen_excluding_b_hash_must_be_verified_before_first_step": True,
            "pre_step_self_attested_hashes_must_remain_exact_on_resume": True,
            "new_lora_bank_authorized": False,
            "all_other_parameters_and_buffers_frozen": True,
        },
        "objective": {
            "formula": (
                "1.0*broad_nll + 0.5*correct_answer_nll + "
                "8.0*side_hinge + 56.0*cross_prefix_hinge"
            ),
            "broad_nll_weight": 1.0,
            "correct_answer_nll_weight": 0.5,
            "side_hinge_weight": 8.0,
            "side_hinge_margin": 0.5,
            "cross_prefix_hinge_weight": _AUTHORIZED_CROSS_WEIGHT,
            "cross_prefix_margin": 0.10,
            "gradient_construction_t": _AUTHORIZED_T,
            "gradient_formula": "g_v40 = g_v39_total + 13 * g_v39_cross_weight4",
            "cross_prefix_weight_28_authorized": False,
        },
        "optimizer": {
            "type": "SGD",
            "implementation": "torch.optim.SGD",
            "fresh": True,
            "learning_rate": 0.003,
            "momentum": 0.0,
            "dampening": 0.0,
            "nesterov": False,
            "weight_decay": 0.0,
            "gradient_clip_norm": 1.0,
            "clip_is_one_global_scalar_over_the_single_b_tensor": True,
            "direction_preserving_after_scalar_clip": True,
            "adam_or_adamw_authorized": False,
            "source_optimizer_state_loaded": False,
            "source_optimizer_file_opened": False,
            "trainable_parameter_count": _V40_TARGET_PARAMETER_COUNT,
            "resume_only_self_hashed_v40_sgd_state": True,
            "resume_optimizer_type_must_remain_exact": True,
        },
        "per_microstep_raw_gradient_guard": {
            "required_before_every_optimizer_step": True,
            "target_parameter_names": [_V40_TARGET_NAME],
            "component_gradient_api": "torch.autograd.grad",
            "backward_for_component_construction_authorized": False,
            "weighted_components": {
                "broad": 1.0,
                "answer": 0.5,
                "side": 8.0,
                "cross": 56.0,
            },
            "total_gradient_formula": "broad + answer + side + cross",
            "scene_gradient_formula": "side + cross",
            "require_total_gradient_finite_and_nonzero": True,
            "directions_checked_if_nonzero": [
                "broad",
                "answer",
                "scene",
                "cross",
            ],
            "require_each_component_gradient_finite": True,
            "require_strictly_positive_total_direction_dot": True,
            "require_strictly_positive_total_direction_cosine": True,
            "fail_stop_before_clip_or_step_on_any_guard_failure": True,
            "assign_exact_summed_raw_gradient_to_parameter_grad": True,
            "scalar_global_clip_only_after_guard_passes": True,
            "momentum_free_sgd_step_only_after_guard_passes": True,
            "persist_every_microstep": [
                "component_norms",
                "total_norm_before_clip",
                "clip_scalar",
                "total_vs_broad_dot_and_cosine",
                "total_vs_answer_dot_and_cosine",
                "total_vs_scene_dot_and_cosine",
                "total_vs_cross_dot_and_cosine",
                "guard_passed",
                "target_hash_before_and_after",
                "frozen_excluding_b_hash_before_and_after",
            ],
            "aggregate_v39_evidence_alone_is_not_a_microstep_pass": True,
        },
        "schedule": {
            "exact_v38_schedule": True,
            "maximum_optimizer_step": 41,
            "saved_optimizer_steps": list(_SAVED_STEPS),
            "per_unit_nll_diagnostics_required_at_steps": list(_DIAGNOSTIC_STEPS),
            "pair_schedule_sha256": _PAIR_SCHEDULE_SHA256,
            "full_schedule_sha256": _FULL_SCHEDULE_SHA256,
            "true_microsteps": True,
        },
        "hard_train_only_gates": {
            "unchanged_from_exact_v38": True,
            **{name: dict(value) for name, value in _GATES.items()},
        },
        "stop_protocol": {
            "evaluate_update8_before_any_update9": True,
            "stop_at_update8_if_gate_fails": True,
            "evaluate_update16_before_any_update17": True,
            "stop_at_update16_if_gate_fails": True,
            "stop_at_update41_if_gate_fails": True,
            "no_gate_relaxation_authorized": True,
            "no_training_past_optimizer_step_41": True,
            "failed_gate_checkpoint_may_be_sealed_but_not_promoted": True,
            "new_terminal_seal_required_after_training": True,
        },
        "data_and_scene_scope": {
            "exact_training_scene_ids": list(_TRAIN_SCENES),
            "exact_training_scene_count": 16,
            "training_qa_only": True,
            "validation_qa_loaded_during_training": False,
            "validation_scene_maps_loaded_during_training": False,
            "oracle_loaded_during_training": False,
            "final_test_loaded_during_training": False,
            "all_occupied_blocks_must_be_processed": True,
            "scene_prefixes_built_before_questions": True,
            "scene_prefixes_question_independent": True,
            "question_dependent_retrieval": False,
        },
        "authorized_output_root": (
            "data_gemma4/checkpoints/gemma4_v40_diverse28_"
            "cross_preserving_l14_query"
        ),
        "validation_access_authorized": False,
        "final_test_access_authorized": False,
        "oracle_access_authorized": False,
        "chat_or_runtime_promotion_authorized": False,
        "selector_execution_authorized": False,
        "embodied_agent_promotion_authorized": False,
    }


def audit_v39_layer14_query_screen(
    screen_report: Path = DEFAULT_SCREEN_REPORT,
    v38_terminal: Path = DEFAULT_V38_TERMINAL,
) -> dict[str, Any]:
    """Authenticate V39 and return its immutable terminal authorization."""

    screen_path = _resolve(screen_report)
    v38_path = _resolve(v38_terminal)
    module_path = _resolve(_SCREEN_MODULE)
    test_path = _resolve(_SCREEN_TEST)
    protected_path = _resolve(_PROTECTED_ARTIFACT)

    screen = _load_locked_json(
        screen_path, _SCREEN_REPORT_SHA256, "V39 gradient-screen report"
    )
    _locked_file(module_path, _SCREEN_MODULE_SHA256, "V39 gradient-screen module")
    _locked_file(test_path, _SCREEN_TEST_SHA256, "V39 gradient-screen tests")
    _locked_file(v38_path, _V38_TERMINAL_SHA256, "V38 revision-2 terminal seal")
    _locked_file(protected_path, _PROTECTED_SHA256, "protected V29 selection artifact")

    report_identity = (
        screen.get("schema_version"),
        screen.get("artifact"),
    )
    if report_identity != (
        1,
        "v39_v28_layer14_query_gradient_cosine_screen",
    ):
        raise ValueError(f"V39 report identity changed: {report_identity}")

    source_surface = _validate_source_and_surface(screen)
    failure = _validate_pass_failure(screen)
    scope = _validate_mutation_and_data_scope(screen)
    math_replay = solve_cross_preserving_interval(screen)
    authorization = _v40_authorization()

    loaded = (screen_path, module_path, test_path, v38_path, protected_path)
    report = {
        "schema_version": 1,
        "artifact": "v39_layer14_query_terminal_gate",
        "seal_revision": 1,
        "passed": True,
        "audit_method": (
            "exact_report_module_test_v38_seal_and_protected_hashes_plus_"
            "persisted_contract_mutation_file_scope_and_gram_replay_only"
        ),
        "v39_diagnostic_completed": True,
        "v39_diagnostic_passed": False,
        "v39_training_or_promotion_authorized_by_screen": False,
        "input_integrity": {
            "screen_report": {
                "path": _relative(screen_path),
                "sha256": _SCREEN_REPORT_SHA256,
            },
            "screen_module": {
                "path": _relative(module_path),
                "sha256": _SCREEN_MODULE_SHA256,
            },
            "screen_tests": {
                "path": _relative(test_path),
                "sha256": _SCREEN_TEST_SHA256,
            },
            "v38_terminal_seal": {
                "path": _relative(v38_path),
                "sha256": _V38_TERMINAL_SHA256,
            },
            "protected_artifact": {
                "path": _relative(protected_path),
                "sha256": _PROTECTED_SHA256,
                "access": "bytes_hashed_only",
                "unchanged": True,
            },
        },
        "source_and_surface_replay": source_surface,
        "single_failure_replay": failure,
        "mutation_and_data_scope_replay": scope,
        "cross_preserving_gradient_math": math_replay,
        "conditional_successor_authorization": authorization,
        "v40_cross_preserving_layer14_query_training_authorized": True,
        "only_exact_successor_authorized": (
            "v40_cross_preserving_layer14_query_training"
        ),
        "arbitrary_training_authorized": False,
        "validation_access_authorized": False,
        "final_test_access_authorized": False,
        "oracle_access_authorized": False,
        "chat_or_runtime_promotion_authorized": False,
        "selector_execution_authorized": False,
        "terminal_process_access_audit": {
            "gemma_loaded": False,
            "checkpoint_tensor_or_metadata_loaded": False,
            "optimizer_opened": False,
            "qa_loaded": False,
            "scene_maps_loaded": False,
            "validation_loaded": False,
            "oracle_loaded": False,
            "final_test_loaded": False,
            "loaded_file_count": len(loaded),
            "loaded_file_inventory": sorted(_relative(path) for path in loaded),
        },
    }
    return json.loads(json.dumps(report, allow_nan=False))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--screen-report", type=Path, default=DEFAULT_SCREEN_REPORT)
    parser.add_argument("--v38-terminal", type=Path, default=DEFAULT_V38_TERMINAL)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    report = audit_v39_layer14_query_screen(args.screen_report, args.v38_terminal)
    _atomic_json(_resolve(args.output), report)
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["audit_v39_layer14_query_screen", "solve_cross_preserving_interval"]
