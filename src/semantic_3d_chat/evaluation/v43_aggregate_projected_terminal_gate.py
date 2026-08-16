"""Seal the negative V43 screen and authorize one bounded V44 pilot."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from semantic_3d_chat.config import PROJECT_ROOT

DEFAULT_SCREEN = Path(
    "reports/gemma4/metrics/v43_aggregate_projected_no_step_diagnostic.json"
)
DEFAULT_OUTPUT = Path(
    "reports/gemma4/metrics/v43_aggregate_projected_screen_terminal_gate.json"
)
V43_SOURCE = Path(
    "src/semantic_3d_chat/evaluation/v43_aggregate_projected_screen.py"
)
V43_TEST = Path("tests/test_v43_aggregate_projected_screen.py")
V42_TERMINAL = Path("reports/gemma4/metrics/v42_delta_line_terminal_gate.json")
PROTECTED = Path(
    "reports/gemma4/metrics/"
    "training_selection_gemma4_color_mirror_full_vocab_v11_resume36.json"
)
V44_CONFIG = Path(
    "configs/experiments/gemma4_diverse28_joint_scene_readout_v44.yaml"
)
V44_OUTPUT = Path(
    "data_gemma4/checkpoints/gemma4_v44_joint_scene_readout_l14_query"
)
V44_SOURCE = Path(
    "data_gemma4/checkpoints/"
    "gemma4_v41_retry1_diverse28_projected_gradient_l14_query/update_000"
)

_PINS = {
    str(DEFAULT_SCREEN): "31c8e1958a7449f48e36524b2161771005d9fc5eba82819f0a3cd99f17a41e6d",
    str(V43_SOURCE): "239083b2143976eeafce6117ce7772d34df2ebdc52e2bb1495c658686d00b196",
    str(V43_TEST): "1c02e2474c2168542a275b7d62e5ff37173579da5be0b1aa54bf7e6748a4559b",
    str(V42_TERMINAL): "1f4f73c782813fd47d1ea8fd659df3545dffe8143bbcacc0d47c9d40baea59e8",
    str(PROTECTED): "c0086f66edbb8854a7938e09c57535bfd47100adbaf3b3c95eeb4b08014ce2f8",
}
_SOURCE_FILES = {
    "adapter.safetensors": "b0bdceb7699e9d97467915c69186f433d3b0fac2b09144d38c3078afe1f70cb0",
    "metadata.json": "331cda3f2ebc1539e8ee27ebbae398be5e19f3fd77d0aa20dde635d569e29d6d",
    "runtime_metadata.json": "690e790b612e0b75323c1f27f7e9afe87243ccc1564c8cc690e86a442cffbfcd",
}
_STEPS = [-0.008, -0.004, 0.0, 0.002, 0.004, 0.008, 0.012, 0.016]
_TRAIN_SCENES = [
    *(f"scene_{index:06d}" for index in range(11, 19)),
    *(f"scene_{index:06d}" for index in range(31, 39)),
]
_TARGET_U0 = "d0834cc588ee2a9edf08aabedfd01e0a6d2b01c6b6ae7e3a3d764eaddf58cc3e"
_FULL_U0 = "7b951c6d7ae4f7b50603159f0bc4dfb4d50b5b40f9325134d78d1de1dae87fc0"
_FROZEN_V41 = "cec01bc088bb87c6bb44e0659eb03aa766f951ddeee706ca9a70edaa080dea5e"
_FROZEN_V44 = "31cb215cf0d1623886d4a79203c501912e66537021da53dd711124abdc2e36fa"
_AUTHORIZED_SOURCE = "b935c7e6ccceb1068f80e679b4159c6ca756f9f81868b954b93ac683e014f5a0"


def _resolve(path: str | Path) -> Path:
    value = Path(path)
    return value.resolve() if value.is_absolute() else (PROJECT_ROOT / value).resolve()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _mapping(value: object, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{field} must be a mapping")
    return value


def _authenticate_inputs() -> dict[str, str]:
    observed: dict[str, str] = {}
    for relative, expected in _PINS.items():
        path = _resolve(relative)
        if path.is_symlink() or not path.is_file() or _sha256(path) != expected:
            raise ValueError(f"V43 terminal input changed or is aliased: {relative}")
        observed[relative] = expected
    source = _resolve(V44_SOURCE)
    if source.is_symlink() or not source.is_dir():
        raise ValueError("V44 source checkpoint changed or is aliased")
    if sorted(path.name for path in source.iterdir()) != sorted(_SOURCE_FILES):
        raise ValueError("V44 source checkpoint inventory changed")
    for name, expected in _SOURCE_FILES.items():
        path = source / name
        if path.is_symlink() or not path.is_file() or _sha256(path) != expected:
            raise ValueError(f"V44 source checkpoint file changed: {name}")
        observed[str(V44_SOURCE / name)] = expected
    return observed


def _validate_negative_screen(screen: Mapping[str, Any]) -> None:
    candidates = screen.get("candidate_results")
    restorations = screen.get("restoration_audit")
    endpoint = _mapping(
        screen.get("update_zero_endpoint_replay"), "V43 endpoint replay"
    )
    final_state = _mapping(screen.get("final_state"), "V43 final state")
    gradient = _mapping(screen.get("gradient_inventory"), "V43 gradient")
    gradient_source = _mapping(
        gradient.get("source_state_after_gradient_measurement"),
        "V43 gradient source state",
    )
    inventory = _mapping(screen.get("candidate_inventory"), "V43 inventory")
    cache = _mapping(screen.get("cache_boundary"), "V43 cache boundary")
    qa = _mapping(screen.get("qa_audit"), "V43 QA audit")
    if (
        screen.get("artifact") != "v43_aggregate_projected_no_step_diagnostic"
        or screen.get("screen_integrity_passed") is not True
        or screen.get("teacher_eligible_candidate_found") is not False
        or screen.get("selected_scalar_step") is not None
        or screen.get("selected_candidate") is not None
        or screen.get("selected_target_sha256") is not None
        or screen.get("selected_candidate_replay") is not None
        or screen.get("optional_selected_candidate_greedy_audit") is not None
        or screen.get("candidate_checkpoint_written") is not False
        or screen.get("optimizer_constructed_or_loaded") is not False
        or screen.get("selector_execution_authorized") is not False
        or screen.get("training_authorized") is not False
        or screen.get("runtime_promotion_authorized") is not False
        or screen.get("validation_qa_loaded") is not False
        or screen.get("oracle_loaded") is not False
        or screen.get("final_test_scenes_touched") is not False
        or screen.get("forbidden_file_accesses") != []
        or not isinstance(candidates, list)
        or len(candidates) != len(_STEPS)
        or [row.get("scalar_step") for row in candidates] != _STEPS
        or any(row.get("teacher_eligible") is not False for row in candidates)
        or not isinstance(restorations, list)
        or len(restorations) != len(_STEPS)
        or [row.get("scalar_step") for row in restorations] != _STEPS
        or any(row.get("passed") is not True for row in restorations)
        or endpoint.get("passed") is not True
        or not all(endpoint.get(name) is True for name in ("pair_metrics", "per_unit_nll", "broad_nll"))
        or final_state.get("restored_exact") is not True
        or final_state.get("target_state_sha256") != _TARGET_U0
        or final_state.get("full_state_sha256") != _FULL_U0
        or final_state.get("frozen_state_sha256") != _FROZEN_V41
        or final_state.get("all_gradients_absent") is not True
        or final_state.get("all_requires_grad_false") is not True
        or gradient_source.get("passed") is not True
        or gradient_source.get("target_state_sha256") != _TARGET_U0
        or gradient_source.get("full_state_sha256") != _FULL_U0
        or gradient_source.get("frozen_state_sha256") != _FROZEN_V41
        or gradient_source.get("all_gradients_absent") is not True
        or inventory.get("fixed_scalar_steps") != _STEPS
        or inventory.get("candidate_hashes_fixed_before_forward_evaluation") is not True
        or cache.get("exact_train_scene_ids") != _TRAIN_SCENES
        or cache.get("exact_train_scene_count") != 16
        or cache.get("validation_environment_maps_loaded") is not False
        or cache.get("oracle_environment_files_loaded") is not False
        or qa.get("train_scene_ids") != _TRAIN_SCENES
        or qa.get("train_question_count") != 384
        or qa.get("train_changed_pair_unit_count") != 25
        or qa.get("validation_qa_loaded") is not False
        or qa.get("deferred_final_qa_loaded") is not False
        or qa.get("oracle_environment_files_loaded") is not False
    ):
        raise ValueError("V43 negative-screen evidence changed")


def _authorization() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "authorization_id": "v44_joint_scene_readout_train_only_pilot",
        "authorized": True,
        "only_exact_action": "one_bounded_v44_joint_scene_readout_training_pilot",
        "authorized_config": str(V44_CONFIG),
        "authorized_output_root": str(V44_OUTPUT),
        "source_checkpoint": str(V44_SOURCE),
        "source_file_sha256": dict(_SOURCE_FILES),
        "source_full_tensor_state_sha256": _FULL_U0,
        "source_authorized_surface_state_sha256": _AUTHORIZED_SOURCE,
        "frozen_excluding_authorized_state_sha256": _FROZEN_V44,
        "trainable_surface": {
            "parameter_names": [
                "block_cross_residual.w_o",
                "lora_banks.extension_v28_stage_b_query.adapters.1.lora_a",
                "lora_banks.extension_v28_stage_b_query.adapters.1.lora_b",
            ],
            "parameter_shapes": [[256, 1536], [4, 1536], [4096, 4]],
            "scene_readout_parameter_count": 393_216,
            "query_parameter_count": 22_528,
            "total_parameter_count": 415_744,
            "block_qkv_frozen": True,
            "gemma_base_and_all_other_lora_banks_frozen": True,
        },
        "optimizer": {
            "implementation": "fresh_torch_adamw_two_groups",
            "source_optimizer_loaded": False,
            "scene_readout_learning_rate": 2.5e-5,
            "query_learning_rate": 2.0e-5,
            "weight_decay": 0.0,
            "foreach": False,
            "fused": False,
            "per_group_gradient_clip_norm": 1.0,
        },
        "objective": {
            "broad_nll_weight": 0.25,
            "pair_correct_nll_weight": 0.5,
            "side_hinge_weight": 8.0,
            "cross_prefix_flip_weight": 8.0,
            "side_hinge_margin": 0.5,
            "cross_prefix_flip_margin": 0.1,
            "source_prefix_trust_weight": 0.001,
            "source_prefix_trust_scale": 0.05,
        },
        "schedule": {
            "maximum_optimizer_updates": 16,
            "checkpoint_steps": [0, 4, 8, 16],
            "update8_must_pass_before_updates_9_through_16": True,
            "update4_is_diagnostic_only": True,
            "true_optimizer_step_per_schedule_row": True,
        },
        "update8_gate": {
            "priority_side_deficit_minimum_improvement": 0.5,
            "complete_units_minimum": 9,
            "positive_sides_minimum": 34,
            "cross_prefix_complete_units_minimum": 17,
            "broad_nll_maximum_increase": 0.02,
            "both_authorized_parameter_groups_must_change": True,
            "frozen_state_must_remain_exact": True,
        },
        "update16_gate": {
            "require_update8_passed": True,
            "priority_side_deficit_minimum_improvement": 0.5,
            "complete_units_minimum": 10,
            "positive_sides_minimum": 35,
            "cross_prefix_complete_units_minimum": 17,
            "complete_physical_pair_coverage_minimum": 5,
            "book_or_picture_complete_units_minimum": 1,
            "greedy_complete_units_minimum": 5,
            "broad_greedy_exact_correct_minimum": 23,
            "broad_nll_maximum_increase": 0.02,
        },
        "scope": {
            "training_qa_and_maps_only": True,
            "all_occupied_blocks_processed": True,
            "question_dependent_scene_processing": False,
            "question_dependent_retrieval": False,
            "validation_access_authorized": False,
            "oracle_access_authorized": False,
            "final_test_access_authorized": False,
            "selector_execution_authorized": False,
            "runtime_promotion_authorized": False,
            "new_terminal_seal_required_after_training": True,
        },
    }


def build_report() -> dict[str, Any]:
    observed = _authenticate_inputs()
    screen = json.loads(_resolve(DEFAULT_SCREEN).read_text(encoding="utf-8"))
    _validate_negative_screen(screen)
    return {
        "schema_version": 1,
        "artifact": "v43_aggregate_projected_screen_terminal_gate",
        "passed": True,
        "screen_sha256": _PINS[str(DEFAULT_SCREEN)],
        "input_sha256": observed,
        "negative_result": {
            "fixed_scalar_steps": list(_STEPS),
            "candidate_count": len(_STEPS),
            "teacher_eligible_candidate_count": 0,
            "update_zero_endpoint_replay_exact": True,
            "all_candidates_restored_exact_u0": True,
            "gradient_measurement_left_source_exact": True,
            "no_optimizer_checkpoint_selector_or_restricted_access": True,
        },
        "conditional_successor_authorization": _authorization(),
        "only_exact_successor_authorized": "v44_joint_scene_readout_train_only_pilot",
        "v44_train_only_pilot_authorized": True,
        "validation_access_authorized": False,
        "selector_execution_authorized": False,
        "runtime_promotion_authorized": False,
    }


def write_report(output: str | Path = DEFAULT_OUTPUT) -> dict[str, Any]:
    path = _resolve(output)
    authorized = _resolve(DEFAULT_OUTPUT)
    if path != authorized:
        raise ValueError(f"V43 terminal output is pinned to {authorized}")
    if path.is_symlink() or path.exists():
        raise FileExistsError(f"V43 terminal is one-shot and will not overwrite {path}")
    report = build_report()
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    os.close(descriptor)
    temporary_path = Path(temporary)
    try:
        temporary_path.write_text(
            json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    print(json.dumps(write_report(args.output), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = ["build_report", "write_report"]
