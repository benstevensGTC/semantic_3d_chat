from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from semantic_3d_chat.evaluation.v35_terminal_gate import (
    _validate_checkpoint_sequence,
    audit_v35_update32,
)

CONFIG = Path("configs/experiments/gemma4_diverse28_block_cross_v35.yaml")
CHECKPOINT_ROOT = Path("data_gemma4/checkpoints/gemma4_v35_diverse28_block_cross")
REPORT = Path("reports/gemma4/metrics/v35_update32_terminal_gate.json")


def test_v35_terminal_gate_replays_exact_train_only_update32_failure() -> None:
    report = audit_v35_update32()
    assert report["passed"] is True
    assert report["gemma_loaded"] is False
    assert report["scene_maps_loaded"] is False
    assert report["qa_loaded"] is False
    assert report["oracle_loaded"] is False
    assert report["final_test_scenes_touched"] is False
    assert report["observed_saved_optimizer_steps"] == [0, 8, 16, 24, 32]
    assert report["stopped_at_optimizer_step"] == 32
    assert report["no_update_040_or_later"] is True
    gate = report["update32_gate_evidence"]
    assert gate["changed_selectivity_ratio_geometric_mean"] == 1.000036597251892
    assert gate["changed_selectivity_over_1_02_count"] == 0
    assert gate["baseline_complete_units"] == 8
    assert gate["terminal_complete_units"] == 9
    assert gate["complete_units_by_family_at_update32"] == {
        "book_support": 0,
        "mirror_lr": 2,
        "picture_support": 0,
    }
    assert gate["training_scenes_only"] is True
    assert gate["passed"] is False


def test_v35_terminal_gate_proves_exact_tensor_and_optimizer_surfaces() -> None:
    report = audit_v35_update32()
    transition = report["tensor_transition"]
    assert transition["terminal_changed_tensor_names"] == [
        "block_cross_residual.w_k",
        "block_cross_residual.w_o",
        "block_cross_residual.w_q",
        "block_cross_residual.w_v",
    ]
    assert transition["terminal_changed_tensor_count"] == 4
    assert transition["terminal_changed_parameter_count"] == 983_040
    assert transition["inherited_v33_tensor_count"] == 168
    assert transition["all_inherited_v33_tensors_bit_exact_at_every_saved_arm"] is True
    assert transition["persistent_block_core_buffer_count"] == 7
    assert transition["persistent_block_core_buffer_element_count"] == 779
    assert (
        transition["all_persistent_block_core_buffers_bit_exact_at_every_saved_arm"]
        is True
    )
    assert transition["only_declared_block_core_matrices_changed"] is True
    optimizer = report["optimizer_transition"]
    assert optimizer["all_saved_adam_states_exact_and_finite"] is True
    assert optimizer["step_one_output_only_progression_proven"] is True
    assert optimizer["saved_optimizer_states"]["32"]["qkv_adam_step"] == 31
    assert optimizer["saved_optimizer_states"]["32"]["output_adam_step"] == 32


def test_v35_terminal_gate_authorizes_only_exact_v36_joint_surface() -> None:
    report = audit_v35_update32()
    source = report["tensor_transition"]["authorized_v36_lora_source"]
    assert source == {
        "all_output_matrices_exact_zero": True,
        "alpha": 16.0,
        "bank": "extension_v30_joint_pair_query",
        "parameter_count": 131_072,
        "rank": 8,
        "state_sha256": (
            "2b1d89fbb9189ac551bf12905cf94036ebaa84696449b31c2b37b69d478fb70d"
        ),
        "target_language_layers": [18, 19, 20, 21],
        "target_module_suffixes": ["self_attn.q_proj"],
        "tensor_count": 8,
    }
    authorization = report["conditional_authorization"]
    assert authorization["authorized"] is True
    assert authorization["stage"] == "v36_joint_block_cross_upper_lora"
    assert authorization["scope"] == (
        "exact_v35_update32_block_cross_plus_existing_exact_zero_"
        "extension_v30_joint_pair_query_joint_only"
    )
    assert authorization["authorized_existing_lora_bank"] == (
        "extension_v30_joint_pair_query"
    )
    assert authorization["authorized_existing_lora_parameter_count"] == 131_072
    assert authorization["optimizer_updates_1_through_8"] == (
        "authorized_existing_lora_bank_only"
    )
    assert authorization["optimizer_updates_9_through_100"] == (
        "authorized_existing_lora_bank_plus_v35_block_cross_matrices"
    )
    assert authorization["fresh_adam_state_required"] is True
    assert authorization["new_lora_bank_authorized"] is False
    assert authorization["all_other_followup_architectures_authorized"] is False
    assert authorization["chat_promotion_authorized"] is False
    assert authorization["final_test_access_authorized"] is False
    assert report["conditional_v36_joint_upper_lora_authorized"] is True
    assert report["v35_chat_promotion_eligible"] is False


def test_v35_terminal_loaded_inventory_contains_no_environment_runtime_input() -> None:
    report = audit_v35_update32()
    inventory = report["loaded_file_inventory"]
    assert inventory
    assert all(
        path == str(CONFIG)
        or path == "reports/gemma4/metrics/v34_update32_terminal_gate.json"
        or path.startswith(f"{CHECKPOINT_ROOT}/update_")
        for path in inventory
    )
    assert all(
        fragment not in path.casefold()
        for path in inventory
        for fragment in ("/qa/", "/maps/", "/oracle/", "final_once", "scene_000025")
    )


def test_v35_terminal_report_is_exact_replay() -> None:
    assert json.loads(REPORT.read_text(encoding="utf-8")) == audit_v35_update32()
    assert hashlib.sha256(CONFIG.read_bytes()).hexdigest() == (
        "c8ddd808b2f338b9d61bcdadacbb0f679a0283d5e28a923fd25a6eab1a221485"
    )
    assert hashlib.sha256(REPORT.read_bytes()).hexdigest() == (
        "88205d018de14fc0518fe695bf7420c44ac832a1ee95eea0e2ae1f41deff4a27"
    )


def test_v35_terminal_gate_rejects_changed_config_bytes(tmp_path: Path) -> None:
    changed = tmp_path / "v35.yaml"
    changed.write_bytes(CONFIG.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="config bytes differ"):
        audit_v35_update32(changed, CHECKPOINT_ROOT)


def test_v35_terminal_gate_rejects_any_later_update_directory(tmp_path: Path) -> None:
    for step in (0, 8, 16, 24, 32, 40):
        (tmp_path / f"update_{step:03d}").mkdir()
    with pytest.raises(ValueError, match="stopped at its contiguous update-32"):
        _validate_checkpoint_sequence(tmp_path)


def test_v35_terminal_make_target_has_no_selection_or_final_dependency() -> None:
    makefile = Path("Makefile").read_text(encoding="utf-8")
    recipe = makefile.split("gemma4-v35-seal-update32:", maxsplit=1)[1].split(
        "\n\n", maxsplit=1
    )[0]
    assert "semantic_3d_chat.evaluation.v35_terminal_gate" in recipe
    assert "select" not in recipe
    assert "final" not in recipe
