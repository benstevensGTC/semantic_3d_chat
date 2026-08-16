from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from semantic_3d_chat.evaluation.v37_terminal_gate import (
    _checkpoint_paths,
    audit_v37_update16,
)

CONFIG = Path("configs/experiments/gemma4_diverse28_scene_ingress_kv_v37.yaml")
CHECKPOINT_ROOT = Path(
    "data_gemma4/checkpoints/gemma4_v37_diverse28_scene_ingress_kv"
)
REPORT = Path("reports/gemma4/metrics/v37_update16_terminal_gate.json")


@pytest.fixture(scope="module")
def terminal() -> dict:
    return audit_v37_update16()


def test_v37_terminal_replays_exact_failed_update16_gate(terminal: dict) -> None:
    assert terminal["passed"] is True
    assert terminal["observed_saved_optimizer_steps"] == [0, 8, 16]
    assert terminal["stopped_at_optimizer_step"] == 16
    assert terminal["no_update_024_or_later"] is True
    assert terminal["v37_train_only_continuation_gate_passed"] is False
    assert terminal["v37_development_selector_legal"] is False
    evidence = terminal["update16_gate_evidence"]
    assert evidence["teacher_complete_units"] == 9
    assert evidence["complete_physical_pair_coverage"] == 4
    assert evidence["teacher_cross_prefix_complete_units"] == 18
    assert evidence["teacher_positive_sides"] == 33
    assert evidence["mean_cross_prefix_margin"] == 1.4349822998046875
    assert evidence["complete_units_by_family"] == {
        "book_support": 0,
        "mirror_lr": 2,
        "picture_support": 0,
    }
    assert evidence["failed_requirements"] == [
        "teacher_complete_units_at_least_10",
        "complete_physical_pair_coverage_at_least_5",
        "teacher_positive_sides_at_least_35",
        "mean_cross_prefix_margin_at_least_source",
        "book_or_picture_teacher_complete",
    ]
    assert evidence["checks"]["passed"] is False


def test_v37_terminal_pins_exact_v36_terminal_and_source(terminal: dict) -> None:
    prior = terminal["exact_v36_prior_and_source"]
    assert prior["terminal_report"]["sha256"] == (
        "cb5b1248a4904dc58a685b64e052f980c02771b59eed5578bdbf2865ddbf5877"
    )
    assert prior["source_tensor_state_sha256"] == (
        "e9b6d1362d58f34aede04817b0c8d81320c616dcd4b64e9c0d3bbe56b5835dd7"
    )
    assert prior["source_file_sha256"]["optimizer.pt"] == (
        "51a76712d87f24af793a28848d743034b9229d5e1df63d02c81e13efb5f12569"
    )
    assert prior["source_optimizer_access"] == "bytes_hashed_only_not_deserialized"


def test_v37_terminal_proves_target_only_tensor_transition(terminal: dict) -> None:
    transition = terminal["tensor_transition"]
    assert transition["source_update_zero_bit_exact_v36_update16"] is True
    assert transition["adapter_tensor_count"] == 179
    assert transition["target_tensor_count"] == 8
    assert transition["target_parameter_count"] == 30_720
    assert transition["frozen_complement_tensor_count"] == 171
    assert transition["update8_changed_exactly_all_eight_target_tensors"] is True
    assert transition["update16_changed_exactly_all_eight_target_tensors"] is True
    assert transition["all_frozen_complement_tensors_bit_exact_at_every_arm"] is True
    assert transition["learned_v36_core_bit_exact_at_every_arm"] is True
    assert transition["learned_v36_query_bank_bit_exact_at_every_arm"] is True
    assert len(transition["changed_tensor_names_by_optimizer_step"]["8"]) == 8
    assert len(transition["changed_tensor_names_by_optimizer_step"]["16"]) == 8
    assert all(
        name.startswith("lora_banks.extension_v23_shared_kv.")
        for step in ("8", "16")
        for name in transition["changed_tensor_names_by_optimizer_step"][step]
    )


def test_v37_terminal_pins_optimizer_files_manifests_and_moments(terminal: dict) -> None:
    transition = terminal["optimizer_transition"]
    assert transition["fresh_v37_adam_verified"] is True
    assert transition["optimizer_integrity_manifest_verified"] is True
    assert transition["source_v36_optimizer_access"] == (
        "bytes_hashed_only_not_deserialized"
    )
    for step in ("8", "16"):
        audit = transition["saved_optimizer_states"][step]
        assert audit["optimizer_step"] == int(step)
        assert audit["group_count"] == 1
        assert audit["moment_tensor_count"] == 16
        assert audit["exact_parameter_order_verified"] is True
        assert audit["exact_adamw_group_schema_verified"] is True
        assert audit["self_hash_linkage_verified"] is True
        assert len(audit["parameter_states_inspected"]) == 8
        assert "optimizer_audit.json" in terminal["saved_file_sha256_by_optimizer_step"][step]


def test_v37_terminal_authorizes_only_exact_v38_successor(terminal: dict) -> None:
    authorization = terminal["conditional_successor_authorization"]
    assert terminal["conditional_v38_authorized"] is True
    assert terminal["arbitrary_continuation_authorized"] is False
    assert terminal["only_exact_conditional_successor_authorized"] == "v38_query_recovery"
    assert authorization["authorized"] is True
    assert authorization["successor"] == "v38_query_recovery"
    assert authorization["source_checkpoint"].endswith(
        "gemma4_v37_diverse28_scene_ingress_kv/update_016"
    )
    assert authorization["chat_promotion_authorized"] is False
    assert authorization["final_test_access_authorized"] is False
    assert authorization["oracle_access_authorized"] is False
    assert terminal["chat_promotion_authorized"] is False
    assert terminal["v37_chat_promotion_eligible"] is False
    assert terminal["final_test_access_authorized"] is False
    assert terminal["oracle_access_authorized"] is False


def test_v38_authorization_pins_exact_k_only_hybrid(terminal: dict) -> None:
    authorization = terminal["conditional_successor_authorization"]
    hybrid = authorization["update_zero_initialization"]
    assert hybrid["scope"] == "v23_k_only_hybrid_from_exact_v36_and_v37"
    assert hybrid["v23_k_only_rollback_from_v36_u16"] is True
    assert hybrid["hybrid_full_tensor_state_sha256"] == (
        "7b951c6d7ae4f7b50603159f0bc4dfb4d50b5b40f9325134d78d1de1dae87fc0"
    )
    assert hybrid["hybrid_v23_bank_state_sha256"] == (
        "5c9233bd96b381e2f63443f8a739a868a21997b28061d90fb407a46d9de2d4cb"
    )
    assert hybrid["hybrid_v30_query_bank_state_sha256"] == (
        "050706c300e6fb0ac8e4cc02e26c565b54a9a89505104302d4ffcedc02124c64"
    )
    assert hybrid["hybrid_block_core_state_sha256"] == (
        "92652fd2dbde2406227503f50717b2031baa1bcbc050902a379ddb9ddb52764f"
    )
    assert hybrid["hybrid_frozen_excluding_v30_query_state_sha256"] == (
        "fe39da221505c1968030c67aacb4e99f1a179e05a97d2906d416afe5fef5ed78"
    )
    assert len(hybrid["differs_from_v36_u16_only_tensor_names"]) == 4
    assert all(
        ".adapters.0." in name or ".adapters.2." in name
        for name in hybrid["differs_from_v36_u16_only_tensor_names"]
    )
    assert len(hybrid["differs_from_v37_u16_only_tensor_names"]) == 4
    assert all(
        ".adapters.1." in name or ".adapters.3." in name
        for name in hybrid["differs_from_v37_u16_only_tensor_names"]
    )


def test_v38_authorization_trains_only_existing_query_bank_with_fresh_adamw(
    terminal: dict,
) -> None:
    authorization = terminal["conditional_successor_authorization"]
    assert authorization["authorized_existing_lora_bank"] == (
        "extension_v30_joint_pair_query"
    )
    assert authorization["authorized_existing_lora_parameter_count"] == 131_072
    assert authorization["authorized_existing_lora_tensor_count"] == 8
    assert authorization["authorized_existing_lora_rank"] == 8
    assert authorization["authorized_existing_lora_alpha"] == 16.0
    assert authorization["authorized_existing_lora_dropout"] == 0.0
    assert authorization["authorized_existing_lora_target_language_layers"] == [
        18,
        19,
        20,
        21,
    ]
    assert authorization["all_four_query_lora_b_tensors_nonzero"] is True
    assert authorization["new_lora_bank_authorized"] is False
    assert authorization["existing_query_bank_reinitialization_authorized"] is False
    optimizer = authorization["optimizer"]
    assert optimizer == {
        "type": "AdamW",
        "fresh_state_required": True,
        "v36_optimizer_state_may_be_loaded": False,
        "v37_optimizer_state_may_be_loaded": False,
        "learning_rate": 2e-5,
        "weight_decay": 0.0,
        "gradient_clip_norm": 1.0,
    }
    frozen = authorization["frozen_surface"]
    assert frozen["v23_shared_kv_frozen"] is True
    assert frozen["block_cross_residual_frozen"] is True
    assert frozen["every_other_tensor_and_buffer_frozen"] is True


def test_v38_authorization_pins_schedule_objective_and_relative_gates(
    terminal: dict,
) -> None:
    authorization = terminal["conditional_successor_authorization"]
    schedule = authorization["schedule"]
    assert schedule["true_optimizer_step_count"] == 41
    assert schedule["saved_optimizer_steps"] == [0, 8, 16, 24, 32, 40, 41]
    assert schedule["exact_pair_schedule_sha256"] == (
        "4e26160c9a36e20ac894ff5e26604e6fad985ff5c84211282fd9807c9e738b3f"
    )
    assert len(schedule["exact_pair_schedule"]) == 41
    assert schedule["exact_pair_schedule"][:8] == [
        {**row, "optimizer_step": index}
        for index, row in enumerate(
            schedule["priority_book_picture_units_steps_1_through_8"], start=1
        )
    ]
    assert schedule["exact_pair_schedule"][8:16] == [
        {**row, "optimizer_step": index}
        for index, row in enumerate(
            schedule["priority_book_picture_units_steps_1_through_8"], start=9
        )
    ]
    assert schedule["one_deterministic_unchanged_broad_row_per_update"] is True
    assert schedule["continuation_past_update_41_authorized"] is False
    assert authorization["objective"] == {
        "broad_answer_nll_weight": 0.5,
        "pair_correct_answer_nll_weight": 1.0,
        "side_hinge_weight": 8.0,
        "side_hinge_margin": 0.5,
        "cross_prefix_maintenance_weight": 1.0,
        "cross_prefix_maintenance_margin": 0.1,
        "additional_terms_authorized": False,
    }
    assert authorization["update8_gate"][
        "priority_teacher_deficit_delta_from_update_zero_maximum"
    ] == -0.5
    assert authorization["update8_gate"]["teacher_positive_sides_minimum"] == 34
    assert authorization["update8_gate"][
        "teacher_cross_prefix_complete_units_minimum"
    ] == 17
    assert authorization["update8_gate"][
        "broad_train_nll_maximum_increase_from_update_zero"
    ] == 0.02
    assert authorization["update16_gate"][
        "priority_teacher_deficit_delta_from_update_zero_maximum"
    ] == -3.12
    assert authorization["update16_gate"]["teacher_positive_sides_minimum"] == 35
    assert authorization["update16_gate"][
        "teacher_cross_prefix_complete_units_minimum"
    ] == 17
    assert authorization["update16_gate"][
        "broad_train_nll_maximum_increase_from_update_zero"
    ] == 0.02
    assert authorization["update41_gate"][
        "priority_teacher_deficit_delta_from_update_zero_maximum"
    ] == -6.24
    assert authorization["update41_gate"]["teacher_complete_units_minimum"] == 12
    assert authorization["update41_gate"]["train_greedy_complete_units_minimum"] == 6
    assert authorization["update41_gate"]["broad_greedy_exact_correct_minimum"] == 23
    assert authorization["update41_gate"][
        "broad_train_nll_maximum_increase_from_update_zero"
    ] == 0.02
    assert authorization["gate_artifact_requirements"][
        "per_unit_correct_answer_nll_must_be_persisted"
    ] is True
    assert authorization["gate_artifact_requirements"][
        "per_unit_rank_nll_must_be_persisted"
    ] is True


def test_v37_terminal_loaded_no_model_qa_map_or_environment_input(terminal: dict) -> None:
    assert terminal["gemma_loaded"] is False
    assert terminal["scene_maps_loaded"] is False
    assert terminal["qa_loaded"] is False
    assert terminal["validation_qa_loaded"] is False
    assert terminal["validation_model_selection_ran"] is False
    assert terminal["oracle_loaded"] is False
    assert terminal["final_test_scenes_touched"] is False
    inventory = terminal["loaded_file_inventory"]
    assert inventory
    assert all(
        fragment not in path.casefold()
        for path in inventory
        for fragment in (
            "/qa/",
            "/maps/",
            "/oracle/",
            "validation.jsonl",
            "final_once",
            "scene_000025",
            "scene_000030",
        )
    )


def test_v37_terminal_protected_artifact_remains_exact(terminal: dict) -> None:
    protected = terminal["protected_artifact"]
    assert protected["access"] == "bytes_hashed_only"
    assert protected["unchanged"] is True
    assert protected["sha256"] == (
        "c0086f66edbb8854a7938e09c57535bfd47100adbaf3b3c95eeb4b08014ce2f8"
    )


def test_v37_terminal_report_is_exact_replay() -> None:
    assert json.loads(REPORT.read_text(encoding="utf-8")) == audit_v37_update16()
    assert hashlib.sha256(CONFIG.read_bytes()).hexdigest() == (
        "38b0ee5a0921d77c909b31a6cad3834f2527589ef43e6c3671d02ae7731fa098"
    )
    assert hashlib.sha256(REPORT.read_bytes()).hexdigest() == (
        "8f8d9cfaf2c8cf564794b9f6d03eaa23f63d4fce96427816f5bc7b3fca9b70c2"
    )


def test_v37_terminal_rejects_changed_config_bytes(tmp_path: Path) -> None:
    changed = tmp_path / "v37.yaml"
    changed.write_bytes(CONFIG.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="config bytes differ"):
        audit_v37_update16(changed, CHECKPOINT_ROOT)


def test_v37_terminal_rejects_any_update24_or_later(tmp_path: Path) -> None:
    for step in (0, 8, 16, 24):
        (tmp_path / f"update_{step:03d}").mkdir()
    with pytest.raises(ValueError, match="stopped at its contiguous update-16"):
        _checkpoint_paths(tmp_path)
