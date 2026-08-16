from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from semantic_3d_chat.config import load_config
from semantic_3d_chat.evaluation.v38_terminal_gate import (
    _checkpoint_paths,
    audit_v38_update8,
)
from semantic_3d_chat.training.train_query_recovery_v38 import (
    replay_v38_gates,
    v38_contract,
)

CONFIG = Path("configs/experiments/gemma4_diverse28_query_recovery_v38.yaml")
CHECKPOINT_ROOT = Path(
    "data_gemma4/checkpoints/gemma4_v38_diverse28_query_recovery"
)
REPORT = Path("reports/gemma4/metrics/v38_update8_terminal_gate.json")


@pytest.fixture(scope="module")
def terminal() -> dict:
    return audit_v38_update8()


def test_v38_terminal_replays_exact_failed_update8_gate(terminal: dict) -> None:
    assert terminal["passed"] is True
    assert terminal["observed_saved_optimizer_steps"] == [0, 8]
    assert terminal["stopped_at_optimizer_step"] == 8
    assert terminal["no_update_016_or_later"] is True
    assert terminal["v38_train_only_continuation_gate_passed"] is False
    assert terminal["v38_development_selector_legal"] is False
    evidence = terminal["update8_gate_evidence"]
    assert evidence["passed"] is False
    assert evidence["source"]["complete_units"] == 9
    assert evidence["source"]["positive_sides"] == 34
    assert evidence["source"]["cross_prefix_complete_units"] == 17
    assert evidence["source"]["broad_train_nll"] == 2.9013306349515915
    assert evidence["update8"]["complete_units"] == 9
    assert evidence["update8"]["complete_physical_pair_coverage"] == 4
    assert evidence["update8"]["positive_sides"] == 33
    assert evidence["update8"]["cross_prefix_complete_units"] == 16
    assert evidence["update8"]["broad_train_nll"] == 2.9220473815997443
    assert evidence["update8"]["priority_side_deficit"] == 31.062952995300293
    assert evidence["failed_requirements"] == [
        "priority_teacher_deficit_improved_at_least_0_5",
        "teacher_positive_sides_at_least_34",
        "teacher_cross_complete_units_at_least_17",
        "broad_nll_within_hybrid_update_zero_plus_0_02",
    ]
    assert evidence["checks"]["passed"] is False


def test_v38_public_replay_uses_nested_behavioral_baseline_schema() -> None:
    config = load_config(CONFIG)
    contract = v38_contract(config)
    metadata = json.loads(
        (CHECKPOINT_ROOT / "update_008" / "metadata.json").read_text(encoding="utf-8")
    )
    gate8, gate16, gate41 = replay_v38_gates(metadata, contract)
    assert gate8 == metadata["v38_query_recovery"]["update8_train_only_gate"]
    assert gate8 is not None and gate8["passed"] is False
    assert gate16 is None
    assert gate41 is None


def test_v38_terminal_proves_exact_query_only_tensor_transition(terminal: dict) -> None:
    transition = terminal["tensor_transition"]
    assert transition["adapter_tensor_count"] == 179
    assert transition["query_tensor_count"] == 8
    assert transition["query_parameter_count"] == 131_072
    assert transition["frozen_tensor_count"] == 171
    assert transition["update_zero_bit_exact_authenticated_hybrid"] is True
    assert transition["update8_changed_exactly_all_eight_query_tensors"] is True
    assert transition["all_frozen_tensors_bit_exact"] is True
    assert transition["all_tensors_finite"] is True
    assert transition["changed_tensor_names_by_optimizer_step"]["0"] == []
    changed = transition["changed_tensor_names_by_optimizer_step"]["8"]
    assert len(changed) == 8
    assert all(
        name.startswith("lora_banks.extension_v30_joint_pair_query.")
        for name in changed
    )
    assert transition["state_sha256_by_optimizer_step"]["0"][
        "full_tensor_state_sha256"
    ] == "7b951c6d7ae4f7b50603159f0bc4dfb4d50b5b40f9325134d78d1de1dae87fc0"
    assert transition["state_sha256_by_optimizer_step"]["8"][
        "full_tensor_state_sha256"
    ] == "6af96e291df87ea03f608c5db069e4a535e756fbb94bb52bd1446eb11a3859b6"


def test_v38_terminal_authenticates_own_optimizer_only(terminal: dict) -> None:
    transition = terminal["optimizer_transition"]
    assert transition["fresh_v38_adam_verified"] is True
    assert transition["optimizer_integrity_manifest_verified"] is True
    assert transition["source_or_rollback_optimizer_opened"] is False
    audit = transition["saved_optimizer_states"]["8"]
    assert audit["optimizer_step"] == 8
    assert audit["group_count"] == 1
    assert audit["moment_tensor_count"] == 16
    assert audit["exact_parameter_order_verified"] is True
    assert audit["exact_adamw_group_schema_verified"] is True
    assert audit["fresh_v38_adam_verified"] is True
    assert audit["self_hash_linkage_verified"] is True
    assert audit["optimizer_sha256"] == (
        "6cf931a84157e24ab593ad786733e7ddbd57522ed17985db44fad4d3d0c0d089"
    )
    source = terminal["exact_v37_terminal_and_source"]
    assert source["source_optimizer_access"] == "not_opened_not_hashed_not_deserialized"
    assert source["rollback_optimizer_access"] == (
        "not_opened_not_hashed_not_deserialized"
    )


def test_v38_terminal_selector_refuses_before_validation(terminal: dict) -> None:
    refusal = terminal["selector_refusal"]
    assert refusal["selector_refused_incomplete_envelope"] is True
    assert refusal["validation_evaluator_constructed"] is False
    assert refusal["selector_output_written"] is False
    assert refusal["refusal_type"] == "FileNotFoundError"
    assert "exact completed update-41 envelope" in refusal["refusal_message"]


def test_v38_terminal_binds_completed_tomography_and_scope_extension(
    terminal: dict,
) -> None:
    assert terminal["seal_revision"] == 2
    summary = terminal["query_delta_tomography_summary"]
    assert summary["artifact"] == "v38_query_delta_tomography_summary"
    assert summary["source_terminal_report_sha256"] == (
        "0b637bf6a57ed1a2903e9c58e313fa2539c3dabc636444572c2018c1ee5e6b7f"
    )
    scale = summary["full_bank_scale_screen"]
    assert scale["observed_scale_factors"] == [
        -1.0,
        -0.5,
        0.0,
        0.25,
        0.5,
        0.75,
        1.0,
        1.5,
        2.0,
    ]
    assert scale["eligible_candidate_count"] == 0
    assert scale["best_priority_deficit_gain"] == 0.304239
    assert scale["best_priority_deficit_gain_scale"] == 0.5
    assert scale["best_gain_candidate_metrics"] == {
        "complete_units": 8,
        "complete_physical_pair_coverage": 3,
        "positive_sides": 33,
    }
    masks = summary["layer_mask_scale_one_screen"]
    assert masks["observed_nonempty_mask_count"] == 15
    assert masks["eligible_candidate_count"] == 0
    assert masks["best_retention_shaped_mask"] == {
        "layers": [19, 20, 21],
        "complete_units": 9,
        "complete_physical_pair_coverage": 4,
        "positive_sides": 34,
        "cross_prefix_complete_units": 20,
        "priority_deficit_gain": 0.101911,
        "eligible": False,
    }
    assert summary["aggregate_result"]["eligible_candidate_count"] == 0
    scope = summary["scope_audit"]
    assert scope["candidate_grid_fully_within_prior_artifact_authorization"] is False
    assert scope["unplanned_candidate_grid_extension"] is True
    assert scope["unplanned_scale_extensions"] == [-1.0, -0.5, 1.5, 2.0]
    assert len(scope["unplanned_layer_masks"]) == 6
    assert scope["retroactive_authorization_claimed"] is False
    assert scope["reported_parameter_writes"] is False
    assert scope["reported_optimizer_step_calls"] is False
    assert scope["raw_per_candidate_trace_bound"] is False


def test_v38_terminal_authorizes_only_no_write_v28_l14_gradient_screen(
    terminal: dict,
) -> None:
    authorization = terminal["conditional_successor_authorization"]
    assert terminal["query_delta_tomography_completed"] is True
    assert terminal["conditional_v39_query_delta_tomography_authorized"] is False
    assert terminal[
        "conditional_v39_v28_layer14_gradient_cosine_screen_authorized"
    ] is True
    assert terminal["v39_training_authorized"] is False
    assert terminal["arbitrary_continuation_authorized"] is False
    assert terminal["only_exact_conditional_successor_authorized"] == (
        "v39_v28_layer14_gradient_cosine_screen"
    )
    assert authorization["authorized"] is True
    assert authorization["successor"] == "v39_v28_layer14_gradient_cosine_screen"
    assert authorization["existing_lora_bank"] == "extension_v28_stage_b_query"
    assert authorization["existing_adapter_index"] == 1
    assert authorization["target_language_layer"] == 14
    assert authorization["target_tensor_count"] == 2
    assert authorization["target_parameter_count"] == 22_528
    assert authorization["target_rank"] == 4
    assert authorization["target_alpha"] == 8.0
    assert authorization["target_dropout"] == 0.0
    assert authorization["target_source_state_sha256"] == (
        "9ff9d535a094f96328483c46ff8c8ea5fca30edc35878492976c35f8674a9f87"
    )
    assert authorization["gradient_computation_authorized"] is True
    assert authorization[
        "backward_or_autograd_grad_for_measurement_authorized"
    ] is True
    assert authorization["training_authorized"] is False
    assert authorization["optimizer_construction_authorized"] is False
    assert authorization["optimizer_step_authorized"] is False
    assert authorization["parameter_update_authorized"] is False
    assert authorization["parameter_or_buffer_write_authorized"] is False
    assert authorization["source_optimizer_access_authorized"] is False
    assert authorization["update8_optimizer_access_authorized"] is False
    assert authorization["new_lora_bank_authorized"] is False
    assert authorization["validation_access_authorized"] is False
    assert authorization["final_test_access_authorized"] is False
    assert authorization["oracle_access_authorized"] is False
    assert authorization["chat_promotion_authorized"] is False
    assert authorization["separate_terminal_seal_required_for_any_training"] is True
    assert authorization["frozen_excluding_target_tensor_count"] == 177
    assert authorization["frozen_excluding_target_state_sha256"] == (
        "7f33e541d36de33b10ceeac25e5f40374bffd1cf4b234af7a6b6341198b85360"
    )


def test_v38_terminal_loaded_no_gemma_qa_map_or_environment_input(
    terminal: dict,
) -> None:
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
    assert not any(
        path.endswith("optimizer.pt")
        and (
            "gemma4_v36_diverse28_joint_block_cross" in path
            or "gemma4_v37_diverse28_scene_ingress_kv" in path
        )
        for path in inventory
    )


def test_v38_terminal_protected_artifact_remains_exact(terminal: dict) -> None:
    protected = terminal["protected_artifact"]
    assert protected["access"] == "bytes_hashed_only"
    assert protected["unchanged"] is True
    assert protected["sha256"] == (
        "c0086f66edbb8854a7938e09c57535bfd47100adbaf3b3c95eeb4b08014ce2f8"
    )


def test_v38_terminal_report_is_exact_replay() -> None:
    assert json.loads(REPORT.read_text(encoding="utf-8")) == audit_v38_update8()
    assert hashlib.sha256(CONFIG.read_bytes()).hexdigest() == (
        "df884cdebed805fb783d68981011c2a66f1a37dc27aa8ecb529e1b981d25a7c5"
    )


def test_v38_terminal_rejects_changed_config_bytes(tmp_path: Path) -> None:
    changed = tmp_path / "v38.yaml"
    changed.write_bytes(CONFIG.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="config bytes differ"):
        audit_v38_update8(changed, CHECKPOINT_ROOT)


def test_v38_terminal_rejects_any_update16_or_later(tmp_path: Path) -> None:
    for step in (0, 8, 16):
        (tmp_path / f"update_{step:03d}").mkdir()
    with pytest.raises(ValueError, match="stopped at its contiguous failed update-8"):
        _checkpoint_paths(tmp_path)
