from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from semantic_3d_chat.evaluation.v36_terminal_gate import (
    _checkpoint_paths,
    audit_v36_update16,
)

CONFIG = Path("configs/experiments/gemma4_diverse28_joint_block_cross_v36.yaml")
CHECKPOINT_ROOT = Path("data_gemma4/checkpoints/gemma4_v36_diverse28_joint_block_cross")
REPORT = Path("reports/gemma4/metrics/v36_update16_terminal_gate.json")


def test_v36_terminal_gate_replays_exact_failed_update16_gate() -> None:
    report = audit_v36_update16()
    assert report["passed"] is True
    assert report["observed_saved_optimizer_steps"] == [0, 8, 16]
    assert report["stopped_at_optimizer_step"] == 16
    assert report["no_update_024_or_later"] is True
    assert report["v36_train_only_continuation_gate_passed"] is False
    assert report["v36_chat_promotion_eligible"] is False
    evidence = report["update16_gate_evidence"]
    assert evidence["teacher_complete_units"] == 9
    assert evidence["teacher_cross_prefix_complete_units"] == 16
    assert evidence["teacher_positive_sides"] == 34
    assert evidence["mean_cross_prefix_margin"] == 1.4565558433532715
    assert evidence["complete_physical_pair_coverage"] == 4
    assert evidence["failed_requirements"] == [
        "teacher_complete_units_at_least_10",
        "complete_physical_pair_coverage_at_least_5",
    ]
    assert evidence["checks"]["passed"] is False


def test_v36_terminal_gate_proves_exact_staged_tensor_surface() -> None:
    transition = audit_v36_update16()["tensor_transition"]
    assert transition["source_update_zero_bit_exact_v35_update32"] is True
    assert transition["adapter_tensor_count"] == 179
    assert transition["terminal_changed_tensor_count"] == 12
    assert transition["terminal_changed_parameter_count"] == 1_114_112
    assert transition["update8_only_eight_query_lora_tensors_changed"] is True
    assert transition["update16_only_twelve_authorized_tensors_changed"] is True
    assert transition["frozen_nonauthorized_tensor_count"] == 167
    assert transition["all_frozen_nonauthorized_tensors_bit_exact_at_every_arm"] is True
    assert transition["all_persistent_block_core_buffers_bit_exact_at_every_arm"] is True
    states = transition["state_sha256_by_optimizer_step"]
    assert states["0"]["full_tensor_state_sha256"] == (
        "1fe8f278460faeb1e13d9da09051a497965a566565c79a4f6ea28c56a9120326"
    )
    assert states["8"]["full_tensor_state_sha256"] == (
        "958c508fb9a8a59e8943bfb28fb276b43b39feccd89dabd908e196103d717676"
    )
    assert states["16"]["full_tensor_state_sha256"] == (
        "e9b6d1362d58f34aede04817b0c8d81320c616dcd4b64e9c0d3bbe56b5835dd7"
    )


def test_v36_terminal_gate_proves_exact_fresh_adam_staging() -> None:
    optimizer = audit_v36_update16()["optimizer_transition"]
    assert optimizer["fresh_v36_adam_staging_verified"] is True
    assert optimizer["v35_optimizer_state_loaded"] is False
    assert optimizer["saved_optimizer_states"]["8"]["lora_optimizer_step"] == 8
    assert optimizer["saved_optimizer_states"]["8"]["block_core_optimizer_step"] is None
    assert optimizer["saved_optimizer_states"]["16"]["lora_optimizer_step"] == 16
    assert optimizer["saved_optimizer_states"]["16"]["block_core_optimizer_step"] == 8


def test_v36_terminal_authorizes_only_existing_shared_kv_v37_surface() -> None:
    report = audit_v36_update16()
    source = report["tensor_transition"]["conditional_v37_existing_shared_kv_source"]
    assert source == {
        "all_tensors_finite": True,
        "bank": "extension_v23_shared_kv",
        "frozen_complement_state_sha256": (
            "c82b8715aebcb775a6e23cb5cd477520922682b5f41929017f4f91917eafe061"
        ),
        "frozen_complement_tensor_count": 171,
        "parameter_count": 30_720,
        "state_sha256": (
            "91a9eea577cab5a37e840cdf4007722a398415846af91280713bcb2cda0f045c"
        ),
        "tensor_count": 8,
    }
    authorization = report["conditional_authorization"]
    assert authorization["authorized"] is True
    assert authorization["stage"] == "v37_scene_ingress_kv"
    assert authorization["scope"] == "continue_existing_extension_v23_shared_kv_only"
    assert authorization["authorized_existing_lora_bank"] == "extension_v23_shared_kv"
    assert authorization["authorized_existing_lora_parameter_count"] == 30_720
    assert authorization["authorized_existing_lora_rank"] == 4
    assert authorization["authorized_existing_lora_alpha"] == 8.0
    assert authorization["authorized_existing_lora_target_language_layers"] == [13, 14]
    assert authorization["new_lora_bank_authorized"] is False
    assert authorization["existing_shared_kv_bank_reinitialization_authorized"] is False
    assert authorization["source_v36_block_core_frozen_for_all_updates"] is True
    assert authorization["source_v36_query_bank_frozen_for_all_updates"] is True
    assert authorization["fresh_adam_required"] is True
    assert authorization["v36_optimizer_state_may_be_loaded"] is False
    assert authorization["maximum_true_optimizer_steps"] == 64
    assert authorization["all_other_followup_architectures_authorized"] is False
    assert authorization["chat_promotion_authorized"] is False
    assert authorization["final_test_access_authorized"] is False
    assert report["conditional_v37_scene_ingress_kv_authorized"] is True


def test_v36_terminal_rejects_nonexistent_upper_kv_target_design() -> None:
    authorization = audit_v36_update16()["conditional_authorization"]
    paths = authorization["authorized_existing_lora_target_module_paths"]
    assert len(paths) == 4
    assert all("layers.13." in path or "layers.14." in path for path in paths)
    assert all("layers.18." not in path and "layers.21." not in path for path in paths)
    architecture = authorization["shared_kv_architecture_attestation"]
    assert architecture["layers_18_through_21_have_operative_kv_projections"] is False
    assert authorization.get("authorized_new_lora_parameter_count") not in {118_784, 143_360}


def test_v36_terminal_loaded_no_model_qa_map_or_environment_input() -> None:
    report = audit_v36_update16()
    assert report["gemma_loaded"] is False
    assert report["scene_maps_loaded"] is False
    assert report["qa_loaded"] is False
    assert report["validation_qa_loaded"] is False
    assert report["validation_model_selection_ran"] is False
    assert report["oracle_loaded"] is False
    assert report["final_test_scenes_touched"] is False
    inventory = report["loaded_file_inventory"]
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


def test_v36_terminal_protected_artifact_remains_exact() -> None:
    protected = audit_v36_update16()["protected_artifact"]
    assert protected["access"] == "bytes_hashed_only"
    assert protected["unchanged"] is True
    assert protected["sha256"] == (
        "c0086f66edbb8854a7938e09c57535bfd47100adbaf3b3c95eeb4b08014ce2f8"
    )


def test_v36_terminal_report_is_exact_replay() -> None:
    assert json.loads(REPORT.read_text(encoding="utf-8")) == audit_v36_update16()
    assert hashlib.sha256(CONFIG.read_bytes()).hexdigest() == (
        "d684da6451b54de3c17af9a7bd5bc2bf6756ecc064cbe54c5e19d53f82f326a1"
    )
    assert hashlib.sha256(REPORT.read_bytes()).hexdigest() == (
        "cb5b1248a4904dc58a685b64e052f980c02771b59eed5578bdbf2865ddbf5877"
    )


def test_v36_terminal_rejects_changed_config_bytes(tmp_path: Path) -> None:
    changed = tmp_path / "v36.yaml"
    changed.write_bytes(CONFIG.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="config bytes differ"):
        audit_v36_update16(changed, CHECKPOINT_ROOT)


def test_v36_terminal_rejects_any_update24_or_later(tmp_path: Path) -> None:
    for step in (0, 8, 16, 24):
        (tmp_path / f"update_{step:03d}").mkdir()
    with pytest.raises(ValueError, match="stopped at its contiguous update-16"):
        _checkpoint_paths(tmp_path)


def test_v36_terminal_make_target_has_no_selection_or_final_dependency() -> None:
    makefile = Path("Makefile").read_text(encoding="utf-8")
    recipe = makefile.split("gemma4-v36-seal-update16:", maxsplit=1)[1].split(
        "\n\n", maxsplit=1
    )[0]
    assert "semantic_3d_chat.evaluation.v36_terminal_gate" in recipe
    assert "select" not in recipe
    assert "final" not in recipe
