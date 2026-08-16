from __future__ import annotations

from pathlib import Path

import pytest

from semantic_3d_chat.config import load_config
from semantic_3d_chat.evaluation.gemma4_tool_decoder_preregistration_v2 import (
    _ExactDecoderSurface,
)
from semantic_3d_chat.evaluation.gemma4_tool_decoder_training_authorization_v2_2 import (
    build_cpu_authorization_v2_2,
)
from semantic_3d_chat.language.gemma4_tool_decoder_v2 import (
    NumericToolContextProjectorV2,
    tool_decoder_lora_settings_v2,
)
from semantic_3d_chat.language.gemma4_tool_decoder_v2_checkpoint import (
    build_runtime_metadata_v2,
    load_runtime_checkpoint_v2,
    publish_runtime_checkpoint_v2,
    validate_saved_runtime_probe_v2,
)
from semantic_3d_chat.language.lora import (
    initialize_lora_adapter_state,
    install_lora_adapters,
)
from semantic_3d_chat.training.train_gemma4_tool_decoder_v2 import (
    authenticate_training_authorization_v2,
)


@pytest.fixture(scope="module")
def surface():
    model = _ExactDecoderSurface().requires_grad_(False)
    installation = install_lora_adapters(model, tool_decoder_lora_settings_v2())
    assert installation is not None
    initialize_lora_adapter_state(installation, seed=2026081218)
    return installation, NumericToolContextProjectorV2()


def _provenance() -> dict[str, str]:
    values = ("1", "2", "3", "4", "5", "6")
    return dict(
        zip(
            (
                "base_checkpoint_sha256",
                "preregistration_sha256",
                "cpu_preflight_sha256",
                "training_authorization_sha256",
                "clearance_cache_sha256",
                "prefix_inventory_sha256",
            ),
            (value * 64 for value in values),
            strict=True,
        )
    )


def _passing_evaluation() -> dict[str, object]:
    teacher = {
        "all_heldout_rows_scored": True,
        "sample_count": 2268,
        "scene_count": 8,
        "answer_token_nll": 0.1,
        "answer_token_accuracy": 0.99,
        "exact_sequence_accuracy": 0.9,
        "teacher_forced_argmax_valid_schema_rate": 0.99,
        "teacher_forced_argmax_tool_accuracy": 0.95,
    }
    teacher_gate = {
        "schema": "semantic_3d_chat.gemma4_tool_decoder_teacher_forced_gate.v2",
        "checks": {
            "all_heldout_rows_scored": True,
            "sample_count": True,
            "scene_count": True,
            "answer_token_nll": True,
            "answer_token_accuracy": True,
            "exact_sequence_accuracy": True,
            "teacher_forced_argmax_valid_schema_rate": True,
            "teacher_forced_argmax_tool_accuracy": True,
        },
        "passed": True,
        "failed": [],
        "evaluated_before_greedy_generation": True,
    }
    causal_checks = {
        "sample_count_per_condition": True,
        "wrong_scene_nll_increase": True,
        "zero_scene_nll_increase": True,
        "wrong_robot_targeted_nll_increase": True,
        "zero_robot_targeted_nll_increase": True,
        "wrong_target_targeted_nll_increase": True,
        "zero_target_targeted_nll_increase": True,
        "wrong_clearance_targeted_nll_increase": True,
        "zero_clearance_targeted_nll_increase": True,
    }
    causal = {
        "sample_count_per_condition": 448,
        "drops_from_primary": {
            mode: {
                "answer_token_nll_increase": 0.1,
                "by_family": {
                    family: {"answer_token_nll_increase": 0.1}
                    for family in (
                        "face",
                        "approach",
                        "left_right",
                        "obstacle",
                        "collision_recovery",
                    )
                },
            }
            for mode in (
                "wrong_scene",
                "zero_scene",
                "wrong_robot",
                "zero_robot",
                "wrong_target",
                "zero_target",
                "wrong_clearance",
                "zero_clearance",
            )
        },
    }
    causal_gate = {
        "schema": "semantic_3d_chat.gemma4_tool_decoder_teacher_causal_gate.v2_1",
        "checks": causal_checks,
        "passed": True,
        "failed": [],
        "evaluated_before_greedy_generation": True,
    }
    primary = {
        "exact_json_accuracy": 0.9,
        "valid_schema_rate": 1.0,
        "tool_accuracy": 0.95,
        "turn_sign_accuracy": 0.95,
        "argument_mae_normalized": 0.1,
        "unsafe_motion_count": 0,
    }
    return {
        "conditions": {"primary": primary},
        "primary_large": primary,
        "drops_from_primary": {},
        "greedy_output_change_rate_from_primary": {
            "wrong_clearance": 0.2,
            "zero_clearance": 0.2,
        },
        "all_heldout_teacher_forced": teacher,
        "teacher_forced_early_gate": teacher_gate,
        "teacher_forced_causal_controls": causal,
        "teacher_forced_causal_gate": causal_gate,
    }


def test_checkpoint_metadata_is_runtime_only_and_exact(surface) -> None:
    installation, projector = surface
    metadata = build_runtime_metadata_v2(
        installation,
        projector,
        weights_sha256="a" * 64,
        provenance=_provenance(),
        promoted=True,
    )
    assert metadata["environmental_text_inputs"] == []
    assert metadata["oracle_inputs_at_runtime"] is False
    assert metadata["max_new_tokens"] == 24
    assert metadata["runtime_required_files"] == [
        "tool_decoder.safetensors",
        "runtime_metadata.json",
    ]


def test_saved_runtime_gate_rejects_noncanonical_and_nonnumeric_state() -> None:
    config = load_config("configs/experiments/gemma4_embodied_tool_decoder_v2.yaml")
    base = {
        "saved_checkpoint_loaded": True,
        "generated_text": '{"arguments":{},"tool":"stop"}',
        "numeric_robot_state": {"x": 0.0, "yaw": 0.0},
        "tool_execution_attempted": True,
        "tool_execution_result": {"success": True, "collision": False},
        "collision_interlock_checked": True,
        "oracle_inputs_loaded": False,
        "environmental_text_inputs": [],
    }
    assert validate_saved_runtime_probe_v2(base, config)["passed"] is True
    assert (
        validate_saved_runtime_probe_v2(
            {**base, "generated_text": '{"tool":"stop","arguments":{}}'}, config
        )["passed"]
        is False
    )
    assert (
        validate_saved_runtime_probe_v2(
            {**base, "numeric_robot_state": "pose unavailable"}, config
        )["passed"]
        is False
    )


def test_publish_strict_load_and_runtime_execution_gate(tmp_path: Path, surface) -> None:
    installation, projector = surface
    destination = tmp_path / "runtime" / "tool_v2"
    config = load_config("configs/experiments/gemma4_embodied_tool_decoder_v2.yaml")

    def probe(staging: Path):
        metadata = load_runtime_checkpoint_v2(
            staging,
            installation,
            projector,
            expected_provenance=_provenance(),
            require_promoted=False,
        )
        assert metadata["status"] == "staged_runtime_probe_only"
        return {
            "saved_checkpoint_loaded": True,
            "generated_text": '{"arguments":{},"tool":"stop"}',
            "numeric_robot_state": {"position": [0.0, 0.0, 0.0], "yaw": 0.0},
            "tool_execution_attempted": True,
            "tool_execution_result": {"success": True, "collision": False},
            "collision_interlock_checked": True,
            "oracle_inputs_loaded": False,
            "environmental_text_inputs": [],
        }

    publication = publish_runtime_checkpoint_v2(
        destination,
        installation,
        projector,
        provenance=_provenance(),
        evaluation=_passing_evaluation(),
        runtime_probe=probe,
        config=config,
    )
    assert publication["published"] is True
    assert {path.name for path in destination.iterdir()} == {
        "tool_decoder.safetensors",
        "runtime_metadata.json",
    }
    loaded = load_runtime_checkpoint_v2(
        destination,
        installation,
        projector,
        expected_provenance=_provenance(),
    )
    assert loaded["status"] == "promoted_runtime"


def test_cpu_authorization_cannot_start_heavy_model_load(tmp_path: Path) -> None:
    missing = tmp_path / "missing.json"
    with pytest.raises(FileNotFoundError):
        authenticate_training_authorization_v2(missing)


def test_sealed_cpu_authorization_has_no_heavy_permission() -> None:
    expected = build_cpu_authorization_v2_2()
    payload, digest = authenticate_training_authorization_v2()
    assert payload == expected
    assert len(digest) == 64
    assert payload["resource_contract"]["checkpoint_selection"] == (
        "fixed_final_update_64_no_posthoc_selection"
    )
    assert payload["full_model_mps_microbatch_authorized"] is False
    assert payload["multi_update_training_authorized"] is False
    assert payload["execution"]["full_model_loaded"] is False
    assert payload["execution"]["optimizer_steps"] == 0
    with pytest.raises(PermissionError, match="cannot authorize"):
        authenticate_training_authorization_v2(
            required_stage="full_model_mps_microbatch"
        )
    with pytest.raises(PermissionError, match="cannot authorize"):
        authenticate_training_authorization_v2(required_stage="multi_update_training")
