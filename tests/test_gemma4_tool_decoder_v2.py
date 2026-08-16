from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import torch
from safetensors import safe_open

from semantic_3d_chat.config import PROJECT_ROOT, load_config
from semantic_3d_chat.evaluation.gemma4_tool_decoder_preregistration_v2 import (
    PREREGISTRATION_PATH,
    V1_FAILURE_PATH,
    V1_FAILURE_SHA256,
    V1_PREREGISTRATION_SHA256,
    _model_snapshot,
    build_tool_decoder_preregistration_v2,
    run_tiny_cpu_backward_smoke_v2,
    run_tool_decoder_preflight_v2,
)
from semantic_3d_chat.language.gemma4_tool_decoder_v2 import (
    INITIAL_LORA_STATE_SHA256,
    INITIAL_PROJECTOR_STATE_SHA256,
    LORA_PARAMETER_COUNT,
    PROJECTOR_PARAMETER_COUNT,
    TARGET_PROJECTION,
    TOTAL_TRAINABLE_PARAMETER_COUNT,
    NumericToolContextProjectorV2,
    canonical_answer_token_ids,
    canonical_tool_json_from_trace,
    tool_decoder_system_prompt,
)
from semantic_3d_chat.language.lora import lora_banks_settings, tensor_state_sha256


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_v1_is_authenticated_terminal_failure_before_training() -> None:
    preregistration = (
        PROJECT_ROOT
        / "reports/gemma4/metrics/gemma4_embodied_tool_decoder_preregistration_v1.json"
    )
    failure_path = PROJECT_ROOT / V1_FAILURE_PATH
    assert _sha256(preregistration) == V1_PREREGISTRATION_SHA256
    assert _sha256(failure_path) == V1_FAILURE_SHA256
    failure = json.loads(failure_path.read_text(encoding="utf-8"))
    assert failure["status"] == "terminal_design_failure_before_training"
    assert failure["execution"] == {
        "checkpoint_published": False,
        "full_checkpoint_loaded": False,
        "full_model_generation_executed": False,
        "mps_used": False,
        "multi_update_training_executed": False,
        "optimizer_steps": 0,
        "training_executed": False,
    }
    assert failure["finding"]["layer_34_loaded_k_proj_present"] is False
    assert failure["finding"]["layer_34_loaded_v_proj_present"] is False


def test_v2_preregistration_is_sealed_and_forbids_training() -> None:
    path = PROJECT_ROOT / PREREGISTRATION_PATH
    artifact = json.loads(path.read_text(encoding="utf-8"))
    assert artifact == build_tool_decoder_preregistration_v2()
    assert artifact["launch_authorization"]["multi_update_training_authorized"] is False
    assert artifact["execution"]["training_executed"] is False
    assert artifact["trainable_surface"]["total_trainable_parameter_count"] == 165_888
    assert artifact["trainable_surface"]["lora"]["exact_target_modules"] == [
        TARGET_PROJECTION
    ]


def test_v2_trainable_count_and_bank_are_exact_and_disjoint() -> None:
    assert LORA_PARAMETER_COUNT == 55_296
    assert PROJECTOR_PARAMETER_COUNT == 110_592
    assert TOTAL_TRAINABLE_PARAMETER_COUNT == 165_888
    config = load_config("configs/experiments/gemma4_embodied_tool_decoder_v2.yaml")
    banks = lora_banks_settings(config)
    selected = banks.bank("embodied_tool_decoder_v2_final_down")
    assert selected.trainable is True
    assert selected.adapter.target_modules == (TARGET_PROJECTION,)
    assert selected.expected_initial_state_sha256 == INITIAL_LORA_STATE_SHA256
    assert all(
        not bank.trainable
        for bank in banks.banks
        if bank.name != "embodied_tool_decoder_v2_final_down"
    )
    all_targets = [target for bank in banks.banks for target in bank.adapter.target_modules]
    assert len(all_targets) == len(set(all_targets))


def test_pinned_snapshot_contains_real_v2_projection_shape() -> None:
    snapshot = _model_snapshot()
    if not snapshot.is_dir():
        pytest.skip("Pinned local Gemma-4 snapshot is unavailable")
    with safe_open(snapshot / "model.safetensors", framework="pt", device="cpu") as archive:
        assert list(archive.get_slice(f"{TARGET_PROJECTION}.weight").get_shape()) == [
            1536,
            12288,
        ]


def test_v2_numeric_projector_is_deterministic_finite_and_shape_checked() -> None:
    first = NumericToolContextProjectorV2()
    second = NumericToolContextProjectorV2()
    assert tensor_state_sha256(first.state_dict()) == INITIAL_PROJECTOR_STATE_SHA256
    assert tensor_state_sha256(second.state_dict()) == INITIAL_PROJECTOR_STATE_SHA256
    target = torch.linspace(-1.0, 1.0, 10).unsqueeze(0)
    clearance = torch.linspace(0.0, 1.0, 24).unsqueeze(0)
    output = first(target, clearance)
    assert output.shape == (1, 4, 1536)
    assert torch.isfinite(output).all()
    assert torch.equal(output, second(target, clearance))
    with pytest.raises(ValueError, match="target_state"):
        first(torch.zeros(1, 9), clearance)
    with pytest.raises(ValueError, match="normalized"):
        first(target, torch.full((1, 24), 1.1))
    with pytest.raises(ValueError, match="NaN"):
        first(target, torch.full((1, 24), float("nan")))


def test_tool_labels_are_exact_minified_json_with_eos() -> None:
    row = {
        "action_index": 0,
        "action_name": "stop",
        "argument_target_normalized": 0.0,
    }
    answer = canonical_tool_json_from_trace(
        row, max_turn_degrees=45.0, max_move_m=0.5
    )
    assert answer == '{"arguments":{},"tool":"stop"}'

    class Tokenizer:
        eos_token_id = 7

        def __call__(self, text: str, **_: object) -> dict[str, torch.Tensor]:
            assert text == answer
            return {"input_ids": torch.tensor([[3, 4, 5]], dtype=torch.long)}

    ids = canonical_answer_token_ids(Tokenizer(), answer, device="cpu")
    assert torch.equal(ids, torch.tensor([[3, 4, 5, 7]], dtype=torch.long))


def test_tool_protocol_contains_no_environment_description() -> None:
    prompt = tool_decoder_system_prompt(max_turn_degrees=45.0, max_move_m=0.5)
    assert "continuous context" in prompt
    assert "JSON" in prompt
    for forbidden in ("chair", "bowl", "table", "lamp", "picture", "scene graph"):
        assert forbidden not in prompt.casefold()


def test_v2_structural_preflight_passes_but_launch_remains_blocked() -> None:
    report = run_tool_decoder_preflight_v2()
    assert report["status"] == "passed_structural_preflight_training_not_authorized"
    assert report["experiment_config"]["target_sets_disjoint"] is True
    assert report["experiment_config"]["all_older_banks_frozen"] is True
    assert report["trainable_parameter_count"] == TOTAL_TRAINABLE_PARAMETER_COUNT
    assert report["clearance_cache_materialized"] is False
    assert report["full_model_mps_microbatch_smoke_executed"] is False
    assert report["multi_update_training_authorized"] is False
    assert report["gemma_model_parameters_loaded"] is False


def test_tiny_true_gemma_cpu_microbatch_proves_gradients_and_forward_influence() -> None:
    pytest.importorskip("transformers.models.gemma4.modeling_gemma4")
    report = run_tiny_cpu_backward_smoke_v2()
    assert report["status"] == "passed"
    assert report["microbatches"] == 1
    assert report["optimizer_steps"] == 0
    assert report["tiny_layer_34_kv_modules_present"] is False
    assert report["zero_output_lora_exact_noop"] is True
    assert report["projector_gradient_l2"] > 0.0
    assert report["lora_b_gradient_l2"] > 0.0
    assert report["lora_a_gradient_l2_expected_zero"] == 0.0
    assert report["nonzero_lora_forward_influence_proved"] is True
    assert report["nonzero_lora_maximum_logit_change"] > 0.0
    assert report["base_model_trainable_parameter_count"] == 0
    assert report["full_checkpoint_loaded"] is False
    assert report["mps_used"] is False
    assert report["training_executed"] is False
