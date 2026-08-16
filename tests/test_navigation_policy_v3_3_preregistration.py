from __future__ import annotations

import json

import pytest

from semantic_3d_chat.evaluation import navigation_policy_v3_3_preregistration as v33


def _passed_preflight() -> dict[str, object]:
    return {
        "required_before_full_gemma_load": True,
        "test_node": v33.ROUTING_TEST_NODE,
        "command": ["python", "-m", "pytest", "-q", v33.ROUTING_TEST_NODE],
        "exit_code": 0,
        "passed": True,
        "stdout_sha256": "a" * 64,
        "stderr_sha256": "b" * 64,
        "real_v3_checkpoint_controller_constructed_on_cpu": True,
        "exact_live_wrapper_install_function_exercised": True,
        "enveloped_scan_then_move_to_then_stop_proved": True,
        "planner_metadata_proved_before_stop": True,
        "full_gemma_model_loaded": False,
        "optimizer_constructed": False,
        "oracle_loaded": False,
    }


def test_v3_3_preregistration_seals_development_scope_and_exact_routing(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    outputs = {name: str(tmp_path / f"future_{name}") for name in v33.SUCCESSOR_OUTPUTS}
    runtime_outputs = {
        name: str(tmp_path / f"future_runtime_{name}") for name in v33.SUCCESSOR_RUNTIME_OUTPUTS
    }
    result = str(tmp_path / "future_result.json")
    monkeypatch.setattr(v33, "_v3_2_provenance", lambda: {"status": "rejected"})
    monkeypatch.setattr(
        v33,
        "_hash_files",
        lambda paths: {str(path): "c" * 64 for path in paths},
    )
    monkeypatch.setattr(
        v33,
        "_hash_trees",
        lambda paths: {str(path): "d" * 64 for path in paths},
    )

    payload = v33.build_preregistration(
        _passed_preflight(),
        successor_outputs=outputs,
        successor_runtime_outputs=runtime_outputs,
        result_output=result,
    )

    assert payload["runtime_version"] == "v3.3"
    assert payload["status"] == "sealed_before_single_v3_3_development_run"
    assert payload["claim_scope"] == {
        "development_calibration": True,
        "same_benchmark_used_for_diagnosis": True,
        "held_out_claim": False,
        "generalization_claim": False,
        "stop_calibration_family_if_rejected": True,
    }
    assert payload["routing_integration_preflight"] == _passed_preflight()
    assert payload["authorized_change"]["kind"] == ("correct_live_protocol_envelope_routing_only")
    assert payload["authorized_change"]["numeric_planner_calibration_unchanged_from_v3_2"]
    assert payload["authorized_change"]["environmental_text_inputs"] == []
    assert payload["authorized_change"]["oracle_inputs_at_runtime"] is False
    assert payload["acceptance_gates"]["minimum_success_count"] == 6
    assert payload["acceptance_gates"]["numeric_planner_action_required"] == "move_to"
    assert payload["calibration"]["single_live_run"] is True
    assert payload["calibration"]["no_post_result_tuning"] is True
    assert payload["successor_outputs"] == outputs
    assert payload["successor_runtime_outputs"] == runtime_outputs
    assert payload["successor_outputs_absent_at_preregistration"] is True
    assert payload["full_gemma_model_loaded_by_preregistration"] is False
    assert payload["benchmark_rerun_completed"] is False
    assert payload["runtime_promotion_authorized"] is False

    # Ensure the payload remains strict JSON rather than silently admitting
    # NaN or implementation-specific values into the sealed contract.
    json.dumps(payload, sort_keys=True, allow_nan=False)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("passed", False),
        ("exit_code", 1),
        ("full_gemma_model_loaded", True),
        ("enveloped_scan_then_move_to_then_stop_proved", False),
        ("planner_metadata_proved_before_stop", False),
    ],
)
def test_v3_3_preregistration_rejects_inexact_routing_preflight(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    value: object,
) -> None:
    outputs = {name: str(tmp_path / f"future_{name}") for name in v33.SUCCESSOR_OUTPUTS}
    runtime_outputs = {
        name: str(tmp_path / f"future_runtime_{name}") for name in v33.SUCCESSOR_RUNTIME_OUTPUTS
    }
    preflight = _passed_preflight()
    preflight[field] = value
    monkeypatch.setattr(v33, "_v3_2_provenance", lambda: {"status": "rejected"})

    with pytest.raises(ValueError, match="exact routing preflight"):
        v33.build_preregistration(
            preflight,
            successor_outputs=outputs,
            successor_runtime_outputs=runtime_outputs,
            result_output=str(tmp_path / "future_result.json"),
        )


def test_v3_3_production_outputs_are_absent_before_sealing() -> None:
    for path in (
        *v33.SUCCESSOR_OUTPUTS.values(),
        *v33.SUCCESSOR_RUNTIME_OUTPUTS.values(),
        v33.DEFAULT_RESULT_OUTPUT,
    ):
        assert not v33._rooted(path).exists()


def test_v3_3_runner_authenticates_before_inference_and_after_result() -> None:
    source = v33._rooted("scripts/run_learned_navigation_benchmark_v3_3.sh")
    contents = source.read_text(encoding="utf-8")

    prereg_auth = contents.index("authenticate --preregistration")
    inference = contents.index("scripts/run_llm_navigation_inference_v3_3.py")
    result = contents.index("result --preregistration")
    result_auth = contents.index("authenticate-result --preregistration")
    assert prereg_auth < inference < result < result_auth
    assert "TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1" in contents
    assert "Refusing to reuse the V3.3 runtime output tree" in contents
