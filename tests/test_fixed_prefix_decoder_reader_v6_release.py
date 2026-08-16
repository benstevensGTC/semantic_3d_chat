from __future__ import annotations

import json
from pathlib import Path

import pytest

from semantic_3d_chat.evaluation import fixed_prefix_decoder_reader_v6_release as release
from semantic_3d_chat.evaluation.fixed_prefix_decoder_reader_v6_preregistration import (
    _canonical_sha256,
    build_preregistration_draft,
)


def test_v6_release_binds_the_independently_audited_frozen_proposal() -> None:
    observed = {
        path: release.sha256_file(path) for path in release.FROZEN_PROPOSAL_HASHES
    }
    assert observed == release.FROZEN_PROPOSAL_HASHES
    assert _canonical_sha256(build_preregistration_draft()) == release.PROPOSAL_SHA256


def test_v6_sealed_payload_authorizes_no_heavy_execution_yet(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        release,
        "_authenticate_frozen_proposal",
        lambda: {
            "source_sha256": release.FROZEN_PROPOSAL_HASHES,
            "canonical_proposal_sha256": release.PROPOSAL_SHA256,
            "local_model_weights_sha256": "a" * 64,
            "local_model_weights_size_bytes": 10_246_621_918,
            "actual_model_bytes_streamed": True,
        },
    )
    payload = release.build_sealed_preregistration()
    assert payload["authorization"] == {
        "sealed": True,
        "full_model_mps_smoke_authorized": False,
        "joint_runtime_smoke_authorized": False,
        "optimizer_construction_authorized": False,
        "multi_update_training_authorized": False,
        "checkpoint_write_authorized": False,
    }
    assert payload["independent_audit"]["deferred_or_final_qa_accessed"] is False


def test_v6_create_once_artifact_refuses_overwrite(tmp_path: Path) -> None:
    target = tmp_path / "release.json"
    path, digest = release._create_once(target, {"passed": True})
    assert path == target
    assert len(digest) == 64
    assert json.loads(target.read_text()) == {"passed": True}
    with pytest.raises(FileExistsError, match="create-once"):
        release._create_once(target, {"passed": False})


def test_v6_stage_source_inventories_are_disjoint_and_explicit() -> None:
    assert "smoke_fixed_prefix_decoder_reader_v6.py" in " ".join(
        release.SMOKE_BOUND_PATHS
    )
    assert "train_fixed_prefix_decoder_reader_v6.py" in " ".join(
        release.TRAINING_BOUND_PATHS
    )
    assert set(release.FROZEN_PROPOSAL_HASHES).isdisjoint(release.ROBOT_STATE_HASHES)
    assert release.MPS_SMOKE_REPORT not in release.SMOKE_BOUND_PATHS
    assert release.RESULT_REPORT not in release.TRAINING_BOUND_PATHS
    assert {
        "src/semantic_3d_chat/chat/runtime.py",
        "src/semantic_3d_chat/chat/runtime_config.py",
        "src/semantic_3d_chat/language/gemma4_answer_tail.py",
        "src/semantic_3d_chat/robot/state_checkpoint.py",
        "src/semantic_3d_chat/robot/state_encoder.py",
        "pyproject.toml",
        "uv.lock",
    }.issubset(release.SMOKE_BOUND_PATHS)


def test_v6_smoke_attempt_claim_is_create_once(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    attempt = tmp_path / "attempt.json"
    terminal = tmp_path / "terminal.json"
    monkeypatch.setattr(release, "MPS_SMOKE_ATTEMPT", str(attempt))
    monkeypatch.setattr(release, "MPS_SMOKE_REPORT", str(terminal))
    monkeypatch.setattr(
        release,
        "authenticate_mps_smoke_release",
        lambda: ({"status": "released"}, "a" * 64),
    )
    first, _digest = release.claim_mps_smoke_attempt()
    assert first == attempt
    assert json.loads(attempt.read_text())["status"] == "claimed_before_model_loading"
    with pytest.raises(FileExistsError, match="create-once"):
        release.claim_mps_smoke_attempt()


def test_v6_passing_smoke_rejects_public_boolean_forgery(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    terminal = tmp_path / "terminal.json"
    release_path = tmp_path / "release.json"
    release_path.write_text("{}", encoding="utf-8")
    terminal.write_text(
        json.dumps(
            {
                "status": "passed",
                "passed": True,
                "authorization_sha256": release.sha256_file(release_path),
                "device": "mps",
                "full_model_loaded": True,
                "mps_used": True,
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(release, "MPS_SMOKE_REPORT", str(terminal))
    monkeypatch.setattr(release, "MPS_SMOKE_RELEASE", str(release_path))
    monkeypatch.setattr(
        release,
        "authenticate_mps_smoke_attempt",
        lambda: ({"status": "claimed"}, "b" * 64),
    )
    with pytest.raises(ValueError, match="schema changed"):
        release._authenticate_passing_smoke()


def test_v6_passing_smoke_authenticator_recomputes_all_bound_invariants(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    terminal = tmp_path / "terminal.json"
    release_path = tmp_path / "release.json"
    loaded_path = str((tmp_path / "model.safetensors").resolve())
    release_path.write_text("{}", encoding="utf-8")
    release_sha = release.sha256_file(release_path)
    attempt_sha = "b" * 64
    b_gradients = {
        target: float(index + 1)
        for index, target in enumerate(release.TARGET_MODULES)
    }
    a_gradients = {target: 0.0 for target in release.TARGET_MODULES}
    samples = {phase: index + 1 for index, phase in enumerate(sorted(release._MPS_MEMORY_PHASES))}
    peak = max(samples.values())
    report = {
        "schema_version": 1,
        "artifact": "gemma4_v54_fixed_prefix_decoder_reader_v6_real_mps_smoke",
        "status": "passed",
        "passed": True,
        "authorization_sha256": release_sha,
        "attempt_sha256": attempt_sha,
        "device": "mps",
        "software_versions": release._EXPECTED_SOFTWARE_VERSIONS,
        "full_model_loaded": True,
        "mps_used": True,
        "optimizer_constructed": False,
        "optimizer_steps": 0,
        "training_executed": False,
        "checkpoint_published": False,
        "answer_tail_equivalence_passed": True,
        "full_vs_tail_selected_logits_exact": True,
        "full_vs_tail_selected_logits_max_abs_difference": 0.0,
        "full_vs_tail_per_token_nll_max_abs_difference": 0.0,
        "full_vs_tail_mean_nll_absolute_difference": 0.0,
        "full_vs_tail_targets_exact": True,
        "full_vs_tail_label_positions_exact": True,
        "full_vs_tail_causal_positions_exact": True,
        "v6_zero_output_exact_noop": True,
        "v6_initial_state_sha256": release.INITIAL_STATE_SHA256,
        "v6_gradient_l2": 3.0,
        "v6_gradient_by_module": {
            target: {
                "lora_a": a_gradients[target],
                "lora_b": b_gradients[target],
                "total_l2": b_gradients[target],
            }
            for target in release.TARGET_MODULES
        },
        "v6_lora_b_gradient_l2_by_target": b_gradients,
        "v6_lora_a_gradient_l2_expected_zero_by_target": a_gradients,
        "both_v6_adapter_gradients_nonzero": True,
        "contrastive_correct_nll": 1.0,
        "contrastive_wrong_nll": 1.25,
        "contrastive_margin": 0.25,
        "broad_nll": 1.5,
        "retention_self_kl": 0.0,
        "joint_zero_output_structural_runtime_coexistence_passed": True,
        "joint_nonzero_semantic_or_tool_behavior_proven": False,
        "joint_zero_output_exact_noop": True,
        "tool_numeric_projector_state_sha256": release.INITIAL_PROJECTOR_STATE_SHA256,
        "joint_state_roundtrip": {
            "reader_state_sha256": release.INITIAL_STATE_SHA256,
            "tool_state_sha256": release.TOOL_INITIAL_LORA_STATE_SHA256,
            "serialized_bytes": 10,
            "strict_state_roundtrip": True,
        },
        "scene_prefix_shape": [1, 258, 1536],
        "question_dependent_scene_retrieval": False,
        "environmental_text_inputs": [],
        "file_access_audit_active_for_entire_execution": True,
        "loaded_files": [loaded_path],
        "loaded_file_count": 1,
        "loaded_file_inventory_sha256": release._path_inventory_sha256([loaded_path]),
        "forbidden_file_accesses": [],
        "deferred_or_final_qa_accessed": False,
        "memory": {
            "peak_process_rss_bytes": 1,
            "mps_current_allocated_bytes": 1,
            "mps_driver_allocated_bytes": peak,
            "mps_driver_allocated_bytes_sampled_peak": peak,
            "mps_driver_sample_count": len(samples),
            "mps_driver_samples_by_phase": samples,
        },
        "elapsed_seconds": 1.0,
    }
    terminal.write_text(json.dumps(report), encoding="utf-8")
    monkeypatch.setattr(release, "MPS_SMOKE_REPORT", str(terminal))
    monkeypatch.setattr(release, "MPS_SMOKE_RELEASE", str(release_path))
    monkeypatch.setattr(
        release,
        "authenticate_mps_smoke_attempt",
        lambda: ({"status": "claimed"}, attempt_sha),
    )

    observed, _digest = release._authenticate_passing_smoke()
    assert observed == report
    report["loaded_files"] = [str((tmp_path / "scene_000057" / "map.npz").resolve())]
    report["loaded_file_inventory_sha256"] = release._path_inventory_sha256(
        report["loaded_files"]
    )
    terminal.write_text(json.dumps(report), encoding="utf-8")
    with pytest.raises(ValueError, match="did not pass"):
        release._authenticate_passing_smoke()


def test_v6_training_release_authentication_allows_terminal_outputs(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    release_path = tmp_path / "training_release.json"
    result_path = tmp_path / "terminal_result.json"
    checkpoint_path = tmp_path / "checkpoint"
    payload = {"artifact": "released"}
    release_path.write_text(json.dumps(payload), encoding="utf-8")
    result_path.write_text("{}", encoding="utf-8")
    checkpoint_path.mkdir()

    monkeypatch.setattr(release, "TRAINING_RELEASE", str(release_path))
    monkeypatch.setattr(release, "RESULT_REPORT", str(result_path))
    monkeypatch.setattr(release, "OUTPUT_CHECKPOINT", str(checkpoint_path))
    monkeypatch.setattr(
        release,
        "_build_training_release",
        lambda *, require_outputs_absent: (
            payload
            if require_outputs_absent is False
            else pytest.fail("authentication required outputs to be absent")
        ),
    )

    observed, digest = release.authenticate_training_release()
    assert observed == payload
    assert digest == release.sha256_file(release_path)
