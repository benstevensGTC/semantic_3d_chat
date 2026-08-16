from __future__ import annotations

import ast
import inspect
import json
import sys
from pathlib import Path
from typing import Any

import pytest

from semantic_3d_chat.evaluation import v96_final_reporting as reporting


def _digest(character: str) -> str:
    return character * 64


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _release_receipt(
    *, deferred_evidence_sha256: str, smoke_sha256: str, release_sha256: str
) -> dict[str, Any]:
    return {
        "phase": "v96_strict_runtime_release_verified",
        "passed": True,
        "candidate_fingerprint_sha256": _digest("1"),
        "candidate_attestation_file_sha256": _digest("9"),
        "candidate_attestation_identity_sha256": _digest("a"),
        "v1_implementation_seal_sha256": _digest("b"),
        "v2_implementation_seal_sha256": _digest("c"),
        "candidate_checkpoint_sha256": _digest("2"),
        "candidate_adapter_sha256": _digest("3"),
        "deferred_final_evidence_sha256": deferred_evidence_sha256,
        "runtime_smoke_sha256": smoke_sha256,
        "release_report_sha256": release_sha256,
        "release_checkpoint_sha256": _digest("4"),
        "release_adapter_sha256": _digest("5"),
        "v95_state_sha256": _digest("6"),
        "v96_state_sha256": _digest("7"),
        "runtime_implementation_inventory_sha256": _digest("8"),
        "release_implementation_inventory_sha256": _digest("d"),
        "scene_ids": list(reporting.SCENE_IDS),
        "checks": {field: True for field in reporting._REQUIRED_RELEASE_CHECKS},
    }


def _robot_evidence(release: dict[str, Any]) -> dict[str, Any]:
    return {
        "artifact": reporting.ROBOT_ARTIFACT,
        "schema_version": 96,
        "status": "authenticated_model_free_embodied_mcp_preflight_only",
        "scene_id": "scene_000001",
        "candidate_fingerprint_sha256": release["candidate_fingerprint_sha256"],
        "release_checkpoint_sha256": release["release_checkpoint_sha256"],
        "release_adapter_sha256": release["release_adapter_sha256"],
        "deferred_final_evidence_sha256": release[
            "deferred_final_evidence_sha256"
        ],
        "runtime_smoke_sha256": release["runtime_smoke_sha256"],
        "runtime_implementation_inventory_sha256": release[
            "runtime_implementation_inventory_sha256"
        ],
        "preflight_summary_sha256": _digest("9"),
        "access_audit_sha256": _digest("a"),
        "mcp_tool_count": 9,
        "strict_input_schemas": True,
        "fixed_memory_tokens": 738,
        "frozen_lora_bank_count": 10,
        "numeric_tool_outputs_only": True,
        "full_memory_recompiled_before_map_commit": True,
        "language_model_loaded": False,
        "blender_started": False,
        "robot_or_map_state_changed": False,
        "navigation_actions_executed": False,
        "navigation_success_measured": False,
        "direct_v96_answer_robot_tokens_authenticated": False,
        "forbidden_access_count": 0,
        "environmental_text_inputs": [],
        "oracle_inputs_at_runtime": False,
        "passed": True,
    }


def _live_robot_evidence(
    release: dict[str, Any], *, audit_path: Path
) -> dict[str, Any]:
    loaded_files = ["/safe/runtime/config.yaml"]
    _write_json(
        audit_path,
        {
            "loaded_files": loaded_files,
            "forbidden_roots": ["/protected/oracle"],
            "forbidden_component_names": ["oracle", "qa"],
            "block_forbidden": True,
            "forbidden_accesses": [],
            "passed": True,
        },
    )
    release_bindings = {
        field: release[field]
        for field in (
            "candidate_fingerprint_sha256",
            "deferred_final_evidence_sha256",
            "runtime_smoke_sha256",
            "release_checkpoint_sha256",
            "release_adapter_sha256",
            "v95_state_sha256",
            "v96_state_sha256",
            "runtime_implementation_inventory_sha256",
        )
    }
    hashes = {
        stage: {
            field: (
                _digest("e")
                if field == "robot_state_encoder_sha256"
                else _digest(character)
            )
            for field in reporting._LIVE_HASH_FIELDS
        }
        for stage, character in (("initial", "a"), ("scan", "b"), ("turn", "c"))
    }
    payload: dict[str, Any] = {
        "schema": "semantic_3d_chat.v96_candidate_mcp_live_smoke.v1",
        "passed": True,
        "transport": "stdio",
        "mcp_sdk_version": "1.0.0",
        "protocol_version": "2025-11-25",
        "server_name": "semantic-3d-chat",
        "server_version": "1.0.0",
        "scene_id": "scene_000025",
        "tool_count": 9,
        "tools": list(reporting._MCP_TOOLS),
        "called_tools": ["get_robot_state", "scan", "turn"],
        "base_checkpoint": "/safe/runtime/checkpoint",
        "scene_memory": "/safe/runtime/memory",
        "runtime_asset": "/safe/runtime/s_000025.blend",
        "audit_report": str(audit_path.absolute()),
        "audit_report_sha256": reporting._sha256(audit_path),
        "loaded_file_count": len(loaded_files),
        "forbidden_access_count": 0,
        "mode": "explicit_v96_candidate_overlay_after_promoted_release",
        "promoted_runtime_release_verified_before_transport": True,
        "server_reauthenticates_promoted_release_before_model_load": True,
        "deferred_final_gate_passed": True,
        "runtime_leakage_gate_passed": True,
        "numeric_tool_outputs_only": True,
        "question_free_full_memory_refresh": True,
        "full_memory_tokens": 738,
        "full_memory_recompiled_before_map_commit": True,
        "robot_state_numeric_binding_exercised": True,
        "language_questions_asked": 0,
        "v96_answer_generation_exercised": False,
        "direct_v96_answer_robot_tokens_authenticated": False,
        "environmental_text_inputs": [],
        "semantic_result_leaks": [],
        "release_bindings": release_bindings,
        "elapsed_seconds": 10.0,
        "initial_map_version": 4,
        "scan_map_version": 5,
        "turn_map_version": 6,
        "explicit_scan_valid_depth_pixels": 100,
        "turn_auto_scan_valid_depth_pixels": 101,
        "initial_source_voxels": 80,
        "scan_source_voxels": 90,
        "turn_source_voxels": 95,
        "initial_processed_voxels": 80,
        "scan_processed_voxels": 90,
        "turn_processed_voxels": 95,
        "bounded_turn_degrees": 15.0,
        "resulting_body_yaw_degrees": 15.0,
        "robot_state_encoder_identity_invariant": True,
        "initial_hashes": hashes["initial"],
        "scan_hashes": hashes["scan"],
        "turn_hashes": hashes["turn"],
    }
    payload.update({field: True for field in reporting._LIVE_CHANGED_FIELDS})
    return payload


@pytest.fixture
def authenticated_bundle(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> dict[str, Any]:
    paths = {
        "known_score": tmp_path / "aggregate" / "known_score.json",
        "known_evidence": tmp_path / "aggregate" / "known_evidence.json",
        "training": tmp_path / "aggregate" / "training.json",
        "deferred_score": tmp_path / "aggregate" / "deferred_score.json",
        "deferred_evidence": tmp_path / "aggregate" / "deferred_evidence.json",
        "smoke": tmp_path / "aggregate" / "smoke.json",
        "release": tmp_path / "aggregate" / "release.json",
        "robot": tmp_path / "aggregate" / "robot.json",
        "live_audit": tmp_path / "aggregate" / "live_access.json",
        "live_robot": tmp_path / "aggregate" / "live_robot.json",
    }
    known_score = {
        "candidate_fingerprint_sha256": _digest("1"),
        "candidate_attestation_file_sha256": _digest("9"),
        "candidate_attestation_identity_sha256": _digest("a"),
        "row_count": 216,
        "scene_count": 6,
        "structured_metrics": {
            "primary_accuracy": 0.875,
            "counterfactual_consistency": 0.75,
        },
        "nll_metrics": {"primary_mean_nll": 0.25, "zero_payload_gap": 0.5},
        "gate_results": {"accuracy": True, "causal_controls": True},
    }
    known_evidence = {
        "candidate_fingerprint_sha256": _digest("1"),
        "candidate_attestation_file_sha256": _digest("9"),
        "candidate_attestation_identity_sha256": _digest("a"),
    }
    training = {
        "status": "fixed_final_training_complete_not_promoted",
        "device": "mps",
        "model_id": "google/gemma-3-4b-it",
        "model_revision": "fixed-revision",
        "strict_input_contract": {"memory_shape": [1, 738, 1536]},
        "trainable_bridge": {"parameter_count": 2072576},
        "elapsed_seconds": 120.5,
        "optimizer_updates": 8,
        "micro_steps_consumed": 32,
        "unique_training_rows": 216,
        "training_scene_count": 6,
        "total_nll_forwards": 108,
        "protected_read_count": 0,
        "known_development_labels_loaded": False,
        "known_development_questions_loaded": False,
        "deferred_final_generated": False,
        "oracle_loaded": False,
        "runtime_promotion_authorized": False,
        "gates": {"complete": True, "protected_read_count_zero": True},
    }
    deferred_score = {
        "candidate_fingerprint_sha256": _digest("1"),
        "candidate_attestation_file_sha256": _digest("9"),
        "candidate_attestation_identity_sha256": _digest("a"),
        "v1_implementation_seal_sha256": _digest("b"),
        "v2_implementation_seal_sha256": _digest("c"),
        "row_count": 216,
        "metrics": {
            "primary": {"correct": 190, "total": 216, "accuracy": 190 / 216},
            "counterfactual": {"canonical_complete_units": 17},
            "nll": {"primary_mean_nll": 0.2, "zero_payload_mean_nll_gap": 0.4},
        },
        "gate_results": {"held_out_accuracy": True, "prefix_invariant": True},
    }
    deferred_evidence = {
        "candidate_fingerprint_sha256": _digest("1"),
        "candidate_attestation_file_sha256": _digest("9"),
        "candidate_attestation_identity_sha256": _digest("a"),
        "v1_implementation_seal_sha256": _digest("b"),
        "v2_implementation_seal_sha256": _digest("c"),
    }
    smoke = {
        "passed": True,
        "promotion_authorized": True,
        "scene_ids": list(reporting.SCENE_IDS),
        "candidate_attestation_file_sha256": _digest("9"),
        "candidate_attestation_identity_sha256": _digest("a"),
        "questions": ["question-one", "question-two"],
        "expected_answers_supplied_to_children": False,
        "behavior_assertions_in_children": False,
        "scenes": {scene_id: {"passed": True} for scene_id in reporting.SCENE_IDS},
        "gates": {"oracle_unavailable": True, "prefix_invariant": True},
    }
    for key, payload in (
        ("known_score", known_score),
        ("known_evidence", known_evidence),
        ("training", training),
        ("deferred_score", deferred_score),
        ("deferred_evidence", deferred_evidence),
        ("smoke", smoke),
    ):
        _write_json(paths[key], payload)
    release = {
        "all_release_gates_passed": True,
        "promotion_decision": "strict_v96_deferred_final_primary",
        "scene_ids": list(reporting.SCENE_IDS),
        "strict_input_contract": {"memory_shape": [1, 738, 1536]},
        "default_runtime_pointer_modified": False,
        "bindings": {
            "deferred_final_evidence_sha256": reporting._sha256(
                paths["deferred_evidence"]
            ),
            "runtime_smoke_sha256": reporting._sha256(paths["smoke"]),
            "runtime_implementation_inventory_sha256": _digest("8"),
            "release_implementation_inventory_sha256": _digest("d"),
            "candidate_attestation_file_sha256": _digest("9"),
            "candidate_attestation_identity_sha256": _digest("a"),
            "v1_implementation_seal_sha256": _digest("b"),
            "v2_implementation_seal_sha256": _digest("c"),
        },
    }
    _write_json(paths["release"], release)
    release_receipt = _release_receipt(
        deferred_evidence_sha256=reporting._sha256(paths["deferred_evidence"]),
        smoke_sha256=reporting._sha256(paths["smoke"]),
        release_sha256=reporting._sha256(paths["release"]),
    )
    known_receipt = {
        "authenticated": True,
        "known_development_gate_passed": True,
        "status": "passed_deferred_final_explicit_unlock_eligible",
        "scene_prefix_question_independent": True,
        "protected_read_count": 0,
        "row_level_content_serialized": False,
        "runtime_promotion_authorized": False,
        "final_score_sha256": reporting._sha256(paths["known_score"]),
        "evidence_sha256": reporting._sha256(paths["known_evidence"]),
        "training_report_sha256": reporting._sha256(paths["training"]),
        "candidate_attestation_file_sha256": _digest("9"),
        "candidate_attestation_identity_sha256": _digest("a"),
        "v1_implementation_seal_sha256": _digest("b"),
        "implementation_seal_sha256": _digest("c"),
    }
    deferred_receipt = {
        "authenticated": True,
        "deferred_final_gate_passed": True,
        "status": "passed_deferred_final_not_runtime_promoted",
        "question_label_isolation_proven": True,
        "prefix_hash_invariant": True,
        "protected_read_count": 0,
        "row_level_content_serialized": False,
        "runtime_promotion_authorized": False,
        "final_score_sha256": reporting._sha256(paths["deferred_score"]),
        "evidence_file_sha256": reporting._sha256(paths["deferred_evidence"]),
        "candidate_attestation_file_sha256": _digest("9"),
        "candidate_attestation_identity_sha256": _digest("a"),
        "v1_implementation_seal_sha256": _digest("b"),
        "v2_implementation_seal_sha256": _digest("c"),
    }
    _write_json(paths["robot"], _robot_evidence(release_receipt))
    _write_json(
        paths["live_robot"],
        _live_robot_evidence(release_receipt, audit_path=paths["live_audit"]),
    )

    for attribute, key in (
        ("KNOWN_SCORE", "known_score"),
        ("KNOWN_EVIDENCE", "known_evidence"),
        ("TRAINING_REPORT", "training"),
        ("DEFERRED_SCORE", "deferred_score"),
        ("DEFERRED_EVIDENCE", "deferred_evidence"),
        ("SMOKE_REPORT", "smoke"),
        ("RELEASE_REPORT", "release"),
    ):
        monkeypatch.setattr(reporting, attribute, paths[key])
    receipts = {
        "known": known_receipt,
        "deferred": deferred_receipt,
        "release": release_receipt,
    }
    monkeypatch.setattr(
        reporting,
        "_run_json_command",
        lambda kind, *, python: dict(receipts[kind]),
    )
    return {
        "paths": paths,
        "receipts": receipts,
        "release": release_receipt,
        "python": tmp_path / "unused-python",
    }


def test_collects_only_authenticated_aggregate_evidence(
    authenticated_bundle: dict[str, Any],
) -> None:
    payload = reporting.collect_v96_measured_integration(
        python=authenticated_bundle["python"],
        robot_evidence_path=authenticated_bundle["paths"]["robot"],
    )

    assert payload["status"] == "authenticated_post_release_measured_addendum"
    assert payload["known_development"]["row_count"] == 216
    assert payload["deferred_final"]["row_count"] == 216
    assert payload["runtime_release"]["scene_count"] == 6
    assert payload["runtime_leakage"]["child_process_count"] == 6
    assert payload["claims"]["held_out_deferred_final_passed"] is True
    assert payload["claims"]["embodied_navigation_success_measured"] is False
    assert payload["aggregate_only"] is True
    assert reporting._is_sha256(payload["integration_identity_sha256"])
    markdown = reporting.render_markdown(payload)
    assert "V96 navigation success measured: **no**" in markdown
    assert "does not replace `reports/final_report.md`" in markdown


@pytest.mark.parametrize(
    ("artifact", "field", "value"),
    (
        ("training", "known_development_labels_loaded", True),
        ("smoke", "expected_answers_supplied_to_children", True),
        ("release", "default_runtime_pointer_modified", True),
    ),
)
def test_refuses_mutated_evidence(
    authenticated_bundle: dict[str, Any],
    artifact: str,
    field: str,
    value: object,
) -> None:
    path = authenticated_bundle["paths"][artifact]
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload[field] = value
    _write_json(path, payload)

    with pytest.raises(ValueError, match="incomplete or inconsistent"):
        reporting.collect_v96_measured_integration(
            python=authenticated_bundle["python"],
            robot_evidence_path=authenticated_bundle["paths"]["robot"],
        )


def test_builds_exact_release_bound_model_free_robot_evidence(
    authenticated_bundle: dict[str, Any], tmp_path: Path
) -> None:
    release = authenticated_bundle["release"]
    loaded_files = ["/safe/runtime/config.yaml", "/safe/runtime/map.npz"]
    access = {
        "loaded_files": loaded_files,
        "forbidden_roots": ["/protected/oracle"],
        "forbidden_component_names": ["oracle", "qa"],
        "block_forbidden": True,
        "forbidden_accesses": [],
        "passed": True,
    }
    audit_path = tmp_path / "aggregate" / "robot_preflight_access.json"
    _write_json(audit_path, access)
    preflight = {
        "schema": "semantic_3d_chat.embodied_mcp_preflight.v1",
        "phase": "embodied_mcp_preflight",
        "passed": True,
        "mode": "semantic_continuous_map_v96_explicit_candidate",
        "scene_id": "scene_000001",
        "loads_language_model": False,
        "loads_blender": False,
        "starts_transport": False,
        "changes_robot_or_map_state": False,
        "scene_prefix_computation_deferred_to_live_startup": True,
        "environmental_text_inputs": [],
        "oracle_inputs_at_runtime": False,
        "loaded_file_count": len(loaded_files),
        "forbidden_access_count": 0,
        "audit_report": str(audit_path.absolute()),
        "action_protocol": {"tool_count": 9, "strict_input_schemas": True},
        "v96_explicit_candidate": {
            "candidate_fingerprint_sha256": release[
                "candidate_fingerprint_sha256"
            ],
            "known_development_gate_passed": True,
            "pass_evidence_authenticated": True,
            "deferred_final_gate_passed": True,
            "runtime_leakage_smoke_passed": True,
            "promoted_runtime_release_verified": True,
            "deferred_final_evidence_sha256": release[
                "deferred_final_evidence_sha256"
            ],
            "runtime_leakage_smoke_sha256": release["runtime_smoke_sha256"],
            "runtime_implementation_inventory_sha256": release[
                "runtime_implementation_inventory_sha256"
            ],
            "verified_release_checkpoint_sha256": release[
                "release_checkpoint_sha256"
            ],
            "verified_release_adapter_sha256": release["release_adapter_sha256"],
            "fixed_memory_tokens": 738,
            "frozen_lora_bank_count": 10,
            "numeric_tool_outputs_only": True,
            "full_memory_recompiled_before_map_commit": True,
            "direct_v96_answer_robot_tokens_authenticated": False,
            "environmental_text_inputs": [],
            "oracle_inputs_at_runtime": False,
        },
    }

    evidence = reporting.build_robot_preflight_evidence(
        preflight,
        access,
        access_audit_path=audit_path,
        release_receipt=release,
    )
    assert evidence["passed"] is True
    assert evidence["navigation_actions_executed"] is False
    assert evidence["navigation_success_measured"] is False
    assert (
        evidence["runtime_implementation_inventory_sha256"]
        == release["runtime_implementation_inventory_sha256"]
    )

    tampered = json.loads(json.dumps(preflight))
    tampered["v96_explicit_candidate"][
        "runtime_implementation_inventory_sha256"
    ] = _digest("f")
    with pytest.raises(ValueError, match="did not authenticate exactly"):
        reporting.build_robot_preflight_evidence(
            tampered,
            access,
            access_audit_path=audit_path,
            release_receipt=release,
        )

    second_audit = tmp_path / "aggregate" / "spliced_access.json"
    _write_json(second_audit, access)
    with pytest.raises(ValueError, match="did not authenticate exactly"):
        reporting.build_robot_preflight_evidence(
            preflight,
            access,
            access_audit_path=second_audit,
            release_receipt=release,
        )


def test_existing_robot_receipt_is_authenticated_without_rerunning_preflight(
    authenticated_bundle: dict[str, Any], capsys: pytest.CaptureFixture[str]
) -> None:
    exit_code = reporting.main(
        [
            "robot-evidence-check",
            "--python",
            str(authenticated_bundle["python"]),
            "--robot-evidence",
            str(authenticated_bundle["paths"]["robot"]),
        ]
    )
    assert exit_code == 0
    result = json.loads(capsys.readouterr().out)
    assert result["phase"] == "v96_robot_preflight_evidence_authenticated"
    assert result["navigation_success_measured"] is False

def test_optionally_authenticates_measured_numeric_live_refresh(
    authenticated_bundle: dict[str, Any],
) -> None:
    payload = reporting.collect_v96_measured_integration(
        python=authenticated_bundle["python"],
        robot_evidence_path=authenticated_bundle["paths"]["robot"],
        live_robot_evidence_path=authenticated_bundle["paths"]["live_robot"],
    )
    assert payload["claims"]["embodied_numeric_mcp_refresh_measured"] is True
    assert payload["claims"]["embodied_navigation_success_measured"] is False
    assert payload["live_robot"]["called_tools"] == [
        "get_robot_state",
        "scan",
        "turn",
    ]
    assert "Finite numeric MCP scan/turn refresh: PASS (4 -> 5 -> 6)" in (
        reporting.render_markdown(payload)
    )

    audit_path = authenticated_bundle["paths"]["live_audit"]
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    audit["forbidden_accesses"] = ["/protected/oracle/scene.json"]
    audit["passed"] = False
    _write_json(audit_path, audit)
    with pytest.raises(ValueError, match="lifetime audit"):
        reporting.collect_v96_measured_integration(
            python=authenticated_bundle["python"],
            robot_evidence_path=authenticated_bundle["paths"]["robot"],
            live_robot_evidence_path=authenticated_bundle["paths"]["live_robot"],
        )


@pytest.mark.parametrize(
    ("mutation", "match"),
    (
        ("release_binding", "did not pass exactly"),
        ("changed_binding", "did not pass exactly"),
        ("map_version", "map versions"),
        ("language_question", "did not pass exactly"),
        ("extra_field", "field inventory"),
        ("identical_hashes", "changed-hash claims"),
    ),
)
def test_live_refresh_evidence_is_exact_and_fail_closed(
    authenticated_bundle: dict[str, Any], mutation: str, match: str
) -> None:
    live_path = authenticated_bundle["paths"]["live_robot"]
    live = json.loads(live_path.read_text(encoding="utf-8"))
    if mutation == "release_binding":
        live["release_bindings"]["runtime_smoke_sha256"] = _digest("f")
    elif mutation == "changed_binding":
        live["turn_changed_scene_prefix_sha256"] = False
    elif mutation == "map_version":
        live["scan_map_version"] = live["initial_map_version"] + 2
    elif mutation == "language_question":
        live["language_questions_asked"] = 1
    elif mutation == "identical_hashes":
        live["scan_hashes"] = dict(live["initial_hashes"])
    else:
        live["unsupported_claim"] = True
    _write_json(live_path, live)

    with pytest.raises(ValueError, match=match):
        reporting.collect_v96_measured_integration(
            python=authenticated_bundle["python"],
            robot_evidence_path=authenticated_bundle["paths"]["robot"],
            live_robot_evidence_path=live_path,
        )


def test_report_pair_is_idempotent_and_refuses_partial_collision(
    authenticated_bundle: dict[str, Any], tmp_path: Path
) -> None:
    payload = reporting.collect_v96_measured_integration(
        python=authenticated_bundle["python"],
        robot_evidence_path=authenticated_bundle["paths"]["robot"],
    )
    metrics = tmp_path / "outputs" / "metrics.json"
    markdown = tmp_path / "outputs" / "report.md"
    first = reporting.build_report_outputs(
        payload, metrics_output=metrics, markdown_output=markdown
    )
    second = reporting.build_report_outputs(
        payload, metrics_output=metrics, markdown_output=markdown
    )
    assert first == second
    assert json.loads(metrics.read_text(encoding="utf-8")) == payload

    partial_metrics = tmp_path / "collision" / "metrics.json"
    conflicting_markdown = tmp_path / "collision" / "report.md"
    conflicting_markdown.parent.mkdir(parents=True)
    conflicting_markdown.write_text("different\n", encoding="utf-8")
    with pytest.raises(FileExistsError, match="differs"):
        reporting.build_report_outputs(
            payload,
            metrics_output=partial_metrics,
            markdown_output=conflicting_markdown,
        )
    assert not partial_metrics.exists()


def test_create_once_output_rolls_back_if_directory_sync_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "output" / "evidence.json"
    original_fsync = reporting.os.fsync
    call_count = 0

    def fail_directory_sync(descriptor: int) -> None:
        nonlocal call_count
        call_count += 1
        if call_count == 2:
            raise OSError("synthetic directory fsync failure")
        original_fsync(descriptor)

    monkeypatch.setattr(reporting.os, "fsync", fail_directory_sync)
    with pytest.raises(OSError, match="directory fsync"):
        reporting._write_exact(output, b"{}\n")
    assert not output.exists()
    assert not list(output.parent.glob(".*.tmp"))


def test_create_once_output_rolls_back_if_temporary_unlink_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "output" / "evidence.json"
    original_unlink = Path.unlink
    failed = False

    def fail_first_temporary_unlink(
        candidate: Path, missing_ok: bool = False
    ) -> None:
        nonlocal failed
        if candidate.name.endswith(".tmp") and not failed:
            failed = True
            raise OSError("synthetic temporary unlink failure")
        original_unlink(candidate, missing_ok=missing_ok)

    monkeypatch.setattr(Path, "unlink", fail_first_temporary_unlink)
    with pytest.raises(OSError, match="temporary unlink"):
        reporting._write_exact(output, b"{}\n")
    assert not output.exists()
    assert not list(output.parent.glob(".*.tmp"))


def test_strict_json_rejects_duplicate_fields() -> None:
    with pytest.raises(ValueError, match="Duplicate"):
        reporting._strict_json_text('{"passed":true,"passed":false}', purpose="test")


def test_release_receipt_requires_exact_sixteen_gate_inventory() -> None:
    receipt = _release_receipt(
        deferred_evidence_sha256=_digest("a"),
        smoke_sha256=_digest("b"),
        release_sha256=_digest("c"),
    )
    receipt["checks"] = {"release_report_identity": True}
    with pytest.raises(ValueError, match="incomplete"):
        reporting._validate_release_receipt(receipt)


def test_python_validator_preserves_virtual_environment_symlink(tmp_path: Path) -> None:
    executable = tmp_path / "python"
    executable.symlink_to(Path(sys.executable).resolve())
    assert reporting._python_executable(executable) == executable.absolute()


@pytest.mark.parametrize("component", ("oracle", "qa", "questions", "predictions"))
def test_reporter_refuses_protected_evidence_paths_before_open(
    tmp_path: Path, component: str
) -> None:
    protected = tmp_path / component / "evidence.json"
    _write_json(protected, {"passed": True})
    with pytest.raises(ValueError, match="protected evidence path"):
        reporting._read_json(protected, purpose="synthetic evidence")


def test_reporting_is_model_free_and_uses_isolated_authenticators() -> None:
    tree = ast.parse(inspect.getsource(reporting))
    modules = {
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module is not None
    }
    imports = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    }
    assert "torch" not in imports
    assert "transformers" not in imports
    assert not any(
        module.startswith(
            ("semantic_3d_chat.training", "semantic_3d_chat.language")
        )
        for module in modules
    )
    assert set(reporting._AUTHENTICATOR_COMMANDS) == {"known", "deferred", "release"}


def test_v96_handoff_is_explicit_and_does_not_replace_default() -> None:
    root = Path(__file__).resolve().parents[1]
    makefile = (root / "Makefile").read_text(encoding="utf-8")
    wrapper = (root / "scripts/run_v96_embodied_preflight_evidence.sh").read_text(
        encoding="utf-8"
    )
    assert "v96-report-check:" in makefile
    assert "v96-handoff-check:" in makefile
    assert "v96-handoff-demo:" in makefile
    assert "v96-handoff-demo: v96-handoff-check" in makefile
    assert "v96-report-live-check:" in makefile
    assert "v96-report-live:" in makefile
    assert "v96-handoff-live:" in makefile
    assert '--live-robot-evidence "$(GEMMA4_V96_LIVE_ROBOT_EVIDENCE)"' in makefile
    assert "v96-report-live-check: v96-report-check" in makefile
    assert (
        "v96-report-live-check: v96-explicit-candidate-embodied-mcp-live-smoke"
        not in makefile
    )
    assert "v96-demo-check" in makefile
    assert "v96_final_reporting" in wrapper
    assert "robot-evidence-check" in wrapper
    assert wrapper.index("robot-evidence-check") < wrapper.index(
        'if [[ ! -f "$V96_ROBOT_ASSET" ]]'
    )
    assert "--check" in wrapper
    assert "--allow-explicit-v96-candidate" in wrapper
    default_demo = next(
        line for line in makefile.splitlines() if line.startswith("demo:")
    )
    assert "v96" not in default_demo.casefold()
