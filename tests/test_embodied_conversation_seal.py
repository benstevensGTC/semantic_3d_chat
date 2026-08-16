from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

import semantic_3d_chat.evaluation.embodied_conversation_seal as seal


def _hash(label: str) -> str:
    return hashlib.sha256(label.encode()).hexdigest()


def _binding(version: int) -> dict[str, object]:
    scene_identity: dict[str, object] = {
        "schema": "semantic_3d_chat.scene_prefix_binding.v2",
        "scene_id": "scene_000001",
        "map_version": version,
        "map_sha256": _hash(f"map-{version}"),
        "scene_prefix_sha256": _hash(f"scene-{version}"),
        "scene_control_signature_sha256": _hash(f"control-{version}"),
        "source_voxels": 74_000 + version,
        "processed_voxels": 8_400 + version,
    }
    binding_hash = seal._canonical_sha256(scene_identity)
    active_identity = {
        **scene_identity,
        "binding_sha256": binding_hash,
        "active_prefix_sha256": _hash(f"active-{version}"),
        "robot_state_sha256": _hash(f"state-{version}"),
        "robot_tokens_sha256": _hash(f"tokens-{version}"),
        "robot_state_encoder_sha256": _hash("encoder"),
    }
    return {
        **active_identity,
        "active_binding_sha256": seal._canonical_sha256(active_identity),
    }


def _receipt(version: int, *, command: str) -> dict[str, object]:
    binding = _binding(version)
    return {
        "success": True,
        "error_code": None,
        "scene_id": "scene_000001",
        "seed": 0,
        "scene_version": version,
        "position_m": [0.0, 0.0, 0.0],
        "camera_position_m": [0.0, 0.0, 1.2],
        "body_yaw_degrees": 15.0 if command == "turn" else 0.0,
        "camera_yaw_degrees": 15.0 if command == "turn" else 0.0,
        "pitch_degrees": 0.0,
        "linear_velocity_xy_m": [0.0, 0.0],
        "angular_velocity_degrees": 0.0,
        "collision": False,
        "last_movement_delta_m": [0.0, 0.0, 0.0],
        "distance_moved": 0.0,
        "turn_degrees": 15.0 if command == "turn" else 0.0,
        "scan_coverage": 0.1 * version,
        "scan_count": version,
        "visible_voxels": 0,
        "valid_depth_pixels": 50_176,
        "observation_id": f"o_{version:06d}",
        "clearance_m": None,
        "action_count": version,
        "stopped": False,
        **binding,
    }


def _navigation(version: int, *, command: str) -> dict[str, object]:
    return {
        "kind": "navigation",
        "command": command,
        "success": True,
        "request_sha256": _hash(f"request-{command}"),
        "target_count": 0,
        "groundings": [],
        "navigation": None,
        "action_receipts": [_receipt(version, command=command)],
        "prefix_binding": _binding(version),
        "environmental_text_inputs": [],
        "question_dependent_scene_retrieval": False,
    }


def _records() -> list[dict[str, object]]:
    startup_binding = _binding(0)
    startup = {
        "phase": "embodied_conversation_ready",
        "scene_id": "scene_000001",
        "runtime": {
            "phase": "scene_ready",
            "scene_id": "scene_000001",
            "runtime_kind": "continuous_scene_question_control",
            "questions_answered": 0,
            "scene_prefix_computed_before_question": True,
            "prequestion_scene_key_value_cache": True,
            "scene_prefix_hash": startup_binding["scene_prefix_sha256"],
            "scene_control_signature_sha256": startup_binding["scene_control_signature_sha256"],
            "environmental_text_inputs": [],
            "answer_text_runtime_loaded": False,
            "answer_class_codebook_runtime_loaded": False,
            "warnings": ["numeric configuration hash differs"],
        },
        "prefix_binding": startup_binding,
        "scene_prefix_computed_before_question": True,
        "environmental_text_inputs": [],
        "local_inference": True,
        "bounded_action_protocol": True,
        "strict_fixed_environment_embedding_input": False,
        "question_conditioned_scene_readout_tokens": True,
        "llm_tool_policy": {
            "enabled": False,
            "environmental_text_inputs": [],
        },
        "navigation_policy": {
            "enabled": False,
            "oracle_inputs_at_runtime": False,
            "environmental_text_inputs": [],
        },
        "gemma_tool_decoder": {
            "enabled": False,
            "oracle_inputs_at_runtime": False,
            "environmental_text_inputs": [],
        },
        "learned_navigation_closed_loop": {"enabled": False},
    }
    turn_binding = _binding(2)
    answer = {
        "kind": "answer",
        "request_sha256": _hash("question"),
        "answer": "yes",
        "grounding_xyz_m": [0.1, 0.2, 0.3],
        "grounding_confidence": 0.7,
        "prefix_hash": turn_binding["active_prefix_sha256"],
        "prefix_binding": turn_binding,
        "environmental_text_inputs": [],
    }
    return [startup, _navigation(1, command="scan"), _navigation(2, command="turn"), answer]


def _write(path: Path, records: list[dict[str, object]]) -> bytes:
    raw = b"".join(
        json.dumps(record, sort_keys=True, allow_nan=False).encode() + b"\n" for record in records
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(raw)
    return raw


def _project_fixture(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[Path, list[dict[str, object]]]:
    monkeypatch.setattr(seal, "PROJECT_ROOT", tmp_path)
    records = _records()
    source = tmp_path / "reports" / "examples" / "embodied_live.jsonl"
    _write(source, records)
    return source, records


def test_valid_transcript_seals_exact_v0_v1_v2_answer_chain(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source, _records_value = _project_fixture(tmp_path, monkeypatch)
    raw = source.read_bytes()
    result = seal.seal_embodied_conversation_transcript(source)
    assert result["passed"] is True
    assert result["source_transcript"]["sha256"] == hashlib.sha256(raw).hexdigest()
    assert result["source_transcript"]["record_count"] == 4
    assert result["facts"]["map_versions"] == [0, 1, 2, 2]
    assert result["facts"]["scan"]["changed_map_scene_control_and_active_prefix"] is True
    assert result["facts"]["turn"]["auto_scan"] is True
    assert result["facts"]["answer"]["normalized_answer"] == "yes"
    assert result["facts"]["answer"]["uses_version_two_active_prefix"] is True
    assert result["checks"] == {
        "answer_is_yes": True,
        "answer_uses_v2_active_prefix": True,
        "no_environmental_text": True,
        "scan_changed_map_scene_and_control_cache": True,
        "scan_success_v1": True,
        "startup_v0": True,
        "turn_auto_scan_changed_all_hashes_again": True,
        "turn_positive_15_degrees_success_v2": True,
        "versions_monotonic": True,
    }


@pytest.mark.parametrize(
    ("mutation", "code"),
    [
        (lambda rows: rows[0]["prefix_binding"].__setitem__("map_version", 1), "E_VERSION"),
        (
            lambda rows: rows[1]["action_receipts"][0].__setitem__("success", False),
            "E_ACTION_FAILURE",
        ),
        (
            lambda rows: rows[2]["action_receipts"][0].__setitem__("turn_degrees", 14.0),
            "E_TURN_ANGLE",
        ),
        (
            lambda rows: rows[2]["action_receipts"][0].__setitem__("observation_id", "o_000001"),
            "E_AUTO_SCAN",
        ),
        (lambda rows: rows[3].__setitem__("answer", "no"), "E_ANSWER"),
        (
            lambda rows: rows[3].__setitem__("prefix_hash", _hash("stale")),
            "E_ANSWER_PREFIX",
        ),
    ],
)
def test_required_sequence_fails_closed_on_tampering(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation,
    code: str,
) -> None:
    source, records = _project_fixture(tmp_path, monkeypatch)
    mutation(records)
    _write(source, records)
    with pytest.raises(seal.EmbodiedConversationSealError) as captured:
        seal.seal_embodied_conversation_transcript(source)
    assert captured.value.code == code


def test_refresh_hashes_must_change_map_scene_control_and_active_prefix(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source, records = _project_fixture(tmp_path, monkeypatch)
    scan_binding = records[1]["prefix_binding"]
    startup_hash = records[0]["prefix_binding"]["scene_control_signature_sha256"]
    scan_binding["scene_control_signature_sha256"] = startup_hash
    receipt = records[1]["action_receipts"][0]
    receipt["scene_control_signature_sha256"] = startup_hash
    # Recompute both authenticated binding identities after the malicious edit.
    scene_identity = {
        key: scan_binding[key]
        for key in seal._BINDING_KEYS
        if key
        not in {
            "binding_sha256",
            "active_prefix_sha256",
            "robot_state_sha256",
            "robot_tokens_sha256",
            "robot_state_encoder_sha256",
            "active_binding_sha256",
        }
    }
    scan_binding["binding_sha256"] = seal._canonical_sha256(scene_identity)
    active_identity = {
        key: scan_binding[key] for key in seal._BINDING_KEYS if key != "active_binding_sha256"
    }
    scan_binding["active_binding_sha256"] = seal._canonical_sha256(active_identity)
    for key in seal._BINDING_KEYS:
        receipt[key] = scan_binding[key]
    _write(source, records)
    with pytest.raises(seal.EmbodiedConversationSealError) as captured:
        seal.seal_embodied_conversation_transcript(source)
    assert captured.value.code == "E_REFRESH_UNCHANGED"


@pytest.mark.parametrize(
    "mutation",
    [
        lambda rows: rows[2].__setitem__("environmental_text_inputs", ["hidden room text"]),
        lambda rows: rows[0]["runtime"].__setitem__("scene_caption", "hidden room text"),
        lambda rows: rows[0]["runtime"].__setitem__("answer_text_runtime_loaded", True),
        lambda rows: rows[0]["navigation_policy"].__setitem__("oracle_inputs_at_runtime", True),
    ],
)
def test_environmental_text_and_oracle_leakage_fail_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mutation
) -> None:
    source, records = _project_fixture(tmp_path, monkeypatch)
    mutation(records)
    _write(source, records)
    with pytest.raises(seal.EmbodiedConversationSealError):
        seal.seal_embodied_conversation_transcript(source)


def test_extra_record_duplicate_key_and_nonfinite_json_are_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source, records = _project_fixture(tmp_path, monkeypatch)
    with source.open("ab") as handle:
        handle.write(json.dumps(records[-1]).encode() + b"\n")
    with pytest.raises(seal.EmbodiedConversationSealError) as captured:
        seal.seal_embodied_conversation_transcript(source)
    assert captured.value.code == "E_RECORD_COUNT"

    source.write_text('{"a":1,"a":2}\n{}\n{}\n{}\n', encoding="utf-8")
    with pytest.raises(seal.EmbodiedConversationSealError) as captured:
        seal.seal_embodied_conversation_transcript(source)
    assert captured.value.code == "E_DUPLICATE_JSON_KEY"

    source.write_text('{"a":NaN}\n{}\n{}\n{}\n', encoding="utf-8")
    with pytest.raises(seal.EmbodiedConversationSealError) as captured:
        seal.seal_embodied_conversation_transcript(source)
    assert captured.value.code == "E_NONFINITE_JSON"


def test_successful_seal_is_atomic_immutable_and_does_not_embed_transcript(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source, _records_value = _project_fixture(tmp_path, monkeypatch)
    result = seal.seal_embodied_conversation_transcript(source)
    destination = tmp_path / "reports" / "metrics" / "embodied_seal.json"
    written = seal.write_embodied_conversation_seal(destination, result)
    assert written == destination
    payload = json.loads(destination.read_text())
    assert payload == result
    serialized = destination.read_text()
    assert "grounding_xyz_m" not in serialized
    assert "action_receipts" not in serialized
    with pytest.raises(seal.EmbodiedConversationSealError) as captured:
        seal.write_embodied_conversation_seal(destination, result)
    assert captured.value.code == "E_OUTPUT_EXISTS"


def test_cli_failure_prints_nonseal_summary_and_creates_no_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    source, records = _project_fixture(tmp_path, monkeypatch)
    records[3]["answer"] = "unknown"
    raw = _write(source, records)
    destination = tmp_path / "reports" / "metrics" / "should_not_exist.json"
    code = seal.main(["--transcript", str(source), "--output", str(destination)])
    assert code == 2
    assert not destination.exists()
    output = json.loads(capsys.readouterr().out)
    assert output["passed"] is False
    assert output["sealed"] is False
    assert output["error_code"] == "E_ANSWER"
    assert output["source_transcript"]["sha256"] == hashlib.sha256(raw).hexdigest()


def test_forbidden_and_symlink_transcript_paths_are_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source, _records_value = _project_fixture(tmp_path, monkeypatch)
    link = tmp_path / "reports" / "examples" / "linked.jsonl"
    link.symlink_to(source)
    with pytest.raises(seal.EmbodiedConversationSealError) as captured:
        seal.seal_embodied_conversation_transcript(link)
    assert captured.value.code == "E_SYMLINK"

    forbidden = tmp_path / "data" / "oracle" / "conversation.jsonl"
    forbidden.parent.mkdir(parents=True)
    forbidden.write_bytes(source.read_bytes())
    with pytest.raises(seal.EmbodiedConversationSealError) as captured:
        seal.seal_embodied_conversation_transcript(forbidden)
    assert captured.value.code == "E_FORBIDDEN_PATH"
    failure = seal.failure_summary(forbidden, captured.value)
    assert failure["source_transcript"]["sha256"] is None
    assert failure["source_transcript"]["size_bytes"] is None
