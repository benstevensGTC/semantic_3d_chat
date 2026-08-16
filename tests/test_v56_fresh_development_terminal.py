from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from semantic_3d_chat.evaluation import v56_fresh_development_terminal as terminal
from semantic_3d_chat.evaluation.question_manifest import build_question_manifest


def _write_control_checkpoint(path: Path) -> dict[str, object]:
    path.mkdir()
    weights = path / "control.safetensors"
    weights.write_bytes(b"synthetic-continuous-control")
    metadata: dict[str, object] = {
        "schema_version": 1,
        "architecture": "full_scene_question_control_v1",
        "hidden_size": 1536,
        "attention_dim": 256,
        "control_tokens": 4,
        "uniform_floor": 0.05,
        "output_scale": 0.25,
        "weights_sha256": hashlib.sha256(weights.read_bytes()).hexdigest(),
        "base_checkpoint_sha256": terminal.V54_CHECKPOINT_SHA256,
        "base_runtime_config_sha256": terminal.RUNTIME_CONFIG_EFFECTIVE_SHA256,
        "question_dependent_scene_retrieval": False,
        "complete_scene_prefix_required": True,
        "environmental_text_inputs": [],
    }
    (path / "runtime_metadata.json").write_text(
        json.dumps(metadata), encoding="utf-8"
    )
    return metadata


def _training_report(
    path: Path,
    control_path: Path,
    metadata: dict[str, object],
) -> None:
    payload = {
        "schema_version": 1,
        "artifact": "v56_question_control_training",
        "passed": True,
        "base": {
            "checkpoint_sha256": terminal.V54_CHECKPOINT_SHA256,
            "checkpoint_files": [
                {
                    "path": name,
                    "sha256": digest,
                    "size_bytes": 1,
                }
                for name, digest in terminal.V54_CHECKPOINT_FILES.items()
            ],
            "runtime_config_effective_sha256": (
                terminal.RUNTIME_CONFIG_EFFECTIVE_SHA256
            ),
            "runtime_config_file_sha256": terminal.RUNTIME_CONFIG_FILE_SHA256,
        },
        "inputs": {
            "training_qa_sha256": "3" * 64,
            "training_record_count": terminal.EXPECTED_TRAIN_RECORD_COUNT,
            "training_scene_ids": list(terminal.EXPECTED_TRAIN_SCENES),
            "prefix_cache_manifest_sha256": "4" * 64,
            "prefix_sha256_by_scene": {
                scene_id: hashlib.sha256(scene_id.encode()).hexdigest()
                for scene_id in terminal.EXPECTED_TRAIN_SCENES
            },
            "prefix_cache_created": True,
        },
        "curriculum": {
            "step_count": 3,
            "steps_by_kind": {
                "broad": 1,
                "changed_pair": 1,
                "count_replay": 1,
            },
            "changed_pair_unit_count": 1,
            "schedule_sha256": "5" * 64,
            "paired_two_side_optimizer_steps": True,
        },
        "architecture": {
            "name": metadata["architecture"],
            "hidden_size": metadata["hidden_size"],
            "attention_dim": metadata["attention_dim"],
            "control_tokens": metadata["control_tokens"],
            "uniform_floor": metadata["uniform_floor"],
            "output_scale": metadata["output_scale"],
            "parameter_count": 1234,
        },
        "optimization": {
            "seed": 56056,
            "epochs": 1,
            "learning_rate": 0.001,
            "weight_decay": 0.0,
            "gradient_clip_norm": 1.0,
            "optimizer_steps": 3,
            "device": "cpu",
            "elapsed_seconds": 1.0,
            "epoch_loss": [
                {
                    "epoch": 0,
                    "steps": 3,
                    "mean_answer_ce": 1.0,
                    "minimum_answer_ce": 0.5,
                    "maximum_answer_ce": 1.5,
                }
            ],
            "maximum_preclip_gradient_norm": 1.0,
        },
        "checkpoint": {
            "weights_sha256": hashlib.sha256(
                (control_path / "control.safetensors").read_bytes()
            ).hexdigest(),
            "runtime_metadata_sha256": hashlib.sha256(
                (control_path / "runtime_metadata.json").read_bytes()
            ).hexdigest(),
        },
        "scope": {
            "base_scene_stack_frozen": True,
            "base_parameter_count": 1000,
            "only_control_head_optimized": True,
            "answer_only_cross_entropy": True,
            "paired_two_side_optimizer_steps": True,
            "question_inputs_to_scene_prefix_cache": False,
            "question_dependent_scene_retrieval": False,
            "oracle_loaded": False,
            "fresh_development_loaded": False,
            "deferred_final_loaded": False,
            "optimizer_state_saved": False,
        },
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_control_and_training_evidence_are_exactly_bound(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(terminal, "PROJECT_ROOT", tmp_path)
    base_path = tmp_path / "base"
    base_path.mkdir()
    for name in terminal.V54_CHECKPOINT_FILES:
        (base_path / name).write_bytes(b"x")
    monkeypatch.setattr(terminal, "V54_CHECKPOINT", base_path)
    control_path = tmp_path / "control"
    metadata = _write_control_checkpoint(control_path)
    report_path = tmp_path / "training_report.json"
    _training_report(report_path, control_path, metadata)
    expected_parameter_count = 1234
    monkeypatch.setattr(
        terminal,
        "_load_control_head",
        lambda *_, **__: (
            type("Control", (), {"parameter_count": expected_parameter_count})(),
            metadata,
        ),
    )

    control = terminal._control_checkpoint_identity(control_path)
    training = terminal._training_report_identity(report_path, control)

    assert control["path"] == "control"
    assert len(str(control["sha256"])) == 64
    assert control["parameter_count"] == expected_parameter_count
    assert set(control["files"]) == {
        "control.safetensors",
        "runtime_metadata.json",
    }
    assert training["training_record_count"] == 960
    assert training["training_scene_ids"] == list(terminal.EXPECTED_TRAIN_SCENES)
    assert training["training_qa_sha256"] == "3" * 64
    assert training["prefix_cache_manifest_sha256"] == "4" * 64
    assert training["scope"]["base_parameter_count"] == 1000
    assert all(
        scene_id not in training["training_scene_ids"]
        for scene_id in terminal.EXPECTED_SCENE_IDS
    )


def test_question_identity_requires_complete_fresh_questions_only_manifest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "questions.json"
    rows = [
        {
            "scene_id": terminal.EXPECTED_SCENE_IDS[index % 6],
            "question_id": f"q_{index + 1:06d}",
            "question": f"Synthetic question {index + 1}?",
        }
        for index in range(216)
    ]
    manifest = build_question_manifest(rows, source_qa_sha256="a" * 64)
    path.write_text(json.dumps(manifest.as_dict()), encoding="utf-8")
    monkeypatch.setattr(terminal, "QUESTIONS_PATH", path)

    identity = terminal._question_identity(path)

    assert identity["question_count"] == 216
    assert identity["scene_count"] == 6
    assert identity["reference_sha256"] == "a" * 64
    serialized = json.dumps(identity)
    assert "Synthetic question" not in serialized


def _redirect_outputs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    replacements = {
        "DEFAULT_OUTPUT": tmp_path / "terminal.json",
        "CLAIM_PATH": tmp_path / "claim.json",
        "MODEL_SNAPSHOT_PATH": tmp_path / "model_snapshot.json",
        "PREDICTIONS_PATH": tmp_path / "predictions.jsonl",
        "PREDICTION_PROVENANCE_PATH": tmp_path / "predictions.provenance.json",
        "SCORE_PATH": tmp_path / "score.json",
        "SELECTOR_REPORT_PATH": tmp_path / "selector.json",
    }
    for name, path in replacements.items():
        monkeypatch.setattr(terminal, name, path)


def test_terminal_payload_precommits_fresh_scope_without_answer_access(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _redirect_outputs(tmp_path, monkeypatch)
    events: list[str] = []
    control = {
        "path": "control",
        "sha256": "b" * 64,
        "files": {
            "control.safetensors": {"sha256": "c" * 64, "size_bytes": 1},
            "runtime_metadata.json": {"sha256": "d" * 64, "size_bytes": 1},
        },
        "runtime_metadata": {},
    }
    questions = {
        "path": str(terminal.QUESTIONS_PATH),
        "manifest_sha256": "e" * 64,
        "questions_sha256": "f" * 64,
        "reference_sha256": "1" * 64,
        "question_count": 216,
        "scene_count": 6,
    }
    maps = {
        scene_id: {
            "voxel_map_sha256": hashlib.sha256(scene_id.encode()).hexdigest(),
            "voxel_map_size_bytes": 1,
        }
        for scene_id in terminal.EXPECTED_SCENE_IDS
    }
    monkeypatch.setattr(
        terminal,
        "_authenticate_static_predecessors",
        lambda: {"passed": True},
    )
    monkeypatch.setattr(terminal, "load_runtime_config", lambda _: {"safe": True})
    monkeypatch.setattr(
        terminal,
        "runtime_config_file_sha256",
        lambda _: terminal.RUNTIME_CONFIG_FILE_SHA256,
    )
    monkeypatch.setattr(terminal, "_control_checkpoint_identity", lambda _: control)
    monkeypatch.setattr(
        terminal,
        "_training_report_identity",
        lambda *_: {"path": "training", "sha256": "2" * 64},
    )
    monkeypatch.setattr(terminal, "_question_identity", lambda _: questions)

    def map_manifest(_config: object) -> dict[str, object]:
        events.append("numeric_map_hashes")
        return maps

    monkeypatch.setattr(terminal, "_scene_map_identity", map_manifest)
    monkeypatch.setattr(
        terminal,
        "local_model_snapshot_identity",
        lambda _: events.append("model_snapshot_hashes")
        or {"tree_sha256": "3" * 64, "files": []},
    )
    monkeypatch.setattr(terminal, "_bound_source_hashes", lambda: {"source": "4" * 64})

    payload = terminal.build_terminal_payload(
        control_checkpoint="control",
        training_report="training",
    )

    authorization = payload["authorization"]
    assert authorization["development"]["scene_ids"] == list(
        terminal.EXPECTED_SCENE_IDS
    )
    assert authorization["development"]["atomic_pair_count"] == 3
    assert authorization["thresholds"]["normalized_exact_accuracy_minimum"] == 0.42
    assert authorization["scope"]["deferred_final_access_authorized"] is False
    assert events == ["numeric_map_hashes", "model_snapshot_hashes"]


def test_prior_claim_closes_terminal_before_any_input_authentication(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _redirect_outputs(tmp_path, monkeypatch)
    Path(terminal.CLAIM_PATH).write_text("{}\n", encoding="utf-8")
    events: list[str] = []
    monkeypatch.setattr(
        terminal,
        "_authenticate_static_predecessors",
        lambda: events.append("predecessors") or {},
    )

    with pytest.raises(FileExistsError, match="one-shot output exists"):
        terminal.build_terminal_payload(
            control_checkpoint="control",
            training_report="training",
        )

    assert events == []


def test_terminal_seal_is_create_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _redirect_outputs(tmp_path, monkeypatch)
    monkeypatch.setattr(
        terminal,
        "build_terminal_payload",
        lambda **_: {
            "schema_version": 1,
            "artifact": terminal.ARTIFACT,
            "passed": True,
        },
    )

    sealed = terminal.seal_terminal(
        control_checkpoint="control",
        training_report="training",
        output=terminal.DEFAULT_OUTPUT,
    )

    assert len(sealed["sha256"]) == 64
    with pytest.raises(FileExistsError, match="immutable"):
        terminal.seal_terminal(
            control_checkpoint="control",
            training_report="training",
            output=terminal.DEFAULT_OUTPUT,
        )
