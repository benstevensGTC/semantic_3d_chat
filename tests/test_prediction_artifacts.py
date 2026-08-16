from __future__ import annotations

import json
from pathlib import Path

import pytest

from semantic_3d_chat.evaluation.prediction_artifacts import (
    AtomicPredictionJournal,
    build_prediction_provenance,
    checkpoint_fingerprint,
    effective_config_sha256,
    provenance_path_for,
)


def _inputs(tmp_path: Path):
    config_path = tmp_path / "experiment.yaml"
    config_path.write_text("seed: 7\n", encoding="utf-8")
    references = tmp_path / "questions.jsonl"
    references.write_text(
        json.dumps({"scene_id": "scene_000001", "question_id": "q_001"}) + "\n",
        encoding="utf-8",
    )
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "adapter.safetensors").write_bytes(b"adapter-v1")
    (checkpoint / "metadata.json").write_text('{"epoch": 2}\n', encoding="utf-8")
    (checkpoint / "runtime_metadata.json").write_text(
        '{"schema_version": 3}\n', encoding="utf-8"
    )
    (checkpoint / "optimizer.pt").write_bytes(b"must-not-affect-inference-hash")
    maps_root = tmp_path / "maps"
    map_path = maps_root / "scene_000001" / "voxel_map.npz"
    map_path.parent.mkdir(parents=True)
    map_path.write_bytes(b"continuous-map-v1")
    config = {
        "seed": 7,
        "nested": {"latent_count": 256},
        "paths": {"data_root": str(tmp_path / "data"), "maps_root": str(maps_root)},
        "_config_path": str(config_path),
    }
    provenance = build_prediction_provenance(
        config,
        config_path=config_path,
        checkpoint_path=checkpoint,
        references_path=references,
        scene_ids=["scene_000001"],
        split="test",
        run_kind="primary_continuous_3d",
    )
    return config, config_path, references, checkpoint, provenance


def test_provenance_hashes_effective_config_checkpoint_and_references(tmp_path: Path) -> None:
    config, config_path, references, checkpoint, provenance = _inputs(tmp_path)
    assert len(provenance.config_sha256) == 64
    assert len(provenance.config_file_sha256) == 64
    assert len(provenance.checkpoint_sha256) == 64
    assert len(provenance.references_sha256) == 64
    assert len(provenance.sha256) == 64
    assert {item["path"] for item in provenance.checkpoint_files} == {
        "adapter.safetensors",
        "metadata.json",
        "runtime_metadata.json",
    }
    assert effective_config_sha256(config) == effective_config_sha256(
        {
            "nested": {"latent_count": 256},
            "paths": config["paths"],
            "seed": 7,
            "_another_internal": True,
        }
    )
    assert set(provenance.scene_map_manifest) == {"scene_000001"}
    assert len(provenance.scene_map_manifest_sha256) == 64

    first, _ = checkpoint_fingerprint(checkpoint)
    (checkpoint / "optimizer.pt").write_bytes(b"changed optimizer")
    assert checkpoint_fingerprint(checkpoint)[0] == first
    (checkpoint / "runtime_metadata.json").write_text(
        '{"schema_version": 3, "contract": "changed"}\n', encoding="utf-8"
    )
    assert checkpoint_fingerprint(checkpoint)[0] != first
    (checkpoint / "runtime_metadata.json").write_text(
        '{"schema_version": 3}\n', encoding="utf-8"
    )
    assert checkpoint_fingerprint(checkpoint)[0] == first
    (checkpoint / "adapter.safetensors").write_bytes(b"adapter-v2")
    assert checkpoint_fingerprint(checkpoint)[0] != first

    # Both the literal selected config layer and merged settings are recorded.
    config_path.write_text("seed: 8\n", encoding="utf-8")
    changed = build_prediction_provenance(
        config,
        config_path=config_path,
        checkpoint_path=checkpoint,
        references_path=references,
        scene_ids=["scene_000001"],
        split="test",
        run_kind="primary_continuous_3d",
    )
    assert changed.config_file_sha256 != provenance.config_file_sha256


def test_journal_atomically_checkpoints_each_question_and_resumes(tmp_path: Path) -> None:
    *_unused, provenance = _inputs(tmp_path)
    output = tmp_path / "predictions.jsonl"
    journal = AtomicPredictionJournal(output, provenance)
    first = {
        "scene_id": "scene_000001",
        "question_id": "q_001",
        "predicted_answer": "red",
    }
    assert journal.append(first) is True
    assert output.is_file()
    assert not list(tmp_path.glob(".*.tmp"))
    persisted = json.loads(output.read_text(encoding="utf-8"))
    assert persisted["provenance_sha256"] == provenance.sha256
    sidecar = json.loads(provenance_path_for(output).read_text(encoding="utf-8"))
    assert sidecar["config_sha256"] == provenance.config_sha256
    assert sidecar["checkpoint_sha256"] == provenance.checkpoint_sha256
    assert sidecar["references_sha256"] == provenance.references_sha256
    assert sidecar["scene_map_manifest"] == provenance.scene_map_manifest
    assert (
        sidecar["scene_map_manifest_sha256"]
        == provenance.scene_map_manifest_sha256
    )

    resumed = AtomicPredictionJournal(output, provenance, resume=True)
    assert resumed.contains("scene_000001", "q_001")
    assert resumed.append(first) is False
    assert resumed.append(
        {
            "scene_id": "scene_000001",
            "question_id": "q_002",
            "predicted_answer": "blue",
        }
    )
    assert len(output.read_text(encoding="utf-8").splitlines()) == 2


def test_journal_refuses_stale_or_unprovenanced_resume(tmp_path: Path) -> None:
    config, config_path, references, checkpoint, provenance = _inputs(tmp_path)
    output = tmp_path / "predictions.jsonl"
    journal = AtomicPredictionJournal(output, provenance)
    journal.append({"scene_id": "scene_000001", "question_id": "q_001"})

    references.write_text('{"scene_id":"scene_000002","question_id":"q_002"}\n')
    changed = build_prediction_provenance(
        config,
        config_path=config_path,
        checkpoint_path=checkpoint,
        references_path=references,
        scene_ids=["scene_000001"],
        split="test",
        run_kind="primary_continuous_3d",
    )
    with pytest.raises(RuntimeError, match="provenance mismatch"):
        AtomicPredictionJournal(output, changed, resume=True)

    provenance_path_for(output).unlink()
    with pytest.raises(RuntimeError, match="missing provenance sidecar"):
        AtomicPredictionJournal(output, provenance, resume=True)


def test_journal_refuses_resume_after_scene_map_bytes_change(tmp_path: Path) -> None:
    config, config_path, references, checkpoint, provenance = _inputs(tmp_path)
    output = tmp_path / "predictions.jsonl"
    journal = AtomicPredictionJournal(output, provenance)
    journal.append({"scene_id": "scene_000001", "question_id": "q_001"})
    map_path = Path(config["paths"]["maps_root"]) / "scene_000001" / "voxel_map.npz"
    map_path.write_bytes(b"continuous-map-v2")

    changed = build_prediction_provenance(
        config,
        config_path=config_path,
        checkpoint_path=checkpoint,
        references_path=references,
        scene_ids=["scene_000001"],
        split="test",
        run_kind="primary_continuous_3d",
    )

    assert changed.scene_map_manifest_sha256 != provenance.scene_map_manifest_sha256
    with pytest.raises(RuntimeError, match="provenance mismatch"):
        AtomicPredictionJournal(output, changed, resume=True)


def test_checkpoint_fingerprint_requires_sanitized_runtime_metadata(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "adapter.safetensors").write_bytes(b"adapter")

    with pytest.raises(FileNotFoundError, match="missing runtime metadata"):
        checkpoint_fingerprint(checkpoint)


def test_failed_atomic_replace_preserves_previous_question_checkpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    *_unused, provenance = _inputs(tmp_path)
    output = tmp_path / "predictions.jsonl"
    journal = AtomicPredictionJournal(output, provenance)
    journal.append({"scene_id": "scene_000001", "question_id": "q_001"})
    before = output.read_bytes()

    def fail_replace(_source, _destination) -> None:
        raise OSError("simulated interruption")

    monkeypatch.setattr("semantic_3d_chat.evaluation.baseline_io.os.replace", fail_replace)
    with pytest.raises(OSError, match="simulated interruption"):
        journal.append({"scene_id": "scene_000001", "question_id": "q_002"})
    assert output.read_bytes() == before
    assert not journal.contains("scene_000001", "q_002")
