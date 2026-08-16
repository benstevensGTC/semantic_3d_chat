from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
from starlette.testclient import TestClient

import semantic_3d_chat.chat.strict_prefix_web as strict_web
from semantic_3d_chat.chat.file_audit import FileAccessAudit


@dataclass(frozen=True)
class FakeAnswer:
    question: str
    prefix_hash: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "question": self.question,
            "answer": "continuous fixed-prefix answer",
            "grounding_xyz_m": [0.2, -0.4, 0.8],
            "grounding_confidence": 0.75,
            "grounding_support_distance_m": 0.05,
            "prefix_hash": self.prefix_hash,
            "generated_tokens": 3,
            "elapsed_seconds": 0.01,
        }


class FakeRuntime:
    scene_id = "scene_000001"
    scene_prefix_hash = "7" * 64

    def __init__(self) -> None:
        self.questions: list[str] = []
        self.prefix_checks = 0

    @property
    def questions_answered(self) -> int:
        return len(self.questions)

    def current_prefix_hash(self) -> str:
        return self.scene_prefix_hash

    def assert_prefix_unchanged(self) -> None:
        self.prefix_checks += 1
        assert self.current_prefix_hash() == self.scene_prefix_hash

    def answer(self, question: str) -> FakeAnswer:
        self.questions.append(question)
        return FakeAnswer(question=question, prefix_hash=self.scene_prefix_hash)

    def startup_summary(self) -> dict[str, Any]:
        return {
            "scene_id": self.scene_id,
            "prefix_hash": self.scene_prefix_hash,
            "prefix_shape": [1, 258, 1536],
            "scene_latents": 256,
            "language_hidden_dim": 1536,
            "source_voxels": 74_699,
            "processed_voxels": 74_699,
            "occupied_blocks": 3_019,
            "device": "test",
            "prefix_build_seconds": 1.0,
            "scene_prefix_computed_before_question": True,
            "strict_fixed_environment_embedding_input": True,
            "environment_conditioned_input_sha256": self.scene_prefix_hash,
            "question_conditioned_scene_readout_tokens": False,
            "question_dependent_scene_retrieval": False,
        }


def _config(tmp_path: Path) -> dict[str, Any]:
    return {
        "paths": {
            "data_root": str(tmp_path / "data"),
            "reports_root": str(tmp_path / "reports/gemma4"),
            "maps_root": str(tmp_path / "data_gemma4/maps"),
            "checkpoints_root": str(tmp_path / "data_gemma4/checkpoints"),
        },
        "render": {
            "camera_position_m": [0.0, 0.0, 1.4],
            "yaw_degrees": [0.0],
            "pitch_degrees": [0.0],
        },
        "vision": {"backend": "gemma4"},
        "language": {"backend": "gemma4"},
        "scene_encoder": {"global_latents": 256},
    }


def _write_required_artifacts(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    config_path = tmp_path / "gemma4_v54.yaml"
    config_path.write_text("runtime: synthetic\n", encoding="utf-8")
    checkpoint = tmp_path / "data_gemma4/checkpoints/v54/update_000"
    checkpoint.mkdir(parents=True)
    (checkpoint / "adapter.safetensors").write_bytes(b"synthetic-adapter")
    (checkpoint / "runtime_metadata.json").write_text("{}\n", encoding="utf-8")
    map_path = tmp_path / "data_gemma4/maps/scene_000001/voxel_map.npz"
    map_path.parent.mkdir(parents=True)
    map_path.write_bytes(b"synthetic-map")
    preview = tmp_path / "reports/gemma4/figures/scene_000001/map_rgb.png"
    preview.parent.mkdir(parents=True)
    preview.write_bytes(b"synthetic-raster")
    montage = tmp_path / "reports/gemma4/figures/scan_montage.png"
    montage.parent.mkdir(parents=True, exist_ok=True)
    montage.write_bytes(b"synthetic-scan-montage")
    return config_path, checkpoint, map_path, preview


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _use_synthetic_checkpoint_hashes(monkeypatch: pytest.MonkeyPatch, checkpoint: Path) -> None:
    monkeypatch.setattr(
        strict_web,
        "_V54_ADAPTER_SHA256",
        _sha256(checkpoint / "adapter.safetensors"),
    )
    monkeypatch.setattr(
        strict_web,
        "_V54_RUNTIME_METADATA_SHA256",
        _sha256(checkpoint / "runtime_metadata.json"),
    )


def test_strict_web_reuses_exact_environment_hash_and_keeps_visual_human_only(
    tmp_path: Path,
) -> None:
    _write_required_artifacts(tmp_path)
    runtime = FakeRuntime()
    app = strict_web.create_strict_web_app(
        runtime,
        _config(tmp_path),
        project_root=tmp_path,
    )

    with TestClient(app) as client:
        state = client.get("/api/state")
        first = client.post("/api/chat", json={"question": "Is there a chair?"})
        second = client.post("/api/chat", json={"question": "Where is the bowl?"})

    assert state.status_code == first.status_code == second.status_code == 200
    assert state.json()["visuals"] == {
        "map": "/assets/map",
        "overview": "/assets/overview",
    }
    assert state.json()["prefix_built_before_questions"] is True
    assert state.json()["strict_fixed_environment_embedding_input"] is True
    assert state.json()["question_conditioned_scene_readout_tokens"] is False
    assert state.json()["question_dependent_retrieval"] is False
    assert state.json()["human_visuals_are_model_inputs"] is False
    hashes = {
        state.json()["environment_conditioned_input_sha256"],
        first.json()["environment_conditioned_input_sha256"],
        second.json()["environment_conditioned_input_sha256"],
    }
    assert hashes == {runtime.scene_prefix_hash}
    assert runtime.questions == ["Is there a chair?", "Where is the bowl?"]
    assert runtime.prefix_checks >= 6


def test_strict_web_denylist_blocks_every_environmental_text_tree(tmp_path: Path) -> None:
    config = _config(tmp_path)
    blocked = [
        tmp_path / "data/oracle/secret.json",
        tmp_path / "data/qa/train.jsonl",
        tmp_path / "data/rendered/frame.png",
        tmp_path / "data/features/frame.npz",
        tmp_path / "data_gemma4/training/cache.pt",
        tmp_path / "reports/gemma4/scorer_only/references.json",
    ]
    for path in blocked:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("forbidden", encoding="utf-8")
    audit = FileAccessAudit(
        strict_web.strict_forbidden_roots(tmp_path, config),
        forbidden_component_names={
            "oracle",
            "qa",
            "rendered",
            "features",
            "scorer_only",
        },
        block_forbidden=True,
    )

    with audit:
        for path in blocked:
            with pytest.raises(PermissionError, match="Blocked forbidden runtime file read"):
                path.read_bytes()

    assert set(audit.forbidden_accesses()) == {str(path.resolve()) for path in blocked}


def test_strict_web_launcher_builds_prefix_before_serving_without_loading_gemma(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config_path, checkpoint, _map_path, _preview = _write_required_artifacts(tmp_path)
    config = _config(tmp_path)
    _use_synthetic_checkpoint_hashes(monkeypatch, checkpoint)
    runtime = FakeRuntime()
    audit_path = tmp_path / "reports/gemma4/metrics/strict_web_test_audit.json"
    events: list[str] = []

    def fake_load_config(path: Path, audit: FileAccessAudit) -> dict[str, Any]:
        assert path == config_path.resolve()
        audit.record(path)
        events.append("config")
        return config

    def fake_load_runtime(
        loaded_config: dict[str, Any],
        scene_id: str,
        loaded_checkpoint: Path,
        audit: FileAccessAudit,
    ) -> FakeRuntime:
        assert loaded_config is config
        assert scene_id == runtime.scene_id
        assert loaded_checkpoint == checkpoint.resolve()
        assert audit.active is True
        events.append("prefix")
        return runtime

    def fake_serve(app: Any, host: str, port: int) -> None:
        assert events == ["config", "prefix"]
        assert host == "127.0.0.1"
        assert port == 9876
        events.append("serve")
        with TestClient(app) as client:
            state = client.get("/api/state").json()
            one = client.post("/api/chat", json={"question": "Question one?"}).json()
            two = client.post("/api/chat", json={"question": "Question two?"}).json()
            image = client.get("/assets/map")
            montage = client.get("/assets/overview")
        assert image.status_code == 200
        assert montage.status_code == 200
        assert state["visuals"] == {
            "map": "/assets/map",
            "overview": "/assets/overview",
        }
        assert {
            state["environment_conditioned_input_sha256"],
            one["environment_conditioned_input_sha256"],
            two["environment_conditioned_input_sha256"],
        } == {runtime.scene_prefix_hash}

    monkeypatch.setattr(strict_web, "_load_runtime_config", fake_load_config)
    monkeypatch.setattr(strict_web, "_load_static_runtime", fake_load_runtime)
    monkeypatch.setattr(strict_web, "_serve", fake_serve)

    result = strict_web.main(
        [
            "--config",
            str(config_path),
            "--scene",
            runtime.scene_id,
            "--checkpoint",
            str(checkpoint),
            "--host",
            "127.0.0.1",
            "--port",
            "9876",
            "--audit-log",
            str(audit_path),
        ]
    )

    assert result == 0
    assert events == ["config", "prefix", "serve"]
    assert runtime.questions == ["Question one?", "Question two?"]
    report = json.loads(audit_path.read_text(encoding="utf-8"))
    assert report["passed"] is True
    assert report["forbidden_accesses"] == []
    assert str((checkpoint / "metadata.json").resolve()) in report["forbidden_roots"]
    assert str((checkpoint / "metadata.json").resolve()) not in report["loaded_files"]
    output = capsys.readouterr().out
    assert "strict_fixed_prefix_web_ready" in output
    assert runtime.scene_prefix_hash in output


def test_strict_web_check_keeps_implicit_audit_in_memory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config_path, checkpoint, _map_path, _preview = _write_required_artifacts(tmp_path)
    config = _config(tmp_path)
    _use_synthetic_checkpoint_hashes(monkeypatch, checkpoint)
    save_calls: list[Path] = []

    def fake_load_config(path: Path, audit: FileAccessAudit) -> dict[str, Any]:
        audit.record(path)
        return config

    def record_save(_audit: FileAccessAudit, path: str | Path) -> None:
        save_calls.append(Path(path))

    monkeypatch.setattr(strict_web, "_load_runtime_config", fake_load_config)
    monkeypatch.setattr(FileAccessAudit, "save", record_save)

    result = strict_web.main(
        [
            "--config",
            str(config_path),
            "--scene",
            "scene_000001",
            "--checkpoint",
            str(checkpoint),
            "--check",
        ]
    )

    assert result == 0
    assert save_calls == []
    messages = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert messages[-1]["phase"] == "strict_fixed_prefix_web_audit_complete"
    assert messages[-1]["audit_log"] is None


def test_strict_web_check_persists_an_explicit_audit_log(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config_path, checkpoint, _map_path, _preview = _write_required_artifacts(tmp_path)
    config = _config(tmp_path)
    audit_path = tmp_path / "explicit-check-audit.json"
    _use_synthetic_checkpoint_hashes(monkeypatch, checkpoint)

    def fake_load_config(path: Path, audit: FileAccessAudit) -> dict[str, Any]:
        audit.record(path)
        return config

    monkeypatch.setattr(strict_web, "_load_runtime_config", fake_load_config)

    result = strict_web.main(
        [
            "--config",
            str(config_path),
            "--scene",
            "scene_000001",
            "--checkpoint",
            str(checkpoint),
            "--audit-log",
            str(audit_path),
            "--check",
        ]
    )

    assert result == 0
    report = json.loads(audit_path.read_text(encoding="utf-8"))
    assert report["passed"] is True
    assert report["forbidden_accesses"] == []
    messages = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert messages[-1]["audit_log"] == str(audit_path.resolve())


def test_v54_preflight_physically_rejects_training_metadata(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _config_path, checkpoint, _map_path, _preview = _write_required_artifacts(tmp_path)
    # A runtime release cannot colocate training-only semantic text at all.
    training_metadata = checkpoint / "metadata.json"
    training_metadata.write_text(
        '{"counterfactual_family":"book_support","answer_text":"left"}\n',
        encoding="utf-8",
    )
    _use_synthetic_checkpoint_hashes(monkeypatch, checkpoint)
    audit = FileAccessAudit([training_metadata], block_forbidden=True)

    with pytest.raises(ValueError, match="inventory changed"):
        strict_web._validate_v54_checkpoint(checkpoint, audit)

    assert audit.unique_paths == []


def test_strict_web_refuses_non_loopback_even_in_check_mode() -> None:
    assert (
        strict_web.main(
            [
                "--config",
                "configs/runtime/gemma4_v54.yaml",
                "--checkpoint",
                "data_gemma4/checkpoints/gemma4_v54_semantic_greedy_gate/update_000",
                "--host",
                "0.0.0.0",
                "--check",
            ]
        )
        == 2
    )


def test_strict_web_source_has_no_question_control_or_dataset_path() -> None:
    source = Path("src/semantic_3d_chat/chat/strict_prefix_web.py").read_text(encoding="utf-8")
    assert "question_control" not in source
    assert "semantic_3d_chat.data" not in source
    assert "semantic_3d_chat.evaluation" not in source
