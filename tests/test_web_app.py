from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from starlette.testclient import TestClient

from semantic_3d_chat.chat.file_audit import FileAccessAudit
from semantic_3d_chat.chat.web_app import (
    create_web_app,
    resolve_visual_assets,
    validate_visual_assets,
)


@dataclass(frozen=True)
class FakeAnswer:
    question: str
    prefix_hash: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "question": self.question,
            "answer": "continuous answer",
            "grounding_xyz_m": [0.25, -0.5, 0.75],
            "grounding_confidence": 0.7,
            "grounding_support_distance_m": 0.08,
            "prefix_hash": self.prefix_hash,
            "generated_tokens": 2,
            "elapsed_seconds": 0.01,
        }


class FakeRuntime:
    scene_id = "scene_000001"
    scene_prefix_hash = "a" * 64

    def __init__(self) -> None:
        self.answer_calls = 0
        self.prefix_checks = 0

    @property
    def questions_answered(self) -> int:
        return self.answer_calls

    def answer(self, question: str) -> FakeAnswer:
        self.answer_calls += 1
        return FakeAnswer(question=question.strip(), prefix_hash=self.scene_prefix_hash)

    def assert_prefix_unchanged(self) -> None:
        self.prefix_checks += 1

    def startup_summary(self) -> dict[str, Any]:
        return {
            "scene_id": self.scene_id,
            "prefix_hash": self.scene_prefix_hash,
            "prefix_shape": [1, 258, 896],
            "scene_latents": 256,
            "language_hidden_dim": 896,
            "source_voxels": 74_699,
            "processed_voxels": 8_422,
            "occupied_blocks": 3_019,
            "device": "mps",
            "prefix_build_seconds": 3.2,
            "scene_prefix_computed_before_question": True,
            "strict_fixed_environment_embedding_input": True,
            "environment_conditioned_input_sha256": self.scene_prefix_hash,
            "question_conditioned_scene_readout_tokens": False,
            "question_dependent_scene_retrieval": False,
            "checkpoint": "/private/checkpoint/path",
        }


class FakeQuestionConditionedRuntime(FakeRuntime):
    def startup_summary(self) -> dict[str, Any]:
        return {
            **super().startup_summary(),
            "strict_fixed_environment_embedding_input": False,
            "question_conditioned_scene_readout_tokens": True,
        }


def tiny_config() -> dict[str, Any]:
    return {
        "paths": {"reports_root": "reports"},
        "render": {
            "camera_position_m": [0.0, 0.0, 1.4],
            "yaw_degrees": [0.0, 45.0],
            "pitch_degrees": [-25.0, 0.0, 25.0],
        },
    }


def _write_image(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # FileResponse only needs an existing allowlisted raster for these tests.
    path.write_bytes(b"test-raster")


def test_web_chat_reuses_one_prefix_for_multiple_questions(tmp_path: Path) -> None:
    overview = tmp_path / "reports/figures/scan_montage.png"
    map_preview = tmp_path / "reports/figures/scene_000001/map_rgb.png"
    _write_image(overview)
    _write_image(map_preview)
    runtime = FakeRuntime()
    app = create_web_app(runtime, tiny_config(), project_root=tmp_path)

    with TestClient(app) as client:
        state = client.get("/api/state")
        first = client.post("/api/chat", json={"question": "Is there a chair?"})
        second = client.post("/api/chat", json={"question": "Where is the bowl?"})

    assert state.status_code == 200
    assert state.json()["prefix_built_before_questions"] is True
    assert state.json()["question_dependent_retrieval"] is False
    assert state.json()["viewpoint"]["position_m"] == [0.0, 0.0, 1.4]
    assert set(state.json()["visuals"]) == {"overview", "map"}
    assert first.status_code == second.status_code == 200
    assert first.json()["prefix_hash"] == second.json()["prefix_hash"] == "a" * 64
    assert first.json()["prefix_reused"] is second.json()["prefix_reused"] is True
    assert second.json()["questions_answered"] == 2
    assert runtime.answer_calls == 2
    assert runtime.prefix_checks >= 6


def test_web_never_mislabels_question_conditioned_runtime_as_strict(tmp_path: Path) -> None:
    runtime = FakeQuestionConditionedRuntime()
    app = create_web_app(runtime, tiny_config(), project_root=tmp_path)

    with TestClient(app) as client:
        state = client.get("/api/state").json()
        answer = client.post("/api/chat", json={"question": "Where is it?"}).json()

    for payload in (state, answer):
        assert payload["strict_fixed_environment_embedding_input"] is False
        assert payload["question_conditioned_scene_readout_tokens"] is True
        assert payload["question_dependent_retrieval"] is False


def test_web_assets_are_allowlisted_and_audited_without_forbidden_reads(tmp_path: Path) -> None:
    overview = tmp_path / "reports/figures/scan_montage.png"
    _write_image(overview)
    forbidden_root = tmp_path / "data/oracle"
    forbidden_root.mkdir(parents=True)
    audit = FileAccessAudit([forbidden_root])
    runtime = FakeRuntime()
    with audit:
        app = create_web_app(runtime, tiny_config(), project_root=tmp_path, audit=audit)
        with TestClient(app) as client:
            response = client.get("/assets/overview")
            unknown = client.get("/assets/../../data/oracle/secret.png")

    assert response.status_code == 200
    assert unknown.status_code == 404
    assert str(overview.resolve()) in audit.unique_paths
    audit.assert_clean()


def test_visual_asset_validation_rejects_paths_outside_reports(tmp_path: Path) -> None:
    figure_root = tmp_path / "reports/figures"
    safe = figure_root / "scene_000001/map_rgb.png"
    forbidden = tmp_path / "data/oracle/scene_000001.png"
    wrong_suffix = figure_root / "scene_000001/map_rgb.ply"
    for path in (safe, forbidden, wrong_suffix):
        _write_image(path)

    assert validate_visual_assets({"map": safe}, figure_root) == {"map": safe.resolve()}
    for candidate in (forbidden, wrong_suffix):
        try:
            validate_visual_assets({"map": candidate}, figure_root)
        except ValueError:
            pass
        else:
            raise AssertionError(f"Unsafe web visual was accepted: {candidate}")
    try:
        validate_visual_assets({"unexpected": safe}, figure_root)
    except ValueError as exc:
        assert "Unknown web visual" in str(exc)
    else:
        raise AssertionError("Unknown asset name was accepted")


def test_visual_resolution_never_falls_back_to_runtime_data(tmp_path: Path) -> None:
    rendered_overview = tmp_path / "data/rendered/scene_000001/p_000000.png"
    _write_image(rendered_overview)
    figure_root, assets = resolve_visual_assets(tmp_path, "reports", "scene_000001")
    assert figure_root == (tmp_path / "reports/figures").resolve()
    assert assets == {}
    try:
        resolve_visual_assets(tmp_path, "data/oracle", "scene_000001")
    except ValueError as exc:
        assert "forbidden runtime directory" in str(exc)
    else:
        raise AssertionError("An oracle-backed reports root was accepted")


def test_web_state_filters_private_checkpoint_path(tmp_path: Path) -> None:
    runtime = FakeRuntime()
    app = create_web_app(runtime, tiny_config(), project_root=tmp_path)
    with TestClient(app) as client:
        payload = client.get("/api/state").json()
        page = client.get("/")
    assert "checkpoint" not in payload
    assert page.status_code == 200
    assert "continuous full-scene prefix" in page.text


def test_web_rejects_bad_requests_before_runtime_inference(tmp_path: Path) -> None:
    runtime = FakeRuntime()
    app = create_web_app(runtime, tiny_config(), project_root=tmp_path)
    with TestClient(app) as client:
        invalid = client.post(
            "/api/chat", content=b"not-json", headers={"content-type": "application/json"}
        )
        empty = client.post("/api/chat", json={"question": "  "})
        large = client.post("/api/chat", json={"question": "x" * 4_097})
    assert invalid.status_code == empty.status_code == large.status_code == 400
    assert runtime.answer_calls == 0


def test_web_server_has_no_evaluation_or_dataset_imports() -> None:
    source = Path("src/semantic_3d_chat/chat/web_app.py").read_text(encoding="utf-8")
    assert "semantic_3d_chat.data" not in source
    assert "semantic_3d_chat.evaluation" not in source
    assert "qa_generator" not in source
    assert "generate_scene" not in source
