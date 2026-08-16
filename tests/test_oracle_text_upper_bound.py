from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch

from semantic_3d_chat.config import load_config
from semantic_3d_chat.evaluation.baseline_io import sha256_file
from semantic_3d_chat.evaluation.oracle_text_artifacts import (
    load_scene_text_bundle,
)
from semantic_3d_chat.evaluation.oracle_text_predict import (
    LocalGemmaOracleTextAnswerer,
    run_oracle_text_predictions,
)
from semantic_3d_chat.evaluation.oracle_text_prepare import (
    oracle_scene_text,
    prepare_scene_text_bundle,
)
from semantic_3d_chat.evaluation.oracle_text_score import (
    score_oracle_text_predictions,
)
from semantic_3d_chat.evaluation.question_manifest import build_question_manifest


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _oracle(scene_id: str) -> dict[str, Any]:
    return {
        "scene_id": scene_id,
        "room": {
            "size_m": [6.0, 5.0, 3.0],
            "bounds_min_m": [-3.0, -2.5, 0.0],
            "bounds_max_m": [3.0, 2.5, 3.0],
        },
        "instances": [
            {
                "instance_id": "i_floor",
                "category": "floor",
                "kind": "surface",
                "color": {"name": "gray"},
                "expected_center_xyz_m": [0.0, 0.0, 0.0],
                "dimensions_m": [6.0, 5.0, 0.1],
                "pose": {
                    "center_xyz_m": [0.0, 0.0, 0.0],
                    "rotation_euler_degrees": [0.0, 0.0, 0.0],
                },
                "support_surface": None,
                "visible_from_center_scan": True,
            },
            {
                "instance_id": "i_table",
                "category": "table",
                "kind": "object",
                "color": {"name": "brown"},
                "expected_center_xyz_m": [1.0, 2.0, 0.4],
                "dimensions_m": [1.0, 0.8, 0.8],
                "pose": {
                    "center_xyz_m": [1.0, 2.0, 0.4],
                    "rotation_euler_degrees": [0.0, 0.0, 0.0],
                },
                "support_surface": "i_floor",
                "visible_from_center_scan": True,
            },
            {
                "instance_id": "i_cube",
                "category": "cube",
                "kind": "object",
                "color": {"name": "red"},
                "expected_center_xyz_m": [1.0, 2.0, 0.9],
                "dimensions_m": [0.2, 0.2, 0.2],
                "pose": {
                    "center_xyz_m": [1.0, 2.0, 0.9],
                    "rotation_euler_degrees": [180.0, 0.0, 0.0],
                },
                "support_surface": "i_table",
                "visible_from_center_scan": True,
            },
        ],
        "relationships": [
            {
                "subject_instance_id": "i_cube",
                "predicate": "on",
                "object_instance_id": "i_table",
            },
            {
                "subject_instance_id": "i_cube",
                "predicate": "above",
                "object_instance_id": "i_table",
            },
        ],
        # Preparation must not serialize this generator-side counterfactual label.
        "generation": {"change_type": "hidden_counterfactual_name"},
    }


def _fixture(tmp_path: Path, *, question_count: int = 2) -> dict[str, Path | dict[str, Any]]:
    scene_id = "scene_999001"
    references = tmp_path / "answer_references.jsonl"
    rows = [
        {
            "scene_id": scene_id,
            "question_id": "q_000001",
            "question": "What color is the cube?",
            "answer": "red",
            "answer_type": "attribute",
            "target_instance": "i_cube",
        },
        {
            "scene_id": scene_id,
            "question_id": "q_000002",
            "question": "Is the cube on the table?",
            "answer": "yes",
            "answer_type": "presence",
            "target_instance": "i_cube",
        },
    ][:question_count]
    references.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    questions = build_question_manifest(rows, source_qa_sha256=sha256_file(references))
    question_path = tmp_path / "inference_inputs" / "questions.json"
    _write_json(question_path, questions.as_dict())
    oracle_root = tmp_path / "oracle"
    _write_json(oracle_root / scene_id / "oracle.json", _oracle(scene_id))
    scene_text = tmp_path / "evaluation_only" / "oracle_text_control" / "scenes.json"
    predictions = tmp_path / "evaluation_only" / "oracle_text_control" / "predictions.jsonl"
    metrics = tmp_path / "metrics.json"
    config = load_config("configs/experiments/gemma4_oracle_text_v55.yaml")
    return {
        "config": config,
        "references": references,
        "questions": question_path,
        "oracle_root": oracle_root,
        "scene_text": scene_text,
        "predictions": predictions,
        "metrics": metrics,
    }


def test_scene_text_is_complete_question_independent_and_sanitized() -> None:
    oracle = _oracle("scene_999001")
    text = oracle_scene_text(
        oracle,
        camera_position_m=[0.0, 0.0, 1.4],
        camera_yaw_degrees=0.0,
        camera_pitch_degrees=0.0,
    )
    assert "prohibited as input to the primary" in text
    assert "independent of the question" in text
    assert "cube=1" in text
    assert "color=red" in text
    assert "orientation=upside down" in text
    assert "supported_by=table" in text
    assert "cube | on | table" in text
    assert "camera_distance=" in text
    assert "i_cube" not in text
    assert "hidden_counterfactual_name" not in text


def test_local_gemma_answerer_accepts_transformers5_batch_encoding() -> None:
    class Encoded:
        input_ids = torch.tensor([[10, 11, 12]])

    class Tokenizer:
        pad_token_id = 0
        eos_token_id = 2

        @staticmethod
        def apply_chat_template(messages, **_kwargs):
            assert "category=cube" in messages[0]["content"]
            assert messages[1]["content"] == "What color?"
            return Encoded()

        @staticmethod
        def decode(tokens, **_kwargs):
            assert tokens.tolist() == [42]
            return "red"

    class Model:
        @staticmethod
        def generate(**kwargs):
            assert kwargs["input_ids"].shape == (1, 3)
            assert kwargs["do_sample"] is False
            return torch.tensor([[10, 11, 12, 42]])

    answerer = LocalGemmaOracleTextAnswerer.__new__(LocalGemmaOracleTextAnswerer)
    answerer.system_prompt = "Use exact facts."
    answerer.max_answer_tokens = 4
    answerer.local = SimpleNamespace(
        tokenizer=Tokenizer(),
        model=Model(),
        device=torch.device("cpu"),
    )
    assert answerer("category=cube; color=red", "What color?") == "red"


def test_prepare_then_answer_blind_predict_resume_and_score(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    preparation = prepare_scene_text_bundle(
        fixture["config"],
        fixture["questions"],
        fixture["oracle_root"],
        fixture["scene_text"],
        require_v55_development=False,
    )
    assert preparation["question_count"] == 2
    assert preparation["scene_count"] == 1
    bundle = load_scene_text_bundle(fixture["scene_text"])
    assert bundle.question_manifest_sha256 == preparation["question_manifest_sha256"]

    # Prove that inference needs neither simulator oracle nor answer references.
    Path(fixture["oracle_root"]).rename(tmp_path / "oracle_unavailable")
    Path(fixture["references"]).rename(tmp_path / "references_unavailable.jsonl")
    observed: list[tuple[str, str]] = []

    def answerer(scene_text: str, question: str) -> str:
        observed.append((scene_text, question))
        return "red" if "color" in question.casefold() else "yes"

    inference = run_oracle_text_predictions(
        fixture["config"],
        fixture["questions"],
        fixture["scene_text"],
        fixture["predictions"],
        answerer=answerer,
        allow_test_answerer=True,
        require_v55_development=False,
    )
    assert inference["status"] == "complete"
    assert inference["new_prediction_count"] == 2
    assert len(observed) == 2
    assert all("target_instance" not in scene_text for scene_text, _ in observed)

    prediction_rows = [
        json.loads(line)
        for line in Path(fixture["predictions"]).read_text(encoding="utf-8").splitlines()
    ]
    assert {row["answer"] for row in prediction_rows} == {"red", "yes"}
    assert all("question" not in row and "answer_type" not in row for row in prediction_rows)
    assert all(row["prohibited_primary_input"] is True for row in prediction_rows)

    second = run_oracle_text_predictions(
        fixture["config"],
        fixture["questions"],
        fixture["scene_text"],
        fixture["predictions"],
        answerer=lambda _scene, _question: pytest.fail("cached answer was regenerated"),
        allow_test_answerer=True,
        require_v55_development=False,
    )
    assert second["new_prediction_count"] == 0
    assert second["resumed_prediction_count"] == 2
    assert second["scientific_measurement_eligible"] is False

    (tmp_path / "references_unavailable.jsonl").rename(fixture["references"])
    with pytest.raises(ValueError, match="actual local Gemma"):
        score_oracle_text_predictions(
            fixture["references"],
            fixture["predictions"],
            fixture["metrics"],
            require_v55_development=False,
        )
    report = score_oracle_text_predictions(
        fixture["references"],
        fixture["predictions"],
        fixture["metrics"],
        require_v55_development=False,
        require_local_gemma=False,
    )
    assert report["metrics"]["normalized_exact_accuracy"] == 1.0
    assert report["scope"]["model_loaded_by_scorer"] is False
    assert report["scope"]["answer_bearing_references_loaded_by_scorer"] is True
    serialized = Path(fixture["metrics"]).read_text(encoding="utf-8")
    assert "What color is the cube?" not in serialized
    assert '"answer": "red"' not in serialized


def test_injected_answerer_requires_explicit_test_opt_in(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path, question_count=1)
    prepare_scene_text_bundle(
        fixture["config"],
        fixture["questions"],
        fixture["oracle_root"],
        fixture["scene_text"],
        require_v55_development=False,
    )
    with pytest.raises(ValueError, match="test-only"):
        run_oracle_text_predictions(
            fixture["config"],
            fixture["questions"],
            fixture["scene_text"],
            fixture["predictions"],
            answerer=lambda _scene, _question: "red",
            require_v55_development=False,
        )


def test_prediction_is_crash_resumable_one_question_at_a_time(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    prepare_scene_text_bundle(
        fixture["config"],
        fixture["questions"],
        fixture["oracle_root"],
        fixture["scene_text"],
        require_v55_development=False,
    )
    calls = 0

    def interrupted(_scene_text: str, _question: str) -> str:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("simulated interruption")
        return "red"

    with pytest.raises(RuntimeError, match="simulated interruption"):
        run_oracle_text_predictions(
            fixture["config"],
            fixture["questions"],
            fixture["scene_text"],
            fixture["predictions"],
            answerer=interrupted,
            allow_test_answerer=True,
            require_v55_development=False,
        )
    assert len(Path(fixture["predictions"]).read_text().splitlines()) == 1

    resumed_questions: list[str] = []

    def resumed(_scene_text: str, question: str) -> str:
        resumed_questions.append(question)
        return "yes"

    report = run_oracle_text_predictions(
        fixture["config"],
        fixture["questions"],
        fixture["scene_text"],
        fixture["predictions"],
        answerer=resumed,
        allow_test_answerer=True,
        require_v55_development=False,
    )
    assert resumed_questions == ["Is the cube on the table?"]
    assert report["new_prediction_count"] == 1
    assert report["resumed_prediction_count"] == 1


def test_inference_rejects_oracle_path_and_changed_prepared_input(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path, question_count=1)
    prepare_scene_text_bundle(
        fixture["config"],
        fixture["questions"],
        fixture["oracle_root"],
        fixture["scene_text"],
        require_v55_development=False,
    )
    forbidden = Path(fixture["oracle_root"]) / "prepared_scene_text.json"
    forbidden.write_bytes(Path(fixture["scene_text"]).read_bytes())
    with pytest.raises(ValueError, match="QA/oracle"):
        run_oracle_text_predictions(
            fixture["config"],
            fixture["questions"],
            forbidden,
            fixture["predictions"],
            answerer=lambda _scene, _question: "red",
            allow_test_answerer=True,
            require_v55_development=False,
        )

    run_oracle_text_predictions(
        fixture["config"],
        fixture["questions"],
        fixture["scene_text"],
        fixture["predictions"],
        answerer=lambda _scene, _question: "red",
        allow_test_answerer=True,
        require_v55_development=False,
    )
    payload = json.loads(Path(fixture["scene_text"]).read_text())
    payload["scenes"][0]["scene_text"] += " tampered"
    _write_json(Path(fixture["scene_text"]), payload)
    with pytest.raises(ValueError, match="scene text hash"):
        run_oracle_text_predictions(
            fixture["config"],
            fixture["questions"],
            fixture["scene_text"],
            fixture["predictions"],
            answerer=lambda _scene, _question: "red",
            allow_test_answerer=True,
            require_v55_development=False,
        )


def test_scorer_rejects_predictions_modified_after_completed_report(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path, question_count=1)
    prepare_scene_text_bundle(
        fixture["config"],
        fixture["questions"],
        fixture["oracle_root"],
        fixture["scene_text"],
        require_v55_development=False,
    )
    run_oracle_text_predictions(
        fixture["config"],
        fixture["questions"],
        fixture["scene_text"],
        fixture["predictions"],
        answerer=lambda _scene, _question: "red",
        allow_test_answerer=True,
        require_v55_development=False,
    )
    with Path(fixture["predictions"]).open("a", encoding="utf-8") as handle:
        handle.write("{}\n")
    with pytest.raises(ValueError, match="authenticate predictions"):
        score_oracle_text_predictions(
            fixture["references"],
            fixture["predictions"],
            fixture["metrics"],
            require_v55_development=False,
            require_local_gemma=False,
        )


def test_upper_bound_trust_zones_are_not_imported_by_primary_chat() -> None:
    project = Path(__file__).parents[1]
    predictor_source = project.joinpath(
        "src/semantic_3d_chat/evaluation/oracle_text_predict.py"
    ).read_text(encoding="utf-8")
    scorer_source = project.joinpath(
        "src/semantic_3d_chat/evaluation/oracle_text_score.py"
    ).read_text(encoding="utf-8")
    assert "oracle_text_prepare" not in predictor_source
    assert "oracle_text_score" not in predictor_source
    assert "local_lm" not in scorer_source
    assert "import torch" not in scorer_source
    for runtime_source in project.joinpath("src/semantic_3d_chat/chat").glob("*.py"):
        text = runtime_source.read_text(encoding="utf-8")
        assert "oracle_text_artifacts" not in text
        assert "oracle_text_prepare" not in text
        assert "oracle_text_predict" not in text
        assert "oracle_text_score" not in text
