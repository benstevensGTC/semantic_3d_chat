from __future__ import annotations

import json
from pathlib import Path

import pytest

from semantic_3d_chat.evaluation.ablations import file_sha256
from semantic_3d_chat.evaluation.control_score import score_control_suite


def _write_jsonl(path: Path, records: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in records),
        encoding="utf-8",
    )


def _fixture(tmp_path: Path) -> tuple[Path, Path]:
    references = tmp_path / "references.jsonl"
    reference_records = [
        {
            "scene_id": scene_id,
            "question_id": "q_000001",
            "question": "Is it present?",
            "answer": "yes",
            "answer_type": "presence",
            "target_xyz": None,
        }
        for scene_id in ("scene_000001", "scene_000002")
    ]
    _write_jsonl(references, reference_records)
    controls = tmp_path / "predictions"
    controls.mkdir()
    reports: dict[str, dict] = {}
    fields = {
        "semantic_shuffle": ["semantic"],
        "position_shuffle": ["xyz"],
        "geometry_only": ["semantic"],
        "semantics_without_xyz": ["xyz"],
        "remove_rgb": ["rgb"],
        "remove_normals": ["normal"],
    }
    conditions = (
        "primary",
        "empty_scene_prefix",
        "wrong_scene_prefix",
        "semantic_shuffle",
        "position_shuffle",
        "geometry_only",
        "semantics_without_xyz",
        "remove_rgb",
        "remove_normals",
    )
    for condition in conditions:
        records = []
        scenes = {}
        for index, scene_id in enumerate(("scene_000001", "scene_000002")):
            source_scene = (
                ("scene_000002", "scene_000001")[index]
                if condition == "wrong_scene_prefix"
                else scene_id
            )
            prefix_hash = f"{index + 1:064x}"
            metadata = {
                "affected_fields": fields.get(condition, []),
                "question_dependent_selection": False,
            }
            if condition in {"semantic_shuffle", "position_shuffle"}:
                metadata["permutation_sha256"] = "a" * 64
            scenes[scene_id] = {
                "prefix_hash": prefix_hash,
                "prefix_source_scene_id": source_scene,
                "prefix_built_before_questions": True,
                "metadata": metadata,
            }
            records.append(
                {
                    "scene_id": scene_id,
                    "question_id": "q_000001",
                    "predicted_answer": "yes",
                    "condition": condition,
                    "prefix_hash": prefix_hash,
                    "prefix_source_scene_id": source_scene,
                }
            )
        path = controls / f"{condition}.jsonl"
        _write_jsonl(path, records)
        reports[condition] = {
            "path": str(path),
            "sha256": file_sha256(path),
            "scenes": scenes,
        }
    manifest = controls / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "question_dependent_retrieval": False,
                "one_prefix_per_scene_condition": True,
                "conditions": reports,
            }
        ),
        encoding="utf-8",
    )
    return manifest, references


def test_control_score_authenticates_and_scores_complete_suite(tmp_path: Path) -> None:
    manifest, references = _fixture(tmp_path)
    report = score_control_suite(
        manifest,
        references,
        output_directory=tmp_path / "metrics",
    )

    assert set(report["results"]) == {
        "primary",
        "empty_scene_prefix",
        "wrong_scene_prefix",
        "semantic_shuffle",
        "position_shuffle",
        "geometry_only",
        "semantics_without_xyz",
        "remove_rgb",
        "remove_normals",
    }
    assert report["results"]["primary"]["normalized_exact_accuracy"] == 1.0
    assert report["results"]["position_shuffle"]["exact_accuracy_delta_vs_primary"] == 0.0
    assert Path(report["summary_path"]).is_file()
    assert all(Path(result["metrics_path"]).is_file() for result in report["results"].values())


def test_control_score_rejects_changed_prediction_after_manifest_receipt(
    tmp_path: Path,
) -> None:
    manifest, references = _fixture(tmp_path)
    prediction = manifest.parent / "position_shuffle.jsonl"
    prediction.write_text(prediction.read_text(encoding="utf-8") + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="hash mismatch"):
        score_control_suite(
            manifest,
            references,
            output_directory=tmp_path / "metrics",
        )


def test_control_score_rejects_question_varying_prefix_receipt(tmp_path: Path) -> None:
    manifest, references = _fixture(tmp_path)
    prediction = manifest.parent / "primary.jsonl"
    records = [json.loads(line) for line in prediction.read_text().splitlines()]
    records[0]["prefix_hash"] = "f" * 64
    _write_jsonl(prediction, records)
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["conditions"]["primary"]["sha256"] = file_sha256(prediction)
    manifest.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="prefix is not invariant"):
        score_control_suite(
            manifest,
            references,
            output_directory=tmp_path / "metrics",
        )
