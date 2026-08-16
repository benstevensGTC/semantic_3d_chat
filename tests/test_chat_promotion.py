from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from semantic_3d_chat.chat import launch as launch_module
from semantic_3d_chat.chat import promotion as promotion_module
from semantic_3d_chat.chat import runtime as runtime_module
from semantic_3d_chat.chat import runtime_config as runtime_config_module
from semantic_3d_chat.chat.cli import main as cli_main
from semantic_3d_chat.chat.file_audit import FileAccessAudit
from semantic_3d_chat.chat.launch import ChatLaunch, resolve_chat_launch
from semantic_3d_chat.chat.promotion import (
    create_chat_promotion,
    create_held_out_final_evidence,
    resolve_primary_pointer,
    sha256_file,
    validate_chat_promotion,
    write_primary_pointer,
)
from semantic_3d_chat.chat.runtime_config import (
    load_runtime_config,
    runtime_config_file_sha256,
)
from semantic_3d_chat.chat.web_app import main as web_main
from semantic_3d_chat.config import artifact_root
from semantic_3d_chat.evaluation import prediction_artifacts as prediction_artifacts_module
from semantic_3d_chat.evaluation.metrics import score_predictions
from semantic_3d_chat.evaluation.prediction_artifacts import build_prediction_provenance
from semantic_3d_chat.evaluation.question_manifest import build_question_manifest

RUNTIME_CONFIG = Path("configs/runtime/gemma4_primary.yaml").resolve()


@pytest.fixture(autouse=True)
def _stub_large_local_model_snapshot(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    real_artifact_root = promotion_module.artifact_root
    isolated_maps_root = tmp_path / "sanitized_maps"
    monkeypatch.setattr(
        promotion_module,
        "artifact_root",
        lambda config, kind: (
            isolated_maps_root
            if kind == "maps"
            else real_artifact_root(config, kind)
        ),
    )
    real_project_path = prediction_artifacts_module.project_path
    monkeypatch.setattr(
        prediction_artifacts_module,
        "project_path",
        lambda config, kind, *parts: (
            isolated_maps_root.joinpath(*parts)
            if kind == "maps"
            else real_project_path(config, kind, *parts)
        ),
    )
    monkeypatch.setattr(
        promotion_module,
        "local_model_snapshot_identity",
        lambda *_args, **_kwargs: {
            "tree_sha256": "9" * 64,
            "file_count": 9,
        },
    )


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in records),
        encoding="utf-8",
    )


def _prediction_provenance(
    *,
    runtime_config: Path,
    checkpoint: Path,
    questions: Path,
    run_kind: str,
    condition: str,
) -> dict:
    config = load_runtime_config(runtime_config)
    return build_prediction_provenance(
        config,
        config_path=runtime_config,
        checkpoint_path=checkpoint,
        references_path=questions,
        scene_ids=["scene_000003", "scene_000004"],
        split="test",
        run_kind=run_kind,
        condition=condition,
    ).as_dict()


def _evidence(
    tmp_path: Path,
    runtime_config: Path = RUNTIME_CONFIG,
    *,
    selected_update: int = 7,
) -> tuple[Path, Path, Path, Path]:
    checkpoint = tmp_path / "checkpoint" / f"update_{selected_update:03d}"
    checkpoint.mkdir(parents=True)
    (checkpoint / "adapter.safetensors").write_bytes(b"continuous-adapter")
    _write_json(checkpoint / "runtime_metadata.json", {"schema_version": 3})
    adapter_hash = sha256_file(checkpoint / "adapter.safetensors")

    selector = tmp_path / "selector.json"
    _write_json(
        selector,
        {
            "passed": True,
            "development_selection_passed": True,
            "chat_promotion_eligible": True,
            "selected_checkpoint": str(checkpoint.resolve()),
            "selected_update": selected_update,
            "final_test_scenes_touched": False,
            "chat_promotion": {
                "eligible": True,
                "evaluated": True,
                "checks": {
                    "development_checkpoint_selected": True,
                    "changed_complete_pair_threshold_met": True,
                    "aggregate_validation_exact_accuracy_retained": True,
                },
            },
        },
    )
    references = tmp_path / "test_references.jsonl"
    reference_records = [
        {
            "scene_id": "scene_000003",
            "question_id": "q_000001",
            "question": "Which side?",
            "answer": "left",
            "answer_type": "spatial_relation",
            "target_xyz": [0.0, 0.0, 0.0],
            "counterfactual_pair_id": "pair_000001",
        },
        {
            "scene_id": "scene_000004",
            "question_id": "q_000002",
            "question": "Which side?",
            "answer": "right",
            "answer_type": "spatial_relation",
            "target_xyz": [1.0, 0.0, 0.0],
            "counterfactual_pair_id": "pair_000001",
        },
    ]
    _write_jsonl(references, reference_records)
    questions = tmp_path / "questions.json"
    question_manifest = build_question_manifest(
        reference_records,
        source_qa_sha256=sha256_file(references),
    )
    _write_json(questions, question_manifest.as_dict())
    maps_root = promotion_module.artifact_root(
        load_runtime_config(runtime_config), "maps"
    )
    for scene_id in ("scene_000003", "scene_000004"):
        map_path = maps_root / scene_id / "voxel_map.npz"
        map_path.parent.mkdir(parents=True, exist_ok=True)
        map_path.write_bytes(f"continuous-map:{scene_id}".encode("ascii"))
    primary_provenance = _prediction_provenance(
        runtime_config=runtime_config,
        checkpoint=checkpoint,
        questions=questions,
        run_kind="continuous_scene_static",
        condition="all_questions",
    )
    primary_provenance_path = tmp_path / "predictions.jsonl.provenance.json"
    _write_json(primary_provenance_path, primary_provenance)
    predictions = tmp_path / "predictions.jsonl"
    prediction_records = [
        {
            "scene_id": "scene_000003",
            "question_id": "q_000001",
            "predicted_answer": "left",
            "grounding_xyz": [0.0, 0.0, 0.0],
            "prefix_hash": "c" * 64,
            "provenance_sha256": primary_provenance["provenance_sha256"],
        },
        {
            "scene_id": "scene_000004",
            "question_id": "q_000002",
            "predicted_answer": "right",
            "grounding_xyz": [1.0, 0.0, 0.0],
            "prefix_hash": "d" * 64,
            "provenance_sha256": primary_provenance["provenance_sha256"],
        },
    ]
    _write_jsonl(predictions, prediction_records)
    chance_condition = json.dumps(
        {"condition": "empty_scene_prefix", "max_questions_per_scene": None},
        sort_keys=True,
        separators=(",", ":"),
    )
    chance_provenance = _prediction_provenance(
        runtime_config=runtime_config,
        checkpoint=checkpoint,
        questions=questions,
        run_kind="continuous_scene_control",
        condition=chance_condition,
    )
    chance_provenance_path = tmp_path / "chance.jsonl.provenance.json"
    _write_json(chance_provenance_path, chance_provenance)
    chance_predictions = tmp_path / "chance.jsonl"
    chance_prediction_records = [
        {
            "scene_id": "scene_000003",
            "question_id": "q_000001",
            "predicted_answer": "left",
            "grounding_xyz": [0.5, 0.0, 0.0],
            "provenance_sha256": chance_provenance["provenance_sha256"],
        },
        {
            "scene_id": "scene_000004",
            "question_id": "q_000002",
            "predicted_answer": "left",
            "grounding_xyz": [0.5, 0.0, 0.0],
            "provenance_sha256": chance_provenance["provenance_sha256"],
        },
    ]
    _write_jsonl(chance_predictions, chance_prediction_records)
    metrics = tmp_path / "metrics.json"
    common_metrics = {
        "references_path": str(references.resolve()),
        "references_sha256": sha256_file(references),
    }
    _write_json(
        metrics,
        {
            **score_predictions(reference_records, prediction_records),
            **common_metrics,
            "predictions_path": str(predictions.resolve()),
            "predictions_sha256": sha256_file(predictions),
        },
    )
    chance_metrics = tmp_path / "chance_metrics.json"
    _write_json(
        chance_metrics,
        {
            **score_predictions(reference_records, chance_prediction_records),
            **common_metrics,
            "predictions_path": str(chance_predictions.resolve()),
            "predictions_sha256": sha256_file(chance_predictions),
        },
    )
    split_manifest = tmp_path / "splits.json"
    _write_json(
        split_manifest,
        {
            "splits": {
                "train": ["scene_000001"],
                "validation": ["scene_000002"],
                "test": ["scene_000003", "scene_000004"],
            }
        },
    )
    final = tmp_path / "final.json"
    create_held_out_final_evidence(
        runtime_config_path=runtime_config,
        checkpoint=checkpoint,
        metrics_path=metrics,
        predictions_path=predictions,
        prediction_provenance_path=primary_provenance_path,
        chance_metrics_path=chance_metrics,
        chance_predictions_path=chance_predictions,
        chance_prediction_provenance_path=chance_provenance_path,
        split_manifest_path=split_manifest,
        output_path=final,
    )
    runtime_hash = sha256_file(checkpoint / "runtime_metadata.json")
    config_hash = runtime_config_file_sha256(runtime_config)
    leakage = tmp_path / "leakage.json"
    loaded_runtime_config = load_runtime_config(runtime_config)
    _write_json(
        leakage,
        {
            "schema_version": 1,
            "passed": True,
            "scene_id": "scene_000003",
            "checkpoint": str(checkpoint.resolve()),
            "runtime_config": str(runtime_config.resolve()),
            "oracle_directory": str(
                artifact_root(loaded_runtime_config, "oracle").resolve()
            ),
            "oracle_was_renamed": True,
            "oracle_unavailable_during_inference": True,
            "oracle_restored": True,
            "prefix_computed_before_first_question": True,
            "prefix_invariant": True,
            "checkpoint_adapter_sha256": adapter_hash,
            "checkpoint_runtime_metadata_sha256": runtime_hash,
            "runtime_config_file_sha256": config_hash,
            "forbidden_accesses": [],
            "question_count": 3,
            "prefix_hash": "c" * 64,
            "prefix_hashes": ["c" * 64] * 4,
            "answers": [
                {"question": "Question one?", "prefix_hash": "c" * 64},
                {"question": "Question two?", "prefix_hash": "c" * 64},
                {"question": "Question three?", "prefix_hash": "c" * 64},
            ],
            "loaded_files": [
                str(runtime_config.resolve()),
                str((checkpoint / "adapter.safetensors").resolve()),
                str((checkpoint / "runtime_metadata.json").resolve()),
            ],
            "failure": None,
        },
    )
    return checkpoint, selector, final, leakage


def _create(tmp_path: Path) -> tuple[Path, Path, Path, Path, Path]:
    checkpoint, selector, final, leakage = _evidence(tmp_path)
    create_chat_promotion(
        runtime_config_path=RUNTIME_CONFIG,
        checkpoint=checkpoint,
        selector_report_path=selector,
        final_evidence_path=final,
        leakage_report_path=leakage,
    )
    return checkpoint, selector, final, leakage, checkpoint / "promotion.json"


def test_promotion_creation_binds_all_evidence_and_runtime_inputs(tmp_path: Path) -> None:
    checkpoint, selector, final, leakage, promotion_path = _create(tmp_path)
    promotion = validate_chat_promotion(checkpoint, RUNTIME_CONFIG)
    assert promotion_path.is_file()
    assert promotion["status"] == "accepted"
    assert promotion["selector_report_sha256"] == sha256_file(selector)
    assert promotion["final_evidence_sha256"] == sha256_file(final)
    assert promotion["leakage_report_sha256"] == sha256_file(leakage)
    assert promotion["checkpoint_runtime_metadata_sha256"] == sha256_file(
        checkpoint / "runtime_metadata.json"
    )
    assert promotion["model_snapshot_sha256"] == "9" * 64
    assert promotion["model_snapshot_file_count"] == 9
    final_payload = json.loads(final.read_text(encoding="utf-8"))
    assert promotion["scene_runtime_manifest"] == final_payload["scene_runtime_manifest"]
    assert set(promotion["scene_runtime_manifest"]) == {
        "scene_000003",
        "scene_000004",
    }
    for scene_id, entry in promotion["scene_runtime_manifest"].items():
        assert set(entry) == {
            "voxel_map_sha256",
            "voxel_map_size_bytes",
            "scene_prefix_sha256",
        }
        assert len(entry["voxel_map_sha256"]) == 64
        assert entry["voxel_map_size_bytes"] > 0
        assert entry["scene_prefix_sha256"] in {"c" * 64, "d" * 64}
        assert "/" not in scene_id


def test_promotion_rejects_voxel_map_swapped_after_final_evidence(
    tmp_path: Path,
) -> None:
    checkpoint, selector, final, leakage = _evidence(tmp_path)
    config = load_runtime_config(RUNTIME_CONFIG)
    map_path = (
        promotion_module.artifact_root(config, "maps")
        / "scene_000003"
        / "voxel_map.npz"
    )
    map_path.write_bytes(b"swapped-map-bytes")

    with pytest.raises(ValueError, match="map manifest bytes changed"):
        create_chat_promotion(
            runtime_config_path=RUNTIME_CONFIG,
            checkpoint=checkpoint,
            selector_report_path=selector,
            final_evidence_path=final,
            leakage_report_path=leakage,
        )


def test_promoted_chat_verifies_scene_map_and_computed_prefix(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkpoint, *_rest, promotion_path = _create(tmp_path)
    promotion = json.loads(promotion_path.read_text(encoding="utf-8"))
    config = load_runtime_config(RUNTIME_CONFIG)
    maps_root = promotion_module.artifact_root(config, "maps")
    monkeypatch.setattr(
        launch_module,
        "project_path",
        lambda _config, _kind, scene_id, _filename: (
            maps_root / scene_id / "voxel_map.npz"
        ),
    )
    launch = ChatLaunch(
        config_path=RUNTIME_CONFIG,
        checkpoint_path=checkpoint,
        config=config,
        promotion=promotion,
    )

    assert launch.verify_scene_map("scene_000003") == (
        maps_root / "scene_000003" / "voxel_map.npz"
    ).resolve()
    launch.verify_scene_prefix(
        "scene_000003",
        loaded_scene_id="scene_000003",
        prefix_sha256="c" * 64,
    )

    with pytest.raises(ValueError, match="not attested"):
        launch.verify_scene_map("scene_000001")
    with pytest.raises(ValueError, match="Loaded runtime scene does not match"):
        launch.verify_scene_prefix(
            "scene_000003",
            loaded_scene_id="scene_000004",
            prefix_sha256="c" * 64,
        )
    with pytest.raises(ValueError, match="Computed scene prefix does not match"):
        launch.verify_scene_prefix(
            "scene_000003",
            loaded_scene_id="scene_000003",
            prefix_sha256="d" * 64,
        )

    (maps_root / "scene_000003" / "voxel_map.npz").write_bytes(
        (maps_root / "scene_000004" / "voxel_map.npz").read_bytes()
    )
    with pytest.raises(ValueError, match="voxel-map bytes do not match"):
        launch.verify_scene_map("scene_000003")


def test_primary_pointer_cli_verifies_map_before_load_and_prefix_after_load(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []
    config = {
        "paths": {
            "data_root": str(tmp_path / "data"),
            "reports_root": str(tmp_path / "reports"),
        }
    }

    class _Launch:
        checkpoint_path = tmp_path / "checkpoint"
        is_production_gemma = True

        def __init__(self) -> None:
            self.config = config

        def verify_scene_map(self, scene_id: str, *, audit=None) -> None:
            assert scene_id == "scene_000003"
            assert audit is not None
            events.append("map_verified")

        def verify_scene_prefix(
            self,
            requested_scene_id: str,
            *,
            loaded_scene_id: str,
            prefix_sha256: str,
        ) -> None:
            assert requested_scene_id == loaded_scene_id == "scene_000003"
            assert prefix_sha256 == "c" * 64
            events.append("prefix_verified")

    class _Answer:
        def to_dict(self) -> dict[str, object]:
            return {
                "answer": "left",
                "grounding_xyz_m": [0.0, 0.0, 0.0],
                "grounding_confidence": 1.0,
                "grounding_support_distance_m": 0.0,
                "prefix_hash": "c" * 64,
            }

    class _Runtime:
        scene_id = "scene_000003"
        scene_prefix_hash = "c" * 64

        @classmethod
        def load(cls, *_args, **_kwargs):
            assert events == ["map_verified"]
            events.append("runtime_loaded")
            return cls()

        def startup_summary(self) -> dict[str, object]:
            return {"scene_id": self.scene_id, "prefix_hash": self.scene_prefix_hash}

        def answer(self, _question: str) -> _Answer:
            events.append("question_answered")
            return _Answer()

        def assert_prefix_unchanged(self) -> None:
            return None

    monkeypatch.setattr(
        launch_module, "resolve_chat_launch", lambda **_kwargs: _Launch()
    )
    monkeypatch.setattr(runtime_module, "StaticChatRuntime", _Runtime)

    result = cli_main(
        [
            "--primary-pointer",
            str(tmp_path / "primary.json"),
            "--scene",
            "scene_000003",
            "--question",
            "Which side?",
            "--audit-log",
            str(tmp_path / "audit.json"),
            "--chat-log",
            str(tmp_path / "chat.jsonl"),
        ]
    )

    assert result == 0
    assert events == [
        "map_verified",
        "runtime_loaded",
        "prefix_verified",
        "question_answered",
    ]


def test_runtime_rejects_local_model_snapshot_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkpoint, *_ = _create(tmp_path)
    monkeypatch.setattr(
        promotion_module,
        "local_model_snapshot_identity",
        lambda *_args, **_kwargs: {
            "tree_sha256": "8" * 64,
            "file_count": 9,
        },
    )
    with pytest.raises(ValueError, match="invalid or stale"):
        validate_chat_promotion(checkpoint, RUNTIME_CONFIG)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("chat_promotion_eligible", False, "denied chat promotion"),
        ("selected_checkpoint", "/tmp/wrong", "does not match"),
    ],
)
def test_promotion_creation_rejects_false_or_mismatched_selector(
    tmp_path: Path,
    field: str,
    value: object,
    message: str,
) -> None:
    checkpoint, selector, final, leakage = _evidence(tmp_path)
    payload = json.loads(selector.read_text(encoding="utf-8"))
    payload[field] = value
    _write_json(selector, payload)
    with pytest.raises(ValueError, match=message):
        create_chat_promotion(
            runtime_config_path=RUNTIME_CONFIG,
            checkpoint=checkpoint,
            selector_report_path=selector,
            final_evidence_path=final,
            leakage_report_path=leakage,
        )
    assert not (checkpoint / "promotion.json").exists()


def test_promotion_accepts_selected_update_zero_with_exact_checkpoint_suffix(
    tmp_path: Path,
) -> None:
    checkpoint, selector, final, leakage = _evidence(tmp_path, selected_update=0)

    promotion = create_chat_promotion(
        runtime_config_path=RUNTIME_CONFIG,
        checkpoint=checkpoint,
        selector_report_path=selector,
        final_evidence_path=final,
        leakage_report_path=leakage,
    )

    assert checkpoint.name == "update_000"
    assert (checkpoint / "promotion.json").is_file()
    assert promotion["selector_selected_update"] == 0
    assert validate_chat_promotion(checkpoint, RUNTIME_CONFIG)[
        "selector_selected_update"
    ] == 0


def test_promotion_rejects_selected_update_that_does_not_bind_checkpoint_suffix(
    tmp_path: Path,
) -> None:
    checkpoint, selector, final, leakage = _evidence(tmp_path)
    payload = json.loads(selector.read_text(encoding="utf-8"))
    payload["selected_update"] = 8
    _write_json(selector, payload)

    with pytest.raises(ValueError, match="does not match.*checkpoint suffix"):
        create_chat_promotion(
            runtime_config_path=RUNTIME_CONFIG,
            checkpoint=checkpoint,
            selector_report_path=selector,
            final_evidence_path=final,
            leakage_report_path=leakage,
        )
    assert not (checkpoint / "promotion.json").exists()


def test_promotion_creation_fails_closed_without_final_or_leakage_evidence(
    tmp_path: Path,
) -> None:
    checkpoint, selector, final, leakage = _evidence(tmp_path)
    final.unlink()
    with pytest.raises(FileNotFoundError, match="held-out final evidence"):
        create_chat_promotion(
            runtime_config_path=RUNTIME_CONFIG,
            checkpoint=checkpoint,
            selector_report_path=selector,
            final_evidence_path=final,
            leakage_report_path=leakage,
        )
    assert not (checkpoint / "promotion.json").exists()


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("duplicate_question", "distinct questions"),
        ("wrong_oracle", "oracle_directory"),
    ],
)
def test_promotion_rejects_weak_or_misdirected_leakage_evidence(
    tmp_path: Path,
    mutation: str,
    message: str,
) -> None:
    checkpoint, selector, final, leakage = _evidence(tmp_path)
    payload = json.loads(leakage.read_text(encoding="utf-8"))
    if mutation == "duplicate_question":
        payload["answers"][1]["question"] = payload["answers"][0]["question"]
    else:
        payload["oracle_directory"] = str((tmp_path / "unrelated" / "oracle").resolve())
    _write_json(leakage, payload)

    with pytest.raises(ValueError, match=message):
        create_chat_promotion(
            runtime_config_path=RUNTIME_CONFIG,
            checkpoint=checkpoint,
            selector_report_path=selector,
            final_evidence_path=final,
            leakage_report_path=leakage,
        )


def test_promotion_creation_rejects_tampered_final_prediction_artifact(
    tmp_path: Path,
) -> None:
    checkpoint, selector, final, leakage = _evidence(tmp_path)
    final_payload = json.loads(final.read_text(encoding="utf-8"))
    predictions = Path(final_payload["predictions_path"])
    with predictions.open("a", encoding="utf-8") as handle:
        handle.write("{}\n")
    with pytest.raises(ValueError, match="observed"):
        create_chat_promotion(
            runtime_config_path=RUNTIME_CONFIG,
            checkpoint=checkpoint,
            selector_report_path=selector,
            final_evidence_path=final,
            leakage_report_path=leakage,
        )
    assert not (checkpoint / "promotion.json").exists()


def test_promotion_rejects_leakage_prefix_not_used_by_primary_predictions(
    tmp_path: Path,
) -> None:
    checkpoint, selector, final, leakage = _evidence(tmp_path)
    payload = json.loads(leakage.read_text(encoding="utf-8"))
    payload["prefix_hash"] = "e" * 64
    payload["prefix_hashes"] = ["e" * 64] * 4
    for answer in payload["answers"]:
        answer["prefix_hash"] = "e" * 64
    _write_json(leakage, payload)

    with pytest.raises(ValueError, match="does not match the primary"):
        create_chat_promotion(
            runtime_config_path=RUNTIME_CONFIG,
            checkpoint=checkpoint,
            selector_report_path=selector,
            final_evidence_path=final,
            leakage_report_path=leakage,
        )


def _performance_metrics(
    *, pair_accuracy: float, changed_rate: float, grounding_error: float
) -> dict:
    return {
        "counterfactual": {
            "eligible_pairs": 3,
            "expected_change_pairs": 3,
            "malformed_pair_groups": 0,
            "pair_accuracy": pair_accuracy,
            "changed_when_expected_rate": changed_rate,
        },
        "grounding": {
            "target_count": 6,
            "prediction_count": 6,
            "coverage": 1.0,
            "mean_coordinate_error_m": grounding_error,
        },
    }


def test_final_performance_gate_rejects_counterfactual_collapse_above_text_chance() -> None:
    runtime = load_runtime_config(RUNTIME_CONFIG)
    primary = _performance_metrics(
        pair_accuracy=0.0,
        changed_rate=0.0,
        grounding_error=0.25,
    )
    chance = _performance_metrics(
        pair_accuracy=0.0,
        changed_rate=0.0,
        grounding_error=1.0,
    )
    summary = promotion_module._final_performance_summary(primary, chance, runtime)
    assert summary["performance_passed"] is False


def test_final_performance_gate_rejects_grounding_worse_than_empty_prefix() -> None:
    runtime = load_runtime_config(RUNTIME_CONFIG)
    primary = _performance_metrics(
        pair_accuracy=1.0,
        changed_rate=1.0,
        grounding_error=2.0,
    )
    chance = _performance_metrics(
        pair_accuracy=0.0,
        changed_rate=0.0,
        grounding_error=1.0,
    )
    summary = promotion_module._final_performance_summary(primary, chance, runtime)
    assert summary["performance_passed"] is False


def test_held_out_evidence_binds_sanitized_questions_to_scored_references(
    tmp_path: Path,
) -> None:
    references = tmp_path / "references.jsonl"
    reference_records = [
        {
            "scene_id": "scene_000003",
            "question_id": "q_000001",
            "question": "Which side?",
            "answer": "left",
        }
    ]
    _write_jsonl(references, reference_records)
    mismatched_records = [
        {
            "scene_id": "scene_000003",
            "question_id": "q_000001",
            "question": "How many?",
        }
    ]
    manifest_path = tmp_path / "questions.json"
    manifest = build_question_manifest(
        mismatched_records,
        source_qa_sha256=sha256_file(references),
    )
    _write_json(manifest_path, manifest.as_dict())

    with pytest.raises(ValueError, match="does not exactly match"):
        promotion_module._validate_question_manifest_binding(
            manifest_path,
            references,
        )

    source_mismatch = build_question_manifest(
        reference_records,
        source_qa_sha256="d" * 64,
    )
    _write_json(manifest_path, source_mismatch.as_dict())
    with pytest.raises(ValueError, match="not derived from"):
        promotion_module._validate_question_manifest_binding(
            manifest_path,
            references,
        )


def test_runtime_rejects_runtime_metadata_tamper_after_promotion(tmp_path: Path) -> None:
    checkpoint, *_ = _create(tmp_path)
    _write_json(checkpoint / "runtime_metadata.json", {"schema_version": 99})
    with pytest.raises(ValueError, match="invalid or stale"):
        validate_chat_promotion(checkpoint, RUNTIME_CONFIG)


def test_runtime_rejects_promotion_copied_to_a_different_checkpoint_path(
    tmp_path: Path,
) -> None:
    checkpoint, *_ = _create(tmp_path / "source")
    copied = tmp_path / "copied"
    shutil.copytree(checkpoint, copied)
    with pytest.raises(ValueError, match="invalid or stale"):
        validate_chat_promotion(copied, RUNTIME_CONFIG)


def test_promotion_creation_rejects_symlinked_evidence_before_read(tmp_path: Path) -> None:
    checkpoint, selector, final, leakage = _evidence(tmp_path)
    selector_target = tmp_path / "selector-target.json"
    selector.rename(selector_target)
    selector.symlink_to(selector_target)

    with pytest.raises(ValueError, match="selector report must not use symbolic-link"):
        create_chat_promotion(
            runtime_config_path=RUNTIME_CONFIG,
            checkpoint=checkpoint,
            selector_report_path=selector,
            final_evidence_path=final,
            leakage_report_path=leakage,
        )
    assert not (checkpoint / "promotion.json").exists()


def test_promotion_and_direct_launch_reject_checkpoint_directory_symlink(
    tmp_path: Path,
) -> None:
    checkpoint, selector, final, leakage = _evidence(tmp_path / "evidence")
    alias = tmp_path / "checkpoint-alias"
    alias.symlink_to(checkpoint, target_is_directory=True)

    with pytest.raises(ValueError, match="Checkpoint directory must not use symbolic-link"):
        create_chat_promotion(
            runtime_config_path=RUNTIME_CONFIG,
            checkpoint=alias,
            selector_report_path=selector,
            final_evidence_path=final,
            leakage_report_path=leakage,
        )

    create_chat_promotion(
        runtime_config_path=RUNTIME_CONFIG,
        checkpoint=checkpoint,
        selector_report_path=selector,
        final_evidence_path=final,
        leakage_report_path=leakage,
    )
    with pytest.raises(ValueError, match="Checkpoint directory must not use symbolic-link"):
        resolve_chat_launch(
            config_path=RUNTIME_CONFIG,
            checkpoint=alias,
            primary_pointer=None,
        )


def test_promotion_rejects_checkpoint_inside_forbidden_runtime_tree(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "data" / "oracle" / "checkpoint"
    checkpoint.mkdir(parents=True)
    (checkpoint / "adapter.safetensors").write_bytes(b"continuous-adapter")
    _write_json(checkpoint / "runtime_metadata.json", {"schema_version": 3})

    with pytest.raises(ValueError, match="forbidden runtime path"):
        promotion_module._checkpoint_files(checkpoint)


def test_promotion_rejects_nested_environment_text_in_runtime_metadata(
    tmp_path: Path,
) -> None:
    metadata = tmp_path / "runtime_metadata.json"
    _write_json(
        metadata,
        {
            "schema_version": 3,
            "dense_alignment": {"category_hint": "chair"},
        },
    )
    with pytest.raises(ValueError, match="forbidden environmental text"):
        promotion_module._validate_checkpoint_runtime_metadata(metadata)


def test_promotion_validation_never_opens_caller_supplied_config_path_alias(
    tmp_path: Path,
) -> None:
    checkpoint, *_ = _create(tmp_path / "evidence")
    oracle = tmp_path / "data" / "oracle"
    oracle.mkdir(parents=True)
    forbidden_config = oracle / "runtime.yaml"
    forbidden_config.write_text(RUNTIME_CONFIG.read_text(encoding="utf-8"), encoding="utf-8")
    config = load_runtime_config(RUNTIME_CONFIG)
    config["_config_path"] = str(forbidden_config)
    audit = FileAccessAudit([oracle], block_forbidden=True)

    with audit, pytest.raises(ValueError, match="_config_path does not match"):
        validate_chat_promotion(
            checkpoint,
            RUNTIME_CONFIG,
            config,
            record_file=audit.record,
        )

    assert str(forbidden_config.resolve()) not in audit.unique_paths


def test_primary_pointer_resolves_only_exact_hash_bound_promotion(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project = tmp_path / "project"
    runtime_root = project / "configs/runtime"
    runtime_root.mkdir(parents=True)
    safe_config = runtime_root / "gemma4.yaml"
    safe_config.write_text(RUNTIME_CONFIG.read_text(encoding="utf-8"), encoding="utf-8")
    monkeypatch.setattr(promotion_module, "PROJECT_ROOT", project)
    monkeypatch.setattr(runtime_config_module, "PROJECT_ROOT", project)
    monkeypatch.setattr(runtime_config_module, "RUNTIME_CONFIG_ROOT", runtime_root.resolve())

    checkpoint, selector, final, leakage = _evidence(tmp_path / "evidence", safe_config)
    create_chat_promotion(
        runtime_config_path=safe_config,
        checkpoint=checkpoint,
        selector_report_path=selector,
        final_evidence_path=final,
        leakage_report_path=leakage,
    )
    pointer = runtime_root / "primary.json"
    write_primary_pointer(
        pointer,
        runtime_config_path=safe_config,
        checkpoint=checkpoint,
    )
    resolved_config, resolved_checkpoint = resolve_primary_pointer(pointer)
    assert resolved_config == safe_config.resolve()
    assert resolved_checkpoint == checkpoint.resolve()

    with (checkpoint / "promotion.json").open("a", encoding="utf-8") as handle:
        handle.write("\n")
    with pytest.raises(ValueError, match="promotion hash is stale"):
        resolve_primary_pointer(pointer)


def test_primary_pointer_publication_rejects_post_promotion_map_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project = tmp_path / "project"
    runtime_root = project / "configs/runtime"
    runtime_root.mkdir(parents=True)
    safe_config = runtime_root / "gemma4.yaml"
    safe_config.write_text(RUNTIME_CONFIG.read_text(encoding="utf-8"), encoding="utf-8")
    monkeypatch.setattr(promotion_module, "PROJECT_ROOT", project)
    monkeypatch.setattr(runtime_config_module, "PROJECT_ROOT", project)
    monkeypatch.setattr(
        runtime_config_module, "RUNTIME_CONFIG_ROOT", runtime_root.resolve()
    )
    checkpoint, selector, final, leakage = _evidence(
        tmp_path / "evidence", safe_config
    )
    create_chat_promotion(
        runtime_config_path=safe_config,
        checkpoint=checkpoint,
        selector_report_path=selector,
        final_evidence_path=final,
        leakage_report_path=leakage,
    )
    maps_root = promotion_module.artifact_root(
        load_runtime_config(safe_config), "maps"
    )
    (maps_root / "scene_000003" / "voxel_map.npz").write_bytes(b"drifted")
    pointer = runtime_root / "primary.json"

    with pytest.raises(ValueError, match="voxel-map bytes changed"):
        write_primary_pointer(
            pointer,
            runtime_config_path=safe_config,
            checkpoint=checkpoint,
        )

    assert not pointer.exists()


def test_primary_pointer_rejects_symbolic_link_before_read(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project = tmp_path / "project"
    runtime_root = project / "configs/runtime"
    runtime_root.mkdir(parents=True)
    target = tmp_path / "outside.json"
    _write_json(target, {})
    pointer = runtime_root / "primary.json"
    pointer.symlink_to(target)
    monkeypatch.setattr(promotion_module, "PROJECT_ROOT", project)
    with pytest.raises(ValueError, match="symbolic-link"):
        resolve_primary_pointer(pointer)


def test_direct_chat_launch_rejects_experiment_config_before_read_and_unpromoted_best(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="training/experiment"):
        resolve_chat_launch(
            config_path="configs/experiments/gemma4_diverse28_joint_pair_v31.yaml",
            checkpoint=tmp_path / "candidate",
            primary_pointer=None,
        )

    checkpoint = tmp_path / "best"
    checkpoint.mkdir()
    (checkpoint / "adapter.safetensors").write_bytes(b"candidate")
    _write_json(checkpoint / "runtime_metadata.json", {})
    with pytest.raises(FileNotFoundError, match="not behaviorally promoted"):
        resolve_chat_launch(
            config_path=RUNTIME_CONFIG,
            checkpoint=checkpoint,
            primary_pointer=None,
        )


def test_direct_chat_launch_rejects_inherited_gemma_and_copied_alias_before_read(
    tmp_path: Path,
) -> None:
    inherited = Path("configs/gemma4_e2b.yaml").resolve()
    copied_alias = tmp_path / "copied.yaml"
    copied_alias.write_text(inherited.read_text(encoding="utf-8"), encoding="utf-8")
    audit = FileAccessAudit(block_forbidden=True)

    with audit, pytest.raises(ValueError, match="explicit legacy"):
        resolve_chat_launch(
            config_path=inherited,
            checkpoint=tmp_path / "candidate",
            primary_pointer=None,
            audit=audit,
        )
    assert str(inherited) not in audit.unique_paths

    with audit, pytest.raises(ValueError, match="explicit legacy"):
        resolve_chat_launch(
            config_path=copied_alias,
            checkpoint=tmp_path / "candidate",
            primary_pointer=None,
            audit=audit,
        )
    assert str(copied_alias.resolve()) not in audit.unique_paths


def test_direct_chat_launch_rejects_config_symlink_and_parent_alias_before_read(
    tmp_path: Path,
) -> None:
    config_alias = tmp_path / "runtime-alias.yaml"
    config_alias.symlink_to(RUNTIME_CONFIG)
    parent_alias = tmp_path / "runtime-parent"
    parent_alias.symlink_to(RUNTIME_CONFIG.parent, target_is_directory=True)
    audit = FileAccessAudit(block_forbidden=True)

    for candidate in (config_alias, parent_alias / RUNTIME_CONFIG.name):
        with audit, pytest.raises(ValueError, match="symbolic-link path components"):
            resolve_chat_launch(
                config_path=candidate,
                checkpoint=tmp_path / "candidate",
                primary_pointer=None,
                audit=audit,
            )
        assert str(RUNTIME_CONFIG) not in audit.unique_paths

@pytest.mark.parametrize(("entrypoint", "label"), [(cli_main, "cli"), (web_main, "web")])
def test_cli_and_web_reject_experiment_config_before_open(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    entrypoint,
    label: str,
) -> None:
    audit_path = tmp_path / f"{label}.json"
    result = entrypoint(
        [
            "--config",
            "configs/experiments/gemma4_diverse28_joint_pair_v31.yaml",
            "--checkpoint",
            str(tmp_path / "candidate"),
            "--audit-log",
            str(audit_path),
        ]
    )
    assert result == 2
    assert "refuses training/experiment configs" in capsys.readouterr().err
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    assert not [
        path for path in audit["loaded_files"] if "/configs/experiments/" in path
    ]
