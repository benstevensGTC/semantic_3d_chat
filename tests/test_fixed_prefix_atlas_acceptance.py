from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from semantic_3d_chat.chat.fixed_prefix_runtime import FixedPrefixAtlasChatRuntime
from semantic_3d_chat.evaluation import fixed_prefix_atlas_gate as gate
from semantic_3d_chat.evaluation import fixed_prefix_atlas_leakage as leakage
from semantic_3d_chat.evaluation.prediction_artifacts import (
    PredictionProvenance,
    provenance_path_for,
)
from semantic_3d_chat.evaluation.question_manifest import build_question_manifest


def _jsonl_bytes(rows: list[dict]) -> bytes:
    return b"".join(
        (json.dumps(row, sort_keys=True, allow_nan=False) + "\n").encode("utf-8") for row in rows
    )


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _prepared_run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, object]:
    references = [
        {
            "scene_id": "scene_000101",
            "question_id": "q_000001",
            "question": "Is there a chair?",
            "answer": "yes",
            "answer_type": "presence",
        },
        {
            "scene_id": "scene_000101",
            "question_id": "q_000002",
            "question": "How many chairs are present?",
            "answer": "two",
            "answer_type": "count",
            "count": 2,
        },
    ]
    reference_bytes = _jsonl_bytes(references)
    references_path = tmp_path / "scorer" / "references.jsonl"
    references_path.parent.mkdir()
    references_path.write_bytes(reference_bytes)
    manifest = build_question_manifest(
        references,
        source_qa_sha256=_sha256(reference_bytes),
    )
    questions_path = tmp_path / "questions" / "manifest.json"
    questions_path.parent.mkdir()
    questions_path.write_text(
        json.dumps(manifest.as_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    config_path = tmp_path / "runtime.yaml"
    config_path.write_text("runtime: true\n", encoding="utf-8")
    base_checkpoint = tmp_path / "base"
    atlas_checkpoint = tmp_path / "atlas"
    base_checkpoint.mkdir()
    atlas_checkpoint.mkdir()

    expected = PredictionProvenance(
        config_path=str(config_path.resolve()),
        config_sha256="a" * 64,
        config_file_sha256="b" * 64,
        checkpoint_path=f"{base_checkpoint.resolve()} + {atlas_checkpoint.resolve()}",
        checkpoint_sha256="c" * 64,
        checkpoint_files=(),
        references_path=str(questions_path.resolve()),
        references_sha256=_sha256(questions_path.read_bytes()),
        scene_map_manifest_sha256="d" * 64,
        scene_map_manifest={
            "scene_000101": {
                "voxel_map_sha256": "e" * 64,
                "voxel_map_size_bytes": 17,
            }
        },
        split="validation",
        run_kind=gate.RUN_KIND,
        condition=gate.CONDITION,
    )
    monkeypatch.setattr(gate, "_expected_provenance", lambda **_kwargs: expected)
    prefix = "f" * 64
    predictions = [
        {
            "scene_id": "scene_000101",
            "question_id": "q_000001",
            "predicted_answer": "yes",
            "grounding_xyz": [0.0, 0.1, 0.2],
            "grounding_confidence": 0.8,
            "prefix_hash": prefix,
            "generated_tokens": 1,
            "elapsed_seconds": 0.1,
            "question_dependent_scene_processing": False,
            "language_model_environment_conditioning_question_dependent": False,
            "auxiliary_grounding_question_conditioned": True,
            "auxiliary_grounding_affects_language_model": False,
            "provenance_sha256": expected.sha256,
        },
        {
            "scene_id": "scene_000101",
            "question_id": "q_000002",
            "predicted_answer": "three",
            "grounding_xyz": [0.2, 0.3, 0.4],
            "grounding_confidence": 0.7,
            "prefix_hash": prefix,
            "generated_tokens": 1,
            "elapsed_seconds": 0.2,
            "question_dependent_scene_processing": False,
            "language_model_environment_conditioning_question_dependent": False,
            "auxiliary_grounding_question_conditioned": True,
            "auxiliary_grounding_affects_language_model": False,
            "provenance_sha256": expected.sha256,
        },
    ]
    predictions_path = tmp_path / "predictions" / "atlas.jsonl"
    predictions_path.parent.mkdir()
    predictions_path.write_bytes(_jsonl_bytes(predictions))
    provenance_path_for(predictions_path).write_text(
        json.dumps(expected.as_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return {
        "config_path": config_path,
        "questions_path": questions_path,
        "base_checkpoint": base_checkpoint,
        "atlas_checkpoint": atlas_checkpoint,
        "predictions_path": predictions_path,
        "references_path": references_path,
        "references_sha256": _sha256(reference_bytes),
        "expected": expected,
        "predictions": predictions,
    }


def _score_kwargs(tmp_path: Path, prepared: dict[str, object]) -> dict[str, object]:
    return {
        "config_path": prepared["config_path"],
        "questions_manifest": prepared["questions_path"],
        "base_checkpoint": prepared["base_checkpoint"],
        "atlas_checkpoint": prepared["atlas_checkpoint"],
        "predictions_path": prepared["predictions_path"],
        "references_path": prepared["references_path"],
        "expected_references_sha256": prepared["references_sha256"],
        "split": "validation",
        "minimum_normalized_exact_accuracy": 0.5,
        "launch_claim_path": tmp_path / "terminal" / "claim.json",
        "output_path": tmp_path / "terminal" / "metrics.json",
    }


def test_fixed_atlas_terminal_authenticates_and_reports_standard_metrics(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prepared = _prepared_run(tmp_path, monkeypatch)
    kwargs = _score_kwargs(tmp_path, prepared)

    report = gate.score_fixed_prefix_atlas(**kwargs)

    assert report["passed"] is True
    assert report["integrity_passed"] is True
    assert report["metrics"]["normalized_exact_accuracy"] == 0.5
    assert report["metrics"]["count"]["accuracy"] == 0.0
    assert report["metrics"]["presence"]["f1"] == 1.0
    assert report["strict_fixed_prefix"]["prefix_invariant_within_every_scene"] is True
    assert report["strict_fixed_prefix"]["question_count"] == 2
    assert Path(kwargs["launch_claim_path"]).is_file()
    assert Path(kwargs["output_path"]).is_file()


def test_fixed_atlas_terminal_rejects_provenance_before_consuming_attempt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prepared = _prepared_run(tmp_path, monkeypatch)
    provenance_path = provenance_path_for(prepared["predictions_path"])
    value = json.loads(provenance_path.read_text(encoding="utf-8"))
    value["condition"] = "question_selected_points"
    provenance_path.write_text(json.dumps(value), encoding="utf-8")
    kwargs = _score_kwargs(tmp_path, prepared)

    with pytest.raises(ValueError, match="exact runtime inputs"):
        gate.score_fixed_prefix_atlas(**kwargs)

    assert not Path(kwargs["launch_claim_path"]).exists()
    assert not Path(kwargs["output_path"]).exists()


def test_fixed_atlas_terminal_rejects_question_changed_prefix_before_references(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prepared = _prepared_run(tmp_path, monkeypatch)
    rows = list(prepared["predictions"])
    rows[1] = {**rows[1], "prefix_hash": "1" * 64}
    Path(prepared["predictions_path"]).write_bytes(_jsonl_bytes(rows))
    kwargs = _score_kwargs(tmp_path, prepared)

    with pytest.raises(ValueError, match="changed a scene's fixed"):
        gate.score_fixed_prefix_atlas(**kwargs)

    assert not Path(kwargs["launch_claim_path"]).exists()


def test_fixed_atlas_reference_failure_occurs_after_immutable_claim(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prepared = _prepared_run(tmp_path, monkeypatch)
    bad_references = [
        {
            "scene_id": "scene_000101",
            "question_id": "q_000002",
            "question": "Different order",
            "answer": "two",
            "answer_type": "count",
        }
    ]
    bad_bytes = _jsonl_bytes(bad_references)
    Path(prepared["references_path"]).write_bytes(bad_bytes)
    kwargs = _score_kwargs(tmp_path, prepared)
    kwargs["expected_references_sha256"] = _sha256(bad_bytes)

    with pytest.raises(ValueError, match="manifest's source hash"):
        gate.score_fixed_prefix_atlas(**kwargs)

    assert Path(kwargs["launch_claim_path"]).is_file()
    assert not Path(kwargs["output_path"]).exists()


def test_v67_launch_requires_explicit_preregistration_before_reference_open(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prepared = _prepared_run(tmp_path, monkeypatch)
    kwargs = _score_kwargs(tmp_path, prepared)
    kwargs["launch_claim_path"] = tmp_path / "terminal" / "v67_claim.json"
    reference_opened = False

    def forbidden_reference_open(*_args: object, **_kwargs: object) -> object:
        nonlocal reference_opened
        reference_opened = True
        raise AssertionError("reference input was opened")

    monkeypatch.setattr(gate, "_load_references", forbidden_reference_open)

    with pytest.raises(ValueError, match="explicit --preregistration"):
        gate.score_fixed_prefix_atlas(**kwargs)

    assert reference_opened is False
    assert not Path(kwargs["launch_claim_path"]).exists()


@pytest.mark.parametrize("v67_gate_passed", [True, False])
def test_v67_gate_is_sealed_before_reference_and_controls_terminal_status(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    v67_gate_passed: bool,
) -> None:
    prepared = _prepared_run(tmp_path, monkeypatch)
    kwargs = _score_kwargs(tmp_path, prepared)
    preregistration = tmp_path / "public" / "v67_preregistration.json"
    preregistration.parent.mkdir()
    preregistration.write_text("{}\n", encoding="utf-8")
    kwargs["preregistration_path"] = preregistration
    reference_opened = False
    expected_boundary = {
        "preregistration_sha256": "1" * 64,
        "terminal_gate_source_sha256": "2" * 64,
    }
    in_memory_contract = {"thresholds": {"normalized_exact_accuracy": {"minimum": 0.5}}}

    def fake_boundary(**_kwargs: object) -> tuple[dict, dict]:
        assert reference_opened is False
        return expected_boundary, in_memory_contract

    original_load_references = gate._load_references

    def tracked_reference_open(*args: object, **call_kwargs: object) -> object:
        nonlocal reference_opened
        claim_path = Path(kwargs["launch_claim_path"])
        assert claim_path.is_file()
        claim = json.loads(claim_path.read_text(encoding="utf-8"))
        assert claim["v67_strict_atlas"]["source_boundary_validated_before_reference_open"]
        assert claim["v67_strict_atlas"]["preregistration_sha256"] == "1" * 64
        result = original_load_references(*args, **call_kwargs)
        reference_opened = True
        return result

    original_sha256_file = gate.sha256_file

    def reject_post_reference_input_hash(path: str | Path) -> str:
        if reference_opened:
            raise AssertionError(f"input reopened after scorer reference: {path}")
        return original_sha256_file(path)

    v67_result = {
        "passed": v67_gate_passed,
        "checks": {"natural_canonical_exact": v67_gate_passed},
        "metrics": {"natural_canonical_exact": 2 if v67_gate_passed else 0},
        "thresholds": {"natural_canonical_exact": {"minimum": 1, "total": 2}},
    }
    monkeypatch.setattr(gate, "_validate_v67_pre_reference_boundary", fake_boundary)
    monkeypatch.setattr(gate, "_load_references", tracked_reference_open)
    monkeypatch.setattr(gate, "sha256_file", reject_post_reference_input_hash)
    monkeypatch.setattr(gate, "_score_v67_terminal_metrics", lambda **_kwargs: v67_result)

    report = gate.score_fixed_prefix_atlas(**kwargs)

    assert reference_opened is True
    assert report["passed"] is v67_gate_passed
    assert report["status"] == ("terminal_pass" if v67_gate_passed else "terminal_fail")
    assert report["v67_strict_atlas"]["source_boundary"] == expected_boundary
    assert report["v67_strict_atlas"]["terminal_gate"] == v67_result
    claim_bytes = Path(kwargs["launch_claim_path"]).read_bytes()
    assert report["inputs"]["launch_claim_sha256"] == hashlib.sha256(claim_bytes).hexdigest()


def test_training_root_isolation_restores_after_exception(tmp_path: Path) -> None:
    training = tmp_path / "data_gemma4" / "training"
    training.mkdir(parents=True)
    marker = training / "numeric.bin"
    marker.write_bytes(b"continuous")

    with (
        pytest.raises(RuntimeError, match="synthetic failure"),
        leakage.training_root_temporarily_unavailable(training) as state,
    ):
        assert state.renamed is True
        assert not training.exists()
        assert state.hidden is not None and state.hidden.is_dir()
        raise RuntimeError("synthetic failure")

    assert marker.read_bytes() == b"continuous"


def test_fixed_atlas_leakage_hides_training_and_requires_atlas_file_reads(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    training = tmp_path / "data_gemma4" / "training"
    training.mkdir(parents=True)
    (training / "teacher.bin").write_bytes(b"training-only")
    atlas = tmp_path / "atlas"
    atlas.mkdir()
    for name in ("atlas.safetensors", "runtime_metadata.json"):
        (atlas / name).write_bytes(b"numeric")
    report_path = tmp_path / "leakage.json"

    class Audit:
        def __init__(self) -> None:
            self.forbidden_roots: list[Path] = []
            self.paths: list[str] = []

        def record(self, path: str | Path) -> None:
            self.paths.append(str(Path(path).resolve()))

    runtime = SimpleNamespace()

    def fake_load(
        _cls: type,
        _config: dict,
        _scene: str,
        *,
        base_checkpoint: str | Path,
        atlas_checkpoint: str | Path,
        audit: Audit,
        local_files_only: bool,
    ) -> object:
        assert base_checkpoint
        assert local_files_only is True
        assert not training.exists()
        for name in ("atlas.safetensors", "runtime_metadata.json"):
            audit.record(Path(atlas_checkpoint) / name)
        return runtime

    monkeypatch.setattr(
        FixedPrefixAtlasChatRuntime,
        "load",
        classmethod(fake_load),
    )

    def fake_runner(**kwargs: object) -> dict:
        assert not training.exists()
        audit = Audit()
        kwargs["runtime_loader"]({}, "scene_000101", tmp_path / "base", audit)
        return {
            "passed": True,
            "loaded_files": audit.paths,
            "forbidden_accesses": [],
            "prefix_invariant": True,
            "prefix_computed_before_first_question": True,
            "oracle_unavailable_during_inference": True,
            "oracle_restored": True,
        }

    monkeypatch.setattr(leakage, "run_leakage_evaluation", fake_runner)

    report = leakage.run_fixed_prefix_atlas_leakage(
        config_path=tmp_path / "runtime.yaml",
        scene_id="scene_000101",
        base_checkpoint=tmp_path / "base",
        atlas_checkpoint=atlas,
        training_directory=training,
        report_path=report_path,
    )

    assert report["passed"] is True
    assert report["training_directory_unavailable_during_inference"] is True
    assert report["training_directory_restored"] is True
    assert report["training_artifact_loaded_paths"] == []
    assert report["atlas_checkpoint_files_complete"] is True
    assert (training / "teacher.bin").read_bytes() == b"training-only"
    assert json.loads(report_path.read_text(encoding="utf-8"))["passed"] is True
