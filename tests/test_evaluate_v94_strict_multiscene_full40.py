from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

import semantic_3d_chat.evaluation.evaluate_v94_strict_multiscene_full40 as v94_eval
from semantic_3d_chat.config import PROJECT_ROOT
from semantic_3d_chat.evaluation.baseline_io import read_jsonl
from semantic_3d_chat.evaluation.evaluate_v94_strict_multiscene_full40 import (
    _CACHE_MANIFEST_FIELDS,
    _PREDICTION_FIELDS,
    CACHE_ARTIFACT,
    EXPECTED_SCENE_IDS,
    MEMORY_SHAPE,
    PAIR_SCENE,
    PREDICTION_ARTIFACT,
    _canonical_sha256,
    _runtime_audit,
    _validate_predictions,
    load_evaluation_memory_cache_v94,
    score_records_v94,
    shuffle_atlas_values_v94,
    zero_environment_payload_v94,
)
from semantic_3d_chat.evaluation.question_manifest import load_question_manifest
from semantic_3d_chat.evaluation.v56_fresh_development_score import (
    EXPECTED_TYPE_COUNTS,
)
from semantic_3d_chat.evaluation.v75_official_validation_contract import (
    DEFAULT_QUESTIONS_MANIFEST,
)
from semantic_3d_chat.language.prefix_injection import prefix_sha256
from semantic_3d_chat.language.v81_structured_dense_atlas_sidecar import (
    split_v75_v2_prefix_v81,
)


def _memory() -> torch.Tensor:
    generator = torch.Generator().manual_seed(94)
    return torch.randn(MEMORY_SHAPE, generator=generator, dtype=torch.bfloat16)


def _source_hashes() -> dict[str, str]:
    names = (
        "source_runtime_config_sha256",
        "source_v85_adapter_sha256",
        "source_v85_metadata_sha256",
        "source_controller_weights_sha256",
        "source_controller_metadata_sha256",
        "source_probe_weights_sha256",
        "source_probe_metadata_sha256",
    )
    return {name: f"{index:064x}" for index, name in enumerate(names, 1)}


def _build_cache(root: Path) -> tuple[dict[str, str], dict[str, str]]:
    root.mkdir()
    entries: dict[str, dict[str, object]] = {}
    hashes: dict[str, str] = {}
    for index, scene_id in enumerate(EXPECTED_SCENE_IDS):
        memory = (_memory().float() + index).to(torch.bfloat16)
        path = root / f"{scene_id}.safetensors"
        save_file({"scene_memory": memory}, str(path))
        digest = __import__("hashlib").sha256(path.read_bytes()).hexdigest()
        hashes[scene_id] = prefix_sha256(memory)
        entries[scene_id] = {
            "filename": path.name,
            "file_sha256": digest,
            "file_size_bytes": path.stat().st_size,
            "memory_sha256": hashes[scene_id],
        }
    sources = _source_hashes()
    manifest = {
        "artifact": CACHE_ARTIFACT,
        "schema_version": 1,
        "scene_ids": list(EXPECTED_SCENE_IDS),
        "scene_count": 6,
        "shape_each": list(MEMORY_SHAPE),
        "dtype": "bfloat16",
        "compiled_before_questions": True,
        "question_inputs_used": False,
        "question_dependent_retrieval": False,
        "all_memory_slots_retained": True,
        "environmental_text_inputs": [],
        **sources,
        "scenes": entries,
    }
    assert set(manifest) == _CACHE_MANIFEST_FIELDS
    (root / "manifest.json").write_text(
        json.dumps(manifest, sort_keys=True), encoding="utf-8"
    )
    return sources, hashes


def test_v94_controls_preserve_exact_layout_and_boundaries() -> None:
    memory = _memory()
    zero = zero_environment_payload_v94(memory)
    shuffled = shuffle_atlas_values_v94(memory)
    original = split_v75_v2_prefix_v81(memory)
    changed = split_v75_v2_prefix_v81(shuffled)

    assert zero.shape == shuffled.shape == memory.shape
    assert torch.equal(zero[:, :1], memory[:, :1])
    assert torch.equal(zero[:, -1:], memory[:, -1:])
    assert torch.count_nonzero(zero[:, 1:-1]).item() == 0
    assert torch.equal(changed.probe_keys, original.probe_keys)
    assert torch.equal(changed.base_latents, original.base_latents)
    assert torch.equal(changed.atlas_values, original.atlas_values.roll(1, 1))
    assert not torch.equal(shuffled, memory)


def test_v94_numeric_cache_round_trip_is_exact_and_rejects_extra_files(
    tmp_path: Path,
) -> None:
    cache = tmp_path / "evaluation_cache"
    sources, expected_hashes = _build_cache(cache)
    memories, manifest = load_evaluation_memory_cache_v94(
        cache, expected_source_hashes=sources
    )

    assert set(memories) == set(EXPECTED_SCENE_IDS)
    assert manifest["question_inputs_used"] is False
    assert {
        scene_id: prefix_sha256(memory) for scene_id, memory in memories.items()
    } == expected_hashes

    (cache / "unexpected.txt").write_text("no", encoding="utf-8")
    with pytest.raises(ValueError, match="unexpected files"):
        load_evaluation_memory_cache_v94(cache, expected_source_hashes=sources)


def test_v94_predictor_audit_blocks_labels_but_allows_sanitized_questions() -> None:
    labels = PROJECT_ROOT / "data_diverse52/qa/validation.jsonl"
    questions = PROJECT_ROOT / DEFAULT_QUESTIONS_MANIFEST
    audit = _runtime_audit()

    with audit:
        assert questions.read_bytes()
        with pytest.raises(PermissionError, match="Blocked forbidden"):
            labels.read_bytes()
    assert str(labels.resolve()) in audit.forbidden_accesses()


def test_v94_predictor_binds_all_memories_before_opening_questions(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    events: list[str] = []
    config = {
        "outputs": {
            "evaluation_predictions": str(tmp_path / "predictions.jsonl"),
            "evaluation_question_manifest": str(
                PROJECT_ROOT / DEFAULT_QUESTIONS_MANIFEST
            ),
        }
    }

    def load_memories(*_args: object, **_kwargs: object) -> tuple[dict, dict, dict]:
        events.append("all_six_memories_bound")
        return {}, {}, {}

    class QuestionBoundaryReached(RuntimeError):
        pass

    def open_questions(_path: object) -> None:
        events.append("sanitized_questions_opened")
        raise QuestionBoundaryReached

    monkeypatch.setattr(v94_eval, "_load_config", lambda _path: config)
    monkeypatch.setattr(
        v94_eval, "_load_bound_memories_before_questions", load_memories
    )
    monkeypatch.setattr(
        v94_eval, "validate_official_question_manifest", open_questions
    )

    with pytest.raises(QuestionBoundaryReached):
        v94_eval.predict_question_only_v94(tmp_path / "config.yaml")
    assert events == ["all_six_memories_bound", "sanitized_questions_opened"]


def _prediction_row(reference: dict[str, object], answer: str) -> dict[str, object]:
    scene_id = str(reference["scene_id"])
    digest = "a" * 64
    return {
        "artifact": PREDICTION_ARTIFACT,
        "scene_id": scene_id,
        "question_id": str(reference["question_id"]),
        "paired_scene_id": PAIR_SCENE[scene_id],
        "v94_prediction": answer,
        "v85_parent_prediction": "unknown",
        "paired_wrong_prediction": "unknown",
        "zero_payload_prediction": "unknown",
        "shuffled_atlas_prediction": "unknown",
        "memory_sha256": digest,
        "paired_memory_sha256": digest,
        "zero_memory_sha256": "b" * 64,
        "shuffled_memory_sha256": "c" * 64,
        "prefix_hash_unchanged": True,
        "elapsed_seconds": 1.0,
        "provenance_sha256": "d" * 64,
    }


def _gates() -> dict[str, object]:
    result: dict[str, object] = {
        "canonical_accuracy_minimum": 0.65,
        "canonical_accuracy_margin_over_exact_v85_same_216_comparator": 0.05,
        "changed_side_correct_minimum": 14,
        "complete_changed_units_minimum": 6,
        "canonical_prediction_changing_units_minimum": 8,
        "zero_payload_prediction_change_minimum": 6,
        "mean_changed_side_wrong_minus_correct_nll_minimum": 0.15,
        "zero_payload_mean_nll_gap_minimum": 0.5,
        "protected_read_count_maximum": 0,
    }
    minima = {
        "attribute": 24,
        "count": 38,
        "metric": 5,
        "orientation": 5,
        "presence": 30,
        "spatial_relation": 29,
        "support": 16,
    }
    result.update({f"{name}_correct_minimum": value for name, value in minima.items()})
    return result


def test_v94_separate_scorer_compares_exact_same_216_rows() -> None:
    references = read_jsonl(PROJECT_ROOT / "data_diverse52/qa/validation.jsonl")
    assert len(references) == 216
    assert dict(
        sorted(__import__("collections").Counter(
            str(row["answer_type"]) for row in references
        ).items())
    ) == EXPECTED_TYPE_COUNTS
    predictions = [
        _prediction_row(reference, str(reference["answer"]))
        for reference in references
    ]
    metrics = score_records_v94(
        references,
        predictions,
        gates=_gates(),
        nll_metrics={
            "mean_changed_wrong_minus_correct_nll": 0.25,
            "zero_payload_mean_nll_gap": 0.75,
            "shuffled_atlas_mean_nll_gap": 0.25,
        },
        protected_read_count=0,
    )

    assert metrics["v94"]["correct"] == 216
    assert metrics["exact_v85_same_216_comparator"]["total"] == 216
    assert metrics["counterfactual"]["canonical_complete_units"] == 12
    assert metrics["runtime_candidate_gate_passed"] is True
    assert metrics["automatic_runtime_promotion"] is False


def test_v94_prediction_artifact_has_no_labels_and_exact_question_coverage(
    tmp_path: Path,
) -> None:
    manifest = load_question_manifest(PROJECT_ROOT / DEFAULT_QUESTIONS_MANIFEST)
    references = read_jsonl(PROJECT_ROOT / "data_diverse52/qa/validation.jsonl")
    rows = [
        _prediction_row(reference, str(reference["answer"]))
        for reference in references
    ]
    provenance = {
        "artifact": PREDICTION_ARTIFACT,
        "schema_version": 1,
        "labels_opened": False,
    }
    provenance["provenance_sha256"] = _canonical_sha256(provenance)
    for row in rows:
        row["provenance_sha256"] = provenance["provenance_sha256"]
        assert set(row) == _PREDICTION_FIELDS
        assert not ({"answer", "reference_answer", "answer_type"} & set(row))
    output = tmp_path / "predictions.jsonl"
    output.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    (tmp_path / "predictions.jsonl.provenance.json").write_text(
        json.dumps(provenance, sort_keys=True), encoding="utf-8"
    )

    validated, loaded_provenance = _validate_predictions(output, manifest)
    assert len(validated) == 216
    assert loaded_provenance["labels_opened"] is False
