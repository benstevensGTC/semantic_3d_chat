from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from semantic_3d_chat.evaluation import v94_strict_multiscene_evidence as evidence
from semantic_3d_chat.evaluation.question_manifest import QuestionManifest, QuestionRecord


def _questions(path: Path) -> QuestionManifest:
    rows = tuple(
        QuestionRecord(
            scene_id=scene,
            question_id=f"q_{scene_index * 36 + ordinal:06d}",
            question=f"Shared question {ordinal}?",
        )
        for scene_index, scene in enumerate(evidence.SCENE_IDS)
        for ordinal in range(36)
    )
    return QuestionManifest(
        questions=rows,
        questions_sha256="a" * 64,
        source_qa_sha256="b" * 64,
        manifest_path=path,
        manifest_sha256="c" * 64,
    )


def _cache_hashes() -> dict[str, object]:
    return {
        "memory_hashes": {
            scene: f"{index + 1:064x}"
            for index, scene in enumerate(evidence.SCENE_IDS)
        },
        "zero_hashes": {
            scene: f"{index + 101:064x}"
            for index, scene in enumerate(evidence.SCENE_IDS)
        },
        "shuffled_hashes": {
            scene: f"{index + 201:064x}"
            for index, scene in enumerate(evidence.SCENE_IDS)
        },
    }


def _prediction_rows(
    questions: QuestionManifest,
    cache: dict[str, object],
    provenance_sha256: str = "d" * 64,
) -> list[dict[str, object]]:
    result: list[dict[str, object]] = []
    for item in questions.questions:
        ordinal = int(item.question.split()[-1].rstrip("?"))
        paired = evidence.PAIR_SCENE[item.scene_id]
        result.append(
            {
                "artifact": "v94_question_only_same216_predictions_v1",
                "scene_id": item.scene_id,
                "question_id": item.question_id,
                "paired_scene_id": paired,
                "v94_prediction": f"answer-{item.scene_id}-{ordinal}",
                "v85_parent_prediction": "parent",
                "paired_wrong_prediction": f"answer-{paired}-{ordinal}",
                "zero_payload_prediction": "zero",
                "shuffled_atlas_prediction": "atlas-value-shuffle",
                "memory_sha256": cache["memory_hashes"][item.scene_id],  # type: ignore[index]
                "paired_memory_sha256": cache["memory_hashes"][paired],  # type: ignore[index]
                "zero_memory_sha256": cache["zero_hashes"][item.scene_id],  # type: ignore[index]
                "shuffled_memory_sha256": cache["shuffled_hashes"][item.scene_id],  # type: ignore[index]
                "prefix_hash_unchanged": True,
                "elapsed_seconds": float(len(result)),
                "provenance_sha256": provenance_sha256,
            }
        )
    return result


def test_prediction_rows_bind_every_cache_and_control_hash(tmp_path: Path) -> None:
    questions = _questions(tmp_path / "questions.json")
    cache = _cache_hashes()
    rows = _prediction_rows(questions, cache)

    diagnostics = evidence._authenticate_prediction_rows(
        rows,
        questions=questions,
        cache=cache,
        provenance_sha256="d" * 64,
    )

    assert diagnostics["paired_wrong_direct_pair_comparable_count"] == 216
    assert diagnostics["paired_wrong_matches_direct_paired_scene_prediction_count"] == 216
    assert diagnostics["diagnostic_scope"].startswith("posthoc_non_preregistered")


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        ("memory_sha256", "e" * 64),
        ("paired_memory_sha256", "e" * 64),
        ("zero_memory_sha256", "e" * 64),
        ("shuffled_memory_sha256", "e" * 64),
        ("prefix_hash_unchanged", False),
    ],
)
def test_prediction_rows_reject_plausible_but_unbound_hashes(
    tmp_path: Path, field: str, replacement: object
) -> None:
    questions = _questions(tmp_path / "questions.json")
    cache = _cache_hashes()
    rows = _prediction_rows(questions, cache)
    rows[17][field] = replacement

    with pytest.raises(ValueError, match="not bound to cached memory"):
        evidence._authenticate_prediction_rows(
            rows,
            questions=questions,
            cache=cache,
            provenance_sha256="d" * 64,
        )


def test_prediction_rows_reject_incomplete_resume(tmp_path: Path) -> None:
    questions = _questions(tmp_path / "questions.json")
    cache = _cache_hashes()
    rows = _prediction_rows(questions, cache)

    with pytest.raises(ValueError, match="exact 216"):
        evidence._authenticate_prediction_rows(
            rows[:-1],
            questions=questions,
            cache=cache,
            provenance_sha256="d" * 64,
        )


def test_prediction_bundle_rejects_asymmetric_resume_files(tmp_path: Path) -> None:
    prediction_path = tmp_path / "predictions.jsonl"
    prediction_path.write_text("{}\n", encoding="utf-8")
    config = {"outputs": {"evaluation_predictions": prediction_path.name}}
    questions = _questions(tmp_path / "questions.json")

    with pytest.raises(FileNotFoundError, match="prediction provenance"):
        evidence._authenticate_predictions(  # type: ignore[arg-type]
            tmp_path,
            config,
            config_path=tmp_path / "config.yaml",
            questions=questions,
            questions_path=tmp_path / "questions.json",
            cache={},
            candidate={},
        )


def test_provenance_is_bound_to_current_config_question_cache_and_candidate(
    tmp_path: Path,
) -> None:
    questions = _questions(tmp_path / "questions.json")
    cache = {"manifest_canonical_sha256": "3" * 64}
    candidate = {
        "candidate_weights_sha256": "4" * 64,
        "candidate_state_sha256": "5" * 64,
    }
    unsigned = {
        "artifact": "v94_question_only_same216_predictions_v1",
        "schema_version": 1,
        "config_sha256": "2" * 64,
        "question_manifest_sha256": questions.manifest_sha256,
        "questions_sha256": questions.questions_sha256,
        "memory_manifest_sha256": cache["manifest_canonical_sha256"],
        "candidate_weights_sha256": candidate["candidate_weights_sha256"],
        "candidate_state_sha256": candidate["candidate_state_sha256"],
        "scene_ids": list(evidence.SCENE_IDS),
        "row_count": 216,
        "arms": list(evidence.ARMS),
        "labels_opened": False,
        "questions_opened_after_all_memories_bound": True,
        "question_dependent_retrieval": False,
        "environmental_text_inputs": [],
    }
    provenance = {
        **unsigned,
        "provenance_sha256": evidence._canonical_sha256(unsigned),
    }

    assert evidence._authenticate_prediction_provenance(
        provenance,
        config_sha256="2" * 64,
        questions=questions,
        cache=cache,
        candidate=candidate,
    ) == provenance["provenance_sha256"]

    stale = dict(provenance)
    stale["candidate_weights_sha256"] = "6" * 64
    stale["provenance_sha256"] = evidence._canonical_sha256(
        {key: value for key, value in stale.items() if key != "provenance_sha256"}
    )
    with pytest.raises(ValueError, match="self-consistent but not current"):
        evidence._authenticate_prediction_provenance(
            stale,
            config_sha256="2" * 64,
            questions=questions,
            cache=cache,
            candidate=candidate,
        )


def _access_fixture(tmp_path: Path) -> tuple[dict[str, object], dict[str, object]]:
    (tmp_path / "data" / "oracle").mkdir(parents=True)
    (tmp_path / "data" / "qa").mkdir(parents=True)
    (tmp_path / "data_diverse52" / "qa").mkdir(parents=True)
    config_path = (tmp_path / "config.yaml").resolve()
    question_path = (tmp_path / "reports" / "questions" / "questions.json").resolve()
    manifest_path = (tmp_path / "cache" / "manifest.json").resolve()
    memory_paths = {
        scene: (tmp_path / "cache" / f"{scene}.safetensors").resolve()
        for scene in evidence.SCENE_IDS
    }
    candidate_weights = (tmp_path / "candidate" / "bridge.safetensors").resolve()
    candidate_metadata = (tmp_path / "candidate" / "runtime_metadata.json").resolve()
    config = {
        "sources": {
            "runtime_config": "runtime.yaml",
            "frozen_v85_checkpoint": "v85",
            "evaluation_memory_controller": "controller",
            "evaluation_probe_bank": "probes",
            "evaluation_qa_reserved_for_label_scorer": (
                "data_diverse52/qa/validation.jsonl"
            ),
        },
        "outputs": {"evaluation_predictions": "predictions.jsonl"},
    }
    candidate = {
        "candidate_weights_path": candidate_weights,
        "candidate_metadata_path": candidate_metadata,
    }
    cache = {"manifest_path": manifest_path, "memory_paths": memory_paths}
    mandatory = {
        str(config_path),
        str(question_path),
        str(manifest_path),
        *(str(path) for path in memory_paths.values()),
        str(candidate_weights),
        str(candidate_metadata),
        str((tmp_path / "runtime.yaml").resolve()),
        str((tmp_path / "v85" / "adapter.safetensors").resolve()),
        str((tmp_path / "v85" / "runtime_metadata.json").resolve()),
        str((tmp_path / "controller" / "control.safetensors").resolve()),
        str((tmp_path / "controller" / "runtime_metadata.json").resolve()),
        str((tmp_path / "probes" / "probes.safetensors").resolve()),
        str((tmp_path / "probes" / "runtime_metadata.json").resolve()),
    }
    access = {
        "loaded_files": sorted(mandatory),
        "forbidden_roots": sorted(evidence._expected_forbidden_roots(tmp_path)),
        "forbidden_component_names": ["oracle"],
        "block_forbidden": True,
        "forbidden_accesses": [],
        "passed": True,
    }
    context = {
        "config_path": config_path,
        "questions_path": question_path,
        "cache": cache,
        "candidate": candidate,
        "config": config,
    }
    return access, context


def test_completed_access_log_is_required_and_label_free(tmp_path: Path) -> None:
    access, context = _access_fixture(tmp_path)
    evidence._authenticate_access_log(tmp_path, access, **context)  # type: ignore[arg-type]

    label = str((tmp_path / "data_diverse52" / "qa" / "validation.jsonl").resolve())
    access["loaded_files"] = sorted([*access["loaded_files"], label])  # type: ignore[index]
    with pytest.raises(ValueError, match="protected data"):
        evidence._authenticate_access_log(  # type: ignore[arg-type]
            tmp_path, access, **context
        )


def test_atlas_value_shuffle_rolls_values_but_retains_keys_base_and_boundaries() -> None:
    memory = torch.zeros(evidence.MEMORY_SHAPE, dtype=torch.bfloat16)
    memory[:, 0] = 11
    memory[:, -1] = 12
    memory[:, 481:737] = 13
    atlas = memory[:, 1:481].reshape(1, 96, 5, 1536)
    atlas[:, :, 0] = torch.arange(96, dtype=torch.bfloat16).view(1, 96, 1)
    atlas[:, :, 1:] = (
        torch.arange(96, dtype=torch.bfloat16).view(1, 96, 1, 1) + 100
    )

    shuffled = evidence._shuffle_atlas(memory)
    original_atlas = memory[:, 1:481].reshape(1, 96, 5, 1536)
    shuffled_atlas = shuffled[:, 1:481].reshape(1, 96, 5, 1536)

    assert torch.equal(shuffled[:, :1], memory[:, :1])
    assert torch.equal(shuffled[:, 481:], memory[:, 481:])
    assert torch.equal(shuffled_atlas[:, :, 0], original_atlas[:, :, 0])
    assert torch.equal(
        shuffled_atlas[:, :, 1:], original_atlas[:, :, 1:].roll(1, dims=1)
    )


def test_score_binds_predictions_and_preserves_negative_result_authentication(
    tmp_path: Path,
) -> None:
    questions = _questions(tmp_path / "questions.json")
    score_path = tmp_path / "score.json"
    config = {
        "sources": {"evaluation_qa_sha256": "b" * 64},
        "outputs": {"evaluation_score": score_path.name},
    }
    predictions = {
        "predictions_sha256": "1" * 64,
        "prediction_provenance_sha256": "2" * 64,
        "posthoc_non_preregistered_diagnostics": {"paired": True},
    }

    def write_score(*, passed: bool) -> None:
        gates = {"prefix_hash_invariance": True, "protected_read_count_zero": True}
        if not passed:
            gates["behavior_accuracy"] = False
        score = {
            "artifact": "v94_label_isolated_same216_score_v1",
            "schema_version": 94,
            "status": (
                "passed_awaiting_separate_leakage_packaging"
                if passed
                else "measured_gate_not_passed"
            ),
            "row_count": 216,
            "scene_count": 6,
            "question_manifest_sha256": questions.manifest_sha256,
            "reference_sha256": "b" * 64,
            "predictions_sha256": "1" * 64,
            "prediction_provenance_sha256": "2" * 64,
            "labels_opened_only_by_this_scorer": True,
            "answers_or_questions_serialized": False,
            "metrics": {
                "v94": {"accuracy": 0.7},
                "zero_payload": {"accuracy": 0.2},
                "shuffled_atlas": {"accuracy": 0.3},
                "shuffled_atlas_prediction_change_count": 100,
                "runtime_candidate_gates": gates,
                "runtime_candidate_gate_passed": passed,
                "automatic_runtime_promotion": False,
            },
            "runtime_promotion_authorized": False,
        }
        score_path.write_text(json.dumps(score), encoding="utf-8")

    write_score(passed=False)
    result = evidence._authenticate_score(
        tmp_path,
        config,
        questions=questions,
        predictions=predictions,
        require_score=True,
        require_behavior_pass=False,
    )
    assert result is not None and result["behavior_gate_passed"] is False
    with pytest.raises(ValueError, match="behavior gates"):
        evidence._authenticate_score(
            tmp_path,
            config,
            questions=questions,
            predictions=predictions,
            require_score=True,
            require_behavior_pass=True,
        )

    write_score(passed=True)
    result = evidence._authenticate_score(
        tmp_path,
        config,
        questions=questions,
        predictions=predictions,
        require_score=True,
        require_behavior_pass=True,
    )
    assert result is not None
    assert result["posthoc_non_preregistered_diagnostics"][
        "v94_minus_atlas_value_shuffle_accuracy"
    ] == pytest.approx(0.4)


def _attestation_fixture(tmp_path: Path) -> tuple[dict[str, object], dict[str, object]]:
    maps_root = tmp_path / "maps"
    for scene in evidence.SCENE_IDS:
        path = maps_root / scene / "voxel_map.npz"
        path.parent.mkdir(parents=True)
        path.write_bytes(f"numeric map {scene}".encode())
    runtime = tmp_path / "runtime.yaml"
    runtime.write_text("paths:\n  maps_root: maps\n", encoding="utf-8")
    evaluator = tmp_path / "evaluator.py"
    evaluator.write_text("# sealed\n", encoding="utf-8")
    numeric = {
        "runtime_config_sha256": evidence._sha256_file(runtime),
        "frozen_v85_adapter_sha256": "1" * 64,
        "frozen_v85_metadata_sha256": "2" * 64,
        "evaluation_memory_controller_weights_sha256": "3" * 64,
        "evaluation_memory_controller_metadata_sha256": "4" * 64,
        "evaluation_probe_tensor_sha256": "5" * 64,
        "evaluation_probe_metadata_sha256": "6" * 64,
    }
    config: dict[str, object] = {
        "sources": {
            "runtime_config": runtime.name,
            "evaluation_source": evaluator.name,
            "evaluation_source_sha256": evidence._sha256_file(evaluator),
            **numeric,
        },
        "outputs": {
            "evaluation_memory_cache": "experiment/evaluation_cache",
            "evaluation_question_manifest": "reports/questions.json",
        },
    }
    cache_root = tmp_path / "experiment" / "evaluation_cache"
    cache_root.mkdir(parents=True)
    manifest = cache_root / "manifest.json"
    manifest.write_text("{}", encoding="utf-8")
    memory_paths = {}
    for scene in evidence.SCENE_IDS:
        path = cache_root / f"{scene}.safetensors"
        path.write_bytes(scene.encode())
        memory_paths[scene] = path.resolve()
    cache = {
        "manifest_path": manifest.resolve(),
        "manifest_file_sha256": evidence._sha256_file(manifest),
        "manifest_canonical_sha256": evidence._canonical_sha256({}),
        "memory_hashes": {scene: f"{index + 1:064x}" for index, scene in enumerate(evidence.SCENE_IDS)},
        "memory_paths": memory_paths,
    }
    maps = evidence._current_map_inventory(tmp_path, config)  # type: ignore[arg-type]
    attestation_root = tmp_path / "experiment" / evidence.COMPILATION_ATTESTATION_DIRECTORY
    attestation_root.mkdir()
    pre = {
        "artifact": "v94_evaluation_cache_precompile_attestation_v1",
        "schema_version": 1,
        "status": "sealed_before_cache_creation",
        "config_sha256": "7" * 64,
        "evaluator_source_sha256": evidence._sha256_file(evaluator),
        "numeric_compiler_source_sha256": evidence._numeric_compiler_sources(config),  # type: ignore[arg-type]
        "cache_path": "experiment/evaluation_cache",
        "cache_absent_before_compile": True,
        "maps": maps,
        "questions_opened": False,
        "labels_opened": False,
        "oracle_opened": False,
    }
    pre_path = attestation_root / "pre.json"
    pre_path.write_text(json.dumps(pre), encoding="utf-8")
    question_path = (tmp_path / "reports" / "questions.json").resolve()
    loaded = sorted(
        {
            *(str(tmp_path / row["path"]) for row in maps.values()),
            str(manifest.resolve()),
            *(str(path) for path in memory_paths.values()),
        }
    )
    post = {
        "artifact": "v94_evaluation_cache_postcompile_attestation_v1",
        "schema_version": 1,
        "status": "complete_cache_bound_no_questions_or_labels",
        "pre_attestation_sha256": evidence._sha256_file(pre_path),
        "config_sha256": "7" * 64,
        "maps_before": maps,
        "maps_after": maps,
        "maps_unchanged": True,
        "cache_manifest_file_sha256": cache["manifest_file_sha256"],
        "cache_manifest_canonical_sha256": cache["manifest_canonical_sha256"],
        "memory_sha256": cache["memory_hashes"],
        "loaded_files": loaded,
        "loaded_file_inventory_sha256": evidence._canonical_sha256(loaded),
        "forbidden_roots": sorted(
            {*evidence._expected_forbidden_roots(tmp_path), str(question_path)}
        ),
        "forbidden_component_names": ["oracle"],
        "block_forbidden": True,
        "forbidden_accesses": [],
        "protected_read_count": 0,
        "questions_opened": False,
        "labels_opened": False,
        "oracle_opened": False,
    }
    (attestation_root / "post.json").write_text(json.dumps(post), encoding="utf-8")
    return config, cache


def test_compilation_attestation_binds_exact_source_maps_and_access(tmp_path: Path) -> None:
    config, cache = _attestation_fixture(tmp_path)

    result = evidence._authenticate_compilation_attestation(  # type: ignore[arg-type]
        tmp_path, config, config_sha256="7" * 64, cache=cache
    )
    assert set(result) == {"pre_attestation_sha256", "post_attestation_sha256"}

    map_path = tmp_path / "maps" / evidence.SCENE_IDS[0] / "voxel_map.npz"
    map_path.write_bytes(b"changed after compilation")
    with pytest.raises(ValueError, match="precompile attestation is not current"):
        evidence._authenticate_compilation_attestation(  # type: ignore[arg-type]
            tmp_path, config, config_sha256="7" * 64, cache=cache
        )
