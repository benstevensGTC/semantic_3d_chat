from __future__ import annotations

import hashlib
import inspect
import json
from pathlib import Path
from typing import Any

import pytest
import torch
from safetensors.torch import save_file

from semantic_3d_chat.evaluation import authenticate_v95_known_development as authenticator
from semantic_3d_chat.evaluation import nll_v95_known_development as nll_module
from semantic_3d_chat.evaluation import predict_v95_known_development as predictor
from semantic_3d_chat.evaluation import score_v95_known_development as scorer
from semantic_3d_chat.evaluation import v95_known_development_common as common
from semantic_3d_chat.evaluation.question_manifest import (
    QuestionManifest,
    QuestionRecord,
    questions_sha256,
)
from semantic_3d_chat.evaluation.v85_strict_multiscene_preflight import sha256_file_v85
from semantic_3d_chat.evaluation.v95_strict_causal_successor_preflight import (
    payload_permutation_v95,
    permuted_payload_memory_v95,
    zero_payload_memory_v95,
)
from semantic_3d_chat.language.lora import tensor_state_sha256


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _questions() -> QuestionManifest:
    records = tuple(
        QuestionRecord(
            scene_id=scene_id,
            question_id=f"q_{scene_offset * 36 + offset:06d}",
            question=f"Synthetic question {scene_offset * 36 + offset}?",
        )
        for scene_offset, scene_id in enumerate(common.SCENE_IDS)
        for offset in range(36)
    )
    return QuestionManifest(
        questions=records,
        questions_sha256=questions_sha256(records),
        source_qa_sha256=common.REFERENCE_SHA256,
        manifest_path=Path("/synthetic/questions.json"),
        manifest_sha256=_digest("manifest"),
    )


def _memory_hashes() -> dict[str, dict[str, str]]:
    return {
        arm: {scene_id: _digest(f"{arm}:{scene_id}") for scene_id in common.SCENE_IDS}
        for arm in common.ARMS
    }


def _prediction_rows() -> list[dict[str, Any]]:
    manifest = _questions()
    hashes = _memory_hashes()
    provenance = _digest("provenance")
    return [
        common.prediction_row_v95(
            scene_id=row.scene_id,
            question_id=row.question_id,
            predictions={arm: f"synthetic-{arm}" for arm in common.ARMS},
            memory_hashes={arm: hashes[arm][row.scene_id] for arm in common.ARMS},
            provenance_sha256=provenance,
            unchanged=True,
        )
        for row in manifest.questions
    ]


def _structured_metrics(
    *, primary: int = 151, zero: int = 100, permutation: int = 99
) -> dict[str, Any]:
    correct = {
        common.PRIMARY: primary,
        common.ZERO_PAYLOAD: zero,
        common.FULL_INTERIOR_PERMUTATION: permutation,
        common.PAIRED_WRONG_SCENE: 98,
    }
    return {
        "arms": {
            arm: {
                "correct": value,
                "total": common.QUESTION_COUNT,
                "accuracy": value / common.QUESTION_COUNT,
                "by_answer_type": {},
            }
            for arm, value in correct.items()
        },
        "counterfactual": {
            "canonical_correct_sides": 16,
            "canonical_complete_units": 5,
            "canonical_prediction_changed_units": 8,
        },
        "comparisons": {
            arm: {
                "accuracy_drop_from_primary": (primary - correct[arm])
                / common.QUESTION_COUNT,
                "prediction_change_count": 10,
                "prediction_change_rate": 10 / common.QUESTION_COUNT,
            }
            for arm in common.ARMS[1:]
        },
    }


def _nll_metrics(*, changed_gap: float = 0.3) -> dict[str, Any]:
    return {
        "primary_mean_nll": 1.0,
        "paired_wrong_scene_mean_nll": 1.4,
        "zero_payload_mean_nll": 1.6,
        "full_interior_permutation_mean_nll": 1.5,
        "mean_wrong_minus_primary_nll": 0.4,
        "mean_changed_wrong_minus_primary_nll": changed_gap,
        "zero_payload_mean_nll_gap": 0.6,
        "full_interior_permutation_mean_nll_gap": 0.5,
        "row_count_per_arm": common.QUESTION_COUNT,
        "changed_row_count": 24,
    }


def _gate_contract() -> dict[str, Any]:
    return {
        "role": "post_fixed_final_go_no_go_not_checkpoint_selection",
        "scene_count": 6,
        "row_count": 216,
        "labels_opened_after_fixed_final_only": True,
        "labels_opened_by_separate_label_isolated_evaluator_only": True,
        "v94_reference_correct": 143,
        "v94_reference_total": 216,
        "v94_reference_changed_side_correct": 13,
        "v94_reference_complete_changed_units": 2,
        "v94_reference_prediction_changed_units": 4,
        "v95_correct_minimum": 150,
        "v95_accuracy_margin_over_v94_minimum": 0.03,
        "changed_side_correct_minimum": 15,
        "complete_changed_units_minimum": 4,
        "prediction_changed_units_minimum": 7,
        "mean_changed_side_wrong_minus_correct_nll_minimum": 0.2,
        "correct_accuracy_above_zero_payload_required": True,
        "correct_accuracy_above_full_interior_permutation_required": True,
        "correct_nll_below_zero_payload_required": True,
        "correct_nll_below_full_interior_permutation_required": True,
        "fixed_final_checkpoint_may_not_change_after_gate": True,
        "pass_required_before_deferred_final_unlock": True,
    }


def test_zero_and_full_permutation_preserve_native_boundaries() -> None:
    generator = torch.Generator().manual_seed(95)
    memory = torch.randn(common.MEMORY_SHAPE, generator=generator).to(torch.bfloat16)
    zero = zero_payload_memory_v95(memory)
    permutation = permuted_payload_memory_v95(memory, seed=950095)
    indices = payload_permutation_v95(950095)

    assert torch.equal(zero[:, :1], memory[:, :1])
    assert torch.equal(zero[:, -1:], memory[:, -1:])
    assert torch.count_nonzero(zero[:, 1:-1]).item() == 0
    assert len(set(indices.tolist())) == 736
    assert not torch.equal(indices, torch.arange(736))
    assert torch.equal(permutation[:, :1], memory[:, :1])
    assert torch.equal(permutation[:, -1:], memory[:, -1:])
    assert torch.equal(permutation[:, 1:-1], memory[:, 1:-1][:, indices])


def test_bind_all_memories_covers_all_24_arm_scene_inputs() -> None:
    generator = torch.Generator().manual_seed(950095)
    source = {
        scene_id: torch.randn(common.MEMORY_SHAPE, generator=generator).to(
            torch.bfloat16
        )
        for scene_id in common.SCENE_IDS
    }
    bound, hashes = common.bind_all_memories_v95(source, permutation_seed=950095)

    assert tuple(bound) == common.ARMS
    assert all(tuple(bound[arm]) == common.SCENE_IDS for arm in common.ARMS)
    assert sum(len(values) for values in bound.values()) == 24
    for scene_id in common.SCENE_IDS:
        assert bound[common.PAIRED_WRONG_SCENE][scene_id] is source[
            common.PAIR_SCENE[scene_id]
        ]
        assert hashes[common.PRIMARY][scene_id] != hashes[common.ZERO_PAYLOAD][scene_id]
        assert (
            hashes[common.PRIMARY][scene_id]
            != hashes[common.FULL_INTERIOR_PERMUTATION][scene_id]
        )


def test_prediction_rows_are_label_free_and_cover_exact_216() -> None:
    rows = _prediction_rows()
    manifest = _questions()
    hashes = _memory_hashes()
    common.validate_prediction_rows_v95(
        rows,
        questions=manifest,
        memory_hashes=hashes,
        provenance_sha256=_digest("provenance"),
    )
    assert len(rows) == 216
    assert all(set(row) == common.PREDICTION_FIELDS for row in rows)
    prohibited = {"answer", "reference", "target_xyz", "target_instance"}
    assert all(not prohibited.intersection(row) for row in rows)

    with pytest.raises(ValueError, match="exact 216-row coverage"):
        common.validate_prediction_rows_v95(
            rows[:-1],
            questions=manifest,
            memory_hashes=hashes,
            provenance_sha256=_digest("provenance"),
        )


def test_prediction_validation_rejects_duplicate_or_bad_memory_hash() -> None:
    rows = _prediction_rows()
    manifest = _questions()
    hashes = _memory_hashes()
    duplicate = [*rows[:-1], dict(rows[0])]
    with pytest.raises(ValueError):
        common.validate_prediction_rows_v95(
            duplicate,
            questions=manifest,
            memory_hashes=hashes,
            provenance_sha256=_digest("provenance"),
        )
    rows[0]["primary_memory_sha256"] = "not-a-hash"
    with pytest.raises(ValueError):
        common.validate_prediction_rows_v95(
            rows,
            questions=manifest,
            memory_hashes=hashes,
            provenance_sha256=_digest("provenance"),
        )


def test_create_once_bundle_refuses_asymmetric_or_existing_outputs(
    tmp_path: Path,
) -> None:
    paths = tuple(tmp_path / f"part-{index}" for index in range(4))
    common.assert_output_bundle_state_v95(paths, complete=False)
    paths[0].write_text("partial", encoding="utf-8")
    with pytest.raises(FileExistsError, match="asymmetric"):
        common.assert_output_bundle_state_v95(paths, complete=False)
    with pytest.raises(FileNotFoundError, match="incomplete"):
        common.assert_output_bundle_state_v95(paths, complete=True)


def test_fixed_final_mutation_is_rejected() -> None:
    before = {"fingerprint_sha256": _digest("before")}
    common.assert_same_candidate_v95(before, dict(before))
    with pytest.raises(RuntimeError, match="changed during evaluation"):
        common.assert_same_candidate_v95(
            before, {"fingerprint_sha256": _digest("after")}
        )


def test_synthetic_fixed_final_authenticates_exact_six_tensor_inventory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    candidate = tmp_path / "candidate"
    candidate.mkdir()
    state = {
        name: torch.randn(shape, generator=torch.Generator().manual_seed(index))
        for index, (name, shape) in enumerate(
            common.EXPECTED_CANDIDATE_TENSORS.items(), 1
        )
    }
    weights = candidate / "bridge.safetensors"
    save_file(state, str(weights))
    preflight = {
        "config_sha256": _digest("config"),
        "preregistration_sha256": _digest("prereg"),
        "cpu_preflight_sha256": _digest("cpu"),
    }
    config = {
        "outputs": {
            "fixed_final_candidate": str(candidate),
            "training_report": str(tmp_path / "training.json"),
        },
        "sources": {"trainer_source_sha256": _digest("trainer")},
        "training": {
            "row_order_sha256": _digest("row"),
            "cross_scene_schedule_sha256": _digest("wrong"),
            "zero_payload_schedule_sha256": _digest("zero"),
            "permutation_control_schedule_sha256": _digest("permutation"),
        },
        "training_pool": {
            "balanced_class_weight_inventory_sha256": _digest("class")
        },
    }
    bindings = {
        **preflight,
        "trainer_source_sha256": config["sources"]["trainer_source_sha256"],
        "row_order_sha256": config["training"]["row_order_sha256"],
        "cross_scene_schedule_sha256": config["training"][
            "cross_scene_schedule_sha256"
        ],
        "zero_payload_schedule_sha256": config["training"][
            "zero_payload_schedule_sha256"
        ],
        "permutation_control_schedule_sha256": config["training"][
            "permutation_control_schedule_sha256"
        ],
        "fixed_final_optimizer_updates": 480,
        "class_weight_inventory_sha256": config["training_pool"][
            "balanced_class_weight_inventory_sha256"
        ],
        "known_development_labels_opened": False,
        "deferred_final_generated": False,
    }
    metadata = {
        "artifact": "gemma4_v95_strict_causal_successor_fixed_final_v1",
        "schema_version": 95,
        "status": "fixed_final_awaiting_known_development_gate",
        "parent": "fixed_final_nonpromoted_optimization_parent",
        "bank_name": common.FRESH_BANK_NAME,
        "target_modules": list(common.TARGET_MODULES),
        "rank": 8,
        "alpha": 16.0,
        "dropout": 0.0,
        "parameter_count": common.FRESH_PARAMETER_COUNT,
        "state_sha256": tensor_state_sha256(state),
        "weights_sha256": sha256_file_v85(weights),
        "tensor_inventory": sorted(state),
        "environmental_memory_serialized": False,
        "questions_or_answers_serialized": False,
        "oracle_serialized": False,
        "known_development_scored": False,
        "deferred_final_generated": False,
        "runtime_promotion_authorized": False,
        "bindings": bindings,
    }
    metadata_path = candidate / "runtime_metadata.json"
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    training = {
        **preflight,
        "training_report_sha256": _digest("training"),
        "training_report_candidate_metadata_sha256": common.canonical_sha256_v95(
            metadata
        ),
        "training_report_candidate_weights_sha256": sha256_file_v85(weights),
    }
    monkeypatch.setattr(common, "authenticate_training_report_files_v95", lambda *_a, **_k: training)

    fingerprint = common.authenticate_fixed_final_candidate_v95(config)
    assert fingerprint["state_sha256"] == tensor_state_sha256(state)
    assert fingerprint["fixed_final_optimizer_updates"] == 480

    state["adapters.0.lora_b"] = torch.zeros_like(state["adapters.0.lora_b"])
    save_file(state, str(weights))
    with pytest.raises(ValueError, match="bytes differ"):
        common.authenticate_fixed_final_candidate_v95(config)


def test_all_known_development_preregistered_gates_pass_synthetic_success() -> None:
    gates = common.known_development_gate_results_v95(
        _structured_metrics(),
        _nll_metrics(),
        _gate_contract(),
        immutable_fixed_final=True,
        prefix_invariant=True,
        label_isolation_proven=True,
        protected_read_count=0,
    )
    assert gates
    assert all(gates.values())


@pytest.mark.parametrize(
    ("structured", "nll", "immutable", "prefix", "protected"),
    [
        (_structured_metrics(primary=149), _nll_metrics(), True, True, 0),
        (_structured_metrics(), _nll_metrics(changed_gap=0.19), True, True, 0),
        (_structured_metrics(), _nll_metrics(), False, True, 0),
        (_structured_metrics(), _nll_metrics(), True, False, 0),
        (_structured_metrics(), _nll_metrics(), True, True, 1),
    ],
)
def test_known_development_gate_fails_closed_on_each_control(
    structured: dict[str, Any],
    nll: dict[str, Any],
    immutable: bool,
    prefix: bool,
    protected: int,
) -> None:
    gates = common.known_development_gate_results_v95(
        structured,
        nll,
        _gate_contract(),
        immutable_fixed_final=immutable,
        prefix_invariant=prefix,
        label_isolation_proven=True,
        protected_read_count=protected,
    )
    assert not all(gates.values())


def test_known_development_gate_requires_label_process_isolation() -> None:
    gates = common.known_development_gate_results_v95(
        _structured_metrics(),
        _nll_metrics(),
        _gate_contract(),
        immutable_fixed_final=True,
        prefix_invariant=True,
        label_isolation_proven=False,
        protected_read_count=0,
    )
    assert gates["labels_opened_after_fixed_final_only"] is False
    assert gates["labels_opened_by_separate_label_isolated_evaluators_only"] is False


def test_aggregate_guard_rejects_questions_answers_predictions_and_ids() -> None:
    common.assert_aggregate_only_v95(
        {"metrics": {"accuracy": 0.75}, "prediction_sha256": _digest("prediction")}
    )
    for key in ("question", "answer", "prediction", "rows", "scene_id", "question_id"):
        with pytest.raises(ValueError, match="row-level content"):
            common.assert_aggregate_only_v95({"metrics": {key: "leak"}})


def _synthetic_nll_bundle_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[common.EvaluationPathsV95, Path]:
    manifest = _questions()
    memory_hashes = _memory_hashes()
    fingerprint = _digest("candidate")
    fixed = common.FixedInputsV95(
        candidate={"fingerprint_sha256": fingerprint},
        memories={},
        memory_hashes=memory_hashes,
        memory_manifest_sha256=_digest("memory-manifest"),
        memory_paths={},
    )
    final = tmp_path / "known.json"
    predictions = tmp_path / "predictions.jsonl"
    paths = common.EvaluationPathsV95(
        predictions=predictions,
        provenance=tmp_path / "provenance.json",
        prediction_access=tmp_path / "prediction-access.json",
        prediction_completion=tmp_path / "prediction-completion.json",
        structured_score=tmp_path / "structured.json",
        nll=tmp_path / "nll.json",
        nll_access=tmp_path / "nll-access.json",
        nll_completion=tmp_path / "nll-completion.json",
        final_score=final,
        evidence=tmp_path / "evidence.json",
    )
    label = tmp_path / "validation.jsonl"
    config = {
        "known_development_gate": {"labels_path": str(label)},
        "outputs": {"known_development_score": str(final)},
    }
    monkeypatch.setattr(common, "load_config_v95", lambda *_a, **_k: config)
    monkeypatch.setattr(
        common, "authenticate_fixed_inputs_before_questions_v95", lambda *_a, **_k: fixed
    )
    monkeypatch.setattr(common, "load_known_questions_v95", lambda: manifest)
    monkeypatch.setattr(common, "evaluation_paths_v95", lambda _config: paths)
    monkeypatch.setattr(
        common, "mandatory_fixed_input_reads_v95", lambda *_a, **_k: set()
    )
    return paths, label


def test_missing_and_tampered_nll_evidence_fail_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths, label = _synthetic_nll_bundle_files(tmp_path, monkeypatch)
    with pytest.raises(FileNotFoundError, match="incomplete"):
        common.authenticate_nll_bundle_v95("synthetic.yaml")

    report = {
        "artifact": common.NLL_ARTIFACT,
        "schema_version": common.SCHEMA_VERSION,
        "status": "measured_aggregate_only_not_yet_gated",
        "candidate_fingerprint_sha256": _digest("candidate"),
        "memory_manifest_sha256": _digest("memory-manifest"),
        "bound_memory_inventory_sha256": common.canonical_sha256_v95(
            _memory_hashes()
        ),
        "question_manifest_sha256": _digest("manifest"),
        "questions_sha256": _questions().questions_sha256,
        "reference_sha256": common.REFERENCE_SHA256,
        "row_count": 216,
        "scene_count": 6,
        "arms": list(common.ARMS),
        "fixed_final_and_memories_authenticated_before_labels_opened": True,
        "labels_opened_only_by_separate_nll_evaluator": True,
        "row_level_content_serialized": False,
        "metrics": _nll_metrics(),
        "runtime_promotion_authorized": False,
    }
    common.write_json_create_once_v95(paths.nll, report)
    loaded = [str(label.resolve())]
    access = {
        "artifact": "gemma4_v95_file_access_audit_v1",
        "schema_version": 95,
        "loaded_files": loaded,
        "loaded_file_inventory_sha256": common.canonical_sha256_v95(loaded),
        "forbidden_roots": [],
        "forbidden_component_names": ["oracle"],
        "block_forbidden": True,
        "forbidden_accesses": [],
        "protected_read_count": 0,
        "passed": True,
    }
    common.write_json_create_once_v95(paths.nll_access, access)
    completion = {
        "artifact": common.NLL_COMPLETION_ARTIFACT,
        "schema_version": 95,
        "candidate_fingerprint_before": _digest("candidate"),
        "candidate_fingerprint_after": _digest("candidate"),
        "candidate_immutable": True,
        "memory_hashes_invariant": True,
        "nll_sha256": sha256_file_v85(paths.nll),
        "nll_access_sha256": sha256_file_v85(paths.nll_access),
        "row_count_per_arm": 216,
        "changed_row_count": 24,
        "row_level_content_serialized": False,
        "runtime_promotion_authorized": False,
    }
    common.write_json_create_once_v95(paths.nll_completion, completion)
    assert common.authenticate_nll_bundle_v95("synthetic.yaml")["report"] == report

    report["metrics"]["primary_mean_nll"] = 9.0
    paths.nll.write_text(json.dumps(report, sort_keys=True) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="completion/access"):
        common.authenticate_nll_bundle_v95("synthetic.yaml")


def test_predictor_binds_all_memories_before_question_loader_in_source() -> None:
    source = inspect.getsource(predictor.predict_known_development_v95)
    assert source.index("authenticate_fixed_inputs_before_questions_v95") < source.index(
        "load_known_questions_v95"
    )
    assert source.index("load_predictor_stack_v95") < source.index(
        "load_known_questions_v95"
    )
    assert source.index("load_known_questions_v95") < source.index("generate_arm_v95")


def test_nll_authenticates_fixed_inputs_before_opening_references_in_source() -> None:
    source = inspect.getsource(nll_module.measure_known_development_nll_v95)
    assert source.index("authenticate_fixed_inputs_before_questions_v95") < source.index(
        "load_known_questions_v95"
    )
    assert source.index("load_known_questions_v95") < source.index("load_references_v95")
    assert source.index("load_references_v95") < source.index("_measure_nll_v84")


def test_structured_scorer_authenticates_predictions_before_labels_in_source() -> None:
    source = inspect.getsource(scorer.score_known_development_v95)
    assert source.index("authenticate_prediction_bundle_v95") < source.index(
        "load_references_v95"
    )
    assert source.index("load_references_v95") < source.index("structured_metrics_v95")


def test_process_modules_enforce_model_and_label_boundaries() -> None:
    authenticator_source = inspect.getsource(authenticator)
    predictor_source = inspect.getsource(predictor)
    scorer_source = inspect.getsource(scorer)
    nll_source = inspect.getsource(nll_module)
    assert "load_references_v95" not in predictor_source
    assert "read_jsonl" not in predictor_source
    assert "load_local_language_model" not in scorer_source
    assert "_measure_nll_v84" not in scorer_source
    assert "load_references_v95" in nll_source
    assert "_measure_nll_v84" in nll_source
    assert "load_local_language_model" not in authenticator_source
    assert "load_references_v95" not in authenticator_source
    assert "_measure_nll_v84" not in authenticator_source


def test_protected_predictor_roots_include_labels_oracle_and_v94_behavior() -> None:
    config = {
        "known_development_gate": {
            "labels_path": "data_diverse52/qa/validation.jsonl"
        }
    }
    roots = {str(path) for path in common.prediction_forbidden_roots_v95(config)}
    assert any(path.endswith("data_diverse52/qa/validation.jsonl") for path in roots)
    assert any("oracle" in Path(path).parts for path in roots)
    assert any("v94_strict_multiscene_full40_validation_question_only" in path for path in roots)
    assert any(path.endswith("gemma4_v94_strict_multiscene_full40_validation.json") for path in roots)


def test_structured_scorer_blocks_every_qa_source_except_pinned_labels() -> None:
    config = {
        "known_development_gate": {
            "labels_path": "data_diverse52/qa/validation.jsonl"
        },
        "sources": {"training_qa": "data_diverse52/qa/train.jsonl"},
        "deferred_final_lock": {
            "empty_qa_placeholders": [
                "data_diverse52/qa/test.jsonl",
                "data_diverse52/qa/final.jsonl",
            ]
        },
    }
    allowed = common.resolve_v95(
        config["known_development_gate"]["labels_path"]
    )
    roots = set(common.structured_score_forbidden_roots_v95(config))

    assert allowed not in roots
    assert common.resolve_v95(config["sources"]["training_qa"]) in roots
    assert all(
        common.resolve_v95(path) in roots
        for path in config["deferred_final_lock"]["empty_qa_placeholders"]
    )
    assert any("oracle" in path.parts for path in roots)
    assert any("v94_strict_multiscene_full40_validation_question_only" in str(path) for path in roots)


def test_nll_aggregate_schema_has_no_row_level_content() -> None:
    common.validate_nll_metrics_v95(_nll_metrics())
    serialized = json.dumps(_nll_metrics(), sort_keys=True)
    for prohibited in (
        '"question"',
        '"answer"',
        '"prediction"',
        '"scene_id"',
        '"question_id"',
        '"rows"',
    ):
        assert prohibited not in serialized
