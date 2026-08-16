from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest
import torch
from safetensors.torch import save_file

from semantic_3d_chat.evaluation import v96_known_development_common as common
from semantic_3d_chat.evaluation.question_manifest import (
    QuestionManifest,
    QuestionRecord,
    questions_sha256,
)
from semantic_3d_chat.evaluation.score_v96_known_development import (
    stable_invariant_metrics_v96,
    structured_metrics_v96,
)
from semantic_3d_chat.evaluation.v56_fresh_development_score import (
    FAMILY_PAIR_IDS,
    FAMILY_SCENE_PAIRS,
)
from semantic_3d_chat.evaluation.v85_strict_multiscene_preflight import sha256_file_v85
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
    return [
        common.prediction_row_v96(
            scene_id=row.scene_id,
            question_id=row.question_id,
            predictions={arm: f"synthetic-{arm}" for arm in common.ARMS},
            memory_hashes={arm: hashes[arm][row.scene_id] for arm in common.ARMS},
            provenance_sha256=_digest("provenance"),
            unchanged=True,
        )
        for row in manifest.questions
    ]


def _paired_reference_and_predictions(
    *, false_change_units: int = 10
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    references: list[dict[str, Any]] = []
    predictions: list[dict[str, Any]] = []
    for family_index, (family, pair_id) in enumerate(FAMILY_PAIR_IDS.items()):
        left_scene, right_scene = FAMILY_SCENE_PAIRS[family]
        for unit_index in range(36):
            changed = unit_index < 4
            question_key = f"{family}_unit_{unit_index:02d}"
            unit_global = family_index * 32 + max(0, unit_index - 4)
            for side, scene_id in enumerate((left_scene, right_scene)):
                answer = "yes" if not changed or side == 0 else "no"
                question_id = f"q_{family_index}_{side}_{unit_index:02d}"
                reference = {
                    "scene_id": scene_id,
                    "question_id": question_id,
                    "question": "Opaque synthetic question?",
                    "answer": answer,
                    "answer_type": "presence",
                    "counterfactual_pair_id": pair_id,
                    "counterfactual_paired_scene_id": (
                        right_scene if side == 0 else left_scene
                    ),
                    "counterfactual_question_key": question_key,
                    "counterfactual_change_type": family,
                    "counterfactual_expected_change": changed,
                }
                primary = answer
                if (
                    not changed
                    and unit_global < false_change_units
                    and side == 1
                ):
                    primary = "no"
                row = {
                    "scene_id": scene_id,
                    "question_id": question_id,
                    "primary_prediction": primary,
                    "zero_payload_prediction": "no",
                    "full_interior_permutation_prediction": "no",
                    "paired_wrong_scene_prediction": "no" if answer == "yes" else "yes",
                }
                references.append(reference)
                predictions.append(row)
    assert len(references) == common.QUESTION_COUNT
    return references, predictions


def _structured_metrics(
    *,
    primary: int = 160,
    changed_sides: int = 15,
    complete_units: int = 4,
    prediction_changed_units: int = 7,
    false_changes: int = 20,
) -> dict[str, Any]:
    correct = {
        common.PRIMARY: primary,
        common.ZERO_PAYLOAD: 100,
        common.FULL_INTERIOR_PERMUTATION: 99,
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
            "canonical_correct_sides": changed_sides,
            "canonical_complete_units": complete_units,
            "canonical_prediction_changed_units": prediction_changed_units,
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
        "stable_invariant": {
            "side_count": common.INVARIANT_SIDE_COUNT,
            "unit_count": common.INVARIANT_UNIT_COUNT,
            "invariant_false_change_count": false_changes,
            "invariant_false_change_rate": false_changes
            / common.INVARIANT_SIDE_COUNT,
        },
    }


def _nll_metrics(*, changed_gap: float = 0.2) -> dict[str, Any]:
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
        "changed_row_count": common.CHANGED_SIDE_COUNT,
    }


def _gate_contract() -> dict[str, Any]:
    return {
        "role": "post_fixed_final_go_no_go_not_checkpoint_selection",
        "scene_count": 6,
        "row_count": 216,
        "changed_side_total": 24,
        "changed_unit_total": 12,
        "invariant_side_total": 192,
        "labels_opened_after_fixed_final_only": True,
        "labels_opened_by_separate_label_isolated_evaluator_only": True,
        "v96_correct_minimum": 160,
        "changed_side_correct_minimum": 15,
        "complete_changed_units_minimum": 4,
        "prediction_changed_units_minimum": 7,
        "invariant_false_change_maximum": 20,
        "mean_changed_side_wrong_minus_correct_nll_minimum": 0.2,
        "correct_accuracy_above_zero_payload_required": True,
        "correct_accuracy_above_full_interior_permutation_required": True,
        "correct_nll_below_zero_payload_required": True,
        "correct_nll_below_full_interior_permutation_required": True,
        "fixed_final_checkpoint_may_not_change_after_gate": True,
        "pass_required_before_deferred_final_unlock": True,
    }


def test_bind_all_memories_covers_every_arm_scene_and_preserves_boundaries() -> None:
    generator = torch.Generator().manual_seed(960096)
    source = {
        scene_id: torch.randn(common.MEMORY_SHAPE, generator=generator).to(
            torch.bfloat16
        )
        for scene_id in common.SCENE_IDS
    }
    bound, hashes = common.bind_all_memories_v96(source)

    assert tuple(bound) == common.ARMS
    assert sum(len(values) for values in bound.values()) == 24
    for scene_id in common.SCENE_IDS:
        assert bound[common.PAIRED_WRONG_SCENE][scene_id] is source[
            common.PAIR_SCENE[scene_id]
        ]
        assert torch.equal(bound[common.ZERO_PAYLOAD][scene_id][:, :1], source[scene_id][:, :1])
        assert torch.equal(bound[common.ZERO_PAYLOAD][scene_id][:, -1:], source[scene_id][:, -1:])
        assert hashes[common.PRIMARY][scene_id] != hashes[common.ZERO_PAYLOAD][scene_id]
        assert (
            hashes[common.PRIMARY][scene_id]
            != hashes[common.FULL_INTERIOR_PERMUTATION][scene_id]
        )


def test_prediction_rows_are_label_free_and_require_exact_216_coverage() -> None:
    rows = _prediction_rows()
    common.validate_prediction_rows_v96(
        rows,
        questions=_questions(),
        memory_hashes=_memory_hashes(),
        provenance_sha256=_digest("provenance"),
    )
    assert all(set(row) == common.PREDICTION_FIELDS for row in rows)
    assert all(
        not {"answer", "reference", "target_xyz", "target_instance"}.intersection(row)
        for row in rows
    )
    with pytest.raises(ValueError, match="exact 216-row coverage"):
        common.validate_prediction_rows_v96(
            rows[:-1],
            questions=_questions(),
            memory_hashes=_memory_hashes(),
            provenance_sha256=_digest("provenance"),
        )


def test_stable_invariant_false_change_is_explicit_20_of_192() -> None:
    references, rows = _paired_reference_and_predictions(false_change_units=10)
    indexed = {(row["scene_id"], row["question_id"]): row for row in rows}
    metric = stable_invariant_metrics_v96(references, indexed)
    assert metric == {
        "side_count": 192,
        "unit_count": 96,
        "invariant_false_change_count": 20,
        "invariant_false_change_rate": 20 / 192,
    }
    full = structured_metrics_v96(references, rows)
    assert full["stable_invariant"] == metric
    assert full["counterfactual"]["canonical_correct_sides"] == 24
    assert full["counterfactual"]["canonical_complete_units"] == 12
    assert full["counterfactual"]["canonical_prediction_changed_units"] == 12


def test_stable_invariant_metric_rejects_missing_or_noninvariant_units() -> None:
    references, rows = _paired_reference_and_predictions()
    indexed = {(row["scene_id"], row["question_id"]): row for row in rows}
    with pytest.raises(ValueError, match="inventory"):
        stable_invariant_metrics_v96(references[:-1], indexed)
    references[24]["answer"] = "no"
    with pytest.raises(ValueError, match="semantics"):
        stable_invariant_metrics_v96(references, indexed)


def test_all_v96_gate_boundaries_pass_inclusively() -> None:
    gates = common.known_development_gate_results_v96(
        _structured_metrics(),
        _nll_metrics(),
        _gate_contract(),
        immutable_fixed_final=True,
        frozen_v95_parent_immutable=True,
        prefix_invariant=True,
        label_isolation_proven=True,
        protected_read_count=0,
    )
    assert gates and all(gates.values())


@pytest.mark.parametrize(
    ("structured", "nll", "immutable", "parent", "prefix", "labels", "protected"),
    [
        (_structured_metrics(primary=159), _nll_metrics(), True, True, True, True, 0),
        (_structured_metrics(changed_sides=14), _nll_metrics(), True, True, True, True, 0),
        (_structured_metrics(complete_units=3), _nll_metrics(), True, True, True, True, 0),
        (
            _structured_metrics(prediction_changed_units=6),
            _nll_metrics(),
            True,
            True,
            True,
            True,
            0,
        ),
        (_structured_metrics(false_changes=21), _nll_metrics(), True, True, True, True, 0),
        (_structured_metrics(), _nll_metrics(changed_gap=0.199), True, True, True, True, 0),
        (_structured_metrics(), _nll_metrics(), False, True, True, True, 0),
        (_structured_metrics(), _nll_metrics(), True, False, True, True, 0),
        (_structured_metrics(), _nll_metrics(), True, True, False, True, 0),
        (_structured_metrics(), _nll_metrics(), True, True, True, False, 0),
        (_structured_metrics(), _nll_metrics(), True, True, True, True, 1),
    ],
)
def test_v96_gate_fails_closed_on_each_required_control(
    structured: dict[str, Any],
    nll: dict[str, Any],
    immutable: bool,
    parent: bool,
    prefix: bool,
    labels: bool,
    protected: int,
) -> None:
    gates = common.known_development_gate_results_v96(
        structured,
        nll,
        _gate_contract(),
        immutable_fixed_final=immutable,
        frozen_v95_parent_immutable=parent,
        prefix_invariant=prefix,
        label_isolation_proven=labels,
        protected_read_count=protected,
    )
    assert not all(gates.values())


def test_aggregate_guard_and_create_once_bundle_fail_closed(tmp_path: Path) -> None:
    common.assert_aggregate_only_v96(
        {"metrics": {"accuracy": 0.75}, "prediction_sha256": _digest("p")}
    )
    for key in ("question", "answer", "prediction", "rows", "scene_id", "question_id"):
        with pytest.raises(ValueError, match="row-level content"):
            common.assert_aggregate_only_v96({"metrics": {key: "leak"}})

    paths = tuple(tmp_path / f"part-{index}" for index in range(4))
    common.assert_output_bundle_state_v96(paths, complete=False)
    paths[0].write_text("partial", encoding="utf-8")
    with pytest.raises(FileExistsError, match="asymmetric"):
        common.assert_output_bundle_state_v96(paths, complete=False)
    with pytest.raises(FileNotFoundError, match="incomplete"):
        common.assert_output_bundle_state_v96(paths, complete=True)


def test_prediction_read_boundary_allows_only_pinned_sanitized_questions() -> None:
    config = {
        "known_development_gate": {
            "labels_path": "data_diverse52/qa/validation.jsonl"
        }
    }
    forbidden = set(common.prediction_forbidden_roots_v96(config))
    label = common.resolve_v96("data_diverse52/qa/validation.jsonl")
    assert any(root == label or root in label.parents for root in forbidden)
    assert common.QUESTION_MANIFEST.resolve() not in forbidden
    question_root = common.QUESTION_MANIFEST.parent
    assert all(
        path.resolve() in forbidden
        for path in question_root.iterdir()
        if path.resolve() != common.QUESTION_MANIFEST.resolve()
    )
    own = common.evaluation_paths_v96(config)
    assert own.predictions not in forbidden
    assert own.provenance not in forbidden


def test_synthetic_v96_fixed_final_authenticates_two_tensor_bank(
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
        "topology_smoke_sha256": _digest("topology"),
    }
    config: dict[str, Any] = {
        "outputs": {
            "fixed_final_candidate": str(candidate),
            "training_report": str(tmp_path / "training.json"),
        },
        "sources": {"trainer_source_sha256": _digest("trainer")},
        "training": {
            "schedule_sha256": _digest("schedule"),
            "invariant_subset_sha256": _digest("invariant-subset"),
        },
        "training_pool": {
            "balanced_class_weight_inventory_sha256": _digest("class"),
            "changed_family_weight_inventory_sha256": _digest("changed"),
            "invariant_family_weight_inventory_sha256": _digest("invariant"),
        },
    }
    bindings = common._candidate_binding_contract_v96(
        config,
        preflight,
        v95_parent_evidence_sha256=_digest("v95-evidence"),
    )
    metadata = {
        "artifact": "gemma4_v96_atomic_pair_repair_fixed_final_v1",
        "schema_version": 96,
        "status": "fixed_final_awaiting_known_development_gate",
        "parent": "v95_fixed_final_nonpromoted_optimization_parent",
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
    (candidate / "runtime_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    training = {
        **preflight,
        "training_report_sha256": _digest("training"),
        "training_report_candidate_metadata_sha256": common.canonical_sha256_v96(
            metadata
        ),
        "training_report_candidate_weights_sha256": sha256_file_v85(weights),
    }
    monkeypatch.setattr(
        common, "authenticate_training_report_files_v96", lambda *_a, **_k: training
    )
    monkeypatch.setattr(
        common,
        "authenticate_parent_v95_v96",
        lambda _config: {
            "v95_state_sha256": _digest("v95"),
            "v95_evidence_sha256": _digest("v95-evidence"),
        },
    )

    fingerprint = common.authenticate_fixed_final_candidate_v96(config)
    assert fingerprint["state_sha256"] == tensor_state_sha256(state)
    assert fingerprint["frozen_v95_state_sha256"] == _digest("v95")
    assert fingerprint["fixed_final_optimizer_updates"] == 285

    metadata["known_development_scored"] = True
    (candidate / "runtime_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="metadata contract"):
        common.authenticate_fixed_final_candidate_v96(config)


def test_v96_constants_match_preregistered_scope_without_opening_labels() -> None:
    assert common.QUESTION_COUNT == 216
    assert common.CHANGED_SIDE_COUNT == 24
    assert common.CHANGED_UNIT_COUNT == 12
    assert common.INVARIANT_SIDE_COUNT == 192
    assert common.INVARIANT_UNIT_COUNT == 96
    assert common.PERMUTATION_SEED == 950095
    assert common.MEMORY_SHAPE == (1, 738, 1536)
    assert common.ARMS == (
        "primary",
        "zero_payload",
        "full_interior_permutation",
        "paired_wrong_scene",
    )


def test_evaluator_rejects_an_alternate_unsealed_config_path(tmp_path: Path) -> None:
    common.assert_bound_config_path_v96(common.CONFIG)
    with pytest.raises(ValueError, match="not the sealed default"):
        common.assert_bound_config_path_v96(tmp_path / "alternate.yaml")


def _no_row_content(value: object) -> bool:
    try:
        common.assert_aggregate_only_v96(value)
    except ValueError:
        return False
    return True


def test_metric_helpers_remain_aggregate_only() -> None:
    assert _no_row_content(_structured_metrics())
    assert _no_row_content(_nll_metrics())
    assert isinstance(_gate_contract(), Mapping)
