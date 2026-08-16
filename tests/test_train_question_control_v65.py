from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from semantic_3d_chat.evaluation.v62_pair_disjoint_preregistration import (
    TRAIN_PAIR_IDS,
)
from semantic_3d_chat.training import train_question_control_v65 as v65
from semantic_3d_chat.training.train_question_control_v63 import V63Row


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _record(
    *,
    pair_index: int,
    unit_index: int,
    side_index: int,
    exact: bool,
    supported: bool = True,
) -> dict[str, object]:
    pair_id = TRAIN_PAIR_IDS[pair_index]
    opaque = f"{pair_index}:{unit_index}:{side_index}"
    canonical = _digest(f"canonical:{opaque}:{exact}")
    reference = canonical if exact else _digest(f"reference:{opaque}")
    return {
        "scene_id": f"scene_{pair_index * 100 + unit_index * 2 + side_index:06d}",
        "question_id": f"q_{pair_index:02d}_{unit_index:02d}_{side_index}",
        "pair_id": pair_id,
        "question_key": f"cfq_{pair_index:02d}_{unit_index:02d}",
        "answer_class_id": v65._answer_class_id("yes"),
        "fold_class_supported": supported,
        "raw_prediction_sha256": _digest(f"raw:{opaque}"),
        "canonical_prediction_sha256": canonical,
        "reference_canonical_sha256": reference,
        "scoring_contract_sha256": _digest(f"contract:{opaque}"),
        "canonical_exact": exact,
    }


def _behavior_inventory(
    *,
    complete_units: int,
    single_side_units: int,
) -> list[dict[str, object]]:
    units_per_pair = [4, 4, 4, 4, 3, 3, 3, 3, 3, 3, 3, 3]
    records: list[dict[str, object]] = []
    ordinal = 0
    for pair_index, unit_count in enumerate(units_per_pair):
        for local_unit in range(unit_count):
            if ordinal < complete_units:
                exact_sides = (True, True)
            elif ordinal < complete_units + single_side_units:
                exact_sides = (True, False)
            else:
                exact_sides = (False, False)
            records.extend(
                _record(
                    pair_index=pair_index,
                    unit_index=local_unit,
                    side_index=side_index,
                    exact=exact,
                )
                for side_index, exact in enumerate(exact_sides)
            )
            ordinal += 1
    assert ordinal == 40
    return records


def _synthetic_codebook_inputs() -> tuple[
    tuple[V63Row, ...],
    dict[tuple[str, str], torch.Tensor],
]:
    generator = torch.Generator().manual_seed(6501)
    rows: list[V63Row] = []
    teachers: dict[tuple[str, str], torch.Tensor] = {}
    for class_index in range(21):
        for side_index in range(2):
            row = V63Row(
                scene_id=f"scene_{class_index * 2 + side_index:06d}",
                question_id=f"q_{class_index:02d}_{side_index}",
                question=f"Training question {class_index}?",
                pair_id=TRAIN_PAIR_IDS[class_index % len(TRAIN_PAIR_IDS)],
                question_key=f"cfq_{class_index:02d}",
                route_label=True,
                answer=(f"Class {class_index}!" if side_index == 0 else f"class {class_index}"),
            )
            rows.append(row)
            teachers[row.key] = torch.randn(
                1,
                4,
                16,
                generator=generator,
            )
    return tuple(rows), teachers


def test_parser_exposes_only_training_and_behavioral_work_boundaries() -> None:
    destinations = {action.dest for action in v65._parser()._actions if action.dest != "help"}

    assert {
        "baseline_lock",
        "filtered_train_qa",
        "teacher_cache",
        "prefix_cache",
        "source_v60_checkpoint",
        "work_directory",
        "output_checkpoint",
        "training_report",
    } <= destinations
    assert {
        "validation_questions",
        "internal_validation_questions",
        "scorer_references",
        "scorer_sidecar",
        "questions_manifest",
        "predictions",
        "oracle",
        "heldout",
        "preregistration",
    }.isdisjoint(destinations)


def test_answer_codebook_is_deterministic_numeric_and_equal_answer_exact() -> None:
    rows, teachers = _synthetic_codebook_inputs()

    first = v65.build_answer_prototype_codebook(
        rows,
        teachers,
        expected_prompt_shape=(1, 4, 16),
    )
    second = v65.build_answer_prototype_codebook(
        tuple(reversed(rows)),
        dict(reversed(list(teachers.items()))),
        expected_prompt_shape=(1, 4, 16),
    )

    assert len(first.prototypes) == 21
    assert len(first.targets) == 42
    assert first.sha256 == second.sha256
    assert first.manifest["scope"] == "final_all_training"
    assert first.manifest["fold_local"] is False
    assert first.manifest["scene_question_teachers_outside_source_pairs_used"] is False
    assert first.manifest["answer_strings_serialized"] is False
    for class_index in range(21):
        members = [
            row for row in rows if row.answer.casefold().removesuffix("!") == f"class {class_index}"
        ]
        left, right = (first.targets[row.key] for row in members)
        assert torch.equal(left, right)
        assert left.untyped_storage().data_ptr() == right.untyped_storage().data_ptr()
        class_id = first.class_by_key[members[0].key]
        assert torch.equal(first.prototypes[class_id], second.prototypes[class_id])
    serialized = json.dumps(first.manifest, sort_keys=True)
    assert not any(row.answer in serialized for row in rows)

    basis = v65.codebook_output_basis(first, requested_rank=128)
    assert tuple(basis.shape) == (16, 16)
    assert v65._basis_coverage(first.prototypes, basis)["minimum_cosine"] > 0.999


def test_pair_fold_partition_excludes_every_held_example() -> None:
    rows: list[V63Row] = []
    targets: dict[tuple[str, str], torch.Tensor] = {}
    for pair_index, pair_id in enumerate(TRAIN_PAIR_IDS):
        for side_index in range(2):
            row = V63Row(
                f"scene_{pair_index * 2 + side_index:06d}",
                f"q_{pair_index:02d}_{side_index}",
                "Same paired question?",
                pair_id,
                f"cfq_{pair_index:02d}",
                True,
                "left" if side_index == 0 else "right",
            )
            rows.append(row)
            targets[row.key] = torch.ones(1, 4, 1536) * (pair_index + side_index + 1)

    for held_pair_id in TRAIN_PAIR_IDS:
        train_rows, held_rows, train_targets, held_targets = v65._pair_fold_partition(
            rows,
            targets,
            held_pair_id=held_pair_id,
        )
        assert all(row.pair_id != held_pair_id for row in train_rows)
        assert all(row.pair_id == held_pair_id for row in held_rows)
        assert set(train_targets).isdisjoint(held_targets)
        assert set(train_targets) | set(held_targets) == set(targets)


def test_behavioral_gates_use_actual_sides_units_and_fold_hits() -> None:
    held_records = _behavior_inventory(complete_units=30, single_side_units=4)
    # Exact preregistered closed-vocabulary coverage: 60/80 sides and 28/40
    # fully supported units, distributed over eight eligible folds.
    for index, record in enumerate(held_records):
        record["fold_class_supported"] = index < 56 or index % 2 == 0 and index < 64
    held_metrics = v65.behavioral_metrics(held_records)

    assert held_metrics["side_exact"] == 64
    assert held_metrics["side_total"] == 80
    assert held_metrics["supported_side_total"] == 60
    assert held_metrics["fully_supported_unit_total"] == 28
    assert held_metrics["unit_total"] == 40
    checks = v65.assess_cv_behavior(held_metrics)
    assert checks["held_supported_side_total"] is True
    assert checks["held_unsupported_side_total"] is True

    weakened = [dict(record) for record in held_records]
    for record in weakened:
        if record["fold_class_supported"] is True and record["canonical_exact"] is True:
            record["canonical_exact"] = False
            record["canonical_prediction_sha256"] = _digest(
                f"now-wrong:{record['scene_id']}:{record['question_id']}"
            )
    assert not v65.assess_cv_behavior(v65.behavioral_metrics(weakened))["held_supported_side_exact"]

    final_metrics = v65.behavioral_metrics(
        _behavior_inventory(complete_units=36, single_side_units=4)
    )
    assert final_metrics["side_exact"] == 76
    assert final_metrics["complete_units"] == 36
    assert all(v65.assess_final_behavior(final_metrics).values())
    incomplete = dict(final_metrics)
    incomplete["complete_units"] = 35
    assert v65.assess_final_behavior(incomplete)["train_complete_units"] is False


def test_behavior_generation_records_hashes_and_not_training_text() -> None:
    rows = (
        V63Row(
            "scene_000001",
            "q_000001",
            "Which side?",
            TRAIN_PAIR_IDS[0],
            "cfq_opaque",
            True,
            "left",
        ),
        V63Row(
            "scene_000002",
            "q_000002",
            "Which side?",
            TRAIN_PAIR_IDS[0],
            "cfq_opaque",
            True,
            "right",
        ),
    )

    class FakeControl:
        @staticmethod
        def audit() -> SimpleNamespace:
            return SimpleNamespace(control_used=True)

        @staticmethod
        def forward_from_signature(
            _signature: torch.Tensor,
            _question: torch.Tensor,
        ) -> SimpleNamespace:
            return SimpleNamespace(control_tokens=torch.zeros(1, 4, 1536))

    def generate(**kwargs: object) -> str:
        prefix = kwargs["scene_prefix"]
        assert isinstance(prefix, torch.Tensor)
        return "right" if float(prefix.flatten()[0]) > 0.5 else "left"

    records = v65._behavior_rows(
        FakeControl(),  # type: ignore[arg-type]
        rows,
        signatures={row.scene_id: torch.zeros(1, 1) for row in rows},
        questions={row.key: torch.zeros(1, 1, 1536) for row in rows},
        class_by_key={row.key: v65._answer_class_id(row.answer) for row in rows},
        supported_keys={row.key for row in rows},
        runtime=object(),
        prefixes={
            "scene_000001": torch.zeros(1, 1, 1),
            "scene_000002": torch.ones(1, 1, 1),
        },
        device=torch.device("cpu"),
        model_dtype=torch.float32,
        generator_fn=generate,
    )

    assert all(record["canonical_exact"] is True for record in records)
    serialized = json.dumps(records, sort_keys=True)
    assert "Which side?" not in serialized
    assert '"left"' not in serialized
    assert '"right"' not in serialized


def test_suppressed_changed_route_is_scored_through_literal_no_control_path() -> None:
    row = V63Row(
        "scene_000001",
        "q_000001",
        "Which side?",
        TRAIN_PAIR_IDS[0],
        "cfq_opaque",
        True,
        "left",
    )

    class SuppressedControl:
        @staticmethod
        def audit() -> SimpleNamespace:
            return SimpleNamespace(control_used=False)

        @staticmethod
        def forward_from_signature(
            _signature: torch.Tensor,
            _question: torch.Tensor,
        ) -> SimpleNamespace:
            return SimpleNamespace(control_tokens=torch.ones(1, 4, 1536))

    def generate(**kwargs: object) -> str:
        assert kwargs["control_tokens"] is None
        return "left"

    records = v65._behavior_rows(
        SuppressedControl(),  # type: ignore[arg-type]
        (row,),
        signatures={row.scene_id: torch.zeros(1, 1)},
        questions={row.key: torch.zeros(1, 1, 1536)},
        class_by_key={row.key: v65._answer_class_id(row.answer)},
        supported_keys=set(),
        runtime=object(),
        prefixes={row.scene_id: torch.zeros(1, 1, 1)},
        device=torch.device("cpu"),
        model_dtype=torch.float32,
        generator_fn=generate,
    )

    assert records[0]["canonical_exact"] is True
    assert records[0]["fold_class_supported"] is False


def test_activation_summary_reports_both_route_distributions_without_text() -> None:
    rows = tuple(
        V63Row(
            f"scene_{index:06d}",
            f"q_{index:06d}",
            f"Secret question {index}?",
            TRAIN_PAIR_IDS[0],
            f"cfq_{index:06d}",
            index < 2,
            f"secret answer {index}",
        )
        for index in range(4)
    )

    class ActivationControl:
        activation_rms_threshold = 0.01

        @staticmethod
        def activation_rms(control_rms: torch.Tensor) -> torch.Tensor:
            return control_rms.max(dim=-1).values

        @staticmethod
        def forward_from_signature(
            signature: torch.Tensor,
            _question: torch.Tensor,
        ) -> SimpleNamespace:
            value = signature.reshape(1, 1).expand(1, 4)
            return SimpleNamespace(control_rms=value)

    values = (0.02, 0.009, 0.004, 0.03)
    summary = v65.route_activation_summary(
        ActivationControl(),  # type: ignore[arg-type]
        rows,
        signatures={row.scene_id: torch.tensor([values[index]]) for index, row in enumerate(rows)},
        questions={row.key: torch.zeros(1, 1, 1536) for row in rows},
    )

    assert summary["changed"]["active_count"] == 1
    assert summary["changed"]["inactive_count"] == 1
    assert summary["retention"]["active_count"] == 1
    assert summary["retention"]["inactive_count"] == 1
    assert summary["all_changed_active"] is False
    assert summary["all_retention_inactive"] is False
    serialized = json.dumps(summary, sort_keys=True)
    assert "Secret question" not in serialized
    assert "secret answer" not in serialized


def test_retention_gate_records_fail_cleanly_instead_of_raising() -> None:
    def record(value: bool) -> dict[str, object]:
        return {
            "scene_id": "scene_000001",
            "question_id": "q_000001",
            "pair_id": TRAIN_PAIR_IDS[0],
            "baseline_raw_output_sha256": "a" * 64,
            "exact_no_control_route": value,
            "runtime_output_identity_by_construction": value,
            "activation_rms_below_threshold": value,
        }

    failed = v65.assess_retention_gate((record(False),), final=False, expected_count=1)
    assert failed == {
        "retention_inventory_exact": True,
        "every_retention_row_exact_no_control": False,
        "base_output_identity_by_construction": False,
    }

    inconsistent = record(False)
    inconsistent["activation_rms_below_threshold"] = True
    with pytest.raises(ValueError, match="internally inconsistent"):
        v65.assess_retention_gate((inconsistent,), final=False, expected_count=1)


def test_resumable_fold_artifacts_are_strict_hash_only_and_create_once(
    tmp_path: Path,
) -> None:
    work = tmp_path / "work"
    manifest = {
        "schema_version": 1,
        "artifact": "test",
        "run_signature_sha256": "a" * 64,
    }
    v65.prepare_work_directory(work, manifest)
    v65.prepare_work_directory(work, manifest)
    pair_id = TRAIN_PAIR_IDS[0]
    records = [
        _record(pair_index=0, unit_index=0, side_index=side, exact=True) for side in range(2)
    ]
    retention = []
    payload = {
        "schema_version": 2,
        "artifact": v65._FOLD_ARTIFACT,
        "run_signature_sha256": "a" * 64,
        "held_pair_id": pair_id,
        "training_pair_count": 11,
        "held_scene_question_examples_used_for_optimization": False,
        "fold_local_codebook": {
            "class_ids": [v65._answer_class_id("yes")],
            "sha256": "b" * 64,
        },
        "held_teacher_used_in_codebook_or_basis": False,
        "generation_semantics": v65._GENERATION_SEMANTICS,
        "fit": {"optimizer_steps": 1},
        "prompt_reconstruction": {"mean_prompt_cosine": 0.5},
        "behavior": v65.behavioral_metrics(records),
        "changed_records": records,
        "retention": v65.assess_retention_gate(retention, final=False, expected_count=0),
        "held_retention_count": 0,
        "retention_records": retention,
    }

    contaminated = json.loads(json.dumps(payload))
    contaminated["changed_records"][0]["answer"] = "yes"
    with pytest.raises(ValueError, match="record contract changed"):
        v65._save_fold(work, contaminated)
    v65._save_fold(work, payload)
    completed = v65._load_completed_folds(
        work,
        run_signature_sha256="a" * 64,
    )
    assert completed[pair_id] == payload
    assert "yes" not in (work / "folds" / f"{pair_id}.json").read_text(encoding="utf-8")
    with pytest.raises(FileExistsError):
        v65._save_fold(work, payload)
    with pytest.raises(ValueError, match="manifest differs"):
        v65.prepare_work_directory(work, {**manifest, "schema_version": 2})
