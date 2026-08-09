from __future__ import annotations

import pytest
import torch

from semantic_3d_chat.config import load_config
from semantic_3d_chat.data.dataset import QARecord
from semantic_3d_chat.training.checkpointing import (
    load_optimizer_checkpoint,
    save_optimizer_checkpoint,
)
from semantic_3d_chat.training.train_adapter import (
    select_training_records,
    training_artifact_paths,
    training_selection_summary,
    validate_output_namespace,
)


def _record(scene_id: str, index: int) -> QARecord:
    return QARecord(
        scene_id=scene_id,
        question_id=f"q_{index:06d}",
        question=f"Question {index}?",
        answer="yes",
        answer_type="presence",
        target_xyz=None,
    )


def _typed_record(
    scene_id: str,
    index: int,
    answer_type: str,
    *,
    pair_id: str | None = None,
    key: str | None = None,
    role: str | None = None,
) -> QARecord:
    return QARecord(
        scene_id=scene_id,
        question_id=f"q_{scene_id}_{index:03d}",
        question=f"Question {index}?",
        answer="yes",
        answer_type=answer_type,
        target_xyz=None,
        counterfactual_pair_id=pair_id,
        counterfactual_question_key=key,
        counterfactual_expected_change=pair_id is not None,
        counterfactual_role=role,
        counterfactual_change_type="synthetic" if pair_id else None,
    )


def test_per_scene_question_cap_is_deterministic_and_balanced() -> None:
    records = [
        *[_record("scene_000001", index) for index in range(5)],
        *[_record("scene_000002", index) for index in range(5)],
    ]

    selected = select_training_records(records, max_questions_per_scene=2)

    assert [(record.scene_id, record.question_id) for record in selected] == [
        ("scene_000001", "q_000000"),
        ("scene_000001", "q_000001"),
        ("scene_000002", "q_000000"),
        ("scene_000002", "q_000001"),
    ]
    assert select_training_records(records, max_questions=3, max_questions_per_scene=2) == [
        records[0],
        records[1],
        records[5],
    ]


def test_selection_covers_types_and_keeps_changed_counterfactual_units_paired() -> None:
    records: list[QARecord] = []
    for scene_id, role in (("scene_a", "reference"), ("scene_b", "counterfactual")):
        records.extend(
            [
                _typed_record(
                    scene_id,
                    0,
                    "attribute",
                    pair_id="pair_1",
                    key="changed_color",
                    role=role,
                ),
                _typed_record(scene_id, 1, "spatial_relation"),
                _typed_record(scene_id, 2, "presence"),
                _typed_record(scene_id, 3, "count"),
                _typed_record(scene_id, 4, "support"),
                _typed_record(scene_id, 5, "metric"),
                _typed_record(scene_id, 6, "spatial_relation"),
                _typed_record(scene_id, 7, "presence"),
            ]
        )

    selected = select_training_records(records, max_questions_per_scene=6)
    for scene_id in ("scene_a", "scene_b"):
        scene_records = [record for record in selected if record.scene_id == scene_id]
        assert len(scene_records) == 6
        assert {record.answer_type for record in scene_records} == {
            "attribute",
            "spatial_relation",
            "presence",
            "count",
            "support",
            "metric",
        }
        assert [
            record.counterfactual_question_key
            for record in scene_records
            if record.counterfactual_expected_change
        ] == ["changed_color"]

    summary = training_selection_summary(records, selected)
    assert summary["expected_change_units_selected"] == 1
    assert summary["expected_change_units_complete"] == 1
    assert summary["expected_change_units_incomplete"] == 0


def test_selection_rejects_incomplete_or_impossible_counterfactual_selection() -> None:
    incomplete = [
        _typed_record(
            "scene_a",
            0,
            "attribute",
            pair_id="pair_1",
            key="changed_color",
            role="reference",
        )
    ]
    with pytest.raises(ValueError, match="one record from each paired training scene"):
        select_training_records(incomplete, max_questions_per_scene=1)

    paired: list[QARecord] = []
    for index in range(2):
        key = f"changed_{index}"
        paired.extend(
            [
                _typed_record(
                    "scene_a",
                    index,
                    "attribute",
                    pair_id="pair_1",
                    key=key,
                    role="reference",
                ),
                _typed_record(
                    "scene_b",
                    index,
                    "attribute",
                    pair_id="pair_1",
                    key=key,
                    role="counterfactual",
                ),
            ]
        )
    with pytest.raises(ValueError, match="changed counterfactual records but its cap"):
        select_training_records(paired, max_questions_per_scene=1)


def test_multiscene_namespace_is_path_safe_and_separates_all_outputs() -> None:
    config = load_config("configs/experiments/multiscene.yaml")
    checkpoint_root, metrics_path, figure_path = training_artifact_paths(
        config, config["training"]["output_namespace"]
    )

    assert checkpoint_root.parts[-2:] == ("checkpoints", "multiscene")
    assert metrics_path.name == "training_multiscene.json"
    assert figure_path.name == "training_loss_multiscene.png"
    assert config["training"]["max_questions_per_scene"] == 48
    assert validate_output_namespace("run-01") == "run-01"
    for unsafe in ("../escape", "nested/path", "UPPER", "", "."):
        with pytest.raises(ValueError, match="output_namespace"):
            validate_output_namespace(unsafe)


def test_legacy_training_paths_remain_unchanged_without_namespace() -> None:
    config = load_config("configs/default.yaml")
    checkpoint_root, metrics_path, figure_path = training_artifact_paths(config, None)

    assert checkpoint_root.name == "checkpoints"
    assert metrics_path.name == "training.json"
    assert figure_path.name == "training_loss.png"


def test_optimizer_checkpoint_round_trip_supports_epoch_resume(tmp_path) -> None:
    source_model = torch.nn.Linear(3, 2)
    source_optimizer = torch.optim.AdamW(source_model.parameters(), lr=0.01)
    source_model(torch.ones(1, 3)).sum().backward()
    source_optimizer.step()
    saved = save_optimizer_checkpoint(tmp_path, source_optimizer)

    target_model = torch.nn.Linear(3, 2)
    target_optimizer = torch.optim.AdamW(target_model.parameters(), lr=0.5)
    load_optimizer_checkpoint(tmp_path, target_optimizer)

    assert saved.name == "optimizer.pt"
    assert target_optimizer.param_groups[0]["lr"] == pytest.approx(0.01)
    assert len(target_optimizer.state) == len(source_optimizer.state)
