from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

from scripts.evaluate_v74_gemma_behavior import (
    _answer_matches,
    _candidate_model,
    _pair_changes,
    select_smoke_rows_v74,
    shard_rows_v74,
)
from semantic_3d_chat.scene_encoder.question_control_v75 import (
    DenseFullSceneContinuousControlV75,
)
from semantic_3d_chat.training.train_question_control_v73 import (
    load_config_v73,
    load_training_rows_v73,
    split_rows_v73,
)


def _held():
    config = load_config_v73(
        "configs/experiments/gemma4_v73_fullscene_controller.yaml"
    )
    _train, held = split_rows_v73(load_training_rows_v73(config["training_qa"]))
    return held


def test_v74_gemma_smoke_selection_is_pair_complete_and_family_balanced() -> None:
    rows = select_smoke_rows_v74(_held())
    assert len(rows) == 16
    assert len({row.change_type for row in rows}) == 8
    for family in {row.change_type for row in rows}:
        members = [row for row in rows if row.change_type == family]
        assert len(members) == 2
        assert members[0].pair_id == members[1].pair_id
        assert members[0].question == members[1].question
        assert members[0].answer != members[1].answer


def test_v74_behavior_answer_matching_handles_canonical_and_lists() -> None:
    row = _held()[0]
    assert _answer_matches(row, f"The {row.answer}.")
    list_row = replace(row, answer_type="list", answer="book cube")
    assert _answer_matches(list_row, "cube and book", ("book", "cube"))
    assert not _answer_matches(list_row, "cube", ("book", "cube"))


def test_v74_behavior_pair_change_metric_requires_two_different_predictions() -> None:
    records = [
        {"pair_id": "pair_1", "question_key": "q_1", "prediction": "left"},
        {"pair_id": "pair_1", "question_key": "q_1", "prediction": "right"},
        {"pair_id": "pair_2", "question_key": "q_2", "prediction": "yes"},
        {"pair_id": "pair_2", "question_key": "q_2", "prediction": "Yes."},
    ]
    assert _pair_changes(records, "prediction") == 1


def test_v74_behavior_shards_are_disjoint_complete_and_stable() -> None:
    rows = _held()
    shards = [
        shard_rows_v74(rows, shard_count=4, shard_index=index)
        for index in range(4)
    ]
    keys = [{row.key for row in shard} for shard in shards]
    assert all(len(shard) == 96 for shard in shards)
    assert set.union(*keys) == {row.key for row in rows}
    assert sum(len(first & second) for first in keys for second in keys) == len(rows)
    assert shards[2] == shard_rows_v74(rows, shard_count=4, shard_index=2)


@pytest.mark.parametrize(
    "environment_field",
    ("environmental_text_inputs", "environmental_text_inputs_at_inference"),
)
def test_behavior_screen_accepts_explicit_zero_environmental_text_contracts(
    tmp_path: Path,
    environment_field: str,
) -> None:
    torch.manual_seed(31)
    control = DenseFullSceneContinuousControlV75(
        1536,
        torch.eye(1536)[:112],
    )
    candidate = tmp_path / "candidate.safetensors"
    save_file(
        control.state_dict(),
        str(candidate),
        metadata={
            "runtime_promotion_forbidden_until_gemma_gate": "true",
            "answer_codebook_serialized": "false",
            environment_field: "0",
        },
    )

    loaded, metadata = _candidate_model(candidate, torch.device("cpu"))

    assert type(loaded) is DenseFullSceneContinuousControlV75
    assert metadata[environment_field] == "0"
