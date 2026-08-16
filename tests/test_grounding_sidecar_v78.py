from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from semantic_3d_chat.training.grounding_sidecar_v78 import (
    GroundingRecord,
    GroundingSidecarV78,
    _save_candidate,
    _validate_historical_training_path,
    load_historical_grounding_records,
    pair_disjoint_internal_split,
    validate_candidate,
)


def _record(pair: str, scene: str, paired: str, index: int) -> GroundingRecord:
    return GroundingRecord(
        question_id=f"q_{index:06d}",
        scene_id=scene,
        pair_id=pair,
        paired_scene_id=paired,
        question_key=f"cfq_{index:04d}",
        question="Where is the object?",
        target_xyz=(0.1, 0.2, 0.3),
    )


def test_v78_scores_every_scene_token_and_has_expected_shape() -> None:
    torch.manual_seed(1)
    model = GroundingSidecarV78(scene_dim=12, latent_count=8, rank=3, hidden_dim=7)
    question = torch.randn(2, 12)
    scene = torch.randn(2, 8, 12)

    predicted, logits, weights = model(question, scene)

    assert predicted.shape == (2, 3)
    assert logits.shape == (2, 8)
    assert weights.shape == (2, 8)
    assert torch.all(weights > 0)
    assert torch.allclose(weights.sum(dim=-1), torch.ones(2))
    assert torch.all(predicted.abs() <= 1.0)


def test_v78_zero_scene_is_exact_and_question_independent() -> None:
    torch.manual_seed(2)
    model = GroundingSidecarV78(scene_dim=12, latent_count=8, rank=3, hidden_dim=7)
    questions = torch.randn(3, 12)
    zeros = torch.zeros(3, 8, 12)

    predicted, _, weights = model(questions, zeros)

    assert torch.equal(predicted, torch.zeros_like(predicted))
    assert torch.all(weights > 0)
    assert torch.allclose(weights, torch.full_like(weights, 1.0 / 8.0))


def test_v78_scene_tokens_receive_gradients_without_question_only_coordinate_path() -> None:
    torch.manual_seed(3)
    model = GroundingSidecarV78(scene_dim=12, latent_count=8, rank=3, hidden_dim=7)
    question = torch.randn(1, 12)
    scene = torch.randn(1, 8, 12, requires_grad=True)

    predicted, _, _ = model(question, scene)
    predicted.square().sum().backward()

    assert scene.grad is not None
    assert torch.all(scene.grad.abs().sum(dim=-1) > 0)


def test_pair_split_keeps_pairs_and_scenes_disjoint() -> None:
    records: list[GroundingRecord] = []
    for pair_index in range(8):
        pair = f"pair_{pair_index:06d}"
        left = f"scene_{pair_index * 2:06d}"
        right = f"scene_{pair_index * 2 + 1:06d}"
        records.extend(
            [
                _record(pair, left, right, pair_index * 2),
                _record(pair, right, left, pair_index * 2 + 1),
            ]
        )

    train, held, audit = pair_disjoint_internal_split(records)

    assert {row.scene_id for row in train}.isdisjoint({row.scene_id for row in held})
    assert {row.pair_id for row in train}.isdisjoint({row.pair_id for row in held})
    assert audit["pair_disjoint"] is True
    assert audit["scene_disjoint"] is True


@pytest.mark.parametrize("blocked", ["oracle", "official", "validation", "test", "deferred"])
def test_historical_training_path_rejects_forbidden_part(
    tmp_path: Path, blocked: str
) -> None:
    source = tmp_path / "training" / blocked / "rows.jsonl"
    source.parent.mkdir(parents=True)
    source.write_text(json.dumps({"safe": False}) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="must not use"):
        _validate_historical_training_path(source, label="fixture")


@pytest.mark.parametrize(
    "filename",
    ["official_validation.jsonl", "test.jsonl", "deferred-final.jsonl"],
)
def test_historical_training_path_rejects_forbidden_filename(
    tmp_path: Path, filename: str
) -> None:
    source = tmp_path / "training" / filename
    source.parent.mkdir(parents=True)
    source.write_text("{}\n", encoding="utf-8")

    with pytest.raises(ValueError, match="must not use"):
        _validate_historical_training_path(source, label="fixture")


def test_historical_training_path_requires_explicit_training_identity(tmp_path: Path) -> None:
    source = tmp_path / "rows.jsonl"
    source.write_text("{}\n", encoding="utf-8")

    with pytest.raises(ValueError, match="historical training"):
        _validate_historical_training_path(source, label="fixture")


def test_training_loader_immediately_drops_answers_and_instance_ids(tmp_path: Path) -> None:
    source = tmp_path / "training" / "train.jsonl"
    source.parent.mkdir()
    source.write_text(
        json.dumps(
            {
                "answer": "secret semantic answer",
                "target_instance": "named_instance_that_must_be_dropped",
                "question_id": "q_000001",
                "scene_id": "scene_000001",
                "counterfactual_pair_id": "pair_000001",
                "counterfactual_paired_scene_id": "scene_000002",
                "counterfactual_question_key": "cfq_000001",
                "question": "Where is the object?",
                "target_xyz": [1.0, 2.0, 0.5],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    records = load_historical_grounding_records(source)

    assert len(records) == 1
    assert not hasattr(records[0], "answer")
    assert not hasattr(records[0], "target_instance")


def test_candidate_is_exact_two_file_numeric_seal(tmp_path: Path) -> None:
    torch.manual_seed(4)
    model = GroundingSidecarV78(scene_dim=12, latent_count=8, rank=3, hidden_dim=7)
    candidate = tmp_path / "candidate"

    audit = _save_candidate(
        candidate,
        model,
        prefix_manifest_sha256="a" * 64,
        model_id="local/model",
        model_revision="revision",
        room_min=(-3.0, -2.5, 0.0),
        room_max=(3.0, 2.5, 3.0),
        seed=4,
    )
    validated = validate_candidate(candidate)

    assert audit["files"] == ["grounding.safetensors", "metadata.json"]
    assert validated["weights_sha256"] == audit["weights_sha256"]
    metadata = validated["metadata"]
    assert metadata["environmental_text_inputs"] == []
    assert metadata["question_text_serialized"] is False
    assert metadata["target_coordinates_serialized"] is False
    assert metadata["runtime_promotion_authorized"] is False
