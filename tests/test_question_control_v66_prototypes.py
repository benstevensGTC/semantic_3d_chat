from __future__ import annotations

from collections.abc import Mapping
from typing import ClassVar

import pytest
import torch

from semantic_3d_chat.training.question_control_v66_prototypes import (
    answer_class_id_v66,
    build_hybrid_answer_prototype_codebook_v66,
    lm_native_answer_prototype,
)
from semantic_3d_chat.training.train_question_control_v63 import V63Row


class _Tokenizer:
    _ids: ClassVar[dict[str, list[int]]] = {
        "yes": [1],
        "no": [2],
        "red": [3],
        "book cube": [4, 5],
    }

    def __call__(
        self,
        value: str,
        *,
        add_special_tokens: bool,
        return_tensors: str,
    ) -> Mapping[str, torch.Tensor]:
        assert add_special_tokens is False
        assert return_tensors == "pt"
        return {"input_ids": torch.tensor([self._ids[value]])}


def _embedding() -> torch.nn.Embedding:
    embedding = torch.nn.Embedding(8, 6)
    with torch.no_grad():
        embedding.weight.copy_(torch.arange(48).reshape(8, 6).float() + 1.0)
    return embedding


def _row(
    scene: str,
    question_id: str,
    pair: str,
    answer: str,
    *,
    changed: bool,
) -> V63Row:
    return V63Row(
        scene_id=scene,
        question_id=question_id,
        question=f"question {question_id}",
        pair_id=pair,
        question_key=f"key_{question_id}",
        route_label=changed,
        answer=answer,
        answer_type="presence",
    )


def _native(answer: str) -> torch.Tensor:
    return lm_native_answer_prototype(
        answer,
        tokenizer=_Tokenizer(),
        embedding_layer=_embedding(),
        target_rms=0.1,
    )


def test_lm_native_answer_prototype_is_fixed_shape_numeric_and_deterministic() -> None:
    first = _native("book cube")
    second = _native("book cube")

    assert first.shape == (1, 4, 6)
    assert torch.equal(first, second)
    assert torch.allclose(
        first.square().mean(dim=-1).sqrt(),
        torch.full((1, 4), 0.1),
    )
    assert torch.equal(first[:, 0], first[:, 2])
    assert torch.equal(first[:, 1], first[:, 3])


def test_v66_codebook_reuses_teacher_and_fills_missing_class_numerically() -> None:
    rows = (
        _row("scene_1", "q1", "pair_1", "yes", changed=True),
        _row("scene_2", "q2", "pair_2", "yes", changed=False),
        _row("scene_1", "q3", "pair_1", "no", changed=False),
        _row("scene_2", "q4", "pair_2", "no", changed=False),
    )
    teacher = torch.full((1, 4, 1536), 0.03)
    codebook = build_hybrid_answer_prototype_codebook_v66(
        rows,
        {rows[0].key: teacher},
        native_prototype_provider=lambda _answer: torch.full((1, 4, 1536), 0.07),
        allow_unverified_native_fallback=True,
        expected_class_count=2,
        scope="unit",
    )

    assert len(codebook.targets) == 4
    assert torch.equal(codebook.prototypes[answer_class_id_v66("yes")], teacher)
    assert torch.equal(
        codebook.prototypes[answer_class_id_v66("no")],
        torch.full((1, 4, 1536), 0.07),
    )
    assert codebook.manifest["verified_teacher_prototype_count"] == 1
    assert codebook.manifest["lm_native_prototype_count"] == 1
    assert codebook.manifest["answer_strings_serialized"] is False
    serialized = str(codebook.manifest)
    assert "'yes'" not in serialized
    assert "'no'" not in serialized


def test_v66_fold_codebook_rejects_held_rows_and_foreign_teachers() -> None:
    train = _row("scene_1", "q1", "pair_train", "yes", changed=True)
    held = _row("scene_2", "q2", "pair_held", "no", changed=True)

    with pytest.raises(AssertionError, match="held pair"):
        build_hybrid_answer_prototype_codebook_v66(
            (train, held),
            {train.key: torch.ones(1, 4, 1536)},
            native_prototype_provider=lambda _answer: torch.ones(1, 4, 1536),
            allow_unverified_native_fallback=True,
            scope="fold",
            forbidden_pair_id="pair_held",
        )
    with pytest.raises(AssertionError, match="foreign teacher"):
        build_hybrid_answer_prototype_codebook_v66(
            (train,),
            {held.key: torch.ones(1, 4, 1536)},
            native_prototype_provider=lambda _answer: torch.ones(1, 4, 1536),
            allow_unverified_native_fallback=True,
            scope="fold",
            forbidden_pair_id="pair_held",
        )


def test_v66_fold_support_is_derived_only_from_training_rows() -> None:
    train = (
        _row("scene_1", "q1", "pair_train", "yes", changed=False),
        _row("scene_1", "q2", "pair_train", "red", changed=False),
    )
    held = (
        _row("scene_2", "q3", "pair_held", "yes", changed=False),
        _row("scene_2", "q4", "pair_held", "no", changed=False),
    )
    codebook = build_hybrid_answer_prototype_codebook_v66(
        train,
        {},
        native_prototype_provider=lambda _answer: torch.ones(1, 4, 1536),
        allow_unverified_native_fallback=True,
        scope="fold",
        forbidden_pair_id="pair_held",
    )
    supported = {
        row.key
        for row in held
        if answer_class_id_v66(row.answer) in codebook.prototypes
    }

    assert supported == {held[0].key}
    assert codebook.manifest["forbidden_pair_absent"] is True
    assert all(
        "pair_held" not in record["source_pair_ids"]
        for record in codebook.manifest["records"]
    )


def test_v66_primary_codebook_rejects_unverified_native_fallback() -> None:
    row = _row("scene_1", "q1", "pair_train", "yes", changed=False)

    with pytest.raises(ValueError, match="verified numeric teacher"):
        build_hybrid_answer_prototype_codebook_v66(
            (row,),
            {},
            native_prototype_provider=lambda _answer: torch.ones(1, 4, 1536),
            scope="primary",
        )
