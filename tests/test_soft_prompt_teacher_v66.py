from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from semantic_3d_chat.training import soft_prompt_teacher_v66 as v66
from semantic_3d_chat.training.train_question_control_v63 import V63Row


def _row(
    scene: int,
    question: int,
    pair: int,
    answer: str,
) -> V63Row:
    return V63Row(
        scene_id=f"scene_{scene:06d}",
        question_id=f"q_{question:06d}",
        question=f"training question {question}",
        pair_id=f"pair_{pair:06d}",
        question_key=f"cfq_{question:016x}",
        route_label=False,
        answer=answer,
        answer_type="attribute",
    )


def _metrics() -> dict[str, object]:
    return {
        "steps": 3,
        "initial_mean_nll": 1.0,
        "final_mean_nll": 0.01,
        "maximum_preclip_gradient_norm": 0.5,
        "initial_rms": 0.05,
        "final_rms": 0.10,
        "learning_rate": 0.03,
        "attempt_count": 1,
        "attempt_learning_rates": [0.03],
        "total_forward_steps": 3,
        "training_row_count": 2,
    }


def _production_groups() -> tuple[v66.V66TeacherGroup, ...]:
    counts = {
        "behind": 8,
        "blue": 7,
        "cream": 5,
        "in front": 9,
        "red": 10,
        "terracotta": 7,
        "wood": 10,
    }
    groups: list[v66.V66TeacherGroup] = []
    ordinal = 1
    for answer, count in counts.items():
        class_id = v66._answer_class_id(answer)
        for pair_offset in range(count):
            pair = pair_offset + 1
            first = _row(ordinal * 2, ordinal * 2, pair, answer)
            second = _row(ordinal * 2 + 1, ordinal * 2 + 1, pair, answer)
            groups.append(
                v66.V66TeacherGroup(class_id, f"pair_{pair:06d}", (first, second))
            )
            ordinal += 1
    for answer in ("floor", "green", "no", "upright", "wall", "yellow", "yes"):
        class_id = v66._answer_class_id(answer)
        first = _row(ordinal * 2, ordinal * 2, 99, answer)
        second = _row(ordinal * 2 + 1, ordinal * 2 + 1, 99, answer)
        groups.append(
            v66.V66TeacherGroup(
                class_id,
                "pair_000099",
                (first, second),
                "alternate_pair_coverage",
            )
        )
        ordinal += 1
    return tuple(sorted(groups, key=lambda group: group.key))


def _preflight(tmp_path: Path) -> v66.V66TeacherPreflight:
    groups = _production_groups()
    rows = tuple(row for group in groups for row in group.rows)
    return v66.V66TeacherPreflight(
        config={},
        runtime_config_sha256="1" * 64,
        base_checkpoint=tmp_path / "base",
        base_checkpoint_sha256="2" * 64,
        source_control=torch.nn.Identity(),
        source_control_checkpoint_sha256="3" * 64,
        source_control_metadata={},
        prefixes={},
        prefix_cache_manifest_sha256="4" * 64,
        rows=rows,
        groups=groups,
        filtered_train_jsonl_sha256="5" * 64,
        training_baseline_lock_sha256="6" * 64,
        v62_teacher_metadata_sha256="7" * 64,
        v62_teacher_weights_sha256="8" * 64,
        work_directory=tmp_path / "training" / "work",
        output_artifact=tmp_path / "training" / "final",
        optimizer={
            "learning_rate": 0.03,
            "minimum_steps": 2,
            "maximum_steps": 5,
            "nll_threshold": 0.001,
            "gradient_clip_norm": 1.0,
        },
        run_manifest={"run_signature_sha256": "9" * 64},
    )


def test_missing_class_groups_are_pair_specific_and_two_scene_verified() -> None:
    rows = (
        _row(1, 1, 1, "red"),
        _row(2, 2, 1, "red"),
        _row(3, 3, 2, "blue"),
        _row(4, 4, 2, "blue"),
        _row(5, 5, 3, "yes"),
    )
    teacher_keys = {rows[-1].key}

    groups = v66._group_missing_answer_classes(
        rows,
        teacher_keys,
        enforce_production_inventory=False,
    )

    assert [(group.answer_class_id, group.pair_id) for group in groups] == [
        (v66._answer_class_id("blue"), "pair_000002"),
        (v66._answer_class_id("red"), "pair_000001"),
    ]
    assert all(len(group.rows) == 2 for group in groups)
    assert all(len({row.scene_id for row in group.rows}) == 2 for group in groups)


def test_missing_class_groups_reject_single_scene_prototype() -> None:
    rows = (_row(1, 1, 1, "red"), _row(2, 2, 2, "yes"))

    with pytest.raises(ValueError, match="exactly two scenes"):
        v66._group_missing_answer_classes(
            rows,
            {rows[-1].key},
            enforce_production_inventory=False,
        )


def test_joint_optimizer_reduces_shared_two_row_objective(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    group = v66.V66TeacherGroup(
        v66._answer_class_id("red"),
        "pair_000001",
        (_row(1, 1, 1, "red"), _row(2, 2, 1, "red")),
    )

    def loss(**kwargs: object) -> torch.Tensor:
        prompt = kwargs["prompt"]
        assert isinstance(prompt, torch.Tensor)
        return (prompt - 0.1).square().mean()

    monkeypatch.setattr(v66, "_joint_teacher_nll", loss)
    prompt, metrics = v66._optimize_joint_prompt_adaptive(
        runtime=SimpleNamespace(),
        prefixes={},
        group=group,
        initial_prompt=torch.zeros(1, 4, 1536),
        learning_rate=0.03,
        min_steps=2,
        max_steps=20,
        nll_threshold=1e-4,
        gradient_clip_norm=1.0,
    )

    assert prompt.shape == (1, 4, 1536)
    assert metrics["final_mean_nll"] < metrics["initial_mean_nll"]
    assert metrics["training_row_count"] == 2


def test_final_cache_is_strict_numeric_opaque_and_create_once(tmp_path: Path) -> None:
    preflight = _preflight(tmp_path)
    completed = tuple(
        v66.V66CompletedPrototype(
            group=group,
            prompt=torch.full((1, 4, 1536), 0.01 + index / 1000),
            target_token_ids_sha256=f"{index + 10:064x}",
            optimization=_metrics(),
        )
        for index, group in enumerate(preflight.groups)
    )

    metadata = v66.save_v66_answer_class_teacher_cache(preflight, completed)
    loaded, validated = v66.load_v66_answer_class_teacher_cache(
        preflight.output_artifact
    )

    assert validated == metadata
    assert len(loaded) == 63
    assert metadata["answer_class_count"] == 14
    assert metadata["missing_answer_class_count"] == 7
    assert metadata["alternate_pair_class_count"] == 7
    assert metadata["greedy_canonical_exact"] == 126
    assert metadata["greedy_canonical_total"] == 126
    assert metadata["runtime_load_permitted"] is False
    assert metadata["environmental_text_inputs"] == []
    serialized = (preflight.output_artifact / "metadata.json").read_text(
        encoding="utf-8"
    )
    for answer in (
        "behind",
        "blue",
        "cream",
        "floor",
        "green",
        "in front",
        "no",
        "red",
        "terracotta",
        "upright",
        "wall",
        "wood",
        "yellow",
        "yes",
    ):
        assert json.dumps(answer) not in serialized
    assert '"question"' not in serialized
    assert '"answer"' not in serialized

    # Re-saving the exact cache validates and reuses it without overwriting.
    before = (preflight.output_artifact / "teachers.safetensors").read_bytes()
    assert v66.save_v66_answer_class_teacher_cache(preflight, completed) == metadata
    assert (preflight.output_artifact / "teachers.safetensors").read_bytes() == before


def test_strict_loader_rejects_metadata_text_injection(tmp_path: Path) -> None:
    preflight = _preflight(tmp_path)
    completed = tuple(
        v66.V66CompletedPrototype(
            group=group,
            prompt=torch.full((1, 4, 1536), 0.02),
            target_token_ids_sha256=f"{index + 20:064x}",
            optimization=_metrics(),
        )
        for index, group in enumerate(preflight.groups)
    )
    v66.save_v66_answer_class_teacher_cache(preflight, completed)
    metadata_path = preflight.output_artifact / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["answer"] = "red"
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

    with pytest.raises(ValueError, match="contract changed"):
        v66.load_v66_answer_class_teacher_cache(preflight.output_artifact)
