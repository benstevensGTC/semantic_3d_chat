from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from semantic_3d_chat.chat.question_control_runtime import _load_control_head
from semantic_3d_chat.chat.runtime_config import effective_runtime_config_sha256
from semantic_3d_chat.data.dataset import QARecord
from semantic_3d_chat.language.prefix_injection import prefix_sha256
from semantic_3d_chat.scene_encoder.question_control import FullSceneQuestionControl
from semantic_3d_chat.training.train_question_control_v56 import (
    _load_sanitized_runtime_config,
    assert_answer_only_labels,
    build_curriculum,
    build_runtime_metadata,
    curriculum_summary,
    ensure_prefix_cache,
    load_prefix_cache,
    load_training_records,
    save_control_checkpoint,
    validate_training_scene_ids,
)

_DIGEST_A = "a" * 64
_DIGEST_B = "b" * 64


def _record(
    scene_id: str,
    question_id: str,
    *,
    answer_type: str,
    question: str = "opaque training question?",
    answer: str = "yes",
    pair_id: str | None = None,
    question_key: str | None = None,
    role: str | None = None,
    changed: bool | None = None,
) -> QARecord:
    return QARecord(
        scene_id=scene_id,
        question_id=question_id,
        question=question,
        answer=answer,
        answer_type=answer_type,
        target_xyz=None,
        counterfactual_pair_id=pair_id,
        counterfactual_question_key=question_key,
        counterfactual_expected_change=changed,
        counterfactual_role=role,
        counterfactual_change_type="opaque_change" if changed else None,
    )


def _curriculum_records() -> list[QARecord]:
    paired_question = "same locked physical question?"
    return [
        _record(
            "scene_000011",
            "q_pair_a",
            answer_type="spatial_relation",
            question=paired_question,
            answer="left",
            pair_id="pair_opaque",
            question_key="key_opaque",
            role="reference",
            changed=True,
        ),
        _record(
            "scene_000012",
            "q_pair_b",
            answer_type="spatial_relation",
            question=paired_question,
            answer="right",
            pair_id="pair_opaque",
            question_key="key_opaque",
            role="counterfactual",
            changed=True,
        ),
        _record("scene_000011", "q_count_a", answer_type="count", answer="1"),
        _record("scene_000012", "q_count_b", answer_type="count", answer="2"),
        _record("scene_000011", "q_broad_a", answer_type="presence"),
        _record("scene_000012", "q_broad_b", answer_type="attribute", answer="blue"),
    ]


def test_v56_scene_allowlist_includes_exhausted_dev_and_excludes_sealed_splits() -> None:
    assert validate_training_scene_ids(
        ("scene_000011", "scene_000019", "scene_000024", "scene_000031", "scene_000056")
    ) == (
        "scene_000011",
        "scene_000019",
        "scene_000024",
        "scene_000031",
        "scene_000056",
    )

    for number in (*range(25, 31), *range(57, 63), 10, 63):
        with pytest.raises(ValueError):
            validate_training_scene_ids((f"scene_{number:06d}",))


def test_v56_training_reader_filters_explicit_scenes_but_audits_every_row(
    tmp_path: Path,
) -> None:
    source = tmp_path / "train.jsonl"

    def row(scene_id: str, question_id: str) -> dict[str, str]:
        return {
            "scene_id": scene_id,
            "question_id": question_id,
            "question": "training question?",
            "answer": "yes",
            "answer_type": "presence",
        }

    source.write_text(
        "\n".join(
            json.dumps(value)
            for value in (
                row("scene_000011", "q_1"),
                row("scene_000012", "q_2"),
            )
        )
        + "\n",
        encoding="utf-8",
    )
    records, digest = load_training_records(source, scene_ids=("scene_000011",))
    assert [(record.scene_id, record.question_id) for record in records] == [
        ("scene_000011", "q_1")
    ]
    assert len(digest) == 64

    source.write_text(
        source.read_text(encoding="utf-8")
        + json.dumps(row("scene_000025", "q_forbidden"))
        + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="Deferred final"):
        load_training_records(source, scene_ids=("scene_000011",))


def test_v56_curriculum_is_deterministic_and_keeps_changed_sides_atomic() -> None:
    records = _curriculum_records()
    first = build_curriculum(
        records,
        epochs=2,
        seed=56,
        changed_pair_repeats=2,
        count_replay_repeats=2,
        broad_repeats=1,
        replay_batch_size=2,
    )
    second = build_curriculum(
        records,
        epochs=2,
        seed=56,
        changed_pair_repeats=2,
        count_replay_repeats=2,
        broad_repeats=1,
        replay_batch_size=2,
    )
    assert [step.signature() for step in first] == [step.signature() for step in second]
    assert [step.kind for step in first[:3]] == [
        "changed_pair",
        "count_replay",
        "broad",
    ]
    changed = [step for step in first if step.kind == "changed_pair"]
    assert len(changed) == 4
    assert all(len(step.records) == 2 for step in changed)
    assert all(len({record.scene_id for record in step.records}) == 2 for step in changed)
    summary = curriculum_summary(first)
    assert summary["steps_by_kind"] == {
        "broad": 2,
        "changed_pair": 4,
        "count_replay": 4,
    }
    assert summary["paired_two_side_optimizer_steps"] is True


def test_v56_curriculum_rejects_split_or_malformed_changed_unit() -> None:
    records = _curriculum_records()
    records.pop(1)
    with pytest.raises(ValueError, match="locked two-side"):
        build_curriculum(records, epochs=1, seed=1)


def test_v56_runtime_metadata_is_exact_and_contains_no_training_provenance() -> None:
    control = FullSceneQuestionControl(
        8,
        attention_dim=4,
        control_tokens=2,
        uniform_floor=0.1,
        output_scale=0.2,
    )
    metadata = build_runtime_metadata(
        control,
        weights_sha256=_DIGEST_A,
        base_checkpoint_sha256=_DIGEST_B,
        base_runtime_config_sha256="c" * 64,
    )
    assert set(metadata) == {
        "schema_version",
        "architecture",
        "hidden_size",
        "attention_dim",
        "control_tokens",
        "uniform_floor",
        "output_scale",
        "weights_sha256",
        "base_checkpoint_sha256",
        "base_runtime_config_sha256",
        "question_dependent_scene_retrieval",
        "complete_scene_prefix_required",
        "environmental_text_inputs",
    }
    assert metadata["environmental_text_inputs"] == []
    assert metadata["question_dependent_scene_retrieval"] is False
    encoded = json.dumps(metadata, sort_keys=True)
    assert "scene_000" not in encoded
    assert "training_qa" not in encoded
    assert "oracle" not in encoded
    assert "chair" not in encoded


def test_v56_sanitized_runtime_config_has_locked_effective_hash() -> None:
    config, path = _load_sanitized_runtime_config(
        "configs/runtime/gemma4_v56_question_control.yaml"
    )
    assert path.name == "gemma4_v56_question_control.yaml"
    assert config["_runtime_safe_config"] is True
    assert effective_runtime_config_sha256(config) == (
        "714c60ce9ccb1dff69c72f6618f8afb6f31bc60a830b5ee0fb794fedaa8a321e"
    )


class _FakeRuntime:
    def __init__(self, scene_id: str) -> None:
        value = int(scene_id[-2:]) / 100.0
        self.scene_prefix = torch.full((1, 6, 8), value, dtype=torch.float32)
        self.scene_prefix_hash = prefix_sha256(self.scene_prefix)

    def assert_prefix_unchanged(self) -> None:
        assert prefix_sha256(self.scene_prefix) == self.scene_prefix_hash


def test_v56_prefix_cache_is_exact_reusable_and_tamper_evident(tmp_path: Path) -> None:
    cache_path = tmp_path / "prefix_cache_v56"
    calls: list[str] = []

    def loader(scene_id: str) -> _FakeRuntime:
        calls.append(scene_id)
        return _FakeRuntime(scene_id)

    scenes = ("scene_000011", "scene_000019")
    built = ensure_prefix_cache(
        cache_path,
        scene_ids=scenes,
        base_checkpoint_sha256=_DIGEST_A,
        base_runtime_config_sha256=_DIGEST_B,
        runtime_loader=loader,
    )
    assert built.created is True
    assert calls == ["scene_000019", "scene_000011"]
    assert built.retained_runtime is not None
    assert built.manifest["question_inputs_used"] is False
    assert built.manifest["environmental_text_inputs"] == []
    assert set(built.prefixes) == set(scenes)

    calls.clear()
    reused = ensure_prefix_cache(
        cache_path,
        scene_ids=scenes,
        base_checkpoint_sha256=_DIGEST_A,
        base_runtime_config_sha256=_DIGEST_B,
        runtime_loader=loader,
    )
    assert reused.created is False
    assert reused.retained_runtime is None
    assert calls == []
    assert torch.equal(reused.prefixes[scenes[0]], built.prefixes[scenes[0]])

    with (cache_path / "scene_000011.safetensors").open("ab") as handle:
        handle.write(b"tamper")
    with pytest.raises(ValueError, match="bytes changed"):
        load_prefix_cache(
            cache_path,
            scene_ids=scenes,
            base_checkpoint_sha256=_DIGEST_A,
            base_runtime_config_sha256=_DIGEST_B,
        )


def test_v56_control_checkpoint_is_runtime_minimal_and_loader_compatible(
    tmp_path: Path,
) -> None:
    torch.manual_seed(7)
    control = FullSceneQuestionControl(8, attention_dim=4, control_tokens=2)
    checkpoint = tmp_path / "control_checkpoint_v56"
    hashes = save_control_checkpoint(
        checkpoint,
        control=control,
        base_checkpoint_sha256=_DIGEST_A,
        base_runtime_config_sha256=_DIGEST_B,
    )
    assert {item.name for item in checkpoint.iterdir()} == {
        "control.safetensors",
        "runtime_metadata.json",
    }
    loaded, metadata = _load_control_head(
        checkpoint,
        hidden_size=8,
        device=torch.device("cpu"),
    )
    assert loaded.parameter_count == control.parameter_count
    assert metadata["weights_sha256"] == hashes["weights_sha256"]
    assert all(value.dtype == torch.float32 for value in loaded.state_dict().values())
    with pytest.raises(FileExistsError):
        save_control_checkpoint(
            checkpoint,
            control=control,
            base_checkpoint_sha256=_DIGEST_A,
            base_runtime_config_sha256=_DIGEST_B,
        )


def test_v56_answer_labels_are_strictly_answer_only() -> None:
    answer = torch.tensor([[5, 6]], dtype=torch.long)
    assert_answer_only_labels(torch.tensor([[-100, -100, 5, 6]]), answer)
    with pytest.raises(ValueError, match="answer-only"):
        assert_answer_only_labels(torch.tensor([[-100, 4, 5, 6]]), answer)
    with pytest.raises(ValueError, match="answer-only"):
        assert_answer_only_labels(torch.tensor([[-100, -100, 5, 7]]), answer)
