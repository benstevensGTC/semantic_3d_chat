from __future__ import annotations

import builtins
from pathlib import Path

import pytest
import torch

from semantic_3d_chat.evaluation.v94_strict_multiscene_preflight import (
    EXPECTED_EVALUATION_SCENES,
    EXPECTED_INITIAL_STATE_SHA256,
    causal_sides_v94,
    class_weights_v94,
    derive_contract_v94,
    load_config_v94,
    load_scene_memories_v94,
    load_training_rows_v94,
    lora_preflight_v94,
    training_schedule_v94,
    zero_payload_memory_v94,
)


def test_v94_binds_all_forty_training_scenes_and_fixed_schedule() -> None:
    config = load_config_v94()
    rows = load_training_rows_v94(config)
    schedule = training_schedule_v94(rows)
    memories, hashes = load_scene_memories_v94(config, rows)

    assert len(rows) == 960
    assert len({row.scene_id for row in rows}) == 40
    assert len({row.pair_id for row in rows}) == 20
    assert len(schedule) == 2880
    assert len(memories) == len(hashes) == 40
    assert all(value.shape == (1, 738, 1536) for value in memories.values())
    assert all(value.dtype == torch.bfloat16 for value in memories.values())


def test_v94_inverse_sqrt_weights_and_causal_subset_are_exact() -> None:
    config = load_config_v94()
    rows = load_training_rows_v94(config)
    weights = class_weights_v94(config, rows)
    causal = causal_sides_v94(config, rows)

    assert len(weights) == 29
    assert sum(weights[row.answer_class] for row in rows) / len(rows) == pytest.approx(1.0)
    assert len(causal) == 18
    assert len({row.change_type for row in causal}) == 9
    assert all(row.expected_change for row in causal)


def test_v94_fresh_bank_is_exact_zero_and_native_boundaries_survive_control() -> None:
    config = load_config_v94()
    preflight = lora_preflight_v94(config)
    rows = load_training_rows_v94(config)
    memories, _hashes = load_scene_memories_v94(config, rows)
    memory = memories[min(memories)]
    zero = zero_payload_memory_v94(memory)

    assert preflight["initial_state_sha256"] == EXPECTED_INITIAL_STATE_SHA256
    assert preflight["parameter_count"] == 110592
    assert torch.equal(zero[:, :1], memory[:, :1])
    assert torch.equal(zero[:, -1:], memory[:, -1:])
    assert torch.count_nonzero(zero[:, 1:-1]).item() == 0


def test_v94_draft_derivation_never_opens_reserved_validation_labels(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = load_config_v94()
    reserved = Path(config["sources"]["evaluation_qa_reserved_for_label_scorer"]).resolve()
    original_open = builtins.open
    original_path_open = Path.open

    def guarded_open(file: object, *args: object, **kwargs: object) -> object:
        if Path(file).resolve() == reserved:  # type: ignore[arg-type]
            raise AssertionError("V94 preflight opened reserved labels")
        return original_open(file, *args, **kwargs)  # type: ignore[arg-type]

    def guarded_path_open(path: Path, *args: object, **kwargs: object) -> object:
        if path.resolve() == reserved:
            raise AssertionError("V94 preflight opened reserved labels")
        return original_path_open(path, *args, **kwargs)

    monkeypatch.setattr(builtins, "open", guarded_open)
    monkeypatch.setattr(Path, "open", guarded_path_open)
    result = derive_contract_v94()

    assert result["evaluation_label_file_opened"] is False
    assert result["evaluation_scene_ids"] == list(EXPECTED_EVALUATION_SCENES)
    assert result["full_gemma_model_loaded"] is False
    assert result["optimizer_updates_performed"] == 0
