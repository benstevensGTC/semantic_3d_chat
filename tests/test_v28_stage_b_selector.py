from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import save_file

from semantic_3d_chat.config import load_config
from semantic_3d_chat.evaluation import v28_stage_b_selector as selector
from semantic_3d_chat.evaluation.v28_stage_b_selector import (
    FRESH_BANK_NAME,
    FRESH_BANK_PREFIX,
    _checkpoint_paths,
    _fresh_bank_audit,
    _frozen_tensor_sha256,
    _select_eligible_arm,
    _selection_requirements,
    _validate_runtime_metadata,
    _validate_update_zero,
)


def _write_checkpoint(
    root: Path,
    update: int,
    *,
    validation_nll: float = 2.0,
    fresh_value: float = 0.0,
) -> Path:
    path = root / f"update_{update:03d}"
    path.mkdir()
    state = {
        "scene_model.weight": torch.tensor([7.0]),
        f"{FRESH_BANK_PREFIX}adapters.0.lora_a": torch.tensor([[1.0, 2.0]]),
        f"{FRESH_BANK_PREFIX}adapters.0.lora_b": torch.tensor([[fresh_value]]),
    }
    save_file(state, path / "adapter.safetensors")
    fresh = {
        name.removeprefix(FRESH_BANK_PREFIX): value
        for name, value in state.items()
        if name.startswith(FRESH_BANK_PREFIX)
    }
    fresh_hash = selector.tensor_state_sha256(fresh)
    frozen_hash = _frozen_tensor_sha256(state)
    selected_arm = {"update": 2, "eligible": True}
    metadata = {
        "optimizer_step": update,
        "history": [{"validation_answer_token_nll": validation_nll}],
        "lora_bank_state_sha256": {FRESH_BANK_NAME: fresh_hash},
        "v28_stage_b": {
            "frozen_state_sha256": frozen_hash,
            "stage_a_selected_update": 2,
            "stage_a_selected_arm": selected_arm,
            "source_stage_a_checkpoint": "/approved/stage_a/update_002",
            "update_zero_equivalence": {
                "verified": True,
                "base": "selector_approved_stage_a_checkpoint",
                "bank": FRESH_BANK_NAME,
                "state_sha256": fresh_hash,
                "parameter_count": 3,
                "question_dependent_scene_processing": False,
                "validation_nll_equivalent": True,
            },
        },
    }
    (path / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    runtime_metadata = selector.runtime_checkpoint_metadata(metadata)
    (path / "runtime_metadata.json").write_text(
        json.dumps(runtime_metadata), encoding="utf-8"
    )
    return path


def test_checkpoint_inventory_is_complete_contiguous_and_bounded(tmp_path: Path) -> None:
    complete = tmp_path / "complete"
    complete.mkdir()
    for update in range(3):
        _write_checkpoint(complete, update)
    assert [path.name for path in _checkpoint_paths(complete, expected_final_update=2)] == [
        "update_000",
        "update_001",
        "update_002",
    ]
    with pytest.raises(FileNotFoundError, match="incomplete"):
        _checkpoint_paths(complete, expected_final_update=3)

    gap = tmp_path / "gap"
    gap.mkdir()
    _write_checkpoint(gap, 0)
    _write_checkpoint(gap, 2)
    with pytest.raises(FileNotFoundError, match="not contiguous"):
        _checkpoint_paths(gap)

    missing_file = tmp_path / "missing_file"
    missing_file.mkdir()
    path = _write_checkpoint(missing_file, 0)
    (path / "runtime_metadata.json").unlink()
    with pytest.raises(FileNotFoundError, match="Incomplete Stage-B checkpoint"):
        _checkpoint_paths(missing_file)


def test_frozen_hash_excludes_only_fresh_bank_prefix() -> None:
    base = {
        "scene_model.weight": torch.tensor([1.0]),
        "lora_banks.inherited.adapters.0.lora_a": torch.tensor([2.0]),
        f"{FRESH_BANK_PREFIX}adapters.0.lora_a": torch.tensor([3.0]),
        f"{FRESH_BANK_PREFIX}adapters.0.lora_b": torch.tensor([0.0]),
    }
    changed_fresh = {
        **base,
        f"{FRESH_BANK_PREFIX}adapters.0.lora_b": torch.tensor([9.0]),
    }
    assert _frozen_tensor_sha256(base) == _frozen_tensor_sha256(changed_fresh)
    changed_inherited = {
        **base,
        "lora_banks.inherited.adapters.0.lora_a": torch.tensor([8.0]),
    }
    assert _frozen_tensor_sha256(base) != _frozen_tensor_sha256(changed_inherited)


def test_runtime_metadata_must_match_sanitized_training_state(tmp_path: Path) -> None:
    root = tmp_path / "root"
    root.mkdir()
    checkpoint = _write_checkpoint(root, 0)
    metadata = selector._metadata(checkpoint)

    assert _validate_runtime_metadata(checkpoint, metadata) == (
        selector.runtime_checkpoint_metadata(metadata)
    )

    (checkpoint / "runtime_metadata.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="Runtime/training metadata mismatch"):
        _validate_runtime_metadata(checkpoint, metadata)


def test_update_zero_requires_selected_stage_a_and_exact_zero_bank(tmp_path: Path) -> None:
    root = tmp_path / "root"
    root.mkdir()
    checkpoint = _write_checkpoint(root, 0)
    tensors = selector.load_file(checkpoint / "adapter.safetensors", device="cpu")
    metadata = selector._metadata(checkpoint)
    audit = _fresh_bank_audit(tensors, metadata, expected_parameter_count=3)
    result = _validate_update_zero(
        metadata,
        audit,
        expected_initial_hash=str(audit["state_sha256"]),
    )
    assert result["verified"] is True
    assert result["fresh_bank_exact_zero_output"] is True
    assert result["stage_a_selected_update"] == 2

    bad_tensors = dict(tensors)
    bad_tensors[f"{FRESH_BANK_PREFIX}adapters.0.lora_b"] = torch.ones(1, 1)
    bad_state = selector._fresh_bank_state(bad_tensors)
    bad_metadata = dict(metadata)
    bad_metadata["lora_bank_state_sha256"] = {
        FRESH_BANK_NAME: selector.tensor_state_sha256(bad_state)
    }
    bad_audit = _fresh_bank_audit(
        bad_tensors, bad_metadata, expected_parameter_count=3
    )
    with pytest.raises(ValueError, match="not exact-zero"):
        _validate_update_zero(
            bad_metadata,
            bad_audit,
            expected_initial_hash=str(bad_audit["state_sha256"]),
        )


def test_config_contract_is_rank4_two_layer_fresh_query_bank() -> None:
    config = load_config(
        Path(
            "configs/experiments/"
            "gemma4_color_mirror_post_stack_decoder_stage_b_v28.yaml"
        )
    )
    minimum_color, minimum_mirror, updates, initial_hash, parameter_count = (
        _selection_requirements(config)
    )
    assert (minimum_color, minimum_mirror, updates, parameter_count) == (12, 10, 4, 36864)
    assert len(initial_hash) == 64
    bank = selector.lora_banks_settings(config).bank(FRESH_BANK_NAME)
    assert bank.trainable is True
    assert bank.adapter.rank == 4
    assert bank.adapter.alpha == pytest.approx(8.0)
    assert bank.adapter.target_modules == (
        "model.language_model.layers.13.self_attn.q_proj",
        "model.language_model.layers.14.self_attn.q_proj",
    )


def test_selection_prefers_minimum_nll_then_earliest_update() -> None:
    arms = [
        {"eligible": False, "validation_answer_token_nll": 0.1, "update": 1},
        {"eligible": True, "validation_answer_token_nll": 1.0, "update": 3},
        {"eligible": True, "validation_answer_token_nll": 1.0, "update": 2},
    ]
    assert _select_eligible_arm(arms)["update"] == 2
    assert _select_eligible_arm([{**arms[0]}]) is None


def test_selector_rejects_better_nll_when_causal_retention_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "stage_b"
    root.mkdir()
    checkpoints = [
        _write_checkpoint(root, 0, validation_nll=2.0),
        _write_checkpoint(root, 1, validation_nll=0.5, fresh_value=0.1),
        _write_checkpoint(root, 2, validation_nll=1.0, fresh_value=0.2),
    ]
    initial_tensors = selector.load_file(
        checkpoints[0] / "adapter.safetensors", device="cpu"
    )
    initial_hash = selector.tensor_state_sha256(
        selector._fresh_bank_state(initial_tensors)
    )
    config = {
        "seed": 1,
        "scene": {"room_size_m": [6.0, 5.0, 3.0]},
        "scene_encoder": {"input_voxel_size_m": 0.05},
    }
    monkeypatch.setattr(selector, "load_config", lambda _path: config)
    monkeypatch.setattr(
        selector,
        "_selection_requirements",
        lambda _config: (12, 10, 2, initial_hash, 3),
    )
    monkeypatch.setattr(
        selector,
        "pair_curriculum_settings",
        lambda _config: SimpleNamespace(pair_only_scene_ids=(), max_units_per_pair=6),
    )
    monkeypatch.setattr(
        selector,
        "SceneQADataset",
        lambda _path: SimpleNamespace(records=[object()]),
    )
    monkeypatch.setattr(selector, "project_path", lambda *_args: tmp_path / "unused")
    monkeypatch.setattr(selector, "select_pair_only_records", lambda records, _ids: records)
    monkeypatch.setattr(
        selector,
        "cap_pair_units_per_pair",
        lambda records, _limit, seed: records,
    )
    unit = SimpleNamespace(scene_ids=("scene_000001",))
    monkeypatch.setattr(selector, "build_exact_question_pair_units", lambda _records: [unit] * 12)
    fake_runtime = SimpleNamespace(language=SimpleNamespace(device=torch.device("cpu")))
    monkeypatch.setattr(
        selector.StaticChatRuntime,
        "load",
        staticmethod(lambda *_args, **_kwargs: fake_runtime),
    )

    class StateRecorder:
        def __init__(self) -> None:
            self.index = -1

        def load_state_dict(self, _state, strict: bool) -> None:
            assert strict is True
            self.index += 1

    recorder = StateRecorder()
    monkeypatch.setattr(
        selector, "_runtime_fresh_bank_state_module", lambda *_args: recorder
    )
    monkeypatch.setattr(selector, "load_map_tensors", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(selector, "_pair_role_ids", lambda _config: ("color", "mirror"))
    monkeypatch.setattr(
        selector,
        "_teacher_gate",
        lambda **_kwargs: {"step": recorder.index, "by_pair": {"color": {}, "mirror": {}}},
    )
    score_by_step = {0: (12, 10), 1: (11, 10), 2: (12, 10)}
    count_calls = {0: 0, 1: 0, 2: 0}

    def fake_counts(_gate: dict) -> tuple[int, int]:
        step = recorder.index
        role_index = count_calls[step]
        count_calls[step] += 1
        return score_by_step[step][role_index], 6

    monkeypatch.setattr(selector, "_full_vocab_counts", fake_counts)
    monkeypatch.setattr(selector, "_negative_sides", lambda _gate: {("s", "q")})

    report = selector.select_stage_b(tmp_path / "config.yaml", root)
    assert report["passed"] is True
    assert report["selected_update"] == 2
    assert report["arms"][1]["validation_answer_token_nll"] == pytest.approx(0.5)
    assert report["arms"][1]["eligible"] is False
    assert report["arms"][1]["checks"]["color_retained"] is False
    assert report["arms"][2]["eligible"] is True
    assert report["oracle_loaded"] is False
    assert report["question_text_serialized"] is False
    assert report["question_dependent_scene_processing"] is False
    assert report["question_dependent_retrieval"] is False
    assert report["model_load_count"] == 1
