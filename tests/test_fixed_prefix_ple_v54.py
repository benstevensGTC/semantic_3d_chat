from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from torch import nn

from semantic_3d_chat.evaluation.fixed_prefix_ple_v54_preregistration import (
    TRAIN_SCENES,
    VALIDATION_SCENES,
    authenticate_inputs,
    build_preregistration,
    implementation_source_hashes,
    validate_objective,
    write_preregistration,
)
from semantic_3d_chat.evaluation.ple_reader_preregistration import (
    LORA_PARAMETER_COUNT,
    PROJECTION_IN_FEATURES,
    PROJECTION_OUT_FEATURES,
    TARGET_MODULE,
    reader_lora_settings,
)
from semantic_3d_chat.language.lora import install_lora_adapters
from semantic_3d_chat.training.train_fixed_prefix_ple_v54 import (
    build_schedule,
    load_retention_corpus,
    load_training_records,
    load_validation_records,
    memory_metrics,
)


class _TinyProjectionModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = nn.Module()
        self.model.language_model = nn.Module()
        self.model.language_model.per_layer_model_projection = nn.Linear(
            PROJECTION_IN_FEATURES, PROJECTION_OUT_FEATURES, bias=False
        )


def test_v54_ple_preregistration_is_one_fixed_prefix_reader_arm() -> None:
    contract = build_preregistration()

    assert contract["artifact"] == "gemma4_v54_fixed_prefix_ple_reader_v1"
    assert contract["independence"] == {
        "depends_on_failed_atlas": False,
        "depends_on_failed_question_controllers": False,
        "base": "accepted_v54_258_token_question_independent_prefix",
        "base_prefix_tokens": 258,
        "base_scene_latents": 256,
        "base_hidden_dimension": 1536,
        "all_scene_tokens_fit_inside_sliding_window_with_locked_prompts": True,
    }
    assert contract["trainable_surface"]["exact_target_modules"] == [TARGET_MODULE]
    assert contract["trainable_surface"]["parameter_count"] == LORA_PARAMETER_COUNT
    assert contract["optimization"]["one_arm_only"] is True
    assert contract["optimization"]["maximum_updates"] == 40
    assert contract["runtime_contract"]["environmental_text_inputs"] == []
    assert contract["runtime_contract"]["question_dependent_retrieval"] is False
    assert contract["implementation_source_hashes"] == implementation_source_hashes()
    assert contract["pinned_input_hashes"] == authenticate_inputs()


def test_v54_ple_scene_splits_and_records_are_strictly_disjoint() -> None:
    train = load_training_records()
    validation = load_validation_records()

    assert len(TRAIN_SCENES) == 24
    assert len(VALIDATION_SCENES) == 16
    assert set(TRAIN_SCENES).isdisjoint(VALIDATION_SCENES)
    assert len(train) == 576
    assert len(validation) == 384
    assert sum(row.changed for row in train) == 80
    assert sum(row.changed for row in validation) == 52
    assert all(row.paired_scene_id in TRAIN_SCENES for row in train if row.changed)
    assert all(row.paired_scene_id in VALIDATION_SCENES for row in validation if row.changed)


def test_v54_ple_schedule_covers_every_changed_pair_once() -> None:
    schedule = build_schedule(load_training_records())

    assert len(schedule) == 40
    changed = [row for update in schedule for row in update[:2]]
    broad = [row for update in schedule for row in update[2:]]
    assert len(changed) == 80
    assert len({(row.scene_id, row.question_id) for row in changed}) == 80
    assert len(broad) == 80
    assert all(not row.changed for row in broad)
    assert all(
        first.pair_id == second.pair_id
        and first.pair_question_key == second.pair_question_key
        and first.scene_id != second.scene_id
        for first, second, _broad_a, _broad_b in schedule
    )


def test_v54_ple_objective_is_answer_ce_plus_wrong_prefix_hinge() -> None:
    correct = torch.tensor([0.4], requires_grad=True)
    wrong = torch.tensor([0.5], requires_grad=True)
    total, diagnostics = validate_objective(correct, wrong)

    assert total.item() == pytest.approx(0.55)
    assert diagnostics["wrong_prefix_margins"].item() == pytest.approx(0.1)
    total.backward()
    assert correct.grad is not None and wrong.grad is not None


def test_v54_ple_exact_rank4_projection_surface() -> None:
    model = _TinyProjectionModel().requires_grad_(False)
    installation = install_lora_adapters(model, reader_lora_settings())

    assert installation is not None
    installation.assert_only_lora_trainable(model)
    assert installation.target_names == (TARGET_MODULE,)
    assert installation.parameter_count == 41_984
    assert all(parameter.dtype == torch.float32 for parameter in installation.parameters())


def test_v54_ple_retention_set_is_non_environmental_and_fixed() -> None:
    corpus = load_retention_corpus()
    serialized = json.dumps(corpus).casefold()

    assert len(corpus) == 16
    assert not any(
        term in serialized
        for term in ("chair", "bowl", "lamp", "cube", "picture frame", "scene_")
    )


def test_v54_ple_preregistration_is_create_once(tmp_path: Path) -> None:
    destination = tmp_path / "prereg.json"
    path, digest = write_preregistration(destination)

    assert path == destination.resolve()
    assert len(digest) == 64
    assert json.loads(destination.read_text(encoding="utf-8")) == build_preregistration()
    with pytest.raises(FileExistsError, match="already exists"):
        write_preregistration(destination)


def test_v54_ple_memory_metrics_are_numeric_and_optional_mps() -> None:
    metrics = memory_metrics()

    assert metrics["peak_process_rss_bytes"] > 0
    for key in ("mps_current_allocated_bytes", "mps_driver_allocated_bytes"):
        assert metrics[key] is None or metrics[key] >= 0


def test_v54_ple_source_has_no_question_conditioned_scene_readout() -> None:
    source = Path(
        "src/semantic_3d_chat/training/train_fixed_prefix_ple_v54.py"
    ).read_text(encoding="utf-8")
    launcher = Path("scripts/run_gemma4_v54_fixed_prefix_ple_reader.sh").read_text(
        encoding="utf-8"
    )

    assert "question_dependent_retrieval\": False" in source
    assert "environmental_text_inputs\": []" in source
    assert "top_k" not in source
    assert "atlas" not in launcher.casefold()
    assert "preregister|preflight|smoke|train|authenticate" in launcher
