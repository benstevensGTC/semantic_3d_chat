from __future__ import annotations

from collections import Counter
from types import SimpleNamespace

import pytest
import torch
from safetensors import safe_open

from semantic_3d_chat.scene_encoder.question_control_v74 import (
    DenseFullSceneContinuousControlV74,
)
from semantic_3d_chat.training.finetune_v74_gemma_nll import (
    EXPECTED_TRAIN_FAMILIES,
    assert_exclusive_v74_trainable_surface,
    deterministic_training_schedule_v74,
    save_v74_gemma_nll_diagnostic,
    select_balanced_historical_units_v74,
)
from semantic_3d_chat.training.train_question_control_v73 import (
    load_config_v73,
    load_training_rows_v73,
    split_rows_v73,
)


def _rows():
    config = load_config_v73(
        "configs/experiments/gemma4_v73_fullscene_controller.yaml"
    )
    return split_rows_v73(load_training_rows_v73(config["training_qa"]))


def _small_model() -> DenseFullSceneContinuousControlV74:
    basis = torch.eye(16, dtype=torch.float32)[:3]
    return DenseFullSceneContinuousControlV74(
        16,
        basis,
        environment_latents=4,
        query_count=2,
        model_dimension=4,
    )


def test_balanced_selection_is_train_only_paired_and_deterministic() -> None:
    train, held = _rows()
    first = select_balanced_historical_units_v74(train)
    second = select_balanced_historical_units_v74(train)
    assert [(unit.pair_id, unit.question_key) for unit in first] == [
        (unit.pair_id, unit.question_key) for unit in second
    ]
    assert tuple(unit.change_type for unit in first) == EXPECTED_TRAIN_FAMILIES
    assert len(first) == 9
    assert all(unit.left.question == unit.right.question for unit in first)
    assert len({row.key for unit in first for row in (unit.left, unit.right)}) == 18
    with pytest.raises(ValueError, match="escaped historical train pairs"):
        select_balanced_historical_units_v74(held)


def test_schedule_balances_every_selected_side_each_cycle() -> None:
    train, _held = _rows()
    units = select_balanced_historical_units_v74(train)
    first = deterministic_training_schedule_v74(units, cycles=3, seed=740176)
    second = deterministic_training_schedule_v74(units, cycles=3, seed=740176)
    assert [row.key for row in first] == [row.key for row in second]
    assert len(first) == 54
    assert set(Counter(row.key for row in first).values()) == {3}


def test_exclusive_trainable_audit_rejects_unfrozen_base() -> None:
    runtime = SimpleNamespace(
        language=SimpleNamespace(model=torch.nn.Linear(3, 3))
    )
    model = _small_model()
    with pytest.raises(RuntimeError, match="unexpectedly has trainable"):
        assert_exclusive_v74_trainable_surface(runtime, model)
    runtime.language.model.requires_grad_(False)
    audit = assert_exclusive_v74_trainable_surface(runtime, model)
    assert audit["base_trainable_parameter_count"] == 0
    assert audit["v74_trainable_parameter_count"] == audit["v74_parameter_count"]
    assert audit["only_v74_trainable"] is True


def test_diagnostic_output_is_minimal_finite_and_quarantined(tmp_path) -> None:
    model = _small_model()
    output = tmp_path / "diagnostic.safetensors"
    result = save_v74_gemma_nll_diagnostic(
        output,
        model,
        source_sha256="a" * 64,
        optimizer_steps=54,
        train_behavior_improved=True,
    )
    assert result["sha256"]
    with safe_open(str(output), framework="pt", device="cpu") as handle:
        metadata = dict(handle.metadata() or {})
        assert set(handle.keys()) == {
            "output_basis",
            "key.weight",
            "value.weight",
            "query.weight",
            "coefficient_output.weight",
        }
    assert metadata["runtime_promotion_forbidden_until_gemma_gate"] == "true"
    assert metadata["historical_train_pairs_only"] == "true"
    assert metadata["held_optimization_rows"] == "0"
    assert metadata["answer_codebook_serialized"] == "false"
    assert metadata["environmental_text_inputs"] == "0"
    with pytest.raises(FileExistsError):
        save_v74_gemma_nll_diagnostic(
            output,
            model,
            source_sha256="a" * 64,
            optimizer_steps=54,
            train_behavior_improved=True,
        )
