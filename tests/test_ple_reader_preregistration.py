from __future__ import annotations

import pytest
import torch
from torch import nn

from semantic_3d_chat.evaluation.ple_reader_preregistration import (
    LORA_PARAMETER_COUNT,
    PROJECTION_IN_FEATURES,
    PROJECTION_OUT_FEATURES,
    TARGET_MODULE,
    answer_only_wrong_prefix_objective,
    build_ple_reader_preregistration,
    reader_lora_settings,
    validate_launch_authorization,
    validate_projection_surface,
)
from semantic_3d_chat.language.lora import install_lora_adapters


class _TinyContractModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = nn.Module()
        self.model.language_model = nn.Module()
        self.model.language_model.per_layer_model_projection = nn.Linear(
            PROJECTION_IN_FEATURES, PROJECTION_OUT_FEATURES, bias=False
        )


def _authorization() -> dict[str, object]:
    return {
        "atlas_v2_checkpoint_sha256": "1" * 64,
        "atlas_v2_weights_sha256": "2" * 64,
        "atlas_v2_runtime_metadata_sha256": "3" * 64,
        "atlas_v2_acceptance_report_sha256": "4" * 64,
        "atlas_v2_accepted": True,
        "two_file_numeric_checkpoint_only": True,
        "compiler_frozen": True,
        "compiler_absent_at_runtime": True,
        "base_checkpoint_frozen": True,
        "question_independent_prefix": True,
        "complete_base_scene_prefix_preserved": True,
        "oracle_free_runtime": True,
        "fixed_prefix_tokens": 738,
    }


def test_exact_ple_projection_and_rank4_parameter_surface() -> None:
    model = _TinyContractModel().requires_grad_(False)
    projection = validate_projection_surface(model)
    assert tuple(projection.weight.shape) == (8960, 1536)

    installation = install_lora_adapters(model, reader_lora_settings())
    assert installation is not None
    installation.assert_only_lora_trainable(model)
    assert installation.target_names == (TARGET_MODULE,)
    assert installation.parameter_count == LORA_PARAMETER_COUNT == 41_984
    assert all(parameter.dtype == torch.float32 for parameter in installation.parameters())
    assert torch.count_nonzero(installation.adapters[0].lora_b) == 0


def test_projection_shape_drift_fails_closed() -> None:
    model = _TinyContractModel()
    model.model.language_model.per_layer_model_projection = nn.Linear(8, 9, bias=False)
    with pytest.raises(ValueError, match="contract changed"):
        validate_projection_surface(model)


def test_answer_ce_plus_same_question_wrong_prefix_margin() -> None:
    correct = torch.tensor([0.4, 0.8], requires_grad=True)
    wrong = torch.tensor([0.5, 1.4], requires_grad=True)
    total, diagnostics = answer_only_wrong_prefix_objective(correct, wrong, margin=0.25)

    # CE=.6; hinges=(.15,0), averaged=.075.
    assert total.item() == pytest.approx(0.675)
    assert diagnostics["wrong_prefix_margins"].tolist() == pytest.approx([0.1, 0.6])
    total.backward()
    assert correct.grad is not None and wrong.grad is not None
    assert torch.isfinite(correct.grad).all() and torch.isfinite(wrong.grad).all()


def test_launch_requires_concrete_accepted_atlas_v2_hashes() -> None:
    assert validate_launch_authorization(_authorization())["fixed_prefix_tokens"] == 738
    bad = _authorization()
    bad["atlas_v2_checkpoint_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="placeholder"):
        validate_launch_authorization(bad)
    bad = _authorization()
    bad["compiler_absent_at_runtime"] = False
    with pytest.raises(ValueError, match="compiler_absent_at_runtime"):
        validate_launch_authorization(bad)


def test_preregistration_freezes_reader_leakage_and_retention_contracts() -> None:
    contract = build_ple_reader_preregistration()
    assert contract["status"] == "design_locked_training_not_authorized"
    assert contract["trainable_surface"]["parameter_count"] == 41_984
    assert contract["atlas_v2_launch_authorization"][
        "no_mutable_atlas_v2_source_hash_is_preregistered"
    ] is True
    assert contract["objective"]["wrong_prefix_changes_only_scene_prefix"] is True
    assert contract["runtime_and_leakage"]["environmental_text_inputs"] == []
    assert contract["text_retention_controls"]["measure_ce_kl_and_next_token_top1_agreement"]
    assert contract["execution"] == {
        "training_executed": False,
        "gemma_generation_executed": False,
        "checkpoint_published": False,
        "implementation_must_be_new_files_or_explicitly_unsealed_before_run": True,
    }
