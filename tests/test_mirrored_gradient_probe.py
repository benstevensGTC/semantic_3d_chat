from __future__ import annotations

import hashlib

import pytest
import torch
from torch import nn

from semantic_3d_chat.evaluation.mirrored_gradient_probe import (
    ProbeBankSpec,
    cancellation_metrics,
    grouped_cancellation_metrics,
    initialize_probe_bank,
    objective_gradients,
    probe_parameter_items,
    require_autograd_compatible_scene_tokens,
    side_objective_terms,
    validate_checkpoint_provenance,
)
from semantic_3d_chat.language.lora import LoRASettings, install_lora_adapters


class _TinyAttention(nn.Module):
    def __init__(self, hidden: int = 6) -> None:
        super().__init__()
        self.q_proj = nn.Linear(hidden, hidden, bias=False)
        self.o_proj = nn.Linear(hidden, hidden, bias=False)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.o_proj(torch.tanh(self.q_proj(inputs)))


class _TinyLayer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.self_attn = _TinyAttention()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return inputs + self.self_attn(inputs)


class _TinyLanguageModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.ModuleList([_TinyLayer() for _ in range(5)])

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            inputs = layer(inputs)
        return inputs


class _TinyGemma(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = nn.Module()
        self.model.language_model = _TinyLanguageModel()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.model.language_model(inputs)


def _settings(*layers: int, rank: int = 2) -> LoRASettings:
    return LoRASettings(
        enabled=True,
        rank=rank,
        alpha=2.0 * rank,
        dropout=0.0,
        target_modules=tuple(
            f"model.language_model.layers.{layer}.self_attn.{projection}"
            for layer in layers
            for projection in ("q_proj", "o_proj")
        ),
    )


def test_probe_bank_is_separate_deterministic_zero_output_and_no_step() -> None:
    torch.manual_seed(91)
    model = _TinyGemma().eval().requires_grad_(False)
    persisted = install_lora_adapters(model, _settings(4))
    assert persisted is not None
    with torch.no_grad():
        for adapter in persisted.adapters:
            adapter.lora_b.normal_()
    model.requires_grad_(False)
    inputs = torch.randn(2, 3, 6)
    expected = model(inputs).detach().clone()
    persisted_hash = persisted.state_sha256()

    spec = ProbeBankSpec(layers=(0, 1, 2, 3), rank=8, alpha=16.0, seed=13008)
    probe = install_lora_adapters(model, spec.settings)
    assert probe is not None
    initialize_probe_bank(probe, seed=spec.seed)
    before = probe.state_sha256()

    assert torch.equal(model(inputs), expected)
    assert persisted.state_sha256() == persisted_hash
    assert all(torch.count_nonzero(adapter.lora_b) == 0 for adapter in probe.adapters)
    assert set(probe.target_names).isdisjoint(persisted.target_names)
    probe.assert_only_lora_trainable(model)

    loss = model(inputs).square().mean()
    parameters = probe_parameter_items(probe)
    gradients = torch.autograd.grad(loss, tuple(parameters.values()), allow_unused=False)
    assert any(torch.count_nonzero(gradient) > 0 for gradient in gradients)
    assert all(parameter.grad is None for parameter in model.parameters())
    assert probe.state_sha256() == before
    assert persisted.state_sha256() == persisted_hash

    second_model = _TinyGemma().eval().requires_grad_(False)
    second_probe = install_lora_adapters(second_model, spec.settings)
    assert second_probe is not None
    initialize_probe_bank(second_probe, seed=spec.seed)
    assert second_probe.state_sha256() == before


def test_cancellation_metrics_report_exact_opposition_by_layer_and_bank() -> None:
    gradients_a = {
        "model.language_model.layers.30.self_attn.q_proj.lora_a": torch.tensor([1.0, 2.0]),
        "model.language_model.layers.30.self_attn.q_proj.lora_b": torch.tensor([-3.0]),
        "model.language_model.layers.31.self_attn.o_proj.lora_a": torch.tensor([4.0]),
    }
    gradients_b = {name: -value for name, value in gradients_a.items()}

    direct = cancellation_metrics(gradients_a, gradients_b)
    grouped = grouped_cancellation_metrics(gradients_a, gradients_b)

    assert direct["cancellation_ratio"] == pytest.approx(0.0)
    assert direct["cosine_similarity"] == pytest.approx(-1.0)
    assert grouped["aggregate"] == direct
    assert set(grouped["per_layer"]) == {"30", "31"}
    assert all(
        metrics["cancellation_ratio"] == pytest.approx(0.0)
        for metrics in grouped["per_layer"].values()
    )


def test_exact_side_objective_decomposition_and_autograd_leave_parameters_unchanged() -> None:
    projection = nn.Parameter(torch.tensor([[0.3, -0.2], [-0.1, 0.4], [0.2, 0.1]]))
    hidden = torch.tensor([[[0.5, -0.25], [1.0, 0.5], [-0.5, 0.2]]])
    logits = hidden @ projection.T
    labels = torch.tensor([[-100, 1, 2]])
    before = projection.detach().clone()

    terms, diagnostics = side_objective_terms(
        logits,
        labels,
        candidate_spec=(0, 1, 0),
        candidate_margin=1.0,
        candidate_weight=8.0,
        full_vocab_margin=1.0,
        full_vocab_weight=2.0,
    )
    gradients = objective_gradients(terms, {"probe": projection})

    assert torch.allclose(
        gradients["decoder_total"]["probe"],
        gradients["language_nll"]["probe"]
        + gradients["candidate_hinge_weighted"]["probe"]
        + gradients["full_vocab_hinge_weighted"]["probe"],
    )
    assert terms["decoder_total"] == sum(
        terms[name]
        for name in (
            "language_nll",
            "candidate_hinge_weighted",
            "full_vocab_hinge_weighted",
        )
    )
    assert diagnostics["candidate_hinge_active"]
    assert diagnostics["full_vocab_hinge_active"]
    assert projection.grad is None
    assert torch.equal(projection, before)


def test_checkpoint_provenance_fails_closed_before_git_for_dirty_or_missing_record(
    tmp_path,
) -> None:
    with pytest.raises(TypeError, match="source-provenance"):
        validate_checkpoint_provenance({}, tmp_path)

    dirty = {
        "source_provenance": {
            "schema_version": 1,
            "scope": "repository_excluding_generated_artifacts_v1",
            "available": True,
            "head_commit": "a" * 40,
            "head_tree": "b" * 40,
            "is_clean": False,
            "tracked_diff_sha256": hashlib.sha256(b"").hexdigest(),
        }
    }
    with pytest.raises(ValueError, match="Invalid or dirty"):
        validate_checkpoint_provenance(dirty, tmp_path)


def test_probe_contract_rejects_noncanonical_layer_order() -> None:
    with pytest.raises(ValueError, match="strictly increasing"):
        ProbeBankSpec(layers=(31, 30))
    with pytest.raises(ValueError, match="unique"):
        ProbeBankSpec(layers=(30, 30))


def test_scene_tokens_for_lora_backward_cannot_be_inference_tensors() -> None:
    require_autograd_compatible_scene_tokens(torch.ones(1, 2, 3))
    with torch.inference_mode():
        inference_tokens = torch.ones(1, 2, 3)
    with pytest.raises(RuntimeError, match="inference tensors"):
        require_autograd_compatible_scene_tokens(inference_tokens)
