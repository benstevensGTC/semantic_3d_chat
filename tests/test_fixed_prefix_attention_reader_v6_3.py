from __future__ import annotations

from pathlib import Path

import torch
from torch import nn

from semantic_3d_chat.training import train_fixed_prefix_attention_reader_v6_3 as v63


class _WrappedLikeV54(nn.Module):
    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.base = nn.Linear(in_features, out_features, bias=False)
        self.base.requires_grad_(False)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.base(inputs)


def test_outer_residual_is_exact_zero_identity_and_only_fresh_factors_train() -> None:
    base = _WrappedLikeV54(7, 5)
    inputs = torch.randn(3, 7)
    expected = base(inputs)
    residual = v63.OuterAdditiveRankResidual(base, rank=2, alpha=4.0)
    installation = v63.OuterResidualInstallation(("fake.target",), (residual,))
    with torch.no_grad():
        residual.residual_a.normal_()
        residual.residual_b.zero_()
    assert torch.equal(residual(inputs), expected)
    assert residual.adapter_parameter_count == 24
    assert residual.residual_a.requires_grad
    assert residual.residual_b.requires_grad
    assert not any(parameter.requires_grad for parameter in residual.base.parameters())


def test_zero_output_initialization_has_b_gradient_but_zero_a_gradient() -> None:
    base = _WrappedLikeV54(7, 5)
    residual = v63.OuterAdditiveRankResidual(base, rank=2, alpha=4.0)
    installation = v63.OuterResidualInstallation(("fake.target",), (residual,))
    v63.initialize_outer_residuals(installation, seed=11)
    before = installation.state_sha256()
    residual(torch.randn(4, 7)).square().mean().backward()
    assert residual.residual_a.grad is not None
    assert residual.residual_b.grad is not None
    assert float(residual.residual_a.grad.norm()) == 0.0
    assert float(residual.residual_b.grad.norm()) > 0.0
    assert installation.state_sha256() == before


def test_softplus_margin_side_is_symmetric_scalar_and_prefers_larger_margin() -> None:
    weak, weak_margin = v63.softplus_margin_side(torch.tensor(2.0), torch.tensor(2.0))
    strong, strong_margin = v63.softplus_margin_side(torch.tensor(2.0), torch.tensor(3.0))
    assert float(weak_margin) == 0.0
    assert float(strong_margin) == 1.0
    assert float(strong) < float(weak)


def test_train_pair_units_and_balanced_schedule_cover_each_unit_once() -> None:
    rows = v63.v1.load_training_records()
    units = v63.build_pair_units(rows)
    schedule = v63.build_pilot_schedule(units)
    assert len(units) == 40
    assert len(schedule) == 8
    assert {len(update) for update in schedule} == {5}
    keys = [unit.key for update in schedule for unit in update]
    assert len(keys) == len(set(keys)) == 40
    assert all(unit.first.question == unit.second.question for unit in units)
    assert all(unit.first.answer != unit.second.answer for unit in units)


def test_config_and_source_lock_train_only_no_checkpoint_contract() -> None:
    config = v63._load_config()
    assert config["attention_reader"]["exact_targets"] == list(v63.TARGET_MODULES)
    assert config["attention_reader"]["trainable_parameter_count"] == 30_720
    assert config["pilot"]["updates"] == 8
    assert config["pilot"]["checkpoint_publication"] is False
    source = Path(v63.__file__).read_text(encoding="utf-8")
    assert "load_validation_records(" not in source
    assert "save_file(" not in source
    assert "down_proj" not in " ".join(v63.TARGET_MODULES)
    assert "OUTPUT_CHECKPOINT" not in source


def test_forbidden_roots_cover_validation_prefixes_and_oracle_is_component_blocked() -> None:
    roots = {path.resolve() for path in v63.training_forbidden_roots()}
    prefix_root = v63._resolve(v63.PREFIX_CACHE)
    assert all(
        (prefix_root / f"{scene_id}.safetensors").resolve() in roots
        for scene_id in v63.VALIDATION_SCENES
    )
