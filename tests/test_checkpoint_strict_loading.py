from __future__ import annotations

import pytest
import torch
from torch import nn

from semantic_3d_chat.scene_encoder.global_residual import (
    ZERO_SPATIAL_MEAN_CONTENT_GATE_V1,
    GlobalSceneResidual,
)
from semantic_3d_chat.training.checkpointing import (
    load_adapter_checkpoint,
    save_adapter_checkpoint,
)


class _OldBridge(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.base = nn.Linear(3, 4)


class _ExtendedBridge(_OldBridge):
    def __init__(self) -> None:
        super().__init__()
        self.zero_residual = nn.Linear(4, 4, bias=False)
        nn.init.zeros_(self.zero_residual.weight)


def _legacy_residual() -> GlobalSceneResidual:
    return GlobalSceneResidual(
        scene_dim=8,
        latent_count=4,
        width=3,
        fourier_bands=2,
        initialization_seed=18100,
    )


def _content_gated_residual() -> GlobalSceneResidual:
    return GlobalSceneResidual(
        scene_dim=8,
        latent_count=4,
        width=3,
        fourier_bands=2,
        initialization_seed=18100,
        architecture_version=ZERO_SPATIAL_MEAN_CONTENT_GATE_V1,
        gate_temperature=0.75,
    )


def test_checkpoint_loader_rejects_every_unconsumed_tensor(tmp_path) -> None:
    kept = nn.Linear(3, 2)
    checkpoint = save_adapter_checkpoint(
        tmp_path / "unconsumed",
        {"kept": kept, "stale": nn.Linear(2, 2)},
        {"schema_version": 1},
    )

    with pytest.raises(RuntimeError, match="unconsumed tensor keys.*stale"):
        load_adapter_checkpoint(
            checkpoint,
            {"kept": nn.Linear(3, 2)},
            device="cpu",
        )


def test_checkpoint_loader_allows_only_explicit_missing_module_prefix(tmp_path) -> None:
    source = _OldBridge()
    checkpoint = save_adapter_checkpoint(
        tmp_path / "migration",
        {"bridge": source},
        {"schema_version": 1, "migration": "add_zero_residual"},
    )

    with pytest.raises(RuntimeError, match="Missing key|missing key"):
        load_adapter_checkpoint(checkpoint, {"bridge": _ExtendedBridge()}, device="cpu")
    with pytest.raises(RuntimeError, match="forbidden_missing"):
        load_adapter_checkpoint(
            checkpoint,
            {"bridge": _ExtendedBridge()},
            device="cpu",
            allowed_missing_key_prefixes={"bridge": ("some_other_prefix.",)},
        )

    restored = _ExtendedBridge()
    metadata = load_adapter_checkpoint(
        checkpoint,
        {"bridge": restored},
        device="cpu",
        allowed_missing_key_prefixes={"bridge": ("zero_residual.",)},
    )

    assert metadata == {"schema_version": 1, "migration": "add_zero_residual"}
    assert torch.equal(restored.base.weight, source.base.weight)
    assert torch.equal(restored.base.bias, source.base.bias)
    assert torch.count_nonzero(restored.zero_residual.weight).item() == 0


def test_checkpoint_roundtrips_content_gated_residual_parameters_and_buffers(
    tmp_path,
) -> None:
    source = _content_gated_residual()
    with torch.no_grad():
        source.output_projection.weight.fill_(0.03125)
        source.content_gate_projection.weight.mul_(1.5)
    checkpoint = save_adapter_checkpoint(
        tmp_path / "content_gated_roundtrip",
        {"global_scene_residual": source},
        {"schema_version": 3},
    )
    restored = _content_gated_residual()

    metadata = load_adapter_checkpoint(
        checkpoint,
        {"global_scene_residual": restored},
        device="cpu",
    )

    assert metadata == {"schema_version": 3}
    assert restored.validate_structural_state()["gate_temperature"] == pytest.approx(0.75)
    assert set(restored.state_dict()) == set(source.state_dict())
    for name, expected in source.state_dict().items():
        assert torch.equal(restored.state_dict()[name], expected)


@pytest.mark.parametrize(
    ("source_factory", "target_factory"),
    [
        (_legacy_residual, _content_gated_residual),
        (_content_gated_residual, _legacy_residual),
    ],
)
def test_checkpoint_rejects_legacy_content_gated_cross_load(
    tmp_path,
    source_factory,
    target_factory,
) -> None:
    checkpoint = save_adapter_checkpoint(
        tmp_path / f"cross_load_{source_factory.__name__}",
        {"global_scene_residual": source_factory()},
        {"schema_version": 3},
    )

    with pytest.raises(RuntimeError, match="Missing key|Unexpected key|missing key|unexpected key"):
        load_adapter_checkpoint(
            checkpoint,
            {"global_scene_residual": target_factory()},
            device="cpu",
        )
