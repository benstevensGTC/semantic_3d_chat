from __future__ import annotations

import pytest
import torch
from torch import nn

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
