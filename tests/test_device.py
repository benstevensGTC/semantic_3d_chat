from __future__ import annotations

import pytest
import torch

from semantic_3d_chat.device import safe_dtype


def test_safe_dtype_uses_validated_mps_precisions() -> None:
    device = torch.device("mps")
    assert safe_dtype(device, "float16") is torch.float16
    assert safe_dtype(device, "bfloat16") is torch.bfloat16
    assert safe_dtype(device, "float32") is torch.float32


def test_safe_dtype_keeps_cpu_numerics_in_float32() -> None:
    device = torch.device("cpu")
    assert safe_dtype(device, "float16") is torch.float32
    assert safe_dtype(device, "bfloat16") is torch.float32


def test_safe_dtype_rejects_unknown_name() -> None:
    with pytest.raises(ValueError, match="Unsupported requested dtype"):
        safe_dtype(torch.device("cpu"), "int8")
