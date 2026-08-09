from __future__ import annotations

import logging

import torch

LOGGER = logging.getLogger(__name__)


def select_device(prefer_mps: bool = True) -> torch.device:
    """Select MPS when built and operational, otherwise return CPU."""
    if prefer_mps and torch.backends.mps.is_built() and torch.backends.mps.is_available():
        try:
            probe = torch.ones(4, device="mps")
            if torch.isfinite((probe @ probe).cpu()).all():
                return torch.device("mps")
        except (RuntimeError, NotImplementedError) as exc:
            LOGGER.warning("MPS probe failed; using CPU: %s", exc)
    return torch.device("cpu")


def safe_dtype(device: torch.device, requested: str = "float16") -> torch.dtype:
    """Resolve a requested low-precision dtype only on validated backends.

    PyTorch 2.13 on this project's Apple-Silicon target supports MPS bfloat16;
    the real Gemma 4 smoke test exercises that path before it is selected in a
    config. CPU remains float32 because several transformer kernels still have
    incomplete or unexpectedly slow low-precision CPU implementations.
    """

    if requested not in {"float16", "bfloat16", "float32"}:
        raise ValueError(f"Unsupported requested dtype: {requested}")
    if requested == "float32":
        return torch.float32
    if requested == "float16" and device.type in {"mps", "cuda"}:
        return torch.float16
    if requested == "bfloat16" and device.type == "mps":
        return torch.bfloat16
    if (
        requested == "bfloat16"
        and device.type == "cuda"
        and torch.cuda.is_available()
        and torch.cuda.is_bf16_supported()
    ):
        return torch.bfloat16
    return torch.float32
