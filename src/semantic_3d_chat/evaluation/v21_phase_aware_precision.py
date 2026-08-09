"""Phase-aware 16-bit pair diagnostics for the V21 preflight.

This module deliberately has no dependency on the V20 controller.  It treats
the decoder-visible response for one scene as

``E_j = Q(B_j + D_j) - Q(B_j)``

and separates a pair response into the intended FP32 signal
``R = D_first - D_second``, the actual response
``E = E_first - E_second``, and the base-phase-dependent quantization term
``N = E - R``.

Two counterfactual controls make the otherwise hidden quantization-phase dependence
observable:

* shared-base responses apply both deltas to the same base tensor;
* the common-delta null applies one identical delta to both original bases.

The quantized dtype is an explicit input and must be either FP16 or BF16.  All
reductions run in float64 on CPU so the diagnostics are deterministic on MPS
while the 16-bit casts themselves remain on the input device.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch

from semantic_3d_chat.language.lora import tensor_state_sha256

PHASE_AWARE_PRECISION_PAIR_V1 = "phase_aware_precision_pair_v1"
SUPPORTED_MODEL_DTYPES = frozenset({torch.float16, torch.bfloat16})


def model_dtype_label(model_dtype: torch.dtype) -> str:
    """Return the stable report label for one explicitly supported model dtype."""

    if not isinstance(model_dtype, torch.dtype):
        raise TypeError("model_dtype must be a torch.dtype")
    if model_dtype not in SUPPORTED_MODEL_DTYPES:
        raise ValueError("model_dtype must be torch.float16 or torch.bfloat16")
    return str(model_dtype).removeprefix("torch.")


def _validate_tensor(value: Any, name: str) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if value.dtype is not torch.float32:
        raise ValueError(f"{name} must use float32 before the 16-bit model boundary")
    if value.numel() == 0:
        raise ValueError(f"{name} must be nonempty")
    if not bool(torch.isfinite(value).all().detach().cpu()):
        raise ValueError(f"{name} must contain only finite values")
    return value


def _validate_inputs(values: Mapping[str, Any]) -> dict[str, torch.Tensor]:
    tensors = {name: _validate_tensor(value, name) for name, value in values.items()}
    first = next(iter(tensors.values()))
    for name, value in tensors.items():
        if value.shape != first.shape:
            raise ValueError(
                f"All phase-aware tensors must have the same shape; "
                f"{name} has {tuple(value.shape)} and expected {tuple(first.shape)}"
            )
        if value.device != first.device:
            raise ValueError(
                f"All phase-aware tensors must share one device; "
                f"{name} is on {value.device} and expected {first.device}"
            )
    return tensors


def _rms(value: torch.Tensor) -> float:
    flattened = value.reshape(-1)
    return float(flattened.square().mean().sqrt())


def _optional_ratio(numerator: float, denominator: float) -> float | None:
    return None if denominator == 0.0 else numerator / denominator


def exact_effective_delta(
    base: torch.Tensor,
    raw_delta: torch.Tensor,
    *,
    model_dtype: torch.dtype,
) -> torch.Tensor:
    """Return the exact decoder-visible 16-bit response as detached FP32.

    The addition happens in FP32 and only the completed token tensor is cast.
    This matches the scene-prefix boundary; pre-casting either operand would be
    a different computation.
    """

    model_dtype_label(model_dtype)
    tensors = _validate_inputs({"base": base, "raw_delta": raw_delta})
    base = tensors["base"].detach()
    raw_delta = tensors["raw_delta"].detach()
    with torch.no_grad():
        base_model = base.to(dtype=model_dtype)
        adapted_model = (base + raw_delta).to(dtype=model_dtype)
        effective = adapted_model.float() - base_model.float()
    if not bool(torch.isfinite(effective).all().detach().cpu()):
        raise ValueError("16-bit effective delta contains NaN or infinity")
    return effective


def pair_signal_decomposition(
    raw_pair_delta: torch.Tensor,
    effective_pair_delta: torch.Tensor,
) -> dict[str, Any]:
    """Decompose an effective pair response into aligned and phase-noise terms.

    ``aligned_gain`` is the signed least-squares gain from ``R`` to ``E``.
    ``orthogonal_quantization_rms`` is the component of ``N = E - R``
    orthogonal to ``R``.  Alignment values are ``None`` when ``R`` is exactly
    zero because no signal direction exists.
    """

    tensors = _validate_inputs(
        {
            "raw_pair_delta": raw_pair_delta,
            "effective_pair_delta": effective_pair_delta,
        }
    )
    raw_tensor = tensors["raw_pair_delta"].detach().float()
    effective_tensor = tensors["effective_pair_delta"].detach().float()
    noise_tensor = effective_tensor - raw_tensor

    raw = raw_tensor.cpu().double().reshape(-1)
    effective = effective_tensor.cpu().double().reshape(-1)
    noise = effective - raw
    raw_rms = _rms(raw)
    effective_rms = _rms(effective)
    noise_rms = _rms(noise)
    raw_norm = float(raw.norm())
    effective_norm = float(effective.norm())

    cosine: float | None = None
    aligned_gain: float | None = None
    aligned_effective_rms: float | None = None
    parallel_noise_gain: float | None = None
    parallel_noise_rms: float | None = None
    orthogonal_noise_rms: float | None = None
    orthogonal_noise_to_raw: float | None = None
    orthogonal_noise_fraction: float | None = None
    orthogonality_absolute_dot: float | None = None
    noise_energy_closure_error: float | None = None

    if raw_norm > 0.0:
        raw_energy = float(torch.dot(raw, raw))
        aligned_gain = float(torch.dot(effective, raw)) / raw_energy
        parallel_noise_gain = aligned_gain - 1.0
        parallel_noise = parallel_noise_gain * raw
        orthogonal_noise = noise - parallel_noise
        aligned_effective_rms = aligned_gain * raw_rms
        parallel_noise_rms = _rms(parallel_noise)
        orthogonal_noise_rms = _rms(orthogonal_noise)
        orthogonal_noise_to_raw = orthogonal_noise_rms / raw_rms
        orthogonal_noise_fraction = _optional_ratio(orthogonal_noise_rms, noise_rms)
        orthogonality_absolute_dot = abs(float(torch.dot(orthogonal_noise, raw)))
        noise_energy_closure_error = abs(
            float(torch.dot(noise, noise))
            - float(torch.dot(parallel_noise, parallel_noise))
            - float(torch.dot(orthogonal_noise, orthogonal_noise))
        )
        if effective_norm > 0.0:
            cosine = float(torch.dot(raw, effective)) / (raw_norm * effective_norm)

    closure = effective - (raw + noise)
    return {
        "schema_version": 1,
        "shape": list(raw_tensor.shape),
        "element_count": raw_tensor.numel(),
        "raw_pair_exact_zero": raw_norm == 0.0,
        "effective_pair_exact_zero": effective_norm == 0.0,
        "raw_pair_rms": raw_rms,
        "effective_pair_rms": effective_rms,
        "quantization_pair_error_rms": noise_rms,
        "effective_to_raw_rms_ratio": _optional_ratio(effective_rms, raw_rms),
        "quantization_error_to_raw_rms_ratio": _optional_ratio(noise_rms, raw_rms),
        "raw_effective_cosine": cosine,
        "aligned_gain": aligned_gain,
        "aligned_effective_rms": aligned_effective_rms,
        "parallel_quantization_gain_bias": parallel_noise_gain,
        "parallel_quantization_rms": parallel_noise_rms,
        "orthogonal_quantization_rms": orthogonal_noise_rms,
        "orthogonal_quantization_to_raw_rms_ratio": orthogonal_noise_to_raw,
        "orthogonal_quantization_fraction_of_total_error": orthogonal_noise_fraction,
        "orthogonality_absolute_dot": orthogonality_absolute_dot,
        "noise_energy_closure_absolute_error": noise_energy_closure_error,
        "decomposition_closure_absolute_maximum": float(closure.abs().max()),
        "raw_pair_delta_sha256": tensor_state_sha256({"raw_pair_delta": raw_tensor}),
        "effective_pair_delta_sha256": tensor_state_sha256(
            {"effective_pair_delta": effective_tensor}
        ),
        "quantization_pair_error_sha256": tensor_state_sha256(
            {"quantization_pair_error": noise_tensor}
        ),
    }


def shared_base_pair_response(
    base: torch.Tensor,
    raw_delta_first: torch.Tensor,
    raw_delta_second: torch.Tensor,
    *,
    model_dtype: torch.dtype,
) -> torch.Tensor:
    """Apply both scene deltas to one common 16-bit base and subtract responses."""

    tensors = _validate_inputs(
        {
            "base": base,
            "raw_delta_first": raw_delta_first,
            "raw_delta_second": raw_delta_second,
        }
    )
    first = exact_effective_delta(
        tensors["base"], tensors["raw_delta_first"], model_dtype=model_dtype
    )
    second = exact_effective_delta(
        tensors["base"], tensors["raw_delta_second"], model_dtype=model_dtype
    )
    return first - second


def common_delta_phase_null(
    base_first: torch.Tensor,
    base_second: torch.Tensor,
    common_delta: torch.Tensor,
    *,
    model_dtype: torch.dtype,
) -> torch.Tensor:
    """Measure response difference caused solely by distinct base quantization phases."""

    tensors = _validate_inputs(
        {
            "base_first": base_first,
            "base_second": base_second,
            "common_delta": common_delta,
        }
    )
    first = exact_effective_delta(
        tensors["base_first"], tensors["common_delta"], model_dtype=model_dtype
    )
    second = exact_effective_delta(
        tensors["base_second"], tensors["common_delta"], model_dtype=model_dtype
    )
    return first - second


def phase_aware_pair_diagnostics(
    base_first: torch.Tensor,
    base_second: torch.Tensor,
    raw_delta_first: torch.Tensor,
    raw_delta_second: torch.Tensor,
    *,
    model_dtype: torch.dtype,
) -> dict[str, Any]:
    """Build complete JSON-compatible phase-aware evidence for one scene pair."""

    dtype_label = model_dtype_label(model_dtype)
    tensors = _validate_inputs(
        {
            "base_first": base_first,
            "base_second": base_second,
            "raw_delta_first": raw_delta_first,
            "raw_delta_second": raw_delta_second,
        }
    )
    base_first = tensors["base_first"]
    base_second = tensors["base_second"]
    raw_delta_first = tensors["raw_delta_first"]
    raw_delta_second = tensors["raw_delta_second"]

    raw_pair = raw_delta_first.detach() - raw_delta_second.detach()
    effective_first = exact_effective_delta(base_first, raw_delta_first, model_dtype=model_dtype)
    effective_second = exact_effective_delta(base_second, raw_delta_second, model_dtype=model_dtype)
    effective_pair = effective_first - effective_second
    actual = pair_signal_decomposition(raw_pair, effective_pair)

    shared_on_first = shared_base_pair_response(
        base_first,
        raw_delta_first,
        raw_delta_second,
        model_dtype=model_dtype,
    )
    shared_on_second = shared_base_pair_response(
        base_second,
        raw_delta_first,
        raw_delta_second,
        model_dtype=model_dtype,
    )
    shared_mean = (shared_on_first + shared_on_second) * 0.5
    shared_phase_spread = shared_on_first - shared_on_second
    shared_phase_spread_rms = _rms(shared_phase_spread.detach().float().cpu().double())

    common_delta = (raw_delta_first.detach() + raw_delta_second.detach()) * 0.5
    common_null = common_delta_phase_null(
        base_first,
        base_second,
        common_delta,
        model_dtype=model_dtype,
    )
    common_null_rms = _rms(common_null.detach().float().cpu().double())
    raw_pair_rms = float(actual["raw_pair_rms"])
    effective_pair_rms = float(actual["effective_pair_rms"])

    return {
        "schema_version": 1,
        "algorithm_family": PHASE_AWARE_PRECISION_PAIR_V1,
        "algorithm": f"phase_aware_{dtype_label}_pair_v1",
        "model_dtype": dtype_label,
        "source_dtype": "float32",
        "comparison_dtype": "float64_cpu",
        "shape": list(base_first.shape),
        "element_count": base_first.numel(),
        "definitions": {
            "raw_pair_delta": "raw_delta_first_minus_raw_delta_second",
            "effective_scene_delta": "model_dtype(base_plus_raw_delta)_minus_model_dtype(base)",
            "effective_pair_delta": "effective_first_minus_effective_second",
            "quantization_pair_error": "effective_pair_delta_minus_raw_pair_delta",
            "common_delta": "arithmetic_mean_of_raw_scene_deltas",
        },
        "actual_pair": actual,
        "shared_base": {
            "first_base": pair_signal_decomposition(raw_pair, shared_on_first),
            "second_base": pair_signal_decomposition(raw_pair, shared_on_second),
            "mean_response": pair_signal_decomposition(raw_pair, shared_mean),
            "phase_spread_rms": shared_phase_spread_rms,
            "phase_spread_to_raw_pair_rms_ratio": _optional_ratio(
                shared_phase_spread_rms, raw_pair_rms
            ),
            "phase_spread_sha256": tensor_state_sha256(
                {"shared_base_phase_spread": shared_phase_spread}
            ),
        },
        "common_delta_null": {
            "raw_pair_delta_exact_zero_by_construction": True,
            "common_delta_rms": _rms(common_delta.detach().float().cpu().double()),
            "response_rms": common_null_rms,
            "response_to_raw_pair_rms_ratio": _optional_ratio(common_null_rms, raw_pair_rms),
            "response_to_actual_effective_pair_rms_ratio": _optional_ratio(
                common_null_rms, effective_pair_rms
            ),
            "response_sha256": tensor_state_sha256({"common_delta_phase_null": common_null}),
        },
        "tensor_hashes": {
            "base_first_sha256": tensor_state_sha256({"base_first": base_first}),
            "base_second_sha256": tensor_state_sha256({"base_second": base_second}),
            "raw_delta_first_sha256": tensor_state_sha256({"raw_delta_first": raw_delta_first}),
            "raw_delta_second_sha256": tensor_state_sha256({"raw_delta_second": raw_delta_second}),
            "effective_first_sha256": tensor_state_sha256({"effective_first": effective_first}),
            "effective_second_sha256": tensor_state_sha256({"effective_second": effective_second}),
        },
    }


__all__ = [
    "PHASE_AWARE_PRECISION_PAIR_V1",
    "SUPPORTED_MODEL_DTYPES",
    "common_delta_phase_null",
    "exact_effective_delta",
    "model_dtype_label",
    "pair_signal_decomposition",
    "phase_aware_pair_diagnostics",
    "shared_base_pair_response",
]
