from __future__ import annotations

import json
import math

import pytest
import torch

from semantic_3d_chat.evaluation.v21_phase_aware_precision import (
    PHASE_AWARE_PRECISION_PAIR_V1,
    common_delta_phase_null,
    exact_effective_delta,
    model_dtype_label,
    pair_signal_decomposition,
    phase_aware_pair_diagnostics,
    shared_base_pair_response,
)


@pytest.mark.parametrize("model_dtype", [torch.float16, torch.bfloat16])
def test_exact_effective_delta_matches_post_addition_model_boundary(
    model_dtype: torch.dtype,
) -> None:
    base = torch.tensor([[0.5009, 0.5010, -0.5009, -0.5010]], dtype=torch.float32)
    delta = torch.full_like(base, 1.0e-3)

    observed = exact_effective_delta(base, delta, model_dtype=model_dtype)
    direct = (base + delta).to(model_dtype).float() - base.to(model_dtype).float()
    incorrectly_precast = (base.to(model_dtype) + delta.to(model_dtype)).float() - base.to(
        model_dtype
    ).float()

    assert torch.equal(observed, direct)
    if model_dtype is torch.bfloat16:
        assert not torch.equal(observed, incorrectly_precast)
    assert observed.dtype is torch.float32
    assert observed.requires_grad is False


def test_pair_decomposition_recovers_parallel_gain_and_orthogonal_noise() -> None:
    raw = torch.tensor([1.0, -1.0, 0.0, 0.0], dtype=torch.float32)
    orthogonal = torch.tensor([1.0, 1.0, -1.0, -1.0], dtype=torch.float32)
    effective = 1.5 * raw + orthogonal

    report = pair_signal_decomposition(raw, effective)

    assert report["raw_pair_exact_zero"] is False
    assert report["aligned_gain"] == pytest.approx(1.5)
    assert report["parallel_quantization_gain_bias"] == pytest.approx(0.5)
    assert report["aligned_effective_rms"] == pytest.approx(1.5 * report["raw_pair_rms"])
    assert report["orthogonal_quantization_rms"] == pytest.approx(1.0)
    assert report["orthogonality_absolute_dot"] == pytest.approx(0.0, abs=1.0e-12)
    assert report["noise_energy_closure_absolute_error"] == pytest.approx(0.0, abs=1.0e-12)
    assert report["decomposition_closure_absolute_maximum"] == 0.0
    assert len(report["quantization_pair_error_sha256"]) == 64
    json.dumps(report, allow_nan=False)


def test_common_delta_null_exposes_base_phase_with_zero_raw_pair_signal() -> None:
    base_first = torch.full((1, 64), 0.5009, dtype=torch.float32)
    base_second = torch.full((1, 64), 0.5010, dtype=torch.float32)
    common = torch.full_like(base_first, 1.0e-3)

    null = common_delta_phase_null(
        base_first,
        base_second,
        common,
        model_dtype=torch.bfloat16,
    )
    report = phase_aware_pair_diagnostics(
        base_first,
        base_second,
        common,
        common,
        model_dtype=torch.bfloat16,
    )

    assert torch.count_nonzero(null) == null.numel()
    assert report["actual_pair"]["raw_pair_exact_zero"] is True
    assert report["actual_pair"]["raw_pair_rms"] == 0.0
    assert report["actual_pair"]["effective_pair_rms"] > 0.0
    assert report["actual_pair"]["aligned_gain"] is None
    assert report["actual_pair"]["raw_effective_cosine"] is None
    assert report["common_delta_null"]["response_rms"] == pytest.approx(
        report["actual_pair"]["effective_pair_rms"]
    )
    assert report["common_delta_null"]["response_to_raw_pair_rms_ratio"] is None
    assert report["shared_base"]["first_base"]["effective_pair_exact_zero"] is True
    assert report["shared_base"]["second_base"]["effective_pair_exact_zero"] is True


def test_shared_base_control_exposes_cross_base_phase_sensitivity() -> None:
    base_first = torch.full((2, 32), 0.5009, dtype=torch.float32)
    base_second = torch.full((2, 32), 0.5010, dtype=torch.float32)
    first_delta = torch.full_like(base_first, 1.0e-3)
    second_delta = torch.full_like(base_first, 1.1e-3)

    actual_first = exact_effective_delta(base_first, first_delta, model_dtype=torch.bfloat16)
    actual_second = exact_effective_delta(base_second, second_delta, model_dtype=torch.bfloat16)
    shared_first = shared_base_pair_response(
        base_first,
        first_delta,
        second_delta,
        model_dtype=torch.bfloat16,
    )
    shared_second = shared_base_pair_response(
        base_second,
        first_delta,
        second_delta,
        model_dtype=torch.bfloat16,
    )
    report = phase_aware_pair_diagnostics(
        base_first,
        base_second,
        first_delta,
        second_delta,
        model_dtype=torch.bfloat16,
    )

    assert torch.count_nonzero(actual_first - actual_second) > 0
    assert torch.count_nonzero(shared_first) > 0
    assert torch.count_nonzero(shared_second) == 0
    assert report["actual_pair"]["quantization_error_to_raw_rms_ratio"] > 1.0
    assert report["common_delta_null"]["response_rms"] > 0.0
    assert report["shared_base"]["phase_spread_rms"] > 0.0
    assert report["algorithm_family"] == PHASE_AWARE_PRECISION_PAIR_V1
    assert report["algorithm"] == "phase_aware_bfloat16_pair_v1"
    assert report["model_dtype"] == "bfloat16"
    json.dumps(report, allow_nan=False)


def test_fp16_control_avoids_the_adversarial_bf16_base_phase_artifact() -> None:
    base_first = torch.full((2, 32), 0.5009, dtype=torch.float32)
    base_second = torch.full((2, 32), 0.5010, dtype=torch.float32)
    first_delta = torch.full_like(base_first, 1.0e-3)
    second_delta = torch.full_like(base_first, 1.1e-3)

    fp16 = phase_aware_pair_diagnostics(
        base_first,
        base_second,
        first_delta,
        second_delta,
        model_dtype=torch.float16,
    )
    bf16 = phase_aware_pair_diagnostics(
        base_first,
        base_second,
        first_delta,
        second_delta,
        model_dtype=torch.bfloat16,
    )

    assert fp16["model_dtype"] == "float16"
    assert fp16["algorithm"] == "phase_aware_float16_pair_v1"
    assert fp16["actual_pair"]["effective_pair_rms"] == 0.0
    assert fp16["common_delta_null"]["response_rms"] == 0.0
    assert bf16["actual_pair"]["effective_pair_rms"] > 0.0
    assert bf16["common_delta_null"]["response_rms"] > 0.0
    assert bf16["actual_pair"]["quantization_error_to_raw_rms_ratio"] > 1.0


@pytest.mark.parametrize("model_dtype", [torch.float16, torch.bfloat16])
def test_phase_aware_report_is_deterministic_and_hash_bound(
    model_dtype: torch.dtype,
) -> None:
    generator = torch.Generator().manual_seed(21021)
    base_first = torch.randn(3, 5, generator=generator, dtype=torch.float32)
    base_second = torch.randn(3, 5, generator=generator, dtype=torch.float32)
    first_delta = torch.randn(3, 5, generator=generator, dtype=torch.float32) * 2.0e-2
    second_delta = torch.randn(3, 5, generator=generator, dtype=torch.float32) * 2.0e-2

    first = phase_aware_pair_diagnostics(
        base_first,
        base_second,
        first_delta,
        second_delta,
        model_dtype=model_dtype,
    )
    second = phase_aware_pair_diagnostics(
        base_first.clone(),
        base_second.clone(),
        first_delta.clone(),
        second_delta.clone(),
        model_dtype=model_dtype,
    )

    assert first == second
    assert first["shape"] == [3, 5]
    assert first["element_count"] == 15
    assert first["comparison_dtype"] == "float64_cpu"
    assert first["algorithm"] == f"phase_aware_{model_dtype_label(model_dtype)}_pair_v1"
    assert first["model_dtype"] == model_dtype_label(model_dtype)
    assert all(
        math.isfinite(value)
        for value in (
            first["actual_pair"]["raw_pair_rms"],
            first["actual_pair"]["effective_pair_rms"],
            first["actual_pair"]["quantization_pair_error_rms"],
            first["common_delta_null"]["response_rms"],
        )
    )
    assert all(len(value) == 64 for value in first["tensor_hashes"].values())


@pytest.mark.parametrize(
    ("case", "error"),
    [
        ("dtype", ValueError),
        ("shape", ValueError),
        ("nonfinite", ValueError),
        ("empty", ValueError),
        ("type", TypeError),
    ],
)
def test_phase_aware_inputs_fail_closed(case: str, error: type[Exception]) -> None:
    base_first: object = torch.ones(2, 3, dtype=torch.float32)
    base_second = torch.ones(2, 3, dtype=torch.float32)
    first_delta = torch.full((2, 3), 1.0e-3, dtype=torch.float32)
    second_delta = torch.full((2, 3), 2.0e-3, dtype=torch.float32)
    if case == "dtype":
        base_first = torch.ones(2, 3, dtype=torch.bfloat16)
    elif case == "shape":
        base_second = torch.ones(2, 4, dtype=torch.float32)
    elif case == "nonfinite":
        first_delta[0, 0] = torch.nan
    elif case == "empty":
        second_delta = torch.empty(0, dtype=torch.float32)
    else:
        base_first = [[1.0]]

    with pytest.raises(error):
        phase_aware_pair_diagnostics(
            base_first,  # type: ignore[arg-type]
            base_second,
            first_delta,
            second_delta,
            model_dtype=torch.float16,
        )


@pytest.mark.parametrize(
    ("model_dtype", "error"),
    [
        (torch.float32, ValueError),
        ("float16", TypeError),
    ],
)
def test_model_dtype_must_be_an_explicit_supported_torch_dtype(
    model_dtype: object,
    error: type[Exception],
) -> None:
    base = torch.ones(2, 3, dtype=torch.float32)
    delta = torch.full_like(base, 1.0e-3)

    with pytest.raises(error):
        exact_effective_delta(
            base,
            delta,
            model_dtype=model_dtype,  # type: ignore[arg-type]
        )


@pytest.mark.parametrize("model_dtype", [torch.float16, torch.bfloat16])
@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS is unavailable")
def test_phase_aware_diagnostics_run_model_precision_on_mps_and_reduce_on_cpu(
    model_dtype: torch.dtype,
) -> None:
    base_first = torch.full((2, 8), 0.5009, dtype=torch.float32, device="mps")
    base_second = torch.full((2, 8), 0.5010, dtype=torch.float32, device="mps")
    first_delta = torch.full_like(base_first, 1.0e-3)
    second_delta = torch.full_like(base_first, 1.1e-3)

    report = phase_aware_pair_diagnostics(
        base_first,
        base_second,
        first_delta,
        second_delta,
        model_dtype=model_dtype,
    )

    assert report["comparison_dtype"] == "float64_cpu"
    assert report["model_dtype"] == model_dtype_label(model_dtype)
    json.dumps(report, allow_nan=False)
