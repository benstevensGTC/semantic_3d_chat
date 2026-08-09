from __future__ import annotations

import copy
import hashlib
from pathlib import Path
from typing import Any

import pytest
import torch
from safetensors.torch import save_file

from semantic_3d_chat.config import load_config
from semantic_3d_chat.evaluation import v21_update1_verifier as verifier
from semantic_3d_chat.evaluation.v19_optimizer_state import canonical_v19_adamw_state
from semantic_3d_chat.evaluation.v21_phase_aware_precision import (
    phase_aware_pair_diagnostics,
)
from semantic_3d_chat.evaluation.v21_predicted_update_audit import (
    evaluate_v21_predicted_update,
)
from semantic_3d_chat.language.lora import tensor_state_sha256
from semantic_3d_chat.scene_encoder.signed_x_local_field import (
    SignedXLocalFieldSceneResidual,
)

CONFIG_PATH = "configs/experiments/gemma4_color_mirror_signed_x_local_field_phase_aware_v21.yaml"


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _precision_row() -> dict[str, Any]:
    count = 256 * 1536
    return {
        "schema_version": 1,
        "algorithm": "bfloat16_cast_of_fp32_base_plus_fp32_delta",
        "base_source_dtype": "float32",
        "model_dtype": "bfloat16",
        "comparison_dtype": "float64",
        "element_count": count,
        "changed_element_count": 8,
        "changed_element_fraction": 8 / count,
        "raw_delta_rms": 0.005,
        "effective_delta_rms": 0.006,
        "effective_to_raw_rms_ratio": 1.2,
        "quantization_error_rms": 0.001,
        "quantization_error_to_raw_rms_ratio": 0.2,
        "raw_effective_cosine": 0.9,
        "raw_delta_sha256": _digest("raw"),
        "effective_delta_sha256": _digest("effective"),
    }


@pytest.mark.parametrize(
    "case",
    ["dtype", "algorithm", "fraction", "ratio", "hash", "extra"],
)
def test_precision_cast_audit_is_exactly_bfloat16_and_fail_closed(case: str) -> None:
    row = _precision_row()
    if case == "dtype":
        row["model_dtype"] = "float16"
    elif case == "algorithm":
        row["algorithm"] = "float16_cast_of_fp32_base_plus_fp32_delta"
    elif case == "fraction":
        row["changed_element_fraction"] = 0.5
    elif case == "ratio":
        row["effective_to_raw_rms_ratio"] = 2.0
    elif case == "hash":
        row["raw_delta_sha256"] = "not-a-hash"
    else:
        row["unexpected"] = True
    with pytest.raises(verifier.V21Update1Violation):
        verifier._validate_precision_row(row, "scene_000003")


def test_valid_bfloat16_precision_cast_audit_is_accepted() -> None:
    assert (
        verifier._validate_precision_row(_precision_row(), "scene_000003")["model_dtype"]
        == "bfloat16"
    )


def _decomposition(
    tag: str,
    *,
    raw_rms: float,
    aligned_gain: float,
    orthogonal_rms: float,
) -> dict[str, Any]:
    aligned_rms = aligned_gain * raw_rms
    effective_rms = (aligned_rms**2 + orthogonal_rms**2) ** 0.5
    parallel_gain = aligned_gain - 1.0
    parallel_rms = abs(parallel_gain) * raw_rms
    error_rms = (parallel_rms**2 + orthogonal_rms**2) ** 0.5
    return {
        "schema_version": 1,
        "shape": [1, 256, 1536],
        "element_count": 256 * 1536,
        "raw_pair_exact_zero": False,
        "effective_pair_exact_zero": False,
        "raw_pair_rms": raw_rms,
        "effective_pair_rms": effective_rms,
        "quantization_pair_error_rms": error_rms,
        "effective_to_raw_rms_ratio": effective_rms / raw_rms,
        "quantization_error_to_raw_rms_ratio": error_rms / raw_rms,
        "raw_effective_cosine": aligned_rms / effective_rms,
        "aligned_gain": aligned_gain,
        "aligned_effective_rms": aligned_rms,
        "parallel_quantization_gain_bias": parallel_gain,
        "parallel_quantization_rms": parallel_rms,
        "orthogonal_quantization_rms": orthogonal_rms,
        "orthogonal_quantization_to_raw_rms_ratio": orthogonal_rms / raw_rms,
        "orthogonal_quantization_fraction_of_total_error": orthogonal_rms / error_rms,
        "orthogonality_absolute_dot": 0.0,
        "noise_energy_closure_absolute_error": 0.0,
        "decomposition_closure_absolute_maximum": 0.0,
        "raw_pair_delta_sha256": _digest(f"{tag}-raw"),
        "effective_pair_delta_sha256": _digest(f"{tag}-effective"),
        "quantization_pair_error_sha256": _digest(f"{tag}-error"),
    }


def _phase_row(tag: str, *, mirror: bool) -> dict[str, Any]:
    raw_rms = 0.01 if mirror else 0.005
    gain = 0.8 if mirror else 1.0
    orthogonal = 0.002 if mirror else 0.001
    actual = _decomposition(
        f"{tag}-actual",
        raw_rms=raw_rms,
        aligned_gain=gain,
        orthogonal_rms=orthogonal,
    )
    phase_spread = 0.0001
    null_response = 0.0002
    shared_decompositions = {
        "first_base": _decomposition(
            f"{tag}-first", raw_rms=raw_rms, aligned_gain=gain, orthogonal_rms=orthogonal
        ),
        "second_base": _decomposition(
            f"{tag}-second", raw_rms=raw_rms, aligned_gain=gain, orthogonal_rms=orthogonal
        ),
        "mean_response": _decomposition(
            f"{tag}-mean", raw_rms=raw_rms, aligned_gain=gain, orthogonal_rms=orthogonal
        ),
    }
    for decomposition in shared_decompositions.values():
        decomposition["raw_pair_delta_sha256"] = actual["raw_pair_delta_sha256"]
    return {
        "schema_version": 1,
        "algorithm_family": "phase_aware_precision_pair_v1",
        "algorithm": "phase_aware_bfloat16_pair_v1",
        "model_dtype": "bfloat16",
        "source_dtype": "float32",
        "comparison_dtype": "float64_cpu",
        "shape": [1, 256, 1536],
        "element_count": 256 * 1536,
        "definitions": {
            "raw_pair_delta": "raw_delta_first_minus_raw_delta_second",
            "effective_scene_delta": "model_dtype(base_plus_raw_delta)_minus_model_dtype(base)",
            "effective_pair_delta": "effective_first_minus_effective_second",
            "quantization_pair_error": "effective_pair_delta_minus_raw_pair_delta",
            "common_delta": "arithmetic_mean_of_raw_scene_deltas",
        },
        "actual_pair": actual,
        "shared_base": {
            **shared_decompositions,
            "phase_spread_rms": phase_spread,
            "phase_spread_to_raw_pair_rms_ratio": phase_spread / raw_rms,
            "phase_spread_sha256": _digest(f"{tag}-spread"),
        },
        "common_delta_null": {
            "raw_pair_delta_exact_zero_by_construction": True,
            "common_delta_rms": 0.003,
            "response_rms": null_response,
            "response_to_raw_pair_rms_ratio": null_response / raw_rms,
            "response_to_actual_effective_pair_rms_ratio": (
                null_response / actual["effective_pair_rms"]
            ),
            "response_sha256": _digest(f"{tag}-null"),
        },
        "tensor_hashes": {
            key: _digest(f"{tag}-{key}")
            for key in (
                "base_first_sha256",
                "base_second_sha256",
                "raw_delta_first_sha256",
                "raw_delta_second_sha256",
                "effective_first_sha256",
                "effective_second_sha256",
            )
        },
    }


def _phase_evidence() -> dict[str, Any]:
    return {
        "pair_000001": _phase_row("color", mirror=False),
        "pair_000003": _phase_row("mirror", mirror=True),
    }


def test_phase_evidence_binds_complete_bfloat16_helper_schema() -> None:
    evidence = _phase_evidence()
    assert verifier._validate_phase_evidence(evidence) == evidence


def test_phase_validator_accepts_exact_bfloat16_helper_output() -> None:
    generator = torch.Generator().manual_seed(21)
    shape = (1, 256, 1536)
    base_first = torch.randn(shape, generator=generator, dtype=torch.float32) * 0.2
    base_second = torch.randn(shape, generator=generator, dtype=torch.float32) * 0.2
    delta_first = torch.randn(shape, generator=generator, dtype=torch.float32) * 0.005
    delta_second = torch.randn(shape, generator=generator, dtype=torch.float32) * 0.005
    row = phase_aware_pair_diagnostics(
        base_first,
        base_second,
        delta_first,
        delta_second,
        model_dtype=torch.bfloat16,
    )
    evidence = {"pair_000001": row, "pair_000003": copy.deepcopy(row)}
    assert verifier._validate_phase_evidence(evidence) == evidence


@pytest.mark.parametrize(
    "case",
    [
        "dtype",
        "algorithm",
        "missing_pair",
        "extra_actual",
        "aligned_math",
        "cosine_math",
        "parallel_rms_math",
        "orthogonal_fraction_math",
        "effective_energy_math",
        "error_energy_math",
        "orthogonality",
        "noise_closure",
        "decomposition_closure",
        "phase_spread_ratio",
        "null_raw_ratio",
        "null_effective_ratio",
        "bad_hash",
    ],
)
def test_phase_evidence_tamper_is_rejected(case: str) -> None:
    evidence = _phase_evidence()
    mirror = evidence["pair_000003"]
    if case == "dtype":
        mirror["model_dtype"] = "float16"
    elif case == "algorithm":
        mirror["algorithm"] = "phase_aware_float16_pair_v1"
    elif case == "missing_pair":
        evidence.pop("pair_000001")
    elif case == "extra_actual":
        mirror["actual_pair"]["unexpected"] = 1.0
    elif case == "aligned_math":
        mirror["actual_pair"]["aligned_effective_rms"] *= 2.0
    elif case == "cosine_math":
        mirror["actual_pair"]["raw_effective_cosine"] = 1.0
    elif case == "parallel_rms_math":
        mirror["actual_pair"]["parallel_quantization_rms"] *= 2.0
    elif case == "orthogonal_fraction_math":
        mirror["actual_pair"]["orthogonal_quantization_fraction_of_total_error"] *= 2.0
    elif case == "effective_energy_math":
        actual = mirror["actual_pair"]
        actual["effective_pair_rms"] *= 2.0
        actual["effective_to_raw_rms_ratio"] = actual["effective_pair_rms"] / actual["raw_pair_rms"]
        actual["raw_effective_cosine"] = (
            actual["aligned_effective_rms"] / actual["effective_pair_rms"]
        )
    elif case == "error_energy_math":
        actual = mirror["actual_pair"]
        actual["quantization_pair_error_rms"] *= 2.0
        actual["quantization_error_to_raw_rms_ratio"] = (
            actual["quantization_pair_error_rms"] / actual["raw_pair_rms"]
        )
        actual["orthogonal_quantization_fraction_of_total_error"] = (
            actual["orthogonal_quantization_rms"] / actual["quantization_pair_error_rms"]
        )
    elif case == "orthogonality":
        mirror["actual_pair"]["orthogonality_absolute_dot"] = 1.0e6
    elif case == "noise_closure":
        mirror["actual_pair"]["noise_energy_closure_absolute_error"] = 1.0e6
    elif case == "decomposition_closure":
        mirror["actual_pair"]["decomposition_closure_absolute_maximum"] = 1.0
    elif case == "phase_spread_ratio":
        mirror["shared_base"]["phase_spread_to_raw_pair_rms_ratio"] = 999.0
    elif case == "null_raw_ratio":
        mirror["common_delta_null"]["response_to_raw_pair_rms_ratio"] = 999.0
    elif case == "null_effective_ratio":
        mirror["common_delta_null"]["response_to_actual_effective_pair_rms_ratio"] = 999.0
    else:
        mirror["tensor_hashes"]["base_first_sha256"] = "invalid"
    with pytest.raises(verifier.V21Update1Violation):
        verifier._validate_phase_evidence(evidence)


def _functional_measurements(mirror_margin: float) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for pair_id, scenes in {
        "pair_000001": ("scene_000003", "scene_000004"),
        "pair_000003": ("scene_000007", "scene_000008"),
    }.items():
        margin = 0.5 if pair_id == "pair_000001" else mirror_margin
        for index in range(6):
            result.append(
                {
                    "pair_id": pair_id,
                    "question_key": f"q{index}",
                    "sides": [
                        {
                            "scene_id": scene_id,
                            "candidate_margin": margin,
                            "full_vocab_margin": margin,
                        }
                        for scene_id in scenes
                    ],
                }
            )
    return result


def _functional_audit() -> dict[str, Any]:
    policies = {
        "pair_000001": {
            "candidate_target_margin": 0.25,
            "candidate_hinge_weight": 8.0,
            "full_vocab_target_margin": 0.25,
            "full_vocab_hinge_weight": 2.0,
        },
        "pair_000003": {
            "candidate_target_margin": 1.0,
            "candidate_hinge_weight": 8.0,
            "full_vocab_target_margin": 1.0,
            "full_vocab_hinge_weight": 2.0,
        },
    }
    return evaluate_v21_predicted_update(
        _functional_measurements(-0.5),
        _functional_measurements(-0.25),
        policies=policies,
        color_pair_id="pair_000001",
        mirror_pair_id="pair_000003",
    )


def test_predicted_functional_audit_is_recomputed_from_measurements() -> None:
    audit = _functional_audit()
    contract = {
        "predicted_update_requires": {
            "expected_units_per_pair": 6,
            "expected_sides_per_pair": 12,
        }
    }
    assert verifier._validate_functional_audit(audit, contract) == audit
    tampered = copy.deepcopy(audit)
    tampered["summaries"]["after"]["pair_000003"]["weighted_margin_hinge_objective"] = 0.0
    with pytest.raises(verifier.V21Update1Violation, match="recomputation"):
        verifier._validate_functional_audit(tampered, contract)


def test_implementation_sources_bind_phase_and_functional_helpers() -> None:
    preflight: dict[str, Any] = {}
    for field, relative in verifier._IMPLEMENTATION_SOURCES.items():
        path = verifier.PROJECT_ROOT / relative
        preflight[field] = relative
        preflight[f"{field}_sha256"] = verifier.file_sha256(path)
    observed = verifier._validate_implementation_sources(preflight)
    assert observed["phase_audit_implementation_source"].endswith("v21_phase_aware_precision.py")
    assert observed["functional_audit_implementation_source"].endswith(
        "v21_predicted_update_audit.py"
    )


def _optimizer(contract: dict[str, Any], parameter: torch.nn.Parameter) -> torch.optim.AdamW:
    return torch.optim.AdamW(
        [
            {
                "name": "signed_x_output_projection",
                "params": [parameter],
                "lr": float(contract["learning_rate"]),
                "weight_decay": float(contract["weight_decay"]),
            }
        ],
        betas=tuple(contract["betas"]),
        eps=float(contract["epsilon"]),
        foreach=False,
        fused=False,
        capturable=False,
        maximize=False,
        amsgrad=False,
    )


def test_optimizer_is_deserialized_weights_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    contract = load_config(CONFIG_PATH)["training"]["optimizer"]
    parameter = torch.nn.Parameter(torch.zeros(1536, 128, dtype=torch.float32))
    parameter.grad = torch.ones_like(parameter)
    optimizer = _optimizer(contract, parameter)
    optimizer.step()
    manifest, digest = canonical_v19_adamw_state(optimizer.state_dict(), contract)
    path = tmp_path / "optimizer.pt"
    torch.save(optimizer.state_dict(), path)
    real_load = torch.load
    calls: list[dict[str, Any]] = []

    def recording_load(*args: Any, **kwargs: Any) -> Any:
        calls.append(dict(kwargs))
        return real_load(*args, **kwargs)

    monkeypatch.setattr(torch, "load", recording_load)
    observed = verifier._load_optimizer_evidence(
        path,
        contract=contract,
        expected_manifest=manifest,
        expected_hash=digest,
    )
    assert observed["sha256"] == digest
    assert calls == [{"weights_only": True, "map_location": "cpu"}]


def test_tensor_evidence_binds_bfloat16_runtime_frozen_state(tmp_path: Path) -> None:
    config = load_config(CONFIG_PATH)
    module = SignedXLocalFieldSceneResidual(scene_dim=1536, latent_count=256, content_dim=128)
    with torch.no_grad():
        module.output_projection.weight[0, 0] = 0.125
    signed = {f"signed_x_scene_residual.{key}": value for key, value in module.state_dict().items()}
    global_state = {"global_scene_residual.synthetic": torch.tensor([0.25])}
    scene_state = {
        "scene_model.synthetic": torch.tensor([0.5]),
        "composer.synthetic": torch.tensor([0.75], dtype=torch.bfloat16),
        "grounding.synthetic": torch.tensor([1.0]),
    }
    lora_state = {
        "lora_banks.extension_v13.synthetic": torch.tensor([1.25]),
        "lora_banks.inherited_v12.synthetic": torch.tensor([1.5]),
    }
    path = tmp_path / "adapter.safetensors"
    save_file({**signed, **global_state, **scene_state, **lora_state}, path)
    hashes = {
        "signed": tensor_state_sha256(signed),
        "global": tensor_state_sha256(global_state),
        "scene": tensor_state_sha256(scene_state),
    }
    lora_hashes = {
        bank: tensor_state_sha256({"synthetic": lora_state[f"lora_banks.{bank}.synthetic"]})
        for bank in ("extension_v13", "inherited_v12")
    }
    metadata = {
        "signed_x_scene_residual_state_sha256": hashes["signed"],
        "global_scene_residual_state_sha256": hashes["global"],
        "frozen_global_scene_residual_state_sha256": hashes["global"],
        "frozen_scene_state_sha256": hashes["scene"],
        "frozen_lora_bank_state_sha256": lora_hashes,
        "lora_bank_state_sha256": lora_hashes,
    }
    observed = verifier._load_tensor_evidence(
        path,
        metadata,
        config=config,
        expected_scene=hashes["scene"],
        expected_global=hashes["global"],
        expected_lora=lora_hashes,
    )
    assert observed["scene_state_sha256"] == hashes["scene"]


def test_authorization_schema_uses_phase_gate_and_rejects_v20_gate_name() -> None:
    assert "local_field_rank_precision_phase_gate" in verifier._AUTHORIZATION_CHECKS
    assert "local_field_rank_bf16_selectivity_gate" not in verifier._AUTHORIZATION_CHECKS
    assert "local_field_rank_precision_selectivity_gate" not in verifier._AUTHORIZATION_CHECKS
