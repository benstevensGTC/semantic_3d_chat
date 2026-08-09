from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any

import pytest
import torch
from safetensors.torch import save_file

from semantic_3d_chat.config import config_hash, load_config
from semantic_3d_chat.evaluation import v20_update1_verifier as verifier
from semantic_3d_chat.evaluation.v19_optimizer_state import canonical_v19_adamw_state
from semantic_3d_chat.language.lora import tensor_state_sha256
from semantic_3d_chat.scene_encoder.global_residual import global_scene_residual_settings
from semantic_3d_chat.scene_encoder.signed_x_dispatch import signed_x_scene_residual_settings
from semantic_3d_chat.scene_encoder.signed_x_local_field import SignedXLocalFieldSceneResidual

CONFIG_PATH = "configs/experiments/gemma4_color_mirror_signed_x_local_field_v20.yaml"


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _rich_preflight() -> tuple[dict[str, Any], dict[str, Any]]:
    scenes = ("scene_000003", "scene_000004", "scene_000007", "scene_000008")
    pairs = ("pair_000001", "pair_000003")
    requirements = {
        "maximum_per_scene_raw_and_effective_delta_to_core_rms_ratio": 0.01,
        "minimum_mirror_effective_residual_to_core_rms_ratio": 0.01,
        "minimum_mirror_to_color_normalized_effective_selectivity": 1.5,
        "minimum_local_hidden_spatial_rank": 2,
    }
    structural = {
        "architecture_version": "signed_x_local_field_v2",
        "architecture_marker": 2,
        "scene_dim": 1536,
        "latent_count": 256,
        "content_dim": 128,
        "parameter_count": 196_608,
        "accounted_slot_count": 256,
        "all_slots_accounted": True,
        "signed_x_anchor_mean": 0.0,
        "signed_x_anchor_rms": 1.0,
        "spatial_centering": "all_slots_fp32",
        "spatial_statistic": "centered_local_content_times_unit_rms_signed_x",
        "spatial_reduction": "none",
        "trainable_surface": "bias_free_output_projection_only",
    }
    dependence = {
        "schema_version": 1,
        "probe_shape": [128, 256, 128],
        "hidden_shape": [128, 256, 128],
        "probe_count": 128,
        "paired_centered_perturbations": True,
        "maximum_probe_spatial_mean_absolute": 0.0,
        "minimum_changed_slots_per_probe": 2,
        "maximum_changed_slots_per_probe": 2,
        "changed_slot_union_count": 256,
        "all_input_slots_exercised": True,
        "unperturbed_output_slots_exactly_unchanged": True,
        "exact_two_slot_local_support": True,
        "no_global_moment_broadcast": True,
        "hidden_sha256": _digest("local-hidden"),
    }
    ranks = {
        scene: {
            "schema_version": 1,
            "shape": [1, 256, 128],
            "relative_tolerance": 1.0e-5,
            "minimum_spatial_rank": 2,
            "batches": [
                {
                    "batch_index": 0,
                    "spatial_rank": 2,
                    "stable_rank": 1.25,
                    "maximum_singular_value": 10.0,
                    "rank_threshold": 1.0e-4,
                    "top_singular_values": [10.0, 5.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                }
            ],
        }
        for scene in scenes
    }
    centered = {
        scene: {
            "shape": [1, 256, 128],
            "finite": True,
            "across_slot_mean_absolute_maximum": 0.0,
            "local_hidden_rms": 0.5,
            "local_hidden_sha256": _digest(f"local-hidden-{scene}"),
            "sha256": _digest(f"centered-{scene}"),
            "local_hidden_spatial_rank": copy.deepcopy(ranks[scene]),
        }
        for scene in scenes
    }
    raw_scene: dict[str, Any] = {}
    effective_scene: dict[str, Any] = {}
    casts: dict[str, Any] = {}
    for scene in scenes:
        raw_hash = _digest(f"raw-{scene}")
        effective_hash = _digest(f"effective-{scene}")
        raw_scene[scene] = {
            "shape": [1, 256, 1536],
            "core_rms": 1.0,
            "delta_rms": 0.005,
            "delta_sha256": raw_hash,
            "delta_to_core_rms_ratio": 0.005,
            "total_energy": 1.0,
            "across_slot_mean_energy": 0.0,
            "slot_varying_energy": 1.0,
            "positive_finite_total_energy": True,
            "positive_finite_core_rms": True,
            "across_slot_mean_energy_fraction": 0.0,
            "slot_varying_energy_fraction": 1.0,
            "slot_mean_absolute_maximum": 0.0,
            "delta_absolute_maximum": 0.01,
            "energy_closure_absolute_error": 0.0,
            "dtype": "float32",
        }
        effective_scene[scene] = {
            "shape": [1, 256, 1536],
            "core_rms": 1.0,
            "delta_rms": 0.006,
            "delta_sha256": effective_hash,
            "delta_to_core_rms_ratio": 0.006,
            "total_energy": 1.0,
            "across_slot_mean_energy": 0.0,
            "slot_varying_energy": 1.0,
            "positive_finite_total_energy": True,
            "positive_finite_core_rms": True,
            "across_slot_mean_energy_fraction": 0.0,
            "slot_varying_energy_fraction": 1.0,
            "slot_mean_absolute_maximum": 0.0,
            "delta_absolute_maximum": 0.012,
            "energy_closure_absolute_error": 0.0,
            "dtype": "bfloat16_round_trip_float32_delta",
        }
        casts[scene] = {
            "schema_version": 1,
            "algorithm": "bfloat16_cast_of_fp32_base_plus_fp32_delta",
            "base_source_dtype": "float32",
            "model_dtype": "bfloat16",
            "comparison_dtype": "float64",
            "element_count": 256 * 1536,
            "changed_element_count": 5,
            "changed_element_fraction": 5 / (256 * 1536),
            "raw_delta_rms": 0.005,
            "effective_delta_rms": 0.006,
            "effective_to_raw_rms_ratio": 1.2,
            "quantization_error_rms": 0.001,
            "quantization_error_to_raw_rms_ratio": 0.2,
            "raw_effective_cosine": 0.9,
            "raw_delta_sha256": raw_hash,
            "effective_delta_sha256": effective_hash,
        }
    pair_scenes = {
        "pair_000001": ("scene_000003", "scene_000004"),
        "pair_000003": ("scene_000007", "scene_000008"),
    }
    raw_pair = {}
    for index, pair in enumerate(pairs):
        ratio = 0.01 if index == 0 else 0.02
        raw_pair[pair] = {
            "first_scene_id": pair_scenes[pair][0],
            "second_scene_id": pair_scenes[pair][1],
            "core_pair_difference_rms": 0.5,
            "residual_pair_difference_rms": 0.5 * ratio,
            "residual_to_core_pair_difference_ratio": ratio,
            "residual_core_difference_cosine": 0.1,
            "positive_finite_pair_delta": True,
            "positive_finite_core_difference": True,
        }
    effective_pair = copy.deepcopy(raw_pair)
    gate = verifier.evaluate_v20_structural_gate(
        raw_scene,
        effective_scene,
        raw_pair_metrics=raw_pair,
        effective_pair_metrics=effective_pair,
        bf16_audits=casts,
        structural_state=structural,
        local_dependence=dependence,
        local_hidden_ranks=ranks,
        requirements=requirements,
    )
    preflight = {
        "local_field_structural_state": structural,
        "signed_x_structural_state": copy.deepcopy(structural),
        "local_dependence": dependence,
        "local_hidden_spatial_rank": ranks,
        "centered_content": centered,
        "raw_fp32_centered_scene_delta": raw_scene,
        "bf16_effective_scene_delta": effective_scene,
        "effective_cast_scene_delta": copy.deepcopy(effective_scene),
        "bf16_cast_audit": casts,
        "raw_fp32_centered_pair_delta": raw_pair,
        "bf16_effective_pair_delta": effective_pair,
        "effective_cast_pair_delta": copy.deepcopy(effective_pair),
        "structural_gate": gate,
    }
    return preflight, {"structural_preflight_requires": requirements}


def test_rich_bf16_preflight_reduces_canonically() -> None:
    preflight, contract = _rich_preflight()
    first = verifier._validate_rich_evidence(preflight, contract)
    second = verifier._validate_rich_evidence(copy.deepcopy(preflight), copy.deepcopy(contract))
    assert first == second
    assert first["verified"] is True
    assert len(first["canonical_sha256"]) == 64
    assert first["scene_ids"] == [
        "scene_000003",
        "scene_000004",
        "scene_000007",
        "scene_000008",
    ]


@pytest.mark.parametrize(
    "case",
    [
        "algorithm",
        "alias",
        "rank",
        "gate",
        "empty_scene_checks",
        "empty_selectivity_checks",
        "ratio_above_limit",
    ],
)
def test_rich_preflight_tamper_is_rejected(case: str) -> None:
    preflight, contract = _rich_preflight()
    if case == "algorithm":
        preflight["bf16_cast_audit"]["scene_000003"]["algorithm"] = "wrong"
    elif case == "alias":
        preflight["effective_cast_scene_delta"]["scene_000003"]["delta_sha256"] = _digest("wrong")
    elif case == "rank":
        preflight["local_hidden_spatial_rank"]["scene_000003"]["minimum_spatial_rank"] = 1
    elif case == "gate":
        preflight["structural_gate"]["passed"] = False
    elif case == "empty_scene_checks":
        preflight["structural_gate"]["scene_checks"] = {}
    elif case == "empty_selectivity_checks":
        preflight["structural_gate"]["selectivity_checks"] = {}
    else:
        preflight["raw_fp32_centered_scene_delta"]["scene_000003"]["delta_to_core_rms_ratio"] = 1.0
    with pytest.raises(verifier.V20Update1Violation):
        verifier._validate_rich_evidence(preflight, contract)


@pytest.mark.parametrize(
    "case",
    [
        "structural_extra",
        "structural_bool_as_int",
        "structural_alias_bool_as_int",
        "dependence_extra",
        "dependence_string_bool",
        "rank_extra",
        "rank_bool_as_int",
        "centered_extra",
        "centered_nonfinite",
        "raw_negative_ratio",
        "raw_integer_ratio",
        "raw_fraction_above_one",
        "raw_bool_as_int",
        "effective_nonfinite",
        "effective_alias_bool_as_int",
        "cast_negative_ratio",
        "cast_fraction_above_one",
        "pair_negative_ratio",
        "pair_string_bool",
        "pair_membership_swapped",
        "gate_extra",
        "gate_string_bool",
        "gate_negative_selectivity",
        "requirement_bool_as_int",
    ],
)
def test_rich_preflight_exact_schema_types_and_ranges_are_fail_closed(case: str) -> None:
    preflight, contract = _rich_preflight()
    if case == "structural_extra":
        preflight["local_field_structural_state"]["unexpected"] = 1
        preflight["signed_x_structural_state"] = copy.deepcopy(
            preflight["local_field_structural_state"]
        )
    elif case == "structural_bool_as_int":
        preflight["local_field_structural_state"]["all_slots_accounted"] = 1
        preflight["signed_x_structural_state"] = copy.deepcopy(
            preflight["local_field_structural_state"]
        )
    elif case == "structural_alias_bool_as_int":
        preflight["signed_x_structural_state"]["all_slots_accounted"] = 1
    elif case == "dependence_extra":
        preflight["local_dependence"]["unexpected"] = 1
    elif case == "dependence_string_bool":
        preflight["local_dependence"]["all_input_slots_exercised"] = "true"
    elif case == "rank_extra":
        preflight["local_hidden_spatial_rank"]["scene_000003"]["unexpected"] = 1
        preflight["centered_content"]["scene_000003"]["local_hidden_spatial_rank"] = copy.deepcopy(
            preflight["local_hidden_spatial_rank"]["scene_000003"]
        )
    elif case == "rank_bool_as_int":
        preflight["local_hidden_spatial_rank"]["scene_000003"]["minimum_spatial_rank"] = True
        preflight["centered_content"]["scene_000003"]["local_hidden_spatial_rank"] = copy.deepcopy(
            preflight["local_hidden_spatial_rank"]["scene_000003"]
        )
    elif case == "centered_extra":
        preflight["centered_content"]["scene_000003"]["unexpected"] = 1
    elif case == "centered_nonfinite":
        preflight["centered_content"]["scene_000003"]["local_hidden_rms"] = float("nan")
    elif case == "raw_negative_ratio":
        preflight["raw_fp32_centered_scene_delta"]["scene_000003"]["delta_to_core_rms_ratio"] = -0.1
    elif case == "raw_integer_ratio":
        preflight["raw_fp32_centered_scene_delta"]["scene_000003"]["delta_to_core_rms_ratio"] = 0
    elif case == "raw_fraction_above_one":
        preflight["raw_fp32_centered_scene_delta"]["scene_000003"][
            "slot_varying_energy_fraction"
        ] = 1.1
    elif case == "raw_bool_as_int":
        preflight["raw_fp32_centered_scene_delta"]["scene_000003"][
            "positive_finite_total_energy"
        ] = 1
    elif case == "effective_nonfinite":
        preflight["bf16_effective_scene_delta"]["scene_000003"]["delta_rms"] = float("inf")
        preflight["effective_cast_scene_delta"] = copy.deepcopy(
            preflight["bf16_effective_scene_delta"]
        )
    elif case == "effective_alias_bool_as_int":
        preflight["effective_cast_scene_delta"]["scene_000003"]["positive_finite_total_energy"] = 1
    elif case == "cast_negative_ratio":
        preflight["bf16_cast_audit"]["scene_000003"]["quantization_error_to_raw_rms_ratio"] = -0.1
    elif case == "cast_fraction_above_one":
        preflight["bf16_cast_audit"]["scene_000003"]["changed_element_fraction"] = 1.1
    elif case == "pair_negative_ratio":
        preflight["raw_fp32_centered_pair_delta"]["pair_000001"][
            "residual_to_core_pair_difference_ratio"
        ] = -0.1
    elif case == "pair_string_bool":
        preflight["raw_fp32_centered_pair_delta"]["pair_000001"]["positive_finite_pair_delta"] = (
            "true"
        )
    elif case == "pair_membership_swapped":
        row = preflight["raw_fp32_centered_pair_delta"]["pair_000001"]
        row["first_scene_id"], row["second_scene_id"] = (
            row["second_scene_id"],
            row["first_scene_id"],
        )
    elif case == "gate_extra":
        preflight["structural_gate"]["unexpected"] = True
    elif case == "gate_string_bool":
        preflight["structural_gate"]["passed"] = "true"
    elif case == "gate_negative_selectivity":
        preflight["structural_gate"]["raw_pair_selectivity"][
            "mirror_to_color_normalized_selectivity"
        ] = -1.0
    else:
        contract["structural_preflight_requires"]["minimum_local_hidden_spatial_rank"] = True
    with pytest.raises(verifier.V20Update1Violation):
        verifier._validate_rich_evidence(preflight, contract)


def test_implementation_sources_are_bound_to_exact_canonical_modules(tmp_path: Path) -> None:
    preflight: dict[str, Any] = {}
    for field, relative in verifier._IMPLEMENTATION_SOURCES.items():
        path = verifier.PROJECT_ROOT / relative
        preflight[field] = relative
        preflight[f"{field}_sha256"] = verifier.file_sha256(path)
    verifier._validate_implementation_sources(preflight)

    copied = tmp_path / "v20_structural_preflight.py"
    copied.write_bytes(
        (
            verifier.PROJECT_ROOT / verifier._IMPLEMENTATION_SOURCES["implementation_source"]
        ).read_bytes()
    )
    preflight["implementation_source"] = str(copied)
    with pytest.raises(verifier.V20Update1Violation, match="implementation_source"):
        verifier._validate_implementation_sources(preflight)


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


def test_optimizer_is_loaded_weights_only(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config = load_config(CONFIG_PATH)
    contract = config["training"]["optimizer"]
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


def test_tensor_evidence_binds_v2_marker_weight_state_and_frozen_hashes(
    tmp_path: Path,
) -> None:
    config = load_config(CONFIG_PATH)
    module = SignedXLocalFieldSceneResidual(scene_dim=1536, latent_count=256, content_dim=128)
    with torch.no_grad():
        module.output_projection.weight[0, 0] = 0.125
    signed = {f"signed_x_scene_residual.{key}": value for key, value in module.state_dict().items()}
    global_state = {"global_scene_residual.synthetic": torch.tensor([0.25], dtype=torch.float32)}
    scene_state = {
        "scene_model.synthetic": torch.tensor([0.5], dtype=torch.float32),
        "composer.synthetic": torch.tensor([0.75], dtype=torch.float32),
        "grounding.synthetic": torch.tensor([1.0], dtype=torch.float32),
    }
    lora_state = {
        "lora_banks.extension_v13.synthetic": torch.tensor([1.25], dtype=torch.float32),
        "lora_banks.inherited_v12.synthetic": torch.tensor([1.5], dtype=torch.float32),
    }
    path = tmp_path / "adapter.safetensors"
    save_file({**signed, **global_state, **scene_state, **lora_state}, path)
    signed_hash = tensor_state_sha256(signed)
    global_hash = tensor_state_sha256(global_state)
    scene_hash = tensor_state_sha256(scene_state)
    lora_hashes = {
        bank: tensor_state_sha256({"synthetic": lora_state[f"lora_banks.{bank}.synthetic"]})
        for bank in ("extension_v13", "inherited_v12")
    }
    metadata = {
        "signed_x_scene_residual_state_sha256": signed_hash,
        "global_scene_residual_state_sha256": global_hash,
        "frozen_global_scene_residual_state_sha256": global_hash,
        "frozen_scene_state_sha256": scene_hash,
        "frozen_lora_bank_state_sha256": lora_hashes,
        "lora_bank_state_sha256": lora_hashes,
    }
    observed = verifier._load_tensor_evidence(
        path,
        metadata,
        config=config,
        expected_scene=scene_hash,
        expected_global=global_hash,
        expected_lora=lora_hashes,
    )
    assert observed["signed_x_state_sha256"] == signed_hash
    assert observed["global_scene_residual_state_sha256"] == global_hash
    assert observed["scene_state_sha256"] == scene_hash
    assert observed["lora_bank_state_sha256"] == lora_hashes


def test_wrong_v2_architecture_marker_is_rejected(tmp_path: Path) -> None:
    config = load_config(CONFIG_PATH)
    module = SignedXLocalFieldSceneResidual(scene_dim=1536, latent_count=256, content_dim=128)
    signed = {
        f"signed_x_scene_residual.{key}": value.clone()
        for key, value in module.state_dict().items()
    }
    signed["signed_x_scene_residual.architecture_marker"] = torch.tensor(1, dtype=torch.int64)
    path = tmp_path / "adapter.safetensors"
    save_file(
        {
            **signed,
            "global_scene_residual.synthetic": torch.tensor([0.25]),
            "scene_model.synthetic": torch.tensor([0.5]),
            "lora_banks.extension_v13.synthetic": torch.tensor([0.75]),
            "lora_banks.inherited_v12.synthetic": torch.tensor([1.0]),
        },
        path,
    )
    with pytest.raises(verifier.V20Update1Violation, match="structural state"):
        verifier._load_tensor_evidence(
            path,
            {"signed_x_scene_residual_state_sha256": tensor_state_sha256(signed)},
            config=config,
            expected_scene=_digest("scene"),
            expected_global=_digest("global"),
            expected_lora={
                "extension_v13": _digest("extension"),
                "inherited_v12": _digest("inherited"),
            },
        )


@pytest.mark.parametrize(
    ("key", "value", "match"),
    [
        ("unexpected.tensor", torch.tensor([1.0]), "unrecognized tensor keys"),
        ("composer.injected", torch.tensor([float("nan")]), "nonfinite tensors"),
    ],
)
def test_adapter_rejects_unrecognized_and_nonfinite_extra_tensors(
    tmp_path: Path,
    key: str,
    value: torch.Tensor,
    match: str,
) -> None:
    path = tmp_path / "adapter.safetensors"
    save_file({key: value}, path)
    with pytest.raises(verifier.V20Update1Violation, match=match):
        verifier._load_tensor_evidence(
            path,
            {},
            config=load_config(CONFIG_PATH),
            expected_scene=_digest("scene"),
            expected_global=_digest("global"),
            expected_lora={},
        )


def test_adapter_and_optimizer_inputs_reject_symlinks_before_deserialization(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    adapter_target = tmp_path / "adapter-target.safetensors"
    save_file({"unexpected.tensor": torch.tensor([1.0])}, adapter_target)
    adapter_link = tmp_path / "adapter.safetensors"
    adapter_link.symlink_to(adapter_target)
    load_called = False

    def unexpected_load(*_args: Any, **_kwargs: Any) -> Any:
        nonlocal load_called
        load_called = True
        raise AssertionError("deserializer must not run")

    monkeypatch.setattr(verifier, "load_file", unexpected_load)
    with pytest.raises(verifier.V20Update1Violation, match="symlink"):
        verifier._load_tensor_evidence(
            adapter_link,
            {},
            config=load_config(CONFIG_PATH),
            expected_scene=_digest("scene"),
            expected_global=_digest("global"),
            expected_lora={},
        )
    assert load_called is False

    optimizer_target = tmp_path / "optimizer-target.pt"
    optimizer_target.write_bytes(b"not read")
    optimizer_link = tmp_path / "optimizer.pt"
    optimizer_link.symlink_to(optimizer_target)
    with pytest.raises(verifier.V20Update1Violation, match="symlink"):
        verifier._load_optimizer_evidence(
            optimizer_link,
            contract={},
            expected_manifest={},
            expected_hash=_digest("optimizer"),
        )


def _checkpoint_fixture(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[dict[str, Any], Path]:
    config = load_config(CONFIG_PATH)
    checkpoint = tmp_path / "epoch_001"
    checkpoint.mkdir()
    (checkpoint / "adapter.safetensors").write_bytes(b"adapter")
    (checkpoint / "optimizer.pt").write_bytes(b"optimizer")
    source = tmp_path / "v18" / "epoch_004"
    source.mkdir(parents=True)
    source_metadata = {
        "epoch": 4,
        "output_namespace": "v18",
        "config_hash": "1" * 12,
        "source_provenance": {"source": "v18"},
    }
    _write_json(source / "metadata.json", source_metadata)
    source_artifacts = {
        "adapter_sha256": _digest("adapter"),
        "metadata_sha256": _digest("metadata"),
    }
    signed_initial = signed_x_scene_residual_settings(config).expected_initial_state_sha256
    zero_equivalence = {"verified": True}
    evidence = {
        "source": source,
        "source_metadata": source_metadata,
        "source_artifact_hashes": source_artifacts,
        "source_provenance": {"clean": True},
        "expected_scene_state_sha256": _digest("scene"),
        "expected_global_state_sha256": _digest("global"),
        "expected_lora_state_sha256": {"a": _digest("lora")},
        "pair_unit_selection_sha256": _digest("pair-units"),
        "pair_membership_sha256": _digest("membership"),
        "zero_equivalence": zero_equivalence,
        "predicted_signed_x_state_sha256": _digest("predicted-state"),
        "predicted_output_projection_sha256": _digest("predicted-W"),
        "optimizer_contract": config["training"]["optimizer"],
        "optimizer_manifest": {"synthetic": True},
        "optimizer_hash": _digest("optimizer-state"),
        "rich_preflight_reduction": {"verified": True},
    }
    history = {
        "epoch": 1,
        "pair_batch_count": 12,
        "pair_batch_fraction": 1.0,
        "train_loss": 0.5,
        "pair_candidate_gate": {"teacher": True},
    }
    metadata = {
        "schema_version": 3,
        "epoch": 1,
        "optimizer_step": 1,
        "global_step": 12,
        "history": [history],
        "train_loss": 0.5,
        "pair_candidate_gate": history["pair_candidate_gate"],
        "config_hash": config_hash(config),
        "output_namespace": config["training"]["output_namespace"],
        "gradient_accumulation": 12,
        "freeze_scene_adapter": True,
        "train_global_scene_residual_only": False,
        "train_signed_x_scene_residual_only": True,
        "question_dependent_scene_processing": False,
        "scene_latents": 256,
        "language_hidden_dim": 1536,
        "counterfactual_pair_unit_count": 12,
        "counterfactual_pair_unit_selection_sha256": evidence["pair_unit_selection_sha256"],
        "training_counterfactual_pair_count": 2,
        "training_counterfactual_pair_membership_sha256": evidence["pair_membership_sha256"],
        "max_questions_per_scene": 6,
        "source_provenance": evidence["source_provenance"],
        "global_scene_residual": global_scene_residual_settings(config).contract(),
        "global_scene_residual_parameter_count": 400_128,
        "global_scene_residual_state_sha256": evidence["expected_global_state_sha256"],
        "frozen_global_scene_residual_state_sha256": evidence["expected_global_state_sha256"],
        "signed_x_scene_residual": signed_x_scene_residual_settings(config).contract(),
        "signed_x_scene_residual_parameter_count": 196_608,
        "signed_x_scene_residual_initial_state_sha256": signed_initial,
        "signed_x_scene_residual_state_sha256": evidence["predicted_signed_x_state_sha256"],
        "signed_x_scene_residual_zero_output_equivalence": zero_equivalence,
        "frozen_scene_state_sha256": evidence["expected_scene_state_sha256"],
        "frozen_lora_bank_state_sha256": evidence["expected_lora_state_sha256"],
        "lora_bank_state_sha256": evidence["expected_lora_state_sha256"],
        "lora_trainable_parameter_count": 0,
        "initialize_expected_adapter_sha256": source_artifacts["adapter_sha256"],
        "initialize_expected_metadata_sha256": source_artifacts["metadata_sha256"],
        "initialize_expected_global_scene_residual_state_sha256": evidence[
            "expected_global_state_sha256"
        ],
        "initialize_source_residual_into_frozen_base": True,
        "initialization_provenance": {
            "schema_version": 4,
            "mode": "frozen_v18_residual_base_plus_zero_output_signed_x_residual",
            "checkpoint": str(source),
            "adapter_sha256": source_artifacts["adapter_sha256"],
            "metadata_sha256": source_artifacts["metadata_sha256"],
            "expected_adapter_sha256": source_artifacts["adapter_sha256"],
            "expected_metadata_sha256": source_artifacts["metadata_sha256"],
            "checkpoint_epoch": 4,
            "checkpoint_output_namespace": source_metadata["output_namespace"],
            "checkpoint_config_hash": source_metadata["config_hash"],
            "checkpoint_source_provenance": source_metadata["source_provenance"],
            "source_global_scene_residual_state_sha256": evidence["expected_global_state_sha256"],
            "expected_source_global_scene_residual_state_sha256": evidence[
                "expected_global_state_sha256"
            ],
            "global_scene_residual_frozen": True,
            "signed_x_scene_residual_initial_state_sha256": signed_initial,
            "signed_x_scene_residual_zero_output": True,
            "optimizer_state_loaded": False,
            "history_loaded": False,
        },
    }
    _write_json(checkpoint / "metadata.json", metadata)
    _write_json(tmp_path / "preflight.json", {})
    monkeypatch.setattr(verifier, "capture_git_source_provenance", lambda _root: {"clean": True})
    monkeypatch.setattr(verifier, "_clean_provenance", lambda value, _field: value)
    monkeypatch.setattr(verifier, "_validate_preflight", lambda *_args: evidence)
    monkeypatch.setattr(
        verifier,
        "_load_tensor_evidence",
        lambda *_args, **_kwargs: {
            "signed_x_state_sha256": evidence["predicted_signed_x_state_sha256"],
            "output_projection_sha256": evidence["predicted_output_projection_sha256"],
            "global_scene_residual_state_sha256": evidence["expected_global_state_sha256"],
            "scene_state_sha256": evidence["expected_scene_state_sha256"],
            "lora_bank_state_sha256": evidence["expected_lora_state_sha256"],
        },
    )
    monkeypatch.setattr(
        verifier,
        "_load_optimizer_evidence",
        lambda *_args, **_kwargs: {
            "manifest": evidence["optimizer_manifest"],
            "sha256": evidence["optimizer_hash"],
        },
    )
    return config, checkpoint


def test_epoch_one_boundary_and_report_only_claims(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config, checkpoint = _checkpoint_fixture(tmp_path, monkeypatch)
    report = verifier.verify_update1(config, tmp_path / "preflight.json", checkpoint)
    assert report["match"] is True
    assert report["stage_2_authorized"] is True
    assert report["report_only"] is True
    assert report["model_loaded"] is False
    assert report["scene_map_loaded"] is False
    assert report["oracle_loaded"] is False


def test_checkpoint_child_symlink_is_rejected_before_metadata_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config, checkpoint = _checkpoint_fixture(tmp_path, monkeypatch)
    metadata = checkpoint / "metadata.json"
    target = tmp_path / "metadata-target.json"
    metadata.replace(target)
    metadata.symlink_to(target)
    with pytest.raises(verifier.V20Update1Violation, match="symlink"):
        verifier.verify_update1(config, tmp_path / "preflight.json", checkpoint)


def test_source_child_symlink_and_forbidden_resolved_child_are_rejected(tmp_path: Path) -> None:
    target = tmp_path / "source-metadata-target.json"
    target.write_text("{}", encoding="utf-8")
    source = tmp_path / "source"
    source.mkdir()
    source_child = source / "metadata.json"
    source_child.symlink_to(target)
    with pytest.raises(verifier.V20Update1Violation, match="symlink"):
        verifier._read_json(source_child, "V18 source metadata")

    forbidden = tmp_path / "oracle" / "metadata.json"
    forbidden.parent.mkdir()
    forbidden.write_text("{}", encoding="utf-8")
    with pytest.raises(verifier.V20Update1Violation, match="refuses"):
        verifier._read_json(forbidden, "forbidden child")


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("epoch", 2, "checkpoint.epoch"),
        ("optimizer_step", 2, "checkpoint.optimizer_step"),
        ("global_step", 13, "checkpoint.global_step"),
    ],
)
def test_wrong_epoch_optimizer_or_global_step_is_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    value: int,
    match: str,
) -> None:
    config, checkpoint = _checkpoint_fixture(tmp_path, monkeypatch)
    metadata_path = checkpoint / "metadata.json"
    metadata = json.loads(metadata_path.read_text())
    metadata[field] = value
    _write_json(metadata_path, metadata)
    with pytest.raises(verifier.V20Update1Violation, match=match):
        verifier.verify_update1(config, tmp_path / "preflight.json", checkpoint)


def test_epoch_one_history_must_be_exactly_one_twelve_microstep_entry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config, checkpoint = _checkpoint_fixture(tmp_path, monkeypatch)
    metadata_path = checkpoint / "metadata.json"
    metadata = json.loads(metadata_path.read_text())
    metadata["history"][0]["pair_batch_count"] = 11
    _write_json(metadata_path, metadata)
    with pytest.raises(verifier.V20Update1Violation, match="pair batches"):
        verifier.verify_update1(config, tmp_path / "preflight.json", checkpoint)


@pytest.mark.parametrize("component", ["oracle", "maps", "runtime"])
def test_report_only_verifier_rejects_forbidden_input_paths(tmp_path: Path, component: str) -> None:
    with pytest.raises(verifier.V20Update1Violation, match="refuses"):
        verifier._reject_forbidden_input_path(tmp_path / component / "evidence.json")
