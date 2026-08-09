from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
import torch
import yaml
from safetensors.torch import load_file, save_file

from semantic_3d_chat.config import config_hash
from semantic_3d_chat.evaluation import v18_update1_verifier as verifier_module
from semantic_3d_chat.evaluation.v18_optimizer_state import (
    V18_RESIDUAL_OPTIMIZER_GROUP_NAME,
    canonical_v18_adamw_state,
)
from semantic_3d_chat.evaluation.v18_structural_preflight import (
    EXPECTED_CONTINUATION,
    EXPECTED_ELIGIBILITY,
    EXPECTED_FULL_TEACHER_GATE,
    EXPECTED_RANKING_FIELDS,
    STRUCTURAL_PREFLIGHT_ROLE,
    V18_SCREEN_ROLE,
    StructuralThresholds,
    canonical_sha256,
    file_sha256,
    validate_v18_config_contract,
)
from semantic_3d_chat.evaluation.v18_update1_verifier import (
    STAGE_EXECUTION_METADATA,
    V18Update1Violation,
    main,
    verify_update1,
)
from semantic_3d_chat.language.lora import tensor_state_sha256
from semantic_3d_chat.scene_encoder import global_residual as residual_source
from semantic_3d_chat.scene_encoder.global_residual import (
    ZERO_SPATIAL_MEAN_CONTENT_GATE_V1,
    GlobalSceneResidual,
    global_scene_residual_settings,
)
from semantic_3d_chat.training.source_provenance import SOURCE_SCOPE


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _provenance(name: str) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "scope": SOURCE_SCOPE,
        "available": True,
        "head_commit": _digest(f"{name}-commit")[:40],
        "head_tree": _digest(f"{name}-tree")[:40],
        "is_clean": True,
        "tracked_diff_sha256": hashlib.sha256(b"").hexdigest(),
    }


def _zero_prefix_evidence(prefixes: dict[str, str]) -> dict[str, Any]:
    return {
        "verified": True,
        "question_dependent_scene_processing": False,
        "scene_count": len(prefixes),
        "scene_prefixes": {
            scene_id: {
                "core_prefix_sha256": value,
                "adapted_prefix_sha256": value,
            }
            for scene_id, value in prefixes.items()
        },
    }


@dataclass
class SyntheticBundle:
    config: dict[str, Any]
    config_path: Path
    preflight_path: Path
    checkpoint: Path
    predicted_hash: str
    optimizer_hash: str
    source_provenance: dict[str, Any]


@pytest.fixture(autouse=True)
def _pin_current_source_provenance(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        verifier_module,
        "capture_git_source_provenance",
        lambda _root: _provenance("v18"),
    )


def _build_bundle(tmp_path: Path) -> SyntheticBundle:
    implementation_hash = file_sha256(Path(residual_source.__file__).resolve())
    current_provenance = _provenance("v18")
    v14_provenance = _provenance("v14")
    frozen_scene_hash = _digest("frozen-scene")
    frozen_banks = {
        "inherited_v12": _digest("inherited-v12"),
        "extension_v13": _digest("extension-v13"),
    }

    evidence_directory = tmp_path / "evidence"
    evidence_directory.mkdir()
    v16_evidence = evidence_directory / "v16.json"
    v17_evidence = evidence_directory / "v17.json"
    v16_evidence.write_bytes(b'{"audit":"v16"}\n')
    v17_evidence.write_bytes(b'{"audit":"v17"}\n')

    source = tmp_path / "checkpoints" / "v14" / "epoch_007"
    source.mkdir(parents=True)
    (source / "adapter.safetensors").write_bytes(b"synthetic pinned V14 adapter")
    source_metadata = {
        "schema_version": 3,
        "epoch": 7,
        "output_namespace": "synthetic_v14",
        "config_hash": "a" * 12,
        "source_provenance": v14_provenance,
        "frozen_scene_state_sha256": frozen_scene_hash,
        "frozen_lora_bank_state_sha256": {"inherited_v12": frozen_banks["inherited_v12"]},
        "lora_bank_state_sha256": frozen_banks,
    }
    _write_json(source / "metadata.json", source_metadata)

    residual = GlobalSceneResidual(
        scene_dim=1536,
        latent_count=256,
        width=128,
        fourier_bands=4,
        initialization_seed=18018,
        architecture_version=ZERO_SPATIAL_MEAN_CONTENT_GATE_V1,
        gate_temperature=1.0,
    )
    initial_state = {
        f"global_scene_residual.{name}": value.detach().cpu().clone()
        for name, value in residual.state_dict().items()
    }
    initial_hash = tensor_state_sha256(initial_state)
    position_hash = tensor_state_sha256(
        {"position_features": residual.position_features.detach().cpu()}
    )
    optimizer_parameters = list(residual.parameters())
    for parameter in optimizer_parameters:
        parameter.grad = torch.zeros_like(parameter)
    assert residual.output_projection.weight.grad is not None
    residual.output_projection.weight.grad[0, 0] = -1.0
    exact_optimizer = torch.optim.AdamW(
        [
            {
                "name": V18_RESIDUAL_OPTIMIZER_GROUP_NAME,
                "params": optimizer_parameters,
                "lr": 0.001,
                "weight_decay": 0.0,
            }
        ],
        betas=(0.9, 0.999),
        eps=1.0e-8,
        foreach=False,
        fused=False,
        capturable=False,
        maximize=False,
        amsgrad=False,
    )
    exact_optimizer.step()
    updated_state = {
        f"global_scene_residual.{name}": value.detach().cpu().clone()
        for name, value in residual.state_dict().items()
    }
    predicted_hash = tensor_state_sha256(updated_state)

    ordered_units = [
        {
            "microstep": index,
            "pair_id": "pair_000001" if index <= 6 else "pair_000003",
            "question_key": f"opaque_question_{index:02d}",
            "reference_scene_id": "scene_000003" if index <= 6 else "scene_000007",
            "reference_question_id": f"opaque_reference_{index:02d}",
            "counterfactual_scene_id": "scene_000004" if index <= 6 else "scene_000008",
            "counterfactual_question_id": f"opaque_counterfactual_{index:02d}",
        }
        for index in range(1, 13)
    ]
    prefixes = {
        scene_id: _digest(f"prefix-{scene_id}")
        for scene_id in ("scene_000003", "scene_000004", "scene_000007", "scene_000008")
    }
    expected_hashes = {
        "ordered_unit_sha256": canonical_sha256(ordered_units),
        "source_adapter_sha256": file_sha256(source / "adapter.safetensors"),
        "source_metadata_sha256": file_sha256(source / "metadata.json"),
        "frozen_scene_state_sha256": frozen_scene_hash,
        "frozen_lora_bank_state_sha256": frozen_banks,
        "initial_residual_state_sha256": initial_hash,
        "position_features_sha256": position_hash,
        "selection_sha256": _digest("selection"),
        "pair_membership_sha256": _digest("pair-membership"),
        "core_prefix_sha256": prefixes,
        "v16_gradient_audit_sha256": file_sha256(v16_evidence),
        "v17_lr_response_sha256": file_sha256(v17_evidence),
    }
    optimizer = {
        "name": "AdamW",
        "learning_rate": 0.001,
        "betas": [0.9, 0.999],
        "epsilon": 1.0e-8,
        "weight_decay": 0.0,
        "foreach": False,
        "fused": False,
        "capturable": False,
        "maximize": False,
        "amsgrad": False,
        "gradient_clip_norm": 1.0,
        "accumulation_divisor": 12,
        "step_index": 1,
    }
    optimizer_manifest, optimizer_hash = canonical_v18_adamw_state(
        exact_optimizer.state_dict(), optimizer
    )
    config_path = tmp_path / "resolved_v18.yaml"
    config: dict[str, Any] = {
        "scene_encoder": {
            "global_latents": 256,
            "model_dim": 512,
            "global_scene_residual": {
                "enabled": True,
                "architecture_version": ZERO_SPATIAL_MEAN_CONTENT_GATE_V1,
                "width": 128,
                "fourier_bands": 4,
                "initialization_seed": 18018,
                "gate_temperature": 1.0,
                "expected_initial_state_sha256": initial_hash,
            },
        },
        "training": {
            "batch_size": 2,
            "max_questions_per_scene": 6,
            "language_decoder_gradient_checkpointing": True,
            "initialize_legacy_lora_into_bank": None,
            "initialize_named_lora_freeze_transition": True,
            "output_namespace": "synthetic_v18",
            "initialize_from": str(source),
            "initialize_expected_adapter_sha256": expected_hashes["source_adapter_sha256"],
            "initialize_expected_metadata_sha256": expected_hashes["source_metadata_sha256"],
            "train_global_scene_residual_only": True,
            "freeze_scene_adapter": True,
            "learning_rate": 0.001,
            "weight_decay": 0.0,
            "gradient_clip_norm": 1.0,
            "gradient_accumulation": 12,
            "pair_steps_per_epoch": 12,
            "epochs": 4,
            "pair_only_mode": True,
            "pair_batch_fraction": 1.0,
            "pair_units_per_batch": 1,
            "pair_max_units_per_pair": 6,
            "pair_only_scene_ids": [
                "scene_000003",
                "scene_000004",
                "scene_000007",
                "scene_000008",
            ],
            "pair_ranking_weight": 8.0,
            "pair_ranking_margin": 1.0,
            "pair_ranking_mode": "candidate_logit",
            "pair_full_vocab_ranking_weight": 2.0,
            "pair_full_vocab_ranking_margin": 1.0,
            "grounding_weight": 0.0,
            "grounding_anchor_weight": 0.0,
            "latent_diversity_weight": 0.0,
            "paired_scene_separation_weight": 0.0,
            "spatial_answer_contrastive_weight": 0.0,
            "spatial_answer_warmup_steps": 0,
            "spatial_relation_contrastive_weight": 0.0,
            "spatial_relation_warmup_steps": 0,
            "pair_gate_enabled": True,
            "pair_gate_every_epochs": 1,
            "pair_gate_changed_unit_accuracy": 0.95,
            "pair_gate_prediction_flip_rate": 1.0,
            "pair_gate_wrong_prefix_flip_rate": 1.0,
            "pair_gate_first_answer_token_top1_accuracy": 1.0,
            "pair_gate_stop_when_passed": False,
            "early_stopping_patience": 0,
            "optimizer": copy.deepcopy(optimizer),
        },
        "language": {
            "model_id": "google/gemma-4-E2B-it",
            "revision": "3e22461f65e89153144f8adb70e3b8c2cc9845a7",
            "backend": "gemma4",
            "dtype": "bfloat16",
            "scene_prefix_after_bos": True,
            "scene_boundary_mode": "gemma4_native_image",
            "system_prompt": (
                "You answer using only the continuous 3D scene memory supplied before this "
                "conversation. Do not invent objects or relationships unsupported by the "
                "scene. If there is not enough evidence, answer unknown."
            ),
        },
        "experiment": {"residual_parameter_count": 400_128},
        "structural_preflight": {
            "schema_version": 1,
            "required": True,
            "role": STRUCTURAL_PREFLIGHT_ROLE,
            "architecture_version": ZERO_SPATIAL_MEAN_CONTENT_GATE_V1,
            "spatial_centering": "all_slots_fp32",
            "content_gate": "bias_free_scalar_sigmoid_centered_content",
            "implementation_source_sha256": implementation_hash,
            "source_must_be_clean": True,
            "latent_count": 256,
            "scene_dim": 1536,
            "residual_parameter_count": 400_128,
            "exact_epoch": 1,
            "microsteps": 12,
            "optimizer": copy.deepcopy(optimizer),
            "thresholds": StructuralThresholds().__dict__,
            "expected_hashes": expected_hashes,
            "evidence_paths": {
                "v16_gradient_audit": str(v16_evidence),
                "v17_lr_response": str(v17_evidence),
            },
        },
        "v18_screen": {
            "schema_version": 1,
            "role": V18_SCREEN_ROLE,
            "learning_rate": 0.001,
            "screen_optimizer_updates": 4,
            "conditional_max_optimizer_updates": 12,
            "epoch_tiebreaker": "lower_epoch",
            "execution_stages": {
                **STAGE_EXECUTION_METADATA,
                "predicted_preflight_state_must_match_epoch_001": True,
            },
            "eligibility_requires": EXPECTED_ELIGIBILITY,
            "ranking_descending": list(EXPECTED_RANKING_FIELDS),
            "continuation_requires": EXPECTED_CONTINUATION,
            "full_teacher_gate_requires": EXPECTED_FULL_TEACHER_GATE,
            "greedy_audit_only_after_full_teacher_gate": True,
        },
    }
    config["_config_path"] = str(config_path)
    config_path.write_text(
        yaml.safe_dump({key: value for key, value in config.items() if not key.startswith("_")}),
        encoding="utf-8",
    )
    contract = validate_v18_config_contract(
        config, implementation_source_sha256=implementation_hash
    )

    zero_prefix = _zero_prefix_evidence(prefixes)
    rng_hash = _digest("cpu-rng")
    preflight = {
        "schema_version": 1,
        "audit_type": "v18_exact_ordered_structural_preflight",
        "runtime_eligible": False,
        "uses_supervised_qa_metadata": True,
        "question_dependent_scene_processing": False,
        "live_optimizer_constructed": False,
        "live_optimizer_step_executed": False,
        "optimizer_steps": 0,
        "isolated_clone_optimizer_constructed": True,
        "isolated_clone_optimizer_steps": 1,
        "structural_authorization": True,
        "config_path": str(config_path),
        "config_sha256": config_hash(config, length=64),
        "contract": contract,
        "source_provenance": current_provenance,
        "implementation_source": str(Path(residual_source.__file__).resolve()),
        "implementation_source_sha256": implementation_hash,
        "source_checkpoint": str(source),
        "source_checkpoint_epoch": 7,
        "source_hashes": {
            name: expected_hashes[name]
            for name in (
                "source_adapter_sha256",
                "source_metadata_sha256",
                "frozen_scene_state_sha256",
                "frozen_lora_bank_state_sha256",
            )
        },
        "initial_residual_state_sha256": initial_hash,
        "live_residual_state_sha256_before": initial_hash,
        "live_residual_state_sha256_after": initial_hash,
        "live_parameter_state_unchanged": True,
        "simulated_first_output_projection_state_sha256": predicted_hash,
        "structural_state": {
            "architecture_version": ZERO_SPATIAL_MEAN_CONTENT_GATE_V1,
            "parameter_count": 400_128,
            "latent_count": 256,
            "scene_dim": 1536,
            "gate_temperature": 1.0,
            "spatial_centering": "all_slots_fp32",
            "content_gate": "bias_free_scalar_sigmoid_centered_content",
        },
        "position_features_sha256": position_hash,
        "rng_state": {
            "domains": {
                "cpu": {
                    "available": True,
                    "before_sha256": rng_hash,
                    "after_sha256": rng_hash,
                    "unchanged": True,
                },
                "mps": {
                    "available": False,
                    "before_sha256": None,
                    "after_sha256": None,
                    "unchanged": True,
                },
            },
            "all_available_domains_unchanged": True,
            "restored_after_mismatch": False,
        },
        "zero_output_prefix_equivalence": zero_prefix,
        "selection_sha256": expected_hashes["selection_sha256"],
        "pair_membership_sha256": expected_hashes["pair_membership_sha256"],
        "ordered_unit_sha256": expected_hashes["ordered_unit_sha256"],
        "ordered_units": ordered_units,
        "adamw_contract": optimizer,
        "gradient": {
            "implementation": "isolated_full_residual_torch_adamw_clone",
            "parameter_count": 1536 * 128,
            "clone_optimizer_state_parameter_count": 8,
            "changed_parameter_keys": ["output_projection.weight"],
            "clone_residual_state_sha256": predicted_hash,
            "gradient_sha256": _digest("gradient"),
            "clipped_gradient_sha256": _digest("clipped-gradient"),
            "simulated_update_sha256": _digest("simulated-update"),
            "clone_optimizer_state_tensor_sha256": _digest("optimizer-state"),
            "clone_optimizer_state_manifest": optimizer_manifest,
            "clone_optimizer_state_sha256": optimizer_hash,
        },
        "structural_gate": {"passed": True},
    }
    preflight_path = tmp_path / "v18_preflight.json"
    _write_json(preflight_path, preflight)

    checkpoint = tmp_path / "checkpoints" / "v18" / "epoch_001"
    checkpoint.mkdir(parents=True)
    save_file(updated_state, checkpoint / "adapter.safetensors")
    initialization_provenance = {
        "schema_version": 3,
        "mode": "named_lora_banks_frozen_plus_zero_output_scene_residual",
        "checkpoint": str(source),
        "adapter_sha256": expected_hashes["source_adapter_sha256"],
        "metadata_sha256": expected_hashes["source_metadata_sha256"],
        "expected_adapter_sha256": expected_hashes["source_adapter_sha256"],
        "expected_metadata_sha256": expected_hashes["source_metadata_sha256"],
        "checkpoint_epoch": 7,
        "checkpoint_output_namespace": source_metadata["output_namespace"],
        "checkpoint_config_hash": source_metadata["config_hash"],
        "checkpoint_source_provenance": v14_provenance,
        "optimizer_state_loaded": False,
        "history_loaded": False,
        "source_lora_bank_state_sha256": frozen_banks,
        "all_source_lora_banks_frozen": True,
        "global_scene_residual_initial_state_sha256": initial_hash,
        "global_scene_residual_zero_output": True,
    }
    metadata = {
        "schema_version": 3,
        "epoch": 1,
        "optimizer_step": 1,
        "global_step": 12,
        "history": [{"epoch": 1, "pair_batch_count": 12, "pair_batch_fraction": 1.0}],
        "config_hash": config_hash(config),
        "output_namespace": "synthetic_v18",
        "gradient_accumulation": 12,
        "v18_stage_execution": STAGE_EXECUTION_METADATA,
        "freeze_scene_adapter": True,
        "train_global_scene_residual_only": True,
        "question_dependent_scene_processing": False,
        "scene_latents": 256,
        "language_hidden_dim": 1536,
        "global_scene_residual": global_scene_residual_settings(config).contract(),
        "global_scene_residual_parameter_count": 400_128,
        "global_scene_residual_initial_state_sha256": initial_hash,
        "global_scene_residual_state_sha256": predicted_hash,
        "global_scene_residual_zero_output_equivalence": zero_prefix,
        "frozen_scene_state_sha256": frozen_scene_hash,
        "frozen_lora_bank_state_sha256": frozen_banks,
        "lora_bank_state_sha256": frozen_banks,
        "source_provenance": current_provenance,
        "initialization_provenance": initialization_provenance,
    }
    _write_json(checkpoint / "metadata.json", metadata)
    torch.save(exact_optimizer.state_dict(), checkpoint / "optimizer.pt")
    return SyntheticBundle(
        config,
        config_path,
        preflight_path,
        checkpoint,
        predicted_hash,
        optimizer_hash,
        current_provenance,
    )


def test_exact_match_authorizes_with_safe_exact_optimizer_load(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bundle = _build_bundle(tmp_path)
    real_load = torch.load
    calls: list[dict[str, Any]] = []

    def recording_load(*args: Any, **kwargs: Any) -> Any:
        calls.append(dict(kwargs))
        return real_load(*args, **kwargs)

    monkeypatch.setattr(torch, "load", recording_load)
    report = verify_update1(bundle.config, bundle.preflight_path, bundle.checkpoint)

    assert report["match"] is True
    assert report["stage_2_authorized"] is True
    assert report["model_loaded"] is False
    assert report["oracle_loaded"] is False
    assert report["optimizer_deserialized"] is True
    assert calls == [{"weights_only": True, "map_location": "cpu"}]
    assert report["optimizer_deserialization"]["canonical_state_validated"] is True
    assert report["source_provenance"] == bundle.source_provenance
    assert report["preflight"]["predicted_residual_state_sha256"] == bundle.predicted_hash
    assert report["checkpoint"]["residual_state_sha256"] == bundle.predicted_hash
    assert report["preflight"]["optimizer_state_sha256"] == bundle.optimizer_hash
    assert report["checkpoint"]["optimizer_state_sha256"] == bundle.optimizer_hash
    assert report["checkpoint"]["optimizer_state_manifest"]["state_parameter_count"] == 8
    assert report["checkpoint"]["optimizer_size_bytes"] > 0


def _load_optimizer_state(bundle: SyntheticBundle) -> dict[str, Any]:
    return torch.load(
        bundle.checkpoint / "optimizer.pt",
        weights_only=True,
        map_location="cpu",
    )


def test_arbitrary_optimizer_bytes_are_safely_rejected(tmp_path: Path) -> None:
    bundle = _build_bundle(tmp_path)
    (bundle.checkpoint / "optimizer.pt").write_bytes(b"not a torch checkpoint")

    with pytest.raises(V18Update1Violation, match="safely deserialize") as captured:
        verify_update1(bundle.config, bundle.preflight_path, bundle.checkpoint)

    assert captured.value.optimizer_deserialized is False


@pytest.mark.parametrize(
    ("case", "match"),
    [
        ("moments", "manifest differs"),
        ("parameter_order", "parameter order mismatch"),
        ("options", "AdamW lr mismatch"),
        ("step", "step mismatch"),
    ],
)
def test_tampered_optimizer_state_is_rejected_after_safe_load(
    tmp_path: Path, case: str, match: str
) -> None:
    bundle = _build_bundle(tmp_path)
    state = _load_optimizer_state(bundle)
    if case == "moments":
        state["state"][0]["exp_avg"] = state["state"][0]["exp_avg"].clone()
        state["state"][0]["exp_avg"][0] = 0.125
    elif case == "parameter_order":
        state["param_groups"][0]["params"][0:2] = [1, 0]
    elif case == "options":
        state["param_groups"][0]["lr"] = 0.002
    elif case == "step":
        state["state"][0]["step"] = torch.tensor(2.0)
    torch.save(state, bundle.checkpoint / "optimizer.pt")

    with pytest.raises(V18Update1Violation, match=match) as captured:
        verify_update1(bundle.config, bundle.preflight_path, bundle.checkpoint)

    assert captured.value.optimizer_deserialized is True


def test_current_clean_source_must_equal_preflight_and_checkpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bundle = _build_bundle(tmp_path)
    monkeypatch.setattr(
        verifier_module,
        "capture_git_source_provenance",
        lambda _root: _provenance("different-clean-source"),
    )

    with pytest.raises(V18Update1Violation, match="current/preflight clean source"):
        verify_update1(bundle.config, bundle.preflight_path, bundle.checkpoint)


def test_predicted_state_mismatch_is_rejected(tmp_path: Path) -> None:
    bundle = _build_bundle(tmp_path)
    preflight = _read_json(bundle.preflight_path)
    preflight["simulated_first_output_projection_state_sha256"] = _digest("wrong prediction")
    _write_json(bundle.preflight_path, preflight)

    with pytest.raises(V18Update1Violation, match="predicted preflight"):
        verify_update1(bundle.config, bundle.preflight_path, bundle.checkpoint)


def test_rehashed_tensor_tamper_is_rejected_by_preflight_prediction(tmp_path: Path) -> None:
    bundle = _build_bundle(tmp_path)
    adapter = bundle.checkpoint / "adapter.safetensors"
    tensors = load_file(adapter, device="cpu")
    tensors["global_scene_residual.output_projection.weight"] = tensors[
        "global_scene_residual.output_projection.weight"
    ].clone()
    tensors["global_scene_residual.output_projection.weight"][0, 1] = 0.125
    save_file(tensors, adapter)
    residual = {
        name: value for name, value in tensors.items() if name.startswith("global_scene_residual.")
    }
    metadata = _read_json(bundle.checkpoint / "metadata.json")
    metadata["global_scene_residual_state_sha256"] = tensor_state_sha256(residual)
    _write_json(bundle.checkpoint / "metadata.json", metadata)

    with pytest.raises(V18Update1Violation, match="predicted preflight"):
        verify_update1(bundle.config, bundle.preflight_path, bundle.checkpoint)


@pytest.mark.parametrize(
    ("case", "match"),
    [
        ("epoch", "checkpoint.epoch"),
        ("update", "checkpoint.optimizer_step"),
        ("config", "checkpoint.config_hash"),
        ("preflight", "preflight.config_sha256"),
        ("frozen", "checkpoint frozen scene hash"),
        ("stage", "v18_stage_execution"),
    ],
)
def test_wrong_epoch_update_config_preflight_or_frozen_contract_is_rejected(
    tmp_path: Path, case: str, match: str
) -> None:
    bundle = _build_bundle(tmp_path)
    if case == "preflight":
        preflight = _read_json(bundle.preflight_path)
        preflight["config_sha256"] = _digest("wrong config")
        _write_json(bundle.preflight_path, preflight)
    else:
        metadata_path = bundle.checkpoint / "metadata.json"
        metadata = _read_json(metadata_path)
        if case == "epoch":
            metadata["epoch"] = 2
        elif case == "update":
            metadata["optimizer_step"] = 2
        elif case == "config":
            metadata["config_hash"] = "0" * 12
        elif case == "frozen":
            metadata["frozen_scene_state_sha256"] = _digest("wrong frozen scene")
        elif case == "stage":
            metadata["v18_stage_execution"]["stage_2_load_history"] = False
        _write_json(metadata_path, metadata)

    with pytest.raises(V18Update1Violation, match=match):
        verify_update1(bundle.config, bundle.preflight_path, bundle.checkpoint)


def test_changed_evidence_file_is_rejected(tmp_path: Path) -> None:
    bundle = _build_bundle(tmp_path)
    evidence_path = Path(
        bundle.config["structural_preflight"]["evidence_paths"]["v16_gradient_audit"]
    )
    evidence_path.write_bytes(b"tampered after preflight\n")

    with pytest.raises(V18Update1Violation, match="pinned v16_gradient_audit evidence hash"):
        verify_update1(bundle.config, bundle.preflight_path, bundle.checkpoint)


def test_cli_writes_denial_report_and_returns_nonzero(tmp_path: Path) -> None:
    bundle = _build_bundle(tmp_path)
    metadata = _read_json(bundle.checkpoint / "metadata.json")
    metadata["epoch"] = 2
    _write_json(bundle.checkpoint / "metadata.json", metadata)
    report_path = tmp_path / "decision.json"

    result = main(
        [
            "--config",
            str(bundle.config_path),
            "--preflight",
            str(bundle.preflight_path),
            "--checkpoint",
            str(bundle.checkpoint),
            "--report",
            str(report_path),
        ]
    )

    assert result == 2
    denial = _read_json(report_path)
    assert denial["match"] is False
    assert denial["stage_2_authorized"] is False
    assert "checkpoint.epoch" in denial["violation"]
    assert denial["optimizer_deserialized"] is False
