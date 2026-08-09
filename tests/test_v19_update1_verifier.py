from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
import torch
from safetensors.torch import load_file, save_file

from semantic_3d_chat.config import config_hash, load_config
from semantic_3d_chat.evaluation import v19_update1_verifier as verifier_module
from semantic_3d_chat.evaluation.v19_optimizer_state import canonical_v19_adamw_state
from semantic_3d_chat.evaluation.v19_structural_preflight import (
    V19_PREFLIGHT_ROLE,
    canonical_sha256,
    exact_clone_adamw_evidence,
    validate_v19_config_contract,
)
from semantic_3d_chat.evaluation.v19_update1_verifier import (
    V19Update1Violation,
    verify_update1,
)
from semantic_3d_chat.language.lora import tensor_state_sha256
from semantic_3d_chat.scene_encoder.global_residual import global_scene_residual_settings
from semantic_3d_chat.scene_encoder.signed_x_residual import (
    SignedXSceneResidual,
    signed_x_scene_residual_settings,
)
from semantic_3d_chat.training.checkpointing import module_collection_state_sha256
from semantic_3d_chat.training.source_provenance import SOURCE_SCOPE
from semantic_3d_chat.training.train_adapter import file_sha256

CONFIG_PATH = "configs/experiments/gemma4_color_mirror_signed_x_moment_v19.yaml"
PAIR_UNIT_SHA256 = "d5928cb783339ef62fff5c14a8c7f85f90d3a7a6cb8edad0a784998082740d3e"


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _provenance(name: str = "v19") -> dict[str, Any]:
    return {
        "schema_version": 1,
        "scope": SOURCE_SCOPE,
        "available": True,
        "head_commit": _digest(f"{name}-commit")[:40],
        "head_tree": _digest(f"{name}-tree")[:40],
        "is_clean": True,
        "tracked_diff_sha256": hashlib.sha256(b"").hexdigest(),
    }


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


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
        betas=tuple(float(item) for item in contract["betas"]),
        eps=float(contract["epsilon"]),
        foreach=bool(contract["foreach"]),
        fused=bool(contract["fused"]),
        capturable=bool(contract["capturable"]),
        maximize=bool(contract["maximize"]),
        amsgrad=bool(contract["amsgrad"]),
    )


@dataclass
class SyntheticBundle:
    config: dict[str, Any]
    preflight_path: Path
    checkpoint: Path
    provenance: dict[str, Any]
    predicted_signed_hash: str
    predicted_output_hash: str
    optimizer_hash: str
    expected_global_hash: str


@pytest.fixture(autouse=True)
def _pin_current_provenance(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        verifier_module,
        "capture_git_source_provenance",
        lambda _root: _provenance(),
    )


@pytest.fixture
def bundle(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> SyntheticBundle:
    config = load_config(CONFIG_PATH)
    contract = validate_v19_config_contract(config)
    training = config["training"]
    optimizer_contract = copy.deepcopy(training["optimizer"])
    current_provenance = _provenance()

    source = (tmp_path / "synthetic_v18" / "epoch_004").resolve()
    source.mkdir(parents=True)
    global_tensors = {
        "global_scene_residual.output_projection.weight": torch.linspace(
            -0.25,
            0.25,
            512,
            dtype=torch.float32,
        ).reshape(32, 16)
    }
    scene_tensors = {
        "scene_model.synthetic_weight": torch.arange(12, dtype=torch.float32).reshape(3, 4),
        "composer.synthetic_boundary": torch.tensor([0.5], dtype=torch.float32),
        "grounding.synthetic_weight": torch.tensor([-0.5, 0.25], dtype=torch.float32),
    }
    lora_tensors = {
        "lora_banks.extension_v13.synthetic_weight": torch.tensor(
            [0.125, -0.25], dtype=torch.float32
        ),
        "lora_banks.inherited_v12.synthetic_weight": torch.tensor(
            [0.375, -0.5], dtype=torch.float32
        ),
    }
    save_file(
        {**global_tensors, **scene_tensors, **lora_tensors},
        source / "adapter.safetensors",
    )
    expected_global_hash = tensor_state_sha256(global_tensors)
    expected_scene_hash = tensor_state_sha256(scene_tensors)
    expected_lora_hashes = {
        bank_name: tensor_state_sha256(
            {
                key[len(f"lora_banks.{bank_name}.") :]: value
                for key, value in lora_tensors.items()
                if key.startswith(f"lora_banks.{bank_name}.")
            }
        )
        for bank_name in ("extension_v13", "inherited_v12")
    }
    source_metadata = {
        "schema_version": 3,
        "epoch": 4,
        "output_namespace": "synthetic_v18",
        "config_hash": "a" * 12,
        "source_provenance": _provenance("synthetic-v18-source"),
        "global_scene_residual_state_sha256": expected_global_hash,
        "lora_bank_state_sha256": expected_lora_hashes,
    }
    _write_json(source / "metadata.json", source_metadata)
    source_artifacts = {
        "adapter_sha256": file_sha256(source / "adapter.safetensors"),
        "metadata_sha256": file_sha256(source / "metadata.json"),
    }
    training["initialize_from"] = str(source)
    training["initialize_expected_adapter_sha256"] = source_artifacts["adapter_sha256"]
    training["initialize_expected_metadata_sha256"] = source_artifacts["metadata_sha256"]
    training["initialize_expected_global_scene_residual_state_sha256"] = expected_global_hash
    expected = contract["expected_hashes"]
    expected.update(
        {
            "source_adapter_sha256": source_artifacts["adapter_sha256"],
            "source_metadata_sha256": source_artifacts["metadata_sha256"],
            "source_scene_state_sha256": expected_scene_hash,
            "source_global_scene_residual_state_sha256": expected_global_hash,
            "source_lora_bank_state_sha256": expected_lora_hashes,
        }
    )
    contract_without_hash = {
        key: value for key, value in contract.items() if key != "contract_sha256"
    }
    contract["contract_sha256"] = canonical_sha256(contract_without_hash)
    monkeypatch.setattr(
        verifier_module,
        "validate_v19_config_contract",
        lambda _config: copy.deepcopy(contract),
    )

    signed = SignedXSceneResidual(scene_dim=1536, latent_count=256, content_dim=128)
    initial_signed_hash = module_collection_state_sha256({"signed_x_scene_residual": signed})
    assert (
        initial_signed_hash
        == signed_x_scene_residual_settings(config).expected_initial_state_sha256
    )
    gradient = torch.linspace(
        -1.0,
        1.0,
        signed.output_projection.weight.numel(),
        dtype=torch.float32,
    ).reshape_as(signed.output_projection.weight)
    signed.output_projection.weight.grad = gradient.clone()
    predicted_weight, predicted = exact_clone_adamw_evidence(signed, optimizer_contract)

    optimizer = _optimizer(optimizer_contract, signed.output_projection.weight)
    torch.nn.utils.clip_grad_norm_(
        [signed.output_projection.weight],
        float(optimizer_contract["gradient_clip_norm"]),
    )
    optimizer.step()
    assert torch.equal(signed.output_projection.weight.detach(), predicted_weight)
    optimizer_manifest, optimizer_hash = canonical_v19_adamw_state(
        optimizer.state_dict(), optimizer_contract
    )
    assert optimizer_manifest == predicted["canonical_adamw_state_manifest"]
    assert optimizer_hash == predicted["canonical_adamw_state_sha256"]

    signed_state = {
        f"signed_x_scene_residual.{key}": value.detach().cpu().contiguous()
        for key, value in signed.state_dict().items()
    }
    predicted_signed_hash = tensor_state_sha256(signed_state)
    predicted_output_hash = tensor_state_sha256(
        {
            "signed_x_scene_residual.output_projection.weight": signed_state[
                "signed_x_scene_residual.output_projection.weight"
            ]
        }
    )
    assert predicted_signed_hash == predicted["predicted_signed_x_scene_residual_state_sha256"]
    assert predicted_output_hash == predicted["predicted_output_weight_sha256"]

    prefix_rows = {
        scene_id: {
            "v18_base_prefix_sha256": _digest(f"prefix-{scene_id}"),
            "signed_x_adapted_prefix_sha256": _digest(f"prefix-{scene_id}"),
        }
        for scene_id in ("scene_000003", "scene_000004", "scene_000007", "scene_000008")
    }
    zero_equivalence = {
        "verified": True,
        "base": "loaded_frozen_global_scene_residual",
        "question_dependent_scene_processing": False,
        "all_scene_slots_accounted": True,
        "scene_count": 4,
        "scene_prefixes": prefix_rows,
    }
    combined_source_hash = _digest("combined-frozen-source")
    microsteps = [
        {
            "microstep": index,
            "pair_id": "pair_000001" if index <= 6 else "pair_000003",
            "total_loss": 0.0 if index <= 6 else 1.0,
        }
        for index in range(1, 13)
    ]
    preflight = {
        "schema_version": 1,
        "audit_type": V19_PREFLIGHT_ROLE,
        "runtime_eligible": False,
        "uses_supervised_qa_metadata": True,
        "question_dependent_scene_processing": False,
        "live_optimizer_constructed": False,
        "live_optimizer_step_executed": False,
        "optimizer_steps": 0,
        "isolated_clone_optimizer_constructed": True,
        "isolated_clone_optimizer_steps": 1,
        "authorized": True,
        "structural_authorization": True,
        "authorization_checks": {
            "source_and_config_contracts_passed": True,
            "exact_selection_and_order_passed": True,
            "step_zero_identity_all_scenes": True,
            "color_losses_exactly_zero": True,
            "color_isolated_signed_x_gradient_exactly_zero": True,
            "mirror_signed_x_gradient_finite_nonzero": True,
            "accumulated_signed_x_gradient_finite_nonzero": True,
            "only_signed_x_output_weight_has_gradient": True,
            "predicted_adamw_update_finite_nonzero": True,
            "fp32_centered_all_slot_delta_gate": True,
            "live_source_state_unchanged": True,
            "live_signed_x_state_unchanged": True,
            "rng_state_unchanged": True,
        },
        "config_hash": config_hash(config, length=64),
        "contract": contract,
        "adamw_contract": optimizer_contract,
        "source_provenance": current_provenance,
        "source_checkpoint": str(source),
        "source_checkpoint_epoch": 4,
        "source_artifact_hashes": source_artifacts,
        "frozen_state_hashes": {
            "scene_state_sha256": expected["source_scene_state_sha256"],
            "global_scene_residual_state_sha256": expected_global_hash,
            "lora_bank_state_sha256": expected["source_lora_bank_state_sha256"],
            "combined_source_state_sha256": combined_source_hash,
        },
        "source_hashes": {
            **source_artifacts,
            "scene_state_sha256": expected["source_scene_state_sha256"],
            "global_scene_residual_state_sha256": expected_global_hash,
            "lora_bank_state_sha256": expected["source_lora_bank_state_sha256"],
        },
        "source_metadata_global_residual_state_sha256": expected_global_hash,
        "source_metadata_lora_bank_state_sha256": expected["source_lora_bank_state_sha256"],
        "initial_signed_x_state_sha256": initial_signed_hash,
        "live_source_state_sha256_before": combined_source_hash,
        "live_source_state_sha256_after": combined_source_hash,
        "live_source_state_unchanged": True,
        "live_signed_x_state_sha256_before": initial_signed_hash,
        "live_signed_x_state_sha256_after": initial_signed_hash,
        "live_signed_x_state_unchanged": True,
        "selection_sha256": expected["selection_sha256"],
        "pair_membership_sha256": expected["pair_membership_sha256"],
        "pair_unit_selection_sha256": PAIR_UNIT_SHA256,
        "ordered_unit_sha256": expected["ordered_unit_sha256"],
        "zero_output_prefix_equivalence": zero_equivalence,
        "signed_x_structural_state": signed.validate_structural_state(),
        "microsteps": microsteps,
        "microstep_losses": microsteps,
        "pair_gradient_audit": {
            "color_total_loss_exact_zero": True,
            "color_gradient_exact_zero": True,
            "mirror_gradient_positive_finite": True,
        },
        "gradient": {
            "ordered_microstep_count": 12,
            "accumulated_finite_nonzero": True,
            "predicted_update_nonzero_count": predicted["nonzero_update_count"],
            "changed_parameter_keys": ["output_projection.weight"],
            "predicted_signed_x_state_sha256": predicted_signed_hash,
            "predicted_output_projection_sha256": predicted_output_hash,
            "optimizer_state_manifest": optimizer_manifest,
            "optimizer_state_sha256": optimizer_hash,
        },
        "predicted_output_weight_sha256": predicted_output_hash,
        "predicted_signed_x_scene_residual_state_sha256": predicted_signed_hash,
        "predicted_canonical_adamw_state_manifest": optimizer_manifest,
        "predicted_canonical_adamw_state_sha256": optimizer_hash,
        "structural_gate": {"passed": True},
        "rng_state": {
            "all_available_domains_unchanged": True,
            "restored_after_mismatch": False,
        },
    }
    preflight_path = tmp_path / "v19_preflight.json"
    _write_json(preflight_path, preflight)

    checkpoint = tmp_path / "checkpoints" / "v19" / "epoch_001"
    checkpoint.mkdir(parents=True)
    save_file(
        {**global_tensors, **scene_tensors, **lora_tensors, **signed_state},
        checkpoint / "adapter.safetensors",
    )
    history_row = {
        "epoch": 1,
        "train_loss": 0.5,
        "pair_batch_count": 12,
        "pair_batch_fraction": 1.0,
        "pair_candidate_gate": {"evaluation_type": "synthetic_teacher_gate"},
    }
    metadata = {
        "schema_version": 3,
        "epoch": 1,
        "optimizer_step": 1,
        "global_step": 12,
        "history": [history_row],
        "train_loss": history_row["train_loss"],
        "pair_candidate_gate": history_row["pair_candidate_gate"],
        "config_hash": config_hash(config),
        "output_namespace": training["output_namespace"],
        "gradient_accumulation": 12,
        "freeze_scene_adapter": True,
        "train_global_scene_residual_only": False,
        "train_signed_x_scene_residual_only": True,
        "question_dependent_scene_processing": False,
        "scene_latents": 256,
        "language_hidden_dim": 1536,
        "max_questions_per_scene": 6,
        "counterfactual_pair_unit_count": 12,
        "counterfactual_pair_unit_selection_sha256": PAIR_UNIT_SHA256,
        "training_counterfactual_pair_count": 2,
        "training_counterfactual_pair_membership_sha256": expected["pair_membership_sha256"],
        "source_provenance": current_provenance,
        "global_scene_residual": global_scene_residual_settings(config).contract(),
        "global_scene_residual_parameter_count": 400_128,
        "global_scene_residual_initial_state_sha256": global_scene_residual_settings(
            config
        ).expected_initial_state_sha256,
        "global_scene_residual_state_sha256": expected_global_hash,
        "frozen_global_scene_residual_state_sha256": expected_global_hash,
        "global_scene_residual_zero_output_equivalence": None,
        "signed_x_scene_residual": signed_x_scene_residual_settings(config).contract(),
        "signed_x_scene_residual_parameter_count": 196_608,
        "signed_x_scene_residual_initial_state_sha256": initial_signed_hash,
        "signed_x_scene_residual_state_sha256": predicted_signed_hash,
        "signed_x_scene_residual_zero_output_equivalence": zero_equivalence,
        "frozen_scene_state_sha256": expected["source_scene_state_sha256"],
        "frozen_lora_bank_state_sha256": expected["source_lora_bank_state_sha256"],
        "lora_bank_state_sha256": expected["source_lora_bank_state_sha256"],
        "lora_trainable_parameter_count": 0,
        "initialize_expected_adapter_sha256": source_artifacts["adapter_sha256"],
        "initialize_expected_metadata_sha256": source_artifacts["metadata_sha256"],
        "initialize_expected_global_scene_residual_state_sha256": expected_global_hash,
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
            "optimizer_state_loaded": False,
            "history_loaded": False,
            "source_global_scene_residual_state_sha256": expected_global_hash,
            "expected_source_global_scene_residual_state_sha256": expected_global_hash,
            "global_scene_residual_frozen": True,
            "signed_x_scene_residual_initial_state_sha256": initial_signed_hash,
            "signed_x_scene_residual_zero_output": True,
        },
    }
    _write_json(checkpoint / "metadata.json", metadata)
    torch.save(optimizer.state_dict(), checkpoint / "optimizer.pt")
    return SyntheticBundle(
        config=config,
        preflight_path=preflight_path,
        checkpoint=checkpoint,
        provenance=current_provenance,
        predicted_signed_hash=predicted_signed_hash,
        predicted_output_hash=predicted_output_hash,
        optimizer_hash=optimizer_hash,
        expected_global_hash=expected_global_hash,
    )


def test_exact_update_and_optimizer_match_authorizes_resume(
    bundle: SyntheticBundle, monkeypatch: pytest.MonkeyPatch
) -> None:
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
    assert report["scene_map_loaded"] is False
    assert report["oracle_loaded"] is False
    assert calls == [{"weights_only": True, "map_location": "cpu"}]
    assert report["signed_x_state_sha256"] == bundle.predicted_signed_hash
    assert report["output_projection_sha256"] == bundle.predicted_output_hash
    assert report["optimizer_state_sha256"] == bundle.optimizer_hash
    assert report["frozen_global_scene_residual_state_sha256"] == bundle.expected_global_hash


@pytest.mark.parametrize("location", ["preflight", "checkpoint"])
def test_source_provenance_mismatch_is_rejected(bundle: SyntheticBundle, location: str) -> None:
    path = bundle.preflight_path if location == "preflight" else bundle.checkpoint / "metadata.json"
    value = _read_json(path)
    value["source_provenance"] = _provenance("different")
    _write_json(path, value)

    with pytest.raises(V19Update1Violation, match="provenance"):
        verify_update1(bundle.config, bundle.preflight_path, bundle.checkpoint)


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("schema_version", 2, "preflight.schema_version"),
        ("audit_type", "obsolete-role", "preflight.audit_type"),
        ("authorized", False, "preflight.authorized"),
        ("structural_authorization", False, "structural_authorization"),
    ],
)
def test_preflight_schema_or_authorization_tamper_is_rejected(
    bundle: SyntheticBundle, field: str, value: Any, match: str
) -> None:
    preflight = _read_json(bundle.preflight_path)
    preflight[field] = value
    _write_json(bundle.preflight_path, preflight)

    with pytest.raises(V19Update1Violation, match=match):
        verify_update1(bundle.config, bundle.preflight_path, bundle.checkpoint)


def test_rehashed_signed_tensor_tamper_is_rejected_by_prediction(
    bundle: SyntheticBundle,
) -> None:
    adapter = bundle.checkpoint / "adapter.safetensors"
    tensors = load_file(adapter, device="cpu")
    weight_key = "signed_x_scene_residual.output_projection.weight"
    tensors[weight_key] = tensors[weight_key].clone()
    tensors[weight_key][0, 0] += 0.25
    save_file(tensors, adapter)
    signed = {
        key: value for key, value in tensors.items() if key.startswith("signed_x_scene_residual.")
    }
    metadata = _read_json(bundle.checkpoint / "metadata.json")
    metadata["signed_x_scene_residual_state_sha256"] = tensor_state_sha256(signed)
    _write_json(bundle.checkpoint / "metadata.json", metadata)

    with pytest.raises(V19Update1Violation, match="predicted/actual signed-X state"):
        verify_update1(bundle.config, bundle.preflight_path, bundle.checkpoint)


@pytest.mark.parametrize("case", ["moment", "step", "group"])
def test_optimizer_state_tamper_is_rejected(bundle: SyntheticBundle, case: str) -> None:
    optimizer_path = bundle.checkpoint / "optimizer.pt"
    state = torch.load(optimizer_path, weights_only=True, map_location="cpu")
    if case == "moment":
        state["state"][0]["exp_avg"] = state["state"][0]["exp_avg"].clone()
        state["state"][0]["exp_avg"][0, 0] += 0.125
    elif case == "step":
        state["state"][0]["step"] = torch.tensor(2.0)
    else:
        state["param_groups"][0]["name"] = "wrong"
    torch.save(state, optimizer_path)

    with pytest.raises(V19Update1Violation, match="optimizer|Optimizer|AdamW"):
        verify_update1(bundle.config, bundle.preflight_path, bundle.checkpoint)


def test_preflight_optimizer_hash_tamper_is_rejected(bundle: SyntheticBundle) -> None:
    preflight = _read_json(bundle.preflight_path)
    wrong = _digest("wrong optimizer state")
    preflight["gradient"]["optimizer_state_sha256"] = wrong
    preflight["predicted_canonical_adamw_state_sha256"] = wrong
    _write_json(bundle.preflight_path, preflight)

    with pytest.raises(V19Update1Violation, match="optimizer manifest hash"):
        verify_update1(bundle.config, bundle.preflight_path, bundle.checkpoint)


def test_preflight_pair_unit_hash_must_match_trainer_contract(
    bundle: SyntheticBundle,
) -> None:
    preflight = _read_json(bundle.preflight_path)
    preflight["pair_unit_selection_sha256"] = _digest("wrong pair-unit schema")
    _write_json(bundle.preflight_path, preflight)

    with pytest.raises(V19Update1Violation, match="preflight pair-unit hash"):
        verify_update1(bundle.config, bundle.preflight_path, bundle.checkpoint)


def test_frozen_global_tensor_tamper_is_rejected(bundle: SyntheticBundle) -> None:
    adapter = bundle.checkpoint / "adapter.safetensors"
    tensors = load_file(adapter, device="cpu")
    key = "global_scene_residual.output_projection.weight"
    tensors[key] = tensors[key].clone()
    tensors[key][0, 0] += 0.25
    save_file(tensors, adapter)

    with pytest.raises(V19Update1Violation, match="global residual metadata hash"):
        verify_update1(bundle.config, bundle.preflight_path, bundle.checkpoint)


@pytest.mark.parametrize(
    ("key", "match"),
    [
        ("scene_model.synthetic_weight", "frozen scene tensor hash"),
        (
            "lora_banks.extension_v13.synthetic_weight",
            "frozen LoRA tensor hashes",
        ),
    ],
)
def test_frozen_scene_or_lora_tensor_tamper_is_rejected(
    bundle: SyntheticBundle, key: str, match: str
) -> None:
    adapter = bundle.checkpoint / "adapter.safetensors"
    tensors = load_file(adapter, device="cpu")
    tensors[key] = tensors[key].clone()
    tensors[key].reshape(-1)[0] += 0.25
    save_file(tensors, adapter)

    with pytest.raises(V19Update1Violation, match=match):
        verify_update1(bundle.config, bundle.preflight_path, bundle.checkpoint)


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("epoch", 2, "checkpoint.epoch"),
        ("optimizer_step", 2, "checkpoint.optimizer_step"),
        ("global_step", 13, "checkpoint.global_step"),
    ],
)
def test_wrong_epoch_or_step_is_rejected(
    bundle: SyntheticBundle, field: str, value: int, match: str
) -> None:
    metadata_path = bundle.checkpoint / "metadata.json"
    metadata = _read_json(metadata_path)
    metadata[field] = value
    _write_json(metadata_path, metadata)

    with pytest.raises(V19Update1Violation, match=match):
        verify_update1(bundle.config, bundle.preflight_path, bundle.checkpoint)


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        (lambda history: history.append(copy.deepcopy(history[0])), "exactly epoch one"),
        (lambda history: history[0].update(epoch=2), "history epoch"),
        (lambda history: history[0].update(pair_batch_count=11), "history pair batches"),
        (lambda history: history[0].update(pair_batch_fraction=0.5), "history pair fraction"),
    ],
)
def test_history_tamper_is_rejected(bundle: SyntheticBundle, mutation, match: str) -> None:
    metadata_path = bundle.checkpoint / "metadata.json"
    metadata = _read_json(metadata_path)
    mutation(metadata["history"])
    _write_json(metadata_path, metadata)

    with pytest.raises(V19Update1Violation, match=match):
        verify_update1(bundle.config, bundle.preflight_path, bundle.checkpoint)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("adapter_sha256", _digest("wrong source adapter")),
        ("metadata_sha256", _digest("wrong source metadata")),
        ("expected_adapter_sha256", _digest("wrong expected adapter")),
        ("expected_metadata_sha256", _digest("wrong expected metadata")),
        ("source_global_scene_residual_state_sha256", _digest("wrong source global")),
        (
            "expected_source_global_scene_residual_state_sha256",
            _digest("wrong expected global"),
        ),
    ],
)
def test_initialization_provenance_pin_tamper_is_rejected(
    bundle: SyntheticBundle, field: str, value: str
) -> None:
    metadata_path = bundle.checkpoint / "metadata.json"
    metadata = _read_json(metadata_path)
    metadata["initialization_provenance"][field] = value
    _write_json(metadata_path, metadata)

    with pytest.raises(V19Update1Violation, match=f"initialization provenance {field}"):
        verify_update1(bundle.config, bundle.preflight_path, bundle.checkpoint)


@pytest.mark.parametrize(
    ("case", "match"),
    [
        ("contract", "signed-X contract"),
        ("initial_hash", "initial signed-X hash"),
        ("anchor", "signed-X structural state"),
    ],
)
def test_signed_x_contract_or_anchor_tamper_is_rejected(
    bundle: SyntheticBundle, case: str, match: str
) -> None:
    if case == "anchor":
        adapter = bundle.checkpoint / "adapter.safetensors"
        tensors = load_file(adapter, device="cpu")
        key = "signed_x_scene_residual.signed_x_anchors"
        tensors[key] = tensors[key].clone()
        tensors[key][0] += 0.25
        save_file(tensors, adapter)
        metadata = _read_json(bundle.checkpoint / "metadata.json")
        signed = {
            key: value
            for key, value in tensors.items()
            if key.startswith("signed_x_scene_residual.")
        }
        metadata["signed_x_scene_residual_state_sha256"] = tensor_state_sha256(signed)
        _write_json(bundle.checkpoint / "metadata.json", metadata)
    else:
        metadata_path = bundle.checkpoint / "metadata.json"
        metadata = _read_json(metadata_path)
        if case == "contract":
            metadata["signed_x_scene_residual"]["architecture_version"] = "wrong"
        else:
            metadata["signed_x_scene_residual_initial_state_sha256"] = _digest("wrong initial")
        _write_json(metadata_path, metadata)

    with pytest.raises(V19Update1Violation, match=match):
        verify_update1(bundle.config, bundle.preflight_path, bundle.checkpoint)


@pytest.mark.parametrize("field", ["frozen_scene_state_sha256", "frozen_lora_bank_state_sha256"])
def test_frozen_source_metadata_tamper_is_rejected(bundle: SyntheticBundle, field: str) -> None:
    metadata_path = bundle.checkpoint / "metadata.json"
    metadata = _read_json(metadata_path)
    metadata[field] = _digest(f"wrong-{field}")
    _write_json(metadata_path, metadata)

    with pytest.raises(V19Update1Violation, match="frozen"):
        verify_update1(bundle.config, bundle.preflight_path, bundle.checkpoint)
