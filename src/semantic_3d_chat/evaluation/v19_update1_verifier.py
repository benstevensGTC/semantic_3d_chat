"""Report-only verifier for V19's separately executed first optimizer update."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import pickle
import re
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
from safetensors import SafetensorError
from safetensors.torch import load_file

from semantic_3d_chat.config import PROJECT_ROOT, config_hash, load_config
from semantic_3d_chat.evaluation.v19_optimizer_state import (
    V19AdamWStateViolation,
    canonical_v19_adamw_state,
    validate_v19_adamw_state_manifest,
)
from semantic_3d_chat.evaluation.v19_structural_preflight import (
    V19_PREFLIGHT_ROLE,
    V19StructuralPreflightViolation,
    validate_v19_config_contract,
)
from semantic_3d_chat.language.lora import tensor_state_sha256
from semantic_3d_chat.scene_encoder.global_residual import global_scene_residual_settings
from semantic_3d_chat.scene_encoder.signed_x_residual import (
    SignedXSceneResidual,
    signed_x_scene_residual_settings,
)
from semantic_3d_chat.training.source_provenance import (
    capture_git_source_provenance,
    require_clean_committed_source,
)
from semantic_3d_chat.training.train_adapter import file_sha256

UPDATE1_VERIFIER_TYPE = "v19_exact_update1_match_verifier"
_SHA256 = re.compile(r"[0-9a-f]{64}")
_SHORT_SHA = re.compile(r"[0-9a-f]{12}")
_SIGNED_PREFIX = "signed_x_scene_residual."
_GLOBAL_PREFIX = "global_scene_residual."
_SCENE_PREFIXES = ("scene_model.", "composer.", "grounding.")
_LORA_BANK_PREFIX = "lora_banks."
_EMPTY_SHA256 = hashlib.sha256(b"").hexdigest()


class V19Update1Violation(ValueError):
    """A mismatch that denies V19 stage-two resume."""


def _fail(message: str) -> None:
    raise V19Update1Violation(message)


def _mapping(value: Any, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        _fail(f"{field} must be an object")
    return value


def _sha256(value: Any, field: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        _fail(f"{field} must be a lowercase SHA-256 digest")
    return value


def _equal(observed: Any, expected: Any, field: str) -> None:
    if observed != expected:
        _fail(f"{field} mismatch: expected={expected!r} observed={observed!r}")


def _exact_int(observed: Any, expected: int, field: str) -> None:
    if isinstance(observed, bool) or not isinstance(observed, int) or observed != expected:
        _fail(f"{field} mismatch: expected={expected} observed={observed!r}")


def _positive_int(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        _fail(f"{field} must be a positive integer")
    return value


def _resolve(value: str | Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def _display_path(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def _read_json(path: Path, field: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        _fail(f"Cannot read {field} JSON at {path}: {error}")
    return dict(_mapping(value, field))


def _clean_provenance(value: Any, field: str) -> dict[str, Any]:
    provenance = dict(_mapping(value, field))
    try:
        require_clean_committed_source(provenance)
    except RuntimeError as error:
        _fail(f"{field} is not clean committed source provenance: {error}")
    _equal(
        provenance.get("tracked_diff_sha256"),
        _EMPTY_SHA256,
        f"{field}.tracked_diff_sha256",
    )
    return provenance


def _finite(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        _fail(f"{field} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        _fail(f"{field} must be finite")
    return result


def _validate_zero_equivalence(value: Any) -> dict[str, Any]:
    equivalence = dict(_mapping(value, "preflight signed-X zero equivalence"))
    for key, expected in {
        "verified": True,
        "base": "loaded_frozen_global_scene_residual",
        "question_dependent_scene_processing": False,
        "all_scene_slots_accounted": True,
        "scene_count": 4,
    }.items():
        _equal(equivalence.get(key), expected, f"preflight zero equivalence {key}")
    prefixes = _mapping(
        equivalence.get("scene_prefixes"),
        "preflight zero equivalence scene prefixes",
    )
    if len(prefixes) != 4:
        _fail("Preflight zero equivalence must contain exactly four scenes")
    for scene_id, raw in prefixes.items():
        if not isinstance(scene_id, str) or not scene_id:
            _fail("Preflight zero equivalence has an invalid scene ID")
        row = _mapping(raw, f"preflight zero equivalence {scene_id}")
        if set(row) != {
            "v18_base_prefix_sha256",
            "signed_x_adapted_prefix_sha256",
        }:
            _fail(f"Preflight zero equivalence {scene_id} keys mismatch")
        base = _sha256(
            row.get("v18_base_prefix_sha256"),
            f"preflight zero equivalence {scene_id} base",
        )
        adapted = _sha256(
            row.get("signed_x_adapted_prefix_sha256"),
            f"preflight zero equivalence {scene_id} adapted",
        )
        _equal(adapted, base, f"preflight zero equivalence {scene_id} identity")
    return equivalence


def _validate_preflight(
    config: dict[str, Any],
    preflight: Mapping[str, Any],
    current_provenance: Mapping[str, Any],
) -> dict[str, Any]:
    try:
        contract = validate_v19_config_contract(config)
    except (TypeError, ValueError, RuntimeError, V19StructuralPreflightViolation) as error:
        _fail(f"V19 config contract is invalid: {error}")
    _equal(preflight.get("schema_version"), 1, "preflight.schema_version")
    _equal(preflight.get("audit_type"), V19_PREFLIGHT_ROLE, "preflight.audit_type")
    _equal(preflight.get("authorized"), True, "preflight.authorized")
    _equal(
        preflight.get("structural_authorization"),
        True,
        "preflight.structural_authorization",
    )
    for key, expected in {
        "runtime_eligible": False,
        "uses_supervised_qa_metadata": True,
        "question_dependent_scene_processing": False,
        "live_optimizer_constructed": False,
        "live_optimizer_step_executed": False,
        "optimizer_steps": 0,
        "isolated_clone_optimizer_constructed": True,
        "isolated_clone_optimizer_steps": 1,
    }.items():
        _equal(preflight.get(key), expected, f"preflight.{key}")
    authorization_checks = _mapping(
        preflight.get("authorization_checks"),
        "preflight authorization checks",
    )
    expected_authorization_checks = {
        "source_and_config_contracts_passed",
        "exact_selection_and_order_passed",
        "step_zero_identity_all_scenes",
        "color_losses_exactly_zero",
        "color_isolated_signed_x_gradient_exactly_zero",
        "mirror_signed_x_gradient_finite_nonzero",
        "accumulated_signed_x_gradient_finite_nonzero",
        "only_signed_x_output_weight_has_gradient",
        "predicted_adamw_update_finite_nonzero",
        "fp32_centered_all_slot_delta_gate",
        "live_source_state_unchanged",
        "live_signed_x_state_unchanged",
        "rng_state_unchanged",
    }
    if set(authorization_checks) != expected_authorization_checks:
        _fail("Preflight authorization-check schema mismatch")
    if any(value is not True for value in authorization_checks.values()):
        _fail("Every preflight authorization check must be exactly true")
    _equal(preflight.get("config_hash"), config_hash(config, length=64), "preflight.config_hash")
    _equal(preflight.get("contract"), contract, "preflight.contract")
    provenance = _clean_provenance(preflight.get("source_provenance"), "preflight provenance")
    _equal(provenance, dict(current_provenance), "current/preflight provenance")

    training = _mapping(config.get("training"), "training")
    source = _resolve(str(training["initialize_from"]))
    _equal(preflight.get("source_checkpoint"), _display_path(source), "preflight source")
    _exact_int(preflight.get("source_checkpoint_epoch"), 4, "preflight source epoch")
    source_hashes = dict(
        _mapping(preflight.get("source_artifact_hashes"), "source artifact hashes")
    )
    for key, filename, config_key in (
        ("adapter_sha256", "adapter.safetensors", "initialize_expected_adapter_sha256"),
        ("metadata_sha256", "metadata.json", "initialize_expected_metadata_sha256"),
    ):
        observed = file_sha256(source / filename)
        expected = _sha256(training.get(config_key), f"training.{config_key}")
        _equal(observed, expected, f"source {key}")
        _equal(source_hashes.get(key), expected, f"preflight source {key}")
    source_metadata = _read_json(source / "metadata.json", "V18 source metadata")
    _exact_int(source_metadata.get("epoch"), 4, "V18 source epoch")

    expected_hashes = _mapping(contract.get("expected_hashes"), "V19 expected hashes")
    expected_scene = _sha256(
        expected_hashes.get("source_scene_state_sha256"),
        "configured source scene state",
    )
    expected_global = _sha256(
        training.get("initialize_expected_global_scene_residual_state_sha256"),
        "configured source global residual",
    )
    _equal(
        expected_hashes.get("source_global_scene_residual_state_sha256"),
        expected_global,
        "contract/configured source global residual",
    )
    expected_lora = dict(
        _mapping(
            expected_hashes.get("source_lora_bank_state_sha256"),
            "configured source LoRA states",
        )
    )
    _equal(
        source_metadata.get("global_scene_residual_state_sha256"),
        expected_global,
        "V18 source metadata global residual state",
    )
    _equal(
        source_metadata.get("lora_bank_state_sha256"),
        expected_lora,
        "V18 source metadata LoRA states",
    )
    frozen_hashes = _mapping(preflight.get("frozen_state_hashes"), "frozen state hashes")
    for key, expected in {
        "scene_state_sha256": expected_scene,
        "global_scene_residual_state_sha256": expected_global,
        "lora_bank_state_sha256": expected_lora,
    }.items():
        _equal(frozen_hashes.get(key), expected, f"preflight frozen {key}")
    _sha256(frozen_hashes.get("combined_source_state_sha256"), "combined frozen source state")
    observed_source_hashes = _mapping(preflight.get("source_hashes"), "preflight source hashes")
    _equal(
        observed_source_hashes.get("adapter_sha256"),
        source_hashes["adapter_sha256"],
        "source adapter duplicate",
    )
    _equal(
        observed_source_hashes.get("metadata_sha256"),
        source_hashes["metadata_sha256"],
        "source metadata duplicate",
    )
    _equal(observed_source_hashes.get("scene_state_sha256"), expected_scene, "source scene state")
    _equal(
        observed_source_hashes.get("global_scene_residual_state_sha256"),
        expected_global,
        "source global residual state",
    )
    _equal(observed_source_hashes.get("lora_bank_state_sha256"), expected_lora, "source LoRA state")
    _equal(
        preflight.get("source_metadata_global_residual_state_sha256"),
        expected_global,
        "source metadata global residual state",
    )
    _equal(
        preflight.get("source_metadata_lora_bank_state_sha256"),
        expected_lora,
        "source metadata LoRA state",
    )

    signed_settings = signed_x_scene_residual_settings(config)
    initial_signed = _sha256(
        signed_settings.expected_initial_state_sha256,
        "configured initial signed-X state",
    )
    for key in (
        "initial_signed_x_state_sha256",
        "live_signed_x_state_sha256_before",
        "live_signed_x_state_sha256_after",
    ):
        _equal(preflight.get(key), initial_signed, f"preflight.{key}")
    _equal(preflight.get("live_signed_x_state_unchanged"), True, "preflight live signed-X state")
    _equal(preflight.get("live_source_state_unchanged"), True, "preflight live source state")
    _equal(
        preflight.get("live_source_state_sha256_before"),
        frozen_hashes.get("combined_source_state_sha256"),
        "preflight live source before",
    )
    _equal(
        preflight.get("live_source_state_sha256_after"),
        frozen_hashes.get("combined_source_state_sha256"),
        "preflight live source after",
    )
    selection_hash = _sha256(preflight.get("selection_sha256"), "preflight selection hash")
    _equal(selection_hash, expected_hashes.get("selection_sha256"), "preflight selection hash")
    pair_membership_hash = _sha256(
        preflight.get("pair_membership_sha256"), "preflight pair-membership hash"
    )
    _equal(
        pair_membership_hash,
        expected_hashes.get("pair_membership_sha256"),
        "preflight pair-membership hash",
    )
    pair_unit_hash = _sha256(
        preflight.get("pair_unit_selection_sha256"), "preflight pair-unit hash"
    )
    _equal(
        pair_unit_hash,
        expected_hashes.get("pair_unit_selection_sha256"),
        "preflight pair-unit hash",
    )
    ordered_hash = _sha256(preflight.get("ordered_unit_sha256"), "preflight ordered-unit hash")
    _equal(ordered_hash, expected_hashes.get("ordered_unit_sha256"), "preflight ordered-unit hash")

    microsteps = preflight.get("microsteps")
    if not isinstance(microsteps, list) or len(microsteps) != 12:
        _fail("Preflight must contain exactly twelve ordered microsteps")
    _equal(preflight.get("microstep_losses"), microsteps, "preflight microstep duplicate")
    for expected_index, microstep in enumerate(microsteps, start=1):
        row = _mapping(microstep, f"preflight microstep {expected_index}")
        _exact_int(row.get("microstep"), expected_index, f"preflight microstep {expected_index}")
        _finite(row.get("total_loss"), f"preflight microstep {expected_index} total loss")

    zero_equivalence = _validate_zero_equivalence(preflight.get("zero_output_prefix_equivalence"))

    optimizer_contract = dict(_mapping(training.get("optimizer"), "training.optimizer"))
    _equal(
        dict(_mapping(preflight.get("adamw_contract"), "preflight AdamW contract")),
        optimizer_contract,
        "preflight AdamW contract",
    )
    structural_gate = _mapping(preflight.get("structural_gate"), "preflight structural gate")
    _equal(structural_gate.get("passed"), True, "preflight structural gate")
    rng_state = _mapping(preflight.get("rng_state"), "preflight RNG state")
    _equal(
        rng_state.get("all_available_domains_unchanged"),
        True,
        "preflight RNG state",
    )
    _equal(rng_state.get("restored_after_mismatch"), False, "preflight RNG restoration")
    signed_structure = _mapping(
        preflight.get("signed_x_structural_state"),
        "preflight signed-X structural state",
    )
    for key, expected in {
        "architecture_version": signed_settings.architecture_version,
        "scene_dim": 1536,
        "latent_count": 256,
        "content_dim": 128,
        "parameter_count": 196_608,
        "accounted_slot_count": 256,
        "all_slots_accounted": True,
        "spatial_centering": "all_slots_fp32",
        "trainable_surface": "bias_free_output_projection_only",
    }.items():
        _equal(signed_structure.get(key), expected, f"preflight signed-X structure {key}")
    pair_gradient = _mapping(preflight.get("pair_gradient_audit"), "pair gradient audit")
    _equal(pair_gradient.get("color_total_loss_exact_zero"), True, "color zero loss")
    _equal(pair_gradient.get("color_gradient_exact_zero"), True, "color zero gradient")
    _equal(pair_gradient.get("mirror_gradient_positive_finite"), True, "mirror gradient")

    gradient = _mapping(preflight.get("gradient"), "preflight gradient")
    _equal(
        gradient.get("changed_parameter_keys"),
        ["output_projection.weight"],
        "preflight changed parameters",
    )
    predicted_state = _sha256(
        gradient.get("predicted_signed_x_state_sha256"),
        "predicted signed-X state",
    )
    predicted_output = _sha256(
        gradient.get("predicted_output_projection_sha256"),
        "predicted output weight",
    )
    if predicted_state == initial_signed:
        _fail("Preflight predicted signed-X state did not change")
    _equal(
        gradient.get("ordered_microstep_count"),
        12,
        "preflight gradient microstep count",
    )
    _equal(gradient.get("accumulated_finite_nonzero"), True, "preflight accumulated gradient")
    _positive_int(
        gradient.get("predicted_update_nonzero_count"),
        "preflight predicted nonzero update count",
    )
    optimizer_manifest = dict(
        _mapping(gradient.get("optimizer_state_manifest"), "preflight optimizer manifest")
    )
    try:
        calculated_optimizer_hash = validate_v19_adamw_state_manifest(
            optimizer_manifest,
            optimizer_contract,
        )
    except V19AdamWStateViolation as error:
        _fail(f"Preflight optimizer manifest is invalid: {error}")
    optimizer_hash = _sha256(gradient.get("optimizer_state_sha256"), "preflight optimizer hash")
    _equal(calculated_optimizer_hash, optimizer_hash, "preflight optimizer manifest hash")
    _equal(
        preflight.get("predicted_output_weight_sha256"),
        predicted_output,
        "preflight top-level predicted output",
    )
    _equal(
        preflight.get("predicted_signed_x_scene_residual_state_sha256"),
        predicted_state,
        "preflight top-level predicted signed-X state",
    )
    _equal(
        preflight.get("predicted_canonical_adamw_state_manifest"),
        optimizer_manifest,
        "preflight top-level optimizer manifest",
    )
    _equal(
        preflight.get("predicted_canonical_adamw_state_sha256"),
        optimizer_hash,
        "preflight top-level optimizer hash",
    )
    return {
        "source": source,
        "source_metadata": source_metadata,
        "source_artifact_hashes": source_hashes,
        "source_provenance": provenance,
        "frozen_state_hashes": dict(frozen_hashes),
        "expected_scene_state_sha256": expected_scene,
        "expected_global_state_sha256": expected_global,
        "expected_lora_state_sha256": expected_lora,
        "optimizer_contract": optimizer_contract,
        "optimizer_manifest": optimizer_manifest,
        "optimizer_hash": optimizer_hash,
        "predicted_signed_x_state_sha256": predicted_state,
        "predicted_output_projection_sha256": predicted_output,
        "zero_equivalence": zero_equivalence,
        "pair_unit_selection_sha256": pair_unit_hash,
        "pair_membership_sha256": pair_membership_hash,
    }


def _load_tensor_evidence(
    adapter_path: Path,
    metadata: Mapping[str, Any],
    *,
    expected_frozen_scene_sha256: str,
    expected_frozen_global_sha256: str,
    expected_frozen_lora_sha256: Mapping[str, str],
) -> dict[str, Any]:
    try:
        tensors = load_file(adapter_path, device="cpu")
    except (OSError, RuntimeError, ValueError, SafetensorError) as error:
        _fail(f"Cannot read checkpoint adapter tensors: {error}")
    signed = {key: value for key, value in tensors.items() if key.startswith(_SIGNED_PREFIX)}
    expected_signed = {
        f"{_SIGNED_PREFIX}signed_x_anchors": ((256,), torch.float32),
        f"{_SIGNED_PREFIX}output_projection.weight": ((1536, 128), torch.float32),
    }
    _equal(set(signed), set(expected_signed), "checkpoint signed-X tensor keys")
    for key, (shape, dtype) in expected_signed.items():
        _equal(tuple(signed[key].shape), shape, f"checkpoint shape {key}")
        _equal(signed[key].dtype, dtype, f"checkpoint dtype {key}")
        if not bool(torch.isfinite(signed[key]).all()):
            _fail(f"Checkpoint tensor is nonfinite: {key}")
    signed_module = SignedXSceneResidual(
        scene_dim=1536,
        latent_count=256,
        content_dim=128,
    )
    try:
        signed_module.load_state_dict(
            {key[len(_SIGNED_PREFIX) :]: value for key, value in signed.items()},
            strict=True,
        )
        signed_structure = signed_module.validate_structural_state()
    except (RuntimeError, TypeError, ValueError) as error:
        _fail(f"Checkpoint signed-X structural state is invalid: {error}")
    _equal(signed_structure.get("parameter_count"), 196_608, "checkpoint signed-X parameter count")
    _equal(signed_structure.get("accounted_slot_count"), 256, "checkpoint signed-X slot coverage")
    _equal(
        signed_structure.get("all_slots_accounted"), True, "checkpoint signed-X all-slot coverage"
    )
    signed_hash = tensor_state_sha256(signed)
    _equal(
        signed_hash,
        metadata.get("signed_x_scene_residual_state_sha256"),
        "checkpoint signed-X metadata hash",
    )
    output_hash = tensor_state_sha256(
        {
            f"{_SIGNED_PREFIX}output_projection.weight": signed[
                f"{_SIGNED_PREFIX}output_projection.weight"
            ]
        }
    )

    global_state = {key: value for key, value in tensors.items() if key.startswith(_GLOBAL_PREFIX)}
    if not global_state or any(
        not bool(torch.isfinite(value).all()) for value in global_state.values()
    ):
        _fail("Checkpoint global residual tensor subset is missing or nonfinite")
    global_hash = tensor_state_sha256(global_state)
    _equal(
        global_hash,
        metadata.get("global_scene_residual_state_sha256"),
        "checkpoint global residual metadata hash",
    )
    _equal(
        global_hash,
        metadata.get("frozen_global_scene_residual_state_sha256"),
        "checkpoint frozen global residual hash",
    )
    _equal(
        global_hash,
        expected_frozen_global_sha256,
        "checkpoint/configured frozen global residual hash",
    )

    scene_state = {
        key: value
        for key, value in tensors.items()
        if any(key.startswith(prefix) for prefix in _SCENE_PREFIXES)
    }
    if not scene_state or any(
        not bool(torch.isfinite(value).all()) for value in scene_state.values()
    ):
        _fail("Checkpoint frozen scene tensor subset is missing or nonfinite")
    scene_hash = tensor_state_sha256(scene_state)
    _equal(
        scene_hash,
        metadata.get("frozen_scene_state_sha256"),
        "checkpoint frozen scene tensor hash",
    )
    _equal(
        scene_hash,
        expected_frozen_scene_sha256,
        "checkpoint/configured frozen scene tensor hash",
    )

    expected_lora = dict(expected_frozen_lora_sha256)
    observed_lora: dict[str, str] = {}
    consumed_lora_keys: set[str] = set()
    for bank_name in sorted(expected_lora):
        prefix = f"{_LORA_BANK_PREFIX}{bank_name}."
        bank_state = {
            key[len(prefix) :]: value for key, value in tensors.items() if key.startswith(prefix)
        }
        if not bank_state or any(
            not bool(torch.isfinite(value).all()) for value in bank_state.values()
        ):
            _fail(f"Checkpoint frozen LoRA bank {bank_name!r} is missing or nonfinite")
        consumed_lora_keys.update(key for key in tensors if key.startswith(prefix))
        observed_lora[bank_name] = tensor_state_sha256(bank_state)
    all_lora_keys = {key for key in tensors if key.startswith(_LORA_BANK_PREFIX)}
    _equal(all_lora_keys, consumed_lora_keys, "checkpoint frozen LoRA tensor keys")
    _equal(
        observed_lora,
        metadata.get("frozen_lora_bank_state_sha256"),
        "checkpoint frozen LoRA tensor hashes",
    )
    _equal(
        observed_lora,
        metadata.get("lora_bank_state_sha256"),
        "checkpoint LoRA metadata tensor hashes",
    )
    _equal(observed_lora, expected_lora, "checkpoint/configured frozen LoRA tensor hashes")
    return {
        "signed_x_state_sha256": signed_hash,
        "output_projection_sha256": output_hash,
        "global_scene_residual_state_sha256": global_hash,
        "scene_state_sha256": scene_hash,
        "lora_bank_state_sha256": observed_lora,
    }


def _load_optimizer_evidence(
    optimizer_path: Path,
    *,
    contract: Mapping[str, Any],
    expected_manifest: Mapping[str, Any],
    expected_hash: str,
) -> dict[str, Any]:
    try:
        state = torch.load(optimizer_path, weights_only=True, map_location="cpu")
    except (
        EOFError,
        OSError,
        RuntimeError,
        TypeError,
        ValueError,
        pickle.UnpicklingError,
    ) as error:
        _fail(f"Cannot safely deserialize checkpoint optimizer: {error}")
    try:
        manifest, digest = canonical_v19_adamw_state(state, contract)
    except V19AdamWStateViolation as error:
        _fail(f"Checkpoint optimizer state violates V19 contract: {error}")
    _equal(manifest, dict(expected_manifest), "checkpoint/preflight optimizer manifest")
    _equal(digest, expected_hash, "checkpoint/preflight optimizer hash")
    return {"manifest": manifest, "sha256": digest}


def verify_update1(
    config: dict[str, Any],
    preflight_path: str | Path,
    checkpoint_path: str | Path,
) -> dict[str, Any]:
    """Verify the exact epoch-one tensor/moment chain without loading Gemma."""

    current_provenance = _clean_provenance(
        capture_git_source_provenance(PROJECT_ROOT),
        "current source provenance",
    )
    preflight_file = _resolve(preflight_path)
    checkpoint = _resolve(checkpoint_path)
    preflight = _read_json(preflight_file, "V19 preflight")
    evidence = _validate_preflight(config, preflight, current_provenance)
    metadata = _read_json(checkpoint / "metadata.json", "V19 epoch-one metadata")
    _exact_int(metadata.get("schema_version"), 3, "checkpoint.schema_version")
    _exact_int(metadata.get("epoch"), 1, "checkpoint.epoch")
    _exact_int(metadata.get("optimizer_step"), 1, "checkpoint.optimizer_step")
    _exact_int(metadata.get("global_step"), 12, "checkpoint.global_step")
    history = metadata.get("history")
    if not isinstance(history, list) or len(history) != 1:
        _fail("Checkpoint history must contain exactly epoch one")
    history_row = _mapping(history[0], "checkpoint history epoch one")
    _exact_int(history_row.get("epoch"), 1, "checkpoint history epoch")
    _exact_int(history_row.get("pair_batch_count"), 12, "checkpoint history pair batches")
    _equal(history_row.get("pair_batch_fraction"), 1.0, "checkpoint history pair fraction")
    _finite(history_row.get("train_loss"), "checkpoint history train loss")
    _equal(
        metadata.get("train_loss"), history_row.get("train_loss"), "checkpoint/history train loss"
    )
    _equal(
        metadata.get("pair_candidate_gate"),
        history_row.get("pair_candidate_gate"),
        "checkpoint/history final teacher gate",
    )
    _equal(metadata.get("config_hash"), config_hash(config), "checkpoint.config_hash")
    if (
        not isinstance(metadata.get("config_hash"), str)
        or _SHORT_SHA.fullmatch(metadata["config_hash"]) is None
    ):
        _fail("Checkpoint config hash must be a 12-character lowercase digest")
    training = _mapping(config.get("training"), "training")
    for key, expected in {
        "output_namespace": training.get("output_namespace"),
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
    }.items():
        _equal(metadata.get(key), expected, f"checkpoint.{key}")
    _equal(
        metadata.get("source_provenance"),
        evidence["source_provenance"],
        "checkpoint/preflight source provenance",
    )
    _equal(
        metadata.get("global_scene_residual"),
        global_scene_residual_settings(config).contract(),
        "checkpoint global residual contract",
    )
    _exact_int(
        metadata.get("global_scene_residual_parameter_count"),
        400_128,
        "checkpoint global residual parameter count",
    )
    _equal(
        metadata.get("global_scene_residual_initial_state_sha256"),
        global_scene_residual_settings(config).expected_initial_state_sha256,
        "checkpoint initial global residual hash",
    )
    _equal(
        metadata.get("global_scene_residual_state_sha256"),
        evidence["expected_global_state_sha256"],
        "checkpoint frozen global residual state",
    )
    _equal(
        metadata.get("frozen_global_scene_residual_state_sha256"),
        evidence["expected_global_state_sha256"],
        "checkpoint declared frozen global residual state",
    )
    _equal(
        metadata.get("global_scene_residual_zero_output_equivalence"),
        None,
        "checkpoint inherited global residual equivalence",
    )
    _equal(
        metadata.get("signed_x_scene_residual"),
        signed_x_scene_residual_settings(config).contract(),
        "checkpoint signed-X contract",
    )
    _exact_int(
        metadata.get("signed_x_scene_residual_parameter_count"),
        196_608,
        "checkpoint signed-X parameter count",
    )
    _equal(
        metadata.get("signed_x_scene_residual_initial_state_sha256"),
        signed_x_scene_residual_settings(config).expected_initial_state_sha256,
        "checkpoint initial signed-X hash",
    )
    if metadata.get("signed_x_scene_residual_state_sha256") == metadata.get(
        "signed_x_scene_residual_initial_state_sha256"
    ):
        _fail("Checkpoint signed-X state did not change after update one")
    _equal(
        metadata.get("signed_x_scene_residual_zero_output_equivalence"),
        evidence["zero_equivalence"],
        "checkpoint/preflight zero-output equivalence",
    )
    _equal(
        metadata.get("frozen_scene_state_sha256"),
        evidence["expected_scene_state_sha256"],
        "checkpoint frozen scene state",
    )
    _equal(
        metadata.get("frozen_lora_bank_state_sha256"),
        evidence["expected_lora_state_sha256"],
        "checkpoint frozen LoRA states",
    )
    _equal(
        metadata.get("lora_bank_state_sha256"),
        evidence["expected_lora_state_sha256"],
        "checkpoint LoRA states",
    )
    _exact_int(metadata.get("lora_trainable_parameter_count"), 0, "checkpoint trainable LoRA count")
    if "v18_stage_execution" in metadata:
        _fail("V19 checkpoint improperly carries the completed V18 stage controller")
    for key, expected in {
        "initialize_expected_adapter_sha256": evidence["source_artifact_hashes"]["adapter_sha256"],
        "initialize_expected_metadata_sha256": evidence["source_artifact_hashes"][
            "metadata_sha256"
        ],
        "initialize_expected_global_scene_residual_state_sha256": evidence[
            "expected_global_state_sha256"
        ],
        "initialize_source_residual_into_frozen_base": True,
    }.items():
        _equal(metadata.get(key), expected, f"checkpoint.{key}")

    provenance = _mapping(metadata.get("initialization_provenance"), "initialization provenance")
    source_metadata = evidence["source_metadata"]
    for key, expected in {
        "schema_version": 4,
        "mode": "frozen_v18_residual_base_plus_zero_output_signed_x_residual",
        "adapter_sha256": evidence["source_artifact_hashes"]["adapter_sha256"],
        "metadata_sha256": evidence["source_artifact_hashes"]["metadata_sha256"],
        "expected_adapter_sha256": evidence["source_artifact_hashes"]["adapter_sha256"],
        "expected_metadata_sha256": evidence["source_artifact_hashes"]["metadata_sha256"],
        "checkpoint_epoch": 4,
        "checkpoint_output_namespace": source_metadata.get("output_namespace"),
        "checkpoint_config_hash": source_metadata.get("config_hash"),
        "checkpoint_source_provenance": source_metadata.get("source_provenance"),
        "source_global_scene_residual_state_sha256": evidence["expected_global_state_sha256"],
        "expected_source_global_scene_residual_state_sha256": evidence[
            "expected_global_state_sha256"
        ],
        "global_scene_residual_frozen": True,
        "signed_x_scene_residual_initial_state_sha256": signed_x_scene_residual_settings(
            config
        ).expected_initial_state_sha256,
        "signed_x_scene_residual_zero_output": True,
        "optimizer_state_loaded": False,
        "history_loaded": False,
    }.items():
        _equal(provenance.get(key), expected, f"initialization provenance {key}")
    _equal(
        _resolve(str(provenance.get("checkpoint"))),
        evidence["source"],
        "initialization source path",
    )

    tensor_evidence = _load_tensor_evidence(
        checkpoint / "adapter.safetensors",
        metadata,
        expected_frozen_scene_sha256=evidence["expected_scene_state_sha256"],
        expected_frozen_global_sha256=evidence["expected_global_state_sha256"],
        expected_frozen_lora_sha256=evidence["expected_lora_state_sha256"],
    )
    _equal(
        tensor_evidence["signed_x_state_sha256"],
        evidence["predicted_signed_x_state_sha256"],
        "predicted/actual signed-X state",
    )
    _equal(
        tensor_evidence["output_projection_sha256"],
        evidence["predicted_output_projection_sha256"],
        "predicted/actual signed-X output weight",
    )
    optimizer_evidence = _load_optimizer_evidence(
        checkpoint / "optimizer.pt",
        contract=evidence["optimizer_contract"],
        expected_manifest=evidence["optimizer_manifest"],
        expected_hash=evidence["optimizer_hash"],
    )
    return {
        "schema_version": 1,
        "audit_type": UPDATE1_VERIFIER_TYPE,
        "match": True,
        "stage_2_authorized": True,
        "report_only": True,
        "model_loaded": False,
        "scene_map_loaded": False,
        "oracle_loaded": False,
        "optimizer_deserialized": True,
        "optimizer_deserialization": {
            "weights_only": True,
            "map_location": "cpu",
            "canonical_state_validated": True,
        },
        "source_provenance": dict(current_provenance),
        "config_hash": config_hash(config, length=64),
        "preflight_sha256": file_sha256(_resolve(preflight_path)),
        "checkpoint": _display_path(checkpoint),
        "checkpoint_artifact_hashes": {
            "adapter_sha256": file_sha256(checkpoint / "adapter.safetensors"),
            "metadata_sha256": file_sha256(checkpoint / "metadata.json"),
            "optimizer_sha256": file_sha256(checkpoint / "optimizer.pt"),
        },
        "signed_x_state_sha256": tensor_evidence["signed_x_state_sha256"],
        "output_projection_sha256": tensor_evidence["output_projection_sha256"],
        "frozen_global_scene_residual_state_sha256": tensor_evidence[
            "global_scene_residual_state_sha256"
        ],
        "frozen_scene_state_sha256": tensor_evidence["scene_state_sha256"],
        "frozen_lora_bank_state_sha256": tensor_evidence["lora_bank_state_sha256"],
        "optimizer_state_manifest": optimizer_evidence["manifest"],
        "optimizer_state_sha256": optimizer_evidence["sha256"],
    }


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
        temporary = Path(handle.name)
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--preflight", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--report", required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    report = verify_update1(config, args.preflight, args.checkpoint)
    destination = _resolve(args.report)
    _atomic_json(destination, report)
    print(
        json.dumps(
            {
                "phase": "v19_update1_verifier",
                "report": str(destination.relative_to(PROJECT_ROOT)),
                "match": True,
                "stage_2_authorized": True,
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
