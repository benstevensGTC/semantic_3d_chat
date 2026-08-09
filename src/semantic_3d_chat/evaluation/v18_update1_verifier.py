"""Report-only verifier for the exact V18 update-one checkpoint.

The verifier authorizes the second V18 training stage only when the resolved
configuration, no-live-step structural preflight, epoch-one checkpoint
metadata, exact residual tensors, and safely deserialized AdamW moments form
one closed hash chain.  It never loads Gemma, scene maps, or QA/oracle data.
"""

from __future__ import annotations

import argparse
import json
import pickle
import re
import sys
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch
from safetensors import SafetensorError
from safetensors.torch import load_file

from semantic_3d_chat.config import PROJECT_ROOT, config_hash, load_config
from semantic_3d_chat.evaluation.v18_optimizer_state import (
    V18AdamWStateViolation,
    canonical_v18_adamw_state,
    validate_v18_adamw_state_manifest,
)
from semantic_3d_chat.evaluation.v18_structural_preflight import (
    canonical_sha256,
    file_sha256,
    validate_v18_config_contract,
)
from semantic_3d_chat.language.lora import tensor_state_sha256
from semantic_3d_chat.scene_encoder import global_residual as residual_source
from semantic_3d_chat.scene_encoder.global_residual import (
    ZERO_SPATIAL_MEAN_CONTENT_GATE_V1,
    global_scene_residual_settings,
)
from semantic_3d_chat.training.source_provenance import (
    capture_git_source_provenance,
    require_clean_committed_source,
)

UPDATE1_VERIFIER_TYPE = "v18_exact_update1_match_verifier"
STAGE_EXECUTION_METADATA = {
    "stage_1_exact_v14_restart_updates": 1,
    "stage_1_stop_required": True,
    "stage_2_resume_from_epoch": 1,
    "stage_2_load_optimizer_state": True,
    "stage_2_load_history": True,
    "stage_2_target_total_optimizer_updates": 4,
}
_SHA256 = re.compile(r"[0-9a-f]{64}")
_SHORT_CONFIG_HASH = re.compile(r"[0-9a-f]{12}")
_RESIDUAL_PREFIX = "global_scene_residual."
_RESIDUAL_TENSORS: dict[str, tuple[tuple[int, ...], torch.dtype, bool]] = {
    f"{_RESIDUAL_PREFIX}position_features": ((256, 27), torch.float32, False),
    f"{_RESIDUAL_PREFIX}gate_temperature": ((), torch.float32, False),
    f"{_RESIDUAL_PREFIX}scene_norm.weight": ((1536,), torch.float32, True),
    f"{_RESIDUAL_PREFIX}scene_norm.bias": ((1536,), torch.float32, True),
    f"{_RESIDUAL_PREFIX}scene_projection.weight": ((128, 1536), torch.float32, True),
    f"{_RESIDUAL_PREFIX}scene_projection.bias": ((128,), torch.float32, True),
    f"{_RESIDUAL_PREFIX}position_projection.weight": ((128, 27), torch.float32, True),
    f"{_RESIDUAL_PREFIX}position_projection.bias": ((128,), torch.float32, True),
    f"{_RESIDUAL_PREFIX}content_gate_projection.weight": ((1, 128), torch.float32, True),
    f"{_RESIDUAL_PREFIX}output_projection.weight": ((1536, 128), torch.float32, True),
}


class V18Update1Violation(ValueError):
    """A mismatch that denies the exact stage-two resume."""

    def __init__(self, message: str, *, optimizer_deserialized: bool = False) -> None:
        super().__init__(message)
        self.optimizer_deserialized = optimizer_deserialized


def _fail(message: str, *, optimizer_deserialized: bool = False) -> None:
    raise V18Update1Violation(message, optimizer_deserialized=optimizer_deserialized)


def _mapping(value: Any, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        _fail(f"{field} must be an object")
    return value


def _sequence(value: Any, field: str) -> Sequence[Any]:
    if isinstance(value, (str, bytes, bytearray)) or not isinstance(value, Sequence):
        _fail(f"{field} must be a sequence")
    return value


def _exact_int(value: Any, expected: int, field: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value != expected:
        _fail(f"{field} mismatch: expected={expected} observed={value!r}")


def _expect_equal(observed: Any, expected: Any, field: str) -> None:
    if observed != expected:
        _fail(f"{field} mismatch: expected={expected!r} observed={observed!r}")


def _expect_sha256(value: Any, field: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        _fail(f"{field} must be a lowercase SHA-256 digest")
    return value


def _read_json(path: Path, field: str) -> Mapping[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        _fail(f"Cannot read {field} JSON at {path}: {error}")
    return _mapping(value, field)


def _resolve_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def _expect_same_path(observed: Any, expected: Path, field: str) -> None:
    if not isinstance(observed, str) or _resolve_path(observed) != expected.resolve():
        _fail(f"{field} mismatch: expected={expected} observed={observed!r}")


def _require_clean_provenance(value: Any, field: str) -> dict[str, Any]:
    provenance = dict(_mapping(value, field))
    try:
        require_clean_committed_source(provenance)
    except RuntimeError as error:
        _fail(f"{field} is not valid clean source provenance: {error}")
    return provenance


def _validate_zero_prefix_evidence(
    value: Any, expected_prefixes: Mapping[str, str], field: str
) -> None:
    evidence = _mapping(value, field)
    _expect_equal(evidence.get("verified"), True, f"{field}.verified")
    _expect_equal(
        evidence.get("question_dependent_scene_processing"),
        False,
        f"{field}.question_dependent_scene_processing",
    )
    _exact_int(evidence.get("scene_count"), len(expected_prefixes), f"{field}.scene_count")
    scenes = _mapping(evidence.get("scene_prefixes"), f"{field}.scene_prefixes")
    _expect_equal(set(scenes), set(expected_prefixes), f"{field}.scene_ids")
    for scene_id, expected_hash in expected_prefixes.items():
        observed = _mapping(scenes[scene_id], f"{field}.{scene_id}")
        expected = {
            "core_prefix_sha256": expected_hash,
            "adapted_prefix_sha256": expected_hash,
        }
        _expect_equal(dict(observed), expected, f"{field}.{scene_id}")


def _validate_rng_evidence(value: Any) -> None:
    evidence = _mapping(value, "preflight.rng_state")
    _expect_equal(
        evidence.get("all_available_domains_unchanged"),
        True,
        "preflight.rng_state.all_available_domains_unchanged",
    )
    _expect_equal(
        evidence.get("restored_after_mismatch"),
        False,
        "preflight.rng_state.restored_after_mismatch",
    )
    domains = _mapping(evidence.get("domains"), "preflight.rng_state.domains")
    _expect_equal(set(domains), {"cpu", "mps"}, "preflight.rng_state.domains")
    for name in ("cpu", "mps"):
        domain = _mapping(domains[name], f"preflight.rng_state.domains.{name}")
        available = domain.get("available")
        if not isinstance(available, bool):
            _fail(f"preflight.rng_state.domains.{name}.available must be boolean")
        if name == "cpu" and available is not True:
            _fail("preflight CPU RNG evidence must be available")
        _expect_equal(domain.get("unchanged"), True, f"preflight RNG {name} unchanged")
        before = domain.get("before_sha256")
        after = domain.get("after_sha256")
        if available:
            _expect_sha256(before, f"preflight RNG {name} before")
            _expect_sha256(after, f"preflight RNG {name} after")
            _expect_equal(after, before, f"preflight RNG {name} before/after")
        elif before is not None or after is not None:
            _fail(f"Unavailable preflight RNG domain {name} must have null hashes")


def _validate_preflight(
    config: Mapping[str, Any],
    contract: Mapping[str, Any],
    preflight: Mapping[str, Any],
    current_provenance: Mapping[str, Any],
) -> dict[str, Any]:
    expected_hashes = _mapping(contract["expected_hashes"], "contract.expected_hashes")
    checks = {
        "schema_version": preflight.get("schema_version") == 1,
        "audit_type": preflight.get("audit_type") == "v18_exact_ordered_structural_preflight",
        "runtime_eligible": preflight.get("runtime_eligible") is False,
        "uses_supervised_qa_metadata": preflight.get("uses_supervised_qa_metadata") is True,
        "question_dependent_scene_processing": preflight.get("question_dependent_scene_processing")
        is False,
        "live_optimizer_constructed": preflight.get("live_optimizer_constructed") is False,
        "live_optimizer_step_executed": preflight.get("live_optimizer_step_executed") is False,
        "isolated_clone_optimizer_constructed": preflight.get(
            "isolated_clone_optimizer_constructed"
        )
        is True,
        "structural_authorization": preflight.get("structural_authorization") is True,
    }
    failed = sorted(name for name, passed in checks.items() if not passed)
    if failed:
        _fail(f"preflight authorization contract mismatch: {failed}")
    _exact_int(preflight.get("optimizer_steps"), 0, "preflight.optimizer_steps")
    _exact_int(
        preflight.get("isolated_clone_optimizer_steps"),
        1,
        "preflight.isolated_clone_optimizer_steps",
    )
    _expect_equal(
        dict(_mapping(preflight.get("contract"), "preflight.contract")),
        contract,
        "preflight.contract",
    )
    _expect_equal(
        preflight.get("config_sha256"),
        config_hash(dict(config), length=64),
        "preflight.config_sha256",
    )
    config_path = config.get("_config_path")
    if isinstance(config_path, str):
        _expect_same_path(
            preflight.get("config_path"), _resolve_path(config_path), "preflight.config_path"
        )

    implementation_path = Path(residual_source.__file__).resolve()
    implementation_hash = file_sha256(implementation_path)
    _expect_same_path(
        preflight.get("implementation_source"),
        implementation_path,
        "preflight.implementation_source",
    )
    _expect_equal(
        preflight.get("implementation_source_sha256"),
        implementation_hash,
        "preflight.implementation_source_sha256",
    )

    source = _resolve_path(str(_mapping(config.get("training"), "training")["initialize_from"]))
    _expect_same_path(preflight.get("source_checkpoint"), source, "preflight.source_checkpoint")
    _exact_int(preflight.get("source_checkpoint_epoch"), 7, "preflight.source_checkpoint_epoch")
    observed_source_hashes = dict(
        _mapping(preflight.get("source_hashes"), "preflight.source_hashes")
    )
    expected_source_hashes = {
        name: expected_hashes[name]
        for name in (
            "source_adapter_sha256",
            "source_metadata_sha256",
            "frozen_scene_state_sha256",
            "frozen_lora_bank_state_sha256",
        )
    }
    _expect_equal(observed_source_hashes, expected_source_hashes, "preflight.source_hashes")
    for name, filename in (
        ("source_adapter_sha256", "adapter.safetensors"),
        ("source_metadata_sha256", "metadata.json"),
    ):
        path = source / filename
        if not path.is_file():
            _fail(f"Pinned V14 source artifact is missing: {path}")
        _expect_equal(file_sha256(path), expected_hashes[name], f"V14 {name}")

    evidence_hashes: dict[str, str] = {}
    evidence_paths = _mapping(contract["evidence_paths"], "contract.evidence_paths")
    for name, path_value in evidence_paths.items():
        path = _resolve_path(str(path_value))
        if not path.is_file():
            _fail(f"Pinned {name} evidence is missing: {path}")
        observed = file_sha256(path)
        expected = expected_hashes[f"{name}_sha256"]
        _expect_equal(observed, expected, f"pinned {name} evidence hash")
        evidence_hashes[name] = observed

    initial_hash = expected_hashes["initial_residual_state_sha256"]
    for field in (
        "initial_residual_state_sha256",
        "live_residual_state_sha256_before",
        "live_residual_state_sha256_after",
    ):
        _expect_equal(preflight.get(field), initial_hash, f"preflight.{field}")
    _expect_equal(
        preflight.get("live_parameter_state_unchanged"),
        True,
        "preflight.live_parameter_state_unchanged",
    )
    _validate_rng_evidence(preflight.get("rng_state"))

    expected_structural_state = {
        "architecture_version": ZERO_SPATIAL_MEAN_CONTENT_GATE_V1,
        "parameter_count": 400_128,
        "latent_count": 256,
        "scene_dim": 1536,
        "gate_temperature": float(
            _mapping(
                _mapping(config.get("scene_encoder"), "scene_encoder").get("global_scene_residual"),
                "scene_encoder.global_scene_residual",
            )["gate_temperature"]
        ),
        "spatial_centering": "all_slots_fp32",
        "content_gate": "bias_free_scalar_sigmoid_centered_content",
    }
    _expect_equal(
        dict(_mapping(preflight.get("structural_state"), "preflight.structural_state")),
        expected_structural_state,
        "preflight.structural_state",
    )
    for field, hash_field in (
        ("position_features_sha256", "position_features_sha256"),
        ("selection_sha256", "selection_sha256"),
        ("pair_membership_sha256", "pair_membership_sha256"),
        ("ordered_unit_sha256", "ordered_unit_sha256"),
    ):
        _expect_equal(preflight.get(field), expected_hashes[hash_field], f"preflight.{field}")
    ordered_units = list(_sequence(preflight.get("ordered_units"), "preflight.ordered_units"))
    _exact_int(len(ordered_units), 12, "preflight.ordered_units length")
    _expect_equal(
        canonical_sha256(ordered_units),
        expected_hashes["ordered_unit_sha256"],
        "preflight ordered-unit content hash",
    )
    _validate_zero_prefix_evidence(
        preflight.get("zero_output_prefix_equivalence"),
        _mapping(expected_hashes["core_prefix_sha256"], "expected core prefixes"),
        "preflight.zero_output_prefix_equivalence",
    )
    _expect_equal(
        dict(_mapping(preflight.get("adamw_contract"), "preflight.adamw_contract")),
        contract["optimizer"],
        "preflight.adamw_contract",
    )
    structural_gate = _mapping(preflight.get("structural_gate"), "preflight.structural_gate")
    _expect_equal(structural_gate.get("passed"), True, "preflight.structural_gate.passed")
    predicted = _expect_sha256(
        preflight.get("simulated_first_output_projection_state_sha256"),
        "preflight simulated update-one state",
    )
    gradient = _mapping(preflight.get("gradient"), "preflight.gradient")
    _expect_equal(
        gradient.get("implementation"),
        "isolated_full_residual_torch_adamw_clone",
        "preflight clone implementation",
    )
    _exact_int(
        gradient.get("parameter_count"),
        1536 * 128,
        "preflight cloned output parameter count",
    )
    _exact_int(
        gradient.get("clone_optimizer_state_parameter_count"),
        8,
        "preflight clone optimizer state parameter count",
    )
    _expect_equal(
        gradient.get("changed_parameter_keys"),
        ["output_projection.weight"],
        "preflight clone changed parameter keys",
    )
    _expect_equal(
        gradient.get("clone_residual_state_sha256"),
        predicted,
        "predicted preflight clone residual state hash",
    )
    for field in (
        "gradient_sha256",
        "clipped_gradient_sha256",
        "simulated_update_sha256",
        "clone_optimizer_state_tensor_sha256",
    ):
        _expect_sha256(gradient.get(field), f"preflight.gradient.{field}")
    optimizer_manifest = _mapping(
        gradient.get("clone_optimizer_state_manifest"),
        "preflight.gradient.clone_optimizer_state_manifest",
    )
    try:
        calculated_optimizer_hash = validate_v18_adamw_state_manifest(
            optimizer_manifest, contract["optimizer"]
        )
    except V18AdamWStateViolation as error:
        _fail(f"preflight AdamW state manifest is invalid: {error}")
    optimizer_hash = _expect_sha256(
        gradient.get("clone_optimizer_state_sha256"),
        "preflight.gradient.clone_optimizer_state_sha256",
    )
    _expect_equal(
        calculated_optimizer_hash,
        optimizer_hash,
        "preflight canonical AdamW state manifest hash",
    )
    provenance = _require_clean_provenance(
        preflight.get("source_provenance"), "preflight.source_provenance"
    )
    _expect_equal(
        provenance,
        dict(current_provenance),
        "current/preflight clean source provenance",
    )
    return {
        "predicted_residual_state_sha256": predicted,
        "optimizer_state_manifest": dict(optimizer_manifest),
        "optimizer_state_sha256": optimizer_hash,
        "source_checkpoint": source,
        "source_provenance": provenance,
        "evidence_sha256": evidence_hashes,
    }


def _validate_v14_source_metadata(
    source: Path, expected_hashes: Mapping[str, Any]
) -> Mapping[str, Any]:
    metadata = _read_json(source / "metadata.json", "V14 source metadata")
    _exact_int(metadata.get("epoch"), 7, "V14 source metadata.epoch")
    _expect_equal(
        metadata.get("frozen_scene_state_sha256"),
        expected_hashes["frozen_scene_state_sha256"],
        "V14 source frozen scene hash",
    )
    _expect_equal(
        metadata.get("lora_bank_state_sha256"),
        expected_hashes["frozen_lora_bank_state_sha256"],
        "V14 source LoRA-bank hashes",
    )
    _require_clean_provenance(
        metadata.get("source_provenance"), "V14 source metadata.source_provenance"
    )
    return metadata


def _validate_initialization_provenance(
    value: Any,
    *,
    source: Path,
    source_metadata: Mapping[str, Any],
    expected_hashes: Mapping[str, Any],
) -> None:
    provenance = _mapping(value, "checkpoint.initialization_provenance")
    expected_keys = {
        "schema_version",
        "mode",
        "checkpoint",
        "adapter_sha256",
        "metadata_sha256",
        "expected_adapter_sha256",
        "expected_metadata_sha256",
        "checkpoint_epoch",
        "checkpoint_output_namespace",
        "checkpoint_config_hash",
        "checkpoint_source_provenance",
        "optimizer_state_loaded",
        "history_loaded",
        "source_lora_bank_state_sha256",
        "all_source_lora_banks_frozen",
        "global_scene_residual_initial_state_sha256",
        "global_scene_residual_zero_output",
    }
    _expect_equal(set(provenance), expected_keys, "checkpoint.initialization_provenance keys")
    _exact_int(provenance.get("schema_version"), 3, "initialization provenance schema")
    _expect_equal(
        provenance.get("mode"),
        "named_lora_banks_frozen_plus_zero_output_scene_residual",
        "initialization provenance mode",
    )
    _expect_same_path(provenance.get("checkpoint"), source, "initialization source checkpoint")
    scalar_expected = {
        "adapter_sha256": expected_hashes["source_adapter_sha256"],
        "metadata_sha256": expected_hashes["source_metadata_sha256"],
        "expected_adapter_sha256": expected_hashes["source_adapter_sha256"],
        "expected_metadata_sha256": expected_hashes["source_metadata_sha256"],
        "checkpoint_epoch": 7,
        "checkpoint_output_namespace": source_metadata.get("output_namespace"),
        "checkpoint_config_hash": source_metadata.get("config_hash"),
        "checkpoint_source_provenance": source_metadata.get("source_provenance"),
        "optimizer_state_loaded": False,
        "history_loaded": False,
        "source_lora_bank_state_sha256": expected_hashes["frozen_lora_bank_state_sha256"],
        "all_source_lora_banks_frozen": True,
        "global_scene_residual_initial_state_sha256": expected_hashes[
            "initial_residual_state_sha256"
        ],
        "global_scene_residual_zero_output": True,
    }
    for field, expected in scalar_expected.items():
        _expect_equal(provenance.get(field), expected, f"initialization provenance {field}")


def _load_and_validate_residual_state(
    adapter_path: Path, metadata: Mapping[str, Any]
) -> tuple[str, int]:
    if not adapter_path.is_file() or adapter_path.stat().st_size <= 0:
        _fail(f"Checkpoint adapter.safetensors is missing or empty: {adapter_path}")
    try:
        tensors = load_file(adapter_path, device="cpu")
    except (OSError, RuntimeError, ValueError, SafetensorError) as error:
        _fail(f"Cannot read checkpoint safetensors at {adapter_path}: {error}")
    residual = {name: value for name, value in tensors.items() if name.startswith(_RESIDUAL_PREFIX)}
    _expect_equal(set(residual), set(_RESIDUAL_TENSORS), "checkpoint residual tensor keys")
    parameter_count = 0
    for name, (shape, dtype, is_parameter) in _RESIDUAL_TENSORS.items():
        tensor = residual[name]
        _expect_equal(tuple(tensor.shape), shape, f"checkpoint tensor shape {name}")
        _expect_equal(tensor.dtype, dtype, f"checkpoint tensor dtype {name}")
        if not bool(torch.isfinite(tensor).all()):
            _fail(f"checkpoint residual tensor contains non-finite values: {name}")
        if is_parameter:
            parameter_count += tensor.numel()
    _exact_int(parameter_count, 400_128, "checkpoint residual tensor parameter count")
    observed_hash = tensor_state_sha256(residual)
    metadata_hash = _expect_sha256(
        metadata.get("global_scene_residual_state_sha256"),
        "checkpoint global_scene_residual_state_sha256",
    )
    _expect_equal(observed_hash, metadata_hash, "safetensors residual subset hash")
    return observed_hash, parameter_count


def _load_and_validate_optimizer_state(
    optimizer_path: Path,
    *,
    optimizer_contract: Mapping[str, Any],
    expected_manifest: Mapping[str, Any],
    expected_sha256: str,
) -> tuple[dict[str, Any], str]:
    """Safely deserialize and exactly match the full update-one AdamW state."""

    if not optimizer_path.is_file() or optimizer_path.stat().st_size <= 0:
        _fail(f"Checkpoint optimizer.pt is missing or empty: {optimizer_path}")
    try:
        state_dict = torch.load(
            optimizer_path,
            weights_only=True,
            map_location="cpu",
        )
    except (
        EOFError,
        OSError,
        RuntimeError,
        TypeError,
        ValueError,
        pickle.UnpicklingError,
    ) as error:
        _fail(f"Cannot safely deserialize checkpoint optimizer.pt: {error}")
    try:
        observed_manifest, observed_sha256 = canonical_v18_adamw_state(
            state_dict, optimizer_contract
        )
    except V18AdamWStateViolation as error:
        _fail(
            f"Checkpoint AdamW state contract mismatch: {error}",
            optimizer_deserialized=True,
        )
    if observed_manifest != dict(expected_manifest):
        _fail(
            "Checkpoint AdamW state manifest differs from the exact preflight clone",
            optimizer_deserialized=True,
        )
    if observed_sha256 != expected_sha256:
        _fail(
            "Checkpoint canonical AdamW state hash differs from the exact preflight clone: "
            f"expected={expected_sha256} observed={observed_sha256}",
            optimizer_deserialized=True,
        )
    return observed_manifest, observed_sha256


def _validate_checkpoint(
    config: Mapping[str, Any],
    contract: Mapping[str, Any],
    preflight_evidence: Mapping[str, Any],
    checkpoint: Path,
) -> dict[str, Any]:
    metadata_path = checkpoint / "metadata.json"
    adapter_path = checkpoint / "adapter.safetensors"
    optimizer_path = checkpoint / "optimizer.pt"
    metadata = _read_json(metadata_path, "update-one checkpoint metadata")
    _exact_int(metadata.get("schema_version"), 3, "checkpoint.schema_version")
    _exact_int(metadata.get("epoch"), 1, "checkpoint.epoch")
    _exact_int(metadata.get("optimizer_step"), 1, "checkpoint.optimizer_step")
    _exact_int(metadata.get("global_step"), 12, "checkpoint.global_step")
    history = list(_sequence(metadata.get("history"), "checkpoint.history"))
    _exact_int(len(history), 1, "checkpoint.history length")
    first_history = _mapping(history[0], "checkpoint.history[0]")
    _exact_int(first_history.get("epoch"), 1, "checkpoint.history[0].epoch")
    _exact_int(first_history.get("pair_batch_count"), 12, "checkpoint pair batch count")
    _expect_equal(first_history.get("pair_batch_fraction"), 1.0, "checkpoint pair fraction")

    observed_config_hash = metadata.get("config_hash")
    if (
        not isinstance(observed_config_hash, str)
        or _SHORT_CONFIG_HASH.fullmatch(observed_config_hash) is None
    ):
        _fail("checkpoint.config_hash must be a 12-character lowercase digest")
    _expect_equal(observed_config_hash, config_hash(dict(config)), "checkpoint.config_hash")
    training = _mapping(config.get("training"), "training")
    _expect_equal(
        metadata.get("output_namespace"),
        training.get("output_namespace"),
        "checkpoint.output_namespace",
    )
    _expect_equal(metadata.get("gradient_accumulation"), 12, "checkpoint accumulation")
    _expect_equal(
        metadata.get("v18_stage_execution"),
        STAGE_EXECUTION_METADATA,
        "checkpoint.v18_stage_execution",
    )
    _expect_equal(metadata.get("freeze_scene_adapter"), True, "checkpoint frozen scene flag")
    _expect_equal(
        metadata.get("train_global_scene_residual_only"),
        True,
        "checkpoint residual-only flag",
    )
    _expect_equal(
        metadata.get("question_dependent_scene_processing"),
        False,
        "checkpoint question-dependent scene processing",
    )
    _expect_equal(metadata.get("scene_latents"), 256, "checkpoint scene latent count")
    _expect_equal(metadata.get("language_hidden_dim"), 1536, "checkpoint language hidden size")

    expected_hashes = _mapping(contract["expected_hashes"], "contract.expected_hashes")
    expected_residual_contract = global_scene_residual_settings(config).contract()
    _expect_equal(
        metadata.get("global_scene_residual"),
        expected_residual_contract,
        "checkpoint residual contract",
    )
    _exact_int(
        metadata.get("global_scene_residual_parameter_count"),
        400_128,
        "checkpoint residual parameter count",
    )
    _expect_equal(
        metadata.get("global_scene_residual_initial_state_sha256"),
        expected_hashes["initial_residual_state_sha256"],
        "checkpoint initial residual hash",
    )
    _expect_equal(
        metadata.get("frozen_scene_state_sha256"),
        expected_hashes["frozen_scene_state_sha256"],
        "checkpoint frozen scene hash",
    )
    for field in ("frozen_lora_bank_state_sha256", "lora_bank_state_sha256"):
        _expect_equal(
            metadata.get(field),
            expected_hashes["frozen_lora_bank_state_sha256"],
            f"checkpoint {field}",
        )
    _expect_equal(
        metadata.get("source_provenance"),
        preflight_evidence["source_provenance"],
        "checkpoint/preflight source provenance",
    )
    _require_clean_provenance(metadata.get("source_provenance"), "checkpoint.source_provenance")

    source = Path(preflight_evidence["source_checkpoint"])
    source_metadata = _validate_v14_source_metadata(source, expected_hashes)
    _validate_initialization_provenance(
        metadata.get("initialization_provenance"),
        source=source,
        source_metadata=source_metadata,
        expected_hashes=expected_hashes,
    )
    _validate_zero_prefix_evidence(
        metadata.get("global_scene_residual_zero_output_equivalence"),
        _mapping(expected_hashes["core_prefix_sha256"], "expected core prefixes"),
        "checkpoint.global_scene_residual_zero_output_equivalence",
    )

    residual_hash, parameter_count = _load_and_validate_residual_state(adapter_path, metadata)
    _expect_equal(
        residual_hash,
        preflight_evidence["predicted_residual_state_sha256"],
        "predicted preflight versus actual epoch-one residual state hash",
    )
    optimizer_manifest, optimizer_hash = _load_and_validate_optimizer_state(
        optimizer_path,
        optimizer_contract=_mapping(contract["optimizer"], "contract.optimizer"),
        expected_manifest=_mapping(
            preflight_evidence["optimizer_state_manifest"],
            "preflight optimizer state manifest",
        ),
        expected_sha256=str(preflight_evidence["optimizer_state_sha256"]),
    )
    return {
        "metadata": metadata,
        "adapter_path": adapter_path,
        "metadata_path": metadata_path,
        "optimizer_path": optimizer_path,
        "residual_state_sha256": residual_hash,
        "residual_parameter_count": parameter_count,
        "optimizer_state_manifest": optimizer_manifest,
        "optimizer_state_sha256": optimizer_hash,
    }


def verify_update1(
    config: Mapping[str, Any],
    preflight_path: str | Path,
    checkpoint_path: str | Path,
) -> dict[str, Any]:
    """Return deterministic stage-two authorization or raise on any mismatch."""

    implementation_path = Path(residual_source.__file__).resolve()
    implementation_hash = file_sha256(implementation_path)
    current_provenance = capture_git_source_provenance(PROJECT_ROOT)
    try:
        require_clean_committed_source(current_provenance)
    except RuntimeError as error:
        _fail(f"Current verifier source provenance is not clean and committed: {error}")
    try:
        contract = validate_v18_config_contract(
            config, implementation_source_sha256=implementation_hash
        )
    except (TypeError, ValueError) as error:
        _fail(f"V18 config/contract validation failed: {error}")
    preflight_file = _resolve_path(preflight_path)
    checkpoint = _resolve_path(checkpoint_path)
    if not preflight_file.is_file():
        _fail(f"V18 structural-preflight JSON is missing: {preflight_file}")
    if not checkpoint.is_dir():
        _fail(f"V18 epoch_001 checkpoint directory is missing: {checkpoint}")
    preflight = _read_json(preflight_file, "V18 structural preflight")
    preflight_evidence = _validate_preflight(config, contract, preflight, current_provenance)
    checkpoint_evidence = _validate_checkpoint(config, contract, preflight_evidence, checkpoint)

    report: dict[str, Any] = {
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
        "source_provenance": current_provenance,
        "config": {
            "path": config.get("_config_path"),
            "sha256": config_hash(dict(config), length=64),
            "short_hash": config_hash(dict(config)),
            "contract_sha256": contract["contract_sha256"],
            "implementation_source_sha256": implementation_hash,
        },
        "preflight": {
            "path": str(preflight_file),
            "file_sha256": file_sha256(preflight_file),
            "optimizer_steps": 0,
            "isolated_clone_optimizer_steps": 1,
            "live_state_unchanged": True,
            "rng_state_unchanged": True,
            "structural_authorization": True,
            "predicted_residual_state_sha256": preflight_evidence[
                "predicted_residual_state_sha256"
            ],
            "optimizer_state_sha256": preflight_evidence["optimizer_state_sha256"],
            "evidence_sha256": preflight_evidence["evidence_sha256"],
        },
        "checkpoint": {
            "path": str(checkpoint),
            "epoch": 1,
            "optimizer_step": 1,
            "global_step": 12,
            "history_length": 1,
            "adapter_sha256": file_sha256(checkpoint_evidence["adapter_path"]),
            "metadata_sha256": file_sha256(checkpoint_evidence["metadata_path"]),
            "optimizer_sha256": file_sha256(checkpoint_evidence["optimizer_path"]),
            "optimizer_size_bytes": checkpoint_evidence["optimizer_path"].stat().st_size,
            "optimizer_state_sha256": checkpoint_evidence["optimizer_state_sha256"],
            "optimizer_state_manifest": checkpoint_evidence["optimizer_state_manifest"],
            "residual_state_sha256": checkpoint_evidence["residual_state_sha256"],
            "residual_parameter_count": checkpoint_evidence["residual_parameter_count"],
            "v18_stage_execution": STAGE_EXECUTION_METADATA,
        },
    }
    report["match_report_sha256"] = canonical_sha256(report)
    return report


def _write_report(path: Path, report: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f"{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with open(descriptor, "w", encoding="utf-8", closefd=True) as handle:
            json.dump(report, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
        Path(temporary).replace(path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise


def run_verifier(
    config_path: str | Path,
    preflight_path: str | Path,
    checkpoint_path: str | Path,
    report_path: str | Path,
) -> dict[str, Any]:
    """Load inputs, write a machine-readable decision, and fail closed."""

    destination = _resolve_path(report_path)
    try:
        config = load_config(config_path)
        report = verify_update1(config, preflight_path, checkpoint_path)
    except (OSError, TypeError, ValueError) as error:
        denial = {
            "schema_version": 1,
            "audit_type": UPDATE1_VERIFIER_TYPE,
            "match": False,
            "stage_2_authorized": False,
            "report_only": True,
            "model_loaded": False,
            "scene_map_loaded": False,
            "oracle_loaded": False,
            "optimizer_deserialized": bool(getattr(error, "optimizer_deserialized", False)),
            "violation": str(error),
        }
        denial["match_report_sha256"] = canonical_sha256(denial)
        _write_report(destination, denial)
        raise V18Update1Violation(
            str(error),
            optimizer_deserialized=bool(getattr(error, "optimizer_deserialized", False)),
        ) from error
    _write_report(destination, report)
    return report


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--preflight", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--report", required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        report = run_verifier(args.config, args.preflight, args.checkpoint, args.report)
    except V18Update1Violation as error:
        print(json.dumps({"stage_2_authorized": False, "violation": str(error)}), file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "stage_2_authorized": True,
                "report": str(_resolve_path(args.report)),
                "match_report_sha256": report["match_report_sha256"],
            }
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised through main()
    raise SystemExit(main())


__all__ = [
    "STAGE_EXECUTION_METADATA",
    "UPDATE1_VERIFIER_TYPE",
    "V18Update1Violation",
    "main",
    "run_verifier",
    "verify_update1",
]
