"""Create-once aggregate attestation for V96's immutable fixed-final candidate.

This is the only V96 evaluation-auth process allowed to walk the historical
training-authentication chain.  It runs before prediction, loads no model and
opens no known-development question or label.  The label-blind predictor then
authenticates this row-free attestation plus the candidate bytes directly; it
never needs to reopen a training QA source.
"""

from __future__ import annotations

import argparse
import json
import re
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Final

import torch
from safetensors import safe_open
from safetensors.torch import load_file

from semantic_3d_chat.chat.file_audit import FileAccessAudit
from semantic_3d_chat.config import PROJECT_ROOT
from semantic_3d_chat.evaluation.v85_strict_multiscene_preflight import sha256_file_v85
from semantic_3d_chat.evaluation.v96_atomic_pair_repair_preflight import (
    CONFIG,
    FRESH_PARAMETER_COUNT,
    TARGET_MODULES,
    load_config_v96,
)
from semantic_3d_chat.evaluation.v96_evaluation_io_v2 import (
    physical_path_v96_v2,
    read_json_strict_v96_v2,
    write_json_create_once_v96_v2,
)
from semantic_3d_chat.evaluation.v96_known_development_common import (
    EXPECTED_CANDIDATE_TENSORS,
    assert_aggregate_only_v96,
    canonical_sha256_v96,
    evaluation_paths_v96,
    resolve_v96,
)
from semantic_3d_chat.evaluation.v96_known_development_common import (
    authenticate_fixed_final_candidate_v96 as authenticate_full_chain_v1,
)
from semantic_3d_chat.evaluation.v96_known_development_implementation import (
    IMPLEMENTATION_SEAL as V1_IMPLEMENTATION_SEAL,
)
from semantic_3d_chat.evaluation.v96_known_development_implementation import (
    authenticate_evaluation_implementation_v96,
)
from semantic_3d_chat.language.lora import tensor_state_sha256

SCHEMA_VERSION: Final[int] = 96
ARTIFACT: Final[str] = "gemma4_v96_fixed_final_candidate_attestation_v2"
STATUS: Final[str] = "attested_before_known_development_question_io"
ATTESTATION: Final[Path] = (
    PROJECT_ROOT
    / "reports/gemma4/metrics/gemma4_v96_fixed_final_candidate_attestation_v2.json"
)

_EXPECTED_FIELDS: Final[frozenset[str]] = frozenset(
    {
        "artifact",
        "schema_version",
        "status",
        "config_sha256",
        "candidate_fingerprint_sha256",
        "candidate_state_sha256",
        "candidate_weights_sha256",
        "candidate_metadata_file_sha256",
        "candidate_metadata_canonical_sha256",
        "candidate_tensor_inventory_sha256",
        "training_report_sha256",
        "preregistration_sha256",
        "cpu_preflight_sha256",
        "topology_smoke_sha256",
        "frozen_v95_state_sha256",
        "fixed_final_optimizer_updates",
        "v1_implementation_seal_sha256",
        "v2_implementation_seal_sha256",
        "historical_v1_attempt_failed_before_question_io",
        "historical_v1_output_count",
        "training_chain_authenticated_in_separate_model_free_process",
        "training_qa_content_serialized",
        "known_development_questions_opened",
        "known_development_labels_opened",
        "oracle_opened",
        "model_loaded",
        "row_level_content_serialized",
        "runtime_promotion_authorized",
        "base_model_snapshot_inventory_sha256",
        "frozen_bank_expected_state_inventory_sha256",
        "lora_bank_topology_sha256",
        "weights_hashed_not_model_loaded",
        "candidate_auth_unique_paths",
        "candidate_auth_unique_path_count",
        "candidate_auth_unique_path_inventory_sha256",
        "protected_read_count",
        "attestation_identity_sha256",
    }
)
_HEX64: Final[re.Pattern[str]] = re.compile(r"[0-9a-f]{64}")


def _candidate_paths(config: Mapping[str, Any]) -> tuple[Path, Path, Path]:
    root = physical_path_v96_v2(config["outputs"]["fixed_final_candidate"])
    weights = physical_path_v96_v2(root / "bridge.safetensors")
    metadata = physical_path_v96_v2(root / "runtime_metadata.json")
    if not weights.is_file() or not metadata.is_file():
        raise FileNotFoundError("V96 v2 candidate files must be physical regular files")
    return root, weights, metadata


def _tensor_inventory(weights: Path) -> tuple[dict[str, list[Any]], str]:
    with safe_open(str(weights), framework="pt", device="cpu") as archive:
        keys = sorted(archive.keys())
        inventory = {
            key: [*archive.get_slice(key).get_shape(), str(archive.get_tensor(key).dtype)]
            for key in keys
        }
    return inventory, canonical_sha256_v96(inventory)


def _without_identity(payload: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in payload.items() if key != "attestation_identity_sha256"}


def build_candidate_attestation_v96(
    config_path: str | Path = CONFIG,
) -> dict[str, Any]:
    """Authenticate the full historical chain and return only row-free hashes."""

    from semantic_3d_chat.evaluation.v96_known_development_implementation_v2 import (
        _candidate_forbidden_roots,
        authenticate_evaluation_implementation_v96_v2,
    )

    physical_config = physical_path_v96_v2(config_path)
    config = load_config_v96(physical_config, allow_draft=False)
    outputs = evaluation_paths_v96(config)
    historical = tuple(outputs.__dict__.values())
    present = [path for path in historical if path.exists() or path.is_symlink()]
    if present:
        raise RuntimeError(
            "V96 v2 attestation must precede every known-development output"
        )
    forbidden_roots = _candidate_forbidden_roots(config)
    audit = FileAccessAudit(
        forbidden_roots,
        forbidden_component_names={"oracle"},
        block_forbidden=True,
    )
    physical_v1_seal = physical_path_v96_v2(V1_IMPLEMENTATION_SEAL)
    v1 = authenticate_evaluation_implementation_v96(
        seal_path=physical_v1_seal, config_path=config_path
    )
    v2 = authenticate_evaluation_implementation_v96_v2(config_path=config_path)
    with audit:
        candidate = authenticate_full_chain_v1(config, config_path=config_path)
    audit.assert_clean()
    candidate_access = {
        "unique_paths": audit.unique_paths,
        "unique_path_count": len(audit.unique_paths),
        "unique_path_inventory_sha256": canonical_sha256_v96(audit.unique_paths),
        "protected_read_count": 0,
        "known_development_questions_opened": False,
        "known_development_labels_opened": False,
        "oracle_opened": False,
        "model_loaded": False,
    }
    if candidate_access != v2.get("candidate_auth_access"):
        raise RuntimeError("V96 v2 historical-chain access differs from its seal")
    root, weights, _metadata_path = _candidate_paths(config)
    if root.is_symlink() or not root.is_dir():
        raise FileNotFoundError(root)
    inventory, inventory_sha256 = _tensor_inventory(weights)
    if set(inventory) != set(EXPECTED_CANDIDATE_TENSORS):
        raise ValueError("V96 v2 candidate tensor inventory changed")
    topology_smoke_sha256 = sha256_file_v85(
        resolve_v96(config["outputs"]["topology_smoke"])
    )
    observed_pins = {
        key: candidate[key]
        for key in (
            "fingerprint_sha256",
            "weights_sha256",
            "metadata_file_sha256",
            "metadata_canonical_sha256",
            "state_sha256",
            "tensor_inventory_sha256",
            "training_report_sha256",
            "config_sha256",
            "preregistration_sha256",
            "cpu_preflight_sha256",
            "fixed_final_optimizer_updates",
            "frozen_v95_state_sha256",
            "known_development_scored",
            "deferred_final_generated",
            "runtime_promotion_authorized",
        )
    }
    observed_pins["topology_smoke_sha256"] = topology_smoke_sha256
    if observed_pins != v2.get("candidate_pins"):
        raise RuntimeError("V96 v2 candidate differs from the pre-attestation seal")
    # Repeat the full proof with one audit spanning historical authentication,
    # candidate/tensor reads, and every aggregate byte pinned by the receipt.
    verification_audit = FileAccessAudit(
        forbidden_roots,
        forbidden_component_names={"oracle"},
        block_forbidden=True,
    )
    with verification_audit:
        v1_verified = authenticate_evaluation_implementation_v96(
            seal_path=physical_v1_seal, config_path=config_path
        )
        v2_verified = authenticate_evaluation_implementation_v96_v2(
            config_path=config_path
        )
        candidate_verified = authenticate_full_chain_v1(
            config, config_path=config_path
        )
        verified_root, verified_weights, verified_metadata = _candidate_paths(config)
        verification_audit.record(verified_weights)
        verification_audit.record(verified_metadata)
        verified_inventory, verified_inventory_sha256 = _tensor_inventory(
            verified_weights
        )
        verified_metadata_value = read_json_strict_v96_v2(verified_metadata)
        verified_config_sha256 = sha256_file_v85(physical_config)
        verified_weights_sha256 = sha256_file_v85(verified_weights)
        verified_metadata_file_sha256 = sha256_file_v85(verified_metadata)
        verified_metadata_canonical_sha256 = canonical_sha256_v96(
            verified_metadata_value
        )
        aggregate_paths = (
            physical_path_v96_v2(config["outputs"]["training_report"]),
            physical_path_v96_v2(config["outputs"]["preregistration"]),
            physical_path_v96_v2(config["outputs"]["cpu_preflight"]),
            physical_path_v96_v2(config["outputs"]["topology_smoke"]),
        )
        verified_aggregate_hashes: dict[Path, str] = {}
        for aggregate_path in aggregate_paths:
            verification_audit.record(aggregate_path)
            read_json_strict_v96_v2(aggregate_path)
            verified_aggregate_hashes[aggregate_path] = sha256_file_v85(
                aggregate_path
            )
    verification_audit.assert_clean()
    if (
        v1_verified != v1
        or v2_verified != v2
        or candidate_verified != candidate
        or verified_root != root
        or verified_inventory != inventory
        or verified_inventory_sha256 != inventory_sha256
        or canonical_sha256_v96(verified_metadata_value)
        != candidate["metadata_canonical_sha256"]
    ):
        raise RuntimeError("V96 v2 candidate changed during audited attestation")
    payload: dict[str, Any] = {
        "artifact": ARTIFACT,
        "schema_version": SCHEMA_VERSION,
        "status": STATUS,
        "config_sha256": verified_config_sha256,
        "candidate_fingerprint_sha256": candidate["fingerprint_sha256"],
        "candidate_state_sha256": candidate["state_sha256"],
        "candidate_weights_sha256": candidate["weights_sha256"],
        "candidate_metadata_file_sha256": candidate["metadata_file_sha256"],
        "candidate_metadata_canonical_sha256": candidate["metadata_canonical_sha256"],
        "candidate_tensor_inventory_sha256": inventory_sha256,
        "training_report_sha256": candidate["training_report_sha256"],
        "preregistration_sha256": candidate["preregistration_sha256"],
        "cpu_preflight_sha256": candidate["cpu_preflight_sha256"],
        "topology_smoke_sha256": topology_smoke_sha256,
        "frozen_v95_state_sha256": candidate["frozen_v95_state_sha256"],
        "fixed_final_optimizer_updates": candidate["fixed_final_optimizer_updates"],
        "v1_implementation_seal_sha256": v1["seal_sha256"],
        "v2_implementation_seal_sha256": v2["seal_sha256"],
        "historical_v1_attempt_failed_before_question_io": True,
        "historical_v1_output_count": 0,
        "training_chain_authenticated_in_separate_model_free_process": True,
        "training_qa_content_serialized": False,
        "known_development_questions_opened": False,
        "known_development_labels_opened": False,
        "oracle_opened": False,
        "model_loaded": False,
        "row_level_content_serialized": False,
        "runtime_promotion_authorized": False,
        "base_model_snapshot_inventory_sha256": v2["model_snapshot_binding"][
            "inventory_sha256"
        ],
        "frozen_bank_expected_state_inventory_sha256": v2[
            "frozen_bank_expected_state_inventory_sha256"
        ],
        "lora_bank_topology_sha256": v2["lora_bank_topology_sha256"],
        "weights_hashed_not_model_loaded": True,
        "candidate_auth_unique_paths": candidate_access["unique_paths"],
        "candidate_auth_unique_path_count": candidate_access["unique_path_count"],
        "candidate_auth_unique_path_inventory_sha256": candidate_access[
            "unique_path_inventory_sha256"
        ],
        "protected_read_count": 0,
    }
    payload["attestation_identity_sha256"] = canonical_sha256_v96(payload)
    if set(payload) != _EXPECTED_FIELDS:
        raise AssertionError("V96 v2 candidate-attestation field inventory changed")
    assert_aggregate_only_v96(payload)
    # Bind the exact metadata bytes checked by the full-chain authenticator.
    if (
        verified_weights_sha256 != payload["candidate_weights_sha256"]
        or verified_metadata_file_sha256
        != payload["candidate_metadata_file_sha256"]
        or verified_metadata_canonical_sha256
        != payload["candidate_metadata_canonical_sha256"]
        or len(verified_aggregate_hashes) != 4
    ):
        raise RuntimeError("V96 v2 candidate changed during attestation")
    return payload


def seal_candidate_attestation_v96(
    config_path: str | Path = CONFIG,
) -> dict[str, Any]:
    from semantic_3d_chat.evaluation.v96_known_development_implementation import (
        exclusive_evaluation_lock_v96,
    )

    with exclusive_evaluation_lock_v96():
        official_attestation = physical_path_v96_v2(ATTESTATION)
        if official_attestation.exists() or official_attestation.is_symlink():
            raise FileExistsError(official_attestation)
        payload = build_candidate_attestation_v96(config_path)
        # Validate the exact serialized payload through the full authenticator
        # before the irreversible create-once publication.
        official_attestation.parent.mkdir(parents=True, exist_ok=True)
        physical_path_v96_v2(official_attestation)
        with tempfile.TemporaryDirectory(
            prefix=".v96-attestation-prevalidate-", dir=official_attestation.parent
        ) as temporary_directory:
            temporary = Path(temporary_directory) / official_attestation.name
            write_json_create_once_v96_v2(temporary, payload)
            authenticate_candidate_attestation_v96(
                config_path, attestation_path=temporary
            )
        write_json_create_once_v96_v2(official_attestation, payload)
        return authenticate_candidate_attestation_v96(
            config_path, attestation_path=official_attestation
        )


def authenticate_candidate_attestation_v96(
    config_path: str | Path = CONFIG,
    *,
    audit: FileAccessAudit | None = None,
    authenticate_implementation_sources: bool = True,
    expected_implementation_seal_sha256: str | None = None,
    attestation_path: str | Path = ATTESTATION,
) -> dict[str, Any]:
    """Authenticate only aggregate/candidate bytes when source auth is disabled.

    Predictor/NLL processes pass ``authenticate_implementation_sources=False``
    after authenticating the v2 implementation before entering their file
    audit.  This prevents a transitive training-source walk under inference.
    """

    from semantic_3d_chat.evaluation.v96_known_development_implementation_v2 import (
        IMPLEMENTATION_SEAL_V2,
        _candidate_forbidden_roots,
        _candidate_required_paths,
        _validate_candidate_auth_access,
        _validate_frozen_bank_expected_states_v96_v2,
        _validate_lora_bank_topology_v96_v2,
        _validate_model_snapshot_binding_v96_v2,
        authenticate_evaluation_implementation_v96_v2,
    )

    physical_config = physical_path_v96_v2(config_path)
    config = load_config_v96(physical_config, allow_draft=False)
    if authenticate_implementation_sources:
        implementation = authenticate_evaluation_implementation_v96_v2(
            config_path=config_path
        )
        implementation_sha256 = implementation["seal_sha256"]
        sealed_pins = implementation["candidate_pins"]
        sealed_access = implementation["candidate_auth_access"]
        sealed_model_snapshot = implementation["model_snapshot_binding"]
        sealed_frozen_bank_states = implementation["frozen_bank_expected_states"]
        sealed_frozen_bank_inventory_sha256 = implementation[
            "frozen_bank_expected_state_inventory_sha256"
        ]
        sealed_lora_bank_topology = implementation["lora_bank_topology"]
        sealed_lora_bank_topology_sha256 = implementation[
            "lora_bank_topology_sha256"
        ]
    else:
        if expected_implementation_seal_sha256 is None:
            raise ValueError(
                "V96 v2 inference auth requires its pre-audit implementation SHA"
            )
        physical_implementation_seal = physical_path_v96_v2(IMPLEMENTATION_SEAL_V2)
        if audit is not None:
            audit.record(physical_implementation_seal)
        implementation_sha256 = sha256_file_v85(physical_implementation_seal)
        if implementation_sha256 != expected_implementation_seal_sha256:
            raise RuntimeError("V96 v2 implementation seal changed across audit boundary")
        sealed = read_json_strict_v96_v2(physical_implementation_seal)
        sealed_pins = sealed.get("candidate_pins")
        sealed_access = sealed.get("candidate_auth_access")
        sealed_model_snapshot = sealed.get("model_snapshot_binding")
        sealed_frozen_bank_states = sealed.get("frozen_bank_expected_states")
        sealed_frozen_bank_inventory_sha256 = sealed.get(
            "frozen_bank_expected_state_inventory_sha256"
        )
        sealed_lora_bank_topology = sealed.get("lora_bank_topology")
        sealed_lora_bank_topology_sha256 = sealed.get(
            "lora_bank_topology_sha256"
        )
        if not isinstance(sealed_pins, Mapping) or not isinstance(
            sealed_access, Mapping
        ):
            raise ValueError("V96 v2 sealed candidate/access pins are missing")
    sealed_access = _validate_candidate_auth_access(
        sealed_access,
        forbidden_roots=_candidate_forbidden_roots(config),
        required_paths=_candidate_required_paths(
            config, config_path=physical_config
        ),
    )
    sealed_model_snapshot = _validate_model_snapshot_binding_v96_v2(
        sealed_model_snapshot, config=config
    )
    sealed_frozen_bank_states = _validate_frozen_bank_expected_states_v96_v2(
        sealed_frozen_bank_states
    )
    if sealed_frozen_bank_inventory_sha256 != canonical_sha256_v96(
        sealed_frozen_bank_states
    ):
        raise ValueError("V96 v2 sealed frozen-bank inventory hash changed")
    sealed_lora_bank_topology = _validate_lora_bank_topology_v96_v2(
        sealed_lora_bank_topology
    )
    if sealed_lora_bank_topology_sha256 != canonical_sha256_v96(
        sealed_lora_bank_topology
    ):
        raise ValueError("V96 v2 sealed LoRA-bank topology hash changed")
    attestation_source = physical_path_v96_v2(attestation_path)
    for path in (attestation_source,):
        if audit is not None:
            audit.record(path)
    payload = read_json_strict_v96_v2(attestation_source)
    physical_v1_seal = physical_path_v96_v2(V1_IMPLEMENTATION_SEAL)
    read_json_strict_v96_v2(physical_v1_seal)
    root, weights, metadata_path = _candidate_paths(config)
    if (
        root.is_symlink()
        or not root.is_dir()
        or sorted(path.name for path in root.iterdir())
        != ["bridge.safetensors", "runtime_metadata.json"]
    ):
        raise ValueError("V96 v2 fixed-final directory inventory changed")
    for path in (weights, metadata_path):
        if audit is not None:
            audit.record(path)
    metadata = read_json_strict_v96_v2(metadata_path)
    inventory, inventory_sha256 = _tensor_inventory(weights)
    state = load_file(str(weights), device="cpu")
    expected_metadata = {
        "artifact": "gemma4_v96_atomic_pair_repair_fixed_final_v1",
        "schema_version": SCHEMA_VERSION,
        "status": "fixed_final_awaiting_known_development_gate",
        "parent": "v95_fixed_final_nonpromoted_optimization_parent",
        "bank_name": "v96_atomic_pair_repair_bridge",
        "target_modules": list(TARGET_MODULES),
        "rank": 8,
        "alpha": 16.0,
        "dropout": 0.0,
        "parameter_count": FRESH_PARAMETER_COUNT,
        "tensor_inventory": sorted(EXPECTED_CANDIDATE_TENSORS),
    }
    exact_metadata_fields = {
        "artifact",
        "schema_version",
        "status",
        "parent",
        "bank_name",
        "target_modules",
        "rank",
        "alpha",
        "dropout",
        "parameter_count",
        "state_sha256",
        "weights_sha256",
        "tensor_inventory",
        "environmental_memory_serialized",
        "questions_or_answers_serialized",
        "oracle_serialized",
        "known_development_scored",
        "deferred_final_generated",
        "runtime_promotion_authorized",
        "bindings",
    }
    direct_fingerprint = {
        "artifact": "gemma4_v96_fixed_final_fingerprint_v1",
        "directory_inventory": ["bridge.safetensors", "runtime_metadata.json"],
        "weights_sha256": sha256_file_v85(weights),
        "metadata_file_sha256": sha256_file_v85(metadata_path),
        "metadata_canonical_sha256": canonical_sha256_v96(metadata),
        "state_sha256": tensor_state_sha256(state),
        "tensor_inventory_sha256": inventory_sha256,
        "training_report_sha256": sealed_pins.get("training_report_sha256"),
        "config_sha256": sealed_pins.get("config_sha256"),
        "preregistration_sha256": sealed_pins.get("preregistration_sha256"),
        "cpu_preflight_sha256": sealed_pins.get("cpu_preflight_sha256"),
        "fixed_final_optimizer_updates": sealed_pins.get(
            "fixed_final_optimizer_updates"
        ),
        "frozen_v95_state_sha256": sealed_pins.get("frozen_v95_state_sha256"),
        "known_development_scored": False,
        "deferred_final_generated": False,
        "runtime_promotion_authorized": False,
    }
    direct_fingerprint["fingerprint_sha256"] = canonical_sha256_v96(
        direct_fingerprint
    )
    aggregate_paths = {
        "training_report_sha256": physical_path_v96_v2(config["outputs"]["training_report"]),
        "preregistration_sha256": physical_path_v96_v2(config["outputs"]["preregistration"]),
        "cpu_preflight_sha256": physical_path_v96_v2(config["outputs"]["cpu_preflight"]),
        "topology_smoke_sha256": physical_path_v96_v2(config["outputs"]["topology_smoke"]),
    }
    for path in aggregate_paths.values():
        if audit is not None:
            audit.record(path)
        read_json_strict_v96_v2(path)
    current_aggregate_hashes = {
        key: sha256_file_v85(path) for key, path in aggregate_paths.items()
    }
    pin_hash_fields = (
        "fingerprint_sha256",
        "weights_sha256",
        "metadata_file_sha256",
        "metadata_canonical_sha256",
        "state_sha256",
        "tensor_inventory_sha256",
        "training_report_sha256",
        "config_sha256",
        "preregistration_sha256",
        "cpu_preflight_sha256",
        "topology_smoke_sha256",
        "frozen_v95_state_sha256",
    )
    if (
        set(payload) != _EXPECTED_FIELDS
        or payload.get("artifact") != ARTIFACT
        or payload.get("schema_version") != SCHEMA_VERSION
        or payload.get("status") != STATUS
        or payload.get("config_sha256") != sha256_file_v85(physical_config)
        or payload.get("v1_implementation_seal_sha256")
        != sha256_file_v85(physical_v1_seal)
        or payload.get("v2_implementation_seal_sha256") != implementation_sha256
        or payload.get("candidate_fingerprint_sha256")
        != sealed_pins.get("fingerprint_sha256")
        or payload.get("candidate_state_sha256") != sealed_pins.get("state_sha256")
        or payload.get("candidate_weights_sha256")
        != sealed_pins.get("weights_sha256")
        or payload.get("candidate_metadata_file_sha256")
        != sealed_pins.get("metadata_file_sha256")
        or payload.get("candidate_metadata_canonical_sha256")
        != sealed_pins.get("metadata_canonical_sha256")
        or payload.get("candidate_tensor_inventory_sha256")
        != sealed_pins.get("tensor_inventory_sha256")
        or payload.get("training_report_sha256")
        != sealed_pins.get("training_report_sha256")
        or payload.get("preregistration_sha256")
        != sealed_pins.get("preregistration_sha256")
        or payload.get("cpu_preflight_sha256")
        != sealed_pins.get("cpu_preflight_sha256")
        or payload.get("topology_smoke_sha256")
        != sealed_pins.get("topology_smoke_sha256")
        or payload.get("frozen_v95_state_sha256")
        != sealed_pins.get("frozen_v95_state_sha256")
        or payload.get("fixed_final_optimizer_updates")
        != sealed_pins.get("fixed_final_optimizer_updates")
        or payload.get("base_model_snapshot_inventory_sha256")
        != sealed_model_snapshot.get("inventory_sha256")
        or payload.get("frozen_bank_expected_state_inventory_sha256")
        != sealed_frozen_bank_inventory_sha256
        or payload.get("lora_bank_topology_sha256")
        != sealed_lora_bank_topology_sha256
        or payload.get("weights_hashed_not_model_loaded") is not True
        or payload.get("candidate_auth_unique_path_count")
        != sealed_access.get("unique_path_count")
        or payload.get("candidate_auth_unique_paths")
        != sealed_access.get("unique_paths")
        or payload.get("candidate_auth_unique_path_inventory_sha256")
        != sealed_access.get("unique_path_inventory_sha256")
        or not isinstance(payload.get("candidate_auth_unique_paths"), list)
        or payload.get("candidate_auth_unique_paths")
        != sorted(set(payload.get("candidate_auth_unique_paths", [])))
        or payload.get("candidate_auth_unique_path_count")
        != len(payload.get("candidate_auth_unique_paths", []))
        or payload.get("candidate_auth_unique_path_inventory_sha256")
        != canonical_sha256_v96(payload.get("candidate_auth_unique_paths"))
        or payload.get("protected_read_count") != 0
        or any(
            _HEX64.fullmatch(str(sealed_pins.get(field))) is None
            for field in pin_hash_fields
        )
        or any(
            current_aggregate_hashes[field] != sealed_pins.get(field)
            or current_aggregate_hashes[field] != payload.get(field)
            for field in current_aggregate_hashes
        )
        or payload.get("historical_v1_attempt_failed_before_question_io") is not True
        or payload.get("historical_v1_output_count") != 0
        or payload.get("training_chain_authenticated_in_separate_model_free_process")
        is not True
        or any(
            payload.get(field) is not False
            for field in (
                "training_qa_content_serialized",
                "known_development_questions_opened",
                "known_development_labels_opened",
                "oracle_opened",
                "model_loaded",
                "row_level_content_serialized",
                "runtime_promotion_authorized",
            )
        )
        or payload.get("attestation_identity_sha256")
        != canonical_sha256_v96(_without_identity(payload))
        or set(metadata) != exact_metadata_fields
        or any(metadata.get(key) != value for key, value in expected_metadata.items())
        or any(
            metadata.get(field) is not False
            for field in (
                "environmental_memory_serialized",
                "questions_or_answers_serialized",
                "oracle_serialized",
                "known_development_scored",
                "deferred_final_generated",
                "runtime_promotion_authorized",
            )
        )
        or payload.get("candidate_weights_sha256") != sha256_file_v85(weights)
        or payload.get("candidate_metadata_file_sha256")
        != sha256_file_v85(metadata_path)
        or payload.get("candidate_metadata_canonical_sha256")
        != canonical_sha256_v96(metadata)
        or payload.get("candidate_tensor_inventory_sha256") != inventory_sha256
        or metadata.get("weights_sha256") != payload.get("candidate_weights_sha256")
        or metadata.get("state_sha256") != payload.get("candidate_state_sha256")
        or tensor_state_sha256(state) != payload.get("candidate_state_sha256")
        or any(not bool(torch.isfinite(value).all()) for value in state.values())
        or set(inventory) != set(EXPECTED_CANDIDATE_TENSORS)
        or any(
            tuple(inventory[key][:-1]) != EXPECTED_CANDIDATE_TENSORS[key]
            or inventory[key][-1] != "torch.float32"
            for key in inventory
        )
        or direct_fingerprint["fingerprint_sha256"]
        != payload.get("candidate_fingerprint_sha256")
    ):
        raise ValueError("V96 v2 candidate attestation authentication failed")
    assert_aggregate_only_v96(payload)
    return {
        "artifact": "gemma4_v96_fixed_final_fingerprint_v1",
        "directory_inventory": ["bridge.safetensors", "runtime_metadata.json"],
        "fingerprint_sha256": payload["candidate_fingerprint_sha256"],
        "state_sha256": payload["candidate_state_sha256"],
        "weights_sha256": payload["candidate_weights_sha256"],
        "metadata_file_sha256": payload["candidate_metadata_file_sha256"],
        "metadata_canonical_sha256": payload["candidate_metadata_canonical_sha256"],
        "tensor_inventory_sha256": payload["candidate_tensor_inventory_sha256"],
        "training_report_sha256": payload["training_report_sha256"],
        "config_sha256": payload["config_sha256"],
        "preregistration_sha256": payload["preregistration_sha256"],
        "cpu_preflight_sha256": payload["cpu_preflight_sha256"],
        "topology_smoke_sha256": payload["topology_smoke_sha256"],
        "fixed_final_optimizer_updates": payload["fixed_final_optimizer_updates"],
        "frozen_v95_state_sha256": payload["frozen_v95_state_sha256"],
        "attestation_file_sha256": sha256_file_v85(attestation_source),
        "attestation_identity_sha256": payload["attestation_identity_sha256"],
        "v2_implementation_seal_sha256": implementation_sha256,
        "model_snapshot_inventory_sha256": payload[
            "base_model_snapshot_inventory_sha256"
        ],
        "frozen_bank_expected_state_inventory_sha256": payload[
            "frozen_bank_expected_state_inventory_sha256"
        ],
        "lora_bank_topology_sha256": payload["lora_bank_topology_sha256"],
        "weights_hashed_not_model_loaded": True,
        "known_development_scored": False,
        "deferred_final_generated": False,
        "runtime_promotion_authorized": False,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("seal", "authenticate"), nargs="?", default="authenticate")
    parser.add_argument("--config", default=str(CONFIG))
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    result = (
        seal_candidate_attestation_v96(args.config)
        if args.command == "seal"
        else authenticate_candidate_attestation_v96(args.config)
    )
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "ARTIFACT",
    "ATTESTATION",
    "STATUS",
    "authenticate_candidate_attestation_v96",
    "build_candidate_attestation_v96",
    "main",
    "physical_path_v96_v2",
    "read_json_strict_v96_v2",
    "seal_candidate_attestation_v96",
]
