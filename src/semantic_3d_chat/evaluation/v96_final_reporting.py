"""Fail-closed, aggregate-only V96 report and demo-handoff integration.

This module is deliberately post-release and model-free.  It authenticates
known-development, deferred-final, runtime-release, isolated-leakage, and
embodied-MCP readiness evidence in separate children, then renders a new V96
addendum.  It never rewrites the current V89 report or default demo.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Final

from semantic_3d_chat.config import PROJECT_ROOT

SCHEMA_VERSION: Final[int] = 96
ARTIFACT: Final[str] = "gemma4_v96_measured_integration_v1"
ROBOT_ARTIFACT: Final[str] = "gemma4_v96_embodied_mcp_preflight_evidence_v1"
SCENE_IDS: Final[tuple[str, ...]] = tuple(
    f"scene_{index:06d}" for index in range(25, 31)
)

KNOWN_SCORE: Final[Path] = (
    PROJECT_ROOT
    / "reports/gemma4/metrics/gemma4_v96_atomic_pair_repair_known_development.json"
)
KNOWN_EVIDENCE: Final[Path] = KNOWN_SCORE.with_name(
    "gemma4_v96_atomic_pair_repair_known_development_evidence.json"
)
TRAINING_REPORT: Final[Path] = (
    PROJECT_ROOT
    / "reports/gemma4/metrics/gemma4_v96_atomic_pair_repair_training.json"
)
DEFERRED_SCORE: Final[Path] = (
    PROJECT_ROOT / "reports/gemma4/metrics/gemma4_v96_deferred_final.json"
)
DEFERRED_EVIDENCE: Final[Path] = DEFERRED_SCORE.with_name(
    "gemma4_v96_deferred_final_evidence.json"
)
SMOKE_REPORT: Final[Path] = (
    PROJECT_ROOT / "reports/gemma4/metrics/gemma4_v96_strict_runtime_smoke.json"
)
RELEASE_REPORT: Final[Path] = (
    PROJECT_ROOT / "reports/gemma4/metrics/gemma4_v96_strict_runtime_release.json"
)
DEFAULT_ROBOT_EVIDENCE: Final[Path] = (
    PROJECT_ROOT
    / "reports/gemma4/metrics/gemma4_v96_embodied_mcp_preflight_evidence.json"
)
DEFAULT_LIVE_ROBOT_EVIDENCE: Final[Path] = (
    PROJECT_ROOT
    / "reports/gemma4/metrics/v96_explicit_candidate_embodied_mcp_live_smoke_scene_000025.json"
)
DEFAULT_METRICS_OUTPUT: Final[Path] = (
    PROJECT_ROOT / "reports/gemma4/metrics/gemma4_v96_measured_integration.json"
)
DEFAULT_MARKDOWN_OUTPUT: Final[Path] = (
    PROJECT_ROOT / "reports/gemma4/gemma4_v96_measured_report.md"
)
DEFAULT_LIVE_METRICS_OUTPUT: Final[Path] = (
    PROJECT_ROOT / "reports/gemma4/metrics/gemma4_v96_measured_live_integration.json"
)
DEFAULT_LIVE_MARKDOWN_OUTPUT: Final[Path] = (
    PROJECT_ROOT / "reports/gemma4/gemma4_v96_measured_live_report.md"
)

_SHA256: Final[frozenset[str]] = frozenset("0123456789abcdef")
_PROTECTED_PATH_COMPONENTS: Final[frozenset[str]] = frozenset(
    {"oracle", "qa", "questions", "predictions", "scorer", "scorer_only"}
)
_RELEASE_RECEIPT_FIELDS: Final[frozenset[str]] = frozenset(
    {
        "phase",
        "passed",
        "candidate_fingerprint_sha256",
        "candidate_attestation_file_sha256",
        "candidate_attestation_identity_sha256",
        "v1_implementation_seal_sha256",
        "v2_implementation_seal_sha256",
        "candidate_checkpoint_sha256",
        "candidate_adapter_sha256",
        "deferred_final_evidence_sha256",
        "runtime_smoke_sha256",
        "release_report_sha256",
        "release_checkpoint_sha256",
        "release_adapter_sha256",
        "v95_state_sha256",
        "v96_state_sha256",
        "runtime_implementation_inventory_sha256",
        "release_implementation_inventory_sha256",
        "scene_ids",
        "checks",
    }
)
_REQUIRED_RELEASE_CHECKS: Final[frozenset[str]] = frozenset(
    {
        "release_report_identity",
        "release_report_promoted",
        "exact_two_file_checkpoint",
        "checkpoint_fingerprint_matches_release",
        "adapter_byte_identical_to_smoked_candidate",
        "all_six_memory_tensor_files_byte_identical_to_candidate",
        "all_six_memories_bound_to_attested_prefixes",
        "all_six_runtime_maps_bound_to_smoked_bytes",
        "exact_ten_frozen_final_state_banks",
        "deferred_final_binding_exact",
        "candidate_attestation_binding_exact",
        "evaluator_implementation_binding_exact",
        "runtime_smoke_binding_exact",
        "runtime_implementation_binding_exact",
        "release_implementation_binding_exact",
        "runtime_promotion_authorized",
        "candidate_checkpoint_identity_retained_in_smoke",
        "default_runtime_pointer_unchanged",
    }
)
_AUTHENTICATOR_COMMANDS: Final[dict[str, tuple[str, ...]]] = {
    "known": (
        "-m",
        "semantic_3d_chat.evaluation.seal_v96_known_development_v2",
        "--config",
        "configs/experiments/gemma4_v96_atomic_pair_repair.yaml",
        "authenticate",
    ),
    "deferred": (
        "-m",
        "semantic_3d_chat.evaluation.authenticate_v96_deferred_final",
        "final",
    ),
    "release": (
        "-m",
        "semantic_3d_chat.evaluation.v96_strict_runtime_release",
        "verify",
    ),
}
_MCP_TOOLS: Final[tuple[str, ...]] = tuple(
    sorted(
        {
            "get_robot_state",
            "look",
            "turn",
            "move_forward",
            "move_backward",
            "move_to",
            "scan",
            "stop",
            "reset_scene",
        }
    )
)
_LIVE_HASH_FIELDS: Final[tuple[str, ...]] = (
    "map_sha256",
    "scene_prefix_sha256",
    "scene_control_signature_sha256",
    "binding_sha256",
    "active_prefix_sha256",
    "robot_state_sha256",
    "robot_tokens_sha256",
    "robot_state_encoder_sha256",
    "active_binding_sha256",
)
_LIVE_CHANGED_FIELDS: Final[tuple[str, ...]] = tuple(
    f"{stage}_changed_{field}"
    for stage in ("scan", "turn")
    for field in _LIVE_HASH_FIELDS
    if field != "robot_state_encoder_sha256"
)
_LIVE_EVIDENCE_FIELDS: Final[frozenset[str]] = frozenset(
    {
        "schema",
        "passed",
        "transport",
        "mcp_sdk_version",
        "protocol_version",
        "server_name",
        "server_version",
        "scene_id",
        "tool_count",
        "tools",
        "called_tools",
        "base_checkpoint",
        "scene_memory",
        "runtime_asset",
        "audit_report",
        "audit_report_sha256",
        "loaded_file_count",
        "forbidden_access_count",
        "mode",
        "promoted_runtime_release_verified_before_transport",
        "server_reauthenticates_promoted_release_before_model_load",
        "deferred_final_gate_passed",
        "runtime_leakage_gate_passed",
        "numeric_tool_outputs_only",
        "question_free_full_memory_refresh",
        "full_memory_tokens",
        "full_memory_recompiled_before_map_commit",
        "robot_state_numeric_binding_exercised",
        "language_questions_asked",
        "v96_answer_generation_exercised",
        "direct_v96_answer_robot_tokens_authenticated",
        "environmental_text_inputs",
        "semantic_result_leaks",
        "release_bindings",
        "elapsed_seconds",
        "initial_map_version",
        "scan_map_version",
        "turn_map_version",
        "explicit_scan_valid_depth_pixels",
        "turn_auto_scan_valid_depth_pixels",
        "initial_source_voxels",
        "scan_source_voxels",
        "turn_source_voxels",
        "initial_processed_voxels",
        "scan_processed_voxels",
        "turn_processed_voxels",
        "bounded_turn_degrees",
        "resulting_body_yaw_degrees",
        "robot_state_encoder_identity_invariant",
        "initial_hashes",
        "scan_hashes",
        "turn_hashes",
        *_LIVE_CHANGED_FIELDS,
    }
)


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and set(value).issubset(_SHA256)
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(value: object) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _strict_json_text(raw: str, *, purpose: str) -> dict[str, Any]:
    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"Duplicate {purpose} field: {key}")
            result[key] = value
        return result

    value = json.loads(raw, object_pairs_hook=reject_duplicates)
    if not isinstance(value, dict):
        raise TypeError(f"{purpose} must be one JSON object")
    return value


def _physical_file(path: Path, *, purpose: str) -> Path:
    unresolved = Path(os.path.abspath(path))
    if _PROTECTED_PATH_COMPONENTS.intersection(
        component.casefold() for component in unresolved.parts
    ):
        raise ValueError(f"{purpose} is under a protected evidence path")
    current = Path(unresolved.anchor)
    for component in unresolved.parts[1:]:
        current /= component
        if current.is_symlink():
            raise ValueError(f"{purpose} path contains a symbolic link: {current}")
    if not unresolved.is_file():
        raise FileNotFoundError(f"{purpose} is unavailable: {unresolved}")
    return unresolved


def _python_executable(path: Path) -> Path:
    """Validate a Python command while retaining its virtual-environment path."""

    unresolved = Path(os.path.abspath(path))
    try:
        target = unresolved.resolve(strict=True)
    except OSError as error:
        raise FileNotFoundError(f"V96 report Python is unavailable: {unresolved}") from error
    if not target.is_file() or not os.access(target, os.X_OK):
        raise ValueError(f"V96 report Python is not executable: {unresolved}")
    # Invoking the venv link (rather than its resolved base interpreter) preserves
    # pyvenv.cfg discovery and the exact installed package environment.
    return unresolved


def _read_json(path: Path, *, purpose: str) -> dict[str, Any]:
    source = _physical_file(path, purpose=purpose)
    return _strict_json_text(source.read_text(encoding="utf-8"), purpose=purpose)


def _run_json_command(kind: str, *, python: Path) -> dict[str, Any]:
    if kind not in _AUTHENTICATOR_COMMANDS:
        raise ValueError(f"Unknown V96 authenticator: {kind}")
    executable = _python_executable(python)
    environment = dict(os.environ)
    environment.update(
        {
            "PYTHONPATH": str(PROJECT_ROOT / "src"),
            "TRANSFORMERS_OFFLINE": "1",
            "HF_HUB_OFFLINE": "1",
        }
    )
    completed = subprocess.run(
        [str(executable), *_AUTHENTICATOR_COMMANDS[kind]],
        cwd=PROJECT_ROOT,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0:
        detail = completed.stderr.strip() or f"V96 {kind} authentication failed"
        raise RuntimeError(detail)
    return _strict_json_text(completed.stdout, purpose=f"V96 {kind} receipt")


def _mapping(value: object, *, purpose: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{purpose} must be a mapping")
    return value


def _all_true(value: object, *, purpose: str) -> dict[str, bool]:
    gates = _mapping(value, purpose=purpose)
    if not gates or any(type(flag) is not bool for flag in gates.values()):
        raise ValueError(f"{purpose} must contain only boolean gates")
    result = {str(key): bool(flag) for key, flag in gates.items()}
    if not all(result.values()):
        raise ValueError(f"{purpose} did not pass exactly")
    return result


def _finite_tree(value: object, *, purpose: str) -> None:
    if isinstance(value, Mapping):
        for key, nested in value.items():
            if not isinstance(key, str):
                raise TypeError(f"{purpose} contains a non-string key")
            _finite_tree(nested, purpose=f"{purpose}.{key}")
    elif isinstance(value, list):
        for index, nested in enumerate(value):
            _finite_tree(nested, purpose=f"{purpose}[{index}]")
    elif isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"{purpose} contains NaN or infinity")
    elif value is not None and not isinstance(value, (str, int, float, bool)):
        raise TypeError(f"{purpose} contains a non-JSON value")


def _validate_release_receipt(receipt: Mapping[str, Any]) -> dict[str, Any]:
    if set(receipt) != set(_RELEASE_RECEIPT_FIELDS):
        raise ValueError("V96 release receipt field inventory changed")
    checks = _all_true(receipt.get("checks"), purpose="V96 release checks")
    hashes = _RELEASE_RECEIPT_FIELDS - {"phase", "passed", "scene_ids", "checks"}
    if (
        receipt.get("phase") != "v96_strict_runtime_release_verified"
        or receipt.get("passed") is not True
        or receipt.get("scene_ids") != list(SCENE_IDS)
        or set(checks) != set(_REQUIRED_RELEASE_CHECKS)
        or any(not _is_sha256(receipt.get(field)) for field in hashes)
    ):
        raise ValueError("V96 promoted release receipt is incomplete")
    return {**dict(receipt), "checks": checks}


def _validate_robot_evidence(
    evidence: Mapping[str, Any], *, release: Mapping[str, Any]
) -> dict[str, Any]:
    required = {
        "artifact",
        "schema_version",
        "status",
        "scene_id",
        "candidate_fingerprint_sha256",
        "release_checkpoint_sha256",
        "release_adapter_sha256",
        "deferred_final_evidence_sha256",
        "runtime_smoke_sha256",
        "runtime_implementation_inventory_sha256",
        "preflight_summary_sha256",
        "access_audit_sha256",
        "mcp_tool_count",
        "strict_input_schemas",
        "fixed_memory_tokens",
        "frozen_lora_bank_count",
        "numeric_tool_outputs_only",
        "full_memory_recompiled_before_map_commit",
        "language_model_loaded",
        "blender_started",
        "robot_or_map_state_changed",
        "navigation_actions_executed",
        "navigation_success_measured",
        "direct_v96_answer_robot_tokens_authenticated",
        "forbidden_access_count",
        "environmental_text_inputs",
        "oracle_inputs_at_runtime",
        "passed",
    }
    hash_fields = {
        "candidate_fingerprint_sha256",
        "release_checkpoint_sha256",
        "release_adapter_sha256",
        "deferred_final_evidence_sha256",
        "runtime_smoke_sha256",
        "runtime_implementation_inventory_sha256",
        "preflight_summary_sha256",
        "access_audit_sha256",
    }
    if (
        set(evidence) != required
        or evidence.get("artifact") != ROBOT_ARTIFACT
        or evidence.get("schema_version") != SCHEMA_VERSION
        or evidence.get("status")
        != "authenticated_model_free_embodied_mcp_preflight_only"
        or evidence.get("passed") is not True
        or any(not _is_sha256(evidence.get(field)) for field in hash_fields)
        or evidence.get("candidate_fingerprint_sha256")
        != release["candidate_fingerprint_sha256"]
        or evidence.get("release_checkpoint_sha256")
        != release["release_checkpoint_sha256"]
        or evidence.get("release_adapter_sha256")
        != release["release_adapter_sha256"]
        or evidence.get("deferred_final_evidence_sha256")
        != release["deferred_final_evidence_sha256"]
        or evidence.get("runtime_smoke_sha256") != release["runtime_smoke_sha256"]
        or evidence.get("runtime_implementation_inventory_sha256")
        != release["runtime_implementation_inventory_sha256"]
        or evidence.get("mcp_tool_count") != 9
        or evidence.get("strict_input_schemas") is not True
        or evidence.get("fixed_memory_tokens") != 738
        or evidence.get("frozen_lora_bank_count") != 10
        or evidence.get("numeric_tool_outputs_only") is not True
        or evidence.get("full_memory_recompiled_before_map_commit") is not True
        or evidence.get("language_model_loaded") is not False
        or evidence.get("blender_started") is not False
        or evidence.get("robot_or_map_state_changed") is not False
        or evidence.get("navigation_actions_executed") is not False
        or evidence.get("navigation_success_measured") is not False
        or evidence.get("direct_v96_answer_robot_tokens_authenticated") is not False
        or evidence.get("forbidden_access_count") != 0
        or evidence.get("environmental_text_inputs") != []
        or evidence.get("oracle_inputs_at_runtime") is not False
    ):
        raise ValueError("V96 robot evidence is not exact model-free readiness evidence")
    return dict(evidence)


def _validate_live_robot_evidence(
    evidence: Mapping[str, Any], *, release: Mapping[str, Any]
) -> dict[str, Any]:
    """Authenticate finite numeric MCP refresh evidence without overstating it."""

    if set(evidence) != set(_LIVE_EVIDENCE_FIELDS):
        raise ValueError("V96 live robot evidence field inventory changed")
    release_bindings = _mapping(
        evidence.get("release_bindings"), purpose="V96 live release bindings"
    )
    expected_bindings = {
        "candidate_fingerprint_sha256": release["candidate_fingerprint_sha256"],
        "deferred_final_evidence_sha256": release[
            "deferred_final_evidence_sha256"
        ],
        "runtime_smoke_sha256": release["runtime_smoke_sha256"],
        "release_checkpoint_sha256": release["release_checkpoint_sha256"],
        "release_adapter_sha256": release["release_adapter_sha256"],
        "v95_state_sha256": release["v95_state_sha256"],
        "v96_state_sha256": release["v96_state_sha256"],
        "runtime_implementation_inventory_sha256": release[
            "runtime_implementation_inventory_sha256"
        ],
    }
    numeric_counts = (
        "loaded_file_count",
        "explicit_scan_valid_depth_pixels",
        "turn_auto_scan_valid_depth_pixels",
        "initial_source_voxels",
        "scan_source_voxels",
        "turn_source_voxels",
        "initial_processed_voxels",
        "scan_processed_voxels",
        "turn_processed_voxels",
    )
    if (
        evidence.get("schema")
        != "semantic_3d_chat.v96_candidate_mcp_live_smoke.v1"
        or evidence.get("passed") is not True
        or evidence.get("transport") != "stdio"
        or evidence.get("scene_id") != SCENE_IDS[0]
        or evidence.get("tool_count") != 9
        or evidence.get("tools") != list(_MCP_TOOLS)
        or evidence.get("called_tools")
        != ["get_robot_state", "scan", "turn"]
        or evidence.get("mode")
        != "explicit_v96_candidate_overlay_after_promoted_release"
        or evidence.get("promoted_runtime_release_verified_before_transport")
        is not True
        or evidence.get("server_reauthenticates_promoted_release_before_model_load")
        is not True
        or evidence.get("deferred_final_gate_passed") is not True
        or evidence.get("runtime_leakage_gate_passed") is not True
        or evidence.get("numeric_tool_outputs_only") is not True
        or evidence.get("question_free_full_memory_refresh") is not True
        or evidence.get("full_memory_tokens") != 738
        or evidence.get("full_memory_recompiled_before_map_commit") is not True
        or evidence.get("robot_state_numeric_binding_exercised") is not True
        or evidence.get("language_questions_asked") != 0
        or evidence.get("v96_answer_generation_exercised") is not False
        or evidence.get("direct_v96_answer_robot_tokens_authenticated") is not False
        or evidence.get("environmental_text_inputs") != []
        or evidence.get("semantic_result_leaks") != []
        or evidence.get("forbidden_access_count") != 0
        or dict(release_bindings) != expected_bindings
        or any(
            isinstance(evidence.get(field), bool)
            or not isinstance(evidence.get(field), int)
            or int(evidence[field]) < 1
            for field in numeric_counts
        )
        or any(evidence.get(field) is not True for field in _LIVE_CHANGED_FIELDS)
        or evidence.get("robot_state_encoder_identity_invariant") is not True
    ):
        raise ValueError("V96 live numeric MCP evidence did not pass exactly")

    initial_version = evidence.get("initial_map_version")
    scan_version = evidence.get("scan_map_version")
    turn_version = evidence.get("turn_map_version")
    if (
        any(
            isinstance(value, bool) or not isinstance(value, int)
            for value in (initial_version, scan_version, turn_version)
        )
        or int(initial_version) < 0
        or int(scan_version) != int(initial_version) + 1
        or int(turn_version) != int(scan_version) + 1
    ):
        raise ValueError("V96 live numeric MCP map versions are not consecutive")
    for field in ("elapsed_seconds", "bounded_turn_degrees", "resulting_body_yaw_degrees"):
        value = evidence.get(field)
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
        ):
            raise ValueError(f"V96 live numeric MCP field is invalid: {field}")
    if float(evidence["elapsed_seconds"]) <= 0.0 or not (
        0.0 < abs(float(evidence["bounded_turn_degrees"])) <= 180.0
    ):
        raise ValueError("V96 live numeric MCP timing or bounded turn is invalid")

    episode_hashes: dict[str, Mapping[str, Any]] = {}
    for stage in ("initial", "scan", "turn"):
        hashes = _mapping(
            evidence.get(f"{stage}_hashes"),
            purpose=f"V96 live {stage} hashes",
        )
        if set(hashes) != set(_LIVE_HASH_FIELDS) or any(
            not _is_sha256(hashes.get(field)) for field in _LIVE_HASH_FIELDS
        ):
            raise ValueError(f"V96 live {stage} hash inventory changed")
        episode_hashes[stage] = hashes
    encoder_field = "robot_state_encoder_sha256"
    if len({hashes[encoder_field] for hashes in episode_hashes.values()}) != 1:
        raise ValueError("V96 live robot-state encoder identity changed")
    for stage, before, after in (
        ("scan", "initial", "scan"),
        ("turn", "scan", "turn"),
    ):
        for field in _LIVE_HASH_FIELDS:
            if field == encoder_field:
                continue
            changed = episode_hashes[after][field] != episode_hashes[before][field]
            if evidence.get(f"{stage}_changed_{field}") is not changed or not changed:
                raise ValueError(
                    "V96 live numeric MCP changed-hash claims are inconsistent"
                )

    audit_value = evidence.get("audit_report")
    if not isinstance(audit_value, str):
        raise TypeError("V96 live MCP audit path is missing")
    audit_path = Path(audit_value)
    if not audit_path.is_absolute():
        raise ValueError("V96 live MCP audit path must be absolute")
    audit = _read_json(audit_path, purpose="V96 live MCP lifetime audit")
    loaded_files = audit.get("loaded_files")
    if (
        set(audit)
        != {
            "loaded_files",
            "forbidden_roots",
            "forbidden_component_names",
            "block_forbidden",
            "forbidden_accesses",
            "passed",
        }
        or audit.get("passed") is not True
        or audit.get("block_forbidden") is not True
        or audit.get("forbidden_accesses") != []
        or not isinstance(loaded_files, list)
        or not all(isinstance(item, str) for item in loaded_files)
        or len(set(loaded_files)) != len(loaded_files)
        or len(loaded_files) != evidence["loaded_file_count"]
        or not _is_sha256(evidence.get("audit_report_sha256"))
        or _sha256(audit_path) != evidence["audit_report_sha256"]
    ):
        raise ValueError("V96 live MCP lifetime audit did not authenticate")
    return dict(evidence)


def build_robot_preflight_evidence(
    preflight: Mapping[str, Any],
    access_audit: Mapping[str, Any],
    *,
    access_audit_path: Path,
    release_receipt: Mapping[str, Any],
) -> dict[str, Any]:
    """Reduce an audited MCP preflight to a sanitized, release-bound receipt."""

    release = _validate_release_receipt(release_receipt)
    candidate = _mapping(
        preflight.get("v96_explicit_candidate"),
        purpose="V96 embodied candidate summary",
    )
    protocol = _mapping(preflight.get("action_protocol"), purpose="V96 action protocol")
    expected_audit_fields = {
        "loaded_files",
        "forbidden_roots",
        "forbidden_component_names",
        "block_forbidden",
        "forbidden_accesses",
        "passed",
    }
    audit_source = _physical_file(
        access_audit_path, purpose="V96 embodied access audit"
    )
    if _read_json(audit_source, purpose="V96 embodied access audit") != dict(
        access_audit
    ):
        raise ValueError("V96 embodied access audit bytes do not match the receipt")
    preflight_audit = preflight.get("audit_report")
    loaded_files = access_audit.get("loaded_files")
    if (
        preflight.get("schema") != "semantic_3d_chat.embodied_mcp_preflight.v1"
        or preflight.get("phase") != "embodied_mcp_preflight"
        or preflight.get("passed") is not True
        or preflight.get("mode")
        != "semantic_continuous_map_v96_explicit_candidate"
        or not isinstance(preflight.get("scene_id"), str)
        or preflight.get("loads_language_model") is not False
        or preflight.get("loads_blender") is not False
        or preflight.get("starts_transport") is not False
        or preflight.get("changes_robot_or_map_state") is not False
        or preflight.get("environmental_text_inputs") != []
        or preflight.get("oracle_inputs_at_runtime") is not False
        or preflight.get("scene_prefix_computation_deferred_to_live_startup") is not True
        or isinstance(preflight.get("loaded_file_count"), bool)
        or not isinstance(preflight.get("loaded_file_count"), int)
        or preflight.get("loaded_file_count", -1) < 1
        or preflight.get("forbidden_access_count") != 0
        or not isinstance(preflight_audit, str)
        or Path(os.path.abspath(preflight_audit)) != audit_source
        or protocol.get("tool_count") != 9
        or protocol.get("strict_input_schemas") is not True
        or set(access_audit) != expected_audit_fields
        or access_audit.get("passed") is not True
        or access_audit.get("block_forbidden") is not True
        or access_audit.get("forbidden_accesses") != []
        or not isinstance(loaded_files, list)
        or not all(isinstance(path, str) for path in loaded_files)
        or len(set(loaded_files)) != len(loaded_files)
        or preflight.get("loaded_file_count") != len(loaded_files)
        or candidate.get("candidate_fingerprint_sha256")
        != release["candidate_fingerprint_sha256"]
        or candidate.get("known_development_gate_passed") is not True
        or candidate.get("pass_evidence_authenticated") is not True
        or candidate.get("deferred_final_gate_passed") is not True
        or candidate.get("runtime_leakage_smoke_passed") is not True
        or candidate.get("promoted_runtime_release_verified") is not True
        or candidate.get("deferred_final_evidence_sha256")
        != release["deferred_final_evidence_sha256"]
        or candidate.get("runtime_leakage_smoke_sha256")
        != release["runtime_smoke_sha256"]
        or candidate.get("runtime_implementation_inventory_sha256")
        != release["runtime_implementation_inventory_sha256"]
        or candidate.get("verified_release_checkpoint_sha256")
        != release["release_checkpoint_sha256"]
        or candidate.get("verified_release_adapter_sha256")
        != release["release_adapter_sha256"]
        or candidate.get("fixed_memory_tokens") != 738
        or candidate.get("frozen_lora_bank_count") != 10
        or candidate.get("numeric_tool_outputs_only") is not True
        or candidate.get("full_memory_recompiled_before_map_commit") is not True
        or candidate.get("direct_v96_answer_robot_tokens_authenticated") is not False
        or candidate.get("environmental_text_inputs") != []
        or candidate.get("oracle_inputs_at_runtime") is not False
    ):
        raise ValueError("V96 embodied MCP preflight did not authenticate exactly")
    access_audit_sha256 = _sha256(audit_source)
    payload = {
        "artifact": ROBOT_ARTIFACT,
        "schema_version": SCHEMA_VERSION,
        "status": "authenticated_model_free_embodied_mcp_preflight_only",
        "scene_id": preflight["scene_id"],
        "candidate_fingerprint_sha256": release["candidate_fingerprint_sha256"],
        "release_checkpoint_sha256": release["release_checkpoint_sha256"],
        "release_adapter_sha256": release["release_adapter_sha256"],
        "deferred_final_evidence_sha256": release["deferred_final_evidence_sha256"],
        "runtime_smoke_sha256": release["runtime_smoke_sha256"],
        "runtime_implementation_inventory_sha256": release[
            "runtime_implementation_inventory_sha256"
        ],
        "preflight_summary_sha256": _canonical_sha256(preflight),
        "access_audit_sha256": access_audit_sha256,
        "mcp_tool_count": 9,
        "strict_input_schemas": True,
        "fixed_memory_tokens": 738,
        "frozen_lora_bank_count": 10,
        "numeric_tool_outputs_only": True,
        "full_memory_recompiled_before_map_commit": True,
        "language_model_loaded": False,
        "blender_started": False,
        "robot_or_map_state_changed": False,
        "navigation_actions_executed": False,
        "navigation_success_measured": False,
        "direct_v96_answer_robot_tokens_authenticated": False,
        "forbidden_access_count": 0,
        "environmental_text_inputs": [],
        "oracle_inputs_at_runtime": False,
        "passed": True,
    }
    return _validate_robot_evidence(payload, release=release)


def collect_v96_measured_integration(
    *,
    python: Path,
    robot_evidence_path: Path = DEFAULT_ROBOT_EVIDENCE,
    live_robot_evidence_path: Path | None = None,
) -> dict[str, Any]:
    """Authenticate and collect aggregate evidence without loading a model."""

    known = _run_json_command("known", python=python)
    deferred = _run_json_command("deferred", python=python)
    release = _validate_release_receipt(_run_json_command("release", python=python))
    known_score = _read_json(KNOWN_SCORE, purpose="V96 known-development score")
    known_evidence = _read_json(
        KNOWN_EVIDENCE, purpose="V96 known-development evidence"
    )
    training = _read_json(TRAINING_REPORT, purpose="V96 training report")
    deferred_score = _read_json(DEFERRED_SCORE, purpose="V96 deferred-final score")
    deferred_evidence = _read_json(
        DEFERRED_EVIDENCE, purpose="V96 deferred-final evidence"
    )
    smoke = _read_json(SMOKE_REPORT, purpose="V96 isolated runtime smoke")
    promoted = _read_json(RELEASE_REPORT, purpose="V96 promoted release report")
    robot = _validate_robot_evidence(
        _read_json(robot_evidence_path, purpose="V96 robot evidence"),
        release=release,
    )
    live_robot = (
        None
        if live_robot_evidence_path is None
        else _validate_live_robot_evidence(
            _read_json(
                live_robot_evidence_path,
                purpose="V96 live numeric MCP evidence",
            ),
            release=release,
        )
    )

    known_gates = _all_true(
        known_score.get("gate_results"), purpose="V96 known-development gates"
    )
    deferred_gates = _all_true(
        deferred_score.get("gate_results"), purpose="V96 deferred-final gates"
    )
    smoke_gates = _all_true(smoke.get("gates"), purpose="V96 leakage-smoke gates")
    release_gates = _all_true(
        release.get("checks"), purpose="V96 promoted-release checks"
    )
    training_gates = _all_true(training.get("gates"), purpose="V96 training gates")
    promoted_bindings = _mapping(
        promoted.get("bindings"), purpose="V96 promoted-release bindings"
    )
    smoke_scenes = _mapping(smoke.get("scenes"), purpose="V96 smoke scenes")
    if (
        known.get("authenticated") is not True
        or known.get("known_development_gate_passed") is not True
        or known.get("status") != "passed_deferred_final_explicit_unlock_eligible"
        or known.get("scene_prefix_question_independent") is not True
        or known.get("protected_read_count") != 0
        or known.get("row_level_content_serialized") is not False
        or known.get("runtime_promotion_authorized") is not False
        or _sha256(KNOWN_SCORE) != known.get("final_score_sha256")
        or _sha256(KNOWN_EVIDENCE) != known.get("evidence_sha256")
        or known_score.get("candidate_fingerprint_sha256")
        != release["candidate_fingerprint_sha256"]
        or known_evidence.get("candidate_fingerprint_sha256")
        != release["candidate_fingerprint_sha256"]
        or known.get("candidate_attestation_file_sha256")
        != release["candidate_attestation_file_sha256"]
        or known.get("candidate_attestation_identity_sha256")
        != release["candidate_attestation_identity_sha256"]
        or known.get("v1_implementation_seal_sha256")
        != release["v1_implementation_seal_sha256"]
        or known.get("implementation_seal_sha256")
        != release["v2_implementation_seal_sha256"]
        or _sha256(TRAINING_REPORT) != known.get("training_report_sha256")
        or training.get("status") != "fixed_final_training_complete_not_promoted"
        or training.get("protected_read_count") != 0
        or training.get("known_development_labels_loaded") is not False
        or training.get("known_development_questions_loaded") is not False
        or training.get("deferred_final_generated") is not False
        or training.get("oracle_loaded") is not False
        or training.get("runtime_promotion_authorized") is not False
        or training.get("device") != "mps"
        or not isinstance(training.get("elapsed_seconds"), (int, float))
        or isinstance(training.get("elapsed_seconds"), bool)
        or not math.isfinite(float(training["elapsed_seconds"]))
        or float(training["elapsed_seconds"]) <= 0.0
        or deferred.get("authenticated") is not True
        or deferred.get("deferred_final_gate_passed") is not True
        or deferred.get("status") != "passed_deferred_final_not_runtime_promoted"
        or deferred.get("question_label_isolation_proven") is not True
        or deferred.get("prefix_hash_invariant") is not True
        or deferred.get("protected_read_count") != 0
        or deferred.get("row_level_content_serialized") is not False
        or deferred.get("runtime_promotion_authorized") is not False
        or _sha256(DEFERRED_SCORE) != deferred.get("final_score_sha256")
        or _sha256(DEFERRED_EVIDENCE) != deferred.get("evidence_file_sha256")
        or deferred.get("evidence_file_sha256")
        != release["deferred_final_evidence_sha256"]
        or deferred_score.get("candidate_fingerprint_sha256")
        != release["candidate_fingerprint_sha256"]
        or deferred_evidence.get("candidate_fingerprint_sha256")
        != release["candidate_fingerprint_sha256"]
        or deferred.get("candidate_attestation_file_sha256")
        != release["candidate_attestation_file_sha256"]
        or deferred.get("candidate_attestation_identity_sha256")
        != release["candidate_attestation_identity_sha256"]
        or deferred.get("v1_implementation_seal_sha256")
        != release["v1_implementation_seal_sha256"]
        or deferred.get("v2_implementation_seal_sha256")
        != release["v2_implementation_seal_sha256"]
        or _sha256(SMOKE_REPORT) != release["runtime_smoke_sha256"]
        or smoke.get("passed") is not True
        or smoke.get("promotion_authorized") is not True
        or smoke.get("scene_ids") != list(SCENE_IDS)
        or smoke.get("candidate_attestation_file_sha256")
        != release["candidate_attestation_file_sha256"]
        or smoke.get("candidate_attestation_identity_sha256")
        != release["candidate_attestation_identity_sha256"]
        or set(smoke_scenes) != set(SCENE_IDS)
        or smoke.get("expected_answers_supplied_to_children") is not False
        or smoke.get("behavior_assertions_in_children") is not False
        or _sha256(RELEASE_REPORT) != release["release_report_sha256"]
        or promoted.get("all_release_gates_passed") is not True
        or promoted.get("promotion_decision")
        != "strict_v96_deferred_final_primary"
        or promoted.get("scene_ids") != list(SCENE_IDS)
        or promoted.get("default_runtime_pointer_modified") is not False
        or promoted_bindings.get("runtime_smoke_sha256")
        != release["runtime_smoke_sha256"]
        or promoted_bindings.get("deferred_final_evidence_sha256")
        != release["deferred_final_evidence_sha256"]
        or promoted_bindings.get("runtime_implementation_inventory_sha256")
        != release["runtime_implementation_inventory_sha256"]
        or promoted_bindings.get("release_implementation_inventory_sha256")
        != release["release_implementation_inventory_sha256"]
        or promoted_bindings.get("candidate_attestation_file_sha256")
        != release["candidate_attestation_file_sha256"]
        or promoted_bindings.get("candidate_attestation_identity_sha256")
        != release["candidate_attestation_identity_sha256"]
        or promoted_bindings.get("v1_implementation_seal_sha256")
        != release["v1_implementation_seal_sha256"]
        or promoted_bindings.get("v2_implementation_seal_sha256")
        != release["v2_implementation_seal_sha256"]
        or known_score.get("row_count") != 216
        or known_score.get("scene_count") != 6
        or deferred_score.get("row_count") != 216
    ):
        raise ValueError("V96 measured integration evidence is incomplete or inconsistent")

    structured_metrics = _mapping(
        known_score.get("structured_metrics"),
        purpose="known-development structured metrics",
    )
    known_nll = _mapping(
        known_score.get("nll_metrics"), purpose="known-development NLL metrics"
    )
    deferred_metrics = _mapping(
        deferred_score.get("metrics"), purpose="deferred-final metrics"
    )
    for purpose, value in (
        ("known-development structured metrics", structured_metrics),
        ("known-development NLL metrics", known_nll),
        ("deferred-final metrics", deferred_metrics),
    ):
        _finite_tree(value, purpose=purpose)

    limitations = [
        (
            "Finite live numeric MCP scan/turn evidence was not supplied; only model-free MCP readiness was authenticated."
            if live_robot is None
            else "The finite live MCP result measures numeric transport and continuous-memory refresh, not conversational navigation success."
        ),
        "Direct V96 answer generation with robot-state tokens remains unauthenticated and disabled.",
        "Peak training memory was not recorded by the V96 training report.",
        "The project-wide default remains the previously promoted V89 path.",
    ]
    payload = {
        "artifact": ARTIFACT,
        "schema_version": SCHEMA_VERSION,
        "status": "authenticated_post_release_measured_addendum",
        "candidate_fingerprint_sha256": release["candidate_fingerprint_sha256"],
        "candidate_attestation_file_sha256": release[
            "candidate_attestation_file_sha256"
        ],
        "candidate_attestation_identity_sha256": release[
            "candidate_attestation_identity_sha256"
        ],
        "training": {
            "device": training.get("device"),
            "model_id": training.get("model_id"),
            "model_revision": training.get("model_revision"),
            "elapsed_seconds": training.get("elapsed_seconds"),
            "optimizer_updates": training.get("optimizer_updates"),
            "micro_steps_consumed": training.get("micro_steps_consumed"),
            "unique_training_rows": training.get("unique_training_rows"),
            "training_scene_count": training.get("training_scene_count"),
            "total_nll_forwards": training.get("total_nll_forwards"),
            "trainable_parameter_count": _mapping(
                training.get("trainable_bridge"), purpose="trainable bridge"
            ).get("parameter_count"),
            "strict_input_contract": training.get("strict_input_contract"),
            "gates": training_gates,
            "training_report_sha256": _sha256(TRAINING_REPORT),
            "peak_memory_measured": False,
        },
        "known_development": {
            "row_count": known_score.get("row_count"),
            "scene_count": known_score.get("scene_count"),
            "structured_metrics": dict(structured_metrics),
            "nll_metrics": dict(known_nll),
            "gates": known_gates,
            "score_sha256": _sha256(KNOWN_SCORE),
            "evidence_sha256": _sha256(KNOWN_EVIDENCE),
        },
        "deferred_final": {
            "row_count": deferred_score.get("row_count"),
            "metrics": dict(deferred_metrics),
            "gates": deferred_gates,
            "score_sha256": _sha256(DEFERRED_SCORE),
            "evidence_sha256": _sha256(DEFERRED_EVIDENCE),
        },
        "runtime_release": {
            "scene_ids": list(SCENE_IDS),
            "scene_count": len(SCENE_IDS),
            "strict_input_contract": promoted.get("strict_input_contract"),
            "checkpoint_sha256": release["release_checkpoint_sha256"],
            "adapter_sha256": release["release_adapter_sha256"],
            "runtime_implementation_inventory_sha256": release[
                "runtime_implementation_inventory_sha256"
            ],
            "release_implementation_inventory_sha256": release[
                "release_implementation_inventory_sha256"
            ],
            "checks": release_gates,
            "default_runtime_pointer_modified": False,
        },
        "runtime_leakage": {
            "scene_ids": list(SCENE_IDS),
            "child_process_count": len(smoke.get("scenes", {})),
            "questions_per_scene": len(smoke.get("questions", [])),
            "expected_answers_supplied_to_children": False,
            "gates": smoke_gates,
            "smoke_report_sha256": _sha256(SMOKE_REPORT),
        },
        "robot": robot,
        "live_robot": live_robot,
        "claims": {
            "known_development_passed": True,
            "held_out_deferred_final_passed": True,
            "static_runtime_release_promoted": True,
            "oracle_unavailable_runtime_smoke_passed": True,
            "embodied_mcp_model_free_preflight_passed": True,
            "embodied_numeric_mcp_refresh_measured": live_robot is not None,
            "embodied_navigation_success_measured": False,
            "direct_v96_answer_robot_tokens_authenticated": False,
            "current_project_default_changed": False,
        },
        "limitations": limitations,
        "aggregate_only": True,
        "row_level_content_serialized": False,
        "environmental_text_inputs_supplied_to_runtime": [],
        "bindings": {
            "deferred_final_evidence_sha256": release[
                "deferred_final_evidence_sha256"
            ],
            "runtime_smoke_sha256": release["runtime_smoke_sha256"],
            "release_report_sha256": release["release_report_sha256"],
            "candidate_attestation_file_sha256": release[
                "candidate_attestation_file_sha256"
            ],
            "candidate_attestation_identity_sha256": release[
                "candidate_attestation_identity_sha256"
            ],
            "v1_implementation_seal_sha256": release[
                "v1_implementation_seal_sha256"
            ],
            "v2_implementation_seal_sha256": release[
                "v2_implementation_seal_sha256"
            ],
            "robot_evidence_sha256": _sha256(robot_evidence_path),
            "live_robot_evidence_sha256": (
                None
                if live_robot_evidence_path is None
                else _sha256(live_robot_evidence_path)
            ),
        },
    }
    _finite_tree(payload, purpose="V96 measured integration")
    payload["integration_identity_sha256"] = _canonical_sha256(payload)
    return payload


def render_markdown(payload: Mapping[str, Any]) -> str:
    """Render a measured addendum without mutating the current final report."""

    training = _mapping(payload["training"], purpose="training summary")
    known = _mapping(payload["known_development"], purpose="known summary")
    deferred = _mapping(payload["deferred_final"], purpose="deferred summary")
    leakage = _mapping(payload["runtime_leakage"], purpose="leakage summary")
    robot = _mapping(payload["robot"], purpose="robot summary")
    raw_live_robot = payload.get("live_robot")
    if raw_live_robot is not None and not isinstance(raw_live_robot, Mapping):
        raise TypeError("V96 live robot summary is malformed")
    live_robot = raw_live_robot
    limitations = payload.get("limitations")
    if not isinstance(limitations, list) or not all(
        isinstance(item, str) for item in limitations
    ):
        raise TypeError("V96 report limitations are malformed")
    known_json = json.dumps(
        {
            "structured_metrics": known["structured_metrics"],
            "nll_metrics": known["nll_metrics"],
        },
        indent=2,
        sort_keys=True,
        allow_nan=False,
    )
    deferred_json = json.dumps(
        deferred["metrics"], indent=2, sort_keys=True, allow_nan=False
    )
    limitation_lines = "\n".join(f"- {item}" for item in limitations)
    live_outcome = (
        "- Finite numeric MCP scan/turn refresh: not supplied"
        if live_robot is None
        else (
            "- Finite numeric MCP scan/turn refresh: PASS "
            f"({live_robot['initial_map_version']} -> "
            f"{live_robot['scan_map_version']} -> {live_robot['turn_map_version']})"
        )
    )
    live_detail = (
        "No finite live robot result was supplied to this model-free report."
        if live_robot is None
        else (
            "A real official-SDK stdio session called `get_robot_state`, `scan`, "
            "and `turn`; it observed consecutive map versions, positive metric "
            "depth, and changed map/prefix/control/robot bindings with an invariant "
            "robot-state encoder."
        )
    )
    return f"""# Gemma 4 V96 measured integration addendum

Status: `{payload['status']}`

This addendum is generated only after authenticating the V96 known-development
gate, sealed held-out deferred-final gate, promoted static runtime, six-scene
oracle-unavailable leakage smoke, and model-free embodied MCP preflight. It does
not make unsupported claims. This addendum does not replace `reports/final_report.md`
or change the project default.

## Outcome

- Known-development gate: PASS ({known['row_count']} rows, {known['scene_count']} scenes)
- Held-out deferred-final gate: PASS ({deferred['row_count']} rows)
- Static continuous-memory runtime release: PASS ({len(SCENE_IDS)} held-out scenes)
- Oracle-unavailable child-process smoke: PASS ({leakage['child_process_count']} children)
- Embodied MCP readiness: PASS ({robot['mcp_tool_count']} bounded numeric tools)
{live_outcome}
- V96 navigation success measured: **no**

## Training

- Device: `{training['device']}`
- Optimizer updates: {training['optimizer_updates']}
- Micro-steps: {training['micro_steps_consumed']}
- Trainable parameters: {training['trainable_parameter_count']}
- Elapsed seconds: {training['elapsed_seconds']}
- Peak memory: not measured

## Known-development measurements

```json
{known_json}
```

## Held-out deferred-final measurements

```json
{deferred_json}
```

## Runtime and leakage evidence

The released input is the exact fixed 738-token continuous scene memory for each
scene. Every question reuses the same precomputed memory. The runtime smoke ran
with oracle directories physically unavailable, supplied no expected answers to
children, and passed all {len(leakage['gates'])} authenticated gates.

## Embodied evidence

The model-free preflight authenticated the promoted V96 release before model
load, nine bounded MCP tools, numeric-only tool results, a complete-memory
recompiler-before-map-commit contract, and zero forbidden reads. It deliberately
did not load Gemma, start Blender, execute actions, or measure navigation.

{live_detail}

## Limitations

{limitation_lines}

## Evidence identity

`{payload['integration_identity_sha256']}`
"""


def _write_exact(path: Path, payload: bytes) -> bool:
    if path.is_symlink() or (path.exists() and not path.is_file()):
        raise ValueError(f"V96 report output is unsafe: {path}")
    if path.is_file():
        if path.read_bytes() != payload:
            raise FileExistsError(f"Existing V96 report differs: {path}")
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, raw_temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(raw_temporary)
    linked = False
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path, follow_symlinks=False)
        except FileExistsError:
            if path.is_symlink() or not path.is_file() or path.read_bytes() != payload:
                raise
            temporary.unlink()
            return False
        linked = True
        temporary.unlink()
        directory_descriptor = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
        return True
    except BaseException:
        if linked and path.is_file() and path.read_bytes() == payload:
            path.unlink()
            try:
                directory_descriptor = os.open(path.parent, os.O_RDONLY)
                try:
                    os.fsync(directory_descriptor)
                finally:
                    os.close(directory_descriptor)
            except OSError:
                pass
        raise
    finally:
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            pass


def write_robot_evidence(path: Path, payload: Mapping[str, Any]) -> None:
    encoded = (
        json.dumps(dict(payload), indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")
    _write_exact(path, encoded)


def build_report_outputs(
    payload: Mapping[str, Any], *, metrics_output: Path, markdown_output: Path
) -> dict[str, Any]:
    metrics = (
        json.dumps(dict(payload), indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")
    markdown = render_markdown(payload).encode("utf-8")
    if metrics_output.resolve() == markdown_output.resolve():
        raise ValueError("V96 JSON and Markdown outputs must be distinct")
    # Validate both destinations before creating either half of the pair.
    for path, content in ((metrics_output, metrics), (markdown_output, markdown)):
        if path.is_symlink() or (path.exists() and not path.is_file()):
            raise ValueError(f"V96 report output is unsafe: {path}")
        if path.is_file() and path.read_bytes() != content:
            raise FileExistsError(f"Existing V96 report differs: {path}")
    created_metrics = False
    created_markdown = False
    try:
        created_metrics = _write_exact(metrics_output, metrics)
        created_markdown = _write_exact(markdown_output, markdown)
    except BaseException:
        for path, content, created in (
            (markdown_output, markdown, created_markdown),
            (metrics_output, metrics, created_metrics),
        ):
            if created and path.is_file() and path.read_bytes() == content:
                path.unlink()
                directory_descriptor = os.open(path.parent, os.O_RDONLY)
                try:
                    os.fsync(directory_descriptor)
                finally:
                    os.close(directory_descriptor)
        raise
    return {
        "phase": "v96_measured_integration_report_built",
        "passed": True,
        "metrics_output": str(metrics_output),
        "metrics_sha256": hashlib.sha256(metrics).hexdigest(),
        "markdown_output": str(markdown_output),
        "markdown_sha256": hashlib.sha256(markdown).hexdigest(),
        "live_numeric_mcp_refresh_included": payload.get("live_robot") is not None,
        "current_report_modified": False,
        "default_runtime_modified": False,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=("robot-evidence", "robot-evidence-check", "check", "build"),
    )
    parser.add_argument("--python", default=".venv-gemma4/bin/python")
    parser.add_argument("--robot-evidence", default=str(DEFAULT_ROBOT_EVIDENCE))
    parser.add_argument("--live-robot-evidence")
    parser.add_argument("--access-audit")
    parser.add_argument("--metrics-output", default=str(DEFAULT_METRICS_OUTPUT))
    parser.add_argument("--markdown-output", default=str(DEFAULT_MARKDOWN_OUTPUT))
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    python = Path(args.python)
    if not python.is_absolute():
        python = PROJECT_ROOT / python
    try:
        if args.command in {"robot-evidence", "robot-evidence-check"}:
            destination = Path(args.robot_evidence)
            if not destination.is_absolute():
                destination = PROJECT_ROOT / destination
            release = _validate_release_receipt(
                _run_json_command("release", python=python)
            )
            if args.command == "robot-evidence-check":
                payload = _validate_robot_evidence(
                    _read_json(destination, purpose="V96 robot evidence"),
                    release=release,
                )
                result = {
                    "phase": "v96_robot_preflight_evidence_authenticated",
                    "passed": True,
                    "output": str(destination),
                    "sha256": _sha256(destination),
                    "preflight_summary_sha256": payload[
                        "preflight_summary_sha256"
                    ],
                    "navigation_success_measured": False,
                }
            else:
                if args.access_audit is None:
                    raise ValueError("robot-evidence requires --access-audit")
                preflight = _strict_json_text(
                    sys.stdin.read(), purpose="V96 embodied preflight stdout"
                )
                audit_path = Path(args.access_audit)
                if not audit_path.is_absolute():
                    audit_path = PROJECT_ROOT / audit_path
                audit = _read_json(audit_path, purpose="V96 embodied access audit")
                payload = build_robot_preflight_evidence(
                    preflight,
                    audit,
                    access_audit_path=audit_path,
                    release_receipt=release,
                )
                write_robot_evidence(destination, payload)
                result = {
                    "phase": "v96_robot_preflight_evidence_built",
                    "passed": True,
                    "output": str(destination),
                    "sha256": _sha256(destination),
                    "navigation_success_measured": False,
                }
        else:
            robot_path = Path(args.robot_evidence)
            if not robot_path.is_absolute():
                robot_path = PROJECT_ROOT / robot_path
            live_robot_path = (
                None
                if args.live_robot_evidence is None
                else Path(args.live_robot_evidence)
            )
            if live_robot_path is not None and not live_robot_path.is_absolute():
                live_robot_path = PROJECT_ROOT / live_robot_path
            payload = collect_v96_measured_integration(
                python=python,
                robot_evidence_path=robot_path,
                live_robot_evidence_path=live_robot_path,
            )
            if args.command == "check":
                result = {
                    "phase": "v96_measured_integration_report_ready",
                    "passed": True,
                    "integration_identity_sha256": payload[
                        "integration_identity_sha256"
                    ],
                    "live_numeric_mcp_refresh_authenticated": payload.get(
                        "live_robot"
                    )
                    is not None,
                    "would_modify_current_report": False,
                    "would_modify_default_runtime": False,
                }
            else:
                metrics_output = Path(args.metrics_output)
                markdown_output = Path(args.markdown_output)
                if not metrics_output.is_absolute():
                    metrics_output = PROJECT_ROOT / metrics_output
                if not markdown_output.is_absolute():
                    markdown_output = PROJECT_ROOT / markdown_output
                result = build_report_outputs(
                    payload,
                    metrics_output=metrics_output,
                    markdown_output=markdown_output,
                )
    except (OSError, RuntimeError, TypeError, ValueError) as error:
        print(f"V96 final reporting refused: {error}", file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "ARTIFACT",
    "DEFAULT_LIVE_MARKDOWN_OUTPUT",
    "DEFAULT_LIVE_METRICS_OUTPUT",
    "DEFAULT_LIVE_ROBOT_EVIDENCE",
    "DEFAULT_MARKDOWN_OUTPUT",
    "DEFAULT_METRICS_OUTPUT",
    "DEFAULT_ROBOT_EVIDENCE",
    "ROBOT_ARTIFACT",
    "build_report_outputs",
    "build_robot_preflight_evidence",
    "collect_v96_measured_integration",
    "main",
    "render_markdown",
    "write_robot_evidence",
]
