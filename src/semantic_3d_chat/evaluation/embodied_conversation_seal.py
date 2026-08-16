"""Fail-closed sealer for one finite embodied-conversation JSONL proof.

The accepted transcript is deliberately narrow: startup at map version zero,
one explicit scan to version one, one successful +15 degree turn whose
automatic complete-image scan commits version two, and one final ``yes``
answer bound to the version-two active continuous prefix.  The sealer never
loads a model, map, QA file, oracle artifact, or simulator metadata.

Only hashes, numeric versions, opaque IDs, and protocol facts are copied into
the seal.  Raw transcript records are authenticated by SHA-256 but are not
embedded in the output summary.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import tempfile
from collections.abc import Mapping, Sequence
from itertools import pairwise
from pathlib import Path
from typing import Any, Final, NoReturn

from semantic_3d_chat.config import PROJECT_ROOT

SEAL_SCHEMA: Final[str] = "semantic_3d_chat.embodied_conversation_seal.v1"
TRANSCRIPT_RECORD_COUNT: Final[int] = 4
MAX_TRANSCRIPT_BYTES: Final[int] = 8 * 1024 * 1024
_SHA256: Final[re.Pattern[str]] = re.compile(r"[0-9a-f]{64}")
_SCENE_ID: Final[re.Pattern[str]] = re.compile(r"scene_[0-9]{6}")
_OBSERVATION_ID: Final[re.Pattern[str]] = re.compile(r"o_[0-9]{6}")
_FORBIDDEN_PATH_TOKENS: Final[frozenset[str]] = frozenset(
    {
        "oracle",
        "qa",
        "training",
        "validation",
        "validate",
        "test",
        "deferred",
        "final",
    }
)
_FORBIDDEN_ENVIRONMENT_FIELDS: Final[frozenset[str]] = frozenset(
    {
        "blender_object_names",
        "category_names",
        "environment_description",
        "generated_question_answer_metadata",
        "object_categories",
        "object_category",
        "object_inventory",
        "object_labels",
        "object_names",
        "oracle_relationships",
        "relationships",
        "scene_caption",
        "scene_description",
        "scene_graph",
        "segmentation_labels",
        "simulator_metadata",
        "target_instance",
        "textual_scene_graph",
    }
)

_BINDING_KEYS: Final[frozenset[str]] = frozenset(
    {
        "schema",
        "scene_id",
        "map_version",
        "map_sha256",
        "scene_prefix_sha256",
        "scene_control_signature_sha256",
        "source_voxels",
        "processed_voxels",
        "binding_sha256",
        "active_prefix_sha256",
        "robot_state_sha256",
        "robot_tokens_sha256",
        "robot_state_encoder_sha256",
        "active_binding_sha256",
    }
)
_STARTUP_KEYS: Final[frozenset[str]] = frozenset(
    {
        "phase",
        "scene_id",
        "runtime",
        "prefix_binding",
        "scene_prefix_computed_before_question",
        "environmental_text_inputs",
        "local_inference",
        "bounded_action_protocol",
        "strict_fixed_environment_embedding_input",
        "question_conditioned_scene_readout_tokens",
        "llm_tool_policy",
        "navigation_policy",
        "gemma_tool_decoder",
        "learned_navigation_closed_loop",
    }
)
_NAVIGATION_KEYS: Final[frozenset[str]] = frozenset(
    {
        "kind",
        "command",
        "success",
        "request_sha256",
        "target_count",
        "groundings",
        "navigation",
        "action_receipts",
        "prefix_binding",
        "environmental_text_inputs",
        "question_dependent_scene_retrieval",
    }
)
_ANSWER_KEYS: Final[frozenset[str]] = frozenset(
    {
        "kind",
        "request_sha256",
        "answer",
        "grounding_xyz_m",
        "grounding_confidence",
        "prefix_hash",
        "prefix_binding",
        "environmental_text_inputs",
    }
)
_ACTION_RECEIPT_KEYS: Final[frozenset[str]] = (
    frozenset(
        {
            "success",
            "error_code",
            "scene_id",
            "seed",
            "scene_version",
            "position_m",
            "camera_position_m",
            "body_yaw_degrees",
            "camera_yaw_degrees",
            "pitch_degrees",
            "linear_velocity_xy_m",
            "angular_velocity_degrees",
            "collision",
            "last_movement_delta_m",
            "distance_moved",
            "turn_degrees",
            "scan_coverage",
            "scan_count",
            "visible_voxels",
            "valid_depth_pixels",
            "observation_id",
            "clearance_m",
            "action_count",
            "stopped",
            "map_sha256",
        }
    )
    | _BINDING_KEYS
)


class EmbodiedConversationSealError(ValueError):
    """A stable fail-closed transcript-validation error."""

    def __init__(self, code: str, message: str) -> None:
        if not re.fullmatch(r"E_[A-Z0-9_]+", code):
            raise ValueError("Seal error code must be stable uppercase protocol text")
        super().__init__(message)
        self.code = code


def _fail(code: str, message: str) -> NoReturn:
    raise EmbodiedConversationSealError(code, message)


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _canonical_sha256(value: object) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return _sha256_bytes(encoded)


def _rooted(path: str | Path) -> Path:
    value = Path(path).expanduser()
    return Path(os.path.abspath(value if value.is_absolute() else PROJECT_ROOT / value))


def _relative(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def _path_tokens(path: Path) -> set[str]:
    try:
        scoped = path.relative_to(PROJECT_ROOT)
    except ValueError:
        scoped = Path(path.name)
    return {
        token
        for part in scoped.parts
        for token in re.split(r"[^a-z0-9]+", part.casefold())
        if token
    }


def _below_project(path: Path) -> None:
    try:
        path.relative_to(PROJECT_ROOT)
    except ValueError:
        _fail("E_PATH_ESCAPE", "Transcript seals must remain below the project root")


def _reject_symlink_ancestry(path: Path, *, include_leaf: bool) -> None:
    cursor = path if include_leaf else path.parent
    while True:
        if cursor.is_symlink():
            _fail("E_SYMLINK", "Transcript seals cannot traverse symbolic links")
        if cursor == PROJECT_ROOT:
            return
        if cursor.parent == cursor:
            _fail("E_PATH_ESCAPE", "Transcript seal path ancestry escaped the project root")
        cursor = cursor.parent


def _transcript_path(path: str | Path) -> Path:
    source = _rooted(path)
    _below_project(source)
    if _path_tokens(source) & _FORBIDDEN_PATH_TOKENS:
        _fail("E_FORBIDDEN_PATH", "Transcript path crosses a forbidden data boundary")
    _reject_symlink_ancestry(source, include_leaf=True)
    if not source.is_file() or source.is_symlink():
        _fail("E_TRANSCRIPT_MISSING", "Transcript must be an existing regular JSONL file")
    if source.suffix != ".jsonl":
        _fail("E_TRANSCRIPT_SUFFIX", "Transcript must use the .jsonl suffix")
    return source


def _output_path(path: str | Path) -> Path:
    destination = _rooted(path)
    _below_project(destination)
    if _path_tokens(destination) & _FORBIDDEN_PATH_TOKENS:
        _fail("E_FORBIDDEN_PATH", "Seal output crosses a forbidden data boundary")
    _reject_symlink_ancestry(destination, include_leaf=False)
    if destination.suffix != ".json":
        _fail("E_OUTPUT_SUFFIX", "Seal output must use the .json suffix")
    if destination.exists() or destination.is_symlink():
        _fail("E_OUTPUT_EXISTS", "Seal output already exists")
    return destination


def _no_duplicate_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            _fail("E_DUPLICATE_JSON_KEY", "Transcript JSON contains a duplicate object key")
        result[key] = value
    return result


def _reject_constant(value: str) -> NoReturn:
    _fail("E_NONFINITE_JSON", f"Transcript JSON contains forbidden constant {value}")


def _load_records(path: Path) -> tuple[bytes, tuple[dict[str, Any], ...]]:
    raw = path.read_bytes()
    if not raw or len(raw) > MAX_TRANSCRIPT_BYTES:
        _fail("E_TRANSCRIPT_SIZE", "Transcript is empty or exceeds the fixed size bound")
    try:
        decoded = raw.decode("utf-8")
    except UnicodeDecodeError:
        _fail("E_TRANSCRIPT_UTF8", "Transcript must be strict UTF-8")
    lines = decoded.splitlines()
    if len(lines) != TRANSCRIPT_RECORD_COUNT or any(not line.strip() for line in lines):
        _fail("E_RECORD_COUNT", "Transcript must contain exactly four nonempty JSONL records")
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(lines, 1):
        try:
            value = json.loads(
                line,
                object_pairs_hook=_no_duplicate_object,
                parse_constant=_reject_constant,
            )
        except EmbodiedConversationSealError:
            raise
        except json.JSONDecodeError:
            _fail("E_INVALID_JSON", f"Transcript record {line_number} is invalid JSON")
        if not isinstance(value, dict):
            _fail("E_RECORD_TYPE", f"Transcript record {line_number} must be a JSON object")
        records.append(value)
    return raw, tuple(records)


def _exact_keys(value: Mapping[str, Any], expected: frozenset[str], label: str) -> None:
    observed = set(value)
    if observed != set(expected):
        _fail(
            "E_SCHEMA_FIELDS",
            f"{label} fields changed: missing={sorted(expected - observed)} "
            f"unexpected={sorted(observed - expected)}",
        )


def _hash(value: object, label: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        _fail("E_HASH", f"{label} must be a lowercase SHA-256 digest")
    return value


def _integer(value: object, label: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        _fail("E_INTEGER", f"{label} must be an integer >= {minimum}")
    return value


def _finite_number(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        _fail("E_NUMBER", f"{label} must be numeric")
    numeric = float(value)
    if not math.isfinite(numeric):
        _fail("E_NUMBER", f"{label} must be finite")
    return numeric


def _scene_id(value: object, label: str) -> str:
    if not isinstance(value, str) or _SCENE_ID.fullmatch(value) is None:
        _fail("E_SCENE_ID", f"{label} must be an opaque scene ID")
    return value


def _binding(value: object, label: str, *, version: int, scene_id: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        _fail("E_BINDING", f"{label} must be a mapping")
    result = dict(value)
    _exact_keys(result, _BINDING_KEYS, label)
    if result["schema"] != "semantic_3d_chat.scene_prefix_binding.v2":
        _fail("E_BINDING_SCHEMA", f"{label} binding schema changed")
    if _scene_id(result["scene_id"], f"{label}.scene_id") != scene_id:
        _fail("E_SCENE_MISMATCH", f"{label} scene differs from startup")
    if _integer(result["map_version"], f"{label}.map_version") != version:
        _fail("E_VERSION", f"{label} must be map version {version}")
    for field in (
        "map_sha256",
        "scene_prefix_sha256",
        "scene_control_signature_sha256",
        "binding_sha256",
        "active_prefix_sha256",
        "robot_state_sha256",
        "robot_tokens_sha256",
        "robot_state_encoder_sha256",
        "active_binding_sha256",
    ):
        _hash(result[field], f"{label}.{field}")
    _integer(result["source_voxels"], f"{label}.source_voxels", minimum=1)
    _integer(result["processed_voxels"], f"{label}.processed_voxels", minimum=1)
    scene_identity = {
        field: result[field]
        for field in (
            "schema",
            "scene_id",
            "map_version",
            "map_sha256",
            "scene_prefix_sha256",
            "scene_control_signature_sha256",
            "source_voxels",
            "processed_voxels",
        )
    }
    if result["binding_sha256"] != _canonical_sha256(scene_identity):
        _fail("E_BINDING_HASH", f"{label} scene binding hash does not recompute")
    active_identity = {
        **scene_identity,
        "binding_sha256": result["binding_sha256"],
        "active_prefix_sha256": result["active_prefix_sha256"],
        "robot_state_sha256": result["robot_state_sha256"],
        "robot_tokens_sha256": result["robot_tokens_sha256"],
        "robot_state_encoder_sha256": result["robot_state_encoder_sha256"],
    }
    if result["active_binding_sha256"] != _canonical_sha256(active_identity):
        _fail("E_ACTIVE_BINDING_HASH", f"{label} active binding hash does not recompute")
    return result


def _environmental_text_audit(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    environmental_attestations = 0
    oracle_attestations = 0
    answer_text_attestations = 0
    codebook_attestations = 0
    visited_fields = 0

    def walk(value: object, path: str) -> None:
        nonlocal environmental_attestations
        nonlocal oracle_attestations
        nonlocal answer_text_attestations
        nonlocal codebook_attestations
        nonlocal visited_fields
        if isinstance(value, Mapping):
            for raw_key, child in value.items():
                if not isinstance(raw_key, str):
                    _fail("E_NONSTRING_KEY", "Transcript JSON object keys must be strings")
                visited_fields += 1
                key = raw_key.casefold()
                child_path = f"{path}.{raw_key}"
                if key in _FORBIDDEN_ENVIRONMENT_FIELDS:
                    _fail("E_ENVIRONMENTAL_TEXT", f"Forbidden environmental field: {child_path}")
                if key == "environmental_text_inputs":
                    environmental_attestations += 1
                    if child != []:
                        _fail(
                            "E_ENVIRONMENTAL_TEXT",
                            f"Environmental text attestation is nonempty: {child_path}",
                        )
                elif key == "oracle_inputs_at_runtime":
                    oracle_attestations += 1
                    if child is not False:
                        _fail("E_ORACLE_INPUT", f"Oracle input attestation failed: {child_path}")
                elif key == "answer_text_runtime_loaded":
                    answer_text_attestations += 1
                    if child is not False:
                        _fail(
                            "E_ANSWER_TEXT", f"Runtime answer-text attestation failed: {child_path}"
                        )
                elif key == "answer_class_codebook_runtime_loaded":
                    codebook_attestations += 1
                    if child is not False:
                        _fail(
                            "E_ANSWER_CODEBOOK",
                            f"Runtime codebook attestation failed: {child_path}",
                        )
                elif "oracle" in key:
                    _fail("E_ORACLE_FIELD", f"Unexpected oracle-bearing field: {child_path}")
                walk(child, child_path)
        elif isinstance(value, list):
            for index, child in enumerate(value):
                walk(child, f"{path}[{index}]")
        elif isinstance(value, float) and not math.isfinite(value):
            _fail("E_NONFINITE_JSON", f"Nonfinite numeric value at {path}")

    for index, record in enumerate(records):
        walk(record, f"record[{index}]")
    if environmental_attestations < TRANSCRIPT_RECORD_COUNT:
        _fail("E_ENVIRONMENTAL_ATTESTATION", "Every transcript record must deny environmental text")
    return {
        "passed": True,
        "environmental_text_attestation_count": environmental_attestations,
        "all_environmental_text_inputs_empty": True,
        "oracle_false_attestation_count": oracle_attestations,
        "answer_text_not_loaded_attestation_count": answer_text_attestations,
        "answer_codebook_not_loaded_attestation_count": codebook_attestations,
        "forbidden_environment_fields_found": 0,
        "visited_json_fields": visited_fields,
    }


def _action_record(
    value: Mapping[str, Any],
    *,
    label: str,
    command: str,
    version: int,
    scan_count: int,
    scene_id: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    _exact_keys(value, _NAVIGATION_KEYS, label)
    if (
        value["kind"] != "navigation"
        or value["command"] != command
        or value["success"] is not True
        or value["target_count"] != 0
        or value["groundings"] != []
        or value["navigation"] is not None
        or value["environmental_text_inputs"] != []
        or value["question_dependent_scene_retrieval"] is not False
    ):
        _fail("E_ACTION_PROTOCOL", f"{label} is not the required successful {command} action")
    _hash(value["request_sha256"], f"{label}.request_sha256")
    receipts = value["action_receipts"]
    if not isinstance(receipts, list) or len(receipts) != 1 or not isinstance(receipts[0], Mapping):
        _fail("E_ACTION_RECEIPT", f"{label} must contain exactly one action receipt")
    receipt = dict(receipts[0])
    _exact_keys(receipt, _ACTION_RECEIPT_KEYS, f"{label}.receipt")
    if (
        receipt["success"] is not True
        or receipt["error_code"] is not None
        or receipt["collision"] is not False
        or receipt["stopped"] is not False
    ):
        _fail("E_ACTION_FAILURE", f"{label} action receipt is not a clean success")
    if _scene_id(receipt["scene_id"], f"{label}.receipt.scene_id") != scene_id:
        _fail("E_SCENE_MISMATCH", f"{label} receipt scene differs from startup")
    if _integer(receipt["scene_version"], f"{label}.receipt.scene_version") != version:
        _fail("E_VERSION", f"{label} receipt must be scene version {version}")
    if _integer(receipt["scan_count"], f"{label}.receipt.scan_count") != scan_count:
        _fail("E_SCAN_COUNT", f"{label} receipt must have scan count {scan_count}")
    if _integer(receipt["valid_depth_pixels"], f"{label}.valid_depth_pixels", minimum=1) < 1:
        _fail("E_EMPTY_SCAN", f"{label} must contain a successful complete-image scan")
    observation = receipt["observation_id"]
    if not isinstance(observation, str) or _OBSERVATION_ID.fullmatch(observation) is None:
        _fail("E_OBSERVATION", f"{label} observation ID is not opaque")
    for field in (
        "body_yaw_degrees",
        "camera_yaw_degrees",
        "pitch_degrees",
        "angular_velocity_degrees",
        "distance_moved",
        "turn_degrees",
        "scan_coverage",
    ):
        _finite_number(receipt[field], f"{label}.receipt.{field}")
    binding = _binding(
        value["prefix_binding"], f"{label}.prefix_binding", version=version, scene_id=scene_id
    )
    for field in _BINDING_KEYS:
        if receipt[field] != binding[field]:
            _fail("E_RECEIPT_BINDING", f"{label} receipt and active binding disagree on {field}")
    if receipt["map_sha256"] != binding["map_sha256"]:
        _fail("E_RECEIPT_MAP", f"{label} receipt and binding map hashes differ")
    return receipt, binding


def _changed(first: Mapping[str, Any], second: Mapping[str, Any], label: str) -> None:
    required = (
        "map_sha256",
        "scene_prefix_sha256",
        "scene_control_signature_sha256",
        "binding_sha256",
        "active_prefix_sha256",
        "active_binding_sha256",
    )
    unchanged = [field for field in required if first[field] == second[field]]
    if unchanged:
        _fail("E_REFRESH_UNCHANGED", f"{label} did not change required hashes: {unchanged}")
    if first["robot_state_encoder_sha256"] != second["robot_state_encoder_sha256"]:
        _fail("E_STATE_ENCODER", f"{label} changed the robot-state encoder identity")


def seal_embodied_conversation_transcript(path: str | Path) -> dict[str, Any]:
    """Validate and summarize one exact four-record live transcript."""

    source = _transcript_path(path)
    raw, records = _load_records(source)
    startup, scan, turn, answer = records
    _exact_keys(startup, _STARTUP_KEYS, "startup")
    if (
        startup["phase"] != "embodied_conversation_ready"
        or startup["scene_prefix_computed_before_question"] is not True
        or startup["environmental_text_inputs"] != []
        or startup["local_inference"] is not True
        or startup["bounded_action_protocol"] is not True
        or startup["strict_fixed_environment_embedding_input"] is not False
        or startup["question_conditioned_scene_readout_tokens"] is not True
    ):
        _fail("E_STARTUP_PROTOCOL", "Startup does not attest the controlled local embodied runtime")
    scene_id = _scene_id(startup["scene_id"], "startup.scene_id")
    startup_binding = _binding(
        startup["prefix_binding"], "startup.prefix_binding", version=0, scene_id=scene_id
    )
    runtime = startup["runtime"]
    if not isinstance(runtime, Mapping):
        _fail("E_STARTUP_RUNTIME", "Startup runtime summary must be a mapping")
    if (
        runtime.get("phase") != "scene_ready"
        or runtime.get("scene_id") != scene_id
        or runtime.get("runtime_kind") != "continuous_scene_question_control"
        or runtime.get("questions_answered") != 0
        or runtime.get("scene_prefix_computed_before_question") is not True
        or runtime.get("prequestion_scene_key_value_cache") is not True
        or runtime.get("environmental_text_inputs") != []
        or runtime.get("answer_text_runtime_loaded") is not False
        or runtime.get("answer_class_codebook_runtime_loaded") is not False
        or runtime.get("scene_prefix_hash") != startup_binding["scene_prefix_sha256"]
        or runtime.get("scene_control_signature_sha256")
        != startup_binding["scene_control_signature_sha256"]
    ):
        _fail("E_STARTUP_RUNTIME", "Startup runtime/cache facts disagree with the v0 binding")

    scan_receipt, scan_binding = _action_record(
        scan,
        label="scan",
        command="scan",
        version=1,
        scan_count=1,
        scene_id=scene_id,
    )
    if abs(_finite_number(scan_receipt["turn_degrees"], "scan.turn_degrees")) > 1e-9:
        _fail("E_SCAN_MOTION", "Explicit scan receipt unexpectedly includes a turn")
    turn_receipt, turn_binding = _action_record(
        turn,
        label="turn",
        command="turn",
        version=2,
        scan_count=2,
        scene_id=scene_id,
    )
    if abs(_finite_number(turn_receipt["turn_degrees"], "turn.turn_degrees") - 15.0) > 1e-9:
        _fail("E_TURN_ANGLE", "Turn receipt must record exactly +15 degrees")
    if turn_receipt["observation_id"] == scan_receipt["observation_id"]:
        _fail("E_AUTO_SCAN", "Turn must auto-scan a new complete RGB-D observation")

    _changed(startup_binding, scan_binding, "startup-to-scan refresh")
    _changed(scan_binding, turn_binding, "scan-to-turn auto-refresh")

    _exact_keys(answer, _ANSWER_KEYS, "answer")
    if (
        answer["kind"] != "answer"
        or not isinstance(answer["answer"], str)
        or " ".join(answer["answer"].casefold().split()) != "yes"
        or answer["environmental_text_inputs"] != []
    ):
        _fail("E_ANSWER", "Final transcript record must be the answer 'yes'")
    _hash(answer["request_sha256"], "answer.request_sha256")
    answer_binding = _binding(
        answer["prefix_binding"], "answer.prefix_binding", version=2, scene_id=scene_id
    )
    if answer_binding != turn_binding:
        _fail("E_ANSWER_BINDING", "Answer changed or replaced the active v2 binding")
    answer_prefix = _hash(answer["prefix_hash"], "answer.prefix_hash")
    if answer_prefix != turn_binding["active_prefix_sha256"]:
        _fail("E_ANSWER_PREFIX", "Answer did not use the version-two active prefix")

    environmental_audit = _environmental_text_audit(records)
    versions = [
        startup_binding["map_version"],
        scan_binding["map_version"],
        turn_binding["map_version"],
        answer_binding["map_version"],
    ]
    if versions != [0, 1, 2, 2] or any(
        current < previous for previous, current in pairwise(versions)
    ):
        _fail("E_VERSION_MONOTONICITY", "Transcript versions are not the exact 0,1,2,2 sequence")

    return {
        "schema": SEAL_SCHEMA,
        "passed": True,
        "source_transcript": {
            "path": _relative(source),
            "sha256": _sha256_bytes(raw),
            "size_bytes": len(raw),
            "record_count": len(records),
            "strict_utf8_jsonl": True,
            "duplicate_json_keys_rejected": True,
            "nonfinite_json_rejected": True,
        },
        "facts": {
            "scene_id": scene_id,
            "map_versions": versions,
            "versions_monotonic": True,
            "startup": {
                "map_version": 0,
                "map_sha256": startup_binding["map_sha256"],
                "scene_prefix_sha256": startup_binding["scene_prefix_sha256"],
                "scene_control_signature_sha256": startup_binding["scene_control_signature_sha256"],
                "active_prefix_sha256": startup_binding["active_prefix_sha256"],
                "prequestion_scene_key_value_cache": True,
                "questions_answered": 0,
            },
            "scan": {
                "success": True,
                "map_version": 1,
                "scene_version": 1,
                "scan_count": 1,
                "observation_id": scan_receipt["observation_id"],
                "valid_depth_pixels": scan_receipt["valid_depth_pixels"],
                "map_sha256": scan_binding["map_sha256"],
                "scene_prefix_sha256": scan_binding["scene_prefix_sha256"],
                "scene_control_signature_sha256": scan_binding["scene_control_signature_sha256"],
                "active_prefix_sha256": scan_binding["active_prefix_sha256"],
                "changed_map_scene_control_and_active_prefix": True,
            },
            "turn": {
                "success": True,
                "turn_degrees": 15.0,
                "map_version": 2,
                "scene_version": 2,
                "scan_count": 2,
                "auto_scan": True,
                "observation_id": turn_receipt["observation_id"],
                "valid_depth_pixels": turn_receipt["valid_depth_pixels"],
                "map_sha256": turn_binding["map_sha256"],
                "scene_prefix_sha256": turn_binding["scene_prefix_sha256"],
                "scene_control_signature_sha256": turn_binding["scene_control_signature_sha256"],
                "active_prefix_sha256": turn_binding["active_prefix_sha256"],
                "changed_map_scene_control_and_active_prefix_again": True,
            },
            "answer": {
                "normalized_answer": "yes",
                "map_version": 2,
                "active_prefix_sha256": answer_prefix,
                "uses_version_two_active_prefix": True,
                "binding_unchanged_since_turn": True,
            },
        },
        "environmental_text_audit": environmental_audit,
        "checks": {
            "startup_v0": True,
            "scan_success_v1": True,
            "scan_changed_map_scene_and_control_cache": True,
            "turn_positive_15_degrees_success_v2": True,
            "turn_auto_scan_changed_all_hashes_again": True,
            "answer_is_yes": True,
            "answer_uses_v2_active_prefix": True,
            "no_environmental_text": True,
            "versions_monotonic": True,
        },
    }


def write_embodied_conversation_seal(path: str | Path, seal: Mapping[str, Any]) -> Path:
    """Atomically create one immutable successful seal; never overwrite."""

    if seal.get("schema") != SEAL_SCHEMA or seal.get("passed") is not True:
        _fail("E_UNPASSED_SEAL", "Only a successful validated seal may be written")
    destination = _output_path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(seal, indent=2, sort_keys=True, allow_nan=False).encode("utf-8") + b"\n"
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.partial-", dir=destination.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination


def failure_summary(path: str | Path, error: EmbodiedConversationSealError) -> dict[str, Any]:
    """Return a minimal non-seal diagnostic without copying transcript content."""

    source = _rooted(path)
    digest: str | None = None
    size: int | None = None
    try:
        source.relative_to(PROJECT_ROOT)
        safe_components = not bool(_path_tokens(source) & _FORBIDDEN_PATH_TOKENS)
        cursor = source
        symlinked = False
        while cursor != PROJECT_ROOT:
            symlinked = symlinked or cursor.is_symlink()
            if cursor.parent == cursor:
                symlinked = True
                break
            cursor = cursor.parent
        if safe_components and not symlinked and source.is_file():
            raw = source.read_bytes()
            digest = _sha256_bytes(raw)
            size = len(raw)
    except (OSError, ValueError):
        pass
    return {
        "schema": SEAL_SCHEMA,
        "passed": False,
        "sealed": False,
        "error_code": error.code,
        "source_transcript": {
            "path": _relative(source),
            "sha256": digest,
            "size_bytes": size,
        },
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--transcript", required=True, help="Finite four-record JSONL transcript")
    parser.add_argument("--output", required=True, help="New fail-closed JSON seal")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        seal = seal_embodied_conversation_transcript(args.transcript)
        destination = write_embodied_conversation_seal(args.output, seal)
    except EmbodiedConversationSealError as error:
        print(json.dumps(failure_summary(args.transcript, error), sort_keys=True), flush=True)
        return 2
    print(
        json.dumps(
            {
                "passed": True,
                "seal": _relative(destination),
                "seal_sha256": _sha256_bytes(destination.read_bytes()),
                "source_transcript_sha256": seal["source_transcript"]["sha256"],
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


__all__ = [
    "SEAL_SCHEMA",
    "TRANSCRIPT_RECORD_COUNT",
    "EmbodiedConversationSealError",
    "build_parser",
    "failure_summary",
    "main",
    "seal_embodied_conversation_transcript",
    "write_embodied_conversation_seal",
]


if __name__ == "__main__":
    raise SystemExit(main())
