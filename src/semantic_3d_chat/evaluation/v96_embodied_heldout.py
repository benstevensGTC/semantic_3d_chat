"""Fail-closed preregistration and scorer for V96 navigation-held-out scenes.

This downstream evaluator is intentionally inert on import.  It does not
materialize scenes, create a preregistration, open an oracle, run navigation,
or write a report unless an explicit command requests that operation.  Every
operation first requires the already-promoted V96 static release.  Runtime
evidence is fully validated and closed before the scorer opens oracle JSON.

Scenes 25--30 were absent from the accepted V3 navigation train split
(11--24) and validation split (31--37,39).  They are therefore navigation
held-out, although they are necessarily the static V96 release scenes and must
not be described as unseen by the static deferred-final evaluation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import subprocess
import sys
import tempfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

from semantic_3d_chat.config import PROJECT_ROOT, load_config
from semantic_3d_chat.robot.llm_tool_policy import validate_tool_call_text
from semantic_3d_chat.robot.mcp_stdio_runtime import validate_numeric_tool_receipt
from semantic_3d_chat.robot.v96_candidate_refresh import (
    run_isolated_v96_release_verification,
)
from semantic_3d_chat.robot.v96_co_resident_mcp_agent import TRANSPORT_MODE
from semantic_3d_chat.robot.v96_release_action import (
    TRANSFER_MODE,
    V3_POLICY_METADATA_SHA256,
    V3_POLICY_WEIGHTS_SHA256,
    V3_TRAINING_DATASET_SHA256,
)
from semantic_3d_chat.robot.v96_release_embodied import (
    DEFAULT_COMPILER_CHECKPOINT,
    DEFAULT_EMBODIED_CONFIG,
    DEFAULT_PROBE_BANK,
    DEFAULT_ROBOT_STATE_CHECKPOINT,
    RELEASE_SCENE_IDS,
    ROBOT_STATE_FILE_SHA256S,
    V75_CONTROL_FILE_SHA256S,
    V75_PROBE_FILE_SHA256S,
    validate_promoted_v96_release_receipt,
)
from semantic_3d_chat.robot.v96_runtime_source_contract import runtime_source_paths

SCHEMA_VERSION: Final[int] = 2
PREREGISTRATION_SCHEMA: Final[str] = (
    "semantic_3d_chat.v96_embodied_navigation_preregistration.v2"
)
RUNTIME_EVIDENCE_SCHEMA: Final[str] = (
    "semantic_3d_chat.v96_embodied_navigation_runtime_evidence.v2"
)
SCORE_SCHEMA: Final[str] = "semantic_3d_chat.v96_embodied_navigation_score.v2"
NAVIGATION_TRAIN_SCENES: Final[tuple[str, ...]] = tuple(
    f"scene_{index:06d}" for index in range(11, 25)
)
NAVIGATION_VALIDATION_SCENES: Final[tuple[str, ...]] = (
    *(f"scene_{index:06d}" for index in range(31, 38)),
    "scene_000039",
)
DEFAULT_PREREGISTRATION: Final[Path] = (
    PROJECT_ROOT
    / "reports"
    / "gemma4"
    / "metrics"
    / "v96_embodied_navigation_preregistration.json"
)
DEFAULT_RUNTIME_INPUT_ROOT: Final[Path] = (
    PROJECT_ROOT / "reports/gemma4/embodied/runtime_inputs/v96"
)
DEFAULT_RUNTIME_RESULT_ROOT: Final[Path] = (
    PROJECT_ROOT / "reports/gemma4/embodied/runtime_results/v96"
)
DEFAULT_RUNTIME_SCRATCH_ROOT: Final[Path] = (
    PROJECT_ROOT / "data_gemma4/runtime/v96_embodied/heldout_scratch"
)
DEFAULT_SCORE: Final[Path] = (
    PROJECT_ROOT / "reports/gemma4/metrics/v96_embodied_navigation_score.json"
)
DEFAULT_ORACLE_ROOT: Final[Path] = PROJECT_ROOT / "data/oracle"
DEFAULT_RUNTIME_ASSET_ROOT: Final[Path] = PROJECT_ROOT / "data/runtime_assets"
EVALUATION_RESET_SEED: Final[int] = 20260814
NAVIGATION_CHECKPOINT: Final[Path] = (
    PROJECT_ROOT / "data_gemma4/checkpoints/navigation_policy_v3"
)
NAVIGATION_TRAINING_MANIFEST: Final[Path] = (
    PROJECT_ROOT / "data_gemma4/training/navigation_policy_v3/manifest.json"
)
NAVIGATION_TRAINING_TRACES: Final[Path] = (
    PROJECT_ROOT / "data_gemma4/training/navigation_policy_v3/traces.jsonl"
)
NAVIGATION_EXPERIMENT_CONFIG: Final[Path] = (
    PROJECT_ROOT / "configs/experiments/navigation_policy_v3.yaml"
)
NAVIGATION_MANIFEST_SHA256: Final[str] = (
    "005756918c54fbffbb7c6db45e2170174d85f87f278e755e538418d6eb880243"
)
NAVIGATION_TRACES_SHA256: Final[str] = (
    "72434178ff1cf23c2dfeb98d52cb7b4c443fcc8715c1dd4ee883d87ae127e7ad"
)
NAVIGATION_EXPERIMENT_CONFIG_SHA256: Final[str] = (
    "9daf6bdbf2d059064a5b447c984b0cd394cb647c2994499110060dd90ebaea3a"
)
_SHA256_PATTERN: Final[str] = r"[0-9a-f]{64}"
RUNTIME_TASK_INPUT_MANIFEST_SCHEMA: Final[str] = (
    "semantic_3d_chat.v96_embodied_runtime_task_input_manifest.v1"
)
RUNTIME_RESULT_MANIFEST_SCHEMA: Final[str] = (
    "semantic_3d_chat.v96_embodied_runtime_result_root_manifest.v1"
)
SCENE_RESULT_MANIFEST_SCHEMA: Final[str] = (
    "semantic_3d_chat.v96_embodied_runtime_result_manifest.v1"
)
_RUNTIME_SOURCE_PATHS: Final[tuple[str, ...]] = runtime_source_paths()
_RUNTIME_CONFIG_PATHS: Final[tuple[str, ...]] = (
    "configs/runtime/embodied_live.yaml",
    "configs/runtime/embodied_v54.yaml",
    "configs/runtime/gemma4_v54.yaml",
    "configs/runtime/gemma4_v96_strict_multiscene.yaml",
)
_IMPLEMENTATION_SOURCE_PATHS: Final[tuple[str, ...]] = (
    *_RUNTIME_SOURCE_PATHS,
    "Makefile",
    "pyproject.toml",
    "requirements-gemma4-probe.txt",
    "scripts/run_v96_embodied_heldout.py",
    "src/semantic_3d_chat/evaluation/v96_embodied_heldout.py",
    "uv.lock",
)
_AUTO_SCAN_ACTIONS: Final[frozenset[str]] = frozenset(
    {"look", "turn", "move_forward", "move_backward", "move_to"}
)


@dataclass(frozen=True)
class HeldOutTask:
    key: str
    kind: str
    instruction: str

    def public_spec(self, scene_id: str) -> dict[str, Any]:
        return {
            "task_id": f"{scene_id}:{self.key}",
            "scene_id": scene_id,
            "key": self.key,
            "kind": self.kind,
            "instruction": self.instruction,
            "max_steps": 24,
        }


# Fixed before any held-out oracle is opened.  Category names exist only in
# this physically separate evaluator; the runtime sees the literal user-style
# instruction, which is an explicitly permitted text input.
TASK_TEMPLATES: Final[tuple[HeldOutTask, ...]] = (
    HeldOutTask("face_primary", "face", "Face the chair, then stop."),
    HeldOutTask("face_secondary", "face", "Face the lamp, then stop."),
    HeldOutTask("approach_primary", "approach", "Approach the table, then stop."),
    HeldOutTask(
        "scan_approach",
        "approach",
        "Scan the room, then approach the bowl and stop.",
    ),
    HeldOutTask("approach_small", "approach", "Approach the cube, then stop."),
    HeldOutTask("scan_only", "scan", "Scan the room."),
)
_SCORER_ORACLE_CATEGORY_BY_KEY: Final[dict[str, str | None]] = {
    "face_primary": "chair",
    "face_secondary": "floor lamp",
    "approach_primary": "table",
    "scan_approach": "bowl",
    "approach_small": "cube",
    "scan_only": None,
}

ACCEPTANCE_GATES: Final[dict[str, Any]] = {
    "required_scene_count": 6,
    "required_task_count": 36,
    "maximum_face_residual_degrees": 3.0,
    "minimum_approach_center_progress_m": 0.25,
    "maximum_approach_bbox_standoff_m": 0.75,
    "maximum_collision_count": 0,
    "maximum_rejected_action_count": 0,
    "minimum_successful_scan_count": 6,
    "require_map_and_prefix_refresh_after_scan": True,
    "require_all_738_scene_tokens_every_decision": True,
    "require_all_4_robot_tokens_every_decision": True,
    "require_numeric_only_mcp_receipts": True,
    "require_zero_forbidden_runtime_reads": True,
    "require_all_tasks_pass": True,
}


def _canonical_sha256(value: object) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _absolute_nonsymlink(path: str | Path, *, purpose: str) -> Path:
    candidate = Path(path).expanduser()
    rooted = candidate if candidate.is_absolute() else PROJECT_ROOT / candidate
    absolute = Path(os.path.abspath(rooted))
    current = Path(absolute.anchor)
    for component in absolute.parts[1:]:
        current /= component
        if current.is_symlink():
            raise ValueError(f"{purpose} path contains a symbolic link: {current}")
    return absolute


def _read_object(path: Path) -> dict[str, Any]:
    absolute = _absolute_nonsymlink(path, purpose="V96 embodied JSON")
    if absolute.is_symlink() or not absolute.is_file():
        raise FileNotFoundError(path)

    def unique(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"Duplicate JSON field in {path}: {key}")
            result[key] = value
        return result

    value = json.loads(
        absolute.read_text(encoding="utf-8"),
        object_pairs_hook=unique,
        parse_constant=lambda value: (_ for _ in ()).throw(
            ValueError(f"Non-finite JSON constant is forbidden: {value}")
        ),
    )
    if not isinstance(value, dict):
        raise TypeError(f"Expected one JSON object: {path}")
    return value


def _source_inventory(relative_paths: Sequence[str]) -> dict[str, Any]:
    if len(relative_paths) != len(set(relative_paths)):
        raise RuntimeError("V96 embodied source inventory contains duplicate paths")
    entries: list[dict[str, Any]] = []
    for relative in sorted(relative_paths):
        path = _absolute_nonsymlink(
            PROJECT_ROOT / relative,
            purpose="V96 embodied source inventory",
        )
        if not path.is_file():
            raise FileNotFoundError(path)
        entries.append(
            {
                "path": relative,
                "sha256": _file_sha256(path),
                "size_bytes": path.stat().st_size,
            }
        )
    return {
        "files": entries,
        "inventory_sha256": _canonical_sha256(entries),
    }


def _exact_two_file_contract(
    root: Path,
    expected: Mapping[str, str],
) -> dict[str, Any]:
    source = _absolute_nonsymlink(root, purpose="V96 embodied artifact")
    if not source.is_dir() or {item.name for item in source.iterdir()} != set(expected):
        raise ValueError(f"V96 embodied artifact inventory changed: {source}")
    files: list[dict[str, Any]] = []
    for name in sorted(expected):
        path = source / name
        observed = _file_sha256(path)
        if path.is_symlink() or not path.is_file() or observed != expected[name]:
            raise ValueError(f"V96 embodied artifact bytes changed: {path}")
        files.append({"name": name, "sha256": observed, "size_bytes": path.stat().st_size})
    return {"files": files, "inventory_sha256": _canonical_sha256(files)}


def _runtime_asset_contract(scene_id: str, asset_root: Path) -> dict[str, Any]:
    suffix = scene_id.removeprefix("scene_")
    root = _absolute_nonsymlink(
        asset_root / scene_id,
        purpose="V96 sanitized runtime asset",
    )
    asset = root / f"s_{suffix}.blend"
    manifest_path = root / f"s_{suffix}.json"
    if any(path.is_symlink() or not path.is_file() for path in (asset, manifest_path)):
        raise FileNotFoundError(f"V96 sanitized runtime asset is unavailable: {scene_id}")
    manifest = _read_object(manifest_path)
    expected_fields = {
        "schema",
        "scene_id",
        "asset_file",
        "asset_sha256",
        "object_names_opaque",
        "nested_names_opaque",
        "custom_properties_present",
        "external_assets_present",
        "automation_present",
        "animation_present",
        "unsupported_datablocks_present",
        "strict_nested_datablock_audit_passed",
        "mesh_objects",
        "light_objects",
        "materials",
        "collections",
        "node_trees",
    }
    asset_sha256 = _file_sha256(asset)
    if (
        set(manifest) != expected_fields
        or manifest.get("schema") != "semantic_3d_chat.runtime_scene.v2"
        or manifest.get("scene_id") != scene_id
        or manifest.get("asset_file") != asset.name
        or manifest.get("asset_sha256") != asset_sha256
        or manifest.get("object_names_opaque") is not True
        or manifest.get("nested_names_opaque") is not True
        or manifest.get("strict_nested_datablock_audit_passed") is not True
        or any(
            manifest.get(field) is not False
            for field in (
                "custom_properties_present",
                "external_assets_present",
                "automation_present",
                "animation_present",
                "unsupported_datablocks_present",
            )
        )
        or any(
            isinstance(manifest.get(field), bool)
            or not isinstance(manifest.get(field), int)
            or int(manifest[field]) < 0
            for field in (
                "mesh_objects",
                "light_objects",
                "materials",
                "collections",
                "node_trees",
            )
        )
        or int(manifest["mesh_objects"]) < 1
    ):
        raise ValueError(f"V96 runtime asset is not strictly sanitized: {scene_id}")
    value = {
        "scene_id": scene_id,
        "asset_path": asset.relative_to(PROJECT_ROOT).as_posix(),
        "asset_sha256": asset_sha256,
        "asset_size_bytes": asset.stat().st_size,
        "manifest_path": manifest_path.relative_to(PROJECT_ROOT).as_posix(),
        "manifest_file_sha256": _file_sha256(manifest_path),
        "manifest_contract_sha256": _canonical_sha256(manifest),
    }
    value["contract_sha256"] = _canonical_sha256(value)
    return value


def build_embodied_dependency_contract(
    *,
    asset_root: str | Path = DEFAULT_RUNTIME_ASSET_ROOT,
) -> dict[str, Any]:
    """Authenticate every mutable downstream input before held-out runtime."""

    if (
        _file_sha256(NAVIGATION_TRAINING_MANIFEST) != NAVIGATION_MANIFEST_SHA256
        or _file_sha256(NAVIGATION_TRAINING_TRACES) != NAVIGATION_TRACES_SHA256
        or _file_sha256(NAVIGATION_EXPERIMENT_CONFIG)
        != NAVIGATION_EXPERIMENT_CONFIG_SHA256
    ):
        raise ValueError("V96 navigation training provenance changed")
    manifest = _read_object(NAVIGATION_TRAINING_MANIFEST)
    if (
        manifest.get("schema")
        != "semantic_3d_chat.navigation_target_trace_dataset.v3"
        or manifest.get("dataset_sha256") != V3_TRAINING_DATASET_SHA256
        or manifest.get("traces_sha256") != NAVIGATION_TRACES_SHA256
        or manifest.get("train_scene_ids") != list(NAVIGATION_TRAIN_SCENES)
        or manifest.get("validation_scene_ids") != list(NAVIGATION_VALIDATION_SCENES)
        or manifest.get("train_scene_count") != len(NAVIGATION_TRAIN_SCENES)
        or manifest.get("validation_scene_count") != len(NAVIGATION_VALIDATION_SCENES)
        or manifest.get("scene_splits_disjoint") is not True
        or manifest.get("runtime_oracle_inputs") is not False
        or manifest.get("target_coordinates_training_tree_only") is not True
    ):
        raise ValueError("V96 navigation split cannot be authenticated as held out")
    navigation = _exact_two_file_contract(
        NAVIGATION_CHECKPOINT,
        {
            "policy.safetensors": V3_POLICY_WEIGHTS_SHA256,
            "runtime_metadata.json": V3_POLICY_METADATA_SHA256,
        },
    )
    navigation.update(
        {
            "training_dataset_sha256": V3_TRAINING_DATASET_SHA256,
            "training_manifest_file_sha256": NAVIGATION_MANIFEST_SHA256,
            "training_traces_file_sha256": NAVIGATION_TRACES_SHA256,
            "training_experiment_config_sha256": NAVIGATION_EXPERIMENT_CONFIG_SHA256,
            "train_scene_ids": list(NAVIGATION_TRAIN_SCENES),
            "validation_scene_ids": list(NAVIGATION_VALIDATION_SCENES),
        }
    )
    config_inventory = _source_inventory(_RUNTIME_CONFIG_PATHS)
    runtime_sources = _source_inventory(_RUNTIME_SOURCE_PATHS)
    implementation = _source_inventory(_IMPLEMENTATION_SOURCE_PATHS)
    root = Path(os.path.abspath(Path(asset_root).expanduser()))
    assets = {
        scene_id: _runtime_asset_contract(scene_id, root)
        for scene_id in RELEASE_SCENE_IDS
    }
    value: dict[str, Any] = {
        "schema": "semantic_3d_chat.v96_embodied_dependency_contract.v1",
        "navigation_policy": navigation,
        "robot_state_checkpoint": _exact_two_file_contract(
            DEFAULT_ROBOT_STATE_CHECKPOINT, ROBOT_STATE_FILE_SHA256S
        ),
        "question_free_compiler_checkpoint": _exact_two_file_contract(
            DEFAULT_COMPILER_CHECKPOINT, V75_CONTROL_FILE_SHA256S
        ),
        "numeric_probe_bank": _exact_two_file_contract(
            DEFAULT_PROBE_BANK, V75_PROBE_FILE_SHA256S
        ),
        "runtime_config_inventory": config_inventory,
        "runtime_source_inventory": runtime_sources,
        "implementation_source_inventory": implementation,
        "runtime_assets": assets,
        "runtime_asset_inventory_sha256": _canonical_sha256(assets),
        "policy_transfer": TRANSFER_MODE,
        "source_policy_retrained_on_v96": False,
    }
    value["contract_sha256"] = _canonical_sha256(value)
    return value


def _validate_inventory(
    value: object,
    *,
    expected_paths: Sequence[str] | None = None,
    expected_name_sha256s: Mapping[str, str] | None = None,
    label: str,
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{label} inventory is malformed")
    result = dict(value)
    rows = result.get("files")
    if (
        set(result) != {"files", "inventory_sha256"}
        or not isinstance(rows, list)
        or result.get("inventory_sha256") != _canonical_sha256(rows)
    ):
        raise ValueError(f"{label} inventory binding changed")
    if expected_paths is not None:
        expected = sorted(expected_paths)
        if len(rows) != len(expected):
            raise ValueError(f"{label} file count changed")
        for row, path in zip(rows, expected, strict=True):
            if (
                not isinstance(row, Mapping)
                or set(row) != {"path", "sha256", "size_bytes"}
                or row.get("path") != path
                or re.fullmatch(_SHA256_PATTERN, str(row.get("sha256"))) is None
                or isinstance(row.get("size_bytes"), bool)
                or not isinstance(row.get("size_bytes"), int)
                or int(row["size_bytes"]) < 1
            ):
                raise ValueError(f"{label} source row changed")
    elif expected_name_sha256s is not None:
        names = sorted(expected_name_sha256s)
        if len(rows) != len(names):
            raise ValueError(f"{label} file count changed")
        for row, name in zip(rows, names, strict=True):
            if (
                not isinstance(row, Mapping)
                or set(row) != {"name", "sha256", "size_bytes"}
                or row.get("name") != name
                or row.get("sha256") != expected_name_sha256s[name]
                or isinstance(row.get("size_bytes"), bool)
                or not isinstance(row.get("size_bytes"), int)
                or int(row["size_bytes"]) < 1
            ):
                raise ValueError(f"{label} artifact row changed")
    else:  # pragma: no cover - private helper is always constrained
        raise RuntimeError("V96 inventory validator requires an exact inventory")
    return result


def validate_embodied_dependency_contract(value: Mapping[str, Any]) -> dict[str, Any]:
    contract = dict(value)
    digest = contract.pop("contract_sha256", None)
    required = {
        "schema",
        "navigation_policy",
        "robot_state_checkpoint",
        "question_free_compiler_checkpoint",
        "numeric_probe_bank",
        "runtime_config_inventory",
        "runtime_source_inventory",
        "implementation_source_inventory",
        "runtime_assets",
        "runtime_asset_inventory_sha256",
        "policy_transfer",
        "source_policy_retrained_on_v96",
    }
    navigation = contract.get("navigation_policy")
    assets = contract.get("runtime_assets")
    navigation_inventory = None
    if isinstance(navigation, Mapping):
        navigation_inventory = {
            "files": navigation.get("files"),
            "inventory_sha256": navigation.get("inventory_sha256"),
        }
    if (
        set(contract) != required
        or contract.get("schema")
        != "semantic_3d_chat.v96_embodied_dependency_contract.v1"
        or not isinstance(digest, str)
        or re.fullmatch(_SHA256_PATTERN, digest) is None
        or _canonical_sha256(contract) != digest
        or not isinstance(navigation, Mapping)
        or set(navigation)
        != {
            "files",
            "inventory_sha256",
            "training_dataset_sha256",
            "training_manifest_file_sha256",
            "training_traces_file_sha256",
            "training_experiment_config_sha256",
            "train_scene_ids",
            "validation_scene_ids",
        }
        or navigation.get("training_dataset_sha256") != V3_TRAINING_DATASET_SHA256
        or navigation.get("training_manifest_file_sha256")
        != NAVIGATION_MANIFEST_SHA256
        or navigation.get("training_traces_file_sha256")
        != NAVIGATION_TRACES_SHA256
        or navigation.get("training_experiment_config_sha256")
        != NAVIGATION_EXPERIMENT_CONFIG_SHA256
        or navigation.get("train_scene_ids") != list(NAVIGATION_TRAIN_SCENES)
        or navigation.get("validation_scene_ids") != list(NAVIGATION_VALIDATION_SCENES)
        or contract.get("policy_transfer") != TRANSFER_MODE
        or contract.get("source_policy_retrained_on_v96") is not False
        or not isinstance(assets, Mapping)
        or set(assets) != set(RELEASE_SCENE_IDS)
        or contract.get("runtime_asset_inventory_sha256") != _canonical_sha256(assets)
    ):
        raise ValueError("V96 embodied dependency contract changed")
    _validate_inventory(
        navigation_inventory,
        expected_name_sha256s={
            "policy.safetensors": V3_POLICY_WEIGHTS_SHA256,
            "runtime_metadata.json": V3_POLICY_METADATA_SHA256,
        },
        label="V96 navigation policy",
    )
    _validate_inventory(
        contract["robot_state_checkpoint"],
        expected_name_sha256s=ROBOT_STATE_FILE_SHA256S,
        label="V96 robot-state checkpoint",
    )
    _validate_inventory(
        contract["question_free_compiler_checkpoint"],
        expected_name_sha256s=V75_CONTROL_FILE_SHA256S,
        label="V96 question-free compiler checkpoint",
    )
    _validate_inventory(
        contract["numeric_probe_bank"],
        expected_name_sha256s=V75_PROBE_FILE_SHA256S,
        label="V96 numeric probe bank",
    )
    _validate_inventory(
        contract["runtime_config_inventory"],
        expected_paths=_RUNTIME_CONFIG_PATHS,
        label="V96 runtime config",
    )
    _validate_inventory(
        contract["runtime_source_inventory"],
        expected_paths=_RUNTIME_SOURCE_PATHS,
        label="V96 runtime source",
    )
    _validate_inventory(
        contract["implementation_source_inventory"],
        expected_paths=_IMPLEMENTATION_SOURCE_PATHS,
        label="V96 held-out implementation",
    )
    for scene_id in RELEASE_SCENE_IDS:
        row = assets[scene_id]
        suffix = scene_id.removeprefix("scene_")
        if not isinstance(row, Mapping):
            raise TypeError("V96 runtime asset contract is malformed")
        asset_row = dict(row)
        asset_digest = asset_row.pop("contract_sha256", None)
        if (
            set(asset_row)
            != {
                "scene_id",
                "asset_path",
                "asset_sha256",
                "asset_size_bytes",
                "manifest_path",
                "manifest_file_sha256",
                "manifest_contract_sha256",
            }
            or asset_row.get("scene_id") != scene_id
            or asset_row.get("asset_path")
            != f"data/runtime_assets/{scene_id}/s_{suffix}.blend"
            or asset_row.get("manifest_path")
            != f"data/runtime_assets/{scene_id}/s_{suffix}.json"
            or any(
                re.fullmatch(_SHA256_PATTERN, str(asset_row.get(field))) is None
                for field in (
                    "asset_sha256",
                    "manifest_file_sha256",
                    "manifest_contract_sha256",
                )
            )
            or isinstance(asset_row.get("asset_size_bytes"), bool)
            or not isinstance(asset_row.get("asset_size_bytes"), int)
            or int(asset_row["asset_size_bytes"]) < 1
            or re.fullmatch(_SHA256_PATTERN, str(asset_digest)) is None
            or _canonical_sha256(asset_row) != asset_digest
        ):
            raise ValueError("V96 runtime asset binding changed")
    return {**contract, "contract_sha256": digest}


def _task_rows() -> list[dict[str, Any]]:
    return [
        task.public_spec(scene_id)
        for scene_id in RELEASE_SCENE_IDS
        for task in TASK_TEMPLATES
    ]


def runtime_task_input_payload(
    scene_id: str,
    tasks: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    selected = [
        {
            "scene_id": row["scene_id"],
            "task_id": row["task_id"],
            "instruction": row["instruction"],
            "max_steps": row["max_steps"],
        }
        for row in tasks
        if row.get("scene_id") == scene_id
    ]
    if len(selected) != len(TASK_TEMPLATES) or any(
        set(row) != {"scene_id", "task_id", "instruction", "max_steps"}
        or row["scene_id"] != scene_id
        for row in selected
    ):
        raise ValueError("V96 runtime task input is not the fixed sanitized scene inventory")
    return {
        "schema": "semantic_3d_chat.v96_embodied_runtime_task_input.v1",
        "scene_id": scene_id,
        "tasks": selected,
    }


def build_preregistration_payload(
    release_receipt: Mapping[str, Any],
) -> dict[str, Any]:
    receipt = validate_promoted_v96_release_receipt(release_receipt)
    dependencies = validate_embodied_dependency_contract(
        build_embodied_dependency_contract()
    )
    if (
        set(RELEASE_SCENE_IDS) & set(NAVIGATION_TRAIN_SCENES)
        or set(RELEASE_SCENE_IDS) & set(NAVIGATION_VALIDATION_SCENES)
    ):
        raise RuntimeError("V96 navigation-held-out split overlaps policy development")
    tasks = _task_rows()
    if len(tasks) != ACCEPTANCE_GATES["required_task_count"]:
        raise RuntimeError("V96 embodied held-out task inventory changed")
    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "schema": PREREGISTRATION_SCHEMA,
        "status": "fixed_before_navigation_heldout_runtime_or_oracle_scoring",
        "scene_ids": list(RELEASE_SCENE_IDS),
        "navigation_train_scene_ids": list(NAVIGATION_TRAIN_SCENES),
        "navigation_validation_scene_ids": list(NAVIGATION_VALIDATION_SCENES),
        "navigation_held_out": True,
        "static_deferred_final_scenes_reused": True,
        "static_unseen_claim": False,
        "tasks": tasks,
        "runtime_task_input_sha256_by_scene": {
            scene_id: _canonical_sha256(runtime_task_input_payload(scene_id, tasks))
            for scene_id in RELEASE_SCENE_IDS
        },
        "release_receipt_sha256": _canonical_sha256(receipt),
        "dependency_contract": dependencies,
        "dependency_contract_sha256": dependencies["contract_sha256"],
        "scorer_only_target_mapping_sha256": _canonical_sha256(
            _SCORER_ORACLE_CATEGORY_BY_KEY
        ),
        "acceptance_gates": dict(ACCEPTANCE_GATES),
        "runtime_contract": {
            "static_release_must_be_promoted_first": True,
            "release_checkpoint_sha256": receipt["release_checkpoint_sha256"],
            "release_adapter_sha256": receipt["release_adapter_sha256"],
            "v95_state_sha256": receipt["v95_state_sha256"],
            "v96_state_sha256": receipt["v96_state_sha256"],
            "scene_tokens": 738,
            "robot_tokens": 4,
            "active_tokens": 742,
            "hidden_size": 1536,
            "policy_transfer": (
                "frozen_v3_sequence_length_transfer_258_to_738_not_retrained_on_v96"
            ),
            "navigation_policy_weights_sha256": V3_POLICY_WEIGHTS_SHA256,
            "navigation_policy_metadata_sha256": V3_POLICY_METADATA_SHA256,
            "navigation_training_dataset_sha256": V3_TRAINING_DATASET_SHA256,
            "navigation_training_manifest_sha256": NAVIGATION_MANIFEST_SHA256,
            "navigation_training_traces_sha256": NAVIGATION_TRACES_SHA256,
            "navigation_checkpoint_caller_selectable": False,
            "transport": TRANSPORT_MODE,
            "official_stdio_transport_separately_tested": True,
            "auto_scan_after_motion_required": True,
            "numeric_tool_outputs_only": True,
            "environmental_text_inputs": [],
            "oracle_inputs_at_runtime": False,
            "runtime_task_input_fields": ["scene_id", "task_id", "instruction", "max_steps"],
            "scorer_only_category_loaded_by_runtime": False,
            "evaluation_reset_seed": EVALUATION_RESET_SEED,
        },
        "scoring_order": [
            "authenticate_promoted_static_release",
            "authenticate_preregistration",
            "close_and_validate_all_runtime_evidence",
            "only_then_open_oracle_json",
            "score_fixed_metrics",
        ],
    }
    payload["task_inventory_sha256"] = _canonical_sha256(tasks)
    payload["acceptance_gates_sha256"] = _canonical_sha256(ACCEPTANCE_GATES)
    return payload


def preflight(
    *,
    release_verifier: Callable[[], Mapping[str, Any]] = (
        run_isolated_v96_release_verification
    ),
) -> dict[str, Any]:
    """Model-free readiness check; creates no files and opens no oracle."""

    receipt = validate_promoted_v96_release_receipt(release_verifier())
    payload = build_preregistration_payload(receipt)
    return {
        "phase": "v96_embodied_navigation_preregistration_preflight",
        "passed": True,
        "promoted_static_release_verified": True,
        "scene_count": len(RELEASE_SCENE_IDS),
        "task_count": len(payload["tasks"]),
        "task_inventory_sha256": payload["task_inventory_sha256"],
        "dependency_contract_sha256": payload["dependency_contract_sha256"],
        "oracle_opened": False,
        "artifacts_written": False,
    }


def write_preregistration(
    destination: str | Path,
    *,
    release_verifier: Callable[[], Mapping[str, Any]] = (
        run_isolated_v96_release_verification
    ),
) -> dict[str, Any]:
    """Create the preregistration exactly once, only after static promotion."""

    receipt = validate_promoted_v96_release_receipt(release_verifier())
    payload = build_preregistration_payload(receipt)
    path = Path(os.path.abspath(Path(destination).expanduser()))
    if path.exists() or path.is_symlink():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    with path.open("x", encoding="utf-8") as handle:
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())
    return {
        "path": str(path),
        "sha256": _file_sha256(path),
        "payload": payload,
        "oracle_opened": False,
    }


def authenticate_preregistration(
    path: str | Path,
    *,
    release_receipt: Mapping[str, Any],
) -> dict[str, Any]:
    source = Path(os.path.abspath(Path(path).expanduser()))
    observed = _read_object(source)
    expected = build_preregistration_payload(release_receipt)
    if observed != expected:
        raise ValueError("V96 embodied preregistration differs from fixed source")
    return {
        "path": str(source),
        "sha256": _file_sha256(source),
        "payload": observed,
    }


def _write_json_create_once(path: Path, value: Mapping[str, Any]) -> None:
    if path.exists() or path.is_symlink():
        raise FileExistsError(path)
    encoded = json.dumps(dict(value), indent=2, sort_keys=True, allow_nan=False) + "\n"
    with path.open("x", encoding="utf-8") as handle:
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())


def _runtime_input_manifest(
    root: Path,
    preregistration_payload: Mapping[str, Any],
    *,
    preregistration_sha256: str,
) -> dict[str, Any]:
    manifest_path = root / "manifest.json"
    manifest = _read_object(manifest_path)
    rows = manifest.get("scenes")
    fields = {
        "schema",
        "preregistration_sha256",
        "task_inventory_sha256",
        "scenes",
        "inventory_sha256",
    }
    expected_rows: list[dict[str, Any]] = []
    if not isinstance(rows, list):
        raise TypeError("V96 runtime task-input manifest is malformed")
    for scene_id in RELEASE_SCENE_IDS:
        filename = f"t_{hashlib.sha256(scene_id.encode('utf-8')).hexdigest()[:16]}.json"
        path = root / filename
        payload = _read_object(path)
        expected_payload = runtime_task_input_payload(
            scene_id,
            preregistration_payload["tasks"],
        )
        if (
            payload != expected_payload
            or _canonical_sha256(payload)
            != preregistration_payload["runtime_task_input_sha256_by_scene"][scene_id]
        ):
            raise ValueError("V96 runtime task input differs from preregistration")
        expected_rows.append(
            {
                "scene_id": scene_id,
                "file": filename,
                "file_sha256": _file_sha256(path),
                "payload_sha256": _canonical_sha256(payload),
            }
        )
    if (
        set(manifest) != fields
        or manifest.get("schema") != RUNTIME_TASK_INPUT_MANIFEST_SCHEMA
        or manifest.get("preregistration_sha256") != preregistration_sha256
        or manifest.get("task_inventory_sha256")
        != preregistration_payload["task_inventory_sha256"]
        or rows != expected_rows
        or manifest.get("inventory_sha256") != _canonical_sha256(expected_rows)
        or {item.name for item in root.iterdir()}
        != {"manifest.json", *(row["file"] for row in expected_rows)}
    ):
        raise ValueError("V96 runtime task-input inventory changed")
    return manifest


def write_runtime_task_inputs(
    destination: str | Path,
    preregistration_payload: Mapping[str, Any],
    *,
    preregistration_sha256: str,
) -> dict[str, Any]:
    """Write only opaque scene IDs and literal user instructions, exactly once."""

    root = _absolute_nonsymlink(destination, purpose="V96 runtime task-input root")
    if root.exists() or root.is_symlink():
        raise FileExistsError(root)
    root.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{root.name}.", dir=root.parent) as raw:
        stage = Path(raw)
        rows: list[dict[str, Any]] = []
        for scene_id in RELEASE_SCENE_IDS:
            filename = f"t_{hashlib.sha256(scene_id.encode('utf-8')).hexdigest()[:16]}.json"
            payload = runtime_task_input_payload(
                scene_id,
                preregistration_payload["tasks"],
            )
            path = stage / filename
            _write_json_create_once(path, payload)
            rows.append(
                {
                    "scene_id": scene_id,
                    "file": filename,
                    "file_sha256": _file_sha256(path),
                    "payload_sha256": _canonical_sha256(payload),
                }
            )
        manifest = {
            "schema": RUNTIME_TASK_INPUT_MANIFEST_SCHEMA,
            "preregistration_sha256": preregistration_sha256,
            "task_inventory_sha256": preregistration_payload["task_inventory_sha256"],
            "scenes": rows,
            "inventory_sha256": _canonical_sha256(rows),
        }
        _write_json_create_once(stage / "manifest.json", manifest)
        os.replace(stage, root)
    _runtime_input_manifest(
        root,
        preregistration_payload,
        preregistration_sha256=preregistration_sha256,
    )
    return manifest


def _result_paths_from_root(
    result_root: str | Path,
    preregistration_payload: Mapping[str, Any],
    *,
    preregistration_sha256: str,
) -> dict[str, Path]:
    root = _absolute_nonsymlink(result_root, purpose="V96 runtime result root")
    if not root.is_dir():
        raise FileNotFoundError(root)
    top = _read_object(root / "manifest.json")
    top_rows = top.get("scenes")
    expected_top_rows: list[dict[str, Any]] = []
    task_paths: dict[str, Path] = {}
    expected_tasks = {
        row["task_id"]: row for row in preregistration_payload["tasks"]
    }
    if not isinstance(top_rows, list):
        raise TypeError("V96 runtime result-root manifest is malformed")
    for scene_id in RELEASE_SCENE_IDS:
        scene_root = root / scene_id
        scene_manifest_path = scene_root / "manifest.json"
        scene_manifest = _read_object(scene_manifest_path)
        result_rows = scene_manifest.get("results")
        if not isinstance(result_rows, list):
            raise TypeError("V96 runtime scene-result manifest is malformed")
        expected_scene_tasks = [
            row["task_id"]
            for row in preregistration_payload["tasks"]
            if row["scene_id"] == scene_id
        ]
        checked_rows: list[dict[str, Any]] = []
        for task_id in expected_scene_tasks:
            filename = f"r_{hashlib.sha256(task_id.encode('utf-8')).hexdigest()[:16]}.json"
            path = _absolute_nonsymlink(
                scene_root / filename,
                purpose="V96 runtime evidence",
            )
            if not path.is_file():
                raise FileNotFoundError(path)
            row = {
                "task_id": task_id,
                "file": filename,
                "sha256": _file_sha256(path),
            }
            checked_rows.append(row)
            task_paths[task_id] = path
        if (
            set(scene_manifest)
            != {"schema", "scene_id", "results", "inventory_sha256"}
            or scene_manifest.get("schema") != SCENE_RESULT_MANIFEST_SCHEMA
            or scene_manifest.get("scene_id") != scene_id
            or result_rows != checked_rows
            or scene_manifest.get("inventory_sha256")
            != _canonical_sha256(checked_rows)
            or {item.name for item in scene_root.iterdir()}
            != {"manifest.json", *(row["file"] for row in checked_rows)}
        ):
            raise ValueError("V96 runtime scene-result inventory changed")
        expected_top_rows.append(
            {
                "scene_id": scene_id,
                "directory": scene_id,
                "manifest_sha256": _file_sha256(scene_manifest_path),
                "inventory_sha256": scene_manifest["inventory_sha256"],
            }
        )
    if (
        set(top)
        != {
            "schema",
            "preregistration_sha256",
            "release_receipt_sha256",
            "dependency_contract_sha256",
            "scenes",
            "inventory_sha256",
        }
        or top.get("schema") != RUNTIME_RESULT_MANIFEST_SCHEMA
        or top.get("preregistration_sha256") != preregistration_sha256
        or top.get("release_receipt_sha256")
        != preregistration_payload["release_receipt_sha256"]
        or top.get("dependency_contract_sha256")
        != preregistration_payload["dependency_contract_sha256"]
        or top_rows != expected_top_rows
        or top.get("inventory_sha256") != _canonical_sha256(expected_top_rows)
        or {item.name for item in root.iterdir()}
        != {"manifest.json", *RELEASE_SCENE_IDS}
        or set(task_paths) != set(expected_tasks)
    ):
        raise ValueError("V96 runtime result-root inventory changed")
    return task_paths


def _runtime_child_command(
    *,
    scene_id: str,
    task_input: Path,
    output: Path,
    scratch: Path,
    preregistration: Mapping[str, Any],
    preregistration_sha256: str,
) -> list[str]:
    dependencies = preregistration["dependency_contract"]
    asset = dependencies["runtime_assets"][scene_id]
    return [
        sys.executable,
        "-m",
        "semantic_3d_chat.robot.v96_embodied_task_runner",
        "--task-input",
        str(task_input),
        "--expected-task-input-sha256",
        preregistration["runtime_task_input_sha256_by_scene"][scene_id],
        "--preregistration-sha256",
        preregistration_sha256,
        "--expected-release-receipt-sha256",
        preregistration["release_receipt_sha256"],
        "--dependency-contract-sha256",
        preregistration["dependency_contract_sha256"],
        "--runtime-config-inventory-sha256",
        dependencies["runtime_config_inventory"]["inventory_sha256"],
        "--runtime-source-inventory-sha256",
        dependencies["runtime_source_inventory"]["inventory_sha256"],
        "--implementation-source-inventory-sha256",
        dependencies["implementation_source_inventory"]["inventory_sha256"],
        "--runtime-asset-contract-sha256",
        asset["contract_sha256"],
        "--runtime-asset",
        str(PROJECT_ROOT / asset["asset_path"]),
        "--expected-runtime-asset-sha256",
        asset["asset_sha256"],
        "--expected-runtime-manifest-sha256",
        asset["manifest_file_sha256"],
        "--persistent-map",
        str(scratch / scene_id / "semantic_map.npz"),
        "--scan-output",
        str(scratch / scene_id / "scans"),
        "--output",
        str(output),
    ]


def build_runtime_evidence(
    *,
    task_id: str,
    scene_id: str,
    navigation_result: Mapping[str, Any],
    preregistration_sha256: str,
    release_receipt_sha256: str,
    dependency_contract_sha256: str,
    runtime_config_inventory_sha256: str,
    runtime_source_inventory_sha256: str,
    implementation_source_inventory_sha256: str,
    runtime_asset_contract_sha256: str,
    runtime_task_input_sha256: str,
    runtime_access_log: Mapping[str, Any],
) -> dict[str, Any]:
    """Create a hash-bound runtime envelope without semantic target metadata."""

    navigation = dict(navigation_result)
    access_log = dict(runtime_access_log)
    payload = {
        "schema": RUNTIME_EVIDENCE_SCHEMA,
        "task_id": task_id,
        "scene_id": scene_id,
        "navigation": navigation,
        "navigation_sha256": _canonical_sha256(navigation),
        "preregistration_sha256": preregistration_sha256,
        "release_receipt_sha256": release_receipt_sha256,
        "dependency_contract_sha256": dependency_contract_sha256,
        "runtime_config_inventory_sha256": runtime_config_inventory_sha256,
        "runtime_source_inventory_sha256": runtime_source_inventory_sha256,
        "implementation_source_inventory_sha256": (
            implementation_source_inventory_sha256
        ),
        "runtime_asset_contract_sha256": runtime_asset_contract_sha256,
        "runtime_task_input_sha256": runtime_task_input_sha256,
        "runtime_access_log": access_log,
        "runtime_access_log_sha256": _canonical_sha256(access_log),
        "forbidden_runtime_reads": 0,
        "oracle_runtime_reads": 0,
        "environmental_text_inputs": [],
        "scorer_only_target_category_loaded": False,
        "runtime_result_closed": True,
    }
    payload["evidence_identity_sha256"] = _canonical_sha256(payload)
    return payload


def build_runtime_access_log(paths: Sequence[str | Path]) -> dict[str, Any]:
    """Normalize the process audit into one closed, scorer-verifiable inventory."""

    loaded = sorted({str(Path(path).expanduser().resolve()) for path in paths})
    forbidden = [path for path in loaded if _runtime_path_is_forbidden(Path(path))]
    if forbidden:
        raise PermissionError(f"Forbidden V96 embodied runtime reads: {forbidden}")
    return {
        "schema": "semantic_3d_chat.v96_embodied_runtime_access_log.v1",
        "loaded_files": loaded,
        "loaded_file_count": len(loaded),
        "loaded_file_inventory_sha256": _canonical_sha256(loaded),
        "forbidden_accesses": [],
        "oracle_reads": 0,
        "qa_reads": 0,
        "training_reads": 0,
        "scorer_reads": 0,
        "block_forbidden": True,
    }


def _runtime_path_is_forbidden(path: Path) -> bool:
    absolute = path.resolve()
    forbidden_roots = (
        PROJECT_ROOT / "data/oracle",
        PROJECT_ROOT / "data/qa",
        PROJECT_ROOT / "data/rendered",
        PROJECT_ROOT / "data_gemma4/oracle",
        PROJECT_ROOT / "data_gemma4/qa",
        PROJECT_ROOT / "data_gemma4/features",
        PROJECT_ROOT / "data_gemma4/training",
        PROJECT_ROOT / "reports/gemma4/predictions",
        PROJECT_ROOT / "reports/gemma4/questions",
        PROJECT_ROOT / "reports/gemma4/scorer_only",
        PROJECT_ROOT / "configs/experiments",
    )
    if any(
        absolute == root.resolve() or absolute.is_relative_to(root.resolve())
        for root in forbidden_roots
    ):
        return True
    return absolute == Path(__file__).resolve()


def _validate_runtime_access_log(value: object) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError("V96 held-out runtime access log is malformed")
    result = dict(value)
    fields = {
        "schema",
        "loaded_files",
        "loaded_file_count",
        "loaded_file_inventory_sha256",
        "forbidden_accesses",
        "oracle_reads",
        "qa_reads",
        "training_reads",
        "scorer_reads",
        "block_forbidden",
    }
    loaded = result.get("loaded_files")
    if (
        set(result) != fields
        or result.get("schema")
        != "semantic_3d_chat.v96_embodied_runtime_access_log.v1"
        or not isinstance(loaded, list)
        or loaded != sorted(set(loaded))
        or any(not isinstance(path, str) or not Path(path).is_absolute() for path in loaded)
        or isinstance(result.get("loaded_file_count"), bool)
        or not isinstance(result.get("loaded_file_count"), int)
        or result.get("loaded_file_count") != len(loaded)
        or result.get("loaded_file_inventory_sha256") != _canonical_sha256(loaded)
        or result.get("forbidden_accesses") != []
        or any(
            isinstance(result.get(field), bool)
            or not isinstance(result.get(field), int)
            or result.get(field) != 0
            for field in (
                "oracle_reads",
                "qa_reads",
                "training_reads",
                "scorer_reads",
            )
        )
        or result.get("block_forbidden") is not True
        or any(_runtime_path_is_forbidden(Path(path)) for path in loaded)
    ):
        raise ValueError("V96 held-out runtime access-log contract failed")
    return result


def _finite_xy(value: object, label: str) -> tuple[float, float]:
    if not isinstance(value, list) or len(value) < 2:
        raise TypeError(f"{label} must be an XY-compatible list")
    result = (float(value[0]), float(value[1]))
    if not all(math.isfinite(item) for item in result):
        raise ValueError(f"{label} contains a nonfinite coordinate")
    return result


def _finite_vector(value: object, length: int, label: str) -> tuple[float, ...]:
    if not isinstance(value, list) or len(value) != length:
        raise TypeError(f"{label} must be a {length}-value list")
    result = tuple(float(item) for item in value)
    if not all(math.isfinite(item) for item in result):
        raise ValueError(f"{label} contains a nonfinite value")
    return result


def _close(left: float, right: float, *, tolerance: float = 1e-6) -> bool:
    return math.isclose(left, right, rel_tol=0.0, abs_tol=tolerance)


def _angle_residual_degrees(left: float, right: float) -> float:
    return math.degrees(
        math.atan2(
            math.sin(math.radians(left - right)),
            math.cos(math.radians(left - right)),
        )
    )


def _same_vector(left: Sequence[float], right: Sequence[float]) -> bool:
    return len(left) == len(right) and all(
        _close(float(a), float(b)) for a, b in zip(left, right, strict=True)
    )


def _validate_action_transition(
    call: Mapping[str, Any],
    before: Mapping[str, Any],
    after: Mapping[str, Any],
    *,
    observation_expected: bool,
    path: Path,
) -> None:
    """Recompute the bounded kinematic transition from call plus prior state."""

    tool = call.get("tool")
    arguments = call.get("arguments")
    if not isinstance(tool, str) or not isinstance(arguments, Mapping):
        raise TypeError(f"V96 held-out action call is malformed: {path}")
    before_position = _finite_vector(before.get("position_m"), 3, "before position")
    after_position = _finite_vector(after.get("position_m"), 3, "after position")
    before_camera = _finite_vector(
        before.get("camera_position_m"), 3, "before camera position"
    )
    after_camera = _finite_vector(
        after.get("camera_position_m"), 3, "after camera position"
    )
    before_delta = _finite_vector(
        before.get("last_movement_delta_m"), 3, "before movement delta"
    )
    after_delta = _finite_vector(
        after.get("last_movement_delta_m"), 3, "after movement delta"
    )
    before_yaw = float(before["body_yaw_degrees"])
    after_yaw = float(after["body_yaw_degrees"])
    before_camera_yaw = float(before["camera_yaw_degrees"])
    after_camera_yaw = float(after["camera_yaw_degrees"])
    before_pitch = float(before["pitch_degrees"])
    after_pitch = float(after["pitch_degrees"])
    if not all(
        math.isfinite(item)
        for item in (
            before_yaw,
            after_yaw,
            before_camera_yaw,
            after_camera_yaw,
            before_pitch,
            after_pitch,
        )
    ):
        raise ValueError(f"V96 held-out action state is nonfinite: {path}")

    if after.get("success") is not True:
        # Rejected actions cannot satisfy the acceptance gate.  Still require
        # the failure to be a monotonic, non-semantic protocol receipt.
        if (
            after.get("error_code") is None
            or int(after["action_count"]) <= int(before["action_count"])
            or int(after["map_version"]) != int(before["map_version"])
            or after.get("map_sha256") != before.get("map_sha256")
        ):
            raise ValueError(f"V96 held-out rejected-action receipt is invalid: {path}")
        return

    expected_action_increment = 2 if tool in _AUTO_SCAN_ACTIONS else 1
    if (
        int(after["action_count"])
        != int(before["action_count"]) + expected_action_increment
        or after.get("error_code") is not None
        or after.get("collision") is not False
        or not _close(after_position[2], 0.0)
        or not _close(after_camera[0], after_position[0])
        or not _close(after_camera[1], after_position[1])
        or not _close(after_camera[2], before_camera[2])
    ):
        raise ValueError(f"V96 held-out action counter/pose receipt is invalid: {path}")

    expected_position = before_position
    expected_body_yaw = before_yaw
    expected_camera_yaw = before_camera_yaw
    expected_pitch = before_pitch
    expected_distance = 0.0
    expected_turn = 0.0
    expected_delta = before_delta
    if tool == "turn":
        angle = float(arguments["angle_degrees"])
        expected_body_yaw = before_yaw + angle
        expected_camera_yaw = before_camera_yaw + angle
        expected_turn = angle
    elif tool == "look":
        yaw_delta = float(arguments["yaw_delta_degrees"])
        pitch_delta = float(arguments["pitch_delta_degrees"])
        expected_camera_yaw = before_camera_yaw + yaw_delta
        expected_pitch = before_pitch + pitch_delta
        expected_turn = yaw_delta
    elif tool in {"move_forward", "move_backward"}:
        distance = float(arguments["distance_meters"])
        direction = 1.0 if tool == "move_forward" else -1.0
        yaw = math.radians(before_yaw)
        dx = direction * distance * -math.sin(yaw)
        dy = direction * distance * math.cos(yaw)
        expected_position = (
            before_position[0] + dx,
            before_position[1] + dy,
            before_position[2],
        )
        expected_distance = distance
        expected_delta = (dx, dy, 0.0)
    elif tool == "move_to":
        expected_position = (
            float(arguments["x"]),
            float(arguments["y"]),
            before_position[2],
        )
        expected_delta = (
            expected_position[0] - before_position[0],
            expected_position[1] - before_position[1],
            0.0,
        )
        expected_distance = math.hypot(expected_delta[0], expected_delta[1])
    elif tool not in {"scan", "stop"}:
        raise ValueError(f"V96 held-out action tool is unknown: {path}")

    if (
        not _same_vector(after_position, expected_position)
        or not _same_vector(after_delta, expected_delta)
        or abs(_angle_residual_degrees(after_yaw, expected_body_yaw)) > 1e-6
        or abs(_angle_residual_degrees(after_camera_yaw, expected_camera_yaw)) > 1e-6
        or not _close(after_pitch, expected_pitch)
        or not _close(float(after["distance_moved"]), expected_distance)
        or not _close(float(after["turn_degrees"]), expected_turn)
        or (tool == "stop" and after.get("stopped") is not True)
        or (tool != "stop" and after.get("stopped") is not False)
        or observation_expected != (tool == "scan" or tool in _AUTO_SCAN_ACTIONS)
    ):
        raise ValueError(f"V96 held-out action transition does not match its call: {path}")


def _validate_runtime_evidence(
    path: Path,
    expected_task: Mapping[str, Any],
    *,
    preregistration_payload: Mapping[str, Any],
    preregistration_sha256: str,
    runtime_config: Mapping[str, Any],
) -> dict[str, Any]:
    value = _read_object(path)
    fields = {
        "schema",
        "task_id",
        "scene_id",
        "navigation",
        "navigation_sha256",
        "preregistration_sha256",
        "release_receipt_sha256",
        "dependency_contract_sha256",
        "runtime_config_inventory_sha256",
        "runtime_source_inventory_sha256",
        "implementation_source_inventory_sha256",
        "runtime_asset_contract_sha256",
        "runtime_task_input_sha256",
        "runtime_access_log",
        "runtime_access_log_sha256",
        "forbidden_runtime_reads",
        "oracle_runtime_reads",
        "environmental_text_inputs",
        "scorer_only_target_category_loaded",
        "runtime_result_closed",
        "evidence_identity_sha256",
    }
    identity = dict(value)
    evidence_identity = identity.pop("evidence_identity_sha256", None)
    dependencies = preregistration_payload["dependency_contract"]
    scene_asset = dependencies["runtime_assets"][expected_task["scene_id"]]
    access_log = _validate_runtime_access_log(value.get("runtime_access_log"))
    required_runtime_reads = {
        str((PROJECT_ROOT / scene_asset["asset_path"]).resolve()),
        str((PROJECT_ROOT / scene_asset["manifest_path"]).resolve()),
        str((NAVIGATION_CHECKPOINT / "policy.safetensors").resolve()),
        str((NAVIGATION_CHECKPOINT / "runtime_metadata.json").resolve()),
    }
    if (
        set(value) != fields
        or value.get("schema") != RUNTIME_EVIDENCE_SCHEMA
        or value.get("task_id") != expected_task["task_id"]
        or value.get("scene_id") != expected_task["scene_id"]
        or value.get("preregistration_sha256") != preregistration_sha256
        or value.get("release_receipt_sha256")
        != preregistration_payload["release_receipt_sha256"]
        or value.get("dependency_contract_sha256")
        != preregistration_payload["dependency_contract_sha256"]
        or value.get("runtime_config_inventory_sha256")
        != dependencies["runtime_config_inventory"]["inventory_sha256"]
        or value.get("runtime_source_inventory_sha256")
        != dependencies["runtime_source_inventory"]["inventory_sha256"]
        or value.get("implementation_source_inventory_sha256")
        != dependencies["implementation_source_inventory"]["inventory_sha256"]
        or value.get("runtime_asset_contract_sha256")
        != scene_asset["contract_sha256"]
        or value.get("runtime_task_input_sha256")
        != preregistration_payload["runtime_task_input_sha256_by_scene"][
            expected_task["scene_id"]
        ]
        or value.get("runtime_access_log_sha256") != _canonical_sha256(access_log)
        or not required_runtime_reads.issubset(access_log["loaded_files"])
        or isinstance(value.get("forbidden_runtime_reads"), bool)
        or not isinstance(value.get("forbidden_runtime_reads"), int)
        or value.get("forbidden_runtime_reads") != 0
        or isinstance(value.get("oracle_runtime_reads"), bool)
        or not isinstance(value.get("oracle_runtime_reads"), int)
        or value.get("oracle_runtime_reads") != 0
        or value.get("environmental_text_inputs") != []
        or value.get("scorer_only_target_category_loaded") is not False
        or value.get("runtime_result_closed") is not True
        or not isinstance(evidence_identity, str)
        or re.fullmatch(_SHA256_PATTERN, evidence_identity) is None
        or _canonical_sha256(identity) != evidence_identity
    ):
        raise ValueError(f"V96 held-out runtime envelope failed: {path}")
    navigation = value.get("navigation")
    if not isinstance(navigation, Mapping):
        raise TypeError(f"V96 held-out navigation result is malformed: {path}")
    steps = navigation.get("steps")
    expected_instruction_sha256 = hashlib.sha256(
        str(expected_task["instruction"]).encode("utf-8")
    ).hexdigest()
    navigation_fields = {
        "schema",
        "instruction_sha256",
        "termination_reason",
        "step_count",
        "steps",
        "transport",
        "policy_consumed_738_scene_tokens_every_decision",
        "policy_consumed_4_robot_tokens_every_decision",
        "numeric_tool_outputs_only",
        "successful_rgbd_refreshes_verified_before_next_decision",
        "environmental_text_inputs",
        "oracle_inputs_at_runtime",
        "held_out_navigation_claim",
    }
    if (
        set(navigation) != navigation_fields
        or navigation.get("schema")
        != "semantic_3d_chat.v96_co_resident_mcp_navigation.v1"
        or value.get("navigation_sha256") != _canonical_sha256(navigation)
        or navigation.get("instruction_sha256") != expected_instruction_sha256
        or navigation.get("transport") != TRANSPORT_MODE
        or navigation.get("policy_consumed_738_scene_tokens_every_decision")
        is not True
        or navigation.get("policy_consumed_4_robot_tokens_every_decision")
        is not True
        or navigation.get("numeric_tool_outputs_only") is not True
        or navigation.get(
            "successful_rgbd_refreshes_verified_before_next_decision"
        )
        is not True
        or navigation.get("environmental_text_inputs") != []
        or navigation.get("oracle_inputs_at_runtime") is not False
        or navigation.get("held_out_navigation_claim") is not False
        or not isinstance(steps, list)
        or not steps
        or navigation.get("step_count") != len(steps)
    ):
        raise ValueError(f"V96 held-out navigation contract failed: {path}")

    validated_steps: list[dict[str, Any]] = []
    prior_receipt: dict[str, Any] | None = None
    collisions = 0
    rejections = 0
    successful_scans = 0
    refreshed_scans = 0
    step_fields = {
        "index",
        "instruction_sha256",
        "call",
        "call_sha256",
        "proposal_sha256",
        "before_receipt",
        "before_binding",
        "after_binding",
        "receipt",
        "policy_context_audit",
        "rgbd_observation_expected",
        "map_refresh_verified_before_next_decision",
        "transport",
        "numeric_tool_output_only",
        "environmental_text_inputs",
        "oracle_inputs_at_runtime",
    }
    context_fields = {
        "schema",
        "active_prefix_shape",
        "active_prefix_sha256",
        "full_scene_memory_sha256",
        "base_scene_prefix_sha256",
        "robot_tokens_sha256",
        "map_sha256",
        "scene_tokens_consumed",
        "robot_tokens_consumed",
        "policy_consumed_738_scene_tokens",
        "policy_consumed_4_robot_tokens",
        "complete_scene_memory_used",
        "question_dependent_scene_retrieval",
        "target_grounding_used",
        "all_active_map_voxels_scored_for_target_grounding",
        "grounding_scored_voxels",
        "source_policy_was_retrained_on_v96",
        "transfer",
        "forward",
        "environmental_text_inputs",
        "oracle_inputs_at_runtime",
    }
    forward_fields = {
        "schema",
        "forward_call",
        "scene_shape",
        "robot_shape",
        "scene_prefix_sha256",
        "robot_tokens_sha256",
        "scene_tokens_processed",
        "robot_tokens_processed",
        "hidden_size",
        "all_scene_tokens_enter_attention_keys_and_values",
        "all_scene_tokens_enter_global_mean",
        "robot_tokens_enter_robot_value_mean",
        "question_dependent_scene_selection",
        "top_k_scene_selection",
        "environmental_text_inputs",
        "oracle_inputs_at_runtime",
        "transfer",
    }
    transfer_fields = {
        "source_scene_token_count",
        "target_scene_token_count",
        "robot_token_count",
        "active_token_count",
        "hidden_size",
        "source_weights_sha256",
        "source_metadata_sha256",
        "source_training_dataset_sha256",
        "source_training_status",
        "transfer_mode",
        "weights_changed",
        "retrained_on_v96",
        "held_out_navigation_claim",
        "every_scene_token_processed_by_attention",
        "every_scene_token_processed_by_global_mean",
        "environmental_text_inputs",
        "oracle_inputs_at_runtime",
    }
    continuity_fields = {
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
        "scan_coverage",
        "scan_count",
        "action_count",
        "stopped",
        "schema",
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
    for index, raw_step in enumerate(steps):
        if not isinstance(raw_step, Mapping):
            raise TypeError(f"V96 held-out step is malformed: {path}")
        step = dict(raw_step)
        before = step.get("before_receipt")
        after = step.get("receipt")
        context = step.get("policy_context_audit")
        call = step.get("call")
        if not all(isinstance(item, Mapping) for item in (before, after, context, call)):
            raise TypeError(f"V96 held-out step fields are malformed: {path}")
        before = dict(before)
        after = dict(after)
        context = dict(context)
        call = dict(call)
        raw_before_sha256 = _canonical_sha256(before)
        raw_after_sha256 = _canonical_sha256(after)
        before = validate_numeric_tool_receipt(
            before,
            require_continuous_binding=True,
        )
        after = validate_numeric_tool_receipt(
            after,
            require_continuous_binding=True,
        )
        if (
            _canonical_sha256(before) != raw_before_sha256
            or _canonical_sha256(after) != raw_after_sha256
        ):
            raise ValueError(f"V96 held-out receipt required JSON type coercion: {path}")
        tool = call.get("tool")
        observation_expected = tool == "scan" or tool in _AUTO_SCAN_ACTIONS
        refresh_verified = step.get("map_refresh_verified_before_next_decision")
        canonical_call = json.dumps(
            call,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
        revalidated_call = validate_tool_call_text(
            canonical_call,
            runtime_config,
            robot_state=before,
        )
        normalized_call = (
            None
            if revalidated_call.call is None
            else json.dumps(
                revalidated_call.call.as_dict(),
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=True,
                allow_nan=False,
            )
        )
        before_subset = {
            field: before[field]
            for field in (
                "schema",
                "scene_id",
                "scene_version",
                "map_version",
                "map_sha256",
                "scene_prefix_sha256",
                "scene_control_signature_sha256",
                "active_prefix_sha256",
                "robot_state_sha256",
                "robot_tokens_sha256",
                "active_binding_sha256",
            )
        }
        after_subset = {
            field: after[field]
            for field in before_subset
        }
        forward = context.get("forward")
        transfer = context.get("transfer")
        grounding_required = expected_task["kind"] in {"face", "approach"}
        if (
            set(step) != step_fields
            or step.get("index") != index
            or step.get("instruction_sha256") != expected_instruction_sha256
            or step.get("transport") != TRANSPORT_MODE
            or step.get("numeric_tool_output_only") is not True
            or step.get("environmental_text_inputs") != []
            or step.get("oracle_inputs_at_runtime") is not False
            or step.get("rgbd_observation_expected") is not observation_expected
            or (
                after.get("success") is True
                and observation_expected
                and refresh_verified is not True
            )
            or (
                (after.get("success") is not True or not observation_expected)
                and refresh_verified is not False
            )
            or set(context) != context_fields
            or context.get("schema")
            != "semantic_3d_chat.v96_release_action_context.v1"
            or context.get("policy_consumed_738_scene_tokens") is not True
            or context.get("policy_consumed_4_robot_tokens") is not True
            or context.get("scene_tokens_consumed") != 738
            or context.get("robot_tokens_consumed") != 4
            or context.get("active_prefix_shape") != [1, 742, 1536]
            or context.get("complete_scene_memory_used") is not True
            or context.get("question_dependent_scene_retrieval") is not False
            or context.get("source_policy_was_retrained_on_v96") is not False
            or context.get("environmental_text_inputs") != []
            or context.get("oracle_inputs_at_runtime") is not False
            or context.get("active_prefix_sha256")
            != before.get("active_prefix_sha256")
            or context.get("full_scene_memory_sha256")
            != before.get("scene_control_signature_sha256")
            or context.get("robot_tokens_sha256")
            != before.get("robot_tokens_sha256")
            or context.get("map_sha256") != before.get("map_sha256")
            or context.get("base_scene_prefix_sha256")
            != before.get("scene_prefix_sha256")
            or context.get("target_grounding_used") is not grounding_required
            or (
                grounding_required
                and (
                    context.get("all_active_map_voxels_scored_for_target_grounding")
                    is not True
                    or context.get("grounding_scored_voxels")
                    != before.get("source_voxels")
                )
            )
            or (
                not grounding_required
                and (
                    context.get("all_active_map_voxels_scored_for_target_grounding")
                    is not None
                    or context.get("grounding_scored_voxels") is not None
                )
            )
            or not isinstance(forward, Mapping)
            or set(forward) != forward_fields
            or forward.get("schema")
            != "semantic_3d_chat.v96_navigation_transfer_forward.v1"
            or isinstance(forward.get("forward_call"), bool)
            or forward.get("forward_call") != index + 1
            or forward.get("scene_shape") != [1, 738, 1536]
            or forward.get("robot_shape") != [1, 4, 1536]
            or forward.get("scene_prefix_sha256")
            != before.get("scene_control_signature_sha256")
            or forward.get("robot_tokens_sha256") != before.get("robot_tokens_sha256")
            or forward.get("scene_tokens_processed") != 738
            or forward.get("robot_tokens_processed") != 4
            or forward.get("hidden_size") != 1536
            or forward.get("all_scene_tokens_enter_attention_keys_and_values") is not True
            or forward.get("all_scene_tokens_enter_global_mean") is not True
            or forward.get("robot_tokens_enter_robot_value_mean") is not True
            or forward.get("question_dependent_scene_selection") is not False
            or forward.get("top_k_scene_selection") is not False
            or forward.get("environmental_text_inputs") != []
            or forward.get("oracle_inputs_at_runtime") is not False
            or not isinstance(transfer, Mapping)
            or not isinstance(forward.get("transfer"), Mapping)
            or set(transfer) != transfer_fields
            or dict(transfer) != dict(forward.get("transfer", {}))
            or transfer.get("source_scene_token_count") != 258
            or transfer.get("target_scene_token_count") != 738
            or transfer.get("robot_token_count") != 4
            or transfer.get("active_token_count") != 742
            or transfer.get("hidden_size") != 1536
            or transfer.get("source_weights_sha256") != V3_POLICY_WEIGHTS_SHA256
            or transfer.get("source_metadata_sha256") != V3_POLICY_METADATA_SHA256
            or transfer.get("source_training_dataset_sha256")
            != V3_TRAINING_DATASET_SHA256
            or transfer.get("transfer_mode") != TRANSFER_MODE
            or transfer.get("source_training_status")
            != "supervised_continuous_semantic_grounded_navigation_policy_v3"
            or transfer.get("weights_changed") is not False
            or transfer.get("retrained_on_v96") is not False
            or transfer.get("held_out_navigation_claim") is not False
            or transfer.get("every_scene_token_processed_by_attention") is not True
            or transfer.get("every_scene_token_processed_by_global_mean") is not True
            or transfer.get("environmental_text_inputs") != []
            or transfer.get("oracle_inputs_at_runtime") is not False
            or revalidated_call.call is None
            or revalidated_call.error_code is not None
            or revalidated_call.call.as_dict() != call
            or normalized_call != canonical_call
            or step.get("call_sha256")
            != hashlib.sha256(canonical_call.encode("utf-8")).hexdigest()
            or step.get("proposal_sha256")
            != hashlib.sha256(canonical_call.encode("utf-8")).hexdigest()
            or step.get("before_binding") != before_subset
            or step.get("after_binding") != after_subset
            or before.get("scene_id") != expected_task["scene_id"]
            or after.get("scene_id") != expected_task["scene_id"]
            or before.get("success") is not True
            or (
                prior_receipt is not None
                and any(before[field] != prior_receipt[field] for field in continuity_fields)
            )
        ):
            raise ValueError(f"V96 held-out 738+4 action binding failed: {path}")
        _validate_action_transition(
            call,
            before,
            after,
            observation_expected=observation_expected,
            path=path,
        )
        if after.get("success") is True and observation_expected:
            if (
                int(after["map_version"]) != int(before["map_version"]) + 1
                or int(after["scan_count"]) != int(before["scan_count"]) + 1
                or after["map_sha256"] == before["map_sha256"]
                or after["scene_control_signature_sha256"]
                == before["scene_control_signature_sha256"]
                or after["active_prefix_sha256"] == before["active_prefix_sha256"]
                or int(after["valid_depth_pixels"]) < 1
                or int(after["visible_voxels"]) < 1
                or not isinstance(after.get("observation_id"), str)
            ):
                raise ValueError(f"V96 held-out RGB-D transition is invalid: {path}")
        elif after.get("success") is True and (
            after["map_version"] != before["map_version"]
            or after["map_sha256"] != before["map_sha256"]
            or after["scene_control_signature_sha256"]
            != before["scene_control_signature_sha256"]
        ):
            raise ValueError(f"V96 non-observation action changed scene memory: {path}")
        prior_receipt = after
        collisions += int(after.get("collision") is True)
        rejections += int(after.get("success") is not True)
        if tool == "scan" and after.get("success") is True:
            successful_scans += 1
            refreshed = (
                int(after["map_version"]) > int(before["map_version"])
                and after["map_sha256"] != before["map_sha256"]
                and after["scene_control_signature_sha256"]
                != before["scene_control_signature_sha256"]
            )
            refreshed_scans += int(refreshed)
        validated_steps.append(step)
    initial_receipt = dict(validated_steps[0]["before_receipt"])
    final_receipt = dict(validated_steps[-1]["receipt"])
    if (
        initial_receipt.get("seed") != EVALUATION_RESET_SEED
        or initial_receipt.get("map_version") != 0
        or initial_receipt.get("scene_version") != 0
        or initial_receipt.get("scan_count") != 0
        or initial_receipt.get("action_count") != 0
        or initial_receipt.get("stopped") is not False
        or initial_receipt.get("collision") is not False
        or (
            navigation.get("termination_reason") == "stop"
            and final_receipt.get("stopped") is not True
        )
    ):
        raise ValueError(f"V96 held-out task did not start from an isolated reset: {path}")
    return {
        "task_id": value["task_id"],
        "scene_id": value["scene_id"],
        "task_key": expected_task["key"],
        "kind": expected_task["kind"],
        "oracle_category": _SCORER_ORACLE_CATEGORY_BY_KEY[expected_task["key"]],
        "result_path": str(path),
        "result_sha256": _file_sha256(path),
        "steps": validated_steps,
        "initial_receipt": initial_receipt,
        "final_receipt": final_receipt,
        "collision_count": collisions,
        "rejected_action_count": rejections,
        "successful_scan_count": successful_scans,
        "refreshed_scan_count": refreshed_scans,
        "termination_reason": navigation["termination_reason"],
        "called_tools": [step["call"]["tool"] for step in validated_steps],
    }


def _oracle_instance(
    oracle: Mapping[str, Any],
    *,
    category: str,
) -> Mapping[str, Any]:
    instances = oracle.get("instances")
    if not isinstance(instances, list):
        raise TypeError("V96 held-out oracle has no instance list")
    matches = [
        row
        for row in instances
        if isinstance(row, Mapping) and row.get("category") == category
    ]
    if len(matches) != 1:
        raise ValueError(f"Expected exactly one held-out oracle category: {category}")
    return matches[0]


def run_heldout_runtime(
    preregistration: str | Path,
    *,
    runtime_input_root: str | Path = DEFAULT_RUNTIME_INPUT_ROOT,
    runtime_result_root: str | Path = DEFAULT_RUNTIME_RESULT_ROOT,
    scratch_root: str | Path = DEFAULT_RUNTIME_SCRATCH_ROOT,
    release_verifier: Callable[[], Mapping[str, Any]] = (
        run_isolated_v96_release_verification
    ),
    subprocess_runner: Callable[..., Any] = subprocess.run,
) -> dict[str, Any]:
    """Run six isolated model processes and publish 36 closed evidence files."""

    receipt = validate_promoted_v96_release_receipt(release_verifier())
    prereg = authenticate_preregistration(
        preregistration,
        release_receipt=receipt,
    )
    payload = prereg["payload"]
    input_root = _absolute_nonsymlink(
        runtime_input_root,
        purpose="V96 runtime task-input root",
    )
    result_root = _absolute_nonsymlink(
        runtime_result_root,
        purpose="V96 runtime result root",
    )
    scratch_target = _absolute_nonsymlink(
        scratch_root,
        purpose="V96 runtime scratch root",
    )
    if result_root.exists() or result_root.is_symlink():
        raise FileExistsError(result_root)
    if input_root.exists():
        input_manifest = _runtime_input_manifest(
            input_root,
            payload,
            preregistration_sha256=prereg["sha256"],
        )
    else:
        input_manifest = write_runtime_task_inputs(
            input_root,
            payload,
            preregistration_sha256=prereg["sha256"],
        )
    input_by_scene = {
        row["scene_id"]: input_root / row["file"]
        for row in input_manifest["scenes"]
    }
    result_root.parent.mkdir(parents=True, exist_ok=True)
    scratch_target.parent.mkdir(parents=True, exist_ok=True)
    runtime_config = load_config(DEFAULT_EMBODIED_CONFIG)
    if (
        runtime_config.get("robot", {}).get("auto_scan_after_motion") is not True
        or runtime_config.get("scene", {}).get("room_size_m") != [6.0, 5.0, 3.0]
        or runtime_config.get("language", {}).get("model_id")
        != "google/gemma-4-E2B-it"
    ):
        raise ValueError("V96 held-out runtime config changed")
    expected_tasks = {row["task_id"]: row for row in payload["tasks"]}
    env = os.environ.copy()
    source_root = str(PROJECT_ROOT / "src")
    existing_pythonpath = env.get("PYTHONPATH")
    env["PYTHONPATH"] = (
        source_root
        if not existing_pythonpath
        else source_root + os.pathsep + existing_pythonpath
    )
    child_commands: list[list[str]] = []
    with tempfile.TemporaryDirectory(
        prefix=f".{result_root.name}.",
        dir=result_root.parent,
    ) as result_raw, tempfile.TemporaryDirectory(
        prefix=f".{scratch_target.name}.",
        dir=scratch_target.parent,
    ) as scratch_raw:
        stage = Path(result_raw)
        scratch = Path(scratch_raw)
        for scene_id in RELEASE_SCENE_IDS:
            command = _runtime_child_command(
                scene_id=scene_id,
                task_input=input_by_scene[scene_id],
                output=stage / scene_id,
                scratch=scratch,
                preregistration=payload,
                preregistration_sha256=prereg["sha256"],
            )
            child_commands.append(command)
            subprocess_runner(
                command,
                cwd=PROJECT_ROOT,
                env=env,
                check=True,
            )

        # Authenticate every child artifact and recompute every numeric action
        # transition before making the result root visible.
        task_paths: dict[str, Path] = {}
        top_rows: list[dict[str, Any]] = []
        for scene_id in RELEASE_SCENE_IDS:
            scene_root = stage / scene_id
            manifest_path = scene_root / "manifest.json"
            manifest = _read_object(manifest_path)
            rows = manifest.get("results")
            if (
                manifest.get("schema") != SCENE_RESULT_MANIFEST_SCHEMA
                or manifest.get("scene_id") != scene_id
                or not isinstance(rows, list)
                or manifest.get("inventory_sha256") != _canonical_sha256(rows)
            ):
                raise ValueError("V96 runtime child returned an invalid manifest")
            for row in rows:
                if not isinstance(row, Mapping):
                    raise TypeError("V96 runtime child result row is malformed")
                task_id = row.get("task_id")
                filename = row.get("file")
                if (
                    not isinstance(task_id, str)
                    or task_id not in expected_tasks
                    or expected_tasks[task_id]["scene_id"] != scene_id
                    or not isinstance(filename, str)
                ):
                    raise ValueError("V96 runtime child result identity changed")
                evidence_path = _absolute_nonsymlink(
                    scene_root / filename,
                    purpose="V96 runtime child evidence",
                )
                if not evidence_path.is_file():
                    raise FileNotFoundError(evidence_path)
                if row.get("sha256") != _file_sha256(evidence_path):
                    raise ValueError("V96 runtime child result hash changed")
                task_paths[task_id] = evidence_path
            top_rows.append(
                {
                    "scene_id": scene_id,
                    "directory": scene_id,
                    "manifest_sha256": _file_sha256(manifest_path),
                    "inventory_sha256": manifest["inventory_sha256"],
                }
            )
        if set(task_paths) != set(expected_tasks):
            raise ValueError("V96 runtime child task inventory is incomplete")
        for task_id in sorted(expected_tasks):
            _validate_runtime_evidence(
                task_paths[task_id],
                expected_tasks[task_id],
                preregistration_payload=payload,
                preregistration_sha256=prereg["sha256"],
                runtime_config=runtime_config,
            )
        top_manifest = {
            "schema": RUNTIME_RESULT_MANIFEST_SCHEMA,
            "preregistration_sha256": prereg["sha256"],
            "release_receipt_sha256": payload["release_receipt_sha256"],
            "dependency_contract_sha256": payload["dependency_contract_sha256"],
            "scenes": top_rows,
            "inventory_sha256": _canonical_sha256(top_rows),
        }
        _write_json_create_once(stage / "manifest.json", top_manifest)
        os.replace(stage, result_root)

    discovered = _result_paths_from_root(
        result_root,
        payload,
        preregistration_sha256=prereg["sha256"],
    )
    return {
        "phase": "v96_embodied_navigation_runtime_complete",
        "passed": True,
        "scene_count": len(RELEASE_SCENE_IDS),
        "task_count": len(discovered),
        "runtime_input_root": str(input_root),
        "runtime_result_root": str(result_root),
        "runtime_result_manifest_sha256": _file_sha256(result_root / "manifest.json"),
        "child_process_count": len(child_commands),
        "forbidden_runtime_reads": 0,
        "oracle_runtime_reads": 0,
        "oracle_opened_by_parent": False,
    }


def _point_to_box_distance(
    point: tuple[float, float],
    minimum: tuple[float, float],
    maximum: tuple[float, float],
) -> float:
    dx = max(minimum[0] - point[0], 0.0, point[0] - maximum[0])
    dy = max(minimum[1] - point[1], 0.0, point[1] - maximum[1])
    return math.hypot(dx, dy)


def score_heldout_results(
    preregistration: str | Path,
    result_paths: Mapping[str, str | Path],
    *,
    oracle_root: str | Path,
    release_verifier: Callable[[], Mapping[str, Any]] = (
        run_isolated_v96_release_verification
    ),
) -> dict[str, Any]:
    """Validate all inference evidence first, then open oracle and score it."""

    receipt = validate_promoted_v96_release_receipt(release_verifier())
    dependencies = validate_embodied_dependency_contract(
        build_embodied_dependency_contract()
    )
    prereg = authenticate_preregistration(
        preregistration,
        release_receipt=receipt,
    )
    expected_tasks = prereg["payload"]["tasks"]
    expected_by_id = {row["task_id"]: row for row in expected_tasks}
    if set(result_paths) != set(expected_by_id):
        raise ValueError("V96 held-out result inventory differs from preregistration")
    runtime_config = load_config(DEFAULT_EMBODIED_CONFIG)
    if (
        runtime_config.get("robot", {}).get("auto_scan_after_motion") is not True
        or runtime_config.get("scene", {}).get("room_size_m") != [6.0, 5.0, 3.0]
        or runtime_config.get("language", {}).get("model_id")
        != "google/gemma-4-E2B-it"
    ):
        raise ValueError("V96 held-out runtime config changed")

    # Close and validate every runtime result before resolving/opening oracle.
    runtime_rows = [
        _validate_runtime_evidence(
            Path(os.path.abspath(Path(result_paths[task_id]).expanduser())),
            expected_by_id[task_id],
            preregistration_payload=prereg["payload"],
            preregistration_sha256=prereg["sha256"],
            runtime_config=runtime_config,
        )
        for task_id in sorted(expected_by_id)
    ]

    root = _absolute_nonsymlink(oracle_root, purpose="V96 scorer-only oracle root")
    canonical_oracle_root = _absolute_nonsymlink(
        DEFAULT_ORACLE_ROOT,
        purpose="V96 scorer-only oracle root",
    )
    if root != canonical_oracle_root or not root.is_dir():
        raise ValueError("V96 held-out scorer requires the canonical isolated oracle root")
    oracle_cache: dict[str, dict[str, Any]] = {}
    oracle_inventory: list[dict[str, Any]] = []
    scores: list[dict[str, Any]] = []
    for runtime in runtime_rows:
        scene_id = runtime["scene_id"]
        if scene_id not in oracle_cache:
            oracle_path = root / scene_id / "oracle.json"
            oracle = _read_object(oracle_path)
            if oracle.get("scene_id") != scene_id:
                raise ValueError("V96 held-out oracle scene identity changed")
            oracle_cache[scene_id] = oracle
            oracle_inventory.append(
                {
                    "scene_id": scene_id,
                    "path": oracle_path.relative_to(PROJECT_ROOT).as_posix(),
                    "sha256": _file_sha256(oracle_path),
                }
            )
        oracle = oracle_cache[scene_id]
        initial = runtime["initial_receipt"]
        final = runtime["final_receipt"]
        initial_xy = _finite_xy(initial.get("position_m"), "initial position")
        final_xy = _finite_xy(final.get("position_m"), "final position")
        checks = {
            "collision_free": runtime["collision_count"] == 0,
            "no_rejected_actions": runtime["rejected_action_count"] == 0,
            "terminated": runtime["termination_reason"] in {"stop", "max_steps"},
        }
        metrics: dict[str, Any] = {}
        category = runtime["oracle_category"]
        if runtime["kind"] == "scan":
            checks["successful_scan"] = runtime["successful_scan_count"] >= 1
            checks["scan_refreshed_map_and_prefix"] = (
                runtime["refreshed_scan_count"] == runtime["successful_scan_count"]
            )
        else:
            if not isinstance(category, str):
                raise TypeError("V96 held-out target task lacks an oracle category")
            target = _oracle_instance(oracle, category=category)
            center = _finite_xy(target.get("expected_center_xyz_m"), "target center")
            if runtime["kind"] == "face":
                delta = (center[0] - final_xy[0], center[1] - final_xy[1])
                desired = math.degrees(math.atan2(-delta[0], delta[1]))
                yaw = float(final["body_yaw_degrees"])
                residual = math.degrees(
                    math.atan2(
                        math.sin(math.radians(desired - yaw)),
                        math.cos(math.radians(desired - yaw)),
                    )
                )
                metrics["absolute_face_residual_degrees"] = abs(residual)
                checks["face_residual"] = abs(residual) <= float(
                    ACCEPTANCE_GATES["maximum_face_residual_degrees"]
                )
                checks["explicit_stop"] = runtime["termination_reason"] == "stop"
            elif runtime["kind"] == "approach":
                bbox = target.get("bbox")
                if not isinstance(bbox, Mapping):
                    raise TypeError("V96 held-out target has no oracle bounding box")
                minimum = _finite_xy(bbox.get("min_xyz_m"), "bbox minimum")
                maximum = _finite_xy(bbox.get("max_xyz_m"), "bbox maximum")
                initial_distance = math.dist(initial_xy, center)
                final_distance = math.dist(final_xy, center)
                progress = initial_distance - final_distance
                standoff = _point_to_box_distance(final_xy, minimum, maximum)
                metrics.update(
                    {
                        "oracle_center_progress_m": progress,
                        "final_oracle_bbox_standoff_m": standoff,
                    }
                )
                checks["approach_progress"] = progress >= float(
                    ACCEPTANCE_GATES["minimum_approach_center_progress_m"]
                )
                checks["approach_standoff"] = standoff <= float(
                    ACCEPTANCE_GATES["maximum_approach_bbox_standoff_m"]
                )
                checks["explicit_stop"] = runtime["termination_reason"] == "stop"
                if runtime["task_key"] == "scan_approach":
                    called_tools = runtime["called_tools"]
                    checks["explicit_scan_before_approach"] = (
                        "scan" in called_tools
                        and called_tools.index("scan")
                        < next(
                            (
                                index
                                for index, tool in enumerate(called_tools)
                                if tool
                                in {"move_forward", "move_backward", "move_to"}
                            ),
                            len(called_tools),
                        )
                        and runtime["successful_scan_count"] >= 1
                    )
            else:
                raise ValueError("Unknown V96 held-out task kind")
        scores.append(
            {
                "task_id": runtime["task_id"],
                "scene_id": scene_id,
                "kind": runtime["kind"],
                "runtime_result_sha256": runtime["result_sha256"],
                "metrics": metrics,
                "checks": checks,
                "passed": all(checks.values()),
            }
        )

    total_collisions = sum(row["collision_count"] for row in runtime_rows)
    total_rejections = sum(row["rejected_action_count"] for row in runtime_rows)
    successful_scans = sum(row["successful_scan_count"] for row in runtime_rows)
    passed = sum(row["passed"] for row in scores)
    gates = {
        "all_tasks_present": len(scores) == ACCEPTANCE_GATES["required_task_count"],
        "all_tasks_pass": passed == len(scores),
        "zero_collisions": total_collisions == 0,
        "zero_rejected_actions": total_rejections == 0,
        "minimum_successful_scans": successful_scans
        >= ACCEPTANCE_GATES["minimum_successful_scan_count"],
    }
    score_payload: dict[str, Any] = {
        "schema": SCORE_SCHEMA,
        "status": "navigation_heldout_evaluation",
        "passed": all(gates.values()),
        "navigation_success_measured": True,
        "navigation_held_out": True,
        "static_unseen_claim": False,
        "runtime_evidence_validated_before_oracle_open": True,
        "runtime_process_read_oracle": False,
        "scorer_read_oracle": True,
        "preregistration_sha256": prereg["sha256"],
        "release_receipt_sha256": prereg["payload"]["release_receipt_sha256"],
        "dependency_contract_sha256": dependencies["contract_sha256"],
        "implementation_source_inventory_sha256": dependencies[
            "implementation_source_inventory"
        ]["inventory_sha256"],
        "scene_count": len(oracle_cache),
        "scorer_oracle_inventory_sha256": _canonical_sha256(oracle_inventory),
        "task_count": len(scores),
        "passed_task_count": passed,
        "collision_count": total_collisions,
        "rejected_action_count": total_rejections,
        "successful_scan_count": successful_scans,
        "gates": gates,
        "tasks": scores,
        "environmental_text_inputs_at_runtime": [],
        "oracle_inputs_at_runtime": False,
    }
    score_payload["score_identity_sha256"] = _canonical_sha256(score_payload)
    return score_payload


def write_heldout_score(
    preregistration: str | Path,
    runtime_result_root: str | Path,
    *,
    oracle_root: str | Path = DEFAULT_ORACLE_ROOT,
    destination: str | Path = DEFAULT_SCORE,
    release_verifier: Callable[[], Mapping[str, Any]] = (
        run_isolated_v96_release_verification
    ),
) -> dict[str, Any]:
    """Authenticate immutable runtime results, score once, and write once."""

    receipt = validate_promoted_v96_release_receipt(release_verifier())
    prereg = authenticate_preregistration(
        preregistration,
        release_receipt=receipt,
    )
    output = _absolute_nonsymlink(destination, purpose="V96 held-out score")
    if output.exists() or output.is_symlink():
        raise FileExistsError(output)
    result_paths = _result_paths_from_root(
        runtime_result_root,
        prereg["payload"],
        preregistration_sha256=prereg["sha256"],
    )
    score = score_heldout_results(
        preregistration,
        result_paths,
        oracle_root=oracle_root,
        release_verifier=lambda: receipt,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    _write_json_create_once(output, score)
    return {
        "phase": "v96_embodied_navigation_score_written",
        "passed": score["passed"],
        "path": str(output),
        "sha256": _file_sha256(output),
        "score": score,
    }


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    subparsers = result.add_subparsers(dest="command", required=True)
    subparsers.add_parser("preflight")
    preregister = subparsers.add_parser("preregister")
    preregister.add_argument("--output", default=str(DEFAULT_PREREGISTRATION))
    authenticate = subparsers.add_parser("authenticate")
    authenticate.add_argument(
        "--preregistration",
        default=str(DEFAULT_PREREGISTRATION),
    )
    runtime = subparsers.add_parser("run")
    runtime.add_argument(
        "--preregistration",
        default=str(DEFAULT_PREREGISTRATION),
    )
    runtime.add_argument(
        "--runtime-input-root",
        default=str(DEFAULT_RUNTIME_INPUT_ROOT),
    )
    runtime.add_argument(
        "--runtime-result-root",
        default=str(DEFAULT_RUNTIME_RESULT_ROOT),
    )
    runtime.add_argument(
        "--scratch-root",
        default=str(DEFAULT_RUNTIME_SCRATCH_ROOT),
    )
    score = subparsers.add_parser("score")
    score.add_argument(
        "--preregistration",
        default=str(DEFAULT_PREREGISTRATION),
    )
    score.add_argument(
        "--runtime-result-root",
        default=str(DEFAULT_RUNTIME_RESULT_ROOT),
    )
    score.add_argument("--oracle-root", default=str(DEFAULT_ORACLE_ROOT))
    score.add_argument("--output", default=str(DEFAULT_SCORE))
    return result


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    if args.command == "preflight":
        report = preflight()
    elif args.command == "preregister":
        report = write_preregistration(args.output)
    elif args.command == "authenticate":
        receipt = validate_promoted_v96_release_receipt(
            run_isolated_v96_release_verification()
        )
        dependencies = build_embodied_dependency_contract()
        authenticated = authenticate_preregistration(
            args.preregistration,
            release_receipt=receipt,
        )
        report = {
            "phase": "v96_embodied_navigation_preregistration_authenticated",
            "passed": True,
            "path": authenticated["path"],
            "sha256": authenticated["sha256"],
            "dependency_contract_sha256": dependencies["contract_sha256"],
            "oracle_opened": False,
        }
    elif args.command == "run":
        report = run_heldout_runtime(
            args.preregistration,
            runtime_input_root=args.runtime_input_root,
            runtime_result_root=args.runtime_result_root,
            scratch_root=args.scratch_root,
        )
    elif args.command == "score":
        report = write_heldout_score(
            args.preregistration,
            args.runtime_result_root,
            oracle_root=args.oracle_root,
            destination=args.output,
        )
    else:  # pragma: no cover - argparse enforces the subcommand
        raise RuntimeError("Unknown V96 embodied evaluator command")
    print(json.dumps(report, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "ACCEPTANCE_GATES",
    "DEFAULT_PREREGISTRATION",
    "DEFAULT_RUNTIME_INPUT_ROOT",
    "DEFAULT_RUNTIME_RESULT_ROOT",
    "DEFAULT_SCORE",
    "NAVIGATION_TRAIN_SCENES",
    "NAVIGATION_VALIDATION_SCENES",
    "PREREGISTRATION_SCHEMA",
    "RUNTIME_EVIDENCE_SCHEMA",
    "TASK_TEMPLATES",
    "HeldOutTask",
    "authenticate_preregistration",
    "build_embodied_dependency_contract",
    "build_preregistration_payload",
    "build_runtime_evidence",
    "preflight",
    "run_heldout_runtime",
    "runtime_task_input_payload",
    "score_heldout_results",
    "validate_embodied_dependency_contract",
    "write_heldout_score",
    "write_preregistration",
    "write_runtime_task_inputs",
]
