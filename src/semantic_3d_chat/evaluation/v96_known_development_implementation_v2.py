"""Create-once source seal for V96's inference-safe evaluator-auth revision.

The v1 sources and seal remain immutable historical evidence.  This successor
binds the separate candidate-attestation stage and every v2 evaluator source
before that attestation or any question-bearing output can be created.
"""

from __future__ import annotations

import argparse
import ast
import functools
import hashlib
import json
import os
import re
import tempfile
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any, Final, ParamSpec, TypeVar

from semantic_3d_chat.chat.file_audit import FileAccessAudit
from semantic_3d_chat.chat.runtime_config import load_runtime_config
from semantic_3d_chat.chat.v94_strict_multiscene_runtime import (
    EXPECTED_BANKS as V94_BANKS,
)
from semantic_3d_chat.config import PROJECT_ROOT
from semantic_3d_chat.evaluation.v95_strict_causal_successor_preflight import (
    FRESH_BANK_NAME as V95_BANK_NAME,
)
from semantic_3d_chat.evaluation.v95_strict_causal_successor_preflight import (
    load_config_v95,
)
from semantic_3d_chat.evaluation.v96_atomic_pair_repair_preflight import (
    CONFIG,
    load_config_v96,
)
from semantic_3d_chat.evaluation.v96_atomic_pair_repair_preflight import (
    FRESH_BANK_NAME as V96_BANK_NAME,
)
from semantic_3d_chat.evaluation.v96_evaluation_io_v2 import (
    physical_path_v96_v2,
    read_json_strict_v96_v2,
    write_json_create_once_v96_v2,
)
from semantic_3d_chat.evaluation.v96_known_development_common import (
    authenticate_fixed_final_candidate_v96 as authenticate_full_chain_v1,
)
from semantic_3d_chat.evaluation.v96_known_development_common import evaluation_paths_v96
from semantic_3d_chat.evaluation.v96_known_development_implementation import (
    IMPLEMENTATION_SEAL as V1_IMPLEMENTATION_SEAL,
)
from semantic_3d_chat.evaluation.v96_known_development_implementation import (
    authenticate_evaluation_implementation_v96,
    contract_source_inventory_v96,
    exclusive_evaluation_lock_v96,
)
from semantic_3d_chat.training.train_v96_atomic_pair_repair import (
    combined_lora_settings_v96,
)

ARTIFACT: Final[str] = "gemma4_v96_known_development_implementation_seal_v2"
SCHEMA_VERSION: Final[int] = 96
IMPLEMENTATION_SEAL_V2: Final[Path] = (
    PROJECT_ROOT
    / "reports/gemma4/metrics/gemma4_v96_known_development_implementation_seal_v2.json"
)
CANDIDATE_ATTESTATION: Final[Path] = (
    PROJECT_ROOT
    / "reports/gemma4/metrics/gemma4_v96_fixed_final_candidate_attestation_v2.json"
)
V95_CONFIG: Final[Path] = (
    PROJECT_ROOT / "configs/experiments/gemma4_v95_strict_causal_successor.yaml"
)
V85_RUNTIME_METADATA: Final[Path] = (
    PROJECT_ROOT
    / "reports/gemma4/artifacts/v85_strict_runtime_candidate/runtime_metadata.json"
)

_SOURCE_ROOT: Final[Path] = PROJECT_ROOT / "src"
_IMPLEMENTATION_ROOT_MODULES_V2: Final[tuple[str, ...]] = (
    "semantic_3d_chat.evaluation.v96_evaluation_io_v2",
    "semantic_3d_chat.evaluation.v96_known_development_candidate_attestation",
    "semantic_3d_chat.evaluation.v96_known_development_common_v2",
    "semantic_3d_chat.evaluation.v96_known_development_implementation_v2",
    "semantic_3d_chat.evaluation.predict_v96_known_development_v2",
    "semantic_3d_chat.evaluation.authenticate_v96_known_development_v2",
    "semantic_3d_chat.evaluation.score_v96_known_development_v2",
    "semantic_3d_chat.evaluation.nll_v96_known_development_v2",
    "semantic_3d_chat.evaluation.seal_v96_known_development_v2",
    # Loaded dynamically by load_future_trainer_v96 in model-bearing stages.
    "semantic_3d_chat.training.train_v96_atomic_pair_repair",
)


def _module_source_v96_v2(module: str) -> Path | None:
    candidate = _SOURCE_ROOT.joinpath(*module.split("."))
    module_file = candidate.with_suffix(".py")
    if module_file.is_file():
        return physical_path_v96_v2(module_file)
    package_file = candidate / "__init__.py"
    if package_file.is_file():
        return physical_path_v96_v2(package_file)
    return None


def _module_identity_v96_v2(path: Path) -> tuple[str, bool]:
    relative = physical_path_v96_v2(path).relative_to(_SOURCE_ROOT)
    parts = list(relative.parts)
    if parts[-1] == "__init__.py":
        return ".".join(parts[:-1]), True
    parts[-1] = Path(parts[-1]).stem
    return ".".join(parts), False


def _package_initializers_v96_v2(module: str) -> list[Path]:
    parts = module.split(".")
    result: list[Path] = []
    for length in range(1, len(parts)):
        source = _module_source_v96_v2(".".join(parts[:length]))
        if source is not None and source.name == "__init__.py":
            result.append(source)
    return result


def _import_targets_v96_v2(path: Path) -> set[str]:
    module, is_package = _module_identity_v96_v2(path)
    package = module if is_package else module.rpartition(".")[0]
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    targets: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            targets.update(alias.name for alias in node.names)
            continue
        if not isinstance(node, ast.ImportFrom):
            continue
        base = node.module or ""
        if node.level:
            package_parts = package.split(".") if package else []
            retained = len(package_parts) - (node.level - 1)
            if retained < 0:
                continue
            base = ".".join(
                (*package_parts[:retained], *((base,) if base else ()))
            )
        if base:
            targets.add(base)
            targets.update(
                f"{base}.{alias.name}"
                for alias in node.names
                if alias.name != "*"
            )
    return {
        target for target in targets if target.startswith("semantic_3d_chat")
    }


def transitive_implementation_sources_v96_v2() -> dict[str, Path]:
    """Return the deterministic first-party AST closure for V96 evaluation.

    This covers direct, conditional, and function-local imports plus package
    initializers.  The one dynamic trainer import is an explicit root.  It
    deliberately excludes unrelated robot, MCP, report, and UI modules.
    """

    pending: list[Path] = []
    for module in _IMPLEMENTATION_ROOT_MODULES_V2:
        source = _module_source_v96_v2(module)
        if source is None:
            raise FileNotFoundError(f"V96 v2 implementation module is absent: {module}")
        pending.append(source)
    observed: set[Path] = set()
    while pending:
        source = physical_path_v96_v2(pending.pop())
        if source in observed:
            continue
        observed.add(source)
        module, _is_package = _module_identity_v96_v2(source)
        pending.extend(_package_initializers_v96_v2(module))
        for target in _import_targets_v96_v2(source):
            dependency = _module_source_v96_v2(target)
            if dependency is not None and dependency not in observed:
                pending.append(dependency)
    forbidden_parts = {"robot", "mcp_server"}
    if any(forbidden_parts.intersection(path.relative_to(_SOURCE_ROOT).parts) for path in observed):
        raise ValueError("V96 v2 evaluator closure reached an unrelated runtime package")
    return {
        f"source:{path.relative_to(PROJECT_ROOT).as_posix()}": path
        for path in sorted(observed)
    }


IMPLEMENTATION_SOURCES_V2: Final[dict[str, Path]] = (
    {
        **transitive_implementation_sources_v96_v2(),
        "static:runtime_config": PROJECT_ROOT
        / "configs/runtime/gemma4_v85_strict_multiscene.yaml",
        "static:v95_config": V95_CONFIG,
        "static:v85_runtime_metadata": V85_RUNTIME_METADATA,
    }
)
_HEX64: Final[re.Pattern[str]] = re.compile(r"[0-9a-f]{64}")

_P = ParamSpec("_P")
_R = TypeVar("_R")


def _sha256(path: Path) -> str:
    raw = Path(path).expanduser()
    candidate = Path(os.path.abspath(raw if raw.is_absolute() else PROJECT_ROOT / raw))
    if any(component.is_symlink() for component in (candidate, *candidate.parents)):
        raise FileNotFoundError(f"V96 v2 source path contains a symlink: {candidate}")
    if not candidate.is_file():
        raise FileNotFoundError(candidate)
    digest = hashlib.sha256()
    with candidate.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _relative(path: Path) -> str:
    try:
        return path.resolve().relative_to(PROJECT_ROOT).as_posix()
    except ValueError as error:
        raise ValueError(f"V96 v2 source escapes project: {path}") from error


def implementation_source_inventory_v96_v2(
    sources: Mapping[str, Path] = IMPLEMENTATION_SOURCES_V2,
) -> dict[str, dict[str, str]]:
    if set(sources) != set(IMPLEMENTATION_SOURCES_V2):
        raise ValueError("V96 v2 implementation source inventory changed")
    payload = {
        name: {"path": _relative(Path(path)), "sha256": _sha256(Path(path))}
        for name, path in sorted(sources.items())
    }
    return payload


def authenticate_runtime_config_input_v96_v2(
    config: Mapping[str, Any],
) -> dict[str, str]:
    sources = config.get("sources")
    if not isinstance(sources, Mapping):
        raise TypeError("V96 v2 config source bindings are missing")
    path = physical_path_v96_v2(str(sources.get("runtime_config", "")))
    observed = _sha256(path)
    expected = sources.get("runtime_config_sha256")
    if not isinstance(expected, str) or _HEX64.fullmatch(expected) is None:
        raise ValueError("V96 v2 runtime-config SHA binding is malformed")
    if observed != expected:
        raise ValueError("V96 v2 runtime config changed after preregistration")
    return {"path": _relative(path), "sha256": observed}


def _validate_frozen_bank_expected_states_v96_v2(
    value: object,
) -> dict[str, str]:
    """Validate the independently sealed nine-bank optimization parent."""

    expected_names = (*V94_BANKS, V95_BANK_NAME)
    if (
        not isinstance(value, Mapping)
        or set(value) != set(expected_names)
        or len(value) != 9
        or any(_HEX64.fullmatch(str(value.get(name))) is None for name in expected_names)
    ):
        raise ValueError("V96 v2 frozen LoRA-bank state inventory changed")
    return {name: str(value[name]) for name in expected_names}


def build_frozen_bank_expected_states_v96_v2(
    config: Mapping[str, Any],
) -> dict[str, str]:
    """Derive the nine frozen state hashes only from sealed config contracts.

    The seven V85 hashes come from metadata whose exact file bytes are pinned
    by the V95 config.  The trained V94 and V95 hashes come from their sealed
    successor configs.  Unpinned mutable checkpoint metadata is never trusted.
    """

    sources = config.get("sources")
    if not isinstance(sources, Mapping):
        raise TypeError("V96 v2 frozen-stack source bindings are missing")
    v95_path = physical_path_v96_v2(str(sources.get("frozen_v95_config", "")))
    if v95_path != physical_path_v96_v2(V95_CONFIG):
        raise ValueError("V96 v2 frozen V95 config path changed")
    expected_v95_config_sha256 = sources.get("frozen_v95_config_sha256")
    if (
        not isinstance(expected_v95_config_sha256, str)
        or _HEX64.fullmatch(expected_v95_config_sha256) is None
        or _sha256(v95_path) != expected_v95_config_sha256
    ):
        raise ValueError("V96 v2 frozen V95 config bytes changed")

    v95 = load_config_v95(v95_path, allow_draft=False)
    v95_sources = v95.get("sources")
    if not isinstance(v95_sources, Mapping):
        raise TypeError("V96 v2 frozen V95 source bindings are missing")
    v85_metadata_path = physical_path_v96_v2(
        Path(str(v95_sources.get("frozen_v85_checkpoint", "")))
        / "runtime_metadata.json"
    )
    if (
        v85_metadata_path != physical_path_v96_v2(V85_RUNTIME_METADATA)
        or _sha256(v85_metadata_path)
        != v95_sources.get("frozen_v85_metadata_sha256")
    ):
        raise ValueError("V96 v2 pinned V85 runtime metadata bytes changed")
    v85_metadata = read_json_strict_v96_v2(v85_metadata_path)
    v85_lora = v85_metadata.get("lora")
    v85_states = v85_metadata.get("lora_bank_state_sha256")
    v94_state = v95.get("frozen_stack", {}).get("v94_bank_state_sha256")
    v95_state = config.get("frozen_stack", {}).get("v95_bank_state_sha256")
    expected_v85_names = V94_BANKS[:-1]
    if (
        not isinstance(v85_states, Mapping)
        or set(v85_states) != set(expected_v85_names)
        or not isinstance(v85_lora, Mapping)
        or v85_lora.get("adapter_parameter_count") != 565_248
        or v85_lora.get("trainable_adapter_parameter_count") != 0
        or tuple(row.get("name") for row in v85_lora.get("banks", ()))
        != expected_v85_names
        or v95.get("frozen_stack", {}).get("v94_bank_name") != V94_BANKS[-1]
        or v95.get("bridge", {}).get("bank_name") != V95_BANK_NAME
        or config.get("frozen_stack", {}).get("v95_bank_name") != V95_BANK_NAME
    ):
        raise ValueError("V96 v2 frozen LoRA-bank topology changed")
    return _validate_frozen_bank_expected_states_v96_v2(
        {
            **{name: v85_states[name] for name in expected_v85_names},
            V94_BANKS[-1]: v94_state,
            V95_BANK_NAME: v95_state,
        }
    )


def _validate_lora_bank_topology_v96_v2(value: object) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError("V96 v2 LoRA-bank topology is missing")
    rows = value.get("banks")
    expected_names = (*V94_BANKS, V95_BANK_NAME, V96_BANK_NAME)
    if (
        set(value) != {"schema_version", "enabled", "banks"}
        or value.get("schema_version") != 2
        or value.get("enabled") is not True
        or not isinstance(rows, list)
        or len(rows) != 10
        or tuple(row.get("name") for row in rows if isinstance(row, Mapping))
        != expected_names
        or any(
            not isinstance(row, Mapping)
            or set(row)
            != {
                "name",
                "trainable",
                "rank",
                "alpha",
                "dropout",
                "target_modules",
                "initialization_algorithm",
                "initialization_seed",
                "expected_initial_state_sha256",
            }
            for row in rows
        )
        or [bool(row["trainable"]) for row in rows] != [False] * 9 + [True]
        or any(
            not isinstance(row.get("target_modules"), list)
            or not row["target_modules"]
            or any(not isinstance(target, str) or not target for target in row["target_modules"])
            for row in rows
        )
    ):
        raise ValueError("V96 v2 LoRA-bank topology changed")
    targets = [target for row in rows for target in row["target_modules"]]
    if len(targets) != len(set(targets)):
        raise ValueError("V96 v2 LoRA-bank targets are no longer disjoint")
    return dict(value)


def build_lora_bank_topology_v96_v2(
    config: Mapping[str, Any],
) -> dict[str, Any]:
    """Build the exact model-free ten-bank settings consumed by inference."""

    build_frozen_bank_expected_states_v96_v2(config)
    runtime_binding = authenticate_runtime_config_input_v96_v2(config)
    runtime = load_runtime_config(
        physical_path_v96_v2(PROJECT_ROOT / runtime_binding["path"])
    )
    return _validate_lora_bank_topology_v96_v2(
        combined_lora_settings_v96(runtime, config).contract()
    )


def _model_cache_root_v96_v2(model_id: str) -> Path:
    configured_cache = os.environ.get("HF_HUB_CACHE")
    if configured_cache:
        hub_root = Path(configured_cache).expanduser()
    else:
        hf_home = Path(
            os.environ.get("HF_HOME", Path.home() / ".cache/huggingface")
        ).expanduser()
        hub_root = hf_home / "hub"
    return physical_path_v96_v2(
        Path(os.path.abspath(hub_root))
        / f"models--{model_id.replace('/', '--')}"
    )


def _validate_model_snapshot_binding_v96_v2(
    value: object, *, config: Mapping[str, Any]
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError("V96 v2 base-model snapshot binding is missing")
    sources = config.get("sources")
    if not isinstance(sources, Mapping):
        raise TypeError("V96 v2 model source bindings are missing")
    model_id = str(sources.get("model_id", ""))
    revision = str(sources.get("model_revision", ""))
    model_root = _model_cache_root_v96_v2(model_id)
    snapshot_root = model_root / "snapshots" / revision
    entries = value.get("logical_entries")
    if not isinstance(entries, Mapping):
        raise TypeError("V96 v2 base-model snapshot entries are missing")
    expected_fields = {
        "model_id",
        "revision",
        "model_cache_root",
        "snapshot_root",
        "logical_entry_count",
        "logical_entries",
        "inventory_sha256",
        "model_blob_sha256_identity",
        "weights_hashed_not_model_loaded",
    }
    if (
        set(value) != expected_fields
        or value.get("model_id") != model_id
        or value.get("revision") != revision
        or value.get("model_cache_root") != str(model_root)
        or value.get("snapshot_root") != str(snapshot_root)
        or value.get("logical_entry_count") != len(entries)
        or not entries
        or value.get("inventory_sha256") != _canonical(entries)
        or value.get("model_blob_sha256_identity")
        != sources.get("model_blob_sha256_identity")
        or value.get("weights_hashed_not_model_loaded") is not True
        or "model.safetensors" not in entries
    ):
        raise ValueError("V96 v2 base-model snapshot binding changed")
    for logical_name, raw_entry in entries.items():
        if (
            not isinstance(logical_name, str)
            or not logical_name
            or Path(logical_name).name != logical_name
            or not isinstance(raw_entry, Mapping)
            or set(raw_entry) != {"physical_target", "size_bytes", "sha256"}
            or not isinstance(raw_entry.get("physical_target"), str)
            or not Path(str(raw_entry.get("physical_target"))).is_absolute()
            or not isinstance(raw_entry.get("size_bytes"), int)
            or isinstance(raw_entry.get("size_bytes"), bool)
            or int(raw_entry.get("size_bytes", -1)) < 0
            or _HEX64.fullmatch(str(raw_entry.get("sha256"))) is None
        ):
            raise ValueError("V96 v2 base-model snapshot entry changed")
        target = Path(str(raw_entry["physical_target"]))
        if str(Path(os.path.abspath(target))) != str(target):
            raise ValueError("V96 v2 base-model target is not normalized")
        try:
            target.relative_to(model_root)
        except ValueError as error:
            raise ValueError("V96 v2 base-model target escaped its model cache") from error
    model_entry = entries["model.safetensors"]
    if model_entry.get("sha256") != sources.get("model_blob_sha256_identity"):
        raise ValueError("V96 v2 base-model weight identity changed")
    return dict(value)


def build_model_snapshot_binding_v96_v2(
    config: Mapping[str, Any], *, audit: FileAccessAudit | None = None
) -> dict[str, Any]:
    """Hash every logical file in the exact local HF revision without loading it."""

    sources = config.get("sources")
    if not isinstance(sources, Mapping):
        raise TypeError("V96 v2 model source bindings are missing")
    model_id = str(sources.get("model_id", ""))
    revision = str(sources.get("model_revision", ""))
    model_root = _model_cache_root_v96_v2(model_id)
    snapshot_root = physical_path_v96_v2(model_root / "snapshots" / revision)
    if not model_root.is_dir() or not snapshot_root.is_dir():
        raise FileNotFoundError("V96 v2 pinned local model snapshot is absent")
    logical_paths = sorted(snapshot_root.iterdir(), key=lambda path: path.name)
    if not logical_paths or any(path.is_dir() for path in logical_paths):
        raise ValueError("V96 v2 model snapshot must contain only logical files")
    entries: dict[str, dict[str, Any]] = {}
    for logical_path in logical_paths:
        try:
            target = logical_path.resolve(strict=True)
        except (FileNotFoundError, OSError) as error:
            raise FileNotFoundError(
                f"V96 v2 model snapshot entry is broken: {logical_path}"
            ) from error
        target = physical_path_v96_v2(target)
        try:
            target.relative_to(model_root)
        except ValueError as error:
            raise ValueError(
                "V96 v2 model snapshot target escaped its model cache"
            ) from error
        if not target.is_file():
            raise FileNotFoundError(target)
        if audit is not None:
            audit.record(target)
        entries[logical_path.name] = {
            "physical_target": str(target),
            "size_bytes": target.stat().st_size,
            "sha256": _sha256(target),
        }
    value: dict[str, Any] = {
        "model_id": model_id,
        "revision": revision,
        "model_cache_root": str(model_root),
        "snapshot_root": str(snapshot_root),
        "logical_entry_count": len(entries),
        "logical_entries": entries,
        "inventory_sha256": _canonical(entries),
        "model_blob_sha256_identity": sources.get("model_blob_sha256_identity"),
        "weights_hashed_not_model_loaded": True,
    }
    return _validate_model_snapshot_binding_v96_v2(value, config=config)


def authenticate_model_snapshot_v96_v2(
    config: Mapping[str, Any],
    *,
    expected: Mapping[str, Any],
    audit: FileAccessAudit | None = None,
) -> dict[str, Any]:
    sealed = _validate_model_snapshot_binding_v96_v2(expected, config=config)
    current = build_model_snapshot_binding_v96_v2(config, audit=audit)
    if current != sealed:
        raise ValueError("V96 v2 local base-model snapshot changed")
    return current


def _known_outputs() -> tuple[Path, ...]:
    paths = evaluation_paths_v96({})
    return tuple(paths.__dict__.values())


def _candidate_forbidden_roots(config: Mapping[str, Any]) -> list[Path]:
    sources = config.get("sources")
    if not isinstance(sources, Mapping):
        raise TypeError("V96 v2 candidate source bindings are missing")
    active_model_cache = _model_cache_root_v96_v2(str(sources.get("model_id", "")))
    return [
        physical_path_v96_v2(config["known_development_gate"]["labels_path"]),
        physical_path_v96_v2(
            "reports/gemma4/questions/v56_fresh_development_validation.json"
        ),
        active_model_cache,
        Path.home() / ".cache/huggingface/hub/models--google--gemma-4-E2B-it",
        Path.home()
        / ".cache/huggingface/transformers/models--google--gemma-4-E2B-it",
        PROJECT_ROOT / "data/oracle",
        *PROJECT_ROOT.glob("data*/oracle"),
    ]


def _candidate_required_paths(
    config: Mapping[str, Any], *, config_path: str | Path
) -> set[str]:
    root = physical_path_v96_v2(config["outputs"]["fixed_final_candidate"])
    return {
        str(physical_path_v96_v2(config_path)),
        str(physical_path_v96_v2(config["sources"]["training_qa"])),
        str(physical_path_v96_v2(root / "bridge.safetensors")),
        str(physical_path_v96_v2(root / "runtime_metadata.json")),
        *(
            str(physical_path_v96_v2(config["outputs"][key]))
            for key in (
                "training_report",
                "preregistration",
                "cpu_preflight",
                "topology_smoke",
            )
        ),
    }


def _validate_candidate_auth_access(
    value: object,
    *,
    forbidden_roots: Sequence[Path],
    required_paths: set[str],
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError("V96 v2 candidate-auth access proof is missing")
    expected_fields = {
        "unique_paths",
        "unique_path_count",
        "unique_path_inventory_sha256",
        "protected_read_count",
        "known_development_questions_opened",
        "known_development_labels_opened",
        "oracle_opened",
        "model_loaded",
    }
    paths = value.get("unique_paths")
    if (
        set(value) != expected_fields
        or not isinstance(paths, list)
        or not paths
        or any(not isinstance(path, str) or not Path(path).is_absolute() for path in paths)
        or paths != sorted(set(paths))
        or any(str(Path(path).resolve()) != path for path in paths)
        or value.get("unique_path_count") != len(paths)
        or value.get("unique_path_inventory_sha256") != _canonical(paths)
        or value.get("protected_read_count") != 0
        or not required_paths <= set(paths or [])
        or any(
            value.get(field) is not False
            for field in (
                "known_development_questions_opened",
                "known_development_labels_opened",
                "oracle_opened",
                "model_loaded",
            )
        )
    ):
        raise ValueError("V96 v2 candidate-auth access proof changed")
    normalized_forbidden = [Path(path).resolve() for path in forbidden_roots]
    for raw_path in paths:
        path = Path(raw_path)
        if "oracle" in {component.casefold() for component in path.parts}:
            raise ValueError("V96 v2 candidate-auth access reached an oracle path")
        for forbidden in normalized_forbidden:
            try:
                path.relative_to(forbidden)
            except ValueError:
                continue
            raise ValueError("V96 v2 candidate-auth access reached a forbidden root")
    return dict(value)


def build_evaluation_implementation_seal_v96_v2(
    *,
    sources: Mapping[str, Path] = IMPLEMENTATION_SOURCES_V2,
    config_path: str | Path = CONFIG,
) -> dict[str, Any]:
    outputs = (*_known_outputs(), CANDIDATE_ATTESTATION)
    present = [_relative(path) for path in outputs if path.exists() or path.is_symlink()]
    if present:
        raise RuntimeError(
            f"V96 v2 seal must precede attestation/evaluation outputs: {present}"
        )
    physical_config = physical_path_v96_v2(config_path)
    physical_v1_seal = physical_path_v96_v2(V1_IMPLEMENTATION_SEAL)
    v1 = authenticate_evaluation_implementation_v96(
        seal_path=physical_v1_seal, config_path=physical_config
    )
    config = load_config_v96(physical_config, allow_draft=False)
    runtime_config_binding = authenticate_runtime_config_input_v96_v2(config)
    # Fail on symlinked candidate/aggregate ancestors before historical auth.
    candidate_root = physical_path_v96_v2(config["outputs"]["fixed_final_candidate"])
    physical_path_v96_v2(candidate_root / "bridge.safetensors")
    physical_path_v96_v2(candidate_root / "runtime_metadata.json")
    for key in ("training_report", "preregistration", "cpu_preflight", "topology_smoke"):
        physical_path_v96_v2(config["outputs"][key])
    forbidden_roots = _candidate_forbidden_roots(config)
    candidate_audit = FileAccessAudit(
        forbidden_roots,
        forbidden_component_names={"oracle"},
        block_forbidden=True,
    )
    with candidate_audit:
        candidate = authenticate_full_chain_v1(config, config_path=physical_config)
    candidate_audit.assert_clean()
    candidate_access = {
        "unique_paths": candidate_audit.unique_paths,
        "unique_path_count": len(candidate_audit.unique_paths),
        "unique_path_inventory_sha256": _canonical(candidate_audit.unique_paths),
        "protected_read_count": 0,
        "known_development_questions_opened": False,
        "known_development_labels_opened": False,
        "oracle_opened": False,
        "model_loaded": False,
    }
    _validate_candidate_auth_access(
        candidate_access,
        forbidden_roots=forbidden_roots,
        required_paths=_candidate_required_paths(
            config, config_path=physical_config
        ),
    )
    model_snapshot_binding = build_model_snapshot_binding_v96_v2(config)
    frozen_bank_expected_states = build_frozen_bank_expected_states_v96_v2(config)
    lora_bank_topology = build_lora_bank_topology_v96_v2(config)
    topology_smoke_sha256 = _sha256(
        PROJECT_ROOT / str(config["outputs"]["topology_smoke"])
    )
    implementation = implementation_source_inventory_v96_v2(sources)
    contract = contract_source_inventory_v96(physical_config)
    payload: dict[str, Any] = {
        "artifact": ARTIFACT,
        "schema_version": SCHEMA_VERSION,
        "status": "sealed_before_candidate_attestation_or_question_io",
        "source_count": len(implementation),
        "sources": implementation,
        "source_inventory_sha256": _canonical(implementation),
        "contract_source_count": len(contract),
        "contract_sources": contract,
        "contract_source_inventory_sha256": _canonical(contract),
        "runtime_config_binding": runtime_config_binding,
        "v1_implementation_seal_sha256": v1["seal_sha256"],
        "v1_source_inventory_sha256": v1["source_inventory_sha256"],
        "candidate_pins": {
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
                "topology_smoke_sha256",
                "fixed_final_optimizer_updates",
                "frozen_v95_state_sha256",
                "known_development_scored",
                "deferred_final_generated",
                "runtime_promotion_authorized",
            )
            if key in candidate
        },
        "candidate_auth_access": candidate_access,
        "model_snapshot_binding": model_snapshot_binding,
        "frozen_bank_expected_states": frozen_bank_expected_states,
        "frozen_bank_expected_state_inventory_sha256": _canonical(
            frozen_bank_expected_states
        ),
        "lora_bank_topology": lora_bank_topology,
        "lora_bank_topology_sha256": _canonical(lora_bank_topology),
        "weights_hashed_not_model_loaded": True,
        "historical_v1_attempt_failed_before_question_io": True,
        "historical_v1_output_count": 0,
        "candidate_attestation_present_before_seal": False,
        "known_development_outputs_present_before_seal": [],
        "questions_opened": False,
        "labels_opened": False,
        "model_loaded": False,
        "runtime_promotion_authorized": False,
    }
    payload["candidate_pins"]["topology_smoke_sha256"] = topology_smoke_sha256
    return payload


def seal_evaluation_implementation_v96_v2(
    *,
    seal_path: Path = IMPLEMENTATION_SEAL_V2,
    sources: Mapping[str, Path] = IMPLEMENTATION_SOURCES_V2,
    config_path: str | Path = CONFIG,
) -> dict[str, Any]:
    with exclusive_evaluation_lock_v96():
        official_seal = physical_path_v96_v2(seal_path)
        payload = build_evaluation_implementation_seal_v96_v2(
            sources=sources, config_path=config_path
        )
        official_seal.parent.mkdir(parents=True, exist_ok=True)
        physical_path_v96_v2(official_seal)
        with tempfile.TemporaryDirectory(
            prefix=".v96-v2-seal-prevalidate-", dir=official_seal.parent
        ) as temporary_directory:
            temporary = Path(temporary_directory) / official_seal.name
            write_json_create_once_v96_v2(temporary, payload)
            authenticate_evaluation_implementation_v96_v2(
                seal_path=temporary,
                sources=sources,
                config_path=config_path,
            )
        write_json_create_once_v96_v2(official_seal, payload)
        return authenticate_evaluation_implementation_v96_v2(
            seal_path=official_seal,
            sources=sources,
            config_path=config_path,
        )


def authenticate_evaluation_implementation_v96_v2(
    *,
    seal_path: Path = IMPLEMENTATION_SEAL_V2,
    sources: Mapping[str, Path] = IMPLEMENTATION_SOURCES_V2,
    config_path: str | Path = CONFIG,
) -> dict[str, Any]:
    physical_config = physical_path_v96_v2(config_path)
    physical_v1_seal = physical_path_v96_v2(V1_IMPLEMENTATION_SEAL)
    v1 = authenticate_evaluation_implementation_v96(
        seal_path=physical_v1_seal, config_path=physical_config
    )
    payload = read_json_strict_v96_v2(seal_path)
    implementation = implementation_source_inventory_v96_v2(sources)
    contract = contract_source_inventory_v96(physical_config)
    config = load_config_v96(physical_config, allow_draft=False)
    runtime_config_binding = authenticate_runtime_config_input_v96_v2(config)
    expected_fields = {
        "artifact",
        "schema_version",
        "status",
        "source_count",
        "sources",
        "source_inventory_sha256",
        "contract_source_count",
        "contract_sources",
        "contract_source_inventory_sha256",
        "runtime_config_binding",
        "v1_implementation_seal_sha256",
        "v1_source_inventory_sha256",
        "candidate_pins",
        "candidate_auth_access",
        "model_snapshot_binding",
        "frozen_bank_expected_states",
        "frozen_bank_expected_state_inventory_sha256",
        "lora_bank_topology",
        "lora_bank_topology_sha256",
        "weights_hashed_not_model_loaded",
        "historical_v1_attempt_failed_before_question_io",
        "historical_v1_output_count",
        "candidate_attestation_present_before_seal",
        "known_development_outputs_present_before_seal",
        "questions_opened",
        "labels_opened",
        "model_loaded",
        "runtime_promotion_authorized",
    }
    if (
        set(payload) != expected_fields
        or payload.get("artifact") != ARTIFACT
        or payload.get("schema_version") != SCHEMA_VERSION
        or payload.get("status")
        != "sealed_before_candidate_attestation_or_question_io"
        or payload.get("source_count") != len(implementation)
        or payload.get("sources") != implementation
        or payload.get("source_inventory_sha256") != _canonical(implementation)
        or payload.get("contract_source_count") != len(contract)
        or payload.get("contract_sources") != contract
        or payload.get("contract_source_inventory_sha256") != _canonical(contract)
        or payload.get("runtime_config_binding") != runtime_config_binding
        or payload.get("v1_implementation_seal_sha256") != v1["seal_sha256"]
        or payload.get("v1_source_inventory_sha256") != v1["source_inventory_sha256"]
        or payload.get("frozen_bank_expected_state_inventory_sha256")
        != _canonical(
            _validate_frozen_bank_expected_states_v96_v2(
                payload.get("frozen_bank_expected_states")
            )
        )
        or payload.get("lora_bank_topology_sha256")
        != _canonical(
            _validate_lora_bank_topology_v96_v2(
                payload.get("lora_bank_topology")
            )
        )
        or not isinstance(payload.get("candidate_pins"), Mapping)
        or set(payload["candidate_pins"])
        != {
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
            "fixed_final_optimizer_updates",
            "frozen_v95_state_sha256",
            "known_development_scored",
            "deferred_final_generated",
            "runtime_promotion_authorized",
        }
        or any(
            payload["candidate_pins"].get(field) is not False
            for field in (
                "known_development_scored",
                "deferred_final_generated",
                "runtime_promotion_authorized",
            )
        )
        or payload["candidate_pins"].get("fixed_final_optimizer_updates") != 285
        or payload["candidate_pins"].get("config_sha256")
        != contract.get("config", {}).get("sha256")
        or any(
            _HEX64.fullmatch(str(payload["candidate_pins"].get(field))) is None
            for field in (
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
        )
        or payload.get("historical_v1_attempt_failed_before_question_io") is not True
        or payload.get("weights_hashed_not_model_loaded") is not True
        or payload.get("historical_v1_output_count") != 0
        or payload.get("candidate_attestation_present_before_seal") is not False
        or payload.get("known_development_outputs_present_before_seal") != []
        or any(
            payload.get(field) is not False
            for field in (
                "questions_opened",
                "labels_opened",
                "model_loaded",
                "runtime_promotion_authorized",
            )
        )
    ):
        raise ValueError("V96 v2 known-development implementation seal changed")
    candidate_access = _validate_candidate_auth_access(
        payload.get("candidate_auth_access"),
        forbidden_roots=_candidate_forbidden_roots(
            config
        ),
        required_paths=_candidate_required_paths(
            config,
            config_path=physical_config,
        ),
    )
    model_snapshot_binding = _validate_model_snapshot_binding_v96_v2(
        payload.get("model_snapshot_binding"), config=config
    )
    frozen_bank_expected_states = build_frozen_bank_expected_states_v96_v2(config)
    if payload.get("frozen_bank_expected_states") != frozen_bank_expected_states:
        raise ValueError("V96 v2 frozen LoRA-bank state inventory changed")
    lora_bank_topology = build_lora_bank_topology_v96_v2(config)
    if payload.get("lora_bank_topology") != lora_bank_topology:
        raise ValueError("V96 v2 LoRA-bank topology changed")
    return {
        "artifact": ARTIFACT,
        "authenticated": True,
        "seal_sha256": _sha256(seal_path),
        "source_inventory_sha256": payload["source_inventory_sha256"],
        "contract_source_inventory_sha256": payload[
            "contract_source_inventory_sha256"
        ],
        "runtime_config_binding": dict(payload["runtime_config_binding"]),
        "v1_implementation_seal_sha256": payload["v1_implementation_seal_sha256"],
        "candidate_pins": dict(payload["candidate_pins"]),
        "candidate_auth_access": candidate_access,
        "model_snapshot_binding": model_snapshot_binding,
        "frozen_bank_expected_states": frozen_bank_expected_states,
        "frozen_bank_expected_state_inventory_sha256": _canonical(
            frozen_bank_expected_states
        ),
        "lora_bank_topology": lora_bank_topology,
        "lora_bank_topology_sha256": _canonical(lora_bank_topology),
        "mandatory_model_paths": sorted(
            str(entry["physical_target"])
            for entry in model_snapshot_binding["logical_entries"].values()
        ),
        "weights_hashed_not_model_loaded": True,
        "mandatory_source_paths": sorted(
            {
                str(physical_path_v96_v2(V1_IMPLEMENTATION_SEAL)),
                *(
                    str(physical_path_v96_v2(PROJECT_ROOT / str(row["path"])))
                    for row in (
                        *payload["sources"].values(),
                        *payload["contract_sources"].values(),
                    )
                ),
            }
        ),
        "source_count": len(implementation),
        "questions_opened": False,
        "labels_opened": False,
        "model_loaded": False,
    }


def hardened_evaluation_stage_v96_v2(
    function: Callable[_P, _R],
) -> Callable[_P, _R]:
    @functools.wraps(function)
    def guarded(*args: _P.args, **kwargs: _P.kwargs) -> _R:
        with exclusive_evaluation_lock_v96():
            authenticate_evaluation_implementation_v96_v2()
            return function(*args, **kwargs)

    return guarded


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("seal", "authenticate"), nargs="?", default="authenticate")
    parser.add_argument("--config", default=str(CONFIG))
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    with exclusive_evaluation_lock_v96():
        result = (
            seal_evaluation_implementation_v96_v2(config_path=args.config)
            if args.command == "seal"
            else authenticate_evaluation_implementation_v96_v2(config_path=args.config)
        )
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "ARTIFACT",
    "CANDIDATE_ATTESTATION",
    "IMPLEMENTATION_SEAL_V2",
    "IMPLEMENTATION_SOURCES_V2",
    "authenticate_evaluation_implementation_v96_v2",
    "authenticate_model_snapshot_v96_v2",
    "authenticate_runtime_config_input_v96_v2",
    "build_evaluation_implementation_seal_v96_v2",
    "build_model_snapshot_binding_v96_v2",
    "hardened_evaluation_stage_v96_v2",
    "implementation_source_inventory_v96_v2",
    "main",
    "seal_evaluation_implementation_v96_v2",
    "transitive_implementation_sources_v96_v2",
]
