"""Fail-closed, resumable orchestration for the one-shot held-out evaluation.

This controller is the only supported transition from development model
selection to materializing deferred scenes 25--30.  It writes an immutable
launch seal *before* invoking Blender and records a content-addressed receipt
after every stage.  A restart revalidates the launch identity and every prior
receipt; it never silently mixes configs, checkpoints, reports, or partial
artifacts.

The command deliberately has no option to choose arbitrary final scenes.  The
six opaque IDs must be the exact deferred ``test`` split in the dataset config.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
from collections import Counter, defaultdict
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import yaml

from semantic_3d_chat.chat.model_snapshot import local_model_snapshot_identity
from semantic_3d_chat.chat.promotion import (
    _checkpoint_files,
    _final_attestation,
    _json_object,
    _reject_symlink_components,
    _selector_attestation,
    _unresolved_rooted,
    create_chat_promotion,
    create_held_out_final_evidence,
    resolve_primary_pointer,
    sha256_file,
    validate_chat_promotion,
    write_primary_pointer,
)
from semantic_3d_chat.chat.runtime_config import (
    load_runtime_config,
    runtime_config_file_sha256,
)
from semantic_3d_chat.config import PROJECT_ROOT, artifact_root, load_config, reports_root
from semantic_3d_chat.data.scene_variants import batch_scene_plans, batch_scene_splits
from semantic_3d_chat.evaluation.question_manifest import load_question_manifest
from semantic_3d_chat.training.checkpointing import validate_runtime_checkpoint_metadata
from semantic_3d_chat.vision.model_registry import get_model_spec

FINAL_SCENE_IDS: Final[tuple[str, ...]] = tuple(
    f"scene_{index:06d}" for index in range(25, 31)
)
LAUNCH_SCHEMA_VERSION: Final[int] = 2
RECEIPT_SCHEMA_VERSION: Final[int] = 1
COMPLETION_SCHEMA_VERSION: Final[int] = 1
STAGES: Final[tuple[str, ...]] = (
    "generate",
    "render",
    "features",
    "maps",
    "qa",
    "questions",
    "primary_predictions",
    "empty_prefix_predictions",
    "score",
    "final_evidence",
    "leakage",
    "promotion",
)
_STATIC_IMPLEMENTATION_INPUTS: Final[tuple[str, ...]] = (
    "pyproject.toml",
    "uv.lock",
    "requirements-gemma4-probe.txt",
)


def _canonical_sha256(value: object) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _rooted(path: str | Path) -> Path:
    candidate = Path(path).expanduser()
    return candidate.resolve() if candidate.is_absolute() else (PROJECT_ROOT / candidate).resolve()


def _relative_or_absolute(path: Path) -> str:
    try:
        return path.resolve().relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return str(path.resolve())


def _atomic_create_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Atomically create a JSON file without ever replacing an existing seal."""

    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError:
            raise FileExistsError(f"Refusing to overwrite immutable artifact: {path}") from None
    finally:
        temporary.unlink(missing_ok=True)


def _config_dependencies(path: str | Path) -> list[dict[str, Any]]:
    """Hash a YAML config and its complete ``_base_`` inheritance chain."""

    result: list[dict[str, Any]] = []
    seen: set[Path] = set()

    def visit(candidate: Path) -> None:
        unresolved = _unresolved_rooted(candidate)
        _reject_symlink_components(unresolved, "Final-once config dependency")
        resolved = unresolved.resolve()
        if resolved in seen:
            raise ValueError(f"Config dependency is cyclic or repeated: {resolved}")
        seen.add(resolved)
        payload = yaml.safe_load(resolved.read_text(encoding="utf-8"))
        if not isinstance(payload, Mapping):
            raise TypeError(f"Configuration must be a mapping: {resolved}")
        result.append(
            {
                "path": _relative_or_absolute(resolved),
                "sha256": sha256_file(resolved),
            }
        )
        base = payload.get("_base_")
        if base is not None:
            visit(resolved.parent / str(base))

    visit(_rooted(path))
    return result


def _implementation_identity() -> list[dict[str, Any]]:
    """Bind every project Python source that can influence the final result."""

    candidates = {
        *(PROJECT_ROOT / relative for relative in _STATIC_IMPLEMENTATION_INPUTS),
        *(PROJECT_ROOT / "src/semantic_3d_chat").rglob("*.py"),
        *(PROJECT_ROOT / "scripts").glob("*.py"),
        *(PROJECT_ROOT / "blender").glob("*.py"),
    }
    missing = [path for path in candidates if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Final-once implementation inputs are missing: {missing}")
    return [
        {
            "path": path.relative_to(PROJECT_ROOT).as_posix(),
            "sha256": sha256_file(path),
        }
        for path in sorted(candidates)
    ]


def _runtime_contract_parameter_counts(
    metadata: Mapping[str, Any], field: str
) -> dict[str, int]:
    value = metadata.get(field, {})
    if not isinstance(value, Mapping):
        raise TypeError(f"Checkpoint runtime metadata has invalid {field}")
    result: dict[str, int] = {}
    for key, count_or_modules in value.items():
        if not isinstance(key, str):
            raise TypeError(f"Checkpoint runtime metadata has invalid {field}")
        if isinstance(count_or_modules, Mapping):
            module_counts = list(count_or_modules.values())
            if any(
                isinstance(count, bool)
                or not isinstance(count, int)
                or count < 1
                for count in module_counts
            ):
                raise ValueError(f"Checkpoint runtime metadata has invalid {field}")
            result[key] = sum(int(count) for count in module_counts)
        elif (
            isinstance(count_or_modules, bool)
            or not isinstance(count_or_modules, int)
            or count_or_modules < 1
        ):
            raise ValueError(f"Checkpoint runtime metadata has invalid {field}")
        else:
            result[key] = count_or_modules
    return result


def _expected_runtime_dimensions(
    dataset_config: Mapping[str, Any], runtime_config: Mapping[str, Any]
) -> tuple[int, int]:
    """Derive dimensions from pinned model/config contracts, not checkpoint claims."""

    dataset_vision = dataset_config.get("vision")
    runtime_vision = runtime_config.get("vision")
    dataset_language = dataset_config.get("language")
    runtime_language = runtime_config.get("language")
    if not all(
        isinstance(value, Mapping)
        for value in (
            dataset_vision,
            runtime_vision,
            dataset_language,
            runtime_language,
        )
    ):
        raise TypeError("Final-once requires explicit vision and language mappings")
    assert isinstance(dataset_vision, Mapping)
    assert isinstance(runtime_vision, Mapping)
    assert isinstance(dataset_language, Mapping)
    assert isinstance(runtime_language, Mapping)

    model_id = dataset_vision.get("model_id")
    revision = dataset_vision.get("revision")
    if not isinstance(model_id, str) or not isinstance(revision, str):
        raise TypeError("Dataset vision model_id/revision must be strings")
    if re.fullmatch(r"[0-9a-f]{40}", revision) is None:
        raise ValueError("Final-once vision revision must be an exact 40-hex Hub commit")
    for section_name, section in (
        ("runtime vision", runtime_vision),
        ("dataset language", dataset_language),
        ("runtime language", runtime_language),
    ):
        if section.get("model_id") != model_id or section.get("revision") != revision:
            raise ValueError(
                f"{section_name} model/revision differs from the sealed dataset vision model"
            )

    spec = get_model_spec(model_id)
    feature_mode = dataset_vision.get("feature_mode")
    expected_mode = (
        "middle_late_projected"
        if spec.architecture == "gemma4"
        else "middle_late_aligned"
    )
    if feature_mode != expected_mode:
        raise ValueError(
            f"Final-once requires {expected_mode} for {spec.architecture}; got {feature_mode}"
        )
    if spec.architecture != "gemma4":
        raise ValueError("This final-once workflow is locked to the Gemma 4 architecture")
    semantic_dim = (2 * spec.native_dim) + spec.aligned_dim
    # Gemma 4's native multimodal projector emits decoder-width tokens.  The
    # registry's aligned dimension is therefore an independent static contract
    # for the causal decoder hidden width used by prefix injection.
    language_hidden_dim = spec.aligned_dim
    return semantic_dim, language_hidden_dim


def _validate_checkpoint_runtime_contract(
    metadata: dict[str, Any], runtime_config: dict[str, Any], dataset_config: dict[str, Any]
) -> list[str]:
    """Run the full static runtime contract without constructing Gemma weights."""

    from semantic_3d_chat.chat.runtime import validate_checkpoint_contract
    from semantic_3d_chat.scene_encoder.block_cross_residual import (
        construct_block_cross_residual,
    )
    from semantic_3d_chat.scene_encoder.dense_alignment import construct_dense_alignment
    from semantic_3d_chat.scene_encoder.dense_sidecar_adapter import (
        construct_dense_sidecar_adapter,
    )

    semantic_dim, language_hidden_dim = _expected_runtime_dimensions(
        dataset_config, runtime_config
    )
    expected_dimensions = {
        "semantic_dim": semantic_dim,
        "language_hidden_dim": language_hidden_dim,
    }
    dimension_mismatches = {
        field: {"checkpoint": metadata.get(field), "required": value}
        for field, value in expected_dimensions.items()
        if metadata.get(field) != value
    }
    if dimension_mismatches:
        raise ValueError(
            "Checkpoint dimensions differ from the pinned Gemma 4 contract: "
            f"{dimension_mismatches}"
        )
    lora_counts = _runtime_contract_parameter_counts(
        metadata, "lora_bank_parameter_counts"
    )
    dense = construct_dense_alignment(runtime_config, semantic_dim=semantic_dim)
    sidecar = construct_dense_sidecar_adapter(
        runtime_config,
        scene_dim=language_hidden_dim,
        latent_count=int(runtime_config["scene_encoder"]["global_latents"]),
    )
    block_cross = construct_block_cross_residual(
        runtime_config,
        scene_dim=language_hidden_dim,
        block_dim=int(runtime_config["scene_encoder"]["model_dim"]),
        latent_count=int(runtime_config["scene_encoder"]["global_latents"]),
    )
    return validate_checkpoint_contract(
        metadata,
        runtime_config,
        semantic_dim=semantic_dim,
        language_hidden_dim=language_hidden_dim,
        lora_parameter_count=int(metadata.get("lora_parameter_count", 0)),
        lora_parameter_counts=lora_counts,
        dense_alignment_parameter_count=(0 if dense is None else dense.parameter_count),
        dense_sidecar_adapter_parameter_count=(
            0 if sidecar is None else sidecar.parameter_count
        ),
        block_cross_residual_parameter_count=(
            0 if block_cross is None else block_cross.parameter_count
        ),
    )


def _checkpoint_identity(
    checkpoint: Path,
    runtime_config: dict[str, Any],
    dataset_config: dict[str, Any],
) -> dict[str, Any]:
    adapter, runtime_metadata, _ = _checkpoint_files(checkpoint)
    runtime_payload = _json_object(runtime_metadata)
    validate_runtime_checkpoint_metadata(runtime_payload)
    warnings = _validate_checkpoint_runtime_contract(
        runtime_payload, runtime_config, dataset_config
    )
    metadata = checkpoint / "metadata.json"
    files = [adapter, runtime_metadata]
    if metadata.is_file():
        files.append(metadata)
    return {
        "path": _relative_or_absolute(checkpoint),
        "files": [
            {
                "path": path.name,
                "sha256": sha256_file(path),
                "size_bytes": path.stat().st_size,
            }
            for path in files
        ],
        "runtime_contract_validated": True,
        "runtime_contract_warnings": warnings,
    }


def _selector_identity(selector_path: Path, checkpoint: Path) -> dict[str, Any]:
    report = _json_object(selector_path)
    summary = _selector_attestation(report, checkpoint)
    return {
        "path": _relative_or_absolute(selector_path),
        "sha256": sha256_file(selector_path),
        "selected_update": summary["selected_update"],
        "chat_promotion_checks_passed": summary["checks_passed"],
    }


def _final_split_contract(config: dict[str, Any]) -> dict[str, Any]:
    plans = batch_scene_plans(config)
    splits = batch_scene_splits(config, plans)
    if splits is None:
        raise ValueError("Final-once requires explicit batch train/validation/test splits")
    parsed = {name: tuple(splits[name]) for name in ("train", "validation", "test")}
    if parsed["test"] != FINAL_SCENE_IDS:
        raise ValueError(
            "Final-once is locked to deferred scenes 25--30; configured test split is "
            f"{parsed['test']}"
        )
    if "test" not in set(config.get("batch", {}).get("deferred_splits", [])):
        raise ValueError("Final-once requires test to remain a declared deferred split")
    flattened = [scene for values in parsed.values() for scene in values]
    if len(flattened) != len(set(flattened)):
        raise ValueError("Final-once split scenes are not disjoint")
    by_id = {plan.scene_id: plan for plan in plans}
    final_plans = [by_id[scene_id] for scene_id in FINAL_SCENE_IDS]
    return {
        "splits": {name: list(values) for name, values in parsed.items()},
        "final_scene_ids": list(FINAL_SCENE_IDS),
        "final_scene_plan_sha256": _canonical_sha256(
            [
                {"scene_id": plan.scene_id, **plan.oracle_metadata()}
                for plan in final_plans
            ]
        ),
    }


def _qa_development_baseline(dataset_config: dict[str, Any]) -> dict[str, Any]:
    qa_root = artifact_root(dataset_config, "qa").resolve()
    result: dict[str, Any] = {}
    for split in ("train", "validation"):
        path = qa_root / f"{split}.jsonl"
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(f"Development QA baseline is unavailable: {path}")
        result[split] = {
            "path": _relative_or_absolute(path),
            "sha256": sha256_file(path),
            "size_bytes": path.stat().st_size,
        }
    return result


@dataclass(frozen=True)
class WorkflowPaths:
    dataset_config: Path
    runtime_config: Path
    selector_report: Path
    checkpoint: Path
    work_root: Path
    primary_pointer: Path
    coordinator_python: Path
    gemma_python: Path
    blender: str

    @property
    def launch(self) -> Path:
        return self.work_root / "launch.json"

    @property
    def receipts(self) -> Path:
        return self.work_root / "stages"

    @property
    def questions(self) -> Path:
        return self.work_root / "questions.json"

    @property
    def primary_predictions(self) -> Path:
        return self.work_root / "primary.jsonl"

    @property
    def empty_control_root(self) -> Path:
        return self.work_root / "controls"

    @property
    def qa_build_root(self) -> Path:
        return self.work_root / "supervision_build"

    @property
    def qa_build_config(self) -> Path:
        return self.qa_build_root / "config.yaml"

    @property
    def empty_predictions(self) -> Path:
        return self.empty_control_root / "empty_scene_prefix.jsonl"

    @property
    def primary_metrics(self) -> Path:
        return self.work_root / "primary_metrics.json"

    @property
    def empty_metrics(self) -> Path:
        return self.work_root / "empty_scene_prefix_metrics.json"

    @property
    def final_evidence(self) -> Path:
        return self.work_root / "held_out_final_evidence.json"

    @property
    def leakage(self) -> Path:
        return self.work_root / "leakage.json"

    @property
    def completion(self) -> Path:
        return self.work_root / "complete.json"


def _validate_workflow_path_identity(paths: WorkflowPaths) -> None:
    for field, path in (
        ("Dataset config", paths.dataset_config),
        ("Runtime config", paths.runtime_config),
        ("Selector report", paths.selector_report),
        ("Checkpoint", paths.checkpoint),
        ("Final-once work root", paths.work_root),
        ("Primary pointer", paths.primary_pointer),
    ):
        unresolved = _unresolved_rooted(path)
        if unresolved != path:
            raise ValueError(f"{field} path is not normalized absolute input: {path}")
        _reject_symlink_components(unresolved, field)


def build_launch_identity(paths: WorkflowPaths) -> dict[str, Any]:
    _validate_workflow_path_identity(paths)
    try:
        paths.primary_pointer.relative_to((PROJECT_ROOT / "configs/runtime").resolve())
    except ValueError as exc:
        raise ValueError("Primary pointer must remain below configs/runtime") from exc
    forbidden_work_components = {"oracle", "qa", "rendered", "features"}
    if forbidden_work_components.intersection(
        component.casefold() for component in paths.work_root.parts
    ):
        raise ValueError("Final-once work root is inside a runtime-forbidden data tree")
    dataset_dependencies = _config_dependencies(paths.dataset_config)
    runtime_dependencies = _config_dependencies(paths.runtime_config)
    dataset = load_config(paths.dataset_config)
    runtime = load_runtime_config(paths.runtime_config)
    dataset_maps_root = artifact_root(dataset, "maps").resolve()
    runtime_maps_root = artifact_root(runtime, "maps").resolve()
    if dataset_maps_root != runtime_maps_root:
        raise ValueError(
            "Final-once dataset and runtime configs must resolve the same maps_root: "
            f"dataset={dataset_maps_root} runtime={runtime_maps_root}"
        )
    protected_roots = [
        artifact_root(dataset, kind).resolve()
        for kind in ("oracle", "qa", "rendered", "features", "maps", "checkpoints")
    ]
    protected_roots.extend(
        [paths.checkpoint, (PROJECT_ROOT / "configs/runtime").resolve()]
    )
    if any(
        paths.work_root == protected or paths.work_root.is_relative_to(protected)
        for protected in protected_roots
    ):
        raise ValueError("Final-once work root overlaps protected runtime/data inputs")
    split_contract = _final_split_contract(dataset)
    checkpoint_identity = _checkpoint_identity(paths.checkpoint, runtime, dataset)
    selector = _selector_identity(paths.selector_report, paths.checkpoint)
    payload = {
        "schema_version": LAUNCH_SCHEMA_VERSION,
        "kind": "gemma4_held_out_final_once",
        "dataset_config": {
            "path": _relative_or_absolute(paths.dataset_config),
            "dependencies": dataset_dependencies,
            "effective_sha256": _canonical_sha256(
                {key: value for key, value in dataset.items() if not key.startswith("_")}
            ),
        },
        "runtime_config": {
            "path": _relative_or_absolute(paths.runtime_config),
            "dependencies": runtime_dependencies,
            "file_sha256": runtime_config_file_sha256(paths.runtime_config),
            "effective_sha256": _canonical_sha256(
                {key: value for key, value in runtime.items() if not key.startswith("_")}
            ),
        },
        "model_snapshot": local_model_snapshot_identity(runtime),
        "selector": selector,
        "checkpoint": checkpoint_identity,
        "split_contract": split_contract,
        "shared_maps_root": _relative_or_absolute(dataset_maps_root),
        "development_qa_baseline": _qa_development_baseline(dataset),
        "executables": {
            "coordinator_python": str(paths.coordinator_python.resolve()),
            "gemma_python": str(paths.gemma_python.resolve()),
            "blender": paths.blender,
        },
        "outputs": {
            "work_root": _relative_or_absolute(paths.work_root),
            "primary_pointer": _relative_or_absolute(paths.primary_pointer),
            "qa_build_config": _relative_or_absolute(paths.qa_build_config),
            "qa_build_config_sha256": _canonical_sha256(
                _qa_build_config_payload(paths)
            ),
        },
        "implementation_files": _implementation_identity(),
        "environmental_text_to_chat": False,
        "question_dependent_scene_selection": False,
    }
    return {**payload, "identity_sha256": _canonical_sha256(payload)}


def _assert_initial_final_footprint_clean(
    paths: WorkflowPaths, dataset_config: dict[str, Any]
) -> None:
    footprint: list[Path] = []
    for kind in ("oracle", "rendered", "features", "maps"):
        root = artifact_root(dataset_config, kind).resolve()
        footprint.extend(root / scene_id for scene_id in FINAL_SCENE_IDS)
    reports = reports_root(dataset_config).resolve()
    footprint.extend(reports / "figures" / scene_id for scene_id in FINAL_SCENE_IDS)
    footprint.extend(
        reports / "metrics" / f"map_{scene_id}.json"
        for scene_id in FINAL_SCENE_IDS
    )
    existing = [path for path in footprint if path.exists()]
    if existing:
        raise FileExistsError(
            "Deferred final-scene artifacts exist without a matching launch seal: "
            f"{existing}"
        )
    test_qa = artifact_root(dataset_config, "qa").resolve() / "test.jsonl"
    if test_qa.exists() and test_qa.read_text(encoding="utf-8").strip():
        raise FileExistsError(
            "Held-out test QA is already materialized without a matching launch seal"
        )
    promotion = paths.checkpoint / "promotion.json"
    if promotion.exists() or paths.primary_pointer.exists():
        raise FileExistsError("Promotion/pointer already exists outside final-once")
    if paths.work_root.exists() and any(paths.work_root.iterdir()):
        raise FileExistsError(
            f"Final-once work root is nonempty but has no launch seal: {paths.work_root}"
        )


def authorize_launch(paths: WorkflowPaths, *, write: bool) -> dict[str, Any]:
    """Validate the selector before any final-scene write and seal the run."""

    identity = build_launch_identity(paths)
    if paths.launch.is_file():
        existing = _json_object(paths.launch)
        if existing != identity:
            raise RuntimeError("Final-once launch identity changed; refusing unsafe resume")
        return identity
    dataset = load_config(paths.dataset_config)
    _assert_initial_final_footprint_clean(paths, dataset)
    if write:
        _atomic_create_json(paths.launch, identity)
    return identity


def _walk_output_files(roots: Sequence[Path]) -> list[Path]:
    files: list[Path] = []
    for root in roots:
        if root.is_symlink():
            raise ValueError(f"Final-once output cannot be a symbolic link: {root}")
        if root.is_file():
            files.append(root.resolve())
        elif root.is_dir():
            nested = sorted(path for path in root.rglob("*") if path.is_file())
            if any(path.is_symlink() for path in nested):
                raise ValueError(f"Final-once output tree contains a symbolic link: {root}")
            files.extend(path.resolve() for path in nested)
        else:
            raise FileNotFoundError(f"Expected final-once output is unavailable: {root}")
    unique = sorted(set(files), key=str)
    if not unique:
        raise ValueError("Final-once stage produced no output files")
    return unique


def output_inventory(roots: Sequence[Path]) -> dict[str, Any]:
    entries = [
        {
            "path": _relative_or_absolute(path),
            "sha256": sha256_file(path),
            "size_bytes": path.stat().st_size,
        }
        for path in _walk_output_files(roots)
    ]
    return {
        "file_count": len(entries),
        "total_size_bytes": sum(int(entry["size_bytes"]) for entry in entries),
        "tree_sha256": _canonical_sha256(entries),
        "files": entries,
    }


def _prediction_provenance(path: Path) -> Path:
    return path.with_name(f"{path.name}.provenance.json")


def _stage_output_roots(
    stage: str, paths: WorkflowPaths, dataset: dict[str, Any]
) -> tuple[Path, ...]:
    oracle = artifact_root(dataset, "oracle").resolve()
    rendered = artifact_root(dataset, "rendered").resolve()
    features = artifact_root(dataset, "features").resolve()
    maps = artifact_root(dataset, "maps").resolve()
    qa = artifact_root(dataset, "qa").resolve()
    reports = reports_root(dataset).resolve()
    manifest_name = str(dataset["batch"].get("manifest_name", "multiscene"))
    batch_manifest = oracle / "batches" / f"{manifest_name}.json"
    if stage == "generate":
        return (
            *(
                path
                for scene_id in FINAL_SCENE_IDS
                for path in (
                    oracle / scene_id / "oracle.json",
                    oracle / scene_id / "scene.blend",
                    rendered / scene_id / "p_000000.png",
                )
            ),
            batch_manifest,
        )
    if stage == "render":
        return tuple(
            [*(rendered / scene_id for scene_id in FINAL_SCENE_IDS)]
            + [oracle / scene_id / "visibility.json" for scene_id in FINAL_SCENE_IDS]
            + [batch_manifest]
        )
    if stage == "features":
        return tuple(features / scene_id for scene_id in FINAL_SCENE_IDS)
    if stage == "maps":
        return (
            *(maps / scene_id / "voxel_map.npz" for scene_id in FINAL_SCENE_IDS),
            *(reports / "figures" / scene_id for scene_id in FINAL_SCENE_IDS),
            *(
                reports / "metrics" / f"map_{scene_id}.json"
                for scene_id in FINAL_SCENE_IDS
            ),
        )
    if stage == "qa":
        return (
            *(
                qa / name
                for name in ("train.jsonl", "validation.jsonl", "test.jsonl", "splits.json")
            ),
            paths.qa_build_root,
        )
    if stage == "questions":
        return (paths.questions,)
    if stage == "primary_predictions":
        return (paths.primary_predictions, _prediction_provenance(paths.primary_predictions))
    if stage == "empty_prefix_predictions":
        return (
            paths.empty_predictions,
            _prediction_provenance(paths.empty_predictions),
            paths.empty_control_root / "manifest.json",
        )
    if stage == "score":
        return (paths.primary_metrics, paths.empty_metrics)
    if stage == "final_evidence":
        return (paths.final_evidence,)
    if stage == "leakage":
        return (paths.leakage,)
    if stage == "promotion":
        return (paths.checkpoint / "promotion.json", paths.primary_pointer)
    raise ValueError(f"Unknown final-once stage: {stage}")


def _receipt_path(paths: WorkflowPaths, stage: str) -> Path:
    return paths.receipts / f"{STAGES.index(stage) + 1:02d}_{stage}.json"


def _validate_receipt(
    receipt_path: Path,
    *,
    stage: str,
    launch_sha256: str,
    roots: Sequence[Path],
) -> dict[str, Any]:
    receipt = _json_object(receipt_path)
    expected_keys = {"schema_version", "stage", "launch_identity_sha256", "outputs"}
    if set(receipt) != expected_keys:
        raise ValueError(f"Stage receipt has an invalid field set: {receipt_path}")
    if (
        receipt.get("schema_version") != RECEIPT_SCHEMA_VERSION
        or receipt.get("stage") != stage
        or receipt.get("launch_identity_sha256") != launch_sha256
    ):
        raise RuntimeError(f"Stage receipt does not belong to this launch: {receipt_path}")
    observed = output_inventory(roots)
    if receipt.get("outputs") != observed:
        raise RuntimeError(f"Stage outputs changed after receipt: {stage}")
    return receipt


def _qa_paths(dataset: dict[str, Any]) -> tuple[Path, Path]:
    root = artifact_root(dataset, "qa").resolve()
    return root / "test.jsonl", root / "splits.json"


def _require_force_authorization(
    paths: WorkflowPaths,
    stage: str,
    launch: Mapping[str, Any] | None,
) -> None:
    if stage not in {"generate", "render"}:
        return
    if launch is None or not paths.launch.is_file():
        raise RuntimeError(
            f"Refusing {stage} --force without an existing final-once launch seal"
        )
    sealed = _json_object(paths.launch)
    if sealed != launch or sealed.get("identity_sha256") != launch.get(
        "identity_sha256"
    ):
        raise RuntimeError(f"Refusing {stage} --force for a different launch identity")
    if _receipt_path(paths, stage).exists():
        raise RuntimeError(f"Refusing {stage} --force after its stage receipt exists")


def build_stage_commands(
    stage: str,
    paths: WorkflowPaths,
    dataset: dict[str, Any],
    *,
    launch: Mapping[str, Any] | None = None,
) -> tuple[tuple[str, ...], ...]:
    py = str(paths.coordinator_python)
    gemma = str(paths.gemma_python)
    dataset_config = str(paths.dataset_config)
    runtime_config = str(paths.runtime_config)
    checkpoint = str(paths.checkpoint)
    qa_test, _ = _qa_paths(dataset)
    if stage in {"generate", "render"}:
        _require_force_authorization(paths, stage, launch)
        return (
            (
                py,
                "scripts/generate_scene_batch.py",
                "--config",
                dataset_config,
                "--stage",
                stage,
                "--split",
                "test",
                "--include-deferred-test",
                "--blender",
                paths.blender,
                "--force",
            ),
        )
    if stage == "features":
        return (
            (
                gemma,
                "-m",
                "semantic_3d_chat.vision.batch_encoder",
                "--config",
                dataset_config,
                "--split",
                "test",
                "--include-deferred-test",
                "--offline",
            ),
        )
    if stage == "maps":
        return tuple(
            (
                gemma,
                "scripts/build_map.py",
                "--config",
                dataset_config,
                "--scene",
                scene_id,
            )
            for scene_id in FINAL_SCENE_IDS
        )
    if stage == "qa":
        return (
            (
                py,
                "-m",
                "semantic_3d_chat.data.qa_generator",
                "--config",
                str(paths.qa_build_config),
                "--include-deferred-test",
            ),
        )
    if stage == "questions":
        return (
            (
                py,
                "-m",
                "semantic_3d_chat.evaluation.prepare_questions",
                "--config",
                runtime_config,
                "--split",
                "test",
                "--qa",
                str(qa_test),
                "--output",
                str(paths.questions),
                "--force",
            ),
        )
    if stage == "primary_predictions":
        return (
            (
                gemma,
                "-m",
                "semantic_3d_chat.evaluation.predict",
                "--config",
                runtime_config,
                "--split",
                "test",
                "--questions-manifest",
                str(paths.questions),
                "--checkpoint",
                checkpoint,
                "--output",
                str(paths.primary_predictions),
            ),
        )
    if stage == "empty_prefix_predictions":
        return (
            (
                gemma,
                "-m",
                "semantic_3d_chat.evaluation.control_predict",
                "--config",
                runtime_config,
                "--split",
                "test",
                "--questions-manifest",
                str(paths.questions),
                "--checkpoint",
                checkpoint,
                "--output-dir",
                str(paths.empty_control_root),
                "--condition",
                "empty_scene_prefix",
            ),
        )
    if stage == "score":
        return (
            (
                py,
                "-m",
                "semantic_3d_chat.evaluation.run",
                "--config",
                runtime_config,
                "--references",
                str(qa_test),
                "--predictions",
                str(paths.primary_predictions),
                "--output",
                str(paths.primary_metrics),
            ),
            (
                py,
                "-m",
                "semantic_3d_chat.evaluation.run",
                "--config",
                runtime_config,
                "--references",
                str(qa_test),
                "--predictions",
                str(paths.empty_predictions),
                "--output",
                str(paths.empty_metrics),
            ),
        )
    if stage == "leakage":
        return (
            (
                gemma,
                "-m",
                "semantic_3d_chat.evaluation.leakage",
                "--config",
                runtime_config,
                "--scene",
                FINAL_SCENE_IDS[0],
                "--checkpoint",
                checkpoint,
                "--output",
                str(paths.leakage),
            ),
        )
    return ()


def _qa_build_config_payload(paths: WorkflowPaths) -> dict[str, Any]:
    return {
        "_base_": str(paths.dataset_config),
        "paths": {"qa_root": str(paths.qa_build_root)},
    }


def _prepare_qa_build(paths: WorkflowPaths) -> None:
    """Create only the sealed isolated QA output config, never rewrite dev QA."""

    expected = _qa_build_config_payload(paths)
    if paths.qa_build_config.exists():
        if _json_object(paths.qa_build_config) != expected:
            raise RuntimeError("Isolated final QA build config changed after launch")
        return
    _atomic_create_json(paths.qa_build_config, expected)


def _atomic_replace_from(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    temporary = Path(temporary_name)
    try:
        with source.open("rb") as input_handle, os.fdopen(descriptor, "wb") as output_handle:
            shutil.copyfileobj(input_handle, output_handle)
            output_handle.flush()
            os.fsync(output_handle.fileno())
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def _publish_isolated_qa(
    paths: WorkflowPaths,
    dataset: dict[str, Any],
    launch: Mapping[str, Any],
) -> None:
    """Publish test supervision while proving development bytes stayed immutable."""

    for split in ("train", "validation"):
        built = paths.qa_build_root / f"{split}.jsonl"
        baseline = launch["development_qa_baseline"][split]
        if not built.is_file() or sha256_file(built) != baseline["sha256"]:
            raise RuntimeError(
                f"Isolated final QA generation does not reproduce development {split}"
            )
        original = _rooted(str(baseline["path"]))
        if sha256_file(original) != baseline["sha256"]:
            raise RuntimeError(f"Development {split} QA changed before final publication")
    built_test = paths.qa_build_root / "test.jsonl"
    built_splits = paths.qa_build_root / "splits.json"
    if not built_test.is_file() or not built_splits.is_file():
        raise FileNotFoundError("Isolated final QA generation did not produce test/splits")
    qa_test, qa_splits = _qa_paths(dataset)
    _atomic_replace_from(built_test, qa_test)
    _atomic_replace_from(built_splits, qa_splits)


def _validate_qa_stage(paths: WorkflowPaths, dataset: dict[str, Any], launch: Mapping[str, Any]) -> None:
    qa_test, splits_path = _qa_paths(dataset)
    for split in ("train", "validation"):
        baseline = launch["development_qa_baseline"][split]
        path = _rooted(str(baseline["path"]))
        if sha256_file(path) != baseline["sha256"]:
            raise RuntimeError(f"Final QA generation changed development {split} records")
    split_payload = _json_object(splits_path)
    if split_payload.get("splits") != launch["split_contract"]["splits"]:
        raise RuntimeError("Generated QA split manifest differs from sealed split contract")
    records = [
        json.loads(line)
        for line in qa_test.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not records or {str(record.get("scene_id")) for record in records} != set(FINAL_SCENE_IDS):
        raise RuntimeError("Held-out QA does not cover exactly all six final scenes")
    balanced = dataset.get("qa", {}).get("balanced_selection", {})
    per_scene = balanced.get("per_scene") if isinstance(balanced, Mapping) else None
    expected_per_scene = per_scene.get("test") if isinstance(per_scene, Mapping) else None
    if (
        balanced.get("enabled") is not True
        or isinstance(expected_per_scene, bool)
        or not isinstance(expected_per_scene, int)
        or expected_per_scene < 1
    ):
        raise RuntimeError("Final-once requires a positive balanced test QA count")
    counts = Counter(str(record.get("scene_id")) for record in records)
    expected_counts = {scene_id: expected_per_scene for scene_id in FINAL_SCENE_IDS}
    if dict(counts) != expected_counts:
        raise RuntimeError(
            f"Held-out QA count differs from the sealed per-scene contract: {dict(counts)}"
        )
    expected_total = expected_per_scene * len(FINAL_SCENE_IDS)
    manifest_counts = split_payload.get("question_counts")
    if (
        not isinstance(manifest_counts, Mapping)
        or manifest_counts.get("test") != expected_total
        or len(records) != expected_total
    ):
        raise RuntimeError("Held-out QA total differs from the configured final count")

    final_plans = {
        plan.scene_id: plan
        for plan in batch_scene_plans(dataset)
        if plan.scene_id in FINAL_SCENE_IDS
    }
    expected_pair_scenes: dict[str, set[str]] = defaultdict(set)
    for plan in final_plans.values():
        if plan.pair_id is None:
            raise RuntimeError("Every sealed final scene must belong to a counterfactual pair")
        expected_pair_scenes[plan.pair_id].add(plan.scene_id)
    if any(len(scene_ids) != 2 for scene_ids in expected_pair_scenes.values()):
        raise RuntimeError("Sealed final counterfactual pairs must each contain two scenes")
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        pair_id = record.get("counterfactual_pair_id")
        question_key = record.get("counterfactual_question_key")
        if pair_id not in expected_pair_scenes or not isinstance(question_key, str):
            raise RuntimeError("Held-out QA record lacks its sealed counterfactual identity")
        grouped[(str(pair_id), question_key)].append(record)
    malformed = [key for key, members in grouped.items() if len(members) != 2]
    if malformed:
        raise RuntimeError(f"Held-out QA has incomplete counterfactual units: {malformed}")
    changed_per_pair: Counter[str] = Counter()
    for (pair_id, _), members in grouped.items():
        if {str(member.get("scene_id")) for member in members} != expected_pair_scenes[pair_id]:
            raise RuntimeError("Held-out QA counterfactual unit uses the wrong scene pair")
        flags = {member.get("counterfactual_expected_change") for member in members}
        if flags not in ({True}, {False}):
            raise RuntimeError("Held-out QA counterfactual change flags disagree")
        if flags == {True}:
            changed_per_pair[pair_id] += 1
    max_changed = balanced.get("max_changed_units_per_pair")
    if (
        isinstance(max_changed, bool)
        or not isinstance(max_changed, int)
        or max_changed < 1
    ):
        raise RuntimeError("Final-once requires max_changed_units_per_pair")
    expected_changed = {
        pair_id: max_changed for pair_id in sorted(expected_pair_scenes)
    }
    if dict(changed_per_pair) != expected_changed:
        raise RuntimeError(
            "Held-out QA does not provide the configured changed-pair coverage: "
            f"{dict(changed_per_pair)}"
        )
    if not any(record.get("target_xyz") is not None for record in records):
        raise RuntimeError("Held-out QA has no grounding targets")


def _validate_questions(paths: WorkflowPaths, dataset: dict[str, Any]) -> None:
    manifest = load_question_manifest(paths.questions)
    if {record.scene_id for record in manifest.questions} != set(FINAL_SCENE_IDS):
        raise RuntimeError("Question-only manifest does not cover exactly final scenes")
    qa_test, _ = _qa_paths(dataset)
    if manifest.source_qa_sha256 != sha256_file(qa_test):
        raise RuntimeError("Question-only manifest is not bound to the final QA file")
    balanced = dataset.get("qa", {}).get("balanced_selection", {})
    per_scene = balanced.get("per_scene") if isinstance(balanced, Mapping) else None
    expected_per_scene = per_scene.get("test") if isinstance(per_scene, Mapping) else None
    if not isinstance(expected_per_scene, int) or isinstance(expected_per_scene, bool):
        raise TypeError("Question-only manifest has no configured final count")
    if manifest.question_count != expected_per_scene * len(FINAL_SCENE_IDS):
        raise RuntimeError("Question-only manifest does not contain the full final set")


def _validate_existing_final_evidence(paths: WorkflowPaths) -> None:
    config = load_runtime_config(paths.runtime_config)
    adapter, runtime_metadata, _ = _checkpoint_files(paths.checkpoint)
    _final_attestation(
        _json_object(paths.final_evidence),
        runtime_config_path=Path(str(config["_config_path"])),
        checkpoint=paths.checkpoint,
        adapter_sha256=sha256_file(adapter),
        runtime_metadata_sha256=sha256_file(runtime_metadata),
        runtime_config_sha256=runtime_config_file_sha256(paths.runtime_config),
    )


def _create_or_validate_final_evidence(paths: WorkflowPaths, dataset: dict[str, Any]) -> None:
    if paths.final_evidence.exists():
        _validate_existing_final_evidence(paths)
        return
    _, splits = _qa_paths(dataset)
    create_held_out_final_evidence(
        runtime_config_path=paths.runtime_config,
        checkpoint=paths.checkpoint,
        metrics_path=paths.primary_metrics,
        predictions_path=paths.primary_predictions,
        prediction_provenance_path=_prediction_provenance(paths.primary_predictions),
        chance_metrics_path=paths.empty_metrics,
        chance_predictions_path=paths.empty_predictions,
        chance_prediction_provenance_path=_prediction_provenance(paths.empty_predictions),
        split_manifest_path=splits,
        output_path=paths.final_evidence,
    )


def _create_or_validate_promotion(paths: WorkflowPaths) -> None:
    promotion_path = paths.checkpoint / "promotion.json"
    expected_hashes = {
        "selector_report_sha256": sha256_file(paths.selector_report),
        "final_evidence_sha256": sha256_file(paths.final_evidence),
        "leakage_report_sha256": sha256_file(paths.leakage),
    }
    if promotion_path.exists():
        promotion = validate_chat_promotion(
            paths.checkpoint, paths.runtime_config
        )
        if any(promotion.get(key) != value for key, value in expected_hashes.items()):
            raise RuntimeError("Existing checkpoint promotion belongs to different evidence")
    else:
        create_chat_promotion(
            runtime_config_path=paths.runtime_config,
            checkpoint=paths.checkpoint,
            selector_report_path=paths.selector_report,
            final_evidence_path=paths.final_evidence,
            leakage_report_path=paths.leakage,
        )
    if paths.primary_pointer.exists():
        config_path, checkpoint_path = resolve_primary_pointer(paths.primary_pointer)
        if config_path != paths.runtime_config or checkpoint_path != paths.checkpoint:
            raise RuntimeError("Existing primary pointer selects a different runtime")
    else:
        write_primary_pointer(
            paths.primary_pointer,
            runtime_config_path=paths.runtime_config,
            checkpoint=paths.checkpoint,
        )


CommandRunner = Callable[[Sequence[str]], None]


def _subprocess_runner(command: Sequence[str]) -> None:
    print("FINAL_ONCE_COMMAND " + shlex.join(command), flush=True)
    environment = os.environ.copy()
    source_root = str(PROJECT_ROOT / "src")
    current_pythonpath = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = (
        source_root if not current_pythonpath else f"{source_root}{os.pathsep}{current_pythonpath}"
    )
    subprocess.run(command, cwd=PROJECT_ROOT, check=True, env=environment)


@contextmanager
def _exclusive_run_lock(paths: WorkflowPaths) -> Iterator[None]:
    """Prevent two resumptions from racing the same immutable stage receipts."""

    lock_path = paths.work_root / "run.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+", encoding="utf-8") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f"Another final-once process holds {lock_path}") from exc
        handle.seek(0)
        handle.truncate()
        handle.write(f"pid={os.getpid()}\n")
        handle.flush()
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _run_authorized_final_once(
    paths: WorkflowPaths,
    launch: Mapping[str, Any],
    *,
    stop_after: str | None = None,
    command_runner: CommandRunner = _subprocess_runner,
) -> dict[str, Any]:
    launch_hash = str(launch["identity_sha256"])
    dataset = load_config(paths.dataset_config)
    completed: list[str] = []
    for stage in STAGES:
        current_launch = authorize_launch(paths, write=False)
        if current_launch["identity_sha256"] != launch_hash:
            raise RuntimeError("Final-once inputs changed between stages")
        receipt_path = _receipt_path(paths, stage)
        roots = _stage_output_roots(stage, paths, dataset)
        if receipt_path.is_file():
            _validate_receipt(
                receipt_path,
                stage=stage,
                launch_sha256=launch_hash,
                roots=roots,
            )
            completed.append(stage)
            if stop_after == stage:
                break
            continue
        # Revalidate all earlier completed receipts immediately before every
        # state-changing stage, including after a process restart.
        for prior in completed:
            _validate_receipt(
                _receipt_path(paths, prior),
                stage=prior,
                launch_sha256=launch_hash,
                roots=_stage_output_roots(prior, paths, dataset),
            )
        if stage == "final_evidence":
            _create_or_validate_final_evidence(paths, dataset)
        elif stage == "promotion":
            _create_or_validate_promotion(paths)
        else:
            if stage == "qa":
                _prepare_qa_build(paths)
            for command in build_stage_commands(
                stage,
                paths,
                dataset,
                launch=launch,
            ):
                command_runner(command)
            if stage == "qa":
                _publish_isolated_qa(paths, dataset, launch)
        if stage == "qa":
            _validate_qa_stage(paths, dataset, launch)
        elif stage == "questions":
            _validate_questions(paths, dataset)
        elif stage == "final_evidence":
            _validate_existing_final_evidence(paths)
        elif stage == "promotion":
            validate_chat_promotion(paths.checkpoint, paths.runtime_config)
            resolved_config, resolved_checkpoint = resolve_primary_pointer(paths.primary_pointer)
            if resolved_config != paths.runtime_config or resolved_checkpoint != paths.checkpoint:
                raise RuntimeError("Published primary pointer does not match launch seal")
        receipt = {
            "schema_version": RECEIPT_SCHEMA_VERSION,
            "stage": stage,
            "launch_identity_sha256": launch_hash,
            "outputs": output_inventory(roots),
        }
        _atomic_create_json(receipt_path, receipt)
        completed.append(stage)
        if stop_after == stage:
            break
    if completed == list(STAGES):
        receipt_hashes = {
            stage: sha256_file(_receipt_path(paths, stage)) for stage in STAGES
        }
        completion_payload = {
            "schema_version": COMPLETION_SCHEMA_VERSION,
            "kind": "gemma4_held_out_final_once_complete",
            "launch_identity_sha256": launch_hash,
            "stage_receipt_sha256": receipt_hashes,
            "checkpoint": _relative_or_absolute(paths.checkpoint),
            "runtime_config": _relative_or_absolute(paths.runtime_config),
            "primary_pointer": _relative_or_absolute(paths.primary_pointer),
            "promotion_sha256": sha256_file(paths.checkpoint / "promotion.json"),
            "final_evidence_sha256": sha256_file(paths.final_evidence),
            "leakage_sha256": sha256_file(paths.leakage),
        }
        if paths.completion.exists():
            if _json_object(paths.completion) != completion_payload:
                raise RuntimeError("Final-once completion seal is stale")
        else:
            _atomic_create_json(paths.completion, completion_payload)
    return {
        "launch_identity_sha256": launch_hash,
        "completed_stages": completed,
        "complete": completed == list(STAGES),
        "work_root": str(paths.work_root),
    }


def run_final_once(
    paths: WorkflowPaths,
    *,
    stop_after: str | None = None,
    command_runner: CommandRunner = _subprocess_runner,
) -> dict[str, Any]:
    if stop_after is not None and stop_after not in STAGES:
        raise ValueError(f"Unknown final-once stop_after stage: {stop_after}")
    launch = authorize_launch(paths, write=True)
    with _exclusive_run_lock(paths):
        # Recheck after acquiring the lock: another process may have completed
        # a stage between our initial authorization and lock acquisition.
        sealed = authorize_launch(paths, write=False)
        if sealed != launch:
            raise RuntimeError("Final-once launch changed before lock acquisition")
        return _run_authorized_final_once(
            paths,
            launch,
            stop_after=stop_after,
            command_runner=command_runner,
        )


def _resolved_executable(value: str, field: str) -> Path:
    candidate = Path(value).expanduser()
    resolved_value = shutil.which(value) if not candidate.is_absolute() else str(candidate)
    if resolved_value is None:
        raise FileNotFoundError(f"{field} executable is unavailable: {value}")
    resolved = Path(resolved_value).resolve()
    if not resolved.is_file() or not os.access(resolved, os.X_OK):
        raise FileNotFoundError(f"{field} executable is not runnable: {resolved}")
    return resolved


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name in ("preflight", "run"):
        command = subparsers.add_parser(name)
        command.add_argument("--dataset-config", default="configs/experiments/diverse28.yaml")
        command.add_argument("--runtime-config", default="configs/runtime/gemma4_primary.yaml")
        command.add_argument("--selector-report", required=True)
        command.add_argument("--checkpoint", required=True)
        command.add_argument("--work-root", default="reports/gemma4/final_once")
        command.add_argument("--primary-pointer", default="configs/runtime/primary.json")
        command.add_argument("--python", default=str(PROJECT_ROOT / ".venv/bin/python"))
        command.add_argument(
            "--gemma-python", default=str(PROJECT_ROOT / ".venv-gemma4/bin/python")
        )
        command.add_argument("--blender", default="blender")
        if name == "run":
            command.add_argument("--stop-after", choices=STAGES)
    return parser.parse_args(argv)


def _workflow_paths(args: argparse.Namespace) -> WorkflowPaths:
    return WorkflowPaths(
        dataset_config=_unresolved_rooted(args.dataset_config),
        runtime_config=_unresolved_rooted(args.runtime_config),
        selector_report=_unresolved_rooted(args.selector_report),
        checkpoint=_unresolved_rooted(args.checkpoint),
        work_root=_unresolved_rooted(args.work_root),
        primary_pointer=_unresolved_rooted(args.primary_pointer),
        coordinator_python=_resolved_executable(args.python, "Python"),
        gemma_python=_resolved_executable(args.gemma_python, "Gemma Python"),
        blender=str(_resolved_executable(args.blender, "Blender")),
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    paths = _workflow_paths(args)
    if args.command == "preflight":
        launch = authorize_launch(paths, write=False)
        print(
            json.dumps(
                {
                    "authorized": True,
                    "writes_performed": False,
                    "launch_identity_sha256": launch["identity_sha256"],
                    "selected_update": launch["selector"]["selected_update"],
                    "checkpoint": launch["checkpoint"]["path"],
                    "final_scene_ids": launch["split_contract"]["final_scene_ids"],
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    result = run_final_once(paths, stop_after=args.stop_after)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


def cli(argv: Sequence[str] | None = None) -> int:
    try:
        return main(argv)
    except (OSError, subprocess.CalledProcessError, RuntimeError, TypeError, ValueError) as exc:
        print(f"Final-once refused: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(cli())


__all__ = [
    "FINAL_SCENE_IDS",
    "STAGES",
    "WorkflowPaths",
    "authorize_launch",
    "build_launch_identity",
    "build_stage_commands",
    "output_inventory",
    "run_final_once",
]
