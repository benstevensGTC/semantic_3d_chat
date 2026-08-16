"""Package, smoke-test, and promote the strict V85 direct-memory runtime.

The runtime child process never imports this module.  In particular, the three
predeclared smoke expectations live here, outside the chat process, and are
applied only after that process has exited and its oracle-unavailable audit has
been written.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Final

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file

from semantic_3d_chat.chat.runtime_config import (
    effective_runtime_config_sha256,
    load_runtime_config,
)
from semantic_3d_chat.config import PROJECT_ROOT, config_hash
from semantic_3d_chat.evaluation.prediction_artifacts import checkpoint_fingerprint
from semantic_3d_chat.language.lora import tensor_state_sha256
from semantic_3d_chat.scene_encoder.v81_scene_memory_artifact import (
    MEMORY_FILENAME,
    METADATA_FILENAME,
    load_v81_scene_memory,
    save_v81_scene_memory,
)
from semantic_3d_chat.training.checkpointing import (
    RUNTIME_METADATA_FILENAME,
    validate_runtime_checkpoint_metadata,
)

SCHEMA_VERSION: Final[int] = 85
BRIDGE_BANK: Final[str] = "v85_strict_multiscene_bridge"
BRIDGE_TARGET: Final[str] = "model.language_model.layers.34.mlp.down_proj"
BRIDGE_STATE_SHA256: Final[str] = (
    "f31b8f99f77f1b7b92dafd74220e5e12ccfa35cbc8630a6d7640f2fe1f93c581"
)
BRIDGE_PARAMETER_COUNT: Final[int] = 55_296
BASE_ADAPTER_SHA256: Final[str] = (
    "6c627f0a0d9efb7100489c24cbd4acafe10456b79280a64a85399a21cb541daf"
)
PREREGISTRATION_SHA256: Final[str] = (
    "4af534bc37cd09fe7431042ff6fb75bd734a267380e1fe425c6c87b2cb42afff"
)
TRAINING_REPORT_SHA256: Final[str] = (
    "d7c352fd0d6c6dec23f80de61f49efe00635aac30988e3a783a7483c97f79e96"
)
DEVELOPMENT_SCORE_SHA256: Final[str] = (
    "202134d8900e105d63f23d1cc1d19d68a882c4464382b7a63b7aa007f2714828"
)
EXPERIMENT_CONFIG_SHA256: Final[str] = (
    "d4f653dc20a7ad129eb9fa92b586c8ca472a49fdb72675cbddb4f03007b4c36d"
)
SOURCE_MEMORY_PREFIX_SHA256: Final[str] = (
    "a428f5147c815839ae7315a0adab952ab210814fb21dcdc5bf13b167f28a6e37"
)
SOURCE_MEMORY_TENSOR_FILE_SHA256: Final[str] = (
    "3427851432d8f2a3609b6205b18b8c0d9a0fcf68d8f3bf0c98c758ac64209ffb"
)

RUNTIME_CONFIG: Final[Path] = PROJECT_ROOT / "configs/runtime/gemma4_v85_strict_multiscene.yaml"
BASE_CHECKPOINT: Final[Path] = (
    PROJECT_ROOT / "data_gemma4/runtime/checkpoints/gemma4_v54_release_v1"
)
BRIDGE_CANDIDATE: Final[Path] = (
    PROJECT_ROOT / "reports/gemma4/artifacts/v85_strict_multiscene_final"
)
SOURCE_MEMORY: Final[Path] = (
    PROJECT_ROOT / "data_gemma4/runtime/scene_memories/v81/scene_000001"
)
CANDIDATE_CHECKPOINT: Final[Path] = (
    PROJECT_ROOT / "reports/gemma4/artifacts/v85_strict_runtime_candidate"
)
CANDIDATE_MEMORY: Final[Path] = (
    PROJECT_ROOT / "reports/gemma4/artifacts/v85_strict_runtime_candidate_memory/scene_000001"
)
RELEASE_CHECKPOINT: Final[Path] = (
    PROJECT_ROOT / "data_gemma4/runtime/checkpoints/gemma4_v85_strict_multiscene_release_v1"
)
RELEASE_MEMORY: Final[Path] = (
    PROJECT_ROOT / "data_gemma4/runtime/scene_memories/v85/scene_000001"
)
SMOKE_CHAT: Final[Path] = PROJECT_ROOT / "reports/gemma4/examples/v85_strict_runtime_smoke.jsonl"
SMOKE_AUDIT: Final[Path] = (
    PROJECT_ROOT / "reports/gemma4/metrics/v85_strict_runtime_smoke_access.json"
)
SMOKE_REPORT: Final[Path] = (
    PROJECT_ROOT / "reports/gemma4/metrics/v85_strict_runtime_smoke.json"
)
RELEASE_REPORT: Final[Path] = (
    PROJECT_ROOT / "reports/gemma4/metrics/gemma4_v85_strict_runtime_release.json"
)
DEVELOPMENT_CACHE: Final[Path] = (
    PROJECT_ROOT / "reports/gemma4/artifacts/v82_strict_dense_reader/development_cache"
)
HELD_MEMORY_REPORT: Final[Path] = (
    PROJECT_ROOT / "reports/gemma4/metrics/v85_strict_runtime_scene_000039_export.json"
)
EQUIVALENCE_PREREGISTRATION: Final[Path] = (
    PROJECT_ROOT / "reports/gemma4/metrics/v85_strict_runtime_equivalence_preregistration.json"
)
EQUIVALENCE_CHAT: Final[Path] = (
    PROJECT_ROOT / "reports/gemma4/examples/v85_strict_runtime_equivalence_scene_000039.jsonl"
)
EQUIVALENCE_AUDIT: Final[Path] = (
    PROJECT_ROOT / "reports/gemma4/metrics/v85_strict_runtime_equivalence_access_scene_000039.json"
)
EQUIVALENCE_REPORT: Final[Path] = (
    PROJECT_ROOT / "reports/gemma4/metrics/v85_strict_runtime_equivalence_scene_000039.json"
)

# Evaluation-only behavior assertions.  They are never imported by the chat CLI.
_SMOKE_CASES: Final[tuple[tuple[str, str], ...]] = (
    ("Is there a chair?", "yes"),
    ("What color is the bowl?", "red"),
    ("Is the bowl left or right of the chair?", "right"),
)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"Expected a JSON object: {path}")
    return value


def _write_json(path: Path, payload: Mapping[str, Any], *, exclusive: bool = True) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    mode = "x" if exclusive else "w"
    with path.open(mode, encoding="utf-8") as handle:
        json.dump(dict(payload), handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _authenticate_v85_sources() -> None:
    expected = {
        PROJECT_ROOT
        / "reports/gemma4/metrics/gemma4_v85_strict_multiscene_preregistration.json": (
            PREREGISTRATION_SHA256
        ),
        PROJECT_ROOT
        / "reports/gemma4/metrics/gemma4_v85_strict_multiscene_training.json": (
            TRAINING_REPORT_SHA256
        ),
        PROJECT_ROOT
        / "reports/gemma4/metrics/gemma4_v85_strict_multiscene_development.json": (
            DEVELOPMENT_SCORE_SHA256
        ),
        PROJECT_ROOT / "configs/experiments/gemma4_v85_strict_multiscene.yaml": (
            EXPERIMENT_CONFIG_SHA256
        ),
        BASE_CHECKPOINT / "adapter.safetensors": BASE_ADAPTER_SHA256,
    }
    mismatches = {
        str(path.relative_to(PROJECT_ROOT)): {
            "expected": digest,
            "observed": None if not path.is_file() else sha256_file(path),
        }
        for path, digest in expected.items()
        if not path.is_file() or sha256_file(path) != digest
    }
    if mismatches:
        raise ValueError(f"V85 immutable source authentication failed: {mismatches}")
    development = _read_json(
        PROJECT_ROOT
        / "reports/gemma4/metrics/gemma4_v85_strict_multiscene_development.json"
    )
    if (
        development.get("separate_leakage_runtime_packaging_authorized") is not True
        or development.get("runtime_promotion_authorized") is not False
        or development.get("automatic_runtime_promotion") is not False
    ):
        raise ValueError("V85 development result does not authorize separate packaging")


def _bridge_tensors() -> dict[str, torch.Tensor]:
    source = load_file(str(BRIDGE_CANDIDATE / "bridge.safetensors"), device="cpu")
    if set(source) != {"lora_a", "lora_b"}:
        raise ValueError("V85 bridge tensor inventory changed")
    state = {
        "adapters.0.lora_a": source["lora_a"].float().contiguous(),
        "adapters.0.lora_b": source["lora_b"].float().contiguous(),
    }
    if (
        tuple(state["adapters.0.lora_a"].shape) != (4, 12_288)
        or tuple(state["adapters.0.lora_b"].shape) != (1_536, 4)
        or tensor_state_sha256(state) != BRIDGE_STATE_SHA256
    ):
        raise ValueError("V85 bridge shape or state identity changed")
    return state


def _runtime_provenance(*, promotion: str, smoke_report_sha256: str | None) -> dict[str, Any]:
    provenance = dict(
        _read_json(BASE_CHECKPOINT / RUNTIME_METADATA_FILENAME)["initialization_provenance"]
    )
    provenance["v85_strict_runtime_release"] = {
        "schema_version": SCHEMA_VERSION,
        "experiment_config_sha256": EXPERIMENT_CONFIG_SHA256,
        "preregistration_sha256": PREREGISTRATION_SHA256,
        "training_report_sha256": TRAINING_REPORT_SHA256,
        "development_score_sha256": DEVELOPMENT_SCORE_SHA256,
        "fixed_bridge_state_sha256": BRIDGE_STATE_SHA256,
        "promotion_decision": promotion,
        "runtime_promotion_authorized": promotion == "strict_experimental_primary",
        "v75_comparator_retained": True,
        "smoke_report_sha256": smoke_report_sha256,
    }
    return provenance


def _frozen_source_stack_sha256() -> str:
    """Hash the exact V54 source stack plus V85, excluding only V35 itself."""

    tensors, _inheritance = _merged_adapter()
    source = {
        name: value
        for name, value in tensors.items()
        if not name.startswith("block_cross_residual.")
    }
    if not source or len(source) >= len(tensors):
        raise RuntimeError("V85 block-cross frozen-source inventory is invalid")
    return tensor_state_sha256(source)


def build_runtime_metadata(
    *, promotion: str, smoke_report_sha256: str | None
) -> dict[str, Any]:
    config = load_runtime_config(RUNTIME_CONFIG)
    metadata = _read_json(BASE_CHECKPOINT / RUNTIME_METADATA_FILENAME)
    metadata["config_hash"] = config_hash(config)
    banks = [dict(record) for record in metadata["lora"]["banks"]]
    banks.append(
        {
            "name": BRIDGE_BANK,
            "trainable": False,
            "rank": 4,
            "alpha": 8.0,
            "dropout": 0.0,
            "target_modules": [BRIDGE_TARGET],
            "initialization_algorithm": "checkpoint_overwrite",
            "initialization_seed": None,
            "expected_initial_state_sha256": BRIDGE_STATE_SHA256,
            "adapter_parameter_count": BRIDGE_PARAMETER_COUNT,
        }
    )
    total = int(metadata["lora"]["adapter_parameter_count"]) + BRIDGE_PARAMETER_COUNT
    metadata["lora"] = {
        "schema_version": 2,
        "enabled": True,
        "banks": banks,
        "adapter_parameter_count": total,
        "trainable_adapter_parameter_count": 0,
    }
    metadata["lora_bank_wrapped_modules"] = {
        **metadata["lora_bank_wrapped_modules"],
        BRIDGE_BANK: [BRIDGE_TARGET],
    }
    metadata["lora_bank_parameter_counts"] = {
        **metadata["lora_bank_parameter_counts"],
        BRIDGE_BANK: {BRIDGE_TARGET: BRIDGE_PARAMETER_COUNT},
    }
    metadata["lora_bank_state_sha256"] = {
        **metadata["lora_bank_state_sha256"],
        BRIDGE_BANK: BRIDGE_STATE_SHA256,
    }
    metadata["lora_parameter_count"] = total
    metadata["lora_trainable_parameter_count"] = 0
    # The generic runtime authenticates every frozen source-stack tensor.  The
    # V85 bank is an intentional new frozen member, so the source-stack digest
    # must be rebound to that exact extended archive.
    metadata["frozen_block_cross_source_stack_state_sha256"] = (
        _frozen_source_stack_sha256()
    )
    metadata["initialization_provenance"] = _runtime_provenance(
        promotion=promotion, smoke_report_sha256=smoke_report_sha256
    )
    validate_runtime_checkpoint_metadata(metadata)
    return metadata


def _merged_adapter() -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    base = load_file(str(BASE_CHECKPOINT / "adapter.safetensors"), device="cpu")
    merged = {name: value.detach().cpu().contiguous() for name, value in base.items()}
    bridge = _bridge_tensors()
    for suffix, value in bridge.items():
        key = f"lora_banks.{BRIDGE_BANK}.{suffix}"
        if key in merged:
            raise ValueError(f"V85 bridge key unexpectedly exists in V54: {key}")
        merged[key] = value
    base_state = tensor_state_sha256(base)
    retained_state = tensor_state_sha256(
        {name: merged[name] for name in base}
    )
    if retained_state != base_state:
        raise RuntimeError("V54 frozen adapter tensor bytes changed while adding V85")
    return merged, {
        "base_adapter_file_sha256": BASE_ADAPTER_SHA256,
        "base_tensor_state_sha256": base_state,
        "packaged_base_subset_state_sha256": retained_state,
        "base_tensor_count": len(base),
        "packaged_tensor_count": len(merged),
        "bridge_state_sha256": BRIDGE_STATE_SHA256,
        "base_tensors_byte_identical": True,
    }


def _atomic_checkpoint(
    destination: Path,
    *,
    metadata: Mapping[str, Any],
    source_adapter: Path | None = None,
) -> dict[str, Any]:
    if destination.exists():
        raise FileExistsError(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{destination.name}.", dir=destination.parent) as raw:
        temporary = Path(raw)
        if source_adapter is None:
            tensors, inheritance = _merged_adapter()
            save_file(tensors, str(temporary / "adapter.safetensors"))
        else:
            shutil.copyfile(source_adapter, temporary / "adapter.safetensors")
            inheritance = {
                "base_adapter_file_sha256": BASE_ADAPTER_SHA256,
                "bridge_state_sha256": BRIDGE_STATE_SHA256,
                "candidate_adapter_bytes_reused_exactly": True,
            }
        _write_json(temporary / RUNTIME_METADATA_FILENAME, metadata)
        if {item.name for item in temporary.iterdir()} != {
            "adapter.safetensors",
            RUNTIME_METADATA_FILENAME,
        }:
            raise RuntimeError("V85 runtime checkpoint is not an exact two-file package")
        os.replace(temporary, destination)
    fingerprint, files = checkpoint_fingerprint(destination)
    return {
        **inheritance,
        "checkpoint_sha256": fingerprint,
        "checkpoint_files": files,
        "adapter_sha256": sha256_file(destination / "adapter.safetensors"),
        "runtime_metadata_sha256": sha256_file(destination / RUNTIME_METADATA_FILENAME),
        "exact_two_file_checkpoint": True,
    }


def _rebind_memory(
    destination: Path, *, checkpoint_sha256: str, runtime_config_sha256: str
) -> dict[str, Any]:
    if destination.exists():
        raise FileExistsError(destination)
    source_metadata = _read_json(SOURCE_MEMORY / METADATA_FILENAME)
    if (
        source_metadata.get("canonical_prefix_sha256") != SOURCE_MEMORY_PREFIX_SHA256
        or source_metadata.get("tensor_file_sha256")
        != SOURCE_MEMORY_TENSOR_FILE_SHA256
        or sha256_file(SOURCE_MEMORY / MEMORY_FILENAME)
        != SOURCE_MEMORY_TENSOR_FILE_SHA256
    ):
        raise ValueError("Source V81 scene-memory bytes changed")
    rebound = dict(source_metadata)
    rebound["source_base_checkpoint_sha256"] = checkpoint_sha256
    rebound["runtime_config_sha256"] = runtime_config_sha256
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{destination.name}.", dir=destination.parent) as raw:
        temporary = Path(raw)
        shutil.copyfile(SOURCE_MEMORY / MEMORY_FILENAME, temporary / MEMORY_FILENAME)
        _write_json(temporary / METADATA_FILENAME, rebound)
        if sha256_file(temporary / MEMORY_FILENAME) != SOURCE_MEMORY_TENSOR_FILE_SHA256:
            raise RuntimeError("Scene-memory tensor file changed during metadata-only rebinding")
        os.replace(temporary, destination)
    loaded = load_v81_scene_memory(
        destination,
        expected_scene_id="scene_000001",
        expected_base_checkpoint_sha256=checkpoint_sha256,
        expected_runtime_config_sha256=runtime_config_sha256,
        expected_model_device="cpu",
    )
    if loaded.metadata["canonical_prefix_sha256"] != SOURCE_MEMORY_PREFIX_SHA256:
        raise RuntimeError("Scene-memory canonical prefix changed during rebinding")
    return {
        "source_memory_tensor_file_sha256": SOURCE_MEMORY_TENSOR_FILE_SHA256,
        "packaged_memory_tensor_file_sha256": sha256_file(
            destination / MEMORY_FILENAME
        ),
        "canonical_prefix_sha256": loaded.metadata["canonical_prefix_sha256"],
        "metadata_only_rebinding": True,
        "memory_tensor_file_bytes_unchanged": True,
        "question_data_used_for_rebinding": False,
        "exact_two_file_scene_memory": {item.name for item in destination.iterdir()}
        == {MEMORY_FILENAME, METADATA_FILENAME},
    }


def prepare_candidate() -> dict[str, Any]:
    _authenticate_v85_sources()
    config = load_runtime_config(RUNTIME_CONFIG)
    metadata = build_runtime_metadata(
        promotion="pending_strict_runtime_leakage", smoke_report_sha256=None
    )
    checkpoint = _atomic_checkpoint(CANDIDATE_CHECKPOINT, metadata=metadata)
    memory = _rebind_memory(
        CANDIDATE_MEMORY,
        checkpoint_sha256=str(checkpoint["checkpoint_sha256"]),
        runtime_config_sha256=effective_runtime_config_sha256(config),
    )
    result = {
        "schema_version": SCHEMA_VERSION,
        "phase": "v85_strict_runtime_candidate_prepared",
        "candidate_checkpoint": str(CANDIDATE_CHECKPOINT.relative_to(PROJECT_ROOT)),
        "candidate_memory": str(CANDIDATE_MEMORY.relative_to(PROJECT_ROOT)),
        "runtime_config_sha256": effective_runtime_config_sha256(config),
        "checkpoint": checkpoint,
        "scene_memory": memory,
        "runtime_metadata_contains_supervision": False,
        "promotion_decision": "pending_strict_runtime_leakage",
    }
    return result


def verify_candidate() -> dict[str, Any]:
    _authenticate_v85_sources()
    if not CANDIDATE_CHECKPOINT.is_dir() or not CANDIDATE_MEMORY.is_dir():
        raise FileNotFoundError("V85 strict runtime candidate package is incomplete")
    metadata = _read_json(CANDIDATE_CHECKPOINT / RUNTIME_METADATA_FILENAME)
    expected = build_runtime_metadata(
        promotion="pending_strict_runtime_leakage", smoke_report_sha256=None
    )
    if metadata != expected:
        raise ValueError("V85 candidate runtime metadata changed")
    fingerprint, files = checkpoint_fingerprint(CANDIDATE_CHECKPOINT)
    tensors = load_file(str(CANDIDATE_CHECKPOINT / "adapter.safetensors"), device="cpu")
    base = load_file(str(BASE_CHECKPOINT / "adapter.safetensors"), device="cpu")
    bridge = _bridge_tensors()
    checks = {
        "exact_two_file_checkpoint": {item["path"] for item in files}
        == {"adapter.safetensors", RUNTIME_METADATA_FILENAME},
        "all_base_tensors_present_and_equal": set(base).issubset(tensors)
        and all(torch.equal(base[name], tensors[name]) for name in base),
        "bridge_a_equal": torch.equal(
            tensors[f"lora_banks.{BRIDGE_BANK}.adapters.0.lora_a"],
            bridge["adapters.0.lora_a"],
        ),
        "bridge_b_equal": torch.equal(
            tensors[f"lora_banks.{BRIDGE_BANK}.adapters.0.lora_b"],
            bridge["adapters.0.lora_b"],
        ),
        "scene_memory_bytes_unchanged": sha256_file(CANDIDATE_MEMORY / MEMORY_FILENAME)
        == SOURCE_MEMORY_TENSOR_FILE_SHA256,
    }
    load_v81_scene_memory(
        CANDIDATE_MEMORY,
        expected_scene_id="scene_000001",
        expected_base_checkpoint_sha256=fingerprint,
        expected_runtime_config_sha256=effective_runtime_config_sha256(
            load_runtime_config(RUNTIME_CONFIG)
        ),
        expected_model_device="cpu",
    )
    if not all(checks.values()):
        raise RuntimeError(f"V85 strict candidate verification failed: {checks}")
    return {
        "phase": "v85_strict_runtime_candidate_verified",
        "checkpoint_sha256": fingerprint,
        "checks": checks,
        "passed": True,
    }


def export_held_memory(scene_id: str = "scene_000039") -> dict[str, Any]:
    """Export one preselected scene-disjoint numeric memory, without QA rows."""

    if scene_id != "scene_000039":
        raise ValueError("V85 first held-memory export is fixed to scene_000039")
    candidate = verify_candidate()
    destination = CANDIDATE_MEMORY.parent / scene_id
    if destination.exists() and HELD_MEMORY_REPORT.is_file():
        return _read_json(HELD_MEMORY_REPORT)
    if destination.exists() or HELD_MEMORY_REPORT.exists():
        raise FileExistsError("V85 held-memory export is partially occupied")
    cache_metadata_path = DEVELOPMENT_CACHE / "metadata.json"
    cache_tensor_path = DEVELOPMENT_CACHE / "training_tensors.safetensors"
    if sha256_file(cache_tensor_path) != (
        "71222bdc34fc836015ea4449622491545515222a7df99c321c1bf7a22fa2b659"
    ):
        raise ValueError("V85 question-free development memory cache changed")
    cache = _read_json(cache_metadata_path)
    scene_ids = cache.get("scene_ids")
    if (
        not isinstance(scene_ids, list)
        or scene_id not in scene_ids
        or cache.get("questions_or_answers_serialized") is not False
        or cache.get("environmental_text_serialized") is not False
        or cache.get("oracle_serialized") is not False
    ):
        raise ValueError("V85 development cache is not the sealed numeric contract")
    with safe_open(str(cache_tensor_path), framework="pt", device="cpu") as handle:
        if handle.metadata() != {
            "artifact": "v82_numeric_reader_cache_v1",
            "schema_version": "82",
            "oracle_serialized": "false",
            "environmental_text_serialized": "false",
            "questions_or_answers_serialized": "false",
        }:
            raise ValueError("V85 development cache tensor metadata changed")
        memories = handle.get_tensor("scene_memories")
    memory = memories[scene_ids.index(scene_id)].unsqueeze(0).contiguous()
    config = load_runtime_config(RUNTIME_CONFIG)
    metadata = save_v81_scene_memory(
        destination,
        memory,
        scene_id=scene_id,
        source_base_checkpoint_sha256=str(candidate["checkpoint_sha256"]),
        runtime_config_sha256=effective_runtime_config_sha256(config),
        source_control_checkpoint_sha256=str(cache["source_controller_sha256"]),
        source_probe_tensor_sha256=str(cache["source_probe_tensor_sha256"]),
    )
    if metadata["canonical_prefix_sha256"] != (
        "f2587d717746678c6d08d14e46ea5e51465f065b586938ce8595cd81a1cfa36a"
    ):
        raise RuntimeError("V85 preselected scene_000039 memory identity changed")
    report = {
        "schema_version": SCHEMA_VERSION,
        "artifact": "gemma4_v85_strict_runtime_held_memory_export_v1",
        "scene_id": scene_id,
        "selection": "fixed_first_scene_in_preregistered_scene_disjoint_development_inventory",
        "selected_before_runtime_behavior": True,
        "behavior_used_for_selection": False,
        "source_cache_sha256": sha256_file(cache_tensor_path),
        "source_cache_questions_or_answers_serialized": False,
        "source_cache_environmental_text_serialized": False,
        "source_cache_oracle_serialized": False,
        "only_scene_memories_tensor_loaded": True,
        "runtime_inventory": sorted(item.name for item in destination.iterdir()),
        "canonical_prefix_sha256": metadata["canonical_prefix_sha256"],
        "runtime_checkpoint_sha256": candidate["checkpoint_sha256"],
        "candidate_only_not_promoted": True,
    }
    _write_json(HELD_MEMORY_REPORT, report)
    return report


def preregister_equivalence() -> dict[str, Any]:
    """Seal all scene-39 prompts and expected V85 outputs before runtime replay."""

    if EQUIVALENCE_PREREGISTRATION.is_file():
        return _read_json(EQUIVALENCE_PREREGISTRATION)
    prediction_path = (
        PROJECT_ROOT
        / "reports/gemma4/predictions/gemma4_v85_strict_multiscene_development.json"
    )
    qa_path = PROJECT_ROOT / "data_diverse52/qa/train.jsonl"
    if sha256_file(prediction_path) != (
        "6894c089aff2e34d80172aa443bb84695fc20d737ef2498f04b879cc1941b3c1"
    ) or sha256_file(qa_path) != (
        "01721bf904b1ab0b65ce8acac6e366287040873cda1356da6c70c4981abe7619"
    ):
        raise ValueError("V85 equivalence source bytes changed")
    predictions = _read_json(prediction_path)
    records = [
        row
        for row in predictions.get("records", [])
        if row.get("scene_id") == "scene_000039"
    ]
    qa_rows = [
        json.loads(line)
        for line in qa_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    questions = {
        str(row["question_id"]): str(row["question"])
        for row in qa_rows
        if row.get("scene_id") == "scene_000039"
    }
    if len(records) != 24 or len(questions) != 24:
        raise ValueError("V85 scene-39 equivalence inventory must contain all 24 rows")
    inventory = [
        {
            "ordinal": ordinal,
            "question_id": str(row["question_id"]),
            "question": questions[str(row["question_id"])],
            "sealed_expected_prediction": str(row["correct_scene_prediction"]),
            "scene_memory_sha256": str(row["scene_memory_sha256"]),
        }
        for ordinal, row in enumerate(records, 1)
    ]
    if any(
        row["scene_memory_sha256"]
        != "f2587d717746678c6d08d14e46ea5e51465f065b586938ce8595cd81a1cfa36a"
        for row in inventory
    ):
        raise ValueError("V85 scene-39 sealed memory identity changed")
    preregistration = {
        "schema_version": SCHEMA_VERSION,
        "artifact": "gemma4_v85_strict_runtime_equivalence_preregistration_v1",
        "status": "sealed_before_runtime_replay",
        "scene_id": "scene_000039",
        "selection_rule": "all_24_rows_for_fixed_first_development_scene",
        "selection_based_on_runtime_outputs": False,
        "checkpoint_selection_or_promotion_permitted": False,
        "source_predictions_sha256": sha256_file(prediction_path),
        "source_qa_sha256": sha256_file(qa_path),
        "row_count": len(inventory),
        "inventory_sha256": _canonical_sha256(inventory),
        "inventory": inventory,
    }
    _write_json(EQUIVALENCE_PREREGISTRATION, preregistration)
    return preregistration


def run_equivalence() -> dict[str, Any]:
    """Replay all predeclared scene-39 prompts through the isolated chat CLI."""

    if EQUIVALENCE_REPORT.is_file():
        return _read_json(EQUIVALENCE_REPORT)
    if EQUIVALENCE_CHAT.exists() or EQUIVALENCE_AUDIT.exists():
        raise FileExistsError("V85 equivalence artifacts are partially occupied")
    preregistration = preregister_equivalence()
    if preregistration["inventory_sha256"] != _canonical_sha256(
        preregistration["inventory"]
    ):
        raise ValueError("V85 equivalence preregistration inventory changed")
    export_held_memory()
    held_memory = CANDIDATE_MEMORY.parent / "scene_000039"
    oracle = PROJECT_ROOT / "data/oracle"
    unavailable = PROJECT_ROOT / f"data/.oracle-unavailable-v85-equivalence-{os.getpid()}"
    if not oracle.is_dir() or unavailable.exists():
        raise FileNotFoundError("Oracle cannot be made unavailable for V85 equivalence")
    command = [
        str(PROJECT_ROOT / ".venv-gemma4/bin/python"),
        "-m",
        "semantic_3d_chat.chat.v85_strict_multiscene_cli",
        "--config",
        str(RUNTIME_CONFIG),
        "--scene",
        "scene_000039",
        "--base-checkpoint",
        str(CANDIDATE_CHECKPOINT),
        "--scene-memory",
        str(held_memory),
        "--audit-log",
        str(EQUIVALENCE_AUDIT),
        "--chat-log",
        str(EQUIVALENCE_CHAT),
        "--allow-candidate",
    ]
    for row in preregistration["inventory"]:
        command.extend(("--question", str(row["question"])))
    environment = dict(os.environ)
    environment.update(
        {
            "PYTHONPATH": str(PROJECT_ROOT / "src"),
            "TRANSFORMERS_OFFLINE": "1",
            "HF_HUB_OFFLINE": "1",
        }
    )
    oracle_unavailable = False
    try:
        os.rename(oracle, unavailable)
        oracle_unavailable = not oracle.exists() and unavailable.is_dir()
        completed = subprocess.run(
            command,
            cwd=PROJECT_ROOT,
            env=environment,
            text=True,
            capture_output=True,
            check=False,
        )
    finally:
        if unavailable.exists():
            os.rename(unavailable, oracle)
    if completed.returncode != 0:
        raise RuntimeError(
            "V85 equivalence child failed: "
            f"returncode={completed.returncode}\nstdout={completed.stdout}\nstderr={completed.stderr}"
        )
    observed = [
        json.loads(line)
        for line in EQUIVALENCE_CHAT.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(observed) != 24:
        raise RuntimeError("V85 equivalence runtime did not emit all 24 rows")
    comparisons = [
        {
            "ordinal": expected["ordinal"],
            "question_id": expected["question_id"],
            "expected": expected["sealed_expected_prediction"],
            "observed": row["answer"],
            "exact_match": row["answer"] == expected["sealed_expected_prediction"],
        }
        for expected, row in zip(
            preregistration["inventory"], observed, strict=True
        )
    ]
    audit = _read_json(EQUIVALENCE_AUDIT)
    prefix_hashes = [str(row["prefix_hash"]) for row in observed]
    input_hashes = [
        str(row["environment_conditioned_input_sha256"]) for row in observed
    ]
    exact = sum(bool(row["exact_match"]) for row in comparisons)
    gates = {
        "all_24_sealed_predictions_exactly_reproduced": exact == 24,
        "oracle_physically_unavailable": oracle_unavailable,
        "oracle_restored": oracle.is_dir(),
        "forbidden_access_count_zero": not audit.get("forbidden_accesses"),
        "prefix_hash_invariant": len(set(prefix_hashes)) == 1,
        "total_environment_conditioned_input_invariant": len(set(input_hashes)) == 1,
        "expected_scene_memory_hash": set(prefix_hashes)
        == {"f2587d717746678c6d08d14e46ea5e51465f065b586938ce8595cd81a1cfa36a"},
    }
    result = {
        "schema_version": SCHEMA_VERSION,
        "artifact": "gemma4_v85_strict_runtime_equivalence_scene_000039_v1",
        "preregistration_sha256": sha256_file(EQUIVALENCE_PREREGISTRATION),
        "inventory_sha256": preregistration["inventory_sha256"],
        "scene_id": "scene_000039",
        "correct": exact,
        "total": len(comparisons),
        "accuracy": exact / len(comparisons),
        "comparisons": comparisons,
        "gates": gates,
        "packaging_numerically_equivalent": all(gates.values()),
        "selection_or_promotion_changed": False,
        "development_reopened_for_selection": False,
    }
    _write_json(EQUIVALENCE_REPORT, result)
    return result


def _normalized_answer(value: object) -> str:
    return str(value).strip().casefold().rstrip(".!?")


def run_smoke() -> dict[str, Any]:
    if SMOKE_REPORT.is_file():
        return _read_json(SMOKE_REPORT)
    if SMOKE_CHAT.exists() or SMOKE_AUDIT.exists():
        raise FileExistsError("V85 smoke artifacts already exist; results are create-once")
    if not CANDIDATE_CHECKPOINT.is_dir() or not CANDIDATE_MEMORY.is_dir():
        raise FileNotFoundError("Run V85 runtime candidate preparation first")
    oracle = PROJECT_ROOT / "data/oracle"
    unavailable = PROJECT_ROOT / f"data/.oracle-unavailable-v85-{os.getpid()}"
    if not oracle.is_dir() or unavailable.exists():
        raise FileNotFoundError("The oracle directory cannot be made physically unavailable")
    command = [
        str(PROJECT_ROOT / ".venv-gemma4/bin/python"),
        "-m",
        "semantic_3d_chat.chat.v85_strict_multiscene_cli",
        "--config",
        str(RUNTIME_CONFIG),
        "--scene",
        "scene_000001",
        "--base-checkpoint",
        str(CANDIDATE_CHECKPOINT),
        "--scene-memory",
        str(CANDIDATE_MEMORY),
        "--audit-log",
        str(SMOKE_AUDIT),
        "--chat-log",
        str(SMOKE_CHAT),
        "--allow-candidate",
    ]
    for question, _expected in _SMOKE_CASES:
        command.extend(("--question", question))
    environment = dict(os.environ)
    environment.update(
        {
            "PYTHONPATH": str(PROJECT_ROOT / "src"),
            "TRANSFORMERS_OFFLINE": "1",
            "HF_HUB_OFFLINE": "1",
        }
    )
    oracle_unavailable_during_runtime = False
    try:
        os.rename(oracle, unavailable)
        oracle_unavailable_during_runtime = not oracle.exists() and unavailable.is_dir()
        completed = subprocess.run(
            command,
            cwd=PROJECT_ROOT,
            env=environment,
            text=True,
            capture_output=True,
            check=False,
        )
    finally:
        if unavailable.exists():
            os.rename(unavailable, oracle)
    if completed.returncode != 0:
        raise RuntimeError(
            "V85 strict runtime smoke failed: "
            f"returncode={completed.returncode}\nstdout={completed.stdout}\nstderr={completed.stderr}"
        )
    rows = [
        json.loads(line)
        for line in SMOKE_CHAT.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(rows) != len(_SMOKE_CASES):
        raise RuntimeError("V85 smoke chat row count changed")
    behavior = []
    for row, (question, expected) in zip(rows, _SMOKE_CASES, strict=True):
        observed = _normalized_answer(row.get("answer"))
        behavior.append(
            {
                "question": question,
                "expected": expected,
                "observed": observed,
                "passed": observed == expected,
            }
        )
    audit = _read_json(SMOKE_AUDIT)
    prefix_hashes = [row.get("prefix_hash") for row in rows]
    input_hashes = [row.get("environment_conditioned_input_sha256") for row in rows]
    gates = {
        "runtime_process_exit_zero": completed.returncode == 0,
        "three_behavior_assertions_pass": all(row["passed"] for row in behavior),
        "oracle_physically_unavailable": oracle_unavailable_during_runtime,
        "oracle_restored_after_runtime": oracle.is_dir(),
        "file_audit_forbidden_read_count_zero": not audit.get("forbidden_accesses"),
        "prefix_hash_identical_for_every_question": len(set(prefix_hashes)) == 1,
        "total_environment_conditioned_input_identical": len(set(input_hashes)) == 1,
        "prefix_and_environment_input_identical": prefix_hashes == input_hashes,
        "source_memory_bytes_unchanged": (
            sha256_file(CANDIDATE_MEMORY / MEMORY_FILENAME)
            == SOURCE_MEMORY_TENSOR_FILE_SHA256
        ),
    }
    report = {
        "schema_version": SCHEMA_VERSION,
        "artifact": "gemma4_v85_strict_runtime_smoke_v1",
        "behavior_assertions_applied_after_chat_process_exit": True,
        "expected_behavior_not_loaded_by_chat_runtime": True,
        "chat_process_stdout_sha256": hashlib.sha256(completed.stdout.encode()).hexdigest(),
        "chat_process_stderr_sha256": hashlib.sha256(completed.stderr.encode()).hexdigest(),
        "behavior": behavior,
        "prefix_hashes": prefix_hashes,
        "environment_conditioned_input_hashes": input_hashes,
        "gates": gates,
        "passed": all(gates.values()),
        "promotion_authorized": all(gates.values()),
        "wrong_or_zero_memory_controls_rerun": False,
        "development_reopened": False,
    }
    _write_json(SMOKE_REPORT, report)
    return report


def promote_release() -> dict[str, Any]:
    _authenticate_v85_sources()
    smoke = _read_json(SMOKE_REPORT)
    if smoke.get("passed") is not True or smoke.get("promotion_authorized") is not True:
        raise ValueError("V85 smoke did not authorize strict runtime promotion")
    smoke_sha256 = sha256_file(SMOKE_REPORT)
    metadata = build_runtime_metadata(
        promotion="strict_experimental_primary", smoke_report_sha256=smoke_sha256
    )
    checkpoint = _atomic_checkpoint(
        RELEASE_CHECKPOINT,
        metadata=metadata,
        source_adapter=CANDIDATE_CHECKPOINT / "adapter.safetensors",
    )
    if checkpoint["adapter_sha256"] != sha256_file(
        CANDIDATE_CHECKPOINT / "adapter.safetensors"
    ):
        raise RuntimeError("Promoted V85 adapter bytes differ from smoked candidate")
    config = load_runtime_config(RUNTIME_CONFIG)
    memory = _rebind_memory(
        RELEASE_MEMORY,
        checkpoint_sha256=str(checkpoint["checkpoint_sha256"]),
        runtime_config_sha256=effective_runtime_config_sha256(config),
    )
    release = {
        "schema_version": SCHEMA_VERSION,
        "artifact": "gemma4_v85_strict_runtime_release_v1",
        "promotion_decision": "strict_experimental_primary",
        "promotion_scope": "strict_direct_continuous_scene_memory_static_chat",
        "v75_comparator_retained": True,
        "runtime_config": str(RUNTIME_CONFIG.relative_to(PROJECT_ROOT)),
        "runtime_config_sha256": effective_runtime_config_sha256(config),
        "checkpoint": checkpoint,
        "scene_memory": memory,
        "bindings": {
            "experiment_config_sha256": EXPERIMENT_CONFIG_SHA256,
            "preregistration_sha256": PREREGISTRATION_SHA256,
            "training_report_sha256": TRAINING_REPORT_SHA256,
            "development_score_sha256": DEVELOPMENT_SCORE_SHA256,
            "runtime_smoke_sha256": smoke_sha256,
        },
        "chat_runtime_loads_evaluation_reports": False,
        "runtime_checkpoint_contains_environmental_text": False,
        "runtime_checkpoint_contains_supervision": False,
        "scene_memory_metadata_only_rebinding": True,
        "scene_memory_tensor_bytes_unchanged": True,
        "all_release_gates_passed": True,
    }
    _write_json(RELEASE_REPORT, release)
    return release


def verify_release() -> dict[str, Any]:
    release = _read_json(RELEASE_REPORT)
    metadata = _read_json(RELEASE_CHECKPOINT / RUNTIME_METADATA_FILENAME)
    validate_runtime_checkpoint_metadata(metadata)
    fingerprint, files = checkpoint_fingerprint(RELEASE_CHECKPOINT)
    loaded = load_v81_scene_memory(
        RELEASE_MEMORY,
        expected_scene_id="scene_000001",
        expected_base_checkpoint_sha256=fingerprint,
        expected_runtime_config_sha256=effective_runtime_config_sha256(
            load_runtime_config(RUNTIME_CONFIG)
        ),
        expected_model_device="cpu",
    )
    provenance = metadata["initialization_provenance"]["v85_strict_runtime_release"]
    checks = {
        "release_report_promoted": release.get("promotion_decision")
        == "strict_experimental_primary",
        "exact_two_file_checkpoint": {item["path"] for item in files}
        == {"adapter.safetensors", RUNTIME_METADATA_FILENAME},
        "checkpoint_fingerprint_matches_release": fingerprint
        == release["checkpoint"]["checkpoint_sha256"],
        "adapter_matches_smoked_candidate": sha256_file(
            RELEASE_CHECKPOINT / "adapter.safetensors"
        )
        == sha256_file(CANDIDATE_CHECKPOINT / "adapter.safetensors"),
        "memory_bytes_match_source": sha256_file(RELEASE_MEMORY / MEMORY_FILENAME)
        == SOURCE_MEMORY_TENSOR_FILE_SHA256,
        "memory_prefix_match_source": loaded.metadata["canonical_prefix_sha256"]
        == SOURCE_MEMORY_PREFIX_SHA256,
        "runtime_metadata_bindings_exact": provenance["preregistration_sha256"]
        == PREREGISTRATION_SHA256
        and provenance["training_report_sha256"] == TRAINING_REPORT_SHA256
        and provenance["development_score_sha256"] == DEVELOPMENT_SCORE_SHA256,
        "runtime_promotion_authorized": provenance["runtime_promotion_authorized"] is True,
        "v75_comparator_retained": provenance["v75_comparator_retained"] is True,
    }
    if not all(checks.values()):
        raise RuntimeError(f"V85 strict runtime release verification failed: {checks}")
    return {"phase": "v85_strict_runtime_release_verified", "checks": checks, "passed": True}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=(
            "prepare",
            "verify-candidate",
            "export-held-memory",
            "preregister-equivalence",
            "run-equivalence",
            "smoke",
            "promote",
            "verify",
        ),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    functions = {
        "prepare": prepare_candidate,
        "verify-candidate": verify_candidate,
        "export-held-memory": export_held_memory,
        "preregister-equivalence": preregister_equivalence,
        "run-equivalence": run_equivalence,
        "smoke": run_smoke,
        "promote": promote_release,
        "verify": verify_release,
    }
    try:
        result = functions[args.command]()
    except (OSError, RuntimeError, TypeError, ValueError) as error:
        print(f"V85 strict runtime {args.command} refused: {error}", file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 2 if args.command == "smoke" and result.get("passed") is not True else 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "BRIDGE_BANK",
    "BRIDGE_STATE_SHA256",
    "CANDIDATE_CHECKPOINT",
    "CANDIDATE_MEMORY",
    "RELEASE_CHECKPOINT",
    "RELEASE_MEMORY",
    "RUNTIME_CONFIG",
    "build_runtime_metadata",
    "export_held_memory",
    "prepare_candidate",
    "preregister_equivalence",
    "promote_release",
    "run_equivalence",
    "run_smoke",
    "verify_candidate",
    "verify_release",
]
