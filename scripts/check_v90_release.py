#!/usr/bin/env python3
"""Model-free, fail-closed preflight for the promoted V90 scene-one demo.

This checker deliberately does not import Torch, Transformers, the V90 runtime,
or the V90 release module.  It authenticates the small on-disk inference
surface before the launcher is allowed to load Gemma.  The release module then
performs its independently sealed authentication and verification gates.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Final

import yaml

PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG: Final[Path] = Path("configs/runtime/gemma4_v90_strict_scene1.yaml")
DEFAULT_CHECKPOINT: Final[Path] = Path(
    "data_gemma4/runtime/checkpoints/gemma4_v90_strict_scene1_release_v1"
)
DEFAULT_MEMORY: Final[Path] = Path("data_gemma4/runtime/scene_memories/v90/scene_000001")
DEFAULT_RELEASE_REPORT: Final[Path] = Path(
    "reports/gemma4/metrics/gemma4_v90_strict_runtime_release.json"
)
DEFAULT_RUNTIME_CLI_SOURCE: Final[Path] = Path("src/semantic_3d_chat/chat/v90_strict_scene1_cli.py")
DEFAULT_RELEASE_SOURCE: Final[Path] = Path(
    "src/semantic_3d_chat/evaluation/v90_strict_runtime_release.py"
)

SCENE_ID: Final[str] = "scene_000001"
RELEASE_ARTIFACT: Final[str] = "gemma4_v90_strict_runtime_release_v1"
PROMOTION_DECISION: Final[str] = "strict_scene1_conversational_primary"
MODEL_ID: Final[str] = "google/gemma-4-E2B-it"
MODEL_REVISION: Final[str] = "3e22461f65e89153144f8adb70e3b8c2cc9845a7"
V90_BANK: Final[str] = "v90_scene1_conversational_bridge"
V90_TARGET: Final[str] = "model.language_model.layers.28.self_attn.o_proj"
EXPECTED_BANKS: Final[tuple[str, ...]] = (
    "inherited_v12",
    "extension_v13",
    "extension_v23_shared_kv",
    "extension_v24_shared_query",
    "extension_v28_stage_b_query",
    "extension_v30_joint_pair_query",
    "v85_strict_multiscene_bridge",
    "v86_scene1_demo_bridge",
    "v87_scene1_balanced_bridge",
    "v88_scene1_augmented_bridge",
    "v89_scene1_retention_bridge",
    V90_BANK,
)
EXPECTED_ADAPTER_PARAMETERS: Final[int] = 901_120
EXPECTED_MEMORY_PREFIX_SHA256: Final[str] = (
    "a428f5147c815839ae7315a0adab952ab210814fb21dcdc5bf13b167f28a6e37"
)
EXPECTED_MEMORY_FILE_SHA256: Final[str] = (
    "3427851432d8f2a3609b6205b18b8c0d9a0fcf68d8f3bf0c98c758ac64209ffb"
)
CHECKPOINT_INVENTORY: Final[frozenset[str]] = frozenset(
    {"adapter.safetensors", "runtime_metadata.json"}
)
MEMORY_INVENTORY: Final[frozenset[str]] = frozenset({"memory.safetensors", "runtime_metadata.json"})
_SHA256 = re.compile(r"[0-9a-f]{64}")


def _rooted(path: str | Path) -> Path:
    candidate = Path(path).expanduser()
    rooted = candidate if candidate.is_absolute() else PROJECT_ROOT / candidate
    return Path(os.path.abspath(rooted))


def _reject_symlink_components(path: Path, purpose: str) -> Path:
    current = Path(path.anchor)
    for component in path.parts[1:]:
        current /= component
        if current.is_symlink():
            raise ValueError(f"V90 demo {purpose} path contains a symbolic link: {current}")
    return path


def _safe_file(path: str | Path, purpose: str) -> Path:
    source = _reject_symlink_components(_rooted(path), purpose)
    if not source.is_file() or source.is_symlink():
        raise FileNotFoundError(f"V90 demo {purpose} is unavailable: {source}")
    return source


def _safe_exact_directory(path: str | Path, *, purpose: str, inventory: frozenset[str]) -> Path:
    source = _reject_symlink_components(_rooted(path), purpose)
    if not source.is_dir() or source.is_symlink():
        raise FileNotFoundError(f"V90 demo {purpose} is unavailable: {source}")
    observed = {item.name for item in source.iterdir()}
    if observed != inventory:
        raise ValueError(
            f"V90 demo {purpose} must have the exact inference inventory; "
            f"expected={sorted(inventory)} observed={sorted(observed)}"
        )
    for name in inventory:
        _safe_file(source / name, f"{purpose} {name}")
    return source


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _read_json(path: Path, purpose: str) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"V90 demo {purpose} must be a JSON object")
    return value


def _read_runtime_config(path: Path) -> tuple[dict[str, Any], str, str]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise TypeError("V90 demo runtime config must be a YAML mapping")
    if raw.get("_runtime_safe_config") is not True:
        raise ValueError("V90 demo runtime config is not marked runtime-safe")
    stable = {str(key): value for key, value in raw.items() if not str(key).startswith("_")}
    return raw, _canonical_sha256(stable), _sha256_file(path)


def _checkpoint_fingerprint(path: Path) -> tuple[str, list[dict[str, Any]]]:
    entries = [
        {
            "path": name,
            "sha256": _sha256_file(path / name),
            "size_bytes": (path / name).stat().st_size,
        }
        for name in ("adapter.safetensors", "runtime_metadata.json")
    ]
    return _canonical_sha256(entries), entries


def _require_sha256(value: object, purpose: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"V90 demo {purpose} is not a lowercase SHA-256 digest")
    return value


def _mapping(value: object, purpose: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"V90 demo {purpose} is not a mapping")
    return value


def _validate_config(config: Mapping[str, Any]) -> str:
    runtime = _mapping(config.get("runtime"), "runtime config runtime section")
    language = _mapping(config.get("language"), "runtime config language section")
    vision = _mapping(config.get("vision"), "runtime config vision section")
    if runtime.get("schema_version") != 1 or runtime.get("production") is not True:
        raise ValueError("V90 demo requires a production schema-one runtime config")
    if (
        language.get("backend") != "gemma4"
        or language.get("model_id") != MODEL_ID
        or language.get("revision") != MODEL_REVISION
        or vision.get("model_id") != MODEL_ID
        or vision.get("revision") != MODEL_REVISION
        or language.get("scene_prefix_after_bos") is not True
    ):
        raise ValueError("V90 demo runtime model or continuous-prefix contract changed")
    banks = _mapping(language.get("lora_banks"), "runtime LoRA banks")
    if tuple(str(name) for name in banks) != EXPECTED_BANKS:
        raise ValueError("V90 demo requires the exact ordered twelve-bank stack")
    if any(
        not isinstance(bank, Mapping) or bank.get("trainable") is not False
        for bank in banks.values()
    ):
        raise ValueError("Every V90 runtime LoRA bank must be frozen")
    bridge = _mapping(banks[V90_BANK], "V90 conversational bridge")
    if (
        bridge.get("target_modules") != [V90_TARGET]
        or bridge.get("rank") != 8
        or float(bridge.get("alpha", -1.0)) != 16.0
        or float(bridge.get("dropout", -1.0)) != 0.0
    ):
        raise ValueError("V90 runtime conversational bridge topology changed")
    return _require_sha256(
        bridge.get("expected_initial_state_sha256"), "configured V90 bridge state"
    )


def _validate_checkpoint(metadata: Mapping[str, Any], *, configured_v90_state: str) -> None:
    if (
        metadata.get("language_backend") != "gemma4"
        or metadata.get("language_model_id") != MODEL_ID
        or metadata.get("language_revision") != MODEL_REVISION
        or metadata.get("question_dependent_scene_processing") is not False
    ):
        raise ValueError("V90 checkpoint model or scene-processing contract changed")
    lora = _mapping(metadata.get("lora"), "checkpoint LoRA contract")
    banks_raw = lora.get("banks")
    if not isinstance(banks_raw, list) or not all(isinstance(row, Mapping) for row in banks_raw):
        raise TypeError("V90 checkpoint bank inventory is malformed")
    banks = list(banks_raw)
    if tuple(str(row.get("name")) for row in banks) != EXPECTED_BANKS:
        raise ValueError("V90 checkpoint does not contain exactly twelve ordered banks")
    if any(row.get("trainable") is not False for row in banks):
        raise ValueError("Every V90 checkpoint bank must be frozen")
    bridge = banks[-1]
    if (
        bridge.get("name") != V90_BANK
        or bridge.get("target_modules") != [V90_TARGET]
        or bridge.get("rank") != 8
        or float(bridge.get("alpha", -1.0)) != 16.0
        or float(bridge.get("dropout", -1.0)) != 0.0
        or bridge.get("adapter_parameter_count") != 28_672
        or bridge.get("expected_initial_state_sha256") != configured_v90_state
    ):
        raise ValueError("V90 checkpoint conversational bridge changed")
    if (
        lora.get("adapter_parameter_count") != EXPECTED_ADAPTER_PARAMETERS
        or lora.get("trainable_adapter_parameter_count") != 0
        or metadata.get("lora_parameter_count") != EXPECTED_ADAPTER_PARAMETERS
        or metadata.get("lora_trainable_parameter_count") != 0
    ):
        raise ValueError("V90 checkpoint adapter parameter inventory changed")
    states = _mapping(metadata.get("lora_bank_state_sha256"), "bank-state bindings")
    if set(states) != set(EXPECTED_BANKS):
        raise ValueError("V90 checkpoint bank-state bindings are not exact")
    if states.get(V90_BANK) != configured_v90_state:
        raise ValueError("V90 checkpoint and config bridge state differ")
    for name, state in states.items():
        _require_sha256(state, f"checkpoint state for {name}")

    initialization = _mapping(metadata.get("initialization_provenance"), "checkpoint provenance")
    release = _mapping(
        initialization.get("v90_strict_runtime_release"),
        "checkpoint V90 release provenance",
    )
    if (
        release.get("schema_version") != 90
        or release.get("promotion_decision") != PROMOTION_DECISION
        or release.get("runtime_promotion_authorized") is not True
        or release.get("model_acceptance_gate_passed") is not True
        or release.get("model_gate_report_authenticated") is not True
        or release.get("held_out_generalization_claim") is not False
    ):
        raise ValueError("V90 checkpoint was not promoted by the strict release gate")


def _validate_memory(
    metadata: Mapping[str, Any], *, checkpoint_sha256: str, config_sha256: str
) -> None:
    if (
        metadata.get("scene_id") != SCENE_ID
        or metadata.get("shape") != [1, 738, 1536]
        or metadata.get("fixed_memory_tokens") != 738
        or metadata.get("canonical_prefix_sha256") != EXPECTED_MEMORY_PREFIX_SHA256
        or metadata.get("tensor_file_sha256") != EXPECTED_MEMORY_FILE_SHA256
        or metadata.get("source_base_checkpoint_sha256") != checkpoint_sha256
        or metadata.get("runtime_config_sha256") != config_sha256
        or metadata.get("compiled_before_user_question") is not True
        or metadata.get("question_inputs_used_for_compilation") is not False
        or metadata.get("question_dependent_retrieval") is not False
        or metadata.get("question_dependent_scene_processing") is not False
        or metadata.get("semantic_or_spatial_top_k_selection") is not False
        or metadata.get("questions_or_answers_serialized") is not False
        or metadata.get("environmental_text_inputs") != []
        or metadata.get("oracle_loaded") is not False
    ):
        raise ValueError("V90 continuous scene-memory contract changed")


def _validate_release_report(
    report: Mapping[str, Any],
    *,
    config_path: Path,
    config_sha256: str,
    checkpoint_sha256: str,
    checkpoint_entries: list[dict[str, Any]],
) -> None:
    try:
        reported_config = _rooted(str(report["runtime_config"]))
    except KeyError as error:
        raise ValueError("V90 release report omitted its runtime config") from error
    if (
        report.get("artifact") != RELEASE_ARTIFACT
        or report.get("schema_version") != 90
        or report.get("scene_id") != SCENE_ID
        or report.get("promotion_decision") != PROMOTION_DECISION
        or report.get("all_release_gates_passed") is not True
        or reported_config != config_path
        or report.get("runtime_config_sha256") != config_sha256
        or report.get("runtime_checkpoint_contains_environmental_text") is not False
        or report.get("runtime_checkpoint_contains_supervision") is not False
        or report.get("chat_runtime_loads_training_or_evaluation_reports") is not False
        or report.get("scene_memory_metadata_only_rebinding") is not True
        or report.get("scene_memory_tensor_bytes_unchanged") is not True
        or report.get("held_out_generalization_claim") is not False
    ):
        raise ValueError("V90 strict release report did not pass exactly")

    checkpoint = _mapping(report.get("checkpoint"), "release checkpoint binding")
    if (
        checkpoint.get("exact_two_file_checkpoint") is not True
        or checkpoint.get("checkpoint_sha256") != checkpoint_sha256
        or checkpoint.get("checkpoint_files") != checkpoint_entries
        or checkpoint.get("adapter_sha256") != checkpoint_entries[0]["sha256"]
        or checkpoint.get("runtime_metadata_sha256") != checkpoint_entries[1]["sha256"]
    ):
        raise ValueError("V90 release checkpoint identity changed")

    memory = _mapping(report.get("scene_memory"), "release scene-memory binding")
    if (
        memory.get("canonical_prefix_sha256") != EXPECTED_MEMORY_PREFIX_SHA256
        or memory.get("exact_two_file_scene_memory") is not True
        or memory.get("memory_tensor_file_bytes_unchanged") is not True
        or memory.get("metadata_only_rebinding") is not True
        or memory.get("question_data_used_for_rebinding") is not False
        or memory.get("packaged_memory_tensor_file_sha256") != EXPECTED_MEMORY_FILE_SHA256
        or memory.get("source_memory_tensor_file_sha256") != EXPECTED_MEMORY_FILE_SHA256
    ):
        raise ValueError("V90 release scene-memory identity changed")

    contract = _mapping(report.get("strict_input_contract"), "strict input contract")
    if (
        contract.get("shape") != [1, 738, 1536]
        or contract.get("continuous_environment_payload_tokens") != 736
        or contract.get("compiled_before_question") is not True
        or contract.get("same_exact_memory_reused_for_every_question") is not True
        or contract.get("question_derived_environmental_tokens") != 0
        or contract.get("question_conditioned_environmental_readout") is not False
        or contract.get("question_dependent_retrieval") is not False
        or contract.get("environmental_text_inputs") != []
    ):
        raise ValueError("V90 release continuous-input contract changed")


def _require_default_path(selected: str | Path, expected: Path, purpose: str) -> None:
    if _rooted(selected) != _rooted(expected):
        raise ValueError(f"V90 strict demo refuses a substituted {purpose}; required={expected}")


def validate_v90_release(
    *,
    scene_id: str = SCENE_ID,
    config_path: str | Path = DEFAULT_CONFIG,
    checkpoint_path: str | Path = DEFAULT_CHECKPOINT,
    memory_path: str | Path = DEFAULT_MEMORY,
    release_report_path: str | Path = DEFAULT_RELEASE_REPORT,
    runtime_cli_source: str | Path = DEFAULT_RUNTIME_CLI_SOURCE,
    release_source: str | Path = DEFAULT_RELEASE_SOURCE,
    require_default_paths: bool = True,
) -> dict[str, Any]:
    """Authenticate the complete V90 inference release without loading a model."""

    if scene_id != SCENE_ID:
        raise ValueError("V90 strict demo accepts only opaque scene_000001")
    if require_default_paths:
        for selected, expected, purpose in (
            (config_path, DEFAULT_CONFIG, "runtime config"),
            (checkpoint_path, DEFAULT_CHECKPOINT, "checkpoint"),
            (memory_path, DEFAULT_MEMORY, "scene memory"),
            (release_report_path, DEFAULT_RELEASE_REPORT, "release report"),
            (runtime_cli_source, DEFAULT_RUNTIME_CLI_SOURCE, "runtime CLI"),
            (release_source, DEFAULT_RELEASE_SOURCE, "release module"),
        ):
            _require_default_path(selected, expected, purpose)

    config_file = _safe_file(config_path, "runtime config")
    checkpoint = _safe_exact_directory(
        checkpoint_path,
        purpose="checkpoint",
        inventory=CHECKPOINT_INVENTORY,
    )
    memory = _safe_exact_directory(
        memory_path,
        purpose="scene memory",
        inventory=MEMORY_INVENTORY,
    )
    release_report_file = _safe_file(release_report_path, "release report")
    runtime_cli = _safe_file(runtime_cli_source, "runtime CLI source")
    release_module = _safe_file(release_source, "release module source")

    config, config_sha256, config_file_sha256 = _read_runtime_config(config_file)
    configured_v90_state = _validate_config(config)
    checkpoint_sha256, checkpoint_entries = _checkpoint_fingerprint(checkpoint)
    checkpoint_metadata = _read_json(checkpoint / "runtime_metadata.json", "checkpoint metadata")
    _validate_checkpoint(checkpoint_metadata, configured_v90_state=configured_v90_state)
    memory_file_sha256 = _sha256_file(memory / "memory.safetensors")
    if memory_file_sha256 != EXPECTED_MEMORY_FILE_SHA256:
        raise ValueError("V90 scene-memory tensor bytes differ from the immutable source")
    memory_metadata = _read_json(memory / "runtime_metadata.json", "memory metadata")
    _validate_memory(
        memory_metadata,
        checkpoint_sha256=checkpoint_sha256,
        config_sha256=config_sha256,
    )
    report = _read_json(release_report_file, "release report")
    _validate_release_report(
        report,
        config_path=config_file,
        config_sha256=config_sha256,
        checkpoint_sha256=checkpoint_sha256,
        checkpoint_entries=checkpoint_entries,
    )

    return {
        "artifact": "gemma4_v90_strict_demo_preflight_v1",
        "schema_version": 90,
        "passed": True,
        "loads_model": False,
        "imports_torch": False,
        "imports_transformers": False,
        "scene_id": SCENE_ID,
        "runtime_config": str(config_file),
        "runtime_config_sha256": config_sha256,
        "runtime_config_file_sha256": config_file_sha256,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": checkpoint_sha256,
        "checkpoint_inventory": sorted(CHECKPOINT_INVENTORY),
        "scene_memory": str(memory),
        "scene_memory_inventory": sorted(MEMORY_INVENTORY),
        "scene_prefix_sha256": EXPECTED_MEMORY_PREFIX_SHA256,
        "scene_memory_file_sha256": memory_file_sha256,
        "scene_prefix_compiled_before_question": True,
        "same_exact_scene_prefix_for_every_question": True,
        "question_dependent_retrieval": False,
        "environmental_text_inputs": [],
        "frozen_lora_bank_count": len(EXPECTED_BANKS),
        "trainable_runtime_parameter_count": 0,
        "release_report": str(release_report_file),
        "runtime_cli_source_sha256": _sha256_file(runtime_cli),
        "release_module_source_sha256": _sha256_file(release_module),
        "promotion_decision": PROMOTION_DECISION,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", default=SCENE_ID)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    parser.add_argument("--scene-memory", default=str(DEFAULT_MEMORY))
    parser.add_argument("--release-report", default=str(DEFAULT_RELEASE_REPORT))
    parser.add_argument("--runtime-cli-source", default=str(DEFAULT_RUNTIME_CLI_SOURCE))
    parser.add_argument("--release-source", default=str(DEFAULT_RELEASE_SOURCE))
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        report = validate_v90_release(
            scene_id=args.scene,
            config_path=args.config,
            checkpoint_path=args.checkpoint,
            memory_path=args.scene_memory,
            release_report_path=args.release_report,
            runtime_cli_source=args.runtime_cli_source,
            release_source=args.release_source,
        )
    except (OSError, RuntimeError, TypeError, ValueError, json.JSONDecodeError) as error:
        print(f"V90 strict demo preflight refused: {error}", file=sys.stderr)
        return 2
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "DEFAULT_CHECKPOINT",
    "DEFAULT_CONFIG",
    "DEFAULT_MEMORY",
    "DEFAULT_RELEASE_REPORT",
    "DEFAULT_RELEASE_SOURCE",
    "DEFAULT_RUNTIME_CLI_SOURCE",
    "EXPECTED_BANKS",
    "SCENE_ID",
    "main",
    "validate_v90_release",
]
