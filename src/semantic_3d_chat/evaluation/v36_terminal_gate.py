"""Seal V36's stopped update-16 failure using immutable bytes and tensors only.

This report-only audit never constructs Gemma and never opens QA, scene maps,
oracle metadata, or deferred final-scene artifacts.  It pins the complete V36
saved-arm envelope, the exact V35 source, sanitized runtime metadata, staged
fresh-Adam state, and every adapter tensor.  It then independently calls the
locked V36 continuation predicate on the recorded train-only diagnostics and
requires the same failed update-16 result.

Passing this audit is not model selection and does not make V36 eligible for
chat.  It conditionally authorizes only the exact bounded V37 scene-ingress KV
readout experiment recorded in ``conditional_authorization``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
import yaml
from safetensors.torch import load_file

from semantic_3d_chat.config import PROJECT_ROOT, config_hash, load_config
from semantic_3d_chat.language.lora import tensor_state_sha256
from semantic_3d_chat.training.checkpointing import (
    RUNTIME_METADATA_FILENAME,
    TRAINING_METADATA_FILENAME,
    runtime_checkpoint_metadata,
    validate_runtime_checkpoint_metadata,
)
from semantic_3d_chat.training.train_joint_block_cross_v36 import (
    _optimizer_step_audit,
    complete_physical_pair_coverage,
    v36_contract,
    v36_settings,
    v36_update16_gate,
)

DEFAULT_CONFIG = Path("configs/experiments/gemma4_diverse28_joint_block_cross_v36.yaml")
DEFAULT_CHECKPOINT_ROOT = Path(
    "data_gemma4/checkpoints/gemma4_v36_diverse28_joint_block_cross"
)
DEFAULT_OUTPUT = Path("reports/gemma4/metrics/v36_update16_terminal_gate.json")

_CONFIG_SHA256 = "d684da6451b54de3c17af9a7bd5bc2bf6756ecc064cbe54c5e19d53f82f326a1"
_CONFIG_HASH = "691928da10ab"
_CONFIG_CHAIN_SHA256 = "9e6918ebbb798be4a008b8899c11f99ed5ce5a0daa17507dac659eb7663724bb"
_V35_TERMINAL_SHA256 = "88205d018de14fc0518fe695bf7420c44ac832a1ee95eea0e2ae1f41deff4a27"
_V35_SOURCE_TENSOR_SHA256 = "1fe8f278460faeb1e13d9da09051a497965a566565c79a4f6ea28c56a9120326"
_V33_INHERITED_SHA256 = "cb7bb3b48ace60212ee5c7f326839bf2ddd993810417de45c9a9cbc666313fe6"
_SOURCE_CORE_SHA256 = "75af995833d9387e3eb01fb022eaade7327e44960466671123a51aa43afa4cf3"
_SOURCE_BANK_SHA256 = "2b1d89fbb9189ac551bf12905cf94036ebaa84696449b31c2b37b69d478fb70d"
_FROZEN_SHA256 = "b394d502f0c32a694c2d1a448cdf3849c47efc4058cb1f1331fe4a97d381b1dc"
_SCHEDULE_SHA256 = "ba32db43173248987fb517069fafb961dc155f16a933822180a49110a6810ada"
_SEPARATION_SHA256 = "3a70f9e883d3f5687896d99d154a85333d1396cd1547e3f4551b40ac904da47f"
_PROTECTED_ARTIFACT = Path(
    "reports/gemma4/metrics/"
    "training_selection_gemma4_color_mirror_full_vocab_v11_resume36.json"
)
_PROTECTED_SHA256 = "c0086f66edbb8854a7938e09c57535bfd47100adbaf3b3c95eeb4b08014ce2f8"
_SOURCE_FILES = {
    "adapter.safetensors": "4ecd5d9a38f4610387f96d36fca6111d2e248d206fd029e471cce0b1114afda0",
    TRAINING_METADATA_FILENAME: "0f106ecf5dccbe49ae1a15977d45610b560042d7f21a7b0b7ea0bf4ebea6af77",
    RUNTIME_METADATA_FILENAME: "fc06bd605b101ef9a64bf5e38cc83e91cdb8a9a37f8825e6e89c0e6a2ebfd7f1",
    "optimizer.pt": "add72932ce8cd8b58260068472ba0b2486d7011c283b4ce6785ae0f99b12b497",
}
_SAVED_STEPS = (0, 8, 16)
_SAVED_FILES = {
    0: {
        "adapter.safetensors": "4ecd5d9a38f4610387f96d36fca6111d2e248d206fd029e471cce0b1114afda0",
        TRAINING_METADATA_FILENAME: "444870f37fb38ddef6131cf20a3119b116e154cfe033aa763ec817b2578761e6",
        RUNTIME_METADATA_FILENAME: "eb4d804fd13faec27461121e467eef7636ce6e8a14b060f0c5479fb6c8dbc33c",
    },
    8: {
        "adapter.safetensors": "c374292011ced79f67df5861d56fc034407eed20a2a1933f20b00bbb6d6ea8b9",
        TRAINING_METADATA_FILENAME: "9f3d29270afa8493036e1ec953be0285e2e101c18a339731c45d3682c75d8ff1",
        RUNTIME_METADATA_FILENAME: "f80d19e5d2f87885929f784c0dc313c08965f0694499ffd08205218f611fd448",
        "optimizer.pt": "4caab0a9721b052f6239c336b9b96153d356abbddd83368fb560cb5eb6dcec85",
    },
    16: {
        "adapter.safetensors": "6ed86fb51502f7330c75cc48b9be970eb0a933eb19da971a7e04726c419c3be5",
        TRAINING_METADATA_FILENAME: "7e7c257a1e42d20b7f2270a0257969ae006c3c27859e707c18d21b5537a89342",
        RUNTIME_METADATA_FILENAME: "63a27773e5d127c063b762cf110c1ed1d4022908bd9e4b843509dc399fe7f6dc",
        "optimizer.pt": "51a76712d87f24af793a28848d743034b9229d5e1df63d02c81e13efb5f12569",
    },
}
_FULL_TENSOR_SHA256 = {
    0: _V35_SOURCE_TENSOR_SHA256,
    8: "958c508fb9a8a59e8943bfb28fb276b43b39feccd89dabd908e196103d717676",
    16: "e9b6d1362d58f34aede04817b0c8d81320c616dcd4b64e9c0d3bbe56b5835dd7",
}
_CORE_SHA256 = {
    0: _SOURCE_CORE_SHA256,
    8: _SOURCE_CORE_SHA256,
    16: "92652fd2dbde2406227503f50717b2031baa1bcbc050902a379ddb9ddb52764f",
}
_BANK_SHA256 = {
    0: _SOURCE_BANK_SHA256,
    8: "d3e57789591194600a3e287f2fd22eea80797ae2452b35b3865ceb99a1551de7",
    16: "050706c300e6fb0ac8e4cc02e26c565b54a9a89505104302d4ffcedc02124c64",
}
_DYNAMIC_STACK_SHA256 = {
    0: _V33_INHERITED_SHA256,
    8: "3f13d915c670d8f58bf956783766346696f33c0e26cffda77b9ddb57ad62fca3",
    16: "9b5b89dde717278329cb95a99874f2e478d1641d9851b29ea851f3635a5ab5b9",
}
_CORE_PREFIX = "block_cross_residual."
_CORE_PARAMETERS = ("w_q", "w_k", "w_v", "w_o")
_CORE_BUFFERS = (
    "architecture_dimensions",
    "architecture_marker",
    "initialization_seed_state",
    "latent_anchors",
    "residual_scale",
    "spatial_temperature",
    "uniform_floor",
)
_BANK_NAME = "extension_v30_joint_pair_query"
_BANK_PREFIX = f"lora_banks.{_BANK_NAME}."
_V37_BANK_NAME = "extension_v23_shared_kv"
_V37_BANK_PREFIX = f"lora_banks.{_V37_BANK_NAME}."
_V37_BANK_SHA256 = "91a9eea577cab5a37e840cdf4007722a398415846af91280713bcb2cda0f045c"
_V37_FROZEN_COMPLEMENT_SHA256 = (
    "c82b8715aebcb775a6e23cb5cd477520922682b5f41929017f4f91917eafe061"
)
_BANK_NAMES = tuple(
    f"{_BANK_PREFIX}adapters.{index}.lora_{side}"
    for index in range(4)
    for side in ("a", "b")
)
_CORE_PARAMETER_NAMES = tuple(f"{_CORE_PREFIX}{name}" for name in _CORE_PARAMETERS)
_AUTHORIZED = frozenset((*_CORE_PARAMETER_NAMES, *_BANK_NAMES))
_EXPECTED_GATE = {
    "complete_physical_pair_coverage_at_least_5": False,
    "decoder_bank_state_changed": True,
    "frozen_nonauthorized_state_exact": True,
    "mean_cross_prefix_margin_strictly_above_v35_source": True,
    "passed": False,
    "residual_rms_at_most_0_075": True,
    "teacher_complete_units_at_least_10": False,
    "teacher_cross_complete_units_at_least_16": True,
    "teacher_positive_sides_at_least_34": True,
    "training_scenes_only": True,
    "unchanged_broad_nll_within_1_02x_source": True,
    "validation_qa_loaded": False,
}


def _resolve(path: str | Path) -> Path:
    value = Path(path)
    return value.resolve() if value.is_absolute() else (PROJECT_ROOT / value).resolve()


def _relative(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _real_file(path: Path, field: str) -> None:
    if path.is_symlink() or not path.is_file():
        raise FileNotFoundError(f"{field} must be a real file: {path}")


def _mapping(value: object, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{field} must be a mapping")
    return value


def _exact_float(value: object, expected: float, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{field} must be numeric")
    result = float(value)
    if not math.isfinite(result) or result != expected:
        raise ValueError(f"{field} changed: expected={expected} observed={result}")
    return result


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f"{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def _config_chain(path: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    seen: set[Path] = set()
    current = path
    while True:
        current = current.resolve()
        _real_file(current, "V36 config provenance")
        if current in seen or PROJECT_ROOT not in current.parents:
            raise ValueError("V36 config inheritance is cyclic or leaves the project")
        seen.add(current)
        content = current.read_bytes()
        rows.append({"path": _relative(current), "sha256": hashlib.sha256(content).hexdigest()})
        raw = yaml.safe_load(content)
        if not isinstance(raw, Mapping):
            raise TypeError("V36 config-chain member must be a mapping")
        base = raw.get("_base_")
        if base is None:
            break
        current = current.parent / str(base)
    encoded = json.dumps(rows, sort_keys=True, separators=(",", ":")).encode()
    if len(rows) != 34 or hashlib.sha256(encoded).hexdigest() != _CONFIG_CHAIN_SHA256:
        raise ValueError("V36 recursive config byte chain changed")
    return rows


def _validate_prior_and_source(config: Mapping[str, Any]) -> tuple[dict[str, Any], list[Path]]:
    contract = v36_contract(config)
    terminal = contract.v35_terminal_report
    _real_file(terminal, "V35 terminal report")
    if _sha256(terminal) != _V35_TERMINAL_SHA256:
        raise ValueError("V35 terminal authorization bytes changed")
    report = _mapping(json.loads(terminal.read_text(encoding="utf-8")), "V35 report")
    authorization = _mapping(report.get("conditional_authorization"), "V35 authorization")
    if (
        report.get("artifact") != "v35_update32_terminal_gate"
        or report.get("passed") is not True
        or report.get("conditional_v36_joint_upper_lora_authorized") is not True
        or authorization.get("authorized") is not True
        or authorization.get("stage") != "v36_joint_block_cross_upper_lora"
        or authorization.get("all_other_followup_architectures_authorized") is not False
        or authorization.get("chat_promotion_authorized") is not False
        or authorization.get("final_test_access_authorized") is not False
    ):
        raise ValueError("Pinned V35 report no longer authorizes exact V36 only")

    source = contract.source_checkpoint
    if source.is_symlink() or not source.is_dir():
        raise FileNotFoundError("V36 exact V35 source checkpoint is absent or aliased")
    if {path.name for path in source.iterdir()} != set(_SOURCE_FILES):
        raise ValueError("Exact V35 source file inventory changed")
    loaded = [terminal]
    for name, expected in _SOURCE_FILES.items():
        path = source / name
        _real_file(path, f"V35 source {name}")
        if _sha256(path) != expected:
            raise ValueError(f"V35 source hash changed: {name}")
        loaded.append(path)
    source_tensors = load_file(source / "adapter.safetensors", device="cpu")
    if tensor_state_sha256(source_tensors) != _V35_SOURCE_TENSOR_SHA256:
        raise ValueError("V35 source tensor-state hash changed")
    return {
        "terminal_report": {"path": _relative(terminal), "sha256": _V35_TERMINAL_SHA256},
        "source_checkpoint": _relative(source),
        "source_file_sha256": dict(_SOURCE_FILES),
        "source_tensor_state_sha256": _V35_SOURCE_TENSOR_SHA256,
    }, loaded


def _checkpoint_paths(root: Path) -> tuple[Path, ...]:
    if root.is_symlink() or not root.is_dir():
        raise FileNotFoundError(f"V36 checkpoint root must be a real directory: {root}")
    observed = sorted(path.name for path in root.iterdir() if path.name.startswith("update_"))
    expected = [f"update_{step:03d}" for step in _SAVED_STEPS]
    if observed != expected:
        raise ValueError(
            "V36 must be stopped at its contiguous update-16 gate: "
            f"observed={observed} expected={expected}"
        )
    result = tuple(root / name for name in expected)
    for step, path in zip(_SAVED_STEPS, result, strict=True):
        if path.is_symlink() or not path.is_dir():
            raise ValueError(f"V36 checkpoint arm is absent or aliased: {path}")
        inventory = {child.name for child in path.iterdir()}
        if inventory != set(_SAVED_FILES[step]):
            raise ValueError(f"V36 update {step} file inventory changed")
        for name, expected_sha in _SAVED_FILES[step].items():
            candidate = path / name
            _real_file(candidate, f"V36 update {step} {name}")
            if _sha256(candidate) != expected_sha:
                raise ValueError(f"V36 update {step} hash changed: {name}")
    return result


def _validate_stage(
    metadata: Mapping[str, Any], step: int, *, prior_history: list[Mapping[str, Any]]
) -> tuple[list[Mapping[str, Any]], Mapping[str, Any]]:
    stage = _mapping(metadata.get("v36_joint_block_cross"), "V36 training stage")
    surface = _mapping(stage.get("trainable_surface"), "V36 trainable surface")
    schedule = _mapping(stage.get("schedule"), "V36 schedule")
    qa = _mapping(stage.get("train_qa_dataset"), "V36 training QA audit")
    cache = _mapping(stage.get("scene_cache"), "V36 scene cache audit")
    expected_stage = "lora_only" if step <= 8 else "joint_full"
    if (
        stage.get("artifact") != "v36_diverse28_joint_block_cross_training"
        or stage.get("optimizer_step") != step
        or stage.get("source_optimizer_step") != 32
        or stage.get("source_v35_tensor_state_sha256") != _V35_SOURCE_TENSOR_SHA256
        or stage.get("inherited_v33_tensor_state_sha256") != _V33_INHERITED_SHA256
        or stage.get("source_block_core_state_sha256") != _SOURCE_CORE_SHA256
        or stage.get("decoder_bank_initial_state_sha256") != _SOURCE_BANK_SHA256
        or stage.get("frozen_nonauthorized_state_sha256") != _FROZEN_SHA256
        or stage.get("source_v35_optimizer_state_loaded") is not False
        or stage.get("fresh_adam") is not True
        or stage.get("optimizer_stage_updates_1_through_8") != "lora_only"
        or stage.get("optimizer_stage_updates_9_through_100") != "joint_core_and_lora"
        or stage.get("validation_qa_loaded") is not False
        or stage.get("oracle_environment_files_loaded") is not False
        or stage.get("deferred_final_scene_ids_loaded") != []
        or stage.get("question_dependent_scene_processing") is not False
        or stage.get("question_dependent_retrieval") is not False
        or stage.get("development_progress_is_not_chat_promotion") is not True
        or stage.get("independent_selector_required") is not True
        or stage.get("separation_reference_sha256") != _SEPARATION_SHA256
        or schedule.get("schedule_sha256") != _SCHEDULE_SHA256
        or schedule.get("optimizer_step_count") != 100
        or schedule.get("pair_unit_count") != 25
        or schedule.get("exact_pair_unit_recurrence") != 4
        or schedule.get("questions_or_answers_serialized_to_runtime") is not False
        or surface.get("active_stage") != expected_stage
        or surface.get("block_core_parameter_count") != 983_040
        or surface.get("decoder_bank_parameter_count") != 131_072
        or surface.get("total_parameter_count") != 1_114_112
        or surface.get("gemma_base_frozen") is not True
        or surface.get("all_other_lora_banks_frozen") is not True
        or surface.get("complete_v33_scene_stack_frozen") is not True
        or surface.get("every_other_parameter_frozen") is not True
        or qa.get("train_question_count") != 384
        or qa.get("train_changed_pair_unit_count") != 25
        or qa.get("validation_qa_loaded") is not False
        or qa.get("deferred_final_qa_loaded") is not False
        or qa.get("oracle_environment_files_loaded") is not False
        or cache.get("scene_count") != 22
        or cache.get("all_voxels_covered") is not True
        or cache.get("all_occupied_blocks_processed") is not True
        or cache.get("question_inputs_to_scene_cache") is not False
        or cache.get("answer_inputs_to_scene_cache") is not False
        or cache.get("validation_qa_loaded") is not False
        or cache.get("oracle_environment_files_loaded") is not False
    ):
        raise ValueError(f"V36 update {step} violates its locked source/data boundary")
    if stage.get("conditional_v35_terminal_gate") != {
        "path": str(_resolve("reports/gemma4/metrics/v35_update32_terminal_gate.json")),
        "sha256": _V35_TERMINAL_SHA256,
    }:
        raise ValueError("V36 prior authorization provenance changed")
    qa_names = [Path(str(path)).name for path in qa.get("loaded_files", [])]
    if qa_names != ["splits.json", "train.jsonl"] or "validation.jsonl" in qa_names:
        raise ValueError("V36 loaded a non-training QA file")
    allowed_scenes = {
        *(f"scene_{value:06d}" for value in range(11, 25)),
        *(f"scene_{value:06d}" for value in range(31, 39)),
    }
    if set(cache.get("scene_ids", [])) != allowed_scenes:
        raise ValueError("V36 scene cache crossed its declared question-free scene envelope")
    map_paths = cache.get("loaded_environment_files")
    if not isinstance(map_paths, list) or {
        Path(str(path)).parent.name for path in map_paths
    } != allowed_scenes:
        raise ValueError("V36 cached map provenance changed")
    if any(f"scene_{value:06d}" in str(path) for value in range(25, 31) for path in map_paths):
        raise ValueError("V36 cached a deferred final scene")

    history = metadata.get("history")
    if not isinstance(history, list) or len(history) != step + 1:
        raise ValueError(f"V36 update {step} history is incomplete")
    if [row.get("optimizer_update") for row in history] != list(range(step + 1)):
        raise ValueError("V36 history is not one row per true optimizer update")
    if prior_history and history[: len(prior_history)] != prior_history:
        raise ValueError("V36 history was rewritten between saved arms")
    for index, raw in enumerate(history):
        row = _mapping(raw, f"V36 history row {index}")
        if (
            row.get("validation_qa_loaded") is not False
            or row.get("oracle_environment_files_loaded") is not False
            or (index and row.get("true_optimizer_step") is not True)
            or row.get("saved_checkpoint") is not (index % 8 == 0)
        ):
            raise ValueError("V36 history crossed its train-only true-step boundary")
    expected_gate: object = None if step < 16 else _EXPECTED_GATE
    if stage.get("update16_train_only_gate") != expected_gate:
        raise ValueError(f"V36 update {step} update-16 gate evidence changed")
    if stage.get("update32_train_only_gate") is not None or stage.get(
        "update64_train_only_gate"
    ) is not None:
        raise ValueError("V36 unexpectedly reached a later continuation gate")
    return list(history), stage


def _validate_metadata(
    paths: tuple[Path, ...], config: Mapping[str, Any]
) -> tuple[dict[int, dict[str, Any]], dict[str, Any]]:
    if config_hash(dict(config)) != _CONFIG_HASH:
        raise ValueError("V36 normalized config hash changed")
    result: dict[int, dict[str, Any]] = {}
    optimizer: dict[str, Any] = {}
    prior_history: list[Mapping[str, Any]] = []
    for step, path in zip(_SAVED_STEPS, paths, strict=True):
        metadata = dict(
            _mapping(
                json.loads((path / TRAINING_METADATA_FILENAME).read_text(encoding="utf-8")),
                f"V36 update-{step} metadata",
            )
        )
        if (
            metadata.get("optimizer_step") != step
            or metadata.get("epoch") != step
            or metadata.get("config_hash") != _CONFIG_HASH
            or metadata.get("block_cross_residual_state_sha256") != _CORE_SHA256[step]
            or metadata.get("frozen_block_cross_source_stack_state_sha256")
            != _DYNAMIC_STACK_SHA256[step]
            or _mapping(metadata.get("lora_bank_state_sha256"), "LoRA hashes").get(
                _BANK_NAME
            )
            != _BANK_SHA256[step]
        ):
            raise ValueError(f"V36 update {step} top-level provenance changed")
        prior_history, stage = _validate_stage(
            metadata, step, prior_history=prior_history
        )
        if stage.get("current_block_source_stack_state_sha256") != _DYNAMIC_STACK_SHA256[step]:
            raise ValueError("V36 dynamic source-stack hash is stale")
        runtime = _mapping(
            json.loads((path / RUNTIME_METADATA_FILENAME).read_text(encoding="utf-8")),
            "V36 runtime metadata",
        )
        validate_runtime_checkpoint_metadata(runtime)
        if dict(runtime) != runtime_checkpoint_metadata(metadata):
            raise ValueError(f"V36 update {step} runtime metadata is not freshly sanitized")
        result[step] = metadata
        if step:
            optimizer[str(step)] = _optimizer_step_audit(path, expected_step=step)
    return result, optimizer


def _core_state(state: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {
        name.removeprefix(_CORE_PREFIX): tensor
        for name, tensor in state.items()
        if name.startswith(_CORE_PREFIX)
    }


def _bank_state(state: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {
        name.removeprefix(_BANK_PREFIX): tensor
        for name, tensor in state.items()
        if name.startswith(_BANK_PREFIX)
    }


def _validate_tensors(paths: tuple[Path, ...]) -> dict[str, Any]:
    initial = load_file(paths[0] / "adapter.safetensors", device="cpu")
    source = load_file(
        _resolve("data_gemma4/checkpoints/gemma4_v35_diverse28_block_cross/update_032/")
        / "adapter.safetensors",
        device="cpu",
    )
    if set(initial) != set(source) or any(
        not torch.equal(initial[name], source[name]) for name in initial
    ):
        raise ValueError("V36 update zero is not tensor-bit-exact V35 update 32")
    if len(initial) != 179 or sum(int(value.numel()) for value in initial.values()) != 13_985_676:
        raise ValueError("V36 adapter tensor inventory changed")
    core0 = _core_state(initial)
    bank0 = _bank_state(initial)
    if set(core0) != {*_CORE_PARAMETERS, *_CORE_BUFFERS} or {
        f"{_BANK_PREFIX}{name}" for name in bank0
    } != set(_BANK_NAMES):
        raise ValueError("V36 core or query-bank inventory changed")
    if tensor_state_sha256(core0) != _SOURCE_CORE_SHA256:
        raise ValueError("V36 source block core changed")
    if tensor_state_sha256(bank0) != _SOURCE_BANK_SHA256:
        raise ValueError("V36 source query bank changed")
    if any(
        bool(torch.count_nonzero(value))
        for name, value in bank0.items()
        if name.endswith("lora_b")
    ):
        raise ValueError("V36 source query bank is not exact-zero output")

    frozen0 = {name: value for name, value in initial.items() if name not in _AUTHORIZED}
    buffers0 = {name: core0[name] for name in _CORE_BUFFERS}
    hashes: dict[str, dict[str, str]] = {}
    changed: dict[str, list[str]] = {}
    for step, path in zip(_SAVED_STEPS, paths, strict=True):
        state = load_file(path / "adapter.safetensors", device="cpu")
        if set(state) != set(initial):
            raise ValueError(f"V36 update {step} tensor inventory changed")
        core = _core_state(state)
        bank = _bank_state(state)
        frozen = {name: value for name, value in state.items() if name not in _AUTHORIZED}
        dynamic = {name: value for name, value in state.items() if not name.startswith(_CORE_PREFIX)}
        observed = {
            "full_tensor_state_sha256": tensor_state_sha256(state),
            "block_core_state_sha256": tensor_state_sha256(core),
            "decoder_bank_state_sha256": tensor_state_sha256(bank),
            "frozen_nonauthorized_state_sha256": tensor_state_sha256(frozen),
            "dynamic_source_stack_state_sha256": tensor_state_sha256(dynamic),
        }
        expected = {
            "full_tensor_state_sha256": _FULL_TENSOR_SHA256[step],
            "block_core_state_sha256": _CORE_SHA256[step],
            "decoder_bank_state_sha256": _BANK_SHA256[step],
            "frozen_nonauthorized_state_sha256": _FROZEN_SHA256,
            "dynamic_source_stack_state_sha256": _DYNAMIC_STACK_SHA256[step],
        }
        if observed != expected:
            raise ValueError(f"V36 update {step} tensor-state hashes changed")
        if any(not torch.equal(frozen[name], frozen0[name]) for name in frozen0):
            raise ValueError(f"V36 update {step} changed a frozen tensor or core buffer")
        if any(not torch.isfinite(state[name]).all() for name in _AUTHORIZED):
            raise ValueError(f"V36 update {step} contains a nonfinite authorized tensor")
        names = sorted(name for name in state if not torch.equal(state[name], initial[name]))
        expected_names = [] if step == 0 else sorted(_BANK_NAMES)
        if step == 16:
            expected_names = sorted(_AUTHORIZED)
        if names != expected_names:
            raise ValueError(f"V36 update {step} changed outside its staged surface")
        hashes[str(step)] = observed
        changed[str(step)] = names
    v37_bank = {
        name.removeprefix(_V37_BANK_PREFIX): value
        for name, value in state.items()
        if name.startswith(_V37_BANK_PREFIX)
    }
    v37_frozen = {
        name: value for name, value in state.items() if not name.startswith(_V37_BANK_PREFIX)
    }
    expected_v37_shapes = {
        "adapters.0.lora_a": (4, 1536),
        "adapters.0.lora_b": (256, 4),
        "adapters.1.lora_a": (4, 1536),
        "adapters.1.lora_b": (256, 4),
        "adapters.2.lora_a": (4, 1536),
        "adapters.2.lora_b": (512, 4),
        "adapters.3.lora_a": (4, 1536),
        "adapters.3.lora_b": (512, 4),
    }
    if (
        set(v37_bank) != set(expected_v37_shapes)
        or any(tuple(v37_bank[name].shape) != shape for name, shape in expected_v37_shapes.items())
        or sum(int(value.numel()) for value in v37_bank.values()) != 30_720
        or tensor_state_sha256(v37_bank) != _V37_BANK_SHA256
        or tensor_state_sha256(v37_frozen) != _V37_FROZEN_COMPLEMENT_SHA256
    ):
        raise ValueError("V36 terminal does not contain the exact authorized V37 source surface")
    return {
        "adapter_tensor_count": 179,
        "adapter_tensor_element_count": 13_985_676,
        "source_update_zero_bit_exact_v35_update32": True,
        "source_v35_tensor_state_sha256": _V35_SOURCE_TENSOR_SHA256,
        "state_sha256_by_optimizer_step": hashes,
        "changed_tensor_names_by_optimizer_step": changed,
        "update8_only_eight_query_lora_tensors_changed": True,
        "update16_only_twelve_authorized_tensors_changed": True,
        "terminal_changed_tensor_count": 12,
        "terminal_changed_parameter_count": 1_114_112,
        "frozen_nonauthorized_tensor_count": len(frozen0),
        "frozen_nonauthorized_tensor_element_count": sum(
            int(value.numel()) for value in frozen0.values()
        ),
        "frozen_nonauthorized_state_sha256": _FROZEN_SHA256,
        "all_frozen_nonauthorized_tensors_bit_exact_at_every_arm": True,
        "persistent_block_core_buffer_names": list(_CORE_BUFFERS),
        "persistent_block_core_buffer_element_count": sum(
            int(value.numel()) for value in buffers0.values()
        ),
        "all_persistent_block_core_buffers_bit_exact_at_every_arm": True,
        "all_authorized_tensors_finite": True,
        "conditional_v37_existing_shared_kv_source": {
            "bank": _V37_BANK_NAME,
            "tensor_count": 8,
            "parameter_count": 30_720,
            "state_sha256": _V37_BANK_SHA256,
            "frozen_complement_tensor_count": len(v37_frozen),
            "frozen_complement_state_sha256": _V37_FROZEN_COMPLEMENT_SHA256,
            "all_tensors_finite": all(torch.isfinite(value).all() for value in v37_bank.values()),
        },
    }


def _gate_evidence(
    metadata: Mapping[str, Any], config: Mapping[str, Any]
) -> dict[str, Any]:
    history = metadata.get("history")
    if not isinstance(history, list) or len(history) != 17:
        raise ValueError("V36 terminal history must end exactly at update 16")
    stage = _mapping(metadata.get("v36_joint_block_cross"), "V36 terminal stage")
    row = _mapping(history[16], "V36 history[16]")
    pair = _mapping(row.get("training_pair_metrics"), "V36 update-16 pair metrics")
    residual = _mapping(row.get("training_residual_diagnostics"), "V36 residual metrics")
    source_nll = _exact_float(
        stage.get("source_broad_train_nll"), 2.967046543955803, "source broad NLL"
    )
    broad_nll = _exact_float(
        row.get("training_broad_nll"), 2.915099874138832, "update-16 broad NLL"
    )
    residual_rms = _exact_float(
        residual.get("aggregate_rms"), 0.008839325979351997, "update-16 residual RMS"
    )
    expected_ints = {
        "complete_units": 9,
        "cross_prefix_complete_units": 16,
        "positive_sides": 34,
    }
    if any(pair.get(field) != value for field, value in expected_ints.items()):
        raise ValueError("V36 update-16 pair summary changed")
    margin = _exact_float(
        pair.get("mean_cross_prefix_margin"),
        1.4565558433532715,
        "update-16 mean cross-prefix margin",
    )
    family = {"book_support": 0, "mirror_lr": 2, "picture_support": 0}
    if pair.get("complete_units_by_family") != family:
        raise ValueError("V36 update-16 priority-family evidence changed")
    coverage = complete_physical_pair_coverage(pair)
    if coverage != 4:
        raise ValueError("V36 update-16 physical-pair coverage changed")
    contract = v36_contract(config)
    replayed = v36_update16_gate(
        pair_metrics=pair,
        broad_nll=broad_nll,
        source_broad_nll=source_nll,
        residual_rms=residual_rms,
        decoder_bank_state_sha256=_BANK_SHA256[16],
        frozen_nonauthorized_state_sha256=_FROZEN_SHA256,
        contract=contract,
    )
    if (
        replayed != _EXPECTED_GATE
        or row.get("update16_train_only_gate") != replayed
        or stage.get("update16_train_only_gate") != replayed
    ):
        raise ValueError("V36 update-16 train-only gate replay changed")
    if pair.get("training_scenes_only") is not True or pair.get(
        "validation_qa_loaded"
    ) is not False:
        raise ValueError("V36 terminal pair metrics crossed the training boundary")
    return {
        "teacher_complete_units": 9,
        "teacher_cross_prefix_complete_units": 16,
        "teacher_positive_sides": 34,
        "mean_cross_prefix_margin": margin,
        "complete_physical_pair_coverage": coverage,
        "complete_units_by_family": family,
        "source_broad_train_nll": source_nll,
        "update16_broad_train_nll": broad_nll,
        "broad_train_nll_ratio_to_source": broad_nll / source_nll,
        "residual_rms": residual_rms,
        "decoder_bank_state_sha256": _BANK_SHA256[16],
        "frozen_nonauthorized_state_sha256": _FROZEN_SHA256,
        "failed_requirements": [
            "teacher_complete_units_at_least_10",
            "complete_physical_pair_coverage_at_least_5",
        ],
        "checks": dict(replayed),
        "training_scenes_only": True,
        "validation_qa_loaded": False,
        "passed": False,
    }


def _v37_authorization(terminal_path: Path) -> dict[str, Any]:
    return {
        "authorized": True,
        "stage": "v37_scene_ingress_kv",
        "scope": "continue_existing_extension_v23_shared_kv_only",
        "source_checkpoint": _relative(terminal_path),
        "source_file_sha256": dict(_SAVED_FILES[16]),
        "source_full_tensor_state_sha256": _FULL_TENSOR_SHA256[16],
        "source_learned_block_core_state_sha256": _CORE_SHA256[16],
        "source_learned_v30_query_bank_state_sha256": _BANK_SHA256[16],
        "source_frozen_nonauthorized_state_sha256": _FROZEN_SHA256,
        "source_existing_shared_kv_bank_state_sha256": _V37_BANK_SHA256,
        "v37_frozen_complement_state_sha256": _V37_FROZEN_COMPLEMENT_SHA256,
        "source_v36_block_core_frozen_for_all_updates": True,
        "source_v36_query_bank_frozen_for_all_updates": True,
        "authorized_existing_lora_bank": "extension_v23_shared_kv",
        "authorized_existing_lora_rank": 4,
        "authorized_existing_lora_alpha": 8.0,
        "authorized_existing_lora_dropout": 0.0,
        "authorized_existing_lora_tensor_count": 8,
        "authorized_existing_lora_parameter_count": 30_720,
        "authorized_existing_lora_target_language_layers": [13, 14],
        "authorized_existing_lora_target_module_paths": [
            "model.language_model.layers.13.self_attn.k_proj",
            "model.language_model.layers.13.self_attn.v_proj",
            "model.language_model.layers.14.self_attn.k_proj",
            "model.language_model.layers.14.self_attn.v_proj",
        ],
        "authorized_existing_lora_parameter_shapes": {
            "layer13_k_proj": {"lora_a": [4, 1536], "lora_b": [256, 4]},
            "layer13_v_proj": {"lora_a": [4, 1536], "lora_b": [256, 4]},
            "layer14_k_proj": {"lora_a": [4, 1536], "lora_b": [512, 4]},
            "layer14_v_proj": {"lora_a": [4, 1536], "lora_b": [512, 4]},
        },
        "new_lora_bank_authorized": False,
        "existing_shared_kv_bank_reinitialization_authorized": False,
        "existing_shared_kv_bank_is_learned_not_zero_output": True,
        "shared_kv_architecture_attestation": {
            "language_layer_count": 35,
            "upper_shared_kv_layer_count": 20,
            "upper_shared_kv_layer_range": [15, 34],
            "last_nonshared_sliding_kv_producer_layer": 13,
            "last_nonshared_full_kv_producer_layer": 14,
            "layers_18_through_21_have_operative_kv_projections": False,
            "duplicate_lora_target_paths_across_banks_forbidden": True,
        },
        "gemma_base_frozen": True,
        "composer_frozen": True,
        "v33_scene_stack_frozen": True,
        "all_other_inherited_lora_banks_frozen": True,
        "every_other_tensor_and_buffer_frozen": True,
        "fresh_adam_required": True,
        "v36_optimizer_state_may_be_loaded": False,
        "learning_rate": 2e-5,
        "weight_decay": 0.0,
        "gradient_clip_norm": 1.0,
        "scene_prefix_shape": [256, 1536],
        "scene_prefix_built_before_questions": True,
        "all_occupied_blocks_processed": True,
        "question_dependent_retrieval": False,
        "scene_prefix_and_block_residual_must_remain_exact_v36_u16": True,
        "source_block_residual_rms": 0.008839325979351997,
        "new_scene_encoder_module_authorized": False,
        "new_scene_tokens_authorized": False,
        "same_v36_train_only_objective_and_coefficients_required": {
            "broad_answer_nll_weight": 0.25,
            "pair_correct_answer_nll_weight": 0.5,
            "side_hinge_weight": 4.0,
            "side_hinge_margin": 0.5,
            "cross_prefix_flip_hinge_weight": 8.0,
            "cross_prefix_flip_margin": 0.25,
            "normalized_residual_penalty_weight": 0.001,
            "residual_penalty_scale": 0.05,
        },
        "same_25_changed_training_units_required": True,
        "one_deterministic_unchanged_broad_row_per_update": True,
        "updates_1_through_8": [
            {"pair_id": "pair_000015", "question_key": "cfq_13b1138d14c52a7c"},
            {"pair_id": "pair_000017", "question_key": "cfq_1c8b8cd72fcde904"},
            {"pair_id": "pair_000015", "question_key": "cfq_163eb92339ad35a5"},
            {"pair_id": "pair_000017", "question_key": "cfq_66aab89cee5bef49"},
            {"pair_id": "pair_000015", "question_key": "cfq_a1c673a1197a0961"},
            {"pair_id": "pair_000017", "question_key": "cfq_d469c4ac156ac42d"},
            {"pair_id": "pair_000015", "question_key": "cfq_ac7ac024c40aaddc"},
            {"pair_id": "pair_000017", "question_key": "cfq_fa3601dfffa80a0e"},
        ],
        "updates_9_through_58": "two_complete_deterministic_25_pair_cycles",
        "updates_59_through_64": [
            {"pair_id": "pair_000015", "question_key": "cfq_13b1138d14c52a7c"},
            {"pair_id": "pair_000017", "question_key": "cfq_1c8b8cd72fcde904"},
            {"pair_id": "pair_000015", "question_key": "cfq_163eb92339ad35a5"},
            {"pair_id": "pair_000017", "question_key": "cfq_66aab89cee5bef49"},
            {"pair_id": "pair_000015", "question_key": "cfq_a1c673a1197a0961"},
            {"pair_id": "pair_000017", "question_key": "cfq_d469c4ac156ac42d"},
        ],
        "full_schedule_sha256_must_be_pinned_before_training": True,
        "maximum_true_optimizer_steps": 64,
        "saved_optimizer_steps": [0, 8, 16, 24, 32, 40, 48, 56, 64],
        "update_zero_must_bit_replay_v36_u16_teacher_nll_residual_and_prefixes": True,
        "update16_gate": {
            "complete_units_minimum": 10,
            "complete_physical_pair_coverage_minimum": 5,
            "cross_prefix_complete_units_minimum": 16,
            "positive_sides_minimum": 35,
            "mean_cross_prefix_margin_minimum": 1.4565558433532715,
            "book_or_picture_teacher_complete_minimum": 1,
            "mirror_teacher_complete_minimum": 2,
            "book_cross_prefix_complete_minimum": 1,
            "picture_cross_prefix_complete_minimum": 2,
            "broad_train_nll_ratio_maximum": 1.02,
            "existing_bank_state_must_change": True,
            "frozen_state_must_remain_exact": True,
            "scene_prefix_and_block_residual_must_remain_exact_v36_u16": True,
        },
        "update32_gate": {
            "require_update16_passed": True,
            "complete_units_minimum": 12,
            "complete_physical_pair_coverage_minimum": 6,
            "cross_prefix_complete_units_minimum": 18,
            "positive_sides_minimum": 37,
            "mean_cross_prefix_margin_minimum": 1.4565558433532715,
            "book_teacher_complete_minimum": 1,
            "picture_teacher_complete_minimum": 1,
            "mirror_teacher_complete_minimum": 2,
            "broad_train_nll_ratio_maximum": 1.03,
            "scene_prefix_and_block_residual_must_remain_exact_v36_u16": True,
        },
        "update64_gate": {
            "require_update32_passed": True,
            "complete_units_minimum": 15,
            "complete_physical_pair_coverage_minimum": 7,
            "cross_prefix_complete_units_minimum": 20,
            "positive_sides_minimum": 40,
            "teacher_complete_each_priority_family_minimum": 1,
            "train_greedy_complete_units_minimum": 6,
            "train_greedy_complete_each_priority_family_minimum": 1,
            "broad_greedy_exact_accuracy_maximum_drop": 0.02,
            "broad_train_nll_ratio_maximum": 1.05,
            "scene_prefix_and_block_residual_must_remain_exact_v36_u16": True,
        },
        "validation_qa_or_model_selection_before_complete_update64": False,
        "oracle_access_authorized": False,
        "chat_promotion_authorized": False,
        "final_test_access_authorized": False,
        "all_other_followup_architectures_authorized": False,
    }


def audit_v36_update16(
    config_path: Path = DEFAULT_CONFIG,
    checkpoint_root: Path = DEFAULT_CHECKPOINT_ROOT,
) -> dict[str, Any]:
    """Return immutable V36 update-16 failure evidence or fail closed."""

    config_file = _resolve(config_path)
    root = _resolve(checkpoint_root)
    _real_file(config_file, "V36 config")
    if _sha256(config_file) != _CONFIG_SHA256:
        raise ValueError("V36 config bytes differ from the stopped update-16 experiment")
    config_chain = _config_chain(config_file)
    config = load_config(config_file)
    settings = v36_settings(config)
    if settings.optimizer_steps != 100 or settings.saved_optimizer_steps[:3] != _SAVED_STEPS:
        raise ValueError("V36 bounded saved-step contract changed")
    prior, source_loaded = _validate_prior_and_source(config)
    paths = _checkpoint_paths(root)
    metadata_by_step, optimizer = _validate_metadata(paths, config)
    tensors = _validate_tensors(paths)
    gate = _gate_evidence(metadata_by_step[16], config)

    protected = _resolve(_PROTECTED_ARTIFACT)
    _real_file(protected, "protected selection artifact")
    if _sha256(protected) != _PROTECTED_SHA256:
        raise ValueError("Protected selection artifact changed")

    loaded_files = [*(_resolve(row["path"]) for row in config_chain), *source_loaded]
    for step, path in zip(_SAVED_STEPS, paths, strict=True):
        loaded_files.extend(path / name for name in _SAVED_FILES[step])
    loaded_files.append(protected)
    inventory = [_relative(path) for path in loaded_files]
    forbidden_fragments = (
        "/qa/",
        "/maps/",
        "/oracle/",
        "validation.jsonl",
        "final_once",
        *(f"scene_{value:06d}" for value in range(25, 31)),
    )
    if any(fragment in path.casefold() for path in inventory for fragment in forbidden_fragments):
        raise RuntimeError("V36 terminal audit loaded a forbidden runtime/environment file")

    return {
        "schema_version": 1,
        "artifact": "v36_update16_terminal_gate",
        "audit_method": (
            "recursive_config_prior_report_exact_source_checkpoint_"
            "v36_metadata_optimizer_runtime_and_tensors_only"
        ),
        "passed": True,
        "gemma_loaded": False,
        "scene_maps_loaded": False,
        "qa_loaded": False,
        "validation_qa_loaded": False,
        "validation_model_selection_ran": False,
        "oracle_loaded": False,
        "final_test_scenes_touched": False,
        "loaded_file_inventory": inventory,
        "loaded_files_confined_to_declared_tensor_only_audit_inputs": True,
        "config": {
            "path": _relative(config_file),
            "sha256": _CONFIG_SHA256,
            "normalized_config_hash": _CONFIG_HASH,
            "recursive_chain_sha256": _CONFIG_CHAIN_SHA256,
            "recursive_chain": config_chain,
        },
        "protected_artifact": {
            "path": _relative(protected),
            "sha256": _PROTECTED_SHA256,
            "access": "bytes_hashed_only",
            "unchanged": True,
        },
        "v35_source_and_authorization": prior,
        "checkpoint_root": _relative(root),
        "observed_saved_optimizer_steps": list(_SAVED_STEPS),
        "stopped_at_optimizer_step": 16,
        "true_optimizer_steps_completed": 16,
        "configured_bounded_optimizer_horizon": 100,
        "no_update_024_or_later": True,
        "saved_file_sha256_by_optimizer_step": {
            str(step): dict(files) for step, files in _SAVED_FILES.items()
        },
        "optimizer_transition": {
            "saved_optimizer_states": optimizer,
            "update8_lora_optimizer_step": 8,
            "update8_block_core_optimizer_step": None,
            "update16_lora_optimizer_step": 16,
            "update16_block_core_optimizer_step": 8,
            "fresh_v36_adam_staging_verified": True,
            "all_saved_adam_moments_finite": True,
            "v35_optimizer_state_loaded": False,
        },
        "tensor_transition": tensors,
        "update16_gate_evidence": gate,
        "v36_train_only_continuation_gate_passed": False,
        "v36_development_selection_ran": False,
        "v36_development_selection_passed": False,
        "v36_chat_promotion_eligible": False,
        "training_data_boundary": {
            "train_qa_used_by_original_training": True,
            "validation_qa_used_by_original_training": False,
            "question_free_validation_scene_prefix_identity_cache_only": True,
            "validation_scene_questions_or_answers_used": False,
            "oracle_environment_files_used": False,
            "deferred_final_scenes_used": False,
        },
        "conditional_authorization": _v37_authorization(paths[-1]),
        "conditional_v37_scene_ingress_kv_authorized": True,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--checkpoint-root", type=Path, default=DEFAULT_CHECKPOINT_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    report = audit_v36_update16(args.config, args.checkpoint_root)
    _atomic_json(_resolve(args.output), report)
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["audit_v36_update16"]
