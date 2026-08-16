"""Seal V35's stopped update-32 failure without runtime environment inputs.

This report-only audit reads immutable configuration and prior-authorization
bytes plus V35 checkpoint metadata, tensors, and Adam state.  It does not load
Gemma, QA rows, scene maps, oracle data, or deferred final scenes.  The audit
proves that V35 stopped at its first train-only causal gate, that every V33
tensor and every persistent block-core buffer stayed exact, and that only the
four declared block-cross matrices changed.

Passing this audit is not model selection.  It conditionally authorizes only a
bounded V36 experiment that jointly continues the exact V35 block core and the
already-present, exact-zero ``extension_v30_joint_pair_query`` upper-decoder
LoRA bank under fresh Adam state.  It cannot authorize chat promotion or
held-out final access.
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
from safetensors.torch import load_file

from semantic_3d_chat.config import PROJECT_ROOT, config_hash, load_config
from semantic_3d_chat.language.lora import tensor_state_sha256
from semantic_3d_chat.training.checkpointing import (
    RUNTIME_METADATA_FILENAME,
    TRAINING_METADATA_FILENAME,
    runtime_checkpoint_metadata,
    validate_runtime_checkpoint_metadata,
)
from semantic_3d_chat.training.train_block_cross_v35 import (
    v35_contract,
    v35_settings,
)

DEFAULT_CONFIG = Path("configs/experiments/gemma4_diverse28_block_cross_v35.yaml")
DEFAULT_CHECKPOINT_ROOT = Path(
    "data_gemma4/checkpoints/gemma4_v35_diverse28_block_cross"
)
DEFAULT_OUTPUT = Path("reports/gemma4/metrics/v35_update32_terminal_gate.json")

_CONFIG_SHA256 = "c8ddd808b2f338b9d61bcdadacbb0f679a0283d5e28a923fd25a6eab1a221485"
_CONFIG_HASH = "f845427e4163"
_V34_TERMINAL_REPORT_SHA256 = (
    "b0833a72ba5bc507178fa07cacc8cbef798fce4de94a5f85f2e402aafb46679f"
)
_V33_SOURCE_STATE_SHA256 = (
    "cb7bb3b48ace60212ee5c7f326839bf2ddd993810417de45c9a9cbc666313fe6"
)
_CORE_INITIAL_STATE_SHA256 = (
    "72ae7f492f5953e58d809b6782d559dc64669637d5d6a79ae98f3a31296a12cd"
)
_CORE_PERSISTENT_STATE_SHA256 = (
    "9533fb741e9bb14498be50a17b76e4139900b6754b1eadae8c47df6b4f0c2c31"
)
_SCHEDULE_SHA256 = "ba32db43173248987fb517069fafb961dc155f16a933822180a49110a6810ada"
_SEPARATION_REFERENCE_SHA256 = (
    "b376a53e744088bba62cceb3e956e0bcf5e81bca2becdefb38f67edb24bc5130"
)
_V36_AUTHORIZED_LORA_BANK = "extension_v30_joint_pair_query"
_V36_AUTHORIZED_LORA_STATE_SHA256 = (
    "2b1d89fbb9189ac551bf12905cf94036ebaa84696449b31c2b37b69d478fb70d"
)
_EXPECTED_SAVED_STEPS = (0, 8, 16, 24, 32)
_CORE_PARAMETER_NAMES = ("w_q", "w_k", "w_v", "w_o")
_CORE_PARAMETER_SHAPES = {
    "w_q": (1536, 256),
    "w_k": (384, 256),
    "w_v": (384, 256),
    "w_o": (256, 1536),
}
_CORE_BUFFER_NAMES = (
    "architecture_dimensions",
    "architecture_marker",
    "initialization_seed_state",
    "latent_anchors",
    "residual_scale",
    "spatial_temperature",
    "uniform_floor",
)
_CORE_STATE_SHA256_BY_STEP = {
    0: _CORE_INITIAL_STATE_SHA256,
    8: "a27ecb3e01493db9dfd481e58637388fd2a18877198319762423ada8d04410ee",
    16: "bf20b34fb7176f6a50a71cf1e61e397d9c9bea7549a23c31f2743b3ffb357b71",
    24: "b367f8c5b541f046edb489b805064e7ccc96e981b8dd06c33d2b182d65f3419e",
    32: "75af995833d9387e3eb01fb022eaade7327e44960466671123a51aa43afa4cf3",
}
_SAVED_FILE_SHA256 = {
    0: {
        "adapter.safetensors": "4da2df2f718afddc3bb39adc3bb3cebb3292f6abd8cef86c1fd1f7fff1be9be5",
        TRAINING_METADATA_FILENAME: "9810f704e1140a18a2598a1f17deebfec5c6a43ac97e55a214800530f087c52f",
        RUNTIME_METADATA_FILENAME: "8bc140c07c4cc3a3f5e22a3ea3cca3c0a468870efbb3ff712beb3d3cacb32839",
    },
    8: {
        "adapter.safetensors": "e07e05e83e83b6bb32e30eef866f8eec3c8f98e54fc75e84da5a2177ff90fc32",
        TRAINING_METADATA_FILENAME: "636224808dd308a5106c051fd17705f0ffa5a7bcef30bdb5b6af46b657f2ffe5",
        RUNTIME_METADATA_FILENAME: "e01e352192692810af34294519741b5bc62da78d1da099cffdbac16faf5e967b",
        "optimizer.pt": "541f57ea350828710a3504e404302bbd43b41bf8a8923e954650a7fd739b2311",
    },
    16: {
        "adapter.safetensors": "fcf74a3dbad63369decd8d69dcb10fdfd5d3eb9f46a08af6dc470f929e4ec7f7",
        TRAINING_METADATA_FILENAME: "8eaa911a49406c9e84012d5b6a739a9180a83bd270de4872e9fb73e08a3cbb90",
        RUNTIME_METADATA_FILENAME: "2d6f1427484474f16e1a180dac502a56531d3818701e13c702f79ca8e197a50f",
        "optimizer.pt": "09c03220278a276764e82dc45aeffc2ce73414177803fd63bcf4ed216215c322",
    },
    24: {
        "adapter.safetensors": "62b835cb28fbc163e01dabcd120ea5baf3661636a0faa19762d2122df746cc70",
        TRAINING_METADATA_FILENAME: "659a7e44a114d6cdc1b2777807dc3d49648a54e8ed04aa6947e328d8a5ad3738",
        RUNTIME_METADATA_FILENAME: "d9905ffc4b1ab5e8422c6da07a2eac4c5ae3c148ec463bba1ea30f1bd1493cf9",
        "optimizer.pt": "149ca677079d992ade558f8e1c9a673c69c9269be327c578893b46cffa224eb1",
    },
    32: {
        "adapter.safetensors": "4ecd5d9a38f4610387f96d36fca6111d2e248d206fd029e471cce0b1114afda0",
        TRAINING_METADATA_FILENAME: "0f106ecf5dccbe49ae1a15977d45610b560042d7f21a7b0b7ea0bf4ebea6af77",
        RUNTIME_METADATA_FILENAME: "fc06bd605b101ef9a64bf5e38cc83e91cdb8a9a37f8825e6e89c0e6a2ebfd7f1",
        "optimizer.pt": "add72932ce8cd8b58260068472ba0b2486d7011c283b4ce6785ae0f99b12b497",
    },
}
_EXPECTED_GATE_CHECKS = {
    "at_least_6_of_8_changed_pairs_over_1_02": False,
    "changed_selectivity_geometric_mean_at_least_1_02": False,
    "no_physical_pair_selectivity_below_0_98": True,
    "passed": False,
    "residual_rms_at_most_0_10": True,
    "train_complete_count_strictly_improved": True,
    "train_mean_margin_strictly_improved": True,
    "training_scenes_only": True,
    "unrelated_median_two_sided_within_1_02": True,
    "unrelated_p90_abs_log_within_log_1_02": True,
}
_EXPECTED_SELECTIVITY_BY_PAIR = {
    "pair_000005": 1.0000895261764526,
    "pair_000006": 1.0000594854354858,
    "pair_000007": 0.9998957514762878,
    "pair_000008": 1.0001171827316284,
    "pair_000015": 1.0002259016036987,
    "pair_000016": 0.9999871850013733,
    "pair_000017": 0.9998766779899597,
    "pair_000018": 1.0000414848327637,
}


def _resolve(path: str | Path) -> Path:
    value = Path(path)
    return value.resolve() if value.is_absolute() else (PROJECT_ROOT / value).resolve()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _mapping(value: object, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{field} must be a mapping")
    return value


def _real_file(path: Path, field: str) -> None:
    if path.is_symlink() or not path.is_file():
        raise FileNotFoundError(f"{field} must be a real file: {path}")


def _relative(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def _exact_float(observed: object, expected: float, field: str) -> float:
    if isinstance(observed, bool) or not isinstance(observed, (int, float)):
        raise TypeError(f"{field} must be numeric")
    value = float(observed)
    if not math.isfinite(value) or value != expected:
        raise ValueError(f"{field} changed: expected={expected} observed={value}")
    return value


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


def _validate_prior_authorization(config: Mapping[str, Any]) -> dict[str, str]:
    contract = v35_contract(config)
    path = contract.terminal_report
    _real_file(path, "V34 terminal authorization")
    observed = _sha256(path)
    if observed != _V34_TERMINAL_REPORT_SHA256:
        raise ValueError("V34 terminal authorization bytes changed")
    report = _mapping(json.loads(path.read_text(encoding="utf-8")), "V34 report")
    expected = {
        "artifact": "v34_update32_terminal_gate",
        "passed": True,
        "conditional_v35_block_cross_residual_authorized": True,
        "v34_chat_promotion_eligible": False,
        "final_test_scenes_touched": False,
        "oracle_loaded": False,
    }
    if any(report.get(key) != value for key, value in expected.items()):
        raise ValueError("Pinned V34 report no longer authorizes only V35")
    authorization = _mapping(
        report.get("conditional_authorization"), "V34 conditional authorization"
    )
    if dict(authorization) != {
        "all_other_followup_architectures_authorized": False,
        "authorized": True,
        "chat_promotion_authorized": False,
        "final_test_access_authorized": False,
        "scope": "exact_zero_block_token_cross_residual_only",
        "stage": "v35_block_cross_residual",
    }:
        raise ValueError("Pinned V34 report authorizes a different successor")
    return {"path": _relative(path), "sha256": observed}


def _validate_checkpoint_sequence(root: Path) -> tuple[Path, ...]:
    if root.is_symlink() or not root.is_dir():
        raise FileNotFoundError(f"V35 checkpoint root must be a real directory: {root}")
    observed = sorted(path.name for path in root.iterdir() if path.name.startswith("update_"))
    expected = [f"update_{step:03d}" for step in _EXPECTED_SAVED_STEPS]
    if observed != expected:
        raise ValueError(
            "V35 must be stopped at its contiguous update-32 early gate: "
            f"observed={observed} expected={expected}"
        )
    paths = tuple(root / name for name in expected)
    for step, path in zip(_EXPECTED_SAVED_STEPS, paths, strict=True):
        if path.is_symlink() or not path.is_dir():
            raise ValueError(f"V35 saved checkpoint must be a real directory: {path}")
        expected_files = set(_SAVED_FILE_SHA256[step])
        observed_files = {child.name for child in path.iterdir()}
        if observed_files != expected_files:
            raise ValueError(
                f"V35 update {step} file inventory changed: "
                f"observed={sorted(observed_files)} expected={sorted(expected_files)}"
            )
        for filename, expected_sha in _SAVED_FILE_SHA256[step].items():
            candidate = path / filename
            _real_file(candidate, f"V35 update {step} {filename}")
            observed_sha = _sha256(candidate)
            if observed_sha != expected_sha:
                raise ValueError(
                    f"V35 update {step} {filename} hash changed: "
                    f"expected={expected_sha} observed={observed_sha}"
                )
    return paths


def _validate_optimizer(path: Path, expected_step: int) -> dict[str, Any]:
    state = torch.load(path / "optimizer.pt", map_location="cpu", weights_only=True)
    if not isinstance(state, Mapping):
        raise TypeError("V35 optimizer checkpoint must be a mapping")
    groups = state.get("param_groups")
    values = state.get("state")
    if not isinstance(groups, list) or not isinstance(values, Mapping) or len(groups) != 2:
        raise ValueError("V35 optimizer must retain exactly two AdamW groups")
    by_name = {str(group.get("name")): group for group in groups if isinstance(group, Mapping)}
    if set(by_name) != {"block_cross_residual.qkv", "block_cross_residual.output"}:
        raise ValueError("V35 optimizer group names changed")
    expected_group = {
        "block_cross_residual.qkv": ([0, 1, 2], 1e-4),
        "block_cross_residual.output": ([3], 2.5e-5),
    }
    for name, (parameter_ids, learning_rate) in expected_group.items():
        group = by_name[name]
        if group.get("params") != parameter_ids:
            raise ValueError(f"V35 optimizer parameter ordering changed for {name}")
        if float(group.get("lr", math.nan)) != learning_rate:
            raise ValueError(f"V35 optimizer learning rate changed for {name}")
        if float(group.get("weight_decay", math.nan)) != 0.0:
            raise ValueError(f"V35 optimizer weight decay changed for {name}")
        if tuple(group.get("betas", ())) != (0.9, 0.999):
            raise ValueError(f"V35 Adam betas changed for {name}")
        if float(group.get("eps", math.nan)) != 1e-8:
            raise ValueError(f"V35 Adam epsilon changed for {name}")
    expected_shapes = {
        0: _CORE_PARAMETER_SHAPES["w_q"],
        1: _CORE_PARAMETER_SHAPES["w_k"],
        2: _CORE_PARAMETER_SHAPES["w_v"],
        3: _CORE_PARAMETER_SHAPES["w_o"],
    }
    if set(values) != set(expected_shapes):
        raise ValueError("V35 saved Adam state must cover exactly four core matrices")
    observed_steps: dict[str, int] = {}
    for parameter_id, expected_shape in expected_shapes.items():
        entry = _mapping(values[parameter_id], f"V35 Adam parameter {parameter_id}")
        if set(entry) != {"step", "exp_avg", "exp_avg_sq"}:
            raise ValueError("V35 Adam parameter-state fields changed")
        step = entry["step"]
        if not isinstance(step, torch.Tensor) or step.numel() != 1:
            raise TypeError("V35 Adam step must be a scalar tensor")
        expected_adam_step = expected_step if parameter_id == 3 else expected_step - 1
        if int(step.item()) != expected_adam_step:
            raise ValueError(
                f"V35 Adam step changed for parameter {parameter_id}: "
                f"expected={expected_adam_step} observed={int(step.item())}"
            )
        observed_steps[str(parameter_id)] = expected_adam_step
        for field in ("exp_avg", "exp_avg_sq"):
            value = entry[field]
            if (
                not isinstance(value, torch.Tensor)
                or tuple(value.shape) != expected_shape
                or value.dtype != torch.float32
                or not torch.isfinite(value).all()
            ):
                raise ValueError(
                    f"V35 Adam {field} is invalid for parameter {parameter_id}"
                )
        if bool((entry["exp_avg_sq"] < 0).any()):
            raise ValueError("V35 Adam second moment contains a negative value")
    return {
        "optimizer_step": expected_step,
        "qkv_adam_step": expected_step - 1,
        "output_adam_step": expected_step,
        "parameter_state_count": 4,
        "parameter_steps_by_optimizer_id": observed_steps,
        "group_parameter_ids": {
            name: list(parameter_ids)
            for name, (parameter_ids, _learning_rate) in expected_group.items()
        },
        "all_moments_finite": True,
        "fresh_staged_adam_progression_verified": True,
    }


def _validate_stage_boundary(stage: Mapping[str, Any], step: int) -> None:
    surface = _mapping(stage.get("trainable_surface"), f"V35 update-{step} surface")
    schedule = _mapping(stage.get("schedule"), f"V35 update-{step} schedule")
    cache = _mapping(stage.get("scene_cache"), f"V35 update-{step} scene cache")
    qa = _mapping(stage.get("train_qa_dataset"), f"V35 update-{step} QA audit")
    zero = _mapping(stage.get("update_zero_equivalence"), f"V35 update-{step} identity")
    if (
        stage.get("artifact") != "v35_diverse28_block_cross_training"
        or stage.get("optimizer_step") != step
        or stage.get("exact_trainable_parameter_count") != 983_040
        or stage.get("source_optimizer_step") != 64
        or stage.get("source_v33_tensor_state_sha256") != _V33_SOURCE_STATE_SHA256
        or stage.get("frozen_block_cross_source_stack_state_sha256")
        != _V33_SOURCE_STATE_SHA256
        or stage.get("gemma_decoder_frozen") is not True
        or stage.get("all_lora_banks_frozen") is not True
        or stage.get("complete_v33_stack_frozen") is not True
        or stage.get("validation_qa_loaded") is not False
        or stage.get("oracle_environment_files_loaded") is not False
        or stage.get("deferred_final_scene_ids_loaded") != []
        or stage.get("question_dependent_scene_processing") is not False
        or stage.get("question_dependent_retrieval") is not False
        or stage.get("development_progress_is_not_chat_promotion") is not True
        or stage.get("independent_selector_required") is not True
        or stage.get("separation_reference_sha256") != _SEPARATION_REFERENCE_SHA256
        or stage.get("separation_unique_changed_pair_count") != 8
        or stage.get("separation_unrelated_pair_count") != 112
        or schedule.get("schedule_sha256") != _SCHEDULE_SHA256
        or schedule.get("optimizer_step_count") != 100
        or schedule.get("pair_unit_count") != 25
        or schedule.get("exact_pair_unit_recurrence") != 4
        or schedule.get("questions_or_answers_serialized_to_runtime") is not False
        or surface.get("parameter_names") != list(_CORE_PARAMETER_NAMES)
        or surface.get("total_parameter_count") != 983_040
        or surface.get("group_parameter_counts") != {"output": 393_216, "qkv": 589_824}
        or surface.get("gemma_decoder_frozen") is not True
        or surface.get("all_lora_banks_frozen") is not True
        or surface.get("complete_v33_stack_frozen") is not True
        or surface.get("every_other_parameter_frozen") is not True
        or cache.get("scene_count") != 22
        or cache.get("all_voxels_covered") is not True
        or cache.get("all_occupied_blocks_processed") is not True
        or cache.get("all_block_tokens_cached") is not True
        or cache.get("all_repeated_normalized_block_positions_cached") is not True
        or cache.get("question_inputs_to_scene_cache") is not False
        or cache.get("answer_inputs_to_scene_cache") is not False
        or cache.get("oracle_environment_files_loaded") is not False
        or cache.get("validation_qa_loaded") is not False
        or qa.get("train_question_count") != 384
        or qa.get("train_changed_pair_unit_count") != 25
        or qa.get("validation_qa_loaded") is not False
        or qa.get("deferred_final_qa_loaded") is not False
        or qa.get("oracle_environment_files_loaded") is not False
        or zero.get("exact_v33_update64_source_tensors") is not True
        or zero.get("exact_zero_residual_identity") is not True
        or zero.get("exact_v33_update64_source_prefixes_all_22_scenes") is not True
        or zero.get("source_prefix_scene_count") != 22
        or zero.get("source_tensor_state_sha256") != _V33_SOURCE_STATE_SHA256
        or zero.get("validation_qa_loaded") is not False
        or zero.get("oracle_environment_files_loaded") is not False
    ):
        raise ValueError(f"V35 update {step} violates its frozen/data boundary")
    expected_gate: object = None if step < 32 else _EXPECTED_GATE_CHECKS
    if stage.get("update32_train_only_gate") != expected_gate:
        raise ValueError(f"V35 update {step} update-32 gate evidence changed")
    if stage.get("update64_train_only_gate") is not None:
        raise ValueError(f"V35 update {step} unexpectedly reached its update-64 gate")


def _validate_saved_metadata(
    paths: tuple[Path, ...], config: Mapping[str, Any]
) -> tuple[dict[int, dict[str, Any]], dict[str, Any]]:
    expected_config_hash = config_hash(dict(config))
    if expected_config_hash != _CONFIG_HASH:
        raise ValueError("V35 normalized config hash changed")
    metadata_by_step: dict[int, dict[str, Any]] = {}
    optimizer_audits: dict[str, Any] = {}
    for step, path in zip(_EXPECTED_SAVED_STEPS, paths, strict=True):
        metadata_raw = json.loads(
            (path / TRAINING_METADATA_FILENAME).read_text(encoding="utf-8")
        )
        metadata = dict(_mapping(metadata_raw, f"V35 update-{step} metadata"))
        runtime_raw = json.loads(
            (path / RUNTIME_METADATA_FILENAME).read_text(encoding="utf-8")
        )
        runtime = _mapping(runtime_raw, f"V35 update-{step} runtime metadata")
        validate_runtime_checkpoint_metadata(runtime)
        if runtime != runtime_checkpoint_metadata(metadata):
            raise ValueError(f"V35 update {step} runtime metadata is not freshly sanitized")
        if metadata.get("config_hash") != expected_config_hash:
            raise ValueError(f"V35 update {step} metadata config hash changed")
        if metadata.get("optimizer_step") != step or metadata.get("epoch") != step:
            raise ValueError(f"V35 update {step} metadata optimizer step changed")
        if metadata.get("frozen_block_cross_source_stack_state_sha256") != (
            _V33_SOURCE_STATE_SHA256
        ):
            raise ValueError(f"V35 update {step} top-level frozen hash changed")
        if metadata.get("block_cross_residual_state_sha256") != (
            _CORE_STATE_SHA256_BY_STEP[step]
        ):
            raise ValueError(f"V35 update {step} core-state hash changed")
        if metadata.get("block_cross_residual_initial_state_sha256") != (
            _CORE_INITIAL_STATE_SHA256
        ):
            raise ValueError(f"V35 update {step} core initial-state hash changed")
        history = metadata.get("history")
        if not isinstance(history, list) or len(history) != step + 1:
            raise ValueError(f"V35 update {step} history must contain update zero through {step}")
        if [row.get("optimizer_update") for row in history if isinstance(row, Mapping)] != list(
            range(step + 1)
        ):
            raise ValueError(f"V35 update {step} history is not contiguous")
        for index, row_raw in enumerate(history):
            row = _mapping(row_raw, f"V35 update-{step} history row {index}")
            if row.get("validation_qa_loaded") is not False:
                raise ValueError("V35 history claims validation QA was loaded")
            if index:
                if row.get("true_optimizer_step") is not True:
                    raise ValueError("V35 history does not prove true optimizer steps")
                expected_saved = index % 8 == 0
                if row.get("saved_checkpoint") is not expected_saved:
                    raise ValueError("V35 saved-checkpoint history markers changed")
        stage = _mapping(metadata.get("v35_block_cross"), f"V35 update-{step} stage")
        _validate_stage_boundary(stage, step)
        expected_terminal = {
            "path": str(_resolve("reports/gemma4/metrics/v34_update32_terminal_gate.json")),
            "sha256": _V34_TERMINAL_REPORT_SHA256,
        }
        if stage.get("conditional_v34_terminal_gate") != expected_terminal:
            raise ValueError(f"V35 update {step} V34 authorization provenance changed")
        metadata_by_step[step] = metadata
        if step:
            optimizer_audits[str(step)] = _validate_optimizer(path, step)
    return metadata_by_step, optimizer_audits


def _core_state(state: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    prefix = "block_cross_residual."
    return {
        name.removeprefix(prefix): value
        for name, value in state.items()
        if name.startswith(prefix)
    }


def _validate_tensor_transition(
    paths: tuple[Path, ...], metadata_by_step: Mapping[int, Mapping[str, Any]]
) -> dict[str, Any]:
    initial = load_file(paths[0] / "adapter.safetensors", device="cpu")
    initial_core = _core_state(initial)
    expected_core_names = set(_CORE_PARAMETER_NAMES) | set(_CORE_BUFFER_NAMES)
    if set(initial_core) != expected_core_names:
        raise ValueError("V35 update-zero block-core state inventory changed")
    inherited_names = sorted(set(initial) - {f"block_cross_residual.{name}" for name in initial_core})
    inherited_initial = {name: initial[name] for name in inherited_names}
    if tensor_state_sha256(inherited_initial) != _V33_SOURCE_STATE_SHA256:
        raise ValueError("V35 update-zero inherited state is not exact V33 update 64")
    lora_prefix = f"lora_banks.{_V36_AUTHORIZED_LORA_BANK}."
    authorized_v36_lora = {
        name.removeprefix(lora_prefix): value
        for name, value in inherited_initial.items()
        if name.startswith(lora_prefix)
    }
    expected_lora_names = {
        f"adapters.{index}.lora_{suffix}"
        for index in range(4)
        for suffix in ("a", "b")
    }
    if (
        set(authorized_v36_lora) != expected_lora_names
        or sum(int(value.numel()) for value in authorized_v36_lora.values()) != 131_072
        or tensor_state_sha256(authorized_v36_lora) != _V36_AUTHORIZED_LORA_STATE_SHA256
        or any(
            bool(torch.count_nonzero(value))
            for name, value in authorized_v36_lora.items()
            if name.endswith(".lora_b")
        )
    ):
        raise ValueError("V35 source does not retain the exact-zero authorized V36 LoRA bank")
    if tensor_state_sha256(initial_core) != _CORE_INITIAL_STATE_SHA256:
        raise ValueError("V35 update-zero core is not its deterministic exact-zero state")
    persistent_initial = {name: initial_core[name] for name in _CORE_BUFFER_NAMES}
    if tensor_state_sha256(persistent_initial) != _CORE_PERSISTENT_STATE_SHA256:
        raise ValueError("V35 update-zero persistent block-core state changed")
    if bool(torch.count_nonzero(initial_core["w_o"])):
        raise ValueError("V35 update-zero output projection is not exact zero")
    for name, shape in _CORE_PARAMETER_SHAPES.items():
        value = initial_core[name]
        if tuple(value.shape) != shape or value.dtype != torch.float32:
            raise ValueError(f"V35 core matrix {name} shape/dtype changed")
    if sum(int(initial_core[name].numel()) for name in _CORE_PARAMETER_NAMES) != 983_040:
        raise ValueError("V35 core trainable parameter count changed")

    hashes_by_step: dict[str, str] = {}
    changed_by_step: dict[str, list[str]] = {}
    terminal: Mapping[str, torch.Tensor] | None = None
    authorized = {f"block_cross_residual.{name}" for name in _CORE_PARAMETER_NAMES}
    for step, path in zip(_EXPECTED_SAVED_STEPS, paths, strict=True):
        state = load_file(path / "adapter.safetensors", device="cpu")
        if set(state) != set(initial):
            raise ValueError(f"V35 update {step} tensor inventory differs from update zero")
        core = _core_state(state)
        observed_core_hash = tensor_state_sha256(core)
        if observed_core_hash != _CORE_STATE_SHA256_BY_STEP[step]:
            raise ValueError(f"V35 update {step} core tensor-state hash changed")
        if metadata_by_step[step].get("block_cross_residual_state_sha256") != observed_core_hash:
            raise ValueError(f"V35 update {step} metadata/tensor core hash mismatch")
        inherited = {name: state[name] for name in inherited_names}
        if tensor_state_sha256(inherited) != _V33_SOURCE_STATE_SHA256 or any(
            not torch.equal(state[name], initial[name]) for name in inherited_names
        ):
            raise ValueError(f"V35 update {step} changed an inherited V33 tensor")
        persistent = {name: core[name] for name in _CORE_BUFFER_NAMES}
        if tensor_state_sha256(persistent) != _CORE_PERSISTENT_STATE_SHA256 or any(
            not torch.equal(core[name], persistent_initial[name]) for name in _CORE_BUFFER_NAMES
        ):
            raise ValueError(f"V35 update {step} changed a persistent block-core buffer")
        if any(not torch.isfinite(core[name]).all() for name in _CORE_PARAMETER_NAMES):
            raise ValueError(f"V35 update {step} core matrix contains NaN or infinity")
        changed = sorted(name for name in state if not torch.equal(initial[name], state[name]))
        if set(changed) - authorized:
            raise ValueError(f"V35 update {step} changed unauthorized tensors: {changed}")
        expected_changed = [] if step == 0 else sorted(authorized)
        if changed != expected_changed:
            raise ValueError(
                f"V35 update {step} changed-matrix set differs from its exact surface"
            )
        hashes_by_step[str(step)] = observed_core_hash
        changed_by_step[str(step)] = changed
        terminal = state
    if terminal is None:
        raise RuntimeError("V35 tensor transition has no terminal state")
    terminal_changed = changed_by_step["32"]
    changed_parameter_count = sum(int(terminal[name].numel()) for name in terminal_changed)
    if changed_parameter_count != 983_040:
        raise ValueError("V35 terminal changed parameter count is not exact")
    return {
        "authorized_changed_tensor_names": sorted(authorized),
        "terminal_changed_tensor_names": terminal_changed,
        "terminal_changed_tensor_count": len(terminal_changed),
        "terminal_changed_parameter_count": changed_parameter_count,
        "inherited_v33_tensor_count": len(inherited_names),
        "inherited_v33_tensor_state_sha256": _V33_SOURCE_STATE_SHA256,
        "all_inherited_v33_tensors_bit_exact_at_every_saved_arm": True,
        "persistent_block_core_buffer_names": list(_CORE_BUFFER_NAMES),
        "persistent_block_core_buffer_count": len(_CORE_BUFFER_NAMES),
        "persistent_block_core_buffer_element_count": sum(
            int(value.numel()) for value in persistent_initial.values()
        ),
        "persistent_block_core_state_sha256": _CORE_PERSISTENT_STATE_SHA256,
        "all_persistent_block_core_buffers_bit_exact_at_every_saved_arm": True,
        "core_state_sha256_by_optimizer_step": hashes_by_step,
        "changed_tensor_names_by_optimizer_step": changed_by_step,
        "only_declared_block_core_matrices_changed": True,
        "authorized_v36_lora_source": {
            "bank": _V36_AUTHORIZED_LORA_BANK,
            "state_sha256": _V36_AUTHORIZED_LORA_STATE_SHA256,
            "tensor_count": len(authorized_v36_lora),
            "parameter_count": 131_072,
            "rank": 8,
            "alpha": 16.0,
            "target_language_layers": [18, 19, 20, 21],
            "target_module_suffixes": ["self_attn.q_proj"],
            "all_output_matrices_exact_zero": True,
        },
    }


def _terminal_gate_evidence(metadata: Mapping[str, Any]) -> dict[str, Any]:
    stage = _mapping(metadata.get("v35_block_cross"), "V35 terminal stage")
    history = metadata.get("history")
    if not isinstance(history, list) or len(history) != 33:
        raise ValueError("V35 terminal history must end exactly at update 32")
    baseline = _mapping(history[0].get("train_pair_metrics"), "V35 baseline pair metrics")
    terminal = _mapping(history[32], "V35 terminal history row")
    separation = _mapping(
        terminal.get("training_prefix_separation"), "V35 terminal separation"
    )
    residual = _mapping(
        terminal.get("training_residual_diagnostics"), "V35 terminal residual"
    )
    pair = _mapping(terminal.get("training_pair_metrics"), "V35 terminal pair metrics")
    gate = _mapping(terminal.get("update32_train_only_gate"), "V35 terminal gate")
    if dict(gate) != _EXPECTED_GATE_CHECKS or stage.get("update32_train_only_gate") != dict(gate):
        raise ValueError("V35 update-32 causal gate outcome changed")
    if separation.get("changed_selectivity_ratios_by_pair") != _EXPECTED_SELECTIVITY_BY_PAIR:
        raise ValueError("V35 update-32 per-pair selectivity changed")
    exact_values = {
        "changed_selectivity_ratio_geometric_mean": (
            separation.get("changed_selectivity_ratio_geometric_mean"),
            1.000036597251892,
        ),
        "changed_selectivity_ratio_minimum": (
            separation.get("changed_selectivity_ratio_minimum"),
            0.9998766779899597,
        ),
        "unrelated_ratio_median": (
            separation.get("unrelated_ratio_median"),
            0.9999830722808838,
        ),
        "unrelated_abs_log_ratio_p90": (
            separation.get("unrelated_abs_log_ratio_p90"),
            0.00010669205221347511,
        ),
        "residual_rms": (residual.get("aggregate_rms"), 0.0049283793196082115),
        "baseline_mean_margin": (baseline.get("mean_margin"), 1.2638574838638306),
        "terminal_mean_margin": (pair.get("mean_margin"), 1.32265043258667),
    }
    parsed = {
        field: _exact_float(observed, expected, field)
        for field, (observed, expected) in exact_values.items()
    }
    exact_ints = {
        "changed_selectivity_over_1_02_count": (
            separation.get("changed_selectivity_over_1_02_count"),
            0,
        ),
        "baseline_complete_units": (baseline.get("complete_units"), 8),
        "terminal_complete_units": (pair.get("complete_units"), 9),
        "baseline_cross_prefix_complete_units": (
            baseline.get("cross_prefix_complete_units"),
            16,
        ),
        "terminal_cross_prefix_complete_units": (
            pair.get("cross_prefix_complete_units"),
            15,
        ),
    }
    for field, (observed, expected) in exact_ints.items():
        if isinstance(observed, bool) or observed != expected:
            raise ValueError(f"V35 {field} changed: expected={expected} observed={observed}")
        parsed[field] = int(observed)
    expected_family = {"book_support": 0, "mirror_lr": 2, "picture_support": 0}
    if baseline.get("complete_units_by_family") != expected_family:
        raise ValueError("V35 baseline family completions changed")
    if pair.get("complete_units_by_family") != expected_family:
        raise ValueError("V35 update-32 family completions changed")
    if (
        separation.get("unique_changed_physical_pair_count") != 8
        or separation.get("all_nonchanged_train_scene_pair_count") != 112
        or separation.get("question_or_answer_text_used") is not False
        or separation.get("oracle_environment_inputs_used") is not False
        or separation.get("validation_scenes_used") is not False
        or residual.get("scene_count") != 16
        or pair.get("training_scenes_only") is not True
        or pair.get("validation_qa_loaded") is not False
    ):
        raise ValueError("V35 terminal metrics do not retain the train-only boundary")
    return {
        **parsed,
        "complete_units_by_family_at_update32": expected_family,
        "changed_selectivity_ratios_by_pair": dict(_EXPECTED_SELECTIVITY_BY_PAIR),
        "required_selectivity_ratio": 1.02,
        "required_changed_pair_coverage": 6,
        "checks": dict(gate),
        "training_scenes_only": True,
        "passed": False,
    }


def audit_v35_update32(
    config_path: Path = DEFAULT_CONFIG,
    checkpoint_root: Path = DEFAULT_CHECKPOINT_ROOT,
) -> dict[str, Any]:
    """Return V35's immutable terminal failure evidence or fail closed."""

    config_file = _resolve(config_path)
    root = _resolve(checkpoint_root)
    _real_file(config_file, "V35 config")
    observed_config_sha = _sha256(config_file)
    if observed_config_sha != _CONFIG_SHA256:
        raise ValueError("V35 config bytes differ from the stopped update-32 experiment")
    config = load_config(config_file)
    contract = v35_contract(config)
    settings = v35_settings(config)
    if contract.saved_optimizer_steps[:5] != _EXPECTED_SAVED_STEPS:
        raise ValueError("V35 configured saved-step prefix changed")
    if settings.optimizer_steps != 100:
        raise ValueError("V35 configured bounded horizon changed")
    prior = _validate_prior_authorization(config)
    paths = _validate_checkpoint_sequence(root)
    metadata_by_step, optimizer_audits = _validate_saved_metadata(paths, config)
    transition = _validate_tensor_transition(paths, metadata_by_step)
    gate = _terminal_gate_evidence(metadata_by_step[32])

    loaded_files = [config_file, contract.terminal_report]
    for step, path in zip(_EXPECTED_SAVED_STEPS, paths, strict=True):
        loaded_files.extend(path / filename for filename in _SAVED_FILE_SHA256[step])
    loaded_inventory = [_relative(path) for path in loaded_files]

    return {
        "schema_version": 1,
        "artifact": "v35_update32_terminal_gate",
        "audit_method": "config_prior_report_checkpoint_metadata_optimizer_and_tensors_only",
        "gemma_loaded": False,
        "scene_maps_loaded": False,
        "qa_loaded": False,
        "oracle_loaded": False,
        "final_test_scenes_touched": False,
        "loaded_file_inventory": loaded_inventory,
        "loaded_files_confined_to_declared_report_only_inputs": True,
        "config": {"path": _relative(config_file), "sha256": observed_config_sha},
        "v34_terminal_gate": prior,
        "v33_source_tensor_state_sha256": _V33_SOURCE_STATE_SHA256,
        "checkpoint_root": _relative(root),
        "observed_saved_optimizer_steps": list(_EXPECTED_SAVED_STEPS),
        "stopped_at_optimizer_step": 32,
        "no_update_040_or_later": True,
        "configured_bounded_optimizer_horizon": 100,
        "true_optimizer_steps_completed": 32,
        "saved_file_sha256_by_optimizer_step": {
            str(step): dict(files) for step, files in _SAVED_FILE_SHA256.items()
        },
        "schedule_sha256": _SCHEDULE_SHA256,
        "separation_reference_sha256": _SEPARATION_REFERENCE_SHA256,
        "optimizer_transition": {
            "saved_optimizer_states": optimizer_audits,
            "all_saved_adam_states_exact_and_finite": True,
            "step_one_output_only_progression_proven": True,
            "qkv_started_at_optimizer_step_two": True,
        },
        "tensor_transition": transition,
        "update32_gate_evidence": gate,
        "v35_development_selection_ran": False,
        "v35_development_selection_passed": False,
        "v35_chat_promotion_eligible": False,
        "conditional_authorization": {
            "authorized": True,
            "stage": "v36_joint_block_cross_upper_lora",
            "scope": (
                "exact_v35_update32_block_cross_plus_existing_exact_zero_"
                "extension_v30_joint_pair_query_joint_only"
            ),
            "source_checkpoint": _relative(paths[-1]),
            "v35_block_cross_matrices_may_continue_training": True,
            "authorized_existing_lora_bank": "extension_v30_joint_pair_query",
            "authorized_existing_lora_state_sha256": (
                _V36_AUTHORIZED_LORA_STATE_SHA256
            ),
            "authorized_existing_lora_parameter_count": 131_072,
            "authorized_existing_lora_rank": 8,
            "authorized_existing_lora_alpha": 16.0,
            "authorized_existing_lora_dropout": 0.0,
            "authorized_existing_lora_target_language_layers": [18, 19, 20, 21],
            "authorized_existing_lora_target_module_suffixes": ["self_attn.q_proj"],
            "authorized_existing_lora_output_matrices_are_exact_zero": True,
            "fresh_adam_state_required": True,
            "optimizer_updates_1_through_8": "authorized_existing_lora_bank_only",
            "optimizer_updates_9_through_100": (
                "authorized_existing_lora_bank_plus_v35_block_cross_matrices"
            ),
            "new_lora_bank_authorized": False,
            "all_non_authorized_inherited_v33_tensors_frozen": True,
            "all_other_preexisting_lora_banks_frozen": True,
            "all_other_followup_architectures_authorized": False,
            "chat_promotion_authorized": False,
            "final_test_access_authorized": False,
        },
        "conditional_v36_joint_upper_lora_authorized": True,
        "passed": True,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--checkpoint-root", type=Path, default=DEFAULT_CHECKPOINT_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    report = audit_v35_update32(args.config, args.checkpoint_root)
    _atomic_json(_resolve(args.output), report)
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["audit_v35_update32"]
