"""Seal V37's stopped update-16 failure using files, metadata, and tensors only.

The audit never constructs Gemma and never opens QA, scene maps, oracle data,
or deferred final-scene artifacts.  It pins the exact V36 authorization/source,
the complete V37 update-0/8/16 envelope (including optimizer integrity
manifests), the target-only tensor transition, and the independently replayed
failed train-only gate.  The only successor it conditionally authorizes is the
exact deterministic V38 query-recovery surface recorded in the report.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
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
from semantic_3d_chat.training.train_scene_ingress_kv_v37 import (
    OPTIMIZER_AUDIT_FILENAME,
    optimizer_step_audit,
    v37_contract,
    v37_update16_gate,
)

DEFAULT_CONFIG = Path("configs/experiments/gemma4_diverse28_scene_ingress_kv_v37.yaml")
DEFAULT_CHECKPOINT_ROOT = Path(
    "data_gemma4/checkpoints/gemma4_v37_diverse28_scene_ingress_kv"
)
DEFAULT_OUTPUT = Path("reports/gemma4/metrics/v37_update16_terminal_gate.json")

_CONFIG_SHA256 = "38b0ee5a0921d77c909b31a6cad3834f2527589ef43e6c3671d02ae7731fa098"
_CONFIG_HASH = "7f865546146d"
_CONFIG_CHAIN_COUNT = 35
_CONFIG_CHAIN_SHA256 = "067dd4e87cf21c5eded6ed3821c0b266cbfa6600754ef9fd5e21272e675fa93e"
_V36_TERMINAL = Path("reports/gemma4/metrics/v36_update16_terminal_gate.json")
_V36_TERMINAL_SHA256 = "cb5b1248a4904dc58a685b64e052f980c02771b59eed5578bdbf2865ddbf5877"
_V36_SOURCE = Path(
    "data_gemma4/checkpoints/gemma4_v36_diverse28_joint_block_cross/update_016"
)
_V36_SOURCE_FILES = {
    "adapter.safetensors": "6ed86fb51502f7330c75cc48b9be970eb0a933eb19da971a7e04726c419c3be5",
    TRAINING_METADATA_FILENAME: "7e7c257a1e42d20b7f2270a0257969ae006c3c27859e707c18d21b5537a89342",
    RUNTIME_METADATA_FILENAME: "63a27773e5d127c063b762cf110c1ed1d4022908bd9e4b843509dc399fe7f6dc",
    "optimizer.pt": "51a76712d87f24af793a28848d743034b9229d5e1df63d02c81e13efb5f12569",
}
_SOURCE_TENSOR_SHA256 = "e9b6d1362d58f34aede04817b0c8d81320c616dcd4b64e9c0d3bbe56b5835dd7"
_SOURCE_CORE_SHA256 = "92652fd2dbde2406227503f50717b2031baa1bcbc050902a379ddb9ddb52764f"
_SOURCE_QUERY_SHA256 = "050706c300e6fb0ac8e4cc02e26c565b54a9a89505104302d4ffcedc02124c64"
_SOURCE_TARGET_SHA256 = "91a9eea577cab5a37e840cdf4007722a398415846af91280713bcb2cda0f045c"
_FROZEN_SHA256 = "c82b8715aebcb775a6e23cb5cd477520922682b5f41929017f4f91917eafe061"
_PROTECTED_ARTIFACT = Path(
    "reports/gemma4/metrics/"
    "training_selection_gemma4_color_mirror_full_vocab_v11_resume36.json"
)
_PROTECTED_SHA256 = "c0086f66edbb8854a7938e09c57535bfd47100adbaf3b3c95eeb4b08014ce2f8"
_SCHEDULE_SHA256 = "76a123412d4bd3aeee012515b37095c22d9cbf9eb56934b622d715daca45fa2b"
_SAVED_STEPS = (0, 8, 16)
_SAVED_FILES = {
    0: {
        "adapter.safetensors": "6ed86fb51502f7330c75cc48b9be970eb0a933eb19da971a7e04726c419c3be5",
        TRAINING_METADATA_FILENAME: "c649124497006322e8fd4e90ccfe26188ff4617611298db868b3eb10f869e738",
        RUNTIME_METADATA_FILENAME: "9d0da4b2ad0304d4c6aaf00f72077ff2d48031c0e73eacbb045a65875daa4715",
    },
    8: {
        "adapter.safetensors": "22375edfae04cea6dd99af97f4657e577a9ce27822bb76de3e00aece88a1847f",
        TRAINING_METADATA_FILENAME: "6e009a99efe7162e2456eda091ddb3d373993411baa88aa2a84703746aa21c6a",
        RUNTIME_METADATA_FILENAME: "0d825414ac989f4112c2ef37a30454d9f22438dfe015b93b6373e6a443868ffa",
        "optimizer.pt": "7109aed09ec0293378e31c28f4a3b5b6dc33a30d534ebe17d8b3ad9b57385a52",
        OPTIMIZER_AUDIT_FILENAME: (
            "c0ea3decd79a4b46e7b88456bcd08d1504d61b5a5b88b9768d847f04feda6bfb"
        ),
    },
    16: {
        "adapter.safetensors": "5b897221fb448be75caf330c6ab81a39d424ecaa7d0e0307f264f3c58c7659fa",
        TRAINING_METADATA_FILENAME: "f696eeb75d08471e8676613ef6a440852df62113d90393ad705d6c85b9985ecc",
        RUNTIME_METADATA_FILENAME: "8b3f70743aa8226549e65d0d5a08882cc311b61bda07eb2ba2d49454267897de",
        "optimizer.pt": "551dda55c9aba84dbd4ac2238fa489a8451f8ad10e091853f3e758f3cbe32ed3",
        OPTIMIZER_AUDIT_FILENAME: (
            "d52170ab9983eeb44a3c264830f9507baab2abc2cc0ecf0ef55f7cf82ef3cc89"
        ),
    },
}
_FULL_TENSOR_SHA256 = {
    0: _SOURCE_TENSOR_SHA256,
    8: "da60d3af8658cea3d9fa0e1664967f37a288ca8f435a7df135b81cd303cdd67e",
    16: "801f80f6fd6a27e7cb815677824fb855c491272d65b81ba464c95200ebd570b9",
}
_TARGET_SHA256 = {
    0: _SOURCE_TARGET_SHA256,
    8: "9be2354f461d5fd03c7b28ac09fc52ac6650569ce4fb47e47d676fd57bc28343",
    16: "b48dd9606db0e83e5f6ced40e20124456b9ce2dde8a7713af9e8dddcc52a6eca",
}
_DYNAMIC_STACK_SHA256 = {
    0: "9b5b89dde717278329cb95a99874f2e478d1641d9851b29ea851f3635a5ab5b9",
    8: "4255197030029fefba626c9283ff0726eec4a1ea412bdbd2efa0b61af2e3869d",
    16: "dca107db5e445fc66ccceb925c80e891a856aec64399876cc5b5a7dc30d97db6",
}
_TARGET_BANK = "extension_v23_shared_kv"
_TARGET_PREFIX = f"lora_banks.{_TARGET_BANK}."
_QUERY_PREFIX = "lora_banks.extension_v30_joint_pair_query."
_CORE_PREFIX = "block_cross_residual."
_TARGET_NAMES = tuple(
    f"{_TARGET_PREFIX}adapters.{index}.lora_{side}"
    for index in range(4)
    for side in ("a", "b")
)
_V38_QUERY_BANK = "extension_v30_joint_pair_query"
_V38_QUERY_PREFIX = f"lora_banks.{_V38_QUERY_BANK}."
_V38_HYBRID_FULL_SHA256 = (
    "7b951c6d7ae4f7b50603159f0bc4dfb4d50b5b40f9325134d78d1de1dae87fc0"
)
_V38_HYBRID_V23_SHA256 = (
    "5c9233bd96b381e2f63443f8a739a868a21997b28061d90fb407a46d9de2d4cb"
)
_V38_FROZEN_EXCLUDING_QUERY_SHA256 = (
    "fe39da221505c1968030c67aacb4e99f1a179e05a97d2906d416afe5fef5ed78"
)
_V38_PRIORITY_UNITS = (
    ("pair_000015", "cfq_13b1138d14c52a7c"),
    ("pair_000017", "cfq_1c8b8cd72fcde904"),
    ("pair_000015", "cfq_163eb92339ad35a5"),
    ("pair_000017", "cfq_66aab89cee5bef49"),
    ("pair_000015", "cfq_a1c673a1197a0961"),
    ("pair_000017", "cfq_d469c4ac156ac42d"),
    ("pair_000015", "cfq_ac7ac024c40aaddc"),
    ("pair_000017", "cfq_fa3601dfffa80a0e"),
)
_V38_CANONICAL_UNITS = (
    ("pair_000005", "cfq_a578dc166be9a217"),
    ("pair_000006", "cfq_0a79d507273195ef"),
    ("pair_000006", "cfq_5c84a2c27d2be251"),
    ("pair_000006", "cfq_8e9855d51fa91bfb"),
    ("pair_000006", "cfq_f1e8f24afac4339b"),
    ("pair_000007", "cfq_736067b51ce93c49"),
    ("pair_000007", "cfq_997610c185204121"),
    ("pair_000007", "cfq_bd500364dac52b59"),
    ("pair_000007", "cfq_f004e89a2284b94a"),
    ("pair_000008", "cfq_39808de4310f3388"),
    ("pair_000015", "cfq_13b1138d14c52a7c"),
    ("pair_000015", "cfq_163eb92339ad35a5"),
    ("pair_000015", "cfq_a1c673a1197a0961"),
    ("pair_000015", "cfq_ac7ac024c40aaddc"),
    ("pair_000016", "cfq_699675ceeaf65406"),
    ("pair_000016", "cfq_9152e66b248e4692"),
    ("pair_000016", "cfq_e08b96268b3c2a10"),
    ("pair_000016", "cfq_efba0c82a7336161"),
    ("pair_000017", "cfq_1c8b8cd72fcde904"),
    ("pair_000017", "cfq_66aab89cee5bef49"),
    ("pair_000017", "cfq_d469c4ac156ac42d"),
    ("pair_000017", "cfq_fa3601dfffa80a0e"),
    ("pair_000018", "cfq_90b3d9852a93ce2a"),
    ("pair_000018", "cfq_b2c2e9aec9f883bc"),
    ("pair_000018", "cfq_f58d3fc750290b0f"),
)
_V38_PAIR_SCHEDULE_SHA256 = (
    "4e26160c9a36e20ac894ff5e26604e6fad985ff5c84211282fd9807c9e738b3f"
)
_TRAIN_SCENES = (
    "scene_000011",
    "scene_000012",
    "scene_000013",
    "scene_000014",
    "scene_000015",
    "scene_000016",
    "scene_000017",
    "scene_000018",
    "scene_000031",
    "scene_000032",
    "scene_000033",
    "scene_000034",
    "scene_000035",
    "scene_000036",
    "scene_000037",
    "scene_000038",
)
_EXPECTED_GATE = {
    "book_cross_prefix_complete": True,
    "book_or_picture_teacher_complete": False,
    "broad_nll_within_absolute_lock": True,
    "complete_physical_pair_coverage_at_least_5": False,
    "frozen_complement_state_exact": True,
    "mean_cross_prefix_margin_at_least_source": False,
    "mirror_teacher_complete_at_least_2": True,
    "passed": False,
    "picture_cross_prefix_complete_at_least_2": True,
    "scene_prefix_and_block_residual_exact": True,
    "target_bank_state_changed": True,
    "teacher_complete_units_at_least_10": False,
    "teacher_cross_complete_units_at_least_16": True,
    "teacher_positive_sides_at_least_35": False,
    "training_scenes_only": True,
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
        _real_file(current, "V37 config provenance")
        if current in seen or (current != PROJECT_ROOT and PROJECT_ROOT not in current.parents):
            raise ValueError("V37 config inheritance is cyclic or leaves the project")
        seen.add(current)
        content = current.read_bytes()
        rows.append({"path": _relative(current), "sha256": hashlib.sha256(content).hexdigest()})
        raw = yaml.safe_load(content)
        if not isinstance(raw, Mapping):
            raise TypeError("V37 config-chain member must be a mapping")
        base = raw.get("_base_")
        if base is None:
            break
        current = current.parent / str(base)
    encoded = json.dumps(rows, sort_keys=True, separators=(",", ":")).encode()
    if (
        len(rows) != _CONFIG_CHAIN_COUNT
        or hashlib.sha256(encoded).hexdigest() != _CONFIG_CHAIN_SHA256
    ):
        raise ValueError("V37 recursive config byte chain changed")
    return rows


def _validate_v36_source(config: Mapping[str, Any]) -> tuple[dict[str, Any], list[Path]]:
    contract = v37_contract(config)
    terminal = _resolve(_V36_TERMINAL)
    _real_file(terminal, "V36 terminal report")
    if _sha256(terminal) != _V36_TERMINAL_SHA256:
        raise ValueError("V36 terminal report bytes changed")
    report = _mapping(json.loads(terminal.read_text(encoding="utf-8")), "V36 terminal report")
    authorization = _mapping(report.get("conditional_authorization"), "V36 authorization")
    if (
        report.get("artifact") != "v36_update16_terminal_gate"
        or report.get("passed") is not True
        or report.get("conditional_v37_scene_ingress_kv_authorized") is not True
        or authorization.get("authorized") is not True
        or authorization.get("stage") != "v37_scene_ingress_kv"
        or authorization.get("source_full_tensor_state_sha256") != _SOURCE_TENSOR_SHA256
        or authorization.get("source_learned_block_core_state_sha256") != _SOURCE_CORE_SHA256
        or authorization.get("source_learned_v30_query_bank_state_sha256")
        != _SOURCE_QUERY_SHA256
        or authorization.get("source_existing_shared_kv_bank_state_sha256")
        != _SOURCE_TARGET_SHA256
        or authorization.get("v37_frozen_complement_state_sha256") != _FROZEN_SHA256
        or dict(_mapping(authorization.get("source_file_sha256"), "V36 source files"))
        != _V36_SOURCE_FILES
        or authorization.get("final_test_access_authorized") is not False
        or authorization.get("oracle_access_authorized") is not False
    ):
        raise ValueError("V36 report no longer authorizes the exact bounded V37 source")

    source = contract.source_checkpoint
    if source != _resolve(_V36_SOURCE) or source.is_symlink() or not source.is_dir():
        raise FileNotFoundError("Exact V36 update-16 source is absent or aliased")
    if {path.name for path in source.iterdir()} != set(_V36_SOURCE_FILES):
        raise ValueError("V36 source file inventory changed")
    loaded = [terminal]
    for name, expected in _V36_SOURCE_FILES.items():
        candidate = source / name
        _real_file(candidate, f"V36 source {name}")
        if _sha256(candidate) != expected:
            raise ValueError(f"V36 source hash changed: {name}")
        loaded.append(candidate)
    tensors = load_file(source / "adapter.safetensors", device="cpu")
    if tensor_state_sha256(tensors) != _SOURCE_TENSOR_SHA256:
        raise ValueError("V36 source tensor state changed")
    return {
        "terminal_report": {"path": _relative(terminal), "sha256": _V36_TERMINAL_SHA256},
        "source_checkpoint": _relative(source),
        "source_file_sha256": dict(_V36_SOURCE_FILES),
        "source_tensor_state_sha256": _SOURCE_TENSOR_SHA256,
        "source_optimizer_access": "bytes_hashed_only_not_deserialized",
    }, loaded


def _checkpoint_paths(root: Path) -> tuple[Path, ...]:
    if root.is_symlink() or not root.is_dir():
        raise FileNotFoundError(f"V37 checkpoint root must be a real directory: {root}")
    observed = sorted(path.name for path in root.iterdir() if path.name.startswith("update_"))
    expected = [f"update_{step:03d}" for step in _SAVED_STEPS]
    if observed != expected:
        raise ValueError(
            "V37 must be stopped at its contiguous update-16 gate: "
            f"observed={observed} expected={expected}"
        )
    paths = tuple(root / name for name in expected)
    for step, path in zip(_SAVED_STEPS, paths, strict=True):
        if path.is_symlink() or not path.is_dir():
            raise ValueError(f"V37 checkpoint arm is absent or aliased: {path}")
        inventory = {child.name for child in path.iterdir()}
        if inventory != set(_SAVED_FILES[step]):
            raise ValueError(f"V37 update {step} file inventory changed")
        for name, expected_sha in _SAVED_FILES[step].items():
            candidate = path / name
            _real_file(candidate, f"V37 update {step} {name}")
            if _sha256(candidate) != expected_sha:
                raise ValueError(f"V37 update {step} hash changed: {name}")
    return paths


def _validate_stage(
    metadata: Mapping[str, Any],
    *,
    step: int,
    authorization: Mapping[str, Any],
    prior_history: list[Mapping[str, Any]],
) -> tuple[list[Mapping[str, Any]], Mapping[str, Any]]:
    stage = _mapping(metadata.get("v37_scene_ingress_kv"), "V37 training stage")
    surface = _mapping(stage.get("trainable_surface"), "V37 trainable surface")
    schedule = _mapping(stage.get("schedule"), "V37 schedule")
    qa = _mapping(stage.get("train_qa_dataset"), "V37 train QA audit")
    cache = _mapping(stage.get("scene_cache"), "V37 scene cache audit")
    source_replay = _mapping(stage.get("source_replay_attestation"), "V37 source replay")
    prefix = _mapping(stage.get("prefix_replay_attestation"), "V37 prefix replay")
    if (
        stage.get("schema_version") != 1
        or stage.get("optimizer_step") != step
        or stage.get("conditional_v36_terminal_gate")
        != {"path": str(_resolve(_V36_TERMINAL)), "sha256": _V36_TERMINAL_SHA256}
        or stage.get("conditional_authorization") != authorization
        or Path(str(stage.get("source_checkpoint"))).resolve() != _resolve(_V36_SOURCE)
        or stage.get("source_file_sha256") != _V36_SOURCE_FILES
        or stage.get("source_v36_tensor_state_sha256") != _SOURCE_TENSOR_SHA256
        or stage.get("source_block_core_state_sha256") != _SOURCE_CORE_SHA256
        or stage.get("source_query_bank_state_sha256") != _SOURCE_QUERY_SHA256
        or stage.get("target_bank_source_state_sha256") != _SOURCE_TARGET_SHA256
        or stage.get("source_optimizer_state_loaded") is not False
        or stage.get("source_optimizer_file_opened") is not False
        or stage.get("fresh_adam") is not True
        or stage.get("validation_qa_loaded") is not False
        or stage.get("oracle_environment_files_loaded") is not False
        or stage.get("deferred_final_scene_ids_loaded") != []
        or stage.get("question_dependent_scene_processing") is not False
        or stage.get("question_dependent_retrieval") is not False
        or stage.get("independent_selector_required") is not True
        or stage.get("frozen_complement_state_sha256") != _FROZEN_SHA256
        or stage.get("learned_block_core_state_sha256") != _SOURCE_CORE_SHA256
        or stage.get("learned_query_bank_state_sha256") != _SOURCE_QUERY_SHA256
        or stage.get("target_bank_state_sha256") != _TARGET_SHA256[step]
        or stage.get("dynamic_block_source_stack_state_sha256") != _DYNAMIC_STACK_SHA256[step]
        or surface.get("target_bank") != _TARGET_BANK
        or surface.get("trainable_tensor_count") != 8
        or surface.get("trainable_parameter_count") != 30_720
        or surface.get("rank") != 4
        or surface.get("alpha") != 8.0
        or surface.get("dropout") != 0.0
        or surface.get("gemma_base_frozen") is not True
        or surface.get("v36_learned_block_core_frozen") is not True
        or surface.get("v36_learned_query_bank_frozen") is not True
        or surface.get("complete_scene_stack_frozen") is not True
        or surface.get("all_other_lora_banks_frozen") is not True
        or surface.get("every_other_tensor_and_buffer_frozen") is not True
        or tuple(surface.get("target_parameter_names", ())) != _TARGET_NAMES
        or schedule.get("optimizer_step_count") != 64
        or schedule.get("pair_unit_count") != 25
        or schedule.get("schedule_sha256") != _SCHEDULE_SHA256
        or schedule.get("true_optimizer_step_per_schedule_row") is not True
        or schedule.get("questions_or_answers_serialized_to_runtime") is not False
        or qa.get("train_question_count") != 384
        or qa.get("train_changed_pair_unit_count") != 25
        or qa.get("validation_qa_loaded") is not False
        or qa.get("deferred_final_qa_loaded") is not False
        or qa.get("oracle_environment_files_loaded") is not False
        or cache.get("scene_count") != 16
        or tuple(cache.get("scene_ids", ())) != _TRAIN_SCENES
        or cache.get("scene_scope") != "training_only"
        or cache.get("authenticated_manifest_scene_count") != 22
        or cache.get("authenticated_manifest_train_subset_count") != 16
        or cache.get("validation_scene_ids_loaded") != []
        or cache.get("validation_environment_maps_loaded") is not False
        or cache.get("deferred_final_scene_ids_loaded") != []
        or cache.get("validation_qa_loaded") is not False
        or cache.get("oracle_environment_files_loaded") is not False
        or source_replay.get("exact_stopped_v36_update16_loaded") is not True
        or source_replay.get("v36_optimizer_file_opened") is not False
        or source_replay.get("v36_optimizer_state_loaded") is not False
        or source_replay.get(
            "current_v36_u16_prefixes_recomputed_deterministically_from_exact_tensors"
        )
        is not True
        or source_replay.get("external_prefix_manifest_used") is not False
        or prefix.get(
            "current_v36_u16_prefixes_recomputed_deterministically_from_exact_tensors"
        )
        is not True
        or prefix.get("external_prefix_manifest_used") is not False
        or prefix.get("source_prefix_scene_count") != 16
        or tuple(prefix.get("source_prefix_scene_ids", ())) != _TRAIN_SCENES
        or prefix.get("source_prefixes_replayed_bit_exact") is not True
        or prefix.get("validation_environment_maps_loaded") is not False
        or prefix.get("validation_qa_loaded") is not False
    ):
        raise ValueError(f"V37 update {step} violates its locked source/data surface")
    loaded_qa = qa.get("loaded_files")
    if (
        not isinstance(loaded_qa, list)
        or [Path(str(path)).name for path in loaded_qa] != ["splits.json", "train.jsonl"]
    ):
        raise ValueError("V37 training QA provenance changed")
    loaded_maps = cache.get("loaded_environment_files")
    if (
        not isinstance(loaded_maps, list)
        or [Path(str(path)).parent.name for path in loaded_maps] != list(_TRAIN_SCENES)
        or any("oracle" in {part.casefold() for part in Path(str(path)).parts} for path in loaded_maps)
    ):
        raise ValueError("V37 training-map provenance changed")
    first = _mapping(prefix.get("source_prefix_sha256_by_scene"), "V37 first prefixes")
    replayed = _mapping(prefix.get("replayed_prefix_sha256_by_scene"), "V37 replayed prefixes")
    if (
        tuple(sorted(first)) != _TRAIN_SCENES
        or dict(first) != dict(replayed)
        or any(re.fullmatch(r"[0-9a-f]{64}", str(value)) is None for value in first.values())
    ):
        raise ValueError("V37 prefix recomputation evidence changed")

    history = metadata.get("history")
    if not isinstance(history, list) or len(history) != step + 1:
        raise ValueError(f"V37 update {step} history is incomplete")
    if [row.get("optimizer_update") for row in history] != list(range(step + 1)):
        raise ValueError("V37 history is not one row per optimizer update")
    if prior_history and history[: len(prior_history)] != prior_history:
        raise ValueError("V37 history was rewritten between saved arms")
    for index, raw in enumerate(history):
        row = _mapping(raw, f"V37 history row {index}")
        if (
            row.get("validation_qa_loaded") is not False
            or row.get("oracle_environment_files_loaded") is not False
            or (index and row.get("true_optimizer_step") is not True)
            or row.get("saved_checkpoint") is not (index % 8 == 0)
            or (index and row.get("optimizer_stage") != "existing_scene_ingress_kv_lora_only")
            or (index and row.get("frozen_residual_descriptive_only") is not True)
            or (index and row.get("residual_penalty_contributes_gradient") is not False)
            or row.get("frozen_complement_state_sha256") != _FROZEN_SHA256
        ):
            raise ValueError("V37 history crossed its train-only true-step boundary")
    expected_gate: object = None if step < 16 else _EXPECTED_GATE
    if stage.get("update16_train_only_gate") != expected_gate:
        raise ValueError("V37 update-16 gate evidence changed")
    if stage.get("update32_train_only_gate") is not None or stage.get(
        "update64_train_only_gate"
    ) is not None:
        raise ValueError("V37 unexpectedly reached a later continuation gate")
    return list(history), stage


def _validate_metadata(
    paths: tuple[Path, ...], config: Mapping[str, Any], authorization: Mapping[str, Any]
) -> tuple[dict[int, dict[str, Any]], dict[str, Any]]:
    if config_hash(dict(config)) != _CONFIG_HASH:
        raise ValueError("V37 normalized config hash changed")
    metadata_by_step: dict[int, dict[str, Any]] = {}
    optimizer_by_step: dict[str, Any] = {}
    prior_history: list[Mapping[str, Any]] = []
    for step, path in zip(_SAVED_STEPS, paths, strict=True):
        metadata = json.loads((path / TRAINING_METADATA_FILENAME).read_text(encoding="utf-8"))
        if metadata.get("config_hash") != _CONFIG_HASH or metadata.get("optimizer_step") != step:
            raise ValueError(f"V37 update {step} config or optimizer step changed")
        runtime = json.loads((path / RUNTIME_METADATA_FILENAME).read_text(encoding="utf-8"))
        validate_runtime_checkpoint_metadata(runtime)
        if runtime != runtime_checkpoint_metadata(metadata):
            raise ValueError(f"V37 update {step} runtime metadata is not sanitized")
        prior_history, _stage = _validate_stage(
            metadata,
            step=step,
            authorization=authorization,
            prior_history=prior_history,
        )
        tensors = load_file(path / "adapter.safetensors", device="cpu")
        if step:
            optimizer_by_step[str(step)] = optimizer_step_audit(
                path, expected_step=step, tensors=tensors
            )
        metadata_by_step[step] = metadata
    return metadata_by_step, optimizer_by_step


def _state(tensors: Mapping[str, torch.Tensor], prefix: str) -> dict[str, torch.Tensor]:
    return {
        name.removeprefix(prefix): value
        for name, value in tensors.items()
        if name.startswith(prefix)
    }


def _validate_tensors(paths: tuple[Path, ...]) -> dict[str, Any]:
    tensors = {
        step: load_file(path / "adapter.safetensors", device="cpu")
        for step, path in zip(_SAVED_STEPS, paths, strict=True)
    }
    source = tensors[0]
    states: dict[str, Any] = {}
    changed_by_step: dict[str, list[str]] = {}
    for step in _SAVED_STEPS:
        current = tensors[step]
        if set(current) != set(source) or len(current) != 179:
            raise ValueError(f"V37 update {step} tensor inventory changed")
        if any(
            current[name].shape != source[name].shape or current[name].dtype != source[name].dtype
            for name in source
        ):
            raise ValueError(f"V37 update {step} tensor shape/dtype changed")
        full_hash = tensor_state_sha256(current)
        target_hash = tensor_state_sha256(_state(current, _TARGET_PREFIX))
        query_hash = tensor_state_sha256(_state(current, _QUERY_PREFIX))
        core_hash = tensor_state_sha256(_state(current, _CORE_PREFIX))
        frozen = {name: value for name, value in current.items() if not name.startswith(_TARGET_PREFIX)}
        frozen_hash = tensor_state_sha256(frozen)
        if (
            full_hash != _FULL_TENSOR_SHA256[step]
            or target_hash != _TARGET_SHA256[step]
            or query_hash != _SOURCE_QUERY_SHA256
            or core_hash != _SOURCE_CORE_SHA256
            or frozen_hash != _FROZEN_SHA256
            or any(not torch.isfinite(value).all() for value in current.values())
        ):
            raise ValueError(f"V37 update {step} tensor-state lock changed")
        changed = sorted(name for name in current if not torch.equal(current[name], source[name]))
        if (step == 0 and changed) or (step and tuple(changed) != tuple(sorted(_TARGET_NAMES))):
            raise ValueError(f"V37 update {step} changed a non-target tensor")
        changed_by_step[str(step)] = changed
        states[str(step)] = {
            "full_tensor_state_sha256": full_hash,
            "target_bank_state_sha256": target_hash,
            "query_bank_state_sha256": query_hash,
            "block_core_state_sha256": core_hash,
            "frozen_complement_state_sha256": frozen_hash,
        }
    return {
        "adapter_tensor_count": 179,
        "target_bank": _TARGET_BANK,
        "target_tensor_count": 8,
        "target_parameter_count": 30_720,
        "frozen_complement_tensor_count": 171,
        "source_update_zero_bit_exact_v36_update16": (
            _SAVED_FILES[0]["adapter.safetensors"]
            == _V36_SOURCE_FILES["adapter.safetensors"]
            and _FULL_TENSOR_SHA256[0] == _SOURCE_TENSOR_SHA256
        ),
        "update8_changed_exactly_all_eight_target_tensors": True,
        "update16_changed_exactly_all_eight_target_tensors": True,
        "all_frozen_complement_tensors_bit_exact_at_every_arm": True,
        "learned_v36_core_bit_exact_at_every_arm": True,
        "learned_v36_query_bank_bit_exact_at_every_arm": True,
        "changed_tensor_names_by_optimizer_step": changed_by_step,
        "state_sha256_by_optimizer_step": states,
    }


def _replay_gate(
    metadata: Mapping[str, Any], config: Mapping[str, Any]
) -> dict[str, Any]:
    contract = v37_contract(config)
    stage = _mapping(metadata.get("v37_scene_ingress_kv"), "V37 terminal stage")
    row = _mapping(metadata.get("history", [None])[-1], "V37 update-16 row")
    pair = _mapping(row.get("training_pair_metrics"), "V37 update-16 pair metrics")
    broad_nll = _exact_float(row.get("training_broad_nll"), 2.9015875508387885, "broad NLL")
    if (
        pair.get("complete_units") != 9
        or pair.get("complete_physical_pair_coverage") != 4
        or pair.get("cross_prefix_complete_units") != 18
        or pair.get("positive_sides") != 33
        or pair.get("complete_units_by_family")
        != {"book_support": 0, "mirror_lr": 2, "picture_support": 0}
        or pair.get("cross_prefix_complete_units_by_family")
        != {"book_support": 2, "mirror_lr": 4, "picture_support": 3}
        or pair.get("training_scenes_only") is not True
        or pair.get("validation_qa_loaded") is not False
    ):
        raise ValueError("V37 update-16 pair gate numerics changed")
    margin = _exact_float(
        pair.get("mean_cross_prefix_margin"), 1.4349822998046875, "cross-prefix margin"
    )
    replayed = v37_update16_gate(
        pair_metrics=pair,
        broad_nll=broad_nll,
        target_bank_state_sha256=_TARGET_SHA256[16],
        frozen_complement_state_sha256=_FROZEN_SHA256,
        residual_exact=row.get("scene_prefix_and_block_residual_exact") is True,
        contract=contract,
    )
    if (
        replayed != _EXPECTED_GATE
        or row.get("update16_train_only_gate") != replayed
        or stage.get("update16_train_only_gate") != replayed
    ):
        raise ValueError("V37 update-16 train-only gate replay changed")
    return {
        "teacher_complete_units": 9,
        "complete_physical_pair_coverage": 4,
        "teacher_cross_prefix_complete_units": 18,
        "teacher_positive_sides": 33,
        "mean_cross_prefix_margin": margin,
        "complete_units_by_family": dict(pair["complete_units_by_family"]),
        "cross_prefix_complete_units_by_family": dict(
            pair["cross_prefix_complete_units_by_family"]
        ),
        "source_broad_train_nll": 2.915099874138832,
        "update16_broad_train_nll": broad_nll,
        "source_residual_rms": 0.008839325979351997,
        "target_bank_state_sha256": _TARGET_SHA256[16],
        "frozen_complement_state_sha256": _FROZEN_SHA256,
        "failed_requirements": [
            "teacher_complete_units_at_least_10",
            "complete_physical_pair_coverage_at_least_5",
            "teacher_positive_sides_at_least_35",
            "mean_cross_prefix_margin_at_least_source",
            "book_or_picture_teacher_complete",
        ],
        "checks": dict(replayed),
        "training_scenes_only": True,
        "validation_qa_loaded": False,
        "passed": False,
    }


def _v38_pair_schedule() -> list[dict[str, Any]]:
    units = (*_V38_PRIORITY_UNITS, *_V38_PRIORITY_UNITS, *_V38_CANONICAL_UNITS)
    rows = [
        {"optimizer_step": step, "pair_id": pair_id, "question_key": question_key}
        for step, (pair_id, question_key) in enumerate(units, start=1)
    ]
    encoded = json.dumps(rows, sort_keys=True, separators=(",", ":")).encode()
    if len(rows) != 41 or hashlib.sha256(encoded).hexdigest() != _V38_PAIR_SCHEDULE_SHA256:
        raise RuntimeError("Pinned V38 pair schedule changed")
    return rows


def _v38_authorization(v37_source: Path) -> dict[str, Any]:
    """Build and independently attest the one allowed deterministic successor."""

    v36 = load_file(_resolve(_V36_SOURCE) / "adapter.safetensors", device="cpu")
    v37 = load_file(v37_source / "adapter.safetensors", device="cpu")
    if tensor_state_sha256(v36) != _SOURCE_TENSOR_SHA256:
        raise ValueError("V38 donor is not exact V36 update 16")
    if tensor_state_sha256(v37) != _FULL_TENSOR_SHA256[16]:
        raise ValueError("V38 primary source is not exact V37 update 16")

    hybrid = dict(v37)
    retained_k_names = tuple(
        f"{_TARGET_PREFIX}adapters.{index}.lora_{side}"
        for index in (0, 2)
        for side in ("a", "b")
    )
    reverted_v_names = tuple(
        f"{_TARGET_PREFIX}adapters.{index}.lora_{side}"
        for index in (1, 3)
        for side in ("a", "b")
    )
    for name in reverted_v_names:
        hybrid[name] = v36[name]

    differs_from_v36 = sorted(
        name for name in hybrid if not torch.equal(hybrid[name], v36[name])
    )
    differs_from_v37 = sorted(
        name for name in hybrid if not torch.equal(hybrid[name], v37[name])
    )
    query = _state(hybrid, _V38_QUERY_PREFIX)
    v23 = _state(hybrid, _TARGET_PREFIX)
    core = _state(hybrid, _CORE_PREFIX)
    frozen = {
        name: value
        for name, value in hybrid.items()
        if not name.startswith(_V38_QUERY_PREFIX)
    }
    query_b = {name: value for name, value in query.items() if name.endswith("lora_b")}
    expected_query_shapes = {
        "adapters.0.lora_a": (8, 1536),
        "adapters.0.lora_b": (2048, 8),
        "adapters.1.lora_a": (8, 1536),
        "adapters.1.lora_b": (4096, 8),
        "adapters.2.lora_a": (8, 1536),
        "adapters.2.lora_b": (2048, 8),
        "adapters.3.lora_a": (8, 1536),
        "adapters.3.lora_b": (2048, 8),
    }
    if (
        differs_from_v36 != sorted(retained_k_names)
        or differs_from_v37 != sorted(reverted_v_names)
        or tensor_state_sha256(hybrid) != _V38_HYBRID_FULL_SHA256
        or tensor_state_sha256(v23) != _V38_HYBRID_V23_SHA256
        or tensor_state_sha256(query) != _SOURCE_QUERY_SHA256
        or tensor_state_sha256(core) != _SOURCE_CORE_SHA256
        or tensor_state_sha256(frozen) != _V38_FROZEN_EXCLUDING_QUERY_SHA256
        or len(query) != 8
        or sum(int(value.numel()) for value in query.values()) != 131_072
        or set(query) != set(expected_query_shapes)
        or any(
            tuple(query[name].shape) != shape
            for name, shape in expected_query_shapes.items()
        )
        or len(query_b) != 4
        or any(not torch.count_nonzero(value) for value in query_b.values())
        or any(not torch.isfinite(value).all() for value in hybrid.values())
    ):
        raise ValueError("Exact deterministic V38 hybrid/query surface changed")

    schedule = _v38_pair_schedule()
    return {
        "authorized": True,
        "successor": "v38_query_recovery",
        "stage": "v38_query_recovery",
        "scope": "deterministic_v23_k_only_hybrid_then_train_existing_v30_query_only",
        "source_checkpoint": _relative(v37_source),
        "source_file_sha256": dict(_SAVED_FILES[16]),
        "source_full_tensor_state_sha256": _FULL_TENSOR_SHA256[16],
        "authorized_existing_lora_bank": _V38_QUERY_BANK,
        "authorized_existing_lora_rank": 8,
        "authorized_existing_lora_alpha": 16.0,
        "authorized_existing_lora_dropout": 0.0,
        "authorized_existing_lora_tensor_count": 8,
        "authorized_existing_lora_parameter_count": 131_072,
        "authorized_existing_lora_target_language_layers": [18, 19, 20, 21],
        "authorized_existing_lora_target_module_paths": [
            "model.language_model.layers.18.self_attn.q_proj",
            "model.language_model.layers.19.self_attn.q_proj",
            "model.language_model.layers.20.self_attn.q_proj",
            "model.language_model.layers.21.self_attn.q_proj",
        ],
        "authorized_existing_lora_parameter_shapes": {
            "layer18_q_proj": {"lora_a": [8, 1536], "lora_b": [2048, 8]},
            "layer19_q_proj": {"lora_a": [8, 1536], "lora_b": [4096, 8]},
            "layer20_q_proj": {"lora_a": [8, 1536], "lora_b": [2048, 8]},
            "layer21_q_proj": {"lora_a": [8, 1536], "lora_b": [2048, 8]},
        },
        "existing_query_bank_is_learned_not_zero_output": True,
        "existing_query_bank_reinitialization_authorized": False,
        "all_four_query_lora_b_tensors_nonzero": True,
        "new_lora_bank_authorized": False,
        "new_scene_encoder_module_authorized": False,
        "new_scene_tokens_authorized": False,
        "all_other_followup_architectures_authorized": False,
        "update_zero_initialization": {
            "scope": "v23_k_only_hybrid_from_exact_v36_and_v37",
            "v23_k_only_rollback_from_v36_u16": True,
            "primary_source_checkpoint": _relative(v37_source),
            "primary_source_full_tensor_state_sha256": _FULL_TENSOR_SHA256[16],
            "donor_checkpoint": _relative(_resolve(_V36_SOURCE)),
            "donor_full_tensor_state_sha256": _SOURCE_TENSOR_SHA256,
            "retain_exact_v37_u16_v23_k_tensor_names": list(retained_k_names),
            "restore_exact_v36_u16_v23_v_tensor_names": list(reverted_v_names),
            "differs_from_v36_u16_only_tensor_names": differs_from_v36,
            "differs_from_v37_u16_only_tensor_names": differs_from_v37,
            "hybrid_full_tensor_state_sha256": _V38_HYBRID_FULL_SHA256,
            "hybrid_v23_bank_state_sha256": _V38_HYBRID_V23_SHA256,
            "hybrid_v30_query_bank_state_sha256": _SOURCE_QUERY_SHA256,
            "hybrid_block_core_state_sha256": _SOURCE_CORE_SHA256,
            "hybrid_frozen_excluding_v30_query_state_sha256": (
                _V38_FROZEN_EXCLUDING_QUERY_SHA256
            ),
            "hybrid_tensor_count": 179,
            "hybrid_materialization_and_hash_verification_before_optimizer": True,
            "source_v36_or_v37_optimizer_state_may_be_loaded": False,
        },
        "trainable_surface": {
            "bank": _V38_QUERY_BANK,
            "tensor_count": 8,
            "parameter_count": 131_072,
            "source_state_sha256": _SOURCE_QUERY_SHA256,
            "state_must_change": True,
        },
        "frozen_surface": {
            "v23_shared_kv_bank_state_sha256": _V38_HYBRID_V23_SHA256,
            "block_cross_residual_state_sha256": _SOURCE_CORE_SHA256,
            "all_nonquery_adapter_tensor_count": 171,
            "all_nonquery_adapter_state_sha256": _V38_FROZEN_EXCLUDING_QUERY_SHA256,
            "v23_shared_kv_frozen": True,
            "block_cross_residual_frozen": True,
            "v33_scene_stack_frozen": True,
            "composer_frozen": True,
            "gemma_base_frozen": True,
            "every_other_tensor_and_buffer_frozen": True,
        },
        "optimizer": {
            "type": "AdamW",
            "fresh_state_required": True,
            "v36_optimizer_state_may_be_loaded": False,
            "v37_optimizer_state_may_be_loaded": False,
            "learning_rate": 2e-5,
            "weight_decay": 0.0,
            "gradient_clip_norm": 1.0,
        },
        "fresh_adamw_required": True,
        "v36_optimizer_state_may_be_loaded": False,
        "v37_optimizer_state_may_be_loaded": False,
        "learning_rate": 2e-5,
        "weight_decay": 0.0,
        "gradient_clip_norm": 1.0,
        "objective": {
            "broad_answer_nll_weight": 0.5,
            "pair_correct_answer_nll_weight": 1.0,
            "side_hinge_weight": 8.0,
            "side_hinge_margin": 0.5,
            "cross_prefix_maintenance_weight": 1.0,
            "cross_prefix_maintenance_margin": 0.10,
            "additional_terms_authorized": False,
        },
        "schedule": {
            "true_optimizer_step_count": 41,
            "priority_book_picture_units_steps_1_through_8": [
                {"pair_id": pair_id, "question_key": question_key}
                for pair_id, question_key in _V38_PRIORITY_UNITS
            ],
            "steps_9_through_16_repeat_steps_1_through_8_exactly": True,
            "canonical_25_unit_cycle_steps_17_through_41": [
                {"pair_id": pair_id, "question_key": question_key}
                for pair_id, question_key in _V38_CANONICAL_UNITS
            ],
            "exact_pair_schedule": schedule,
            "exact_pair_schedule_sha256": _V38_PAIR_SCHEDULE_SHA256,
            "one_deterministic_unchanged_broad_row_per_update": True,
            "full_pair_and_broad_schedule_sha256_must_be_pinned_before_training": True,
            "saved_optimizer_steps": [0, 8, 16, 24, 32, 40, 41],
            "continuation_past_update_41_authorized": False,
        },
        "update_zero_train_only_baseline": {
            "must_be_recomputed_from_exact_hybrid_before_optimizer_step_1": True,
            "inherited_v37_pair_metrics_may_be_used_as_hybrid_baseline": False,
            "priority_teacher_deficit_must_be_persisted": True,
            "broad_train_nll_must_be_persisted": True,
            "all_25_unit_teacher_metrics_must_be_persisted": True,
            "broad_greedy_source_correct": 23,
            "broad_greedy_source_total": 48,
            "training_scenes_only": True,
            "validation_qa_loaded": False,
        },
        "gate_artifact_requirements": {
            "gate_optimizer_steps": [0, 8, 16, 41],
            "per_unit_correct_answer_nll_must_be_persisted": True,
            "per_unit_rank_nll_must_be_persisted": True,
            "per_unit_pair_id_and_question_key_must_be_persisted": True,
            "per_unit_side_and_cross_prefix_correctness_must_be_persisted": True,
        },
        "update8_gate": {
            "priority_teacher_deficit_improvement_from_update_zero_minimum": 0.5,
            "priority_teacher_deficit_delta_from_update_zero_maximum": -0.5,
            "teacher_complete_units_minimum": 9,
            "teacher_positive_sides_minimum": 34,
            "teacher_cross_prefix_complete_units_minimum": 17,
            "broad_train_nll_maximum_increase_from_update_zero": 0.02,
            "frozen_state_must_remain_exact": True,
        },
        "update16_gate": {
            "require_update8_gate_passed": True,
            "priority_teacher_deficit_improvement_from_update_zero_minimum": 3.12,
            "priority_teacher_deficit_delta_from_update_zero_maximum": -3.12,
            "teacher_complete_units_minimum": 10,
            "teacher_positive_sides_minimum": 35,
            "complete_physical_pair_coverage_minimum": 5,
            "book_or_picture_teacher_complete_minimum": 1,
            "teacher_cross_prefix_complete_units_minimum": 17,
            "broad_train_nll_maximum_increase_from_update_zero": 0.02,
            "frozen_state_must_remain_exact": True,
        },
        "update41_gate": {
            "require_update16_gate_passed": True,
            "priority_teacher_deficit_improvement_from_update_zero_minimum": 6.24,
            "priority_teacher_deficit_delta_from_update_zero_maximum": -6.24,
            "teacher_complete_units_minimum": 12,
            "teacher_positive_sides_minimum": 37,
            "complete_physical_pair_coverage_minimum": 6,
            "book_teacher_complete_minimum": 1,
            "picture_teacher_complete_minimum": 1,
            "mirror_teacher_complete_minimum": 2,
            "teacher_cross_prefix_complete_units_minimum": 18,
            "train_greedy_complete_units_minimum": 6,
            "train_greedy_complete_each_priority_family_minimum": 1,
            "broad_greedy_exact_correct_minimum": 23,
            "broad_greedy_exact_total": 48,
            "broad_train_nll_maximum_increase_from_update_zero": 0.02,
            "frozen_state_must_remain_exact": True,
        },
        "scene_prefix_built_before_questions": True,
        "all_occupied_blocks_processed": True,
        "question_dependent_retrieval": False,
        "validation_qa_or_model_selection_before_complete_update41": False,
        "chat_promotion_authorized": False,
        "final_test_access_authorized": False,
        "oracle_access_authorized": False,
    }


def audit_v37_update16(
    config_path: Path = DEFAULT_CONFIG,
    checkpoint_root: Path = DEFAULT_CHECKPOINT_ROOT,
) -> dict[str, Any]:
    config_path = _resolve(config_path)
    checkpoint_root = _resolve(checkpoint_root)
    _real_file(config_path, "V37 config")
    if _sha256(config_path) != _CONFIG_SHA256:
        raise ValueError("V37 config bytes differ from the terminal lock")
    config_chain = _config_chain(config_path)
    config = load_config(config_path)
    prior, prior_files = _validate_v36_source(config)
    v36_report = json.loads(_resolve(_V36_TERMINAL).read_text(encoding="utf-8"))
    authorization = _mapping(v36_report.get("conditional_authorization"), "V36 authorization")
    paths = _checkpoint_paths(checkpoint_root)
    metadata, optimizer = _validate_metadata(paths, config, authorization)
    tensors = _validate_tensors(paths)
    gate = _replay_gate(metadata[16], config)
    successor = _v38_authorization(paths[-1])

    protected = _resolve(_PROTECTED_ARTIFACT)
    _real_file(protected, "protected V29 selection artifact")
    if _sha256(protected) != _PROTECTED_SHA256:
        raise ValueError("Protected V29 selection artifact changed")
    loaded = [
        *(Path(row["path"]) if Path(row["path"]).is_absolute() else PROJECT_ROOT / row["path"]
          for row in config_chain),
        *prior_files,
        protected,
        *(path / name for path in paths for name in _SAVED_FILES[int(path.name[-3:])]),
    ]
    inventory = sorted({_relative(path.resolve()) for path in loaded})
    forbidden = ("/qa/", "/maps/", "/oracle/", "validation.jsonl", "final_once")
    if any(fragment in path.casefold() for path in inventory for fragment in forbidden):
        raise RuntimeError("V37 terminal audit opened a forbidden runtime/environment file")

    report = {
        "schema_version": 1,
        "artifact": "v37_update16_terminal_gate",
        "audit_method": (
            "recursive_config_exact_v36_terminal_source_v37_metadata_runtime_"
            "optimizer_manifest_and_tensor_bytes_only"
        ),
        "passed": True,
        "checkpoint_root": _relative(checkpoint_root),
        "observed_saved_optimizer_steps": list(_SAVED_STEPS),
        "stopped_at_optimizer_step": 16,
        "no_update_024_or_later": True,
        "v37_train_only_continuation_gate_passed": False,
        "v37_development_selector_legal": False,
        "v37_chat_promotion_eligible": False,
        "exact_v36_prior_and_source": prior,
        "config_provenance": {
            "root": _relative(config_path),
            "root_sha256": _CONFIG_SHA256,
            "normalized_config_hash": _CONFIG_HASH,
            "recursive_chain_sha256": _CONFIG_CHAIN_SHA256,
            "recursive_chain": config_chain,
        },
        "saved_file_sha256_by_optimizer_step": {
            str(step): dict(_SAVED_FILES[step]) for step in _SAVED_STEPS
        },
        "tensor_transition": tensors,
        "optimizer_transition": {
            "source_v36_optimizer_access": "bytes_hashed_only_not_deserialized",
            "fresh_v37_adam_verified": True,
            "optimizer_integrity_manifest_verified": True,
            "saved_optimizer_states": optimizer,
        },
        "update16_gate_evidence": gate,
        "conditional_successor_authorization": successor,
        "conditional_v38_authorized": True,
        "arbitrary_continuation_authorized": False,
        "only_exact_conditional_successor_authorized": "v38_query_recovery",
        "successor_requires_separate_root_diagnosis_and_terminal_authorization": False,
        "chat_promotion_authorized": False,
        "final_test_access_authorized": False,
        "oracle_access_authorized": False,
        "validation_qa_loaded": False,
        "validation_model_selection_ran": False,
        "gemma_loaded": False,
        "scene_maps_loaded": False,
        "qa_loaded": False,
        "oracle_loaded": False,
        "final_test_scenes_touched": False,
        "loaded_file_inventory": inventory,
        "protected_artifact": {
            "path": _relative(protected),
            "sha256": _PROTECTED_SHA256,
            "access": "bytes_hashed_only",
            "unchanged": True,
        },
    }
    # Match the persisted JSON value model exactly (for example, Adam ``betas``
    # are tuples in PyTorch but arrays in the terminal artifact).  Returning the
    # normalized payload makes an in-memory replay byte-semantically comparable
    # with a report read back from disk.
    return json.loads(json.dumps(report, allow_nan=False))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--checkpoint-root", type=Path, default=DEFAULT_CHECKPOINT_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    report = audit_v37_update16(args.config, args.checkpoint_root)
    _atomic_json(_resolve(args.output), report)
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["audit_v37_update16"]
