"""Seal V44's failed update-eight gate and authorize one bounded V45 repair.

This is an offline, read-only terminal audit.  It authenticates only pinned
source/config/code/checkpoint artifacts; it does not load Gemma, QA, maps,
validation, oracle, final-scene, selector, or chat inputs.  Its sole positive
authorization is the exact train-only V45 retention-repair pilot encoded in
``conditional_successor_authorization``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file

from semantic_3d_chat.config import PROJECT_ROOT, config_hash, load_config
from semantic_3d_chat.language.lora import tensor_state_sha256
from semantic_3d_chat.training.checkpointing import (
    runtime_checkpoint_metadata,
    validate_runtime_checkpoint_metadata,
)

DEFAULT_CONFIG = Path(
    "configs/experiments/gemma4_diverse28_joint_scene_readout_v44.yaml"
)
DEFAULT_CHECKPOINT_ROOT = Path(
    "data_gemma4/checkpoints/gemma4_v44_joint_scene_readout_l14_query"
)
DEFAULT_OUTPUT = Path(
    "reports/gemma4/metrics/v44_joint_scene_readout_terminal_gate.json"
)
V43_TERMINAL = Path(
    "reports/gemma4/metrics/v43_aggregate_projected_screen_terminal_gate.json"
)
PROTECTED_REPORT = Path(
    "reports/gemma4/metrics/"
    "training_selection_gemma4_color_mirror_full_vocab_v11_resume36.json"
)
V44_TRAINER = Path(
    "src/semantic_3d_chat/training/train_joint_scene_readout_v44.py"
)
V44_TEST = Path("tests/test_train_joint_scene_readout_v44.py")
V45_CONFIG = Path(
    "configs/experiments/gemma4_diverse28_retention_repair_v45.yaml"
)
V45_OUTPUT = Path(
    "data_gemma4/checkpoints/gemma4_v45_retention_repair_l14_query"
)

_PINNED_INPUTS = {
    str(V43_TERMINAL): (
        "013fbe79ac42e842e83989e33f132b9ff3529746a8045feb212ded32e50a2cc2"
    ),
    str(PROTECTED_REPORT): (
        "c0086f66edbb8854a7938e09c57535bfd47100adbaf3b3c95eeb4b08014ce2f8"
    ),
    str(DEFAULT_CONFIG): (
        "a3f3b65dc3a32612060a679cbcc40c115e5b0c4d014670c9a8f7f752e4a7abb7"
    ),
    str(V44_TRAINER): (
        "300e7a1c130ef4be862d3b1357d58f9db17e9db4bba26ccbf24ea662cb248f5d"
    ),
    str(V44_TEST): (
        "577b5d69b53b2351f4ad3d640b302ae0686d4824263efb4219ec6d4488b302dd"
    ),
}
_CONFIG_HASH = "6c5ebf1ac65c"
_CHECKPOINT_FILES = {
    "update_000/adapter.safetensors": (
        "b0bdceb7699e9d97467915c69186f433d3b0fac2b09144d38c3078afe1f70cb0"
    ),
    "update_000/metadata.json": (
        "db23a9bf89236c6e6ddcf5ffc6aa6b42d23e1e94a3340d781c24ed2d992d38f0"
    ),
    "update_000/runtime_metadata.json": (
        "127772da72ee5bba0f4fe2c5c159ae836683f62b85d06b837764cea5bc0ce356"
    ),
    "update_004/adapter.safetensors": (
        "2e46cedae11cdc85d671ef2ad9c12b203a5cb0d997740c5651264fac96d2f709"
    ),
    "update_004/metadata.json": (
        "0f1c4954488910ed293a03da0c582e01dcd7753c5f6bca12c769a869c86330d3"
    ),
    "update_004/optimizer.pt": (
        "662f3f8061035e757e6d77c82ff666ae0159d30b9f0d1eda43813b0ca47808c3"
    ),
    "update_004/runtime_metadata.json": (
        "c1c0ff8b1229094a54c3d51edb85a58f3699edff86abd893b5b7a225b3a5fdb7"
    ),
    "update_008/adapter.safetensors": (
        "22f7e0276a91d45e31893843345e98e310fbffd14147852c05c5c3bec4dc6589"
    ),
    "update_008/metadata.json": (
        "797fcdb87da3391c5196fda15fca4d352846dda2d5dcc49263ca3f7854fcd1b3"
    ),
    "update_008/optimizer.pt": (
        "cdf9eb0c3560be1bc1542963354444eddb7a89ed0d063ffaa769c45231b9d61a"
    ),
    "update_008/runtime_metadata.json": (
        "59542b55239d64a9c28b9b99ec0a39b47c1dd93839753f61d145722ea7c50acf"
    ),
}
_DIRECTORY_INVENTORY = {
    "update_000": [
        "adapter.safetensors",
        "metadata.json",
        "runtime_metadata.json",
    ],
    "update_004": [
        "adapter.safetensors",
        "metadata.json",
        "optimizer.pt",
        "runtime_metadata.json",
    ],
    "update_008": [
        "adapter.safetensors",
        "metadata.json",
        "optimizer.pt",
        "runtime_metadata.json",
    ],
}
_PARAMETER_NAMES = (
    "block_cross_residual.w_o",
    "lora_banks.extension_v28_stage_b_query.adapters.1.lora_a",
    "lora_banks.extension_v28_stage_b_query.adapters.1.lora_b",
)
_PARAMETER_SHAPES = ((256, 1536), (4, 1536), (4096, 4))
_PARAMETER_COUNTS = (393_216, 6_144, 16_384)
_STATE_HASHES = {
    "update_000": {
        "full": "7b951c6d7ae4f7b50603159f0bc4dfb4d50b5b40f9325134d78d1de1dae87fc0",
        "authorized": "b935c7e6ccceb1068f80e679b4159c6ca756f9f81868b954b93ac683e014f5a0",
        "frozen": "31cb215cf0d1623886d4a79203c501912e66537021da53dd711124abdc2e36fa",
    },
    "update_004": {
        "full": "6367a95d443b3c8908cd6ddf12f03bf75621a460e9d8818cae46e74ff4ce306d",
        "authorized": "d12d65cefb2addd8fc1f10f5c7ece6190f79caeb055adbfaceb07572674b374f",
        "frozen": "31cb215cf0d1623886d4a79203c501912e66537021da53dd711124abdc2e36fa",
    },
    "update_008": {
        "full": "ad9b2227e68020ae785084666c9dca58c3d479e5b1e3e4c13461539fcb19c6fb",
        "authorized": "f56fdc4ce31a3e97c80e9a214948b6855fd87ae3b7f96ebf1f152229cf833e02",
        "frozen": "31cb215cf0d1623886d4a79203c501912e66537021da53dd711124abdc2e36fa",
    },
}
_SOURCE_BROAD_NLL = 2.9013306349515915
_BROAD_NLL_MAXIMUM = 2.9213306349515915
_SOURCE_PRIORITY_DEFICIT = 31.113729119300842
_UPDATE8_BROAD_NLL = 2.898227721452713
_UPDATE8_PRIORITY_DEFICIT = 29.92463493347168
_UPDATE8_PRIORITY_IMPROVEMENT = 1.1890941858291626
_UPDATE8_TRUST_RMS = 0.004081375896930695
_TRAIN_SCENES = [
    *(f"scene_{index:06d}" for index in range(11, 19)),
    *(f"scene_{index:06d}" for index in range(31, 39)),
]
_TARGET_SCHEDULE = [
    {"optimizer_update": 1, "question_key": "cfq_5c84a2c27d2be251"},
    {"optimizer_update": 2, "question_key": "cfq_699675ceeaf65406"},
    {"optimizer_update": 3, "question_key": "cfq_5c84a2c27d2be251"},
    {"optimizer_update": 4, "question_key": "cfq_699675ceeaf65406"},
    {"optimizer_update": 5, "question_key": "cfq_163eb92339ad35a5"},
    {"optimizer_update": 6, "question_key": "cfq_163eb92339ad35a5"},
    {"optimizer_update": 7, "question_key": "cfq_163eb92339ad35a5"},
    {"optimizer_update": 8, "question_key": "cfq_163eb92339ad35a5"},
]
_FRAGILE_SIDES = (
    ("pair_000005", "cfq_a578dc166be9a217", 0, 0.4687870144844055),
    ("pair_000006", "cfq_0a79d507273195ef", 0, 0.125),
    ("pair_000006", "cfq_5c84a2c27d2be251", 0, 0.125),
    ("pair_000007", "cfq_736067b51ce93c49", 0, 0.0625),
    ("pair_000007", "cfq_997610c185204121", 0, 0.25),
    ("pair_000016", "cfq_699675ceeaf65406", 0, 0.5),
    ("pair_000016", "cfq_699675ceeaf65406", 1, 0.25),
    ("pair_000018", "cfq_90b3d9852a93ce2a", 1, 0.125),
)
_BOOK_CROSS_SIDES = (
    ("pair_000015", "cfq_13b1138d14c52a7c", 0, 0.1605992317199707),
    ("pair_000015", "cfq_13b1138d14c52a7c", 1, 0.026900842785835266),
    ("pair_000015", "cfq_a1c673a1197a0961", 0, 0.030324697494506836),
    ("pair_000015", "cfq_a1c673a1197a0961", 1, 0.03217540681362152),
)


def _resolve(path: str | Path) -> Path:
    value = Path(path)
    return value.resolve() if value.is_absolute() else (PROJECT_ROOT / value).resolve()


def _relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path.resolve())


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def _mapping(value: object, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{field} must be a mapping")
    return value


def _sequence(value: object, field: str) -> Sequence[Any]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise TypeError(f"{field} must be a sequence")
    return value


def _locked_file(path: Path, expected: str, field: str) -> None:
    if path.is_symlink() or not path.is_file():
        raise FileNotFoundError(f"{field} must be a real file: {path}")
    observed = _sha256(path)
    if observed != expected:
        raise ValueError(
            f"{field} bytes changed: expected {expected}, observed {observed}"
        )


def _read_json(path: Path, expected: str, field: str) -> Mapping[str, Any]:
    _locked_file(path, expected, field)
    with path.open("r", encoding="utf-8") as handle:
        return _mapping(json.load(handle), field)


def _authenticate_inputs() -> dict[str, Any]:
    observed: dict[str, str] = {}
    for relative, expected in _PINNED_INPUTS.items():
        path = _resolve(relative)
        _locked_file(path, expected, relative)
        observed[relative] = expected
    config = load_config(_resolve(DEFAULT_CONFIG))
    if config_hash(config) != _CONFIG_HASH:
        raise ValueError("V44 normalized config hash changed")
    return {
        "file_sha256": observed,
        "normalized_config_hash": _CONFIG_HASH,
        "protected_report_access": "bytes_hashed_only",
    }


def _authenticate_inventory(root: Path) -> dict[str, Any]:
    if root.is_symlink() or not root.is_dir():
        raise FileNotFoundError("V44 checkpoint root must be a real directory")
    if sorted(path.name for path in root.iterdir()) != sorted(_DIRECTORY_INVENTORY):
        raise ValueError("V44 checkpoint root inventory changed")
    for directory_name, expected_entries in _DIRECTORY_INVENTORY.items():
        directory = root / directory_name
        if directory.is_symlink() or not directory.is_dir():
            raise FileNotFoundError(f"V44 {directory_name} must be a real directory")
        if sorted(path.name for path in directory.iterdir()) != expected_entries:
            raise ValueError(f"V44 {directory_name} inventory changed")
    for relative, expected in _CHECKPOINT_FILES.items():
        _locked_file(root / relative, expected, f"V44 {relative}")
    return {
        "root": _relative(root),
        "root_entries": sorted(_DIRECTORY_INVENTORY),
        "directory_entries": dict(_DIRECTORY_INVENTORY),
        "file_sha256": dict(_CHECKPOINT_FILES),
        "manifest_sha256": _canonical_sha256(_CHECKPOINT_FILES),
        "no_update_after_eight_persisted": True,
    }


def _authenticate_runtime(
    root: Path, metadata: Mapping[str, Any], *, step: int
) -> dict[str, Any]:
    relative = f"update_{step:03d}/runtime_metadata.json"
    runtime = _read_json(
        root / relative,
        _CHECKPOINT_FILES[relative],
        f"V44 update-{step} runtime metadata",
    )
    validate_runtime_checkpoint_metadata(runtime)
    if runtime != runtime_checkpoint_metadata(metadata):
        raise ValueError(f"V44 update-{step} runtime metadata is not freshly sanitized")
    return {
        "optimizer_step": step,
        "runtime_metadata_sha256": _CHECKPOINT_FILES[relative],
        "sanitized_runtime_exact": True,
        "training_history_qa_and_gate_fields_absent": True,
    }


def _authenticate_tensors(root: Path) -> dict[str, Any]:
    states: dict[str, Mapping[str, torch.Tensor]] = {}
    for step in (0, 4, 8):
        label = f"update_{step:03d}"
        tensors = load_file(root / label / "adapter.safetensors", device="cpu")
        if len(tensors) != 179:
            raise ValueError(f"V44 {label} tensor count changed")
        if any(not torch.isfinite(value).all() for value in tensors.values()):
            raise ValueError(f"V44 {label} contains a nonfinite tensor")
        if not all(name in tensors for name in _PARAMETER_NAMES):
            raise ValueError(f"V44 {label} authorized tensor inventory changed")
        if tuple(tuple(tensors[name].shape) for name in _PARAMETER_NAMES) != (
            _PARAMETER_SHAPES
        ):
            raise ValueError(f"V44 {label} authorized tensor shapes changed")
        if tuple(int(tensors[name].numel()) for name in _PARAMETER_NAMES) != (
            _PARAMETER_COUNTS
        ):
            raise ValueError(f"V44 {label} authorized parameter counts changed")
        authorized = {name: tensors[name] for name in _PARAMETER_NAMES}
        frozen = {name: value for name, value in tensors.items() if name not in authorized}
        observed = {
            "full": tensor_state_sha256(tensors),
            "authorized": tensor_state_sha256(authorized),
            "frozen": tensor_state_sha256(frozen),
        }
        if observed != _STATE_HASHES[label]:
            raise ValueError(f"V44 {label} tensor-state hash changed: {observed}")
        states[label] = tensors
    baseline = states["update_000"]
    if any(set(state) != set(baseline) for state in states.values()):
        raise ValueError("V44 tensor names differ between checkpoints")
    changed_by_step = {}
    for label in ("update_004", "update_008"):
        changed = sorted(
            name
            for name in baseline
            if not torch.equal(baseline[name], states[label][name])
        )
        if changed != sorted(_PARAMETER_NAMES):
            raise ValueError(f"V44 {label} changed tensors escaped authorization")
        changed_by_step[label] = changed
    return {
        "tensor_count_each_checkpoint": 179,
        "authorized_parameter_names": list(_PARAMETER_NAMES),
        "authorized_parameter_shapes": [list(value) for value in _PARAMETER_SHAPES],
        "authorized_parameter_counts": list(_PARAMETER_COUNTS),
        "authorized_parameter_count": sum(_PARAMETER_COUNTS),
        "state_sha256": dict(_STATE_HASHES),
        "changed_tensor_names": changed_by_step,
        "only_three_authorized_tensors_changed": True,
        "frozen_state_bit_exact_through_update_eight": True,
        "all_tensors_finite": True,
    }


def _unit_by_key(metrics: Mapping[str, Any], question_key: str) -> Mapping[str, Any]:
    units = _sequence(metrics.get("units"), "V44 pair units")
    matches = [
        _mapping(unit, "V44 pair unit")
        for unit in units
        if isinstance(unit, Mapping) and unit.get("question_key") == question_key
    ]
    if len(matches) != 1:
        raise ValueError(f"V44 expected one unit for {question_key}; got {len(matches)}")
    return matches[0]


def _authenticate_retention_source(
    source_metrics: Mapping[str, Any], update8_metrics: Mapping[str, Any]
) -> dict[str, Any]:
    fragile = []
    for pair_id, question_key, side_index, expected_margin in _FRAGILE_SIDES:
        unit = _unit_by_key(source_metrics, question_key)
        margins = _sequence(unit.get("side_margins"), "V44 source side margins")
        if unit.get("pair_id") != pair_id or float(margins[side_index]) != expected_margin:
            raise ValueError(f"V44 fragile-side source changed: {question_key}/{side_index}")
        fragile.append(
            {
                "pair_id": pair_id,
                "question_key": question_key,
                "side_index": side_index,
                "source_margin": expected_margin,
                "retention_floor": 0.125,
            }
        )
    book_cross = []
    for pair_id, question_key, side_index, expected_margin in _BOOK_CROSS_SIDES:
        unit = _unit_by_key(source_metrics, question_key)
        margins = _sequence(
            unit.get("cross_prefix_margins"), "V44 source cross-prefix margins"
        )
        if (
            unit.get("pair_id") != pair_id
            or unit.get("family") != "book_support"
            or unit.get("cross_prefix_complete") is not True
            or float(margins[side_index]) != expected_margin
        ):
            raise ValueError(f"V44 book-cross source changed: {question_key}/{side_index}")
        book_cross.append(
            {
                "pair_id": pair_id,
                "question_key": question_key,
                "side_index": side_index,
                "source_margin": expected_margin,
                "retention_floor": 0.025,
            }
        )
    lost = []
    for pair_id, question_key, side_index in (
        ("pair_000006", "cfq_5c84a2c27d2be251", 0),
        ("pair_000016", "cfq_699675ceeaf65406", 1),
    ):
        unit = _unit_by_key(update8_metrics, question_key)
        margin = float(
            _sequence(unit.get("side_margins"), "V44 update-eight side margins")[
                side_index
            ]
        )
        if unit.get("pair_id") != pair_id or margin != 0.0:
            raise ValueError(f"V44 exact-zero lost side changed: {question_key}/{side_index}")
        lost.append(
            {
                "pair_id": pair_id,
                "question_key": question_key,
                "side_index": side_index,
                "update8_margin": margin,
                "gate4_required_relation": "strictly_greater_than_zero",
            }
        )
    return {
        "fragile_side_constraints": fragile,
        "fragile_side_constraint_count": 8,
        "fragile_side_floor": 0.125,
        "book_cross_constraints": book_cross,
        "book_cross_constraint_count": 4,
        "book_cross_floor": 0.025,
        "lost_side_gate4_constraints": lost,
        "source_is_exact_original_v41_update_zero_pair_metrics": True,
    }


def _expected_update8_gate() -> dict[str, Any]:
    checks = {
        "both_authorized_parameter_groups_changed": True,
        "broad_nll_within_source_plus_0_02": True,
        "frozen_state_exact": True,
        "priority_teacher_deficit_improved_at_least_0_5": True,
        "query_state_changed": True,
        "scene_readout_state_changed": True,
        "teacher_complete_units_at_least_9": False,
        "teacher_cross_complete_units_at_least_17": True,
        "teacher_positive_sides_at_least_34": False,
    }
    return {
        "both_authorized_parameter_groups_changed": True,
        "broad_nll_delta_from_update_zero": -0.003102913498878479,
        "broad_nll_within_source_plus_0_02": True,
        "checks": checks,
        "frozen_state_exact": True,
        "passed": False,
        "priority_teacher_deficit_improved_at_least_0_5": True,
        "priority_teacher_side_deficit": _UPDATE8_PRIORITY_DEFICIT,
        "priority_teacher_side_deficit_improvement": _UPDATE8_PRIORITY_IMPROVEMENT,
        "query_state_changed": True,
        "scene_readout_state_changed": True,
        "source_prefix_trust_rms": _UPDATE8_TRUST_RMS,
        "teacher_complete_units_at_least_9": False,
        "teacher_cross_complete_units_at_least_17": True,
        "teacher_positive_sides_at_least_34": False,
        "training_scenes_only": True,
        "validation_qa_loaded": False,
    }


def _authenticate_history_and_gate(
    metadata: Mapping[str, Mapping[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    update8 = metadata["update_008"]
    histories = {
        label: list(_sequence(value.get("history"), f"V44 {label} history"))
        for label, value in metadata.items()
    }
    if (
        [len(histories[label]) for label in ("update_000", "update_004", "update_008")]
        != [1, 5, 9]
        or histories["update_000"] != histories["update_004"][:1]
        or histories["update_004"] != histories["update_008"][:5]
        or [row.get("optimizer_update") for row in histories["update_008"]]
        != list(range(9))
        or [
            row.get("optimizer_update")
            for row in histories["update_008"]
            if row.get("saved_checkpoint") is True
        ]
        != [0, 4, 8]
    ):
        raise ValueError("V44 checkpoint history/persistence chain changed")
    history = histories["update_008"]
    if any(
        row.get("frozen_state_sha256") != _STATE_HASHES["update_000"]["frozen"]
        or row.get("oracle_environment_files_loaded") is not False
        for row in history
    ):
        raise ValueError("V44 history escaped its frozen/train-only boundary")
    row0 = _mapping(history[0], "V44 update-zero history row")
    row4 = _mapping(history[4], "V44 update-four history row")
    row8 = _mapping(history[8], "V44 update-eight history row")
    source_metrics = _mapping(row0.get("source_pair_metrics"), "V44 source metrics")
    metrics4 = _mapping(row4.get("pair_metrics"), "V44 update-four metrics")
    metrics8 = _mapping(row8.get("pair_metrics"), "V44 update-eight metrics")
    stage8 = _mapping(update8.get("v44_joint_scene_readout"), "V44 update-eight stage")
    expected_gate = _expected_update8_gate()
    if (
        row8.get("update8_train_only_gate") != expected_gate
        or stage8.get("update8_train_only_gate") != expected_gate
        or stage8.get("update16_train_only_gate") is not None
        or row8.get("update16_train_only_gate") is not None
        or float(row0.get("source_broad_train_nll")) != _SOURCE_BROAD_NLL
        or float(row8.get("broad_diagnostic_nll")) != _UPDATE8_BROAD_NLL
        or metrics8.get("complete_units") != 7
        or metrics8.get("positive_sides") != 32
        or metrics8.get("cross_prefix_complete_units") != 18
        or metrics8.get("complete_physical_pair_coverage") != 3
        or _mapping(metrics8.get("complete_units_by_family"), "V44 u8 family counts")
        != {"book_support": 0, "mirror_lr": 1, "picture_support": 0}
        or _mapping(
            metrics8.get("cross_prefix_complete_units_by_family"),
            "V44 u8 cross family counts",
        )
        != {"book_support": 0, "mirror_lr": 4, "picture_support": 3}
        or row8.get("authorized_state_sha256")
        != _STATE_HASHES["update_008"]["authorized"]
        or row8.get("scene_readout_state_sha256")
        != "3b8c0fe1f57e030a3c01c607ea12a716ee4a337fd530ebfa2808392640af6d0e"
        or row8.get("query_state_sha256")
        != "8783ce21b6fc1639d35b451e7895348fa894604b6e758782f3dea48470ea03ab"
        or float(row8.get("source_prefix_trust_rms")) != _UPDATE8_TRUST_RMS
    ):
        raise ValueError("V44 failed update-eight gate evidence changed")
    for metrics in (source_metrics, metrics4, metrics8):
        if (
            metrics.get("training_scenes_only") is not True
            or metrics.get("validation_qa_loaded") is not False
            or metrics.get("unit_count") != 25
            or metrics.get("side_count") != 50
        ):
            raise ValueError("V44 pair diagnostics crossed the train-only boundary")
    retention = _authenticate_retention_source(source_metrics, metrics8)
    gate = {
        "passed": False,
        "failed_checks": [
            "teacher_complete_units_at_least_9",
            "teacher_positive_sides_at_least_34",
        ],
        "passing_checks": sorted(
            name for name, value in expected_gate["checks"].items() if value is True
        ),
        "complete_units": 7,
        "positive_sides": 32,
        "cross_prefix_complete_units": 18,
        "complete_physical_pair_coverage": 3,
        "complete_units_by_family": {
            "book_support": 0,
            "mirror_lr": 1,
            "picture_support": 0,
        },
        "cross_prefix_complete_units_by_family": {
            "book_support": 0,
            "mirror_lr": 4,
            "picture_support": 3,
        },
        "source_broad_nll": _SOURCE_BROAD_NLL,
        "update8_broad_nll": _UPDATE8_BROAD_NLL,
        "broad_nll_delta": _UPDATE8_BROAD_NLL - _SOURCE_BROAD_NLL,
        "source_priority_side_deficit": _SOURCE_PRIORITY_DEFICIT,
        "update8_priority_side_deficit": _UPDATE8_PRIORITY_DEFICIT,
        "priority_side_deficit_improvement": _UPDATE8_PRIORITY_IMPROVEMENT,
        "update8_source_prefix_trust_rms": _UPDATE8_TRUST_RMS,
        "both_authorized_groups_changed": True,
        "frozen_state_exact": True,
        "stopped_before_update_nine": True,
        "update16_gate_absent": True,
    }
    history_audit = {
        "optimizer_updates_executed": list(range(1, 9)),
        "checkpoint_steps_persisted": [0, 4, 8],
        "history_prefixes_bit_exact": True,
        "history_frozen_hash_exact_every_step": True,
        "no_update_after_eight_persisted": True,
    }
    return history_audit, gate, retention


def _authenticate_provenance(
    metadata: Mapping[str, Mapping[str, Any]], v43: Mapping[str, Any]
) -> dict[str, Any]:
    authorization = _mapping(
        v43.get("conditional_successor_authorization"), "V43 V44 authorization"
    )
    expected_terminal = {
        "path": str(_resolve(V43_TERMINAL)),
        "sha256": _PINNED_INPUTS[str(V43_TERMINAL)],
    }
    for label, value in metadata.items():
        stage = _mapping(value.get("v44_joint_scene_readout"), f"V44 {label} stage")
        if (
            stage.get("conditional_authorization") != authorization
            or stage.get("conditional_v43_terminal_gate") != expected_terminal
            or stage.get("source_checkpoint")
            != str(
                _resolve(
                    "data_gemma4/checkpoints/"
                    "gemma4_v41_retry1_diverse28_projected_gradient_l14_query/"
                    "update_000"
                )
            )
            or stage.get("source_full_tensor_state_sha256")
            != _STATE_HASHES["update_000"]["full"]
            or stage.get("frozen_excluding_authorized_state_sha256")
            != _STATE_HASHES["update_000"]["frozen"]
            or stage.get("validation_qa_loaded") is not False
            or stage.get("oracle_environment_files_loaded") is not False
            or stage.get("deferred_final_scene_ids_loaded") != []
            or stage.get("question_dependent_scene_processing") is not False
            or stage.get("question_dependent_retrieval") is not False
        ):
            raise ValueError(f"V44 {label} authorization/train-only provenance changed")
    update8 = metadata["update_008"]
    source = _mapping(update8.get("v41_projected_gradient"), "V44 source V41 stage")
    update_zero = _mapping(source.get("update_zero_attestation"), "V44 source attestation")
    cache = _mapping(
        update_zero.get("training_cache_boundary"), "V44 training cache boundary"
    )
    qa = _mapping(source.get("train_qa_dataset"), "V44 train QA boundary")
    if (
        cache.get("exact_train_scene_ids") != _TRAIN_SCENES
        or cache.get("exact_train_scene_count") != 16
        or cache.get("validation_environment_maps_loaded") is not False
        or cache.get("oracle_environment_files_loaded") is not False
        or qa.get("train_scene_ids") != _TRAIN_SCENES
        or qa.get("train_question_count") != 384
        or qa.get("train_changed_pair_unit_count") != 25
        or qa.get("validation_qa_loaded") is not False
        or qa.get("deferred_final_qa_loaded") is not False
        or qa.get("oracle_environment_files_loaded") is not False
        or update_zero.get("source_optimizer_files_opened") is not False
        or update_zero.get("source_optimizer_states_loaded") is not False
    ):
        raise ValueError("V44 exact training data/source-optimizer boundary changed")
    return {
        "v43_terminal_sha256": _PINNED_INPUTS[str(V43_TERMINAL)],
        "same_exact_v43_authorization_at_updates_zero_four_eight": True,
        "exact_train_scene_ids": list(_TRAIN_SCENES),
        "exact_train_scene_count": 16,
        "train_question_count": 384,
        "train_changed_pair_unit_count": 25,
        "source_optimizer_files_or_states_loaded": False,
        "validation_oracle_and_final_access": False,
        "question_dependent_scene_processing_or_retrieval": False,
    }


def _authorization(retention: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "authorization_id": "v45_train_only_retention_repair_pilot",
        "authorized": True,
        "only_exact_action": "one_bounded_v45_train_only_retention_repair_pilot",
        "authorized_config": str(V45_CONFIG),
        "authorized_output_root": str(V45_OUTPUT),
        "source_checkpoint": str(DEFAULT_CHECKPOINT_ROOT / "update_008"),
        "source_checkpoint_file_sha256": {
            name.removeprefix("update_008/"): digest
            for name, digest in _CHECKPOINT_FILES.items()
            if name.startswith("update_008/")
        },
        "source_adapter_tensor_count": 179,
        "source_full_tensor_state_sha256": _STATE_HASHES["update_008"]["full"],
        "source_authorized_surface_state_sha256": _STATE_HASHES["update_008"][
            "authorized"
        ],
        "source_frozen_excluding_authorized_state_sha256": _STATE_HASHES[
            "update_008"
        ]["frozen"],
        "source_optimizer_policy": {
            "source_optimizer_file_present_and_authenticated": True,
            "source_optimizer_file_open_authorized_by_v45": False,
            "source_optimizer_deserialization_authorized": False,
            "source_optimizer_state_loading_authorized": False,
            "fresh_optimizer_required": True,
        },
        "trainable_surface": {
            "parameter_names": list(_PARAMETER_NAMES),
            "parameter_shapes": [list(value) for value in _PARAMETER_SHAPES],
            "parameter_counts": list(_PARAMETER_COUNTS),
            "scene_readout_parameter_count": 393_216,
            "query_parameter_count": 22_528,
            "total_parameter_count": 415_744,
            "block_qkv_frozen": True,
            "gemma_base_and_all_other_lora_banks_frozen": True,
            "every_other_tensor_and_buffer_frozen": True,
        },
        "optimizer": {
            "implementation": "fresh_torch_adamw_two_groups",
            "source_optimizer_loaded": False,
            "scene_readout_learning_rate": 1.0e-5,
            "query_learning_rate": 8.0e-6,
            "weight_decay": 0.0,
            "foreach": False,
            "fused": False,
            "per_group_gradient_clip_norm": 1.0,
        },
        "objective": {
            "broad_nll_weight": 0.25,
            "pair_correct_nll_weight": 0.5,
            "target_side_hinge_weight": 8.0,
            "target_cross_prefix_hinge_weight": 8.0,
            "target_side_hinge_margin": 0.5,
            "target_cross_prefix_hinge_margin": 0.1,
            "retention_weight": 8.0,
            "retention_formula": (
                "mean(relu(0.125-fragile_side_margin),8)"
                "+mean(relu(0.025-book_cross_prefix_margin),4)"
            ),
            "u8_prefix_trust_weight": 0.001,
            "u8_prefix_trust_scale": 0.05,
            "u8_prefix_reference_computed_before_optimizer_step_one": True,
        },
        "retention_control": {
            **dict(retention),
            "applied_at_every_optimizer_update": True,
            "fragile_side_hinges_mean_normalized_separately": True,
            "book_cross_hinges_mean_normalized_separately": True,
            "two_normalized_means_summed_before_single_weight": True,
        },
        "target_schedule": list(_TARGET_SCHEDULE),
        "schedule": {
            "maximum_optimizer_updates": 8,
            "checkpoint_steps": [0, 2, 4, 6, 8],
            "gate4_must_pass_before_updates_five_through_eight": True,
            "true_optimizer_step_per_target_schedule_row": True,
            "target_schedule_is_fixed_and_nonadaptive": True,
        },
        "reference_baselines": {
            "original_v41_update_zero_priority_side_deficit": (
                _SOURCE_PRIORITY_DEFICIT
            ),
            "original_v41_update_zero_broad_nll": _SOURCE_BROAD_NLL,
            "broad_nll_maximum": _BROAD_NLL_MAXIMUM,
            "prefix_trust_reference": "exact_v44_update_008_full_scene_prefixes",
        },
        "update4_gate": {
            "complete_units_minimum": 9,
            "positive_sides_minimum": 34,
            "cross_prefix_complete_units_minimum": 17,
            "complete_physical_pair_id_coverage_minimum": 4,
            "priority_side_deficit_minimum_improvement_vs_original_v41_u0": 0.5,
            "broad_nll_maximum": _BROAD_NLL_MAXIMUM,
            "lost_side_margins_must_both_be_strictly_positive": [
                {
                    "pair_id": "pair_000006",
                    "question_key": "cfq_5c84a2c27d2be251",
                    "side_index": 0,
                },
                {
                    "pair_id": "pair_000016",
                    "question_key": "cfq_699675ceeaf65406",
                    "side_index": 1,
                },
            ],
            "both_authorized_parameter_groups_must_change": True,
            "frozen_state_must_remain_exact": True,
        },
        "update8_gate": {
            "require_recorded_update4_gate_passed": True,
            "complete_units_minimum": 10,
            "positive_sides_minimum": 35,
            "cross_prefix_complete_units_minimum": 17,
            "complete_physical_pair_id_coverage_minimum": 5,
            "mirror_complete_units_minimum": 2,
            "book_complete_units_minimum": 1,
            "book_cross_prefix_complete_units_minimum": 1,
            "priority_side_deficit_minimum_improvement_vs_original_v41_u0": 0.5,
            "broad_nll_maximum": _BROAD_NLL_MAXIMUM,
            "train_greedy_complete_units_minimum": 5,
            "broad_greedy_exact_correct_minimum": 23,
            "broad_greedy_row_count_exact": 48,
            "lost_side_margins_must_remain_strictly_positive": [
                {
                    "pair_id": "pair_000006",
                    "question_key": "cfq_5c84a2c27d2be251",
                    "side_index": 0,
                },
                {
                    "pair_id": "pair_000016",
                    "question_key": "cfq_699675ceeaf65406",
                    "side_index": 1,
                },
            ],
            "both_authorized_parameter_groups_must_change": True,
            "frozen_state_must_remain_exact": True,
            "u8_prefix_trust_rms_maximum": 0.002,
        },
        "scope": {
            "training_qa_and_maps_only": True,
            "all_occupied_blocks_processed": True,
            "question_dependent_scene_processing": False,
            "question_dependent_retrieval": False,
            "validation_access_authorized": False,
            "oracle_access_authorized": False,
            "final_test_access_authorized": False,
            "selector_execution_authorized": False,
            "runtime_promotion_authorized": False,
            "chat_promotion_authorized": False,
            "new_terminal_seal_required_after_training": True,
        },
    }


def audit_v44_joint_scene_readout(
    config_path: str | Path = DEFAULT_CONFIG,
    checkpoint_root: str | Path = DEFAULT_CHECKPOINT_ROOT,
) -> dict[str, Any]:
    """Authenticate the immutable V44 train-only terminal evidence."""

    config_file = _resolve(config_path)
    root = _resolve(checkpoint_root)
    if config_file != _resolve(DEFAULT_CONFIG):
        raise ValueError("V44 terminal config path is pinned")
    if root != _resolve(DEFAULT_CHECKPOINT_ROOT):
        raise ValueError("V44 terminal checkpoint root is pinned")
    inputs = _authenticate_inputs()
    inventory = _authenticate_inventory(root)
    v43 = _read_json(
        _resolve(V43_TERMINAL),
        _PINNED_INPUTS[str(V43_TERMINAL)],
        "V43 terminal seal",
    )
    metadata = {
        f"update_{step:03d}": _read_json(
            root / f"update_{step:03d}/metadata.json",
            _CHECKPOINT_FILES[f"update_{step:03d}/metadata.json"],
            f"V44 update-{step} metadata",
        )
        for step in (0, 4, 8)
    }
    for step, label in ((0, "update_000"), (4, "update_004"), (8, "update_008")):
        value = metadata[label]
        if (
            value.get("schema_version") != 1
            or value.get("config_hash") != _CONFIG_HASH
            or value.get("epoch") != step
            or value.get("optimizer_step") != step
        ):
            raise ValueError(f"V44 {label} checkpoint identity changed")
    runtime = {
        label: _authenticate_runtime(root, metadata[label], step=step)
        for step, label in ((0, "update_000"), (4, "update_004"), (8, "update_008"))
    }
    tensors = _authenticate_tensors(root)
    history, gate, retention = _authenticate_history_and_gate(metadata)
    provenance = _authenticate_provenance(metadata, v43)
    authorization = _authorization(retention)
    return {
        "schema_version": 1,
        "artifact": "v44_joint_scene_readout_terminal_gate",
        "passed": True,
        "terminal_conclusion": "update8_train_only_gate_failed_stop_is_final",
        "input_integrity": inputs,
        "checkpoint_inventory": inventory,
        "tensor_transition": tensors,
        "runtime_metadata_audit": runtime,
        "history_audit": history,
        "update8_gate_replay": gate,
        "v44_provenance": provenance,
        "execution_conclusion": {
            "updates_one_through_eight_executed": True,
            "update8_checkpoint_persisted": True,
            "update8_gate_passed": False,
            "update_nine_executed": False,
            "update16_checkpoint_or_gate_present": False,
            "frozen_state_exact": True,
            "validation_loaded": False,
            "oracle_loaded": False,
            "final_test_loaded": False,
            "selector_executed": False,
            "runtime_promoted": False,
        },
        "conditional_successor_authorization": authorization,
        "only_exact_successor_authorized": "v45_train_only_retention_repair_pilot",
        "v45_train_only_retention_repair_pilot_authorized": True,
        "arbitrary_training_authorized": False,
        "resume_v44_training_authorized": False,
        "validation_access_authorized": False,
        "oracle_access_authorized": False,
        "final_test_access_authorized": False,
        "selector_execution_authorized": False,
        "chat_or_runtime_promotion_authorized": False,
        "embodied_agent_promotion_authorized": False,
        "terminal_process_access_audit": {
            "gemma_loaded": False,
            "qa_loaded": False,
            "maps_loaded": False,
            "validation_loaded": False,
            "oracle_loaded": False,
            "final_test_loaded": False,
            "adapter_tensors_loaded_on_cpu_for_hash_and_equality_audit": True,
            "optimizer_deserialized": False,
            "optimizer_step_executed": False,
            "protected_report_access": "bytes_hashed_only",
        },
    }


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def write_report(
    output: str | Path = DEFAULT_OUTPUT,
    *,
    config_path: str | Path = DEFAULT_CONFIG,
    checkpoint_root: str | Path = DEFAULT_CHECKPOINT_ROOT,
) -> dict[str, Any]:
    path = _resolve(output)
    if path != _resolve(DEFAULT_OUTPUT):
        raise ValueError("V44 terminal output path is pinned")
    if path.is_symlink() or path.exists():
        raise FileExistsError(f"V44 terminal is one-shot and will not overwrite {path}")
    report = audit_v44_joint_scene_readout(config_path, checkpoint_root)
    _atomic_json(path, report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--checkpoint-root", type=Path, default=DEFAULT_CHECKPOINT_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    report = write_report(
        args.output,
        config_path=args.config,
        checkpoint_root=args.checkpoint_root,
    )
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = ["audit_v44_joint_scene_readout", "write_report"]
