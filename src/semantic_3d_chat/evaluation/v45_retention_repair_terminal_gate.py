"""Seal V45's failed update-four gate and authorize one bounded V46 probe.

This terminal is intentionally an offline, read-only audit.  It authenticates
the exact V44 source, V45 code/config, and the only three V45 checkpoints that
were persisted.  It loads adapter tensors on CPU solely for hashing and exact
equality checks.  It never loads Gemma, QA, maps, validation, oracle, deferred
final scenes, selector inputs, chat inputs, or optimizer state.

The only successor authorization is a fixed, train-only, report-only V46
diagnostic.  It cannot write a candidate checkpoint or authorize validation,
selection, or promotion.
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
    "configs/experiments/gemma4_diverse28_retention_repair_v45.yaml"
)
DEFAULT_CHECKPOINT_ROOT = Path(
    "data_gemma4/checkpoints/gemma4_v45_retention_repair_l14_query"
)
DEFAULT_OUTPUT = Path(
    "reports/gemma4/metrics/v45_retention_repair_terminal_gate.json"
)
V44_TERMINAL = Path(
    "reports/gemma4/metrics/v44_joint_scene_readout_terminal_gate.json"
)
V44_SOURCE = Path(
    "data_gemma4/checkpoints/"
    "gemma4_v44_joint_scene_readout_l14_query/update_008"
)
PROTECTED_REPORT = Path(
    "reports/gemma4/metrics/"
    "training_selection_gemma4_color_mirror_full_vocab_v11_resume36.json"
)
V45_TRAINER = Path(
    "src/semantic_3d_chat/training/train_retention_repair_v45.py"
)
V45_TEST = Path("tests/test_train_retention_repair_v45.py")
V46_DIAGNOSTIC = Path(
    "src/semantic_3d_chat/evaluation/v46_v45_u4_lost_side_screen.py"
)
V46_TEST = Path("tests/test_v46_v45_u4_lost_side_screen.py")
V46_OUTPUT = Path(
    "reports/gemma4/metrics/v46_v45_u4_lost_side_no_step_diagnostic.json"
)

_PINNED_INPUTS = {
    str(V44_TERMINAL): (
        "b968c46c686051e864417b7539db7e90160a1f0b4639af031d02aab005643b67"
    ),
    str(PROTECTED_REPORT): (
        "c0086f66edbb8854a7938e09c57535bfd47100adbaf3b3c95eeb4b08014ce2f8"
    ),
    str(DEFAULT_CONFIG): (
        "9a4b77c43d30d258be9e4e6d60c477ef6af593fe440d61d28082f2f343519436"
    ),
    str(V45_TRAINER): (
        "dcd46f96a510693b45ea511bd2e1daebfdb1a3461f819f04e5b7182ddc78ae28"
    ),
    str(V45_TEST): (
        "6f69c5da39f4b8ba1f1b3ccb1e8e51ff90c8e7b77f25922668d9882179f98f2d"
    ),
    str(V46_DIAGNOSTIC): (
        "acda1857fd8e1c8673d313dabca07dd1834df5fd17d1eb54021f46a1ae451926"
    ),
    str(V46_TEST): (
        "300d58d1f412f5f94e89193bd091f9e15665960aac12035c896f61c2c04d3547"
    ),
}
_CONFIG_HASH = "9a5bb6c41d3d"
_V44_SOURCE_FILES = {
    "adapter.safetensors": (
        "22f7e0276a91d45e31893843345e98e310fbffd14147852c05c5c3bec4dc6589"
    ),
    "metadata.json": (
        "797fcdb87da3391c5196fda15fca4d352846dda2d5dcc49263ca3f7854fcd1b3"
    ),
    "optimizer.pt": (
        "cdf9eb0c3560be1bc1542963354444eddb7a89ed0d063ffaa769c45231b9d61a"
    ),
    "runtime_metadata.json": (
        "59542b55239d64a9c28b9b99ec0a39b47c1dd93839753f61d145722ea7c50acf"
    ),
}
_CHECKPOINT_FILES = {
    "update_000/adapter.safetensors": (
        "22f7e0276a91d45e31893843345e98e310fbffd14147852c05c5c3bec4dc6589"
    ),
    "update_000/metadata.json": (
        "ca31cb6143b7dbf55670bf7a6d78c9b5e35bcd32def4ffd99075470dceaac1b2"
    ),
    "update_000/runtime_metadata.json": (
        "6dab5ea12d9ab5ae9518eb70ece2cce23285c837a980e7c97028775c726d4fe8"
    ),
    "update_002/adapter.safetensors": (
        "2859bf4c308e984d3ca591cb8ebcfcaa16a730e63302ddaafddb2c71cec20693"
    ),
    "update_002/metadata.json": (
        "7552c8d0f6c54082917eb3a8717bde22bf0fdf531eb2ae11e6b327671cc27fbd"
    ),
    "update_002/optimizer.pt": (
        "50568eec0300f7e5f5e05e58ed975c071f2d9d4b2636dd9eda065ddd8799b06a"
    ),
    "update_002/runtime_metadata.json": (
        "f8388e9813be67609ef4bbbeb0fbaf1e6086daa1d10eb0156a3a957119d91a9e"
    ),
    "update_004/adapter.safetensors": (
        "baffb29e31e1ddf0164bf4b9bcf47ab14f61160f3d46e834ceafc3c1a7c66e17"
    ),
    "update_004/metadata.json": (
        "4249bcdec60dd7468e62c0687616a8a820be0bae94289636da33e4379dc7bf6c"
    ),
    "update_004/optimizer.pt": (
        "c409db27ccc6ef68e43c36123519810c3b65a9d579715ff64d5f3595d7da688d"
    ),
    "update_004/runtime_metadata.json": (
        "8beca055a77016f4ce0960b49789e750ab6b34d3edd852888cddec7a4e2980f0"
    ),
}
_DIRECTORY_INVENTORY = {
    "update_000": [
        "adapter.safetensors",
        "metadata.json",
        "runtime_metadata.json",
    ],
    "update_002": [
        "adapter.safetensors",
        "metadata.json",
        "optimizer.pt",
        "runtime_metadata.json",
    ],
    "update_004": [
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
        "full": "ad9b2227e68020ae785084666c9dca58c3d479e5b1e3e4c13461539fcb19c6fb",
        "authorized": "f56fdc4ce31a3e97c80e9a214948b6855fd87ae3b7f96ebf1f152229cf833e02",
        "frozen": "31cb215cf0d1623886d4a79203c501912e66537021da53dd711124abdc2e36fa",
    },
    "update_002": {
        "full": "0ed1e9412d747a492903ed9025bc49c9db4cc71cd7bb6ea13f9f990f01bd607b",
        "authorized": "09020e44b2b2cfec4d894bd9f784441a8a688e9eefffbdd3ddf5f09e5f889c5c",
        "frozen": "31cb215cf0d1623886d4a79203c501912e66537021da53dd711124abdc2e36fa",
    },
    "update_004": {
        "full": "468f493a746c6125f8ebc62d57ca8ae0419160f6e13ce903dd9f40c64aa772c2",
        "authorized": "e4165bb1c2a4664eeb146a48107aead3e69bb576c1604bea39b3b7474d17c696",
        "frozen": "31cb215cf0d1623886d4a79203c501912e66537021da53dd711124abdc2e36fa",
    },
}
_SCENE_READOUT_HASHES = {
    0: "3b8c0fe1f57e030a3c01c607ea12a716ee4a337fd530ebfa2808392640af6d0e",
    1: "7dca773dda25915b2d01895cb9d18328c82ddfade39a65faf44305a9bbf10556",
    2: "1b4b0fc8e7bea167003842e8f859aab59893645cd08e15b06e854a6cf2cda57b",
    3: "beaede25182ed74ca27b09de1f01058c0c301f5ffd33b4b030fc98084aa6be34",
    4: "b87c24dba5c8473a04c903e4926edfc7f613a20e5e27ef9d1e4bac3e2de988ec",
}
_QUERY_HASHES = {
    0: "8783ce21b6fc1639d35b451e7895348fa894604b6e758782f3dea48470ea03ab",
    1: "849c32fcf99cb969aab537f6a55c65e6a00f3e3fb98de08a26b998203e2df04f",
    2: "2719975133348dfc22ba38cdd8f55d9fe849b5acfcda66cdfa241b7ca505c26a",
    3: "e797f9d908a84e598c824a54510dbc4907f65c37fffb55bdd001e402413607d3",
    4: "1eaec1b045d6d1a555012ecc486df22cd2624546001ede43caa9fe092620c0c5",
}
_AUTHORIZED_HISTORY_HASHES = {
    0: _STATE_HASHES["update_000"]["authorized"],
    1: "b9b48a30ee0f5030f9c0447570e15cb5e1973f4682a414ad019435e70c8dabe6",
    2: _STATE_HASHES["update_002"]["authorized"],
    3: "349498713491b38eb76fdf73d1a914b029d22fa0435abcd266d85a5b6a8a3c51",
    4: _STATE_HASHES["update_004"]["authorized"],
}
_TARGETS = (
    ("pair_000006", "cfq_5c84a2c27d2be251"),
    ("pair_000016", "cfq_699675ceeaf65406"),
    ("pair_000006", "cfq_5c84a2c27d2be251"),
    ("pair_000016", "cfq_699675ceeaf65406"),
)
_TRAIN_SCENES = [
    *(f"scene_{index:06d}" for index in range(11, 19)),
    *(f"scene_{index:06d}" for index in range(31, 39)),
]
_SOURCE_BROAD_NLL = 2.898227721452713
_UPDATE4_BROAD_NLL = 2.889571795860926
_ORIGINAL_V41_PRIORITY_DEFICIT = 31.113729119300842
_UPDATE4_PRIORITY_DEFICIT = 29.800106167793274
_UPDATE4_PRIORITY_IMPROVEMENT = 1.3136229515075684
_UPDATE4_TRUST_RMS = 0.0013297934783622622
_BROAD_NLL_MAXIMUM = 2.9213306349515915


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
        _locked_file(_resolve(relative), expected, relative)
        observed[relative] = expected
    config = load_config(_resolve(DEFAULT_CONFIG))
    if config_hash(config) != _CONFIG_HASH:
        raise ValueError("V45 normalized config hash changed")
    return {
        "file_sha256": observed,
        "normalized_config_hash": _CONFIG_HASH,
        "protected_report_access": "bytes_hashed_only",
    }


def _authenticate_v44_source(v44: Mapping[str, Any]) -> dict[str, Any]:
    authorization = _mapping(
        v44.get("conditional_successor_authorization"),
        "V44 successor authorization",
    )
    source = _resolve(V44_SOURCE)
    if source.is_symlink() or not source.is_dir():
        raise FileNotFoundError("V44 update-eight source must be a real directory")
    if sorted(path.name for path in source.iterdir()) != sorted(_V44_SOURCE_FILES):
        raise ValueError("V44 update-eight source inventory changed")
    for name, expected in _V44_SOURCE_FILES.items():
        _locked_file(source / name, expected, f"V44 source {name}")
    if (
        v44.get("schema_version") != 1
        or v44.get("artifact") != "v44_joint_scene_readout_terminal_gate"
        or v44.get("passed") is not True
        or v44.get("terminal_conclusion")
        != "update8_train_only_gate_failed_stop_is_final"
        or v44.get("only_exact_successor_authorized")
        != "v45_train_only_retention_repair_pilot"
        or v44.get("v45_train_only_retention_repair_pilot_authorized") is not True
        or authorization.get("authorization_id")
        != "v45_train_only_retention_repair_pilot"
        or authorization.get("source_full_tensor_state_sha256")
        != _STATE_HASHES["update_000"]["full"]
        or v44.get("validation_access_authorized") is not False
        or v44.get("oracle_access_authorized") is not False
        or v44.get("final_test_access_authorized") is not False
        or v44.get("selector_execution_authorized") is not False
        or v44.get("chat_or_runtime_promotion_authorized") is not False
    ):
        raise ValueError("V44 exact V45 authorization changed")
    return {
        "terminal_path": str(V44_TERMINAL),
        "terminal_sha256": _PINNED_INPUTS[str(V44_TERMINAL)],
        "source_checkpoint": str(V44_SOURCE),
        "source_file_sha256": dict(_V44_SOURCE_FILES),
        "source_full_tensor_state_sha256": _STATE_HASHES["update_000"]["full"],
        "source_authorized_state_sha256": _STATE_HASHES["update_000"][
            "authorized"
        ],
        "source_frozen_state_sha256": _STATE_HASHES["update_000"]["frozen"],
        "source_optimizer_bytes_authenticated_but_not_deserialized": True,
        "exact_v45_authorization_authenticated": True,
    }


def _authenticate_inventory(root: Path) -> dict[str, Any]:
    if root.is_symlink() or not root.is_dir():
        raise FileNotFoundError("V45 checkpoint root must be a real directory")
    observed_root = sorted(path.name for path in root.iterdir())
    if observed_root != sorted(_DIRECTORY_INVENTORY):
        raise ValueError("V45 checkpoint root inventory changed")
    for directory_name, expected_entries in _DIRECTORY_INVENTORY.items():
        directory = root / directory_name
        if directory.is_symlink() or not directory.is_dir():
            raise FileNotFoundError(f"V45 {directory_name} must be a real directory")
        if sorted(path.name for path in directory.iterdir()) != expected_entries:
            raise ValueError(f"V45 {directory_name} inventory changed")
    for relative, expected in _CHECKPOINT_FILES.items():
        _locked_file(root / relative, expected, f"V45 {relative}")
    return {
        "root": _relative(root),
        "root_entries": observed_root,
        "directory_entries": dict(_DIRECTORY_INVENTORY),
        "file_sha256": dict(_CHECKPOINT_FILES),
        "manifest_sha256": _canonical_sha256(_CHECKPOINT_FILES),
        "update_006_absent": not (root / "update_006").exists(),
        "update_008_absent": not (root / "update_008").exists(),
        "no_checkpoint_after_failed_update_four": True,
    }


def _authenticate_runtime(
    root: Path, metadata: Mapping[str, Any], *, step: int
) -> dict[str, Any]:
    relative = f"update_{step:03d}/runtime_metadata.json"
    runtime = _read_json(
        root / relative,
        _CHECKPOINT_FILES[relative],
        f"V45 update-{step} runtime metadata",
    )
    validate_runtime_checkpoint_metadata(runtime)
    if runtime != runtime_checkpoint_metadata(metadata):
        raise ValueError(f"V45 update-{step} runtime metadata is not freshly sanitized")
    return {
        "optimizer_step": step,
        "runtime_metadata_sha256": _CHECKPOINT_FILES[relative],
        "sanitized_runtime_exact": True,
        "training_history_qa_and_gate_fields_absent": True,
    }


def _authenticate_tensors(root: Path) -> dict[str, Any]:
    states: dict[str, Mapping[str, torch.Tensor]] = {}
    for step in (0, 2, 4):
        label = f"update_{step:03d}"
        tensors = load_file(root / label / "adapter.safetensors", device="cpu")
        if len(tensors) != 179:
            raise ValueError(f"V45 {label} tensor count changed")
        if any(not torch.isfinite(value).all() for value in tensors.values()):
            raise ValueError(f"V45 {label} contains a nonfinite tensor")
        if not all(name in tensors for name in _PARAMETER_NAMES):
            raise ValueError(f"V45 {label} authorized tensor inventory changed")
        if tuple(tuple(tensors[name].shape) for name in _PARAMETER_NAMES) != (
            _PARAMETER_SHAPES
        ):
            raise ValueError(f"V45 {label} authorized tensor shapes changed")
        if tuple(int(tensors[name].numel()) for name in _PARAMETER_NAMES) != (
            _PARAMETER_COUNTS
        ):
            raise ValueError(f"V45 {label} authorized parameter counts changed")
        authorized = {name: tensors[name] for name in _PARAMETER_NAMES}
        frozen = {name: value for name, value in tensors.items() if name not in authorized}
        observed = {
            "full": tensor_state_sha256(tensors),
            "authorized": tensor_state_sha256(authorized),
            "frozen": tensor_state_sha256(frozen),
        }
        if observed != _STATE_HASHES[label]:
            raise ValueError(f"V45 {label} tensor-state hash changed: {observed}")
        states[label] = tensors
    baseline = states["update_000"]
    if any(set(state) != set(baseline) for state in states.values()):
        raise ValueError("V45 tensor names differ between checkpoints")
    changed_by_step = {}
    for label in ("update_002", "update_004"):
        changed = sorted(
            name
            for name in baseline
            if not torch.equal(baseline[name], states[label][name])
        )
        if changed != sorted(_PARAMETER_NAMES):
            raise ValueError(f"V45 {label} changed tensors escaped authorization")
        changed_by_step[label] = changed
    return {
        "tensor_count_each_checkpoint": 179,
        "authorized_parameter_names": list(_PARAMETER_NAMES),
        "authorized_parameter_shapes": [list(value) for value in _PARAMETER_SHAPES],
        "authorized_parameter_counts": list(_PARAMETER_COUNTS),
        "authorized_parameter_count": sum(_PARAMETER_COUNTS),
        "state_sha256": dict(_STATE_HASHES),
        "changed_tensor_names": changed_by_step,
        "update_zero_adapter_bytes_equal_v44_source": (
            _CHECKPOINT_FILES["update_000/adapter.safetensors"]
            == _V44_SOURCE_FILES["adapter.safetensors"]
        ),
        "only_three_authorized_tensors_changed": True,
        "frozen_state_bit_exact_through_update_four": True,
        "all_tensors_finite": True,
    }


def _priority_side_deficit(pair_metrics: Mapping[str, Any]) -> float:
    rows = _sequence(pair_metrics.get("units"), "V45 pair metric units")
    if len(rows) != 25:
        raise ValueError("V45 priority deficit requires exactly 25 units")
    counts = {"book_support": 0, "picture_support": 0}
    deficit = 0.0
    for value in rows:
        row = _mapping(value, "V45 pair metric unit")
        family = str(row.get("family"))
        if family not in counts:
            continue
        margins = _sequence(row.get("side_margins"), "V45 side margins")
        if len(margins) != 2:
            raise ValueError("V45 priority row must have two side margins")
        deficit += sum(max(0.0, 0.5 - float(margin)) for margin in margins)
        counts[family] += 1
    if counts != {"book_support": 4, "picture_support": 4}:
        raise ValueError("V45 priority-family inventory changed")
    return deficit


def _unit_by_key(metrics: Mapping[str, Any], question_key: str) -> Mapping[str, Any]:
    matches = [
        _mapping(value, "V45 pair unit")
        for value in _sequence(metrics.get("units"), "V45 pair units")
        if isinstance(value, Mapping) and value.get("question_key") == question_key
    ]
    if len(matches) != 1:
        raise ValueError(f"V45 expected one unit for {question_key}; got {len(matches)}")
    return matches[0]


def _replay_update4_gate(metrics: Mapping[str, Any], row4: Mapping[str, Any]) -> dict[str, Any]:
    if (
        metrics.get("training_scenes_only") is not True
        or metrics.get("validation_qa_loaded") is not False
        or metrics.get("unit_count") != 25
        or metrics.get("side_count") != 50
    ):
        raise ValueError("V45 update-four metrics crossed the train-only boundary")
    deficit = _priority_side_deficit(metrics)
    if deficit != _UPDATE4_PRIORITY_DEFICIT:
        raise ValueError("V45 update-four priority deficit changed")
    lost = []
    for pair_id, question_key, side_index, expected in (
        ("pair_000006", "cfq_5c84a2c27d2be251", 0, -0.0625),
        (
            "pair_000016",
            "cfq_699675ceeaf65406",
            1,
            0.24999994039535522,
        ),
    ):
        unit = _unit_by_key(metrics, question_key)
        margin = float(_sequence(unit.get("side_margins"), "lost-side margins")[side_index])
        if unit.get("pair_id") != pair_id or margin != expected:
            raise ValueError(f"V45 lost-side evidence changed: {question_key}")
        lost.append(
            {
                "pair_id": pair_id,
                "question_key": question_key,
                "side_index": side_index,
                "margin": margin,
                "strictly_positive": margin > 0.0,
            }
        )
    broad_nll = float(row4.get("broad_diagnostic_nll"))
    checks = {
        "both_authorized_parameter_groups_changed": (
            row4.get("scene_readout_state_changed") is True
            and row4.get("query_state_changed") is True
        ),
        "both_lost_side_margins_strictly_positive": all(
            value["strictly_positive"] for value in lost
        ),
        "broad_nll_at_most_authorized_maximum": broad_nll
        <= _BROAD_NLL_MAXIMUM,
        "complete_physical_pair_id_coverage_at_least_4": int(
            metrics.get("complete_physical_pair_coverage", -1)
        )
        >= 4,
        "frozen_state_exact": row4.get("frozen_state_sha256")
        == _STATE_HASHES["update_000"]["frozen"],
        "priority_teacher_deficit_improved_at_least_0_5_vs_original_v41_u0": (
            _ORIGINAL_V41_PRIORITY_DEFICIT - deficit >= 0.5
        ),
        "query_state_changed": row4.get("query_state_changed") is True,
        "scene_readout_state_changed": row4.get("scene_readout_state_changed") is True,
        "teacher_complete_units_at_least_9": int(metrics.get("complete_units", -1))
        >= 9,
        "teacher_cross_complete_units_at_least_17": int(
            metrics.get("cross_prefix_complete_units", -1)
        )
        >= 17,
        "teacher_positive_sides_at_least_34": int(metrics.get("positive_sides", -1))
        >= 34,
    }
    expected_failed = [
        "both_lost_side_margins_strictly_positive",
        "complete_physical_pair_id_coverage_at_least_4",
        "teacher_complete_units_at_least_9",
        "teacher_positive_sides_at_least_34",
    ]
    failed = sorted(name for name, passed in checks.items() if not passed)
    if failed != expected_failed:
        raise ValueError(f"V45 update-four failure set changed: {failed}")
    gate = _mapping(row4.get("update4_train_only_gate"), "V45 update-four gate")
    if (
        gate.get("checks") != checks
        or gate.get("passed") is not False
        or gate.get("priority_teacher_side_deficit") != deficit
        or gate.get("priority_teacher_side_deficit_improvement_vs_original_v41_u0")
        != _UPDATE4_PRIORITY_IMPROVEMENT
        or gate.get("broad_nll") != _UPDATE4_BROAD_NLL
        or gate.get("broad_nll_maximum") != _BROAD_NLL_MAXIMUM
        or gate.get("u8_prefix_trust_rms") != _UPDATE4_TRUST_RMS
        or gate.get("full_train_pair_unit_count") != 25
        or gate.get("full_broad_nll_row_count") != 48
        or gate.get("training_scenes_only") is not True
        or gate.get("validation_qa_loaded") is not False
    ):
        raise ValueError("V45 recorded update-four gate changed")
    return {
        "passed": False,
        "failed_checks": failed,
        "passing_checks": sorted(name for name, passed in checks.items() if passed),
        "checks": checks,
        "complete_units": int(metrics["complete_units"]),
        "positive_sides": int(metrics["positive_sides"]),
        "cross_prefix_complete_units": int(metrics["cross_prefix_complete_units"]),
        "complete_physical_pair_coverage": int(
            metrics["complete_physical_pair_coverage"]
        ),
        "complete_units_by_family": dict(
            _mapping(metrics.get("complete_units_by_family"), "V45 family counts")
        ),
        "cross_prefix_complete_units_by_family": dict(
            _mapping(
                metrics.get("cross_prefix_complete_units_by_family"),
                "V45 cross family counts",
            )
        ),
        "source_broad_nll": _SOURCE_BROAD_NLL,
        "update4_broad_nll": broad_nll,
        "broad_nll_delta": broad_nll - _SOURCE_BROAD_NLL,
        "original_v41_priority_side_deficit": _ORIGINAL_V41_PRIORITY_DEFICIT,
        "update4_priority_side_deficit": deficit,
        "priority_side_deficit_improvement": (
            _ORIGINAL_V41_PRIORITY_DEFICIT - deficit
        ),
        "update4_source_prefix_trust_rms": _UPDATE4_TRUST_RMS,
        "lost_side_evidence": lost,
        "update8_gate_absent": row4.get("update8_train_only_gate") is None,
        "stopped_before_update_five": True,
    }


def _authenticate_history_and_gate(
    metadata: Mapping[str, Mapping[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    histories = {
        label: list(_sequence(value.get("history"), f"V45 {label} history"))
        for label, value in metadata.items()
    }
    if (
        [len(histories[label]) for label in ("update_000", "update_002", "update_004")]
        != [1, 3, 5]
        or histories["update_000"] != histories["update_002"][:1]
        or histories["update_002"] != histories["update_004"][:3]
        or [row.get("optimizer_update") for row in histories["update_004"]]
        != list(range(5))
        or [
            row.get("optimizer_update")
            for row in histories["update_004"]
            if row.get("saved_checkpoint") is True
        ]
        != [0, 2, 4]
    ):
        raise ValueError("V45 checkpoint history/persistence chain changed")
    history = histories["update_004"]
    for step, row in enumerate(history):
        if (
            row.get("frozen_state_sha256")
            != _STATE_HASHES["update_000"]["frozen"]
            or row.get("authorized_state_sha256") != _AUTHORIZED_HISTORY_HASHES[step]
            or row.get("scene_readout_state_sha256") != _SCENE_READOUT_HASHES[step]
            or row.get("query_state_sha256") != _QUERY_HASHES[step]
            or row.get("oracle_environment_files_loaded") is not False
        ):
            raise ValueError(f"V45 history step {step} escaped its exact boundary")
        if step > 0 and (
            row.get("target_pair_id"), row.get("target_question_key")
        ) != _TARGETS[step - 1]:
            raise ValueError(f"V45 history step {step} target changed")
    row0 = _mapping(history[0], "V45 update-zero row")
    row4 = _mapping(history[4], "V45 update-four row")
    source_metrics = _mapping(row0.get("source_pair_metrics"), "V45 source metrics")
    metrics4 = _mapping(row4.get("pair_metrics"), "V45 update-four metrics")
    if (
        source_metrics.get("unit_count") != 25
        or source_metrics.get("side_count") != 50
        or source_metrics.get("complete_units") != 7
        or source_metrics.get("positive_sides") != 32
        or source_metrics.get("cross_prefix_complete_units") != 18
        or source_metrics.get("complete_physical_pair_coverage") != 3
        or source_metrics.get("training_scenes_only") is not True
        or source_metrics.get("validation_qa_loaded") is not False
        or float(row0.get("source_broad_train_nll")) != _SOURCE_BROAD_NLL
        or float(row0.get("source_prefix_trust_rms")) != 0.0
    ):
        raise ValueError("V45 authenticated update-zero behavior changed")
    if (
        metrics4.get("complete_units") != 8
        or metrics4.get("positive_sides") != 32
        or metrics4.get("cross_prefix_complete_units") != 17
        or metrics4.get("complete_physical_pair_coverage") != 3
        or _mapping(metrics4.get("complete_units_by_family"), "V45 family counts")
        != {"book_support": 0, "mirror_lr": 2, "picture_support": 0}
        or _mapping(
            metrics4.get("cross_prefix_complete_units_by_family"),
            "V45 cross family counts",
        )
        != {"book_support": 1, "mirror_lr": 4, "picture_support": 3}
        or float(row4.get("broad_diagnostic_nll")) != _UPDATE4_BROAD_NLL
        or float(row4.get("u8_prefix_trust_rms")) != _UPDATE4_TRUST_RMS
    ):
        raise ValueError("V45 update-four scientific evidence changed")
    gate = _replay_update4_gate(metrics4, row4)
    if row4.get("update8_train_only_gate") is not None:
        raise ValueError("V45 update-eight gate unexpectedly exists")
    return (
        {
            "optimizer_updates_executed": [1, 2, 3, 4],
            "checkpoint_steps_persisted": [0, 2, 4],
            "history_prefixes_bit_exact": True,
            "history_frozen_hash_exact_every_step": True,
            "fixed_targets_executed": [
                {"optimizer_update": index + 1, "pair_id": pair, "question_key": key}
                for index, (pair, key) in enumerate(_TARGETS)
            ],
            "no_update_after_four_persisted": True,
        },
        gate,
    )


def _authenticate_provenance(
    metadata: Mapping[str, Mapping[str, Any]], v44: Mapping[str, Any]
) -> dict[str, Any]:
    authorization = _mapping(
        v44.get("conditional_successor_authorization"),
        "V44 V45 authorization",
    )
    expected_terminal = {
        "path": str(_resolve(V44_TERMINAL)),
        "sha256": _PINNED_INPUTS[str(V44_TERMINAL)],
    }
    for label, value in metadata.items():
        stage = _mapping(value.get("v45_retention_repair"), f"V45 {label} stage")
        if (
            stage.get("conditional_authorization") != authorization
            or stage.get("conditional_v44_terminal_gate") != expected_terminal
            or stage.get("source_checkpoint") != str(_resolve(V44_SOURCE))
            or stage.get("source_full_tensor_state_sha256")
            != _STATE_HASHES["update_000"]["full"]
            or stage.get("source_authorized_surface_state_sha256")
            != _STATE_HASHES["update_000"]["authorized"]
            or stage.get("frozen_excluding_authorized_state_sha256")
            != _STATE_HASHES["update_000"]["frozen"]
            or stage.get("validation_qa_loaded") is not False
            or stage.get("oracle_environment_files_loaded") is not False
            or stage.get("deferred_final_scene_ids_loaded") != []
            or stage.get("selector_execution_authorized") is not False
            or stage.get("runtime_promotion_authorized") is not False
            or stage.get("question_dependent_scene_processing") is not False
            or stage.get("question_dependent_retrieval") is not False
            or stage.get("independent_terminal_seal_required") is not True
        ):
            raise ValueError(f"V45 {label} authorization/train-only provenance changed")
    update4 = metadata["update_004"]
    stage4 = _mapping(update4.get("v45_retention_repair"), "V45 update-four stage")
    source_audit = _mapping(stage4.get("source_audit"), "V45 source audit")
    update_zero = _mapping(
        source_audit.get("live_update_zero_diagnostic_attestation"),
        "V45 live update-zero attestation",
    )
    inherited = _mapping(update4.get("v41_projected_gradient"), "V45 V41 provenance")
    attestation = _mapping(inherited.get("update_zero_attestation"), "V45 data boundary")
    cache = _mapping(attestation.get("training_cache_boundary"), "V45 cache boundary")
    qa = _mapping(inherited.get("train_qa_dataset"), "V45 QA boundary")
    if (
        source_audit.get("source_optimizer_file_opened") is not False
        or source_audit.get("source_optimizer_deserialized") is not False
        or source_audit.get("source_optimizer_state_loaded") is not False
        or update_zero.get("passed") is not True
        or update_zero.get("computed_before_optimizer_construction") is not True
        or update_zero.get("validation_qa_loaded") is not False
        or cache.get("exact_train_scene_ids") != _TRAIN_SCENES
        or cache.get("exact_train_scene_count") != 16
        or cache.get("validation_environment_maps_loaded") is not False
        or cache.get("oracle_environment_files_loaded") is not False
        or qa.get("train_scene_ids") != _TRAIN_SCENES
        or qa.get("train_question_count") != 384
        or qa.get("train_changed_pair_unit_count") != 25
        or qa.get("validation_qa_loaded") is not False
        or qa.get("deferred_final_qa_loaded") is not False
        or qa.get("oracle_environment_files_loaded") is not False
    ):
        raise ValueError("V45 exact training/source-optimizer boundary changed")
    return {
        "v44_terminal_sha256": _PINNED_INPUTS[str(V44_TERMINAL)],
        "same_exact_v44_authorization_at_updates_zero_two_four": True,
        "exact_train_scene_ids": list(_TRAIN_SCENES),
        "exact_train_scene_count": 16,
        "train_question_count": 384,
        "train_changed_pair_unit_count": 25,
        "source_optimizer_files_or_states_loaded": False,
        "validation_oracle_and_final_access": False,
        "question_dependent_scene_processing_or_retrieval": False,
    }


def _authorization() -> dict[str, Any]:
    """Return the sole fixed successor authorization.

    Direction construction and scalar values are part of this immutable seal,
    so an implementation may not silently broaden the screen.
    """

    gradient_specs = [
        {
            "pair_id": "pair_000006",
            "question_key": "cfq_5c84a2c27d2be251",
            "side_index": 0,
            "role": "g5_candidate_direction_source",
        },
        {
            "pair_id": "pair_000016",
            "question_key": "cfq_699675ceeaf65406",
            "side_index": 1,
            "role": "diagnostic_only_never_a_candidate_direction",
        },
        {
            "pair_id": "pair_000006",
            "question_key": "cfq_0a79d507273195ef",
            "side_index": 0,
            "role": "diagnostic_only_never_a_candidate_direction",
        },
    ]
    alpha_grid = [0.125, 0.25, 0.5, 1.0, 2.0]
    return {
        "schema_version": 1,
        "authorization_id": "v46_train_only_checkpoint_gradient_diagnostic",
        "authorized": True,
        "only_exact_action": (
            "one_bounded_read_only_v46_train_checkpoint_gradient_diagnostic"
        ),
        "authorized_script": str(V46_DIAGNOSTIC),
        "authorized_test": str(V46_TEST),
        "authorized_report": str(V46_OUTPUT),
        "explicit_terminal_sha256_cli_required": True,
        "implementation_integrity": {
            "script_sha256": _PINNED_INPUTS[str(V46_DIAGNOSTIC)],
            "test_sha256": _PINNED_INPUTS[str(V46_TEST)],
            "config_sha256": _PINNED_INPUTS[str(DEFAULT_CONFIG)],
        },
        "invocation_contract": {
            "terminal_path": str(DEFAULT_OUTPUT),
            "required_cli_argument": "--expected-v45-terminal-sha256",
            "expected_value": "sha256_of_materialized_v45_terminal_passed_explicitly",
            "v46_must_not_embed_terminal_sha256": True,
            "v46_must_authenticate_terminal_bytes_and_exact_authorization": True,
        },
        "source": {
            "checkpoint": str(DEFAULT_CHECKPOINT_ROOT / "update_004"),
            "full_tensor_state_sha256": _STATE_HASHES["update_004"]["full"],
            "authorized_surface_state_sha256": _STATE_HASHES["update_004"][
                "authorized"
            ],
            "frozen_state_sha256": _STATE_HASHES["update_004"]["frozen"],
            "file_sha256": {
                name.removeprefix("update_004/"): digest
                for name, digest in _CHECKPOINT_FILES.items()
                if name.startswith("update_004/")
            },
            "update_002_is_authenticated_by_terminal_but_not_a_v46_probe_source": True,
        },
        "fixed_data_boundary": {
            "scene_ids": list(_TRAIN_SCENES),
            "scene_count": 16,
            "train_question_count": 384,
            "changed_pair_unit_count": 25,
            "broad_nll_row_count": 48,
            "blocking_file_access_audit_required": True,
            "all_occupied_blocks_processed": True,
            "complete_pre_question_scene_prefixes": True,
            "question_dependent_scene_processing": False,
            "question_dependent_retrieval": False,
        },
        "measurements": {
            "source_full_teacher_forced_pair_metrics": True,
            "source_fixed_full_48_row_broad_nll": True,
            "isolated_side_gradient_specs": gradient_specs,
            "gradient_loss_formula": "negative_selected_side_margin",
            "gradient_groups": {
                "scene_readout": ["block_cross_residual.w_o"],
                "query": list(_PARAMETER_NAMES[1:]),
            },
            "report_gradient_norms_and_pairwise_cosines": True,
        },
        "fresh_adam_sign_line": {
            "source_checkpoint": str(DEFAULT_CHECKPOINT_ROOT / "update_004"),
            "direction_source": {
                "gradient_id": "g5",
                "loss": (
                    "-side_margins[0] for pair_000006/"
                    "cfq_5c84a2c27d2be251 at exact update_004"
                ),
                "autograd_exact_at_source": True,
            },
            "direction_ids": ["g5_scene_sign", "g5_query_sign", "g5_both_sign"],
            "direction_definitions": {
                "g5_scene_sign": "scene_readout_only_sign(g5)",
                "g5_query_sign": "query_only_sign(g5)",
                "g5_both_sign": "scene_readout_and_query_sign(g5)",
            },
            "candidate_formula": "float32_P0-alpha*lr_group*sign(g5)",
            "scene_readout_learning_rate": 1.0e-5,
            "query_learning_rate": 8.0e-6,
            "alpha_grid": alpha_grid,
            "exact_candidate_count": 15,
            "full_25_unit_teacher_metrics_per_candidate": True,
            "full_fixed_48_row_broad_nll_per_candidate": True,
            "weight_decay": 0.0,
            "in_memory_only": True,
            "exact_u4_restoration_before_and_after_every_probe": True,
            "full_tensor_hash_restored_after_every_probe": _STATE_HASHES[
                "update_004"
            ]["full"],
            "adaptive_direction_or_scalar_selection": False,
            "diagnostic_gradient_q699_and_q0a79_used_as_directions": False,
        },
        "forbidden_actions": {
            "optimizer_state_file_open": True,
            "optimizer_state_deserialization": True,
            "optimizer_state_loading": True,
            "optimizer_step": True,
            "candidate_or_checkpoint_write": True,
            "parameter_state_persist": True,
            "validation_access": True,
            "oracle_access": True,
            "final_test_access": True,
            "selector_execution": True,
            "runtime_promotion": True,
            "chat_promotion": True,
            "embodied_promotion": True,
        },
        "scope": {
            "train_only": True,
            "read_only_except_single_report": True,
            "report_only_output": True,
            "no_candidate_is_authorized_by_this_diagnostic": True,
            "new_terminal_seal_required_for_any_successor": True,
            "validation_access_authorized": False,
            "oracle_access_authorized": False,
            "final_test_access_authorized": False,
            "selector_execution_authorized": False,
            "runtime_promotion_authorized": False,
            "chat_promotion_authorized": False,
        },
    }


def audit_v45_retention_repair(
    config_path: str | Path = DEFAULT_CONFIG,
    checkpoint_root: str | Path = DEFAULT_CHECKPOINT_ROOT,
) -> dict[str, Any]:
    """Authenticate immutable V45 train-only negative evidence."""

    config_file = _resolve(config_path)
    root = _resolve(checkpoint_root)
    if config_file != _resolve(DEFAULT_CONFIG):
        raise ValueError("V45 terminal config path is pinned")
    if root != _resolve(DEFAULT_CHECKPOINT_ROOT):
        raise ValueError("V45 terminal checkpoint root is pinned")
    inputs = _authenticate_inputs()
    v44 = _read_json(
        _resolve(V44_TERMINAL),
        _PINNED_INPUTS[str(V44_TERMINAL)],
        "V44 terminal seal",
    )
    v44_source = _authenticate_v44_source(v44)
    inventory = _authenticate_inventory(root)
    metadata = {
        f"update_{step:03d}": _read_json(
            root / f"update_{step:03d}/metadata.json",
            _CHECKPOINT_FILES[f"update_{step:03d}/metadata.json"],
            f"V45 update-{step} metadata",
        )
        for step in (0, 2, 4)
    }
    for step, label in ((0, "update_000"), (2, "update_002"), (4, "update_004")):
        value = metadata[label]
        stage = _mapping(value.get("v45_retention_repair"), f"V45 {label} stage")
        if (
            value.get("schema_version") != 1
            or value.get("config_hash") != _CONFIG_HASH
            or value.get("epoch") != step
            or value.get("optimizer_step") != step
            or stage.get("optimizer_step") != step
        ):
            raise ValueError(f"V45 {label} checkpoint identity changed")
    runtime = {
        label: _authenticate_runtime(root, metadata[label], step=step)
        for step, label in ((0, "update_000"), (2, "update_002"), (4, "update_004"))
    }
    tensors = _authenticate_tensors(root)
    history, gate = _authenticate_history_and_gate(metadata)
    provenance = _authenticate_provenance(metadata, v44)
    authorization = _authorization()
    return {
        "schema_version": 1,
        "artifact": "v45_retention_repair_terminal_gate",
        "passed": True,
        "terminal_conclusion": "update4_train_only_gate_failed_stop_is_final",
        "input_integrity": inputs,
        "v44_source_authentication": v44_source,
        "checkpoint_inventory": inventory,
        "tensor_transition": tensors,
        "runtime_metadata_audit": runtime,
        "history_audit": history,
        "update4_gate_replay": gate,
        "v45_provenance": provenance,
        "execution_conclusion": {
            "updates_one_through_four_executed": True,
            "update4_checkpoint_persisted": True,
            "update4_gate_passed": False,
            "update_five_executed": False,
            "update6_checkpoint_present": False,
            "update8_checkpoint_or_gate_present": False,
            "frozen_state_exact": True,
            "validation_loaded": False,
            "oracle_loaded": False,
            "final_test_loaded": False,
            "selector_executed": False,
            "runtime_promoted": False,
        },
        "conditional_successor_authorization": authorization,
        "only_exact_successor_authorized": (
            "v46_train_only_checkpoint_gradient_diagnostic"
        ),
        "v46_train_only_checkpoint_gradient_diagnostic_authorized": True,
        "arbitrary_training_authorized": False,
        "resume_v45_training_authorized": False,
        "candidate_checkpoint_write_authorized": False,
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
            "optimizer_files_bytes_hashed_only": True,
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
        raise ValueError("V45 terminal output path is pinned")
    if path.is_symlink() or path.exists():
        raise FileExistsError(f"V45 terminal is one-shot and will not overwrite {path}")
    report = audit_v45_retention_repair(config_path, checkpoint_root)
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


__all__ = ["audit_v45_retention_repair", "write_report"]
