"""Seal V47's negative update-four gate without touching restricted data.

This is an immutable, offline audit of the exact bounded V47 train-only run.
It authenticates code/config bytes, the predecessor authorization, every
persisted checkpoint file, sanitized runtime metadata, tensor transitions,
history prefixes, and the recorded update-two/update-four gates.  Adapter
tensors are loaded on CPU only for hashing and equality checks.  Optimizer
files are byte-hashed but never deserialized.  Gemma, QA, maps, validation,
oracle, deferred-final, selector, chat, and embodied inputs are never loaded.

The failed V47 gate does not authorize more training or promotion.  Its only
possible successor is one fixed, report-only, train-split V48 diagnostic.  The
V48 module and test now exist with stable bytes, but their hashes remain
explicit invocation arguments to avoid a source/seal hash cycle.  Placeholder
mode audits V47 but cannot materialize this terminal report.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
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

DEFAULT_CONFIG = Path("configs/experiments/gemma4_diverse28_book_continuation_v47.yaml")
DEFAULT_CHECKPOINT_ROOT = Path("data_gemma4/checkpoints/gemma4_v47_book_continuation_l14_query")
DEFAULT_OUTPUT = Path("reports/gemma4/metrics/v47_book_continuation_terminal_gate.json")
V46_TERMINAL = Path("reports/gemma4/metrics/v46_v45_u4_lost_side_terminal_gate.json")
V46_REPORT = Path("reports/gemma4/metrics/v46_v45_u4_lost_side_no_step_diagnostic.json")
PROTECTED_REPORT = Path(
    "reports/gemma4/metrics/training_selection_gemma4_color_mirror_full_vocab_v11_resume36.json"
)
V47_TRAINER = Path("src/semantic_3d_chat/training/train_book_continuation_v47.py")
V47_TEST = Path("tests/test_train_book_continuation_v47.py")
V48_DIAGNOSTIC = Path("src/semantic_3d_chat/evaluation/v48_v47_u4_dual_margin_screen.py")
V48_TEST = Path("tests/test_v48_v47_u4_dual_margin_screen.py")
V48_OUTPUT = Path("reports/gemma4/metrics/v48_v47_u4_dual_margin_no_step_diagnostic.json")

IMPLEMENTATION_SHA256_PLACEHOLDER = "PENDING_V48_IMPLEMENTATION_SHA256"
_HEX64 = re.compile(r"[0-9a-f]{64}")
_V48_AUTHORIZATION_ID = "v48_v47_u4_dual_margin_no_step_diagnostic"

_PINNED_INPUTS = {
    str(V46_TERMINAL): ("de66c9844786c8718399c75162e0b13313b778c1b7d5fa7edcd4133d4d31b60d"),
    str(V46_REPORT): ("ce48a1fd484fa5dab71c76a2dd3e39194dd6964e068d6762925a02fb73f6aee6"),
    str(PROTECTED_REPORT): ("c0086f66edbb8854a7938e09c57535bfd47100adbaf3b3c95eeb4b08014ce2f8"),
    str(DEFAULT_CONFIG): ("6b15813237b217d8daad446c799127215bbb1366f2b442b61e975787efa4f6b7"),
    str(V47_TRAINER): ("971fdbaf2f6e6b22dc27b83cfa0f6604c2c1145d92509c06bc98410f6927ea22"),
    str(V47_TEST): ("86c3e0e49b0c42b1161227cdc42ff577afca6b37bd0db183ca1648f23125aedd"),
}
_CONFIG_HASH = "9c79d3cb5af0"
_CHECKPOINT_FILES = {
    "update_000/adapter.safetensors": (
        "c47bbb9bacbb5bc8178e9a1797ec47b04ee4a3709042c6b30f6935eacc4686f0"
    ),
    "update_000/metadata.json": (
        "e76e8a905af53fb082684000a6a6e16845b79e0795b6dbb047a2703245198574"
    ),
    "update_000/runtime_metadata.json": (
        "01e645b82c5e533dd2319ef8a97171437b149c4e1ef86201f83fdb22de047987"
    ),
    "update_002/adapter.safetensors": (
        "8cb88aafa34ce5b0021cc74a240cce2275d11a1241fce4149af81671f13465fd"
    ),
    "update_002/metadata.json": (
        "e39e6f624b3f7b5204ac587ce21e21ec77e03239750164fcc9c2b583b5de091a"
    ),
    "update_002/optimizer.pt": ("e6960f67de249a9e485c6126a860fadf038f8e70642392154b06c4c3e910608b"),
    "update_002/runtime_metadata.json": (
        "70840d1fc29daff0ce2327fb7a2ab6c8e7f16cbcc9b8d95ae8c2bb2db27cffdc"
    ),
    "update_004/adapter.safetensors": (
        "8f903f5d1ba93d37ccd6204e3b58c9a5529ff9ee2b74edca0787ecb5a2c62c66"
    ),
    "update_004/metadata.json": (
        "c6affe7f60c094580e2ea5f5d1330f475bf359e0a3a58bfc3bf3b3ada1de0be1"
    ),
    "update_004/optimizer.pt": ("fe66be9cae13951fbfc217e0c512e43366c347181457c9e551230a9d6001db80"),
    "update_004/runtime_metadata.json": (
        "4e3a1af91642c9f2adb0b3e43997455a1aea31f86bf45618459d6005a68d4bbf"
    ),
}
_CHECKPOINT_MANIFEST_SHA256 = "0c457e644a41e4e6af0e31fb64a5bb60655a46f9828e664aae3119256d6e8b64"
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
        "full": "1d5adc1fb0d7a895056b77d38c8a12aba95c9997ec8a94edf68673f9c58fb954",
        "authorized": ("d60b665d9a970433b2ed59e6769b9114468bef608b98eae828268101d39db56c"),
        "frozen": "31cb215cf0d1623886d4a79203c501912e66537021da53dd711124abdc2e36fa",
    },
    "update_002": {
        "full": "16c6432bcb4f74d81577e1857757d5fdacc8b2ffe79dca619646bafa6ca36996",
        "authorized": ("21aea698f93f0a70d4aaadd207ede168e9e8c70dd77a3343b0570a4e05f3cf51"),
        "frozen": "31cb215cf0d1623886d4a79203c501912e66537021da53dd711124abdc2e36fa",
    },
    "update_004": {
        "full": "adfc0400d1a3bb49b278cd3012ab571d01465f2380881f986c085a25474276e5",
        "authorized": ("a23de4988774a966c0d7aac378ede5d15a3fa1d96093c5039f181a62b0bb09b0"),
        "frozen": "31cb215cf0d1623886d4a79203c501912e66537021da53dd711124abdc2e36fa",
    },
}
_AUTHORIZED_HISTORY_HASHES = {
    0: _STATE_HASHES["update_000"]["authorized"],
    1: "3884b84fef87a61468f5e5c0df5581adf030b906ee908b19c68e9e3a3f4a764b",
    2: _STATE_HASHES["update_002"]["authorized"],
    3: "63388b92dac73b05bf64db08bc2428dc3d23ba2d5a99aac2db3cfc0725643191",
    4: _STATE_HASHES["update_004"]["authorized"],
}
_QUERY_HASHES = {
    0: "493f0f2f1034c93cee9164407650f5f96341aa0d0cf448285f420ce8bb721e7e",
    1: "31e16a8fe80edcea01d48c85b9ecc8e43698d09bdbd57d05ee7446736707897e",
    2: "c8911efc4673653db8e8705d69334e7577cc9f99e35487bcf1cf51bb37946ee1",
    3: "3a2d05bfc930b5ba61d85369839ac1cfece6f30a2996bbd11f9262f9a3ba23fc",
    4: "0c3f6a7015b373642e903f372963afc0ad2d30b70def7df438d879df960f67f0",
}
_SCENE_READOUT_HASHES = {
    0: "4d7c17c6cf88c1ff99c52d91f19cff061e4370c73bb0cc1bc4bda149f592dbce",
    1: "f7294823acd0a82657c19046e27b1ec014605d0726c71d349a6d4209bd4cd1a8",
    2: "efa7dbaace12edd0e436fefd2c0a041bac4b7ce7a13a2b1f354a40866ef33f79",
    3: "d28a8d2686614565574b4c65a4b843d512a9e084c5971cd5f0eb41bc1f4a90b9",
    4: "8b2fa6d1a45e21b52b394338e43c8fbd250b68d4cca421139d46b1125bf2f3bd",
}
_PREFIX_TRUST_RMS = {
    0: 0.0,
    1: 0.00044878781773149967,
    2: 0.0008764256490394473,
    3: 0.0011487154988572001,
    4: 0.001376520493067801,
}
_BROAD_QUESTION_IDS = (
    "q_000099",
    "q_000138",
    "q_000053",
    "q_000089",
)
_TARGET_PAIR = "pair_000015"
_TARGET_QUESTION_KEY = "cfq_163eb92339ad35a5"
_SCHEDULE_SHA256 = "5284e4ff377fc6ea4668c73e398929d8589f0ef45953e0da25b78c82508d5ed3"
_TRAIN_SCENES = [
    *(f"scene_{index:06d}" for index in range(11, 19)),
    *(f"scene_{index:06d}" for index in range(31, 39)),
]
_SOURCE_BROAD_NLL = 2.91504575808843
_UPDATE2_BROAD_NLL = 2.918472488721212
_UPDATE4_BROAD_NLL = 2.9172145972649255
_BROAD_NLL_MAXIMUM = 2.9213306349515915
_ORIGINAL_V41_PRIORITY_DEFICIT = 31.113729119300842
_UPDATE4_PRIORITY_DEFICIT = 30.386213302612305
_UPDATE4_PRIORITY_IMPROVEMENT = 0.7275158166885376
_UPDATE4_TRUST_RMS = _PREFIX_TRUST_RMS[4]


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
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _mapping(value: object, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{field} must be a mapping")
    return value


def _sequence(value: object, field: str) -> Sequence[Any]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise TypeError(f"{field} must be a sequence")
    return value


def _finite(value: object, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{field} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{field} must be finite")
    return result


def _lower_hex64(value: object, field: str) -> str:
    if not isinstance(value, str) or _HEX64.fullmatch(value) is None:
        raise ValueError(f"{field} must be 64 lowercase hexadecimal digits")
    return value


def _locked_file(path: Path, expected: str, field: str) -> None:
    if path.is_symlink() or not path.is_file():
        raise FileNotFoundError(f"{field} must be a real file: {path}")
    observed = _sha256(path)
    if observed != expected:
        raise ValueError(f"{field} bytes changed: expected {expected}, observed {observed}")


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
        raise ValueError("V47 normalized config hash changed")
    return {
        "file_sha256": observed,
        "normalized_config_hash": _CONFIG_HASH,
        "protected_v11_access": "bytes_hashed_only",
        "v46_diagnostic_access": "bytes_hashed_only",
    }


def _authenticate_predecessor(terminal: Mapping[str, Any]) -> dict[str, Any]:
    authorization = _mapping(
        terminal.get("conditional_successor_authorization"),
        "V46 V47 authorization",
    )
    integrity = _mapping(
        authorization.get("implementation_integrity"),
        "V46 V47 implementation integrity",
    )
    source = _mapping(authorization.get("source"), "V46 V47 source")
    training = _mapping(authorization.get("training"), "V46 V47 training")
    scope = _mapping(authorization.get("scope"), "V46 V47 scope")
    checks = {
        "identity": terminal.get("schema_version") == 1
        and terminal.get("artifact") == "v46_v45_u4_lost_side_terminal_gate"
        and terminal.get("passed") is True,
        "only_successor": terminal.get("only_exact_successor_authorized")
        == "v47_exact_book_support_continuation",
        "authorization": authorization.get("authorization_id")
        == "v47_exact_book_support_continuation"
        and authorization.get("authorized") is True,
        "paths": authorization.get("authorized_config") == str(DEFAULT_CONFIG)
        and authorization.get("authorized_trainer") == str(V47_TRAINER)
        and authorization.get("authorized_test") == str(V47_TEST)
        and authorization.get("authorized_output") == str(DEFAULT_CHECKPOINT_ROOT),
        "implementation": integrity
        == {
            "config_sha256": _PINNED_INPUTS[str(DEFAULT_CONFIG)],
            "trainer_sha256": _PINNED_INPUTS[str(V47_TRAINER)],
            "test_sha256": _PINNED_INPUTS[str(V47_TEST)],
        },
        "source": source.get("candidate_id") == "g5_both_sign_alpha_1p0"
        and source.get("candidate_full_tensor_state_sha256") == _STATE_HASHES["update_000"]["full"]
        and source.get("candidate_authorized_surface_sha256")
        == _STATE_HASHES["update_000"]["authorized"]
        and source.get("candidate_frozen_state_sha256") == _STATE_HASHES["update_000"]["frozen"]
        and source.get("v46_report_sha256") == _PINNED_INPUTS[str(V46_REPORT)],
        "schedule": training.get("optimizer_steps") == 4
        and training.get("checkpoint_steps") == [0, 2, 4]
        and training.get("target_question_keys") == [_TARGET_QUESTION_KEY] * 4
        and training.get("broad_question_ids") == list(_BROAD_QUESTION_IDS),
        "boundary": scope.get("train_only") is True
        and scope.get("validation_access_authorized") is False
        and scope.get("oracle_access_authorized") is False
        and scope.get("final_test_access_authorized") is False
        and scope.get("selector_execution_authorized") is False
        and scope.get("runtime_promotion_authorized") is False,
    }
    if not all(checks.values()):
        raise ValueError(f"V46 exact V47 authorization changed: {checks}")
    return {
        "terminal_path": str(V46_TERMINAL),
        "terminal_sha256": _PINNED_INPUTS[str(V46_TERMINAL)],
        "v46_report_sha256": _PINNED_INPUTS[str(V46_REPORT)],
        "authorization_id": "v47_exact_book_support_continuation",
        "authorization_checks": checks,
        "exact_v47_authorization_authenticated": True,
    }


def _authenticate_inventory(root: Path) -> dict[str, Any]:
    if root.is_symlink() or not root.is_dir():
        raise FileNotFoundError("V47 checkpoint root must be a real directory")
    observed_root = sorted(path.name for path in root.iterdir())
    if observed_root != sorted(_DIRECTORY_INVENTORY):
        raise ValueError("V47 checkpoint root inventory changed")
    for directory_name, expected_entries in _DIRECTORY_INVENTORY.items():
        directory = root / directory_name
        if directory.is_symlink() or not directory.is_dir():
            raise FileNotFoundError(f"V47 {directory_name} must be a real directory")
        if sorted(path.name for path in directory.iterdir()) != expected_entries:
            raise ValueError(f"V47 {directory_name} inventory changed")
    for relative, expected in _CHECKPOINT_FILES.items():
        _locked_file(root / relative, expected, f"V47 {relative}")
    if _canonical_sha256(_CHECKPOINT_FILES) != _CHECKPOINT_MANIFEST_SHA256:
        raise ValueError("V47 pinned checkpoint manifest hash changed")
    return {
        "root": _relative(root),
        "root_entries": observed_root,
        "directory_entries": dict(_DIRECTORY_INVENTORY),
        "file_sha256": dict(_CHECKPOINT_FILES),
        "manifest_sha256": _CHECKPOINT_MANIFEST_SHA256,
        "checkpoint_steps_persisted": [0, 2, 4],
        "update_006_absent": not (root / "update_006").exists(),
        "update_008_absent": not (root / "update_008").exists(),
        "optimizer_files_bytes_hashed_only": True,
        "optimizer_state_deserialized": False,
    }


def _authenticate_runtime(
    root: Path,
    metadata: Mapping[str, Any],
    *,
    step: int,
) -> dict[str, Any]:
    relative = f"update_{step:03d}/runtime_metadata.json"
    runtime = _read_json(
        root / relative,
        _CHECKPOINT_FILES[relative],
        f"V47 update-{step} runtime metadata",
    )
    validate_runtime_checkpoint_metadata(runtime)
    if runtime != runtime_checkpoint_metadata(metadata):
        raise ValueError(f"V47 update-{step} runtime metadata is not freshly sanitized")
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
            raise ValueError(f"V47 {label} tensor count changed")
        if any(not torch.isfinite(value).all() for value in tensors.values()):
            raise ValueError(f"V47 {label} contains a nonfinite tensor")
        if not all(name in tensors for name in _PARAMETER_NAMES):
            raise ValueError(f"V47 {label} authorized tensor inventory changed")
        shapes = tuple(tuple(tensors[name].shape) for name in _PARAMETER_NAMES)
        counts = tuple(int(tensors[name].numel()) for name in _PARAMETER_NAMES)
        if shapes != _PARAMETER_SHAPES or counts != _PARAMETER_COUNTS:
            raise ValueError(f"V47 {label} authorized tensor shape/count changed")
        authorized = {name: tensors[name] for name in _PARAMETER_NAMES}
        frozen = {name: value for name, value in tensors.items() if name not in authorized}
        observed = {
            "full": tensor_state_sha256(tensors),
            "authorized": tensor_state_sha256(authorized),
            "frozen": tensor_state_sha256(frozen),
        }
        if observed != _STATE_HASHES[label]:
            raise ValueError(f"V47 {label} tensor-state hash changed: {observed}")
        states[label] = tensors
    baseline = states["update_000"]
    if any(set(state) != set(baseline) for state in states.values()):
        raise ValueError("V47 tensor names differ between checkpoints")
    changed_by_step: dict[str, list[str]] = {}
    for label in ("update_002", "update_004"):
        changed = sorted(
            name for name in baseline if not torch.equal(baseline[name], states[label][name])
        )
        if changed != sorted(_PARAMETER_NAMES):
            raise ValueError(f"V47 {label} changed tensors escaped authorization")
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
        "frozen_state_bit_exact_through_update_four": True,
        "all_tensors_finite": True,
    }


def _priority_side_deficit(pair_metrics: Mapping[str, Any]) -> float:
    rows = _sequence(pair_metrics.get("units"), "V47 pair metric units")
    if len(rows) != 25:
        raise ValueError("V47 priority deficit requires exactly 25 units")
    counts = {"book_support": 0, "picture_support": 0}
    deficit = 0.0
    for value in rows:
        row = _mapping(value, "V47 pair metric unit")
        family = str(row.get("family"))
        if family not in counts:
            continue
        margins = _sequence(row.get("side_margins"), "V47 side margins")
        if len(margins) != 2:
            raise ValueError("V47 priority row must have two side margins")
        deficit += sum(max(0.0, 0.5 - _finite(margin, "V47 margin")) for margin in margins)
        counts[family] += 1
    if counts != {"book_support": 4, "picture_support": 4}:
        raise ValueError("V47 priority-family inventory changed")
    return deficit


def _unit_by_key(metrics: Mapping[str, Any], question_key: str) -> Mapping[str, Any]:
    matches = [
        _mapping(value, "V47 pair unit")
        for value in _sequence(metrics.get("units"), "V47 pair units")
        if isinstance(value, Mapping) and value.get("question_key") == question_key
    ]
    if len(matches) != 1:
        raise ValueError(f"V47 expected one unit for {question_key}; got {len(matches)}")
    return matches[0]


def _replay_update4_gate(metrics: Mapping[str, Any], row4: Mapping[str, Any]) -> dict[str, Any]:
    if (
        metrics.get("training_scenes_only") is not True
        or metrics.get("validation_qa_loaded") is not False
        or metrics.get("unit_count") != 25
        or metrics.get("side_count") != 50
    ):
        raise ValueError("V47 update-four metrics crossed the train-only boundary")
    teacher_expected = {
        "complete_units": 8,
        "positive_sides": 33,
        "cross_prefix_complete_units": 17,
        "complete_physical_pair_coverage": 4,
    }
    if any(metrics.get(name) != value for name, value in teacher_expected.items()):
        raise ValueError("V47 update-four teacher-forced counts changed")
    complete_family = dict(
        _mapping(metrics.get("complete_units_by_family"), "V47 complete family counts")
    )
    cross_family = dict(
        _mapping(
            metrics.get("cross_prefix_complete_units_by_family"),
            "V47 cross family counts",
        )
    )
    if complete_family != {"book_support": 0, "mirror_lr": 1, "picture_support": 0}:
        raise ValueError("V47 update-four complete family counts changed")
    if cross_family != {"book_support": 1, "mirror_lr": 4, "picture_support": 2}:
        raise ValueError("V47 update-four cross family counts changed")
    deficit = _priority_side_deficit(metrics)
    if deficit != _UPDATE4_PRIORITY_DEFICIT:
        raise ValueError("V47 update-four priority deficit changed")
    improvement = _ORIGINAL_V41_PRIORITY_DEFICIT - deficit
    if improvement != _UPDATE4_PRIORITY_IMPROVEMENT:
        raise ValueError("V47 update-four priority improvement changed")
    lost: list[dict[str, Any]] = []
    for pair_id, question_key, side_index, expected in (
        ("pair_000006", "cfq_5c84a2c27d2be251", 0, 0.0625),
        ("pair_000016", "cfq_699675ceeaf65406", 1, 0.0),
    ):
        unit = _unit_by_key(metrics, question_key)
        margins = _sequence(unit.get("side_margins"), "V47 lost-side margins")
        margin = _finite(margins[side_index], "V47 lost-side margin")
        if unit.get("pair_id") != pair_id or margin != expected:
            raise ValueError(f"V47 lost-side evidence changed: {question_key}")
        lost.append(
            {
                "pair_id": pair_id,
                "question_key": question_key,
                "side_index": side_index,
                "margin": margin,
                "strictly_positive": margin > 0.0,
            }
        )
    greedy = _mapping(row4.get("training_greedy_metrics"), "V47 greedy metrics")
    greedy_expected = {
        "broad_exact_correct": 23,
        "broad_row_count": 48,
        "changed_rows_exact_correct": 24,
        "changed_row_count": 50,
        "changed_unit_count": 25,
        "complete_units": 4,
        "complete_physical_pair_coverage": 3,
    }
    if any(greedy.get(name) != value for name, value in greedy_expected.items()):
        raise ValueError("V47 update-four greedy counts changed")
    greedy_family = dict(
        _mapping(greedy.get("complete_units_by_family"), "V47 greedy family counts")
    )
    if greedy_family != {"book_support": 0, "mirror_lr": 2, "picture_support": 0}:
        raise ValueError("V47 update-four greedy family counts changed")
    broad_nll = _finite(row4.get("broad_diagnostic_nll"), "V47 broad NLL")
    trust_rms = _finite(row4.get("candidate_prefix_trust_rms"), "V47 prefix RMS")
    checks = {
        "book_complete_units_at_least_1": complete_family["book_support"] >= 1,
        "book_cross_prefix_complete_units_at_least_1": cross_family["book_support"] >= 1,
        "both_authorized_parameter_groups_changed": row4.get("scene_readout_state_changed") is True
        and row4.get("query_state_changed") is True,
        "both_lost_side_margins_remain_strictly_positive": all(
            value["strictly_positive"] for value in lost
        ),
        "broad_greedy_exact_correct_at_least_23_of_48": greedy["broad_exact_correct"] >= 23
        and greedy["broad_row_count"] == 48,
        "broad_nll_at_most_authorized_maximum": broad_nll <= _BROAD_NLL_MAXIMUM,
        "complete_physical_pair_id_coverage_at_least_5": metrics["complete_physical_pair_coverage"]
        >= 5,
        "frozen_state_exact": row4.get("frozen_state_sha256")
        == _STATE_HASHES["update_000"]["frozen"],
        "mirror_complete_units_at_least_2": complete_family["mirror_lr"] >= 2,
        "priority_teacher_deficit_improved_at_least_0_5_vs_original_v41_u0": (improvement >= 0.5),
        "query_state_changed": row4.get("query_state_changed") is True,
        "scene_readout_state_changed": row4.get("scene_readout_state_changed") is True,
        "source_prefix_trust_rms_at_most_0_002": trust_rms <= 0.002,
        "teacher_complete_units_at_least_10": metrics["complete_units"] >= 10,
        "teacher_cross_complete_units_at_least_17": metrics["cross_prefix_complete_units"] >= 17,
        "teacher_positive_sides_at_least_35": metrics["positive_sides"] >= 35,
        "train_greedy_complete_units_at_least_5": greedy["complete_units"] >= 5,
        "update2_integrity_gate_passed": row4.get("update2_integrity_gate", {}).get("passed")
        is True,
    }
    expected_failed = sorted(
        [
            "book_complete_units_at_least_1",
            "both_lost_side_margins_remain_strictly_positive",
            "complete_physical_pair_id_coverage_at_least_5",
            "mirror_complete_units_at_least_2",
            "teacher_complete_units_at_least_10",
            "teacher_positive_sides_at_least_35",
            "train_greedy_complete_units_at_least_5",
        ]
    )
    failed = sorted(name for name, passed in checks.items() if not passed)
    if failed != expected_failed:
        raise ValueError(f"V47 update-four failure set changed: {failed}")
    gate = _mapping(row4.get("update4_final_train_only_gate"), "V47 final gate")
    retention = _mapping(gate.get("retention_diagnostics"), "V47 retention gate")
    if (
        gate.get("checks") != checks
        or gate.get("passed") is not False
        or gate.get("priority_teacher_side_deficit") != deficit
        or gate.get("priority_teacher_side_deficit_improvement_vs_original_v41_u0") != improvement
        or gate.get("broad_nll") != _UPDATE4_BROAD_NLL
        or gate.get("broad_nll_maximum") != _BROAD_NLL_MAXIMUM
        or gate.get("source_prefix_trust_rms") != _UPDATE4_TRUST_RMS
        or gate.get("full_train_pair_unit_count") != 25
        or gate.get("full_broad_nll_row_count") != 48
        or gate.get("training_greedy_metrics") != greedy
        or retention.get("lost_sides") != lost
        or retention.get("both_lost_sides_strictly_positive") is not False
        or gate.get("training_scenes_only") is not True
        or gate.get("validation_qa_loaded") is not False
        or gate.get("selector_execution_authorized") is not False
    ):
        raise ValueError("V47 recorded update-four gate changed")
    return {
        "passed": False,
        "failed_checks": failed,
        "passing_checks": sorted(name for name, passed in checks.items() if passed),
        "checks": checks,
        "teacher_forced": {
            **teacher_expected,
            "complete_units_by_family": complete_family,
            "cross_prefix_complete_units_by_family": cross_family,
            "teacher_complete_threshold": 10,
            "teacher_positive_threshold": 35,
            "teacher_cross_threshold": 17,
            "physical_pair_threshold": 5,
            "mirror_complete_threshold": 2,
            "book_complete_threshold": 1,
            "book_cross_threshold": 1,
        },
        "greedy": {
            **greedy_expected,
            "complete_units_by_family": greedy_family,
            "complete_units_threshold": 5,
            "broad_exact_threshold": 23,
        },
        "source_broad_nll": _SOURCE_BROAD_NLL,
        "update4_broad_nll": broad_nll,
        "broad_nll_maximum": _BROAD_NLL_MAXIMUM,
        "original_v41_priority_side_deficit": _ORIGINAL_V41_PRIORITY_DEFICIT,
        "update4_priority_side_deficit": deficit,
        "priority_side_deficit_improvement": improvement,
        "update4_source_prefix_trust_rms": trust_rms,
        "lost_side_evidence": lost,
        "stopped_before_update_five": True,
    }


def _authenticate_history_and_gate(
    metadata: Mapping[str, Mapping[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    histories = {
        label: list(_sequence(value.get("history"), f"V47 {label} history"))
        for label, value in metadata.items()
    }
    lengths = [len(histories[label]) for label in ("update_000", "update_002", "update_004")]
    if lengths != [1, 3, 5]:
        raise ValueError(f"V47 checkpoint history lengths changed: {lengths}")
    if histories["update_002"][:1] != histories["update_000"]:
        raise ValueError("V47 update-two history does not preserve update zero")
    if histories["update_004"][:3] != histories["update_002"]:
        raise ValueError("V47 update-four history does not preserve update two")
    history = histories["update_004"]
    for step, value in enumerate(history):
        row = _mapping(value, f"V47 history row {step}")
        if (
            row.get("optimizer_update") != step
            or row.get("saved_checkpoint") is not (step in (0, 2, 4))
            or row.get("authorized_state_sha256") != _AUTHORIZED_HISTORY_HASHES[step]
            or row.get("query_state_sha256") != _QUERY_HASHES[step]
            or row.get("scene_readout_state_sha256") != _SCENE_READOUT_HASHES[step]
            or row.get("frozen_state_sha256") != _STATE_HASHES["update_000"]["frozen"]
            or row.get("candidate_prefix_trust_rms") != _PREFIX_TRUST_RMS[step]
            or row.get("validation_qa_loaded") is not False
            or row.get("oracle_environment_files_loaded") is not False
        ):
            raise ValueError(f"V47 history step {step} escaped its exact boundary")
        if step > 0 and (
            row.get("target_pair_id") != _TARGET_PAIR
            or row.get("target_question_key") != _TARGET_QUESTION_KEY
            or row.get("broad_question_id") != _BROAD_QUESTION_IDS[step - 1]
            or row.get("true_optimizer_step") is not True
        ):
            raise ValueError(f"V47 history step {step} schedule changed")
    row0 = _mapping(history[0], "V47 update-zero row")
    metrics0 = _mapping(row0.get("candidate_pair_metrics"), "V47 candidate metrics")
    if (
        metrics0.get("unit_count") != 25
        or metrics0.get("side_count") != 50
        or metrics0.get("complete_units") != 9
        or metrics0.get("positive_sides") != 34
        or metrics0.get("cross_prefix_complete_units") != 18
        or metrics0.get("complete_physical_pair_coverage") != 4
        or row0.get("candidate_broad_train_nll") != _SOURCE_BROAD_NLL
    ):
        raise ValueError("V47 authenticated candidate behavior changed")
    row2 = _mapping(history[2], "V47 update-two row")
    metrics2 = _mapping(row2.get("pair_metrics"), "V47 update-two metrics")
    gate2 = _mapping(row2.get("update2_integrity_gate"), "V47 update-two gate")
    hard2 = _mapping(gate2.get("hard_checks"), "V47 update-two hard checks")
    if (
        metrics2.get("complete_units") != 8
        or metrics2.get("positive_sides") != 32
        or metrics2.get("cross_prefix_complete_units") != 18
        or metrics2.get("complete_physical_pair_coverage") != 3
        or row2.get("broad_diagnostic_nll") != _UPDATE2_BROAD_NLL
        or gate2.get("passed") is not True
        or set(hard2.values()) != {True}
        or gate2.get("training_scenes_only") is not True
        or gate2.get("validation_qa_loaded") is not False
    ):
        raise ValueError("V47 update-two integrity gate changed")
    row4 = _mapping(history[4], "V47 update-four row")
    metrics4 = _mapping(row4.get("pair_metrics"), "V47 update-four metrics")
    replay = _replay_update4_gate(metrics4, row4)
    stage4 = _mapping(
        metadata["update_004"].get("v47_book_continuation"),
        "V47 update-four stage",
    )
    if stage4.get("update2_integrity_gate") != gate2:
        raise ValueError("V47 stage update-two gate differs from history")
    if stage4.get("update4_final_train_only_gate") != row4.get("update4_final_train_only_gate"):
        raise ValueError("V47 stage update-four gate differs from history")
    return (
        {
            "optimizer_updates_executed": [1, 2, 3, 4],
            "checkpoint_steps_persisted": [0, 2, 4],
            "history_lengths": lengths,
            "history_prefixes_bit_exact": True,
            "history_frozen_hash_exact_every_step": True,
            "authorized_state_sha256_by_update": {
                str(step): value for step, value in _AUTHORIZED_HISTORY_HASHES.items()
            },
            "query_state_sha256_by_update": {
                str(step): value for step, value in _QUERY_HASHES.items()
            },
            "scene_readout_state_sha256_by_update": {
                str(step): value for step, value in _SCENE_READOUT_HASHES.items()
            },
            "prefix_trust_rms_by_update": {
                str(step): value for step, value in _PREFIX_TRUST_RMS.items()
            },
            "fixed_target_pair": _TARGET_PAIR,
            "fixed_target_question_key": _TARGET_QUESTION_KEY,
            "fixed_broad_question_ids": list(_BROAD_QUESTION_IDS),
            "update2_integrity_gate_passed": True,
            "no_update_after_four_persisted": True,
        },
        replay,
    )


def _authenticate_provenance(
    metadata: Mapping[str, Mapping[str, Any]],
    predecessor: Mapping[str, Any],
) -> dict[str, Any]:
    authorization = _mapping(
        predecessor.get("conditional_successor_authorization"),
        "V46 V47 authorization",
    )
    for label, value in metadata.items():
        stage = _mapping(value.get("v47_book_continuation"), f"V47 {label} stage")
        surface = _mapping(stage.get("trainable_surface"), f"V47 {label} surface")
        if (
            stage.get("conditional_authorization") != authorization
            or stage.get("conditional_v46_terminal_gate")
            != {
                "path": str(V46_TERMINAL),
                "sha256": _PINNED_INPUTS[str(V46_TERMINAL)],
            }
            or stage.get("v46_report")
            != {
                "path": str(V46_REPORT),
                "sha256": _PINNED_INPUTS[str(V46_REPORT)],
            }
            or stage.get("base_checkpoint")
            != ("data_gemma4/checkpoints/gemma4_v45_retention_repair_l14_query/update_004")
            or stage.get("reconstructed_candidate_id") != "g5_both_sign_alpha_1p0"
            or stage.get("reconstructed_candidate_full_tensor_state_sha256")
            != _STATE_HASHES["update_000"]["full"]
            or stage.get("reconstructed_candidate_authorized_surface_state_sha256")
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
            or surface.get("parameter_names") != list(_PARAMETER_NAMES)
            or surface.get("parameter_shapes") != [list(value) for value in _PARAMETER_SHAPES]
            or surface.get("trainable_parameter_count") != sum(_PARAMETER_COUNTS)
            or surface.get("everything_else_frozen") is not True
        ):
            raise ValueError(f"V47 {label} authorization/train-only provenance changed")
    stage4 = _mapping(
        metadata["update_004"].get("v47_book_continuation"),
        "V47 update-four stage",
    )
    source = _mapping(stage4.get("source_audit"), "V47 source audit")
    schedule = _mapping(stage4.get("schedule_audit"), "V47 schedule audit")
    prefix_hashes = _mapping(
        stage4.get("candidate_prefix_sha256_by_train_scene"),
        "V47 candidate prefix hashes",
    )
    if (
        source.get("source_optimizer_file_opened") is not False
        or source.get("source_optimizer_state_loaded") is not False
        or source.get("optimizer_file_opened") is not False
        or source.get("optimizer_state_deserialized") is not False
        or source.get("optimizer_state_loaded") is not False
        or schedule.get("schema_version") != 1
        or schedule.get("fixed_nonadaptive") is not True
        or schedule.get("one_true_optimizer_step_per_row") is not True
        or schedule.get("schedule_sha256") != _SCHEDULE_SHA256
        or len(_sequence(schedule.get("rows"), "V47 schedule rows")) != 4
        or sorted(prefix_hashes) != sorted(_TRAIN_SCENES)
    ):
        raise ValueError("V47 source, schedule, or train-scene boundary changed")
    return {
        "v46_terminal_sha256": _PINNED_INPUTS[str(V46_TERMINAL)],
        "same_exact_v46_authorization_at_updates_zero_two_four": True,
        "reconstructed_candidate_full_tensor_state_sha256": _STATE_HASHES["update_000"]["full"],
        "reconstructed_candidate_authorized_surface_state_sha256": _STATE_HASHES["update_000"][
            "authorized"
        ],
        "exact_train_scene_ids": list(_TRAIN_SCENES),
        "exact_train_scene_count": 16,
        "schedule_sha256": _SCHEDULE_SHA256,
        "source_optimizer_files_or_states_loaded": False,
        "validation_oracle_and_final_access": False,
        "question_dependent_scene_processing_or_retrieval": False,
    }


def _implementation_reference(
    expected_v48_diagnostic_sha256: str,
    expected_v48_test_sha256: str,
) -> dict[str, Any]:
    values = (expected_v48_diagnostic_sha256, expected_v48_test_sha256)
    placeholders = [value == IMPLEMENTATION_SHA256_PLACEHOLDER for value in values]
    if any(placeholders) and not all(placeholders):
        raise ValueError("V48 implementation hashes must both be placeholders or real")
    if all(placeholders):
        return {
            "status": "pending_stable_v48_implementation_bytes",
            "diagnostic_path": str(V48_DIAGNOSTIC),
            "diagnostic_sha256": IMPLEMENTATION_SHA256_PLACEHOLDER,
            "test_path": str(V48_TEST),
            "test_sha256": IMPLEMENTATION_SHA256_PLACEHOLDER,
            "report_path": str(V48_OUTPUT),
            "implementation_files_opened": False,
            "implementation_authenticated": False,
        }
    diagnostic_digest = _lower_hex64(
        expected_v48_diagnostic_sha256,
        "expected V48 diagnostic SHA256",
    )
    test_digest = _lower_hex64(expected_v48_test_sha256, "expected V48 test SHA256")
    _locked_file(_resolve(V48_DIAGNOSTIC), diagnostic_digest, "V48 diagnostic")
    _locked_file(_resolve(V48_TEST), test_digest, "V48 test")
    return {
        "status": "stable_v48_implementation_bytes_authenticated",
        "diagnostic_path": str(V48_DIAGNOSTIC),
        "diagnostic_sha256": diagnostic_digest,
        "test_path": str(V48_TEST),
        "test_sha256": test_digest,
        "report_path": str(V48_OUTPUT),
        "implementation_files_opened": True,
        "implementation_authenticated": True,
    }


def _v48_authorization(
    expected_v48_diagnostic_sha256: str,
    expected_v48_test_sha256: str,
) -> dict[str, Any]:
    diagnostic_digest = _lower_hex64(
        expected_v48_diagnostic_sha256,
        "expected V48 diagnostic SHA256",
    )
    test_digest = _lower_hex64(expected_v48_test_sha256, "expected V48 test SHA256")
    return {
        "schema_version": 1,
        "authorization_id": _V48_AUTHORIZATION_ID,
        "authorized": True,
        "only_exact_action": ("one_bounded_read_only_v48_train_checkpoint_dual_margin_diagnostic"),
        "authorized_script": str(V48_DIAGNOSTIC),
        "authorized_test": str(V48_TEST),
        "authorized_report": str(V48_OUTPUT),
        "authorized_config": str(DEFAULT_CONFIG),
        "explicit_terminal_sha256_cli_required": True,
        "implementation_integrity": {
            "script_sha256": diagnostic_digest,
            "test_sha256": test_digest,
            "config_sha256": _PINNED_INPUTS[str(DEFAULT_CONFIG)],
        },
        "invocation_contract": {
            "terminal_path": str(DEFAULT_OUTPUT),
            "required_cli_argument": "--expected-v47-terminal-sha256",
            "expected_value": "sha256_of_materialized_v47_terminal_passed_explicitly",
            "v48_must_not_embed_terminal_sha256": True,
            "v48_must_authenticate_terminal_bytes_and_exact_authorization": True,
        },
        "source": {
            "checkpoint": str(DEFAULT_CHECKPOINT_ROOT / "update_004"),
            "full_tensor_state_sha256": _STATE_HASHES["update_004"]["full"],
            "authorized_surface_state_sha256": _STATE_HASHES["update_004"]["authorized"],
            "frozen_state_sha256": _STATE_HASHES["update_004"]["frozen"],
            "file_sha256": {
                name.removeprefix("update_004/"): digest
                for name, digest in _CHECKPOINT_FILES.items()
                if name.startswith("update_004/")
            },
            "optimizer_file_open_authorized": False,
        },
        "measurements": {
            "isolated_side_gradient_specs": [
                {
                    "gradient_id": "g_book",
                    "pair_id": "pair_000015",
                    "question_key": "cfq_163eb92339ad35a5",
                    "side_index": 0,
                    "loss": "negative_selected_side_margin",
                },
                {
                    "gradient_id": "g_mirror",
                    "pair_id": "pair_000016",
                    "question_key": "cfq_699675ceeaf65406",
                    "side_index": 1,
                    "loss": "negative_selected_side_margin",
                },
                {
                    "gradient_id": "g5_guard",
                    "pair_id": "pair_000006",
                    "question_key": "cfq_5c84a2c27d2be251",
                    "side_index": 0,
                    "loss": "negative_selected_side_margin",
                },
            ],
            "normalize_each_nonzero_component": (
                "unit_l2_within_each_scene_or_query_group_before_combination"
            ),
            "report_raw_norms_and_pairwise_cosines_by_group": True,
            "exact_three_torch_autograd_grad_probes_authorized": True,
        },
        "candidate_grid": {
            "direction_ids": [
                "dual_query_sign",
                "dual_both_sign",
                "guarded_both_sign",
            ],
            "alpha_grid": [0.125, 0.25, 0.5, 1.0, 2.0],
            "candidate_formula": ("float32_P0-alpha*lr_group*sign(normalized_component_sum)"),
            "scene_readout_learning_rate": 1.0e-5,
            "query_learning_rate": 8.0e-6,
            "exact_candidate_count": 15,
            "full_25_unit_teacher_metrics_per_candidate": True,
            "full_fixed_48_row_broad_nll_per_candidate": True,
            "candidate_relative_prefix_trust_per_candidate": True,
            "exact_source_restoration_before_and_after_every_probe": True,
            "prehash_all_candidates_before_candidate_forward": True,
        },
        "fixed_evidence": {
            "teacher_complete_units": 8,
            "teacher_positive_sides": 33,
            "teacher_cross_prefix_complete_units": 17,
            "teacher_complete_physical_pair_coverage": 4,
            "teacher_book_complete_units": 0,
            "teacher_mirror_complete_units": 1,
            "teacher_book_cross_prefix_complete_units": 1,
            "greedy_complete_units": 4,
            "greedy_broad_exact_correct": 23,
            "greedy_broad_row_count": 48,
            "priority_deficit_improvement": _UPDATE4_PRIORITY_IMPROVEMENT,
            "broad_nll": _UPDATE4_BROAD_NLL,
            "source_prefix_trust_rms": _UPDATE4_TRUST_RMS,
        },
        "fixed_data_boundary": {
            "scene_ids": list(_TRAIN_SCENES),
            "scene_count": 16,
            "training_scenes_only": True,
            "blocking_file_access_audit_required": True,
            "complete_pre_question_scene_prefixes": True,
            "question_dependent_scene_processing": False,
            "question_dependent_retrieval": False,
        },
        "forbidden_actions": {
            "optimizer_state_file_open": True,
            "optimizer_state_deserialization": True,
            "optimizer_state_loading": True,
            "optimizer_construction": True,
            "optimizer_step": True,
            "backward_or_parameter_gradient_accumulation": True,
            "gradient_outside_exact_three_specs": True,
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
            "report_only_output": True,
            "candidate_checkpoint_write_authorized": False,
            "optimizer_construction_or_step_authorized": False,
            "candidate_selection_authorized": False,
            "greedy_generation_authorized": False,
            "validation_access_authorized": False,
            "oracle_access_authorized": False,
            "final_test_access_authorized": False,
            "selector_execution_authorized": False,
            "runtime_promotion_authorized": False,
        },
    }


def build_terminal_scaffold(
    expected_v48_diagnostic_sha256: str = IMPLEMENTATION_SHA256_PLACEHOLDER,
    expected_v48_test_sha256: str = IMPLEMENTATION_SHA256_PLACEHOLDER,
    *,
    config_path: str | Path = DEFAULT_CONFIG,
    checkpoint_root: str | Path = DEFAULT_CHECKPOINT_ROOT,
) -> dict[str, Any]:
    """Audit immutable V47 and optionally authenticate stable V48 bytes."""

    config_file = _resolve(config_path)
    root = _resolve(checkpoint_root)
    if config_file != _resolve(DEFAULT_CONFIG):
        raise ValueError("V47 terminal config path is pinned")
    if root != _resolve(DEFAULT_CHECKPOINT_ROOT):
        raise ValueError("V47 terminal checkpoint root is pinned")
    inputs = _authenticate_inputs()
    predecessor = _read_json(
        _resolve(V46_TERMINAL),
        _PINNED_INPUTS[str(V46_TERMINAL)],
        "V46 terminal seal",
    )
    predecessor_audit = _authenticate_predecessor(predecessor)
    inventory = _authenticate_inventory(root)
    metadata = {
        f"update_{step:03d}": _read_json(
            root / f"update_{step:03d}/metadata.json",
            _CHECKPOINT_FILES[f"update_{step:03d}/metadata.json"],
            f"V47 update-{step} metadata",
        )
        for step in (0, 2, 4)
    }
    for step, label in ((0, "update_000"), (2, "update_002"), (4, "update_004")):
        value = metadata[label]
        stage = _mapping(value.get("v47_book_continuation"), f"V47 {label} stage")
        if (
            value.get("schema_version") != 1
            or value.get("config_hash") != _CONFIG_HASH
            or value.get("epoch") != step
            or value.get("optimizer_step") != step
            or stage.get("optimizer_step") != step
        ):
            raise ValueError(f"V47 {label} checkpoint identity changed")
    runtime = {
        label: _authenticate_runtime(root, metadata[label], step=step)
        for step, label in ((0, "update_000"), (2, "update_002"), (4, "update_004"))
    }
    tensors = _authenticate_tensors(root)
    history, gate = _authenticate_history_and_gate(metadata)
    provenance = _authenticate_provenance(metadata, predecessor)
    implementation = _implementation_reference(
        expected_v48_diagnostic_sha256,
        expected_v48_test_sha256,
    )
    ready = implementation["implementation_authenticated"] is True
    return {
        "schema_version": 1,
        "artifact": "v47_book_continuation_terminal_gate_scaffold",
        "passed": True,
        "terminal_materialization_authorized": ready,
        "terminal_conclusion": "update4_train_only_gate_failed_stop_is_final",
        "v47_final_train_only_gate_passed": False,
        "input_integrity": inputs,
        "v46_predecessor_authentication": predecessor_audit,
        "checkpoint_inventory": inventory,
        "tensor_transition": tensors,
        "runtime_metadata_audit": runtime,
        "history_audit": history,
        "update4_gate_replay": gate,
        "v47_provenance": provenance,
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
        "v48_implementation_reference": implementation,
        "only_exact_successor_authorized": _V48_AUTHORIZATION_ID if ready else None,
        "arbitrary_training_authorized": False,
        "resume_v47_training_authorized": False,
        "candidate_checkpoint_write_authorized": False,
        "validation_access_authorized": False,
        "oracle_access_authorized": False,
        "final_test_access_authorized": False,
        "selector_execution_authorized": False,
        "runtime_promotion_authorized": False,
        "chat_promotion_authorized": False,
        "embodied_promotion_authorized": False,
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
            "protected_v11_report_access": "bytes_hashed_only",
        },
    }


def build_terminal_report(
    expected_v48_diagnostic_sha256: str,
    expected_v48_test_sha256: str,
    *,
    config_path: str | Path = DEFAULT_CONFIG,
    checkpoint_root: str | Path = DEFAULT_CHECKPOINT_ROOT,
) -> dict[str, Any]:
    """Build the terminal only after both exact V48 files are stable."""

    scaffold = build_terminal_scaffold(
        expected_v48_diagnostic_sha256,
        expected_v48_test_sha256,
        config_path=config_path,
        checkpoint_root=checkpoint_root,
    )
    if scaffold.get("terminal_materialization_authorized") is not True:
        raise ValueError("V47 terminal requires explicit stable V48 implementation hashes")
    return {
        **scaffold,
        "artifact": "v47_book_continuation_terminal_gate",
        "conditional_successor_authorization": _v48_authorization(
            expected_v48_diagnostic_sha256,
            expected_v48_test_sha256,
        ),
        "only_exact_successor_authorized": _V48_AUTHORIZATION_ID,
        "v48_v47_u4_dual_margin_no_step_diagnostic_authorized": True,
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
    expected_v48_diagnostic_sha256: str,
    expected_v48_test_sha256: str,
    config_path: str | Path = DEFAULT_CONFIG,
    checkpoint_root: str | Path = DEFAULT_CHECKPOINT_ROOT,
) -> dict[str, Any]:
    """Materialize the exact terminal once, atomically, at its pinned path."""

    path = _resolve(output)
    if path != _resolve(DEFAULT_OUTPUT):
        raise ValueError("V47 terminal output path is pinned")
    if path.is_symlink() or path.exists():
        raise FileExistsError(f"V47 terminal is one-shot and will not overwrite {path}")
    report = build_terminal_report(
        expected_v48_diagnostic_sha256,
        expected_v48_test_sha256,
        config_path=config_path,
        checkpoint_root=checkpoint_root,
    )
    _atomic_json(path, report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--checkpoint-root", type=Path, default=DEFAULT_CHECKPOINT_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--expected-v48-diagnostic-sha256", required=True)
    parser.add_argument("--expected-v48-test-sha256", required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            write_report(
                args.output,
                expected_v48_diagnostic_sha256=args.expected_v48_diagnostic_sha256,
                expected_v48_test_sha256=args.expected_v48_test_sha256,
                config_path=args.config,
                checkpoint_root=args.checkpoint_root,
            ),
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "IMPLEMENTATION_SHA256_PLACEHOLDER",
    "build_terminal_report",
    "build_terminal_scaffold",
    "write_report",
]
