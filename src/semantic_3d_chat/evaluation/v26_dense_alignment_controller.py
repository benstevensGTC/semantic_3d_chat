"""Fail-closed control plane for the test-isolated V26 dense calibration.

V26 changes no model architecture and relaxes no scientific gate.  It selects
one preregistered optimizer arm after V25's bounded failure, restricts gradient
calibration to scenes 1/2/9/10, reserves mirrored scenes 7/8 for semantic
validation, and makes scenes 3/4/5/6 inaccessible to calibration loaders.  In
particular, the final paired-QA test scenes 5/6 must have zero map and oracle
access before stage one can be authorized.

This controller is report-only.  It never loads questions, scene maps, Gemma
weights, or oracle metadata.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from safetensors import safe_open

from semantic_3d_chat.config import PROJECT_ROOT, config_hash, load_config
from semantic_3d_chat.evaluation import v25_dense_alignment_controller as v25
from semantic_3d_chat.language.lora import lora_banks_settings, tensor_state_sha256
from semantic_3d_chat.scene_encoder.dense_alignment import (
    DenseAlignmentResidual,
    construct_dense_alignment,
    dense_alignment_settings,
    validate_dense_alignment_state,
)
from semantic_3d_chat.training.checkpointing import (
    RUNTIME_METADATA_FILENAME,
    TRAINING_METADATA_FILENAME,
    runtime_checkpoint_metadata,
    validate_runtime_checkpoint_metadata,
)
from semantic_3d_chat.training.dense_alignment_calibration import (
    dense_alignment_calibration_profile,
)
from semantic_3d_chat.training.dense_alignment_supervision import (
    dense_alignment_supervision_settings,
    dense_alignment_warmup_settings,
)
from semantic_3d_chat.training.source_provenance import (
    capture_git_source_provenance,
    require_clean_committed_source,
)

CONFIG_PATH = Path("configs/experiments/gemma4_color_mirror_dense_alignment_v26.yaml")
CALIBRATION_SCREEN_PATH = Path(
    "reports/gemma4/metrics/v26_dense_alignment_calibration_screen.json"
)
CALIBRATION_REPORT_PATH = Path(
    "reports/gemma4/metrics/v26_dense_alignment_calibration.json"
)
CALIBRATION_BRIDGE_PATH = Path(
    "reports/gemma4/artifacts/v26_dense_alignment_bridge.safetensors"
)
QA_SPLIT_MANIFEST_PATH = Path("data/qa/splits.json")
QA_TRAIN_PATH = Path("data/qa/train.jsonl")
PRIMARY_NAMESPACE = "gemma4_v26_dense_alignment"
EXTENSION_NAMESPACE = "gemma4_v26_dense_alignment_extension_u8"

CALIBRATION_SCENES = (
    "scene_000001",
    "scene_000002",
    "scene_000009",
    "scene_000010",
)
HELD_OUT_SCENES = ("scene_000007", "scene_000008")
FORBIDDEN_SCENES = (
    "scene_000003",
    "scene_000004",
    "scene_000005",
    "scene_000006",
)
FINAL_QA_TEST_SCENES = ("scene_000005", "scene_000006")
PAIRED_QA_TRAIN_SCENES = (
    "scene_000003",
    "scene_000004",
    "scene_000007",
    "scene_000008",
)

EXPECTED_CONFIG_SHA256 = (
    "a1aec6fbef0c044ea3f5a17db13476ef1f913d3edc0b0f0db35582faabdfd509"
)
EXPECTED_CONTRACT_SHA256 = (
    "150201b7fe95553237ed53d544b271394876c941d4139ebec7a85d605cbb67d8"
)
EXPECTED_SCREEN_REPORT_SHA256 = (
    "dfac0479f7bc4744de0ae49b2dd9cbcb0695279cb85abd03245dcbb3cb800d54"
)
EXPECTED_CALIBRATION_REPORT_SHA256 = (
    "cf1fb0cb7a907596766844fe4486aa78301a80724e2cdbeedc1f29ce3e2f13d3"
)
EXPECTED_CALIBRATION_BRIDGE_SHA256 = (
    "3340c453ded5152775e34e6e40ccb7e97dda1d7201e7321ff15c408edf92a83a"
)
EXPECTED_DENSE_INITIAL_SHA256 = v25.EXPECTED_DENSE_INITIAL_SHA256
EXPECTED_CALIBRATION_FINAL_SHA256 = (
    "5c8721fd1ff789b7a8a55e82f31fd7faae54b8114616d3a29efd8605190bad6e"
)
EXPECTED_CALIBRATION_HISTORY_SHA256 = (
    "8d5f1144596938c30c94cb6d08866d4e68649685dbaa134068a042838d7b762e"
)
EXPECTED_HELD_OUT_LOCALIZATION_SHA256 = (
    "ac64416615d2149feb1e0920dc72ff586a1a3337bccbbfda145cd8384623d5c2"
)
EXPECTED_SCENE_ACCESS_AUDIT_SHA256 = (
    "40b0753014356ca38e92917caba8742f501c73e6cd64007ee3a18dfc74494453"
)
EXPECTED_QA_SPLIT_MANIFEST_SHA256 = (
    "075a871ad00ca322745159c52d919809bab8d819afb2bb77ac207eb3e5cb7f0c"
)
EXPECTED_QA_SPLIT_FINGERPRINT = (
    "715382f1a851d74947513e5ebf4e932e74de2915f99d3f4ac1c0114f986cd2e2"
)
EXPECTED_QA_TRAIN_SHA256 = (
    "ffa721d57849ade8fdd0811e3e1e62fe807200f710aec780dc4d3dcecd4fb0e0"
)
EXPECTED_PAIR_UNIT_SELECTION_SHA256 = (
    "d5928cb783339ef62fff5c14a8c7f85f90d3a7a6cb8edad0a784998082740d3e"
)
EXPECTED_PAIR_MEMBERSHIP_SHA256 = (
    "99ee448c23fb71b7269a353a54b2156ac55701847af170597dcc351af15cbcbe"
)
EXPECTED_QA_TRAIN_MAP_SHA256 = {
    "scene_000003": "5adbd63a3c1f816b694ef78f92fbb3bf1246661da18d6a1b3bdef35ad3e0df35",
    "scene_000004": "eee7c265ac2e678973748c86ce9eae4233f536c1a9331b35ec1e63ad99edf0d7",
    "scene_000007": "477af89865db2bf34c84b7ca62b5a600ffde6d8ed1ba746f151546c4eb553911",
    "scene_000008": "0e140531cf9342dbf78a5349ff5799b22d2ec465d29cacc8cf270e546b9b3de4",
}
EXPECTED_DENSE_PARAMETER_COUNT = v25.EXPECTED_DENSE_PARAMETER_COUNT
EXPECTED_FROZEN_HASHES = v25.EXPECTED_FROZEN_HASHES
FROZEN_BANKS = v25.FROZEN_BANKS
EXPECTED_SOURCE_ADAPTER_KEY_COUNT = 137
EXPECTED_SOURCE_ADAPTER_KEYS_SHA256 = (
    "65ff3c73cff298abc25c04fe3d80f00f9949a5eea37c9b8d7024ec2fca16d152"
)
EXPECTED_ADAPTER_KEY_COUNT = 141
EXPECTED_ADAPTER_KEYS_SHA256 = (
    "20f171572675f45e0ebd7814977a59e7c907bcb4d30ee9b8a6fa340f50195652"
)

_SHA256 = re.compile(r"[0-9a-f]{64}")
_BRIDGE_KEYS = {
    "dense_aligner.alignment_a",
    "dense_aligner.alignment_b",
    "dense_aligner.architecture_marker",
    "dense_aligner.scaling",
}
_PROHIBITED_TENSOR_PARTS = (
    "oracle",
    "category",
    "caption",
    "prototype",
    "text_embedding",
    "instance",
    "segmentation",
)


class V26ControlViolation(ValueError):
    """A V26 authorization input drifted from the preregistered contract."""


def _fail(message: str) -> None:
    raise V26ControlViolation(message)


def _mapping(value: Any, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        _fail(f"{field} must be a mapping")
    return value


def _sequence(value: Any, field: str) -> Sequence[Any]:
    if isinstance(value, (str, bytes, bytearray)) or not isinstance(value, Sequence):
        _fail(f"{field} must be a sequence")
    return value


def _equal(observed: Any, expected: Any, field: str) -> None:
    scalar = (bool, int, float, str)
    if (
        isinstance(expected, scalar)
        and type(observed) is not type(expected)
        or observed != expected
    ):
        _fail(f"{field} mismatch: expected={expected!r} observed={observed!r}")


def _finite_float(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        _fail(f"{field} must be a finite number")
    parsed = float(value)
    if not math.isfinite(parsed):
        _fail(f"{field} must be a finite number")
    return parsed


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _key_inventory_sha256(keys: Sequence[str] | set[str]) -> str:
    return _canonical_sha256(sorted(keys))


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _resolve(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def _regular_file(path: Path, field: str) -> Path:
    resolved = _resolve(path)
    if resolved.is_symlink() or not resolved.is_file():
        _fail(f"{field} is not a regular non-symlink file: {resolved}")
    return resolved


def _load_json(path: Path, field: str) -> dict[str, Any]:
    resolved = _regular_file(path, field)
    try:
        loaded = json.loads(resolved.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        _fail(f"cannot load {field}: {error}")
    return dict(_mapping(loaded, field))


def _write_json(value: Mapping[str, Any], path: Path) -> Path:
    destination = _resolve(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(dict(value), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return destination


def _require_numeric_hash_only(value: Any, *, field: str = "calibration") -> None:
    if value is None or isinstance(value, (bool, int)):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            _fail(f"{field} contains NaN or infinity")
        return
    if isinstance(value, str):
        if _SHA256.fullmatch(value) is None:
            _fail(f"{field} contains a non-hash string")
        return
    if isinstance(value, Mapping):
        for key, nested in value.items():
            if not isinstance(key, str):
                _fail(f"{field} contains a non-string key")
            _require_numeric_hash_only(nested, field=f"{field}.{key}")
        return
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for index, nested in enumerate(value):
            _require_numeric_hash_only(nested, field=f"{field}[{index}]")
        return
    _fail(f"{field} contains forbidden type {type(value).__name__}")


def _clean_provenance(value: Mapping[str, Any], field: str) -> dict[str, Any]:
    provenance = dict(value)
    try:
        require_clean_committed_source(provenance)
    except RuntimeError as error:
        _fail(f"{field} is not a clean committed source: {error}")
    return provenance


def v26_contract(config: Mapping[str, Any]) -> dict[str, Any]:
    """Normalize the selected optimizer, split, gates, and frozen surfaces."""

    training = _mapping(config.get("training"), "training")
    experiment = _mapping(config.get("experiment"), "experiment")
    screen = _mapping(config.get("v26_screen"), "v26_screen")
    profile = dense_alignment_calibration_profile(config)
    dense = dense_alignment_settings(config)
    supervision = dense_alignment_supervision_settings(config)
    warmup = dense_alignment_warmup_settings(config)
    lora = lora_banks_settings(config).contract()
    banks = _sequence(lora.get("banks"), "language.lora_banks")
    frozen_bank_hashes: dict[str, Any] = {}
    for value in banks:
        bank = _mapping(value, "language.lora_banks entry")
        name = str(bank.get("name"))
        if name in FROZEN_BANKS:
            _equal(bank.get("trainable"), False, f"LoRA bank {name} trainable")
            frozen_bank_hashes[name] = bank.get("expected_initial_state_sha256")
    _equal(set(frozen_bank_hashes), set(FROZEN_BANKS), "frozen LoRA bank set")
    optimizer = _mapping(training.get("optimizer"), "training.optimizer")
    return {
        "schema_version": 1,
        "role": "final_qa_test_isolated_dense_alignment_falsifier",
        "config_sha256": config_hash(dict(config), length=64),
        "profile_version": profile.version,
        "source": {
            "archive_sha256": screen.get("source_archive_summary_sha256"),
            "checkpoint": training.get("initialize_from"),
            "checkpoint_epoch": screen.get("source_checkpoint_epoch"),
            "adapter_sha256": training.get("initialize_expected_adapter_sha256"),
            "metadata_sha256": training.get("initialize_expected_metadata_sha256"),
            "frozen_hashes": {
                "scene": experiment.get("source_scene_state_sha256"),
                "global": experiment.get("source_global_scene_residual_state_sha256"),
                "signed_x": experiment.get(
                    "source_signed_x_scene_residual_state_sha256"
                ),
                **frozen_bank_hashes,
            },
        },
        "dense_alignment": dense.contract(),
        "dense_alignment_trainable_parameter_count": experiment.get(
            "dense_alignment_trainable_parameter_count"
        ),
        "supervision": supervision.contract(),
        "warmup": warmup.contract(),
        "selected_calibration": {
            "screen_report": screen.get("preregistered_screen_report"),
            "screen_report_sha256": screen.get(
                "preregistered_screen_report_sha256"
            ),
            "arm_index": screen.get("selected_arm_index"),
            "initial_state_sha256": screen.get("selected_initial_state_sha256"),
            "final_state_sha256": screen.get("selected_final_state_sha256"),
            "history_sha256": screen.get("selected_history_sha256"),
            "held_out_localization_sha256": screen.get(
                "selected_held_out_localization_sha256"
            ),
            "optimizer": screen.get("selected_optimizer"),
        },
        "split_isolation": {
            "qa_split_manifest": screen.get("qa_split_manifest"),
            "qa_split_manifest_sha256": screen.get("qa_split_manifest_sha256"),
            "calibration_scene_ids": list(profile.calibration_scene_ids),
            "held_out_scene_ids": list(profile.held_out_scene_ids),
            "forbidden_scene_ids": screen.get("forbidden_calibration_scene_ids"),
            "final_qa_test_scene_ids": screen.get("final_qa_test_scene_ids"),
        },
        "paired_qa": {
            "output_namespace": training.get("output_namespace"),
            "pair_steps_per_epoch": training.get("pair_steps_per_epoch"),
            "gradient_accumulation": training.get("gradient_accumulation"),
            "learning_rate": training.get("dense_alignment_learning_rate"),
            "weight_decay": training.get("dense_alignment_weight_decay"),
            "optimizer": dict(optimizer),
        },
        "training_surface": {
            "initialize_named_lora_freeze_for_dense_alignment_transition": training.get(
                "initialize_named_lora_freeze_for_dense_alignment_transition"
            ),
            "train_dense_alignment_only": training.get("train_dense_alignment_only"),
            "freeze_scene_adapter": training.get("freeze_scene_adapter"),
            "train_signed_x_scene_residual_only": training.get(
                "train_signed_x_scene_residual_only"
            ),
            "train_global_scene_residual_only": training.get(
                "train_global_scene_residual_only"
            ),
            "train_lora_with_frozen_scene_residual_stack": training.get(
                "train_lora_with_frozen_scene_residual_stack"
            ),
            "frozen_lora_bank_hashes": frozen_bank_hashes,
        },
        "runtime_isolation": {
            "question_dependent_scene_processing": experiment.get(
                "question_dependent_scene_processing"
            ),
            "question_dependent_retrieval": experiment.get(
                "question_dependent_retrieval"
            ),
            "runtime_oracle_access": experiment.get("runtime_oracle_access"),
            "runtime_category_strings": experiment.get("runtime_category_strings"),
            "runtime_text_prototypes": experiment.get("runtime_text_prototypes"),
            "all_voxels_transformed": experiment.get("all_voxels_transformed"),
        },
        "unchanged_gates": {
            "held_out_localization_requires": screen.get(
                "held_out_localization_requires"
            ),
            "stage_2_requires": screen.get("stage_2_requires"),
            "eligibility_requires": screen.get("eligibility_requires"),
            "continuation_requires": screen.get("continuation_requires"),
            "full_teacher_gate_requires": screen.get("full_teacher_gate_requires"),
            "greedy_audit_only_after_full_teacher_gate": screen.get(
                "greedy_audit_only_after_full_teacher_gate"
            ),
        },
    }


def _validate_contract(config: Mapping[str, Any]) -> tuple[dict[str, Any], str]:
    _equal(
        config_hash(dict(config), length=64),
        EXPECTED_CONFIG_SHA256,
        "resolved V26 config SHA-256",
    )
    contract = v26_contract(config)
    contract_sha = _canonical_sha256(contract)
    _equal(contract_sha, EXPECTED_CONTRACT_SHA256, "normalized V26 contract SHA-256")
    return contract, contract_sha


def _screen_audit(config: Mapping[str, Any]) -> dict[str, Any]:
    screen = _mapping(config.get("v26_screen"), "v26_screen")
    _equal(
        screen.get("preregistered_screen_report"),
        str(CALIBRATION_SCREEN_PATH),
        "V26 screen report path",
    )
    path = _regular_file(CALIBRATION_SCREEN_PATH, "V26 preregistered screen")
    _equal(
        _file_sha256(path),
        EXPECTED_SCREEN_REPORT_SHA256,
        "V26 preregistered screen SHA-256",
    )
    report = _load_json(CALIBRATION_SCREEN_PATH, "V26 preregistered screen")
    _require_numeric_hash_only(report, field="V26 preregistered screen")
    for field, expected in {
        "schema_version": 1,
        "screen_preregistered": True,
        "selection_rule_first_joint_pass_by_fixed_arm_priority": True,
        "initial_state_sha256": EXPECTED_DENSE_INITIAL_SHA256,
        "calibration_scene_count": 4,
        "validation_scene_count": 2,
        "forbidden_scene_count": 4,
        "arm_count": 4,
        "eligible_arm_count": 3,
        "selected_arm_index": 1,
        "recommend_v26_contract": True,
        "qa_run_count": 0,
        "raw_map_write_count": 0,
    }.items():
        _equal(report.get(field), expected, f"V26 preregistered screen {field}")
    arms = _sequence(report.get("arms"), "V26 preregistered arms")
    _equal(len(arms), 4, "V26 preregistered arm count")
    control = _mapping(arms[0], "V26 control arm")
    selected = _mapping(arms[1], "V26 selected arm")
    _equal(control.get("joint_passed"), False, "V26 control arm joint pass")
    for field, expected in {
        "arm_index": 1,
        "learning_rate": 0.005,
        "weight_decay": 0.0001,
        "delta_mse_regularization_weight": 0.01,
        "maximum_optimizer_steps": 40,
        "optimizer_steps_executed": 13,
        "first_training_pass_step": 13,
        "training_gate_passed": True,
        "held_out_evaluated": True,
        "joint_passed": True,
        "initial_state_sha256": EXPECTED_DENSE_INITIAL_SHA256,
        "terminal_state_sha256": EXPECTED_CALIBRATION_FINAL_SHA256,
    }.items():
        _equal(selected.get(field), expected, f"V26 selected arm {field}")
    history = [
        {key: nested for key, nested in _mapping(row, "selected history row").items()
         if key != "state_sha256"}
        for row in _sequence(selected.get("history"), "selected history")
    ]
    _equal(
        _canonical_sha256(history),
        EXPECTED_CALIBRATION_HISTORY_SHA256,
        "V26 selected history SHA-256",
    )
    held_out = _mapping(
        selected.get("held_out_localization"), "V26 selected held-out localization"
    )
    _equal(
        _canonical_sha256(held_out),
        EXPECTED_HELD_OUT_LOCALIZATION_SHA256,
        "V26 selected held-out SHA-256",
    )
    access = _mapping(report.get("access_audit"), "V26 screen access audit")
    for field, expected in {
        "forbidden_map_access_count": 0,
        "forbidden_oracle_access_count": 0,
        "forbidden_zero_access": True,
        "scene_slot_03_map_count": 0,
        "scene_slot_03_oracle_count": 0,
        "scene_slot_04_map_count": 0,
        "scene_slot_04_oracle_count": 0,
        "scene_slot_05_map_count": 0,
        "scene_slot_05_oracle_count": 0,
        "scene_slot_06_map_count": 0,
        "scene_slot_06_oracle_count": 0,
    }.items():
        _equal(access.get(field), expected, f"V26 screen access {field}")
    return {
        "report_path": str(CALIBRATION_SCREEN_PATH),
        "report_sha256": EXPECTED_SCREEN_REPORT_SHA256,
        "selected_arm_index": 1,
        "selected_optimizer_steps": 13,
        "selected_final_state_sha256": EXPECTED_CALIBRATION_FINAL_SHA256,
        "forbidden_zero_access": True,
        "qa_run_count": 0,
    }


def _qa_split_audit(config: Mapping[str, Any]) -> dict[str, Any]:
    screen = _mapping(config.get("v26_screen"), "v26_screen")
    _equal(
        screen.get("qa_split_manifest"),
        str(QA_SPLIT_MANIFEST_PATH),
        "V26 QA split manifest path",
    )
    path = _regular_file(QA_SPLIT_MANIFEST_PATH, "V26 QA split manifest")
    _equal(
        _file_sha256(path),
        EXPECTED_QA_SPLIT_MANIFEST_SHA256,
        "V26 QA split manifest SHA-256",
    )
    manifest = _load_json(QA_SPLIT_MANIFEST_PATH, "V26 QA split manifest")
    train_path = _regular_file(QA_TRAIN_PATH, "V26 QA training corpus")
    _equal(
        _file_sha256(train_path),
        EXPECTED_QA_TRAIN_SHA256,
        "V26 QA training corpus SHA-256",
    )
    observed_map_hashes = {
        scene_id: _file_sha256(
            _regular_file(
                Path("data_gemma4/maps") / scene_id / "voxel_map.npz",
                f"V26 QA training map {scene_id}",
            )
        )
        for scene_id in PAIRED_QA_TRAIN_SCENES
    }
    _equal(
        observed_map_hashes,
        EXPECTED_QA_TRAIN_MAP_SHA256,
        "V26 QA training map SHA-256",
    )
    _equal(manifest.get("schema_version"), 2, "QA split schema")
    _equal(manifest.get("fingerprint"), EXPECTED_QA_SPLIT_FINGERPRINT, "QA split fingerprint")
    splits = _mapping(manifest.get("splits"), "QA splits")
    test = tuple(str(value) for value in _sequence(splits.get("test"), "QA test split"))
    _equal(test, FINAL_QA_TEST_SCENES, "final QA test scenes")
    calibration = set(CALIBRATION_SCENES)
    held_out = set(HELD_OUT_SCENES)
    final_test = set(test)
    forbidden = tuple(
        str(value)
        for value in _sequence(
            screen.get("forbidden_calibration_scene_ids"),
            "forbidden calibration scenes",
        )
    )
    _equal(forbidden, FORBIDDEN_SCENES, "forbidden calibration scenes")
    _equal(
        tuple(
            str(value)
            for value in _sequence(
                screen.get("final_qa_test_scene_ids"), "configured final QA test scenes"
            )
        ),
        FINAL_QA_TEST_SCENES,
        "configured final QA test scenes",
    )
    if calibration & final_test or held_out & final_test:
        _fail("Final QA test scenes overlap calibration or semantic validation")
    if not final_test.issubset(forbidden):
        _fail("Final QA test scenes are not in the fail-closed forbidden set")
    return {
        "manifest_path": str(QA_SPLIT_MANIFEST_PATH),
        "manifest_sha256": EXPECTED_QA_SPLIT_MANIFEST_SHA256,
        "training_corpus_path": str(QA_TRAIN_PATH),
        "training_corpus_sha256": EXPECTED_QA_TRAIN_SHA256,
        "training_map_sha256": dict(EXPECTED_QA_TRAIN_MAP_SHA256),
        "fingerprint": EXPECTED_QA_SPLIT_FINGERPRINT,
        "final_qa_test_split_sha256": _canonical_sha256(list(test)),
        "final_qa_test_scene_count": len(test),
        "calibration_scene_count": len(calibration),
        "held_out_scene_count": len(held_out),
        "forbidden_scene_count": len(forbidden),
        "final_test_disjoint_from_calibration": True,
        "final_test_disjoint_from_held_out": True,
        "final_test_in_forbidden_set": True,
        "question_records_loaded": False,
        "question_file_bytes_hashed_only": True,
    }


def _training_surface_audit(config: Mapping[str, Any]) -> dict[str, Any]:
    contract = v26_contract(config)
    surface = _mapping(contract.get("training_surface"), "V26 training surface")
    expected = {
        "initialize_named_lora_freeze_for_dense_alignment_transition": True,
        "train_dense_alignment_only": True,
        "freeze_scene_adapter": True,
        "train_signed_x_scene_residual_only": False,
        "train_global_scene_residual_only": False,
        "train_lora_with_frozen_scene_residual_stack": False,
        "frozen_lora_bank_hashes": {
            name: EXPECTED_FROZEN_HASHES[name] for name in FROZEN_BANKS
        },
    }
    _equal(dict(surface), expected, "V26 isolated training surface")
    paired = _mapping(contract.get("paired_qa"), "V26 paired QA")
    for field, value in {
        "output_namespace": PRIMARY_NAMESPACE,
        "pair_steps_per_epoch": 12,
        "gradient_accumulation": 12,
        "learning_rate": 0.0003,
        "weight_decay": 0.0,
    }.items():
        _equal(paired.get(field), value, f"V26 paired QA {field}")
    return {
        "only_trainable_module": "dense_alignment",
        "trainable_parameter_count": EXPECTED_DENSE_PARAMETER_COUNT,
        "all_scene_decoder_and_lora_surfaces_frozen": True,
        "paired_qa_optimizer_constructed_after_warmup": True,
        "paired_qa_optimizer_state_starts_empty": True,
        "pair_steps_per_optimizer_update": 12,
    }


def _runtime_isolation_audit(config: Mapping[str, Any]) -> dict[str, Any]:
    experiment = _mapping(config.get("experiment"), "experiment")
    expected = {
        "question_dependent_scene_processing": False,
        "question_dependent_retrieval": False,
        "runtime_oracle_access": False,
        "runtime_category_strings": False,
        "runtime_text_prototypes": False,
        "all_voxels_transformed": True,
    }
    _equal({key: experiment.get(key) for key in expected}, expected, "V26 runtime isolation")
    profile = dense_alignment_calibration_profile(config)
    _equal(profile.version, 26, "V26 calibration profile version")
    _equal(profile.calibration_scene_ids, CALIBRATION_SCENES, "V26 calibration split")
    _equal(profile.held_out_scene_ids, HELD_OUT_SCENES, "V26 held-out split")
    _equal(profile.forbidden_scene_ids, FORBIDDEN_SCENES, "V26 forbidden split")
    warmup = dense_alignment_warmup_settings(config)
    return {
        **expected,
        "calibration_scene_count": 4,
        "held_out_scene_count": 2,
        "forbidden_scene_count": 4,
        "final_qa_test_scene_count": 2,
        "held_out_scene_gradient_access": warmup.held_out_scene_gradient_access,
        "scene_access_audit_required": profile.include_access_audit,
        "runtime_checkpoint_tensor_payload_only": True,
        "environmental_text_serialized": False,
    }


def run_preflight(
    config_path: Path = CONFIG_PATH,
    output: Path | None = None,
    *,
    source_provenance: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Authorize only the exact V26 calibration, never paired QA directly."""

    config = load_config(config_path)
    contract, contract_sha = _validate_contract(config)
    provenance = (
        capture_git_source_provenance(PROJECT_ROOT)
        if source_provenance is None
        else dict(source_provenance)
    )
    clean = _clean_provenance(provenance, "V26 controller provenance")
    try:
        source = v25._source_audit(config)
        dense = v25._dense_alignment_audit(config)
    except ValueError as error:
        _fail(f"V26 inherited source audit failed: {error}")
    report = {
        "schema_version": 1,
        "audit_type": "v26_dense_alignment_structural_preflight",
        "authorized": True,
        "calibration_stage_authorized": True,
        "paired_qa_stage_authorized": False,
        "paired_qa_authorization_condition": "exact_selected_v26_calibration_report",
        "runtime_eligible": False,
        "report_only": True,
        "model_loaded": False,
        "optimizer_constructed": False,
        "optimizer_step_executed": False,
        "optimizer_steps": 0,
        "oracle_loaded": False,
        "question_files_loaded": False,
        "config_path": str(CONFIG_PATH),
        "config_sha256": EXPECTED_CONFIG_SHA256,
        "contract": contract,
        "contract_sha256": contract_sha,
        "source_provenance": clean,
        "source": source,
        "dense_alignment": dense,
        "screen_selection": _screen_audit(config),
        "qa_split_isolation": _qa_split_audit(config),
        "training_surface": _training_surface_audit(config),
        "runtime_isolation": _runtime_isolation_audit(config),
        "warmup": dense_alignment_warmup_settings(config).contract(),
        "supervision": dense_alignment_supervision_settings(config).contract(),
    }
    if output is not None:
        destination = _write_json(report, output)
        report["output"] = str(destination)
    return report


def validate_preflight(path: Path, config_path: Path = CONFIG_PATH) -> dict[str, Any]:
    report = _load_json(path, "V26 preflight")
    config = load_config(config_path)
    contract, contract_sha = _validate_contract(config)
    for field, expected in {
        "schema_version": 1,
        "audit_type": "v26_dense_alignment_structural_preflight",
        "authorized": True,
        "calibration_stage_authorized": True,
        "paired_qa_stage_authorized": False,
        "paired_qa_authorization_condition": "exact_selected_v26_calibration_report",
        "runtime_eligible": False,
        "report_only": True,
        "model_loaded": False,
        "optimizer_constructed": False,
        "optimizer_step_executed": False,
        "optimizer_steps": 0,
        "oracle_loaded": False,
        "question_files_loaded": False,
        "config_path": str(CONFIG_PATH),
        "config_sha256": EXPECTED_CONFIG_SHA256,
        "contract": contract,
        "contract_sha256": contract_sha,
    }.items():
        _equal(report.get(field), expected, f"V26 preflight {field}")
    _clean_provenance(
        _mapping(report.get("source_provenance"), "V26 preflight provenance"),
        "V26 preflight provenance",
    )
    split = _mapping(report.get("qa_split_isolation"), "V26 preflight split isolation")
    _equal(split.get("final_test_in_forbidden_set"), True, "V26 final test forbidden")
    runtime = _mapping(report.get("runtime_isolation"), "V26 preflight runtime")
    _equal(runtime.get("scene_access_audit_required"), True, "V26 access audit required")
    return report


def _bridge_audit(path: Path) -> dict[str, Any]:
    source = _regular_file(path, "V26 calibration bridge")
    bridge_sha = _file_sha256(source)
    _equal(bridge_sha, EXPECTED_CALIBRATION_BRIDGE_SHA256, "V26 bridge SHA-256")
    with safe_open(source, framework="pt", device="cpu") as handle:
        keys = set(handle.keys())
    _equal(keys, _BRIDGE_KEYS, "V26 bridge tensor keys")
    prohibited = sorted(
        key
        for key in keys
        if any(part in key.casefold() for part in _PROHIBITED_TENSOR_PARTS)
    )
    _equal(prohibited, [], "V26 bridge prohibited tensor keys")
    return {
        "bridge_sha256": bridge_sha,
        "tensor_key_count": len(keys),
        "prohibited_tensor_key_count": 0,
        "tensor_payload_only": True,
        "category_strings_serialized": False,
        "text_prototypes_serialized": False,
        "oracle_payload_serialized": False,
    }


def verify_calibration_report(
    *,
    config_path: Path,
    preflight_path: Path,
    calibration_path: Path,
    bridge_path: Path,
    output: Path | None = None,
) -> dict[str, Any]:
    """Authorize stage one only for the exact selected, test-isolated report."""

    config = load_config(config_path)
    _validate_contract(config)
    preflight = validate_preflight(preflight_path, config_path)
    report_path = _regular_file(calibration_path, "V26 calibration report")
    report_sha = _file_sha256(report_path)
    _equal(
        report_sha,
        EXPECTED_CALIBRATION_REPORT_SHA256,
        "V26 deterministic calibration report SHA-256",
    )
    audit = _load_json(calibration_path, "V26 calibration report")
    _require_numeric_hash_only(audit, field="V26 calibration report")
    supervision = dense_alignment_supervision_settings(config)
    warmup = dense_alignment_warmup_settings(config)
    for field, expected in {
        "schema_version": 1,
        "config_sha256": EXPECTED_CONFIG_SHA256,
        "calibration_split_sha256": _canonical_sha256(list(CALIBRATION_SCENES)),
        "held_out_split_sha256": _canonical_sha256(list(HELD_OUT_SCENES)),
        "initial_state_sha256": EXPECTED_DENSE_INITIAL_SHA256,
        "final_state_sha256": EXPECTED_CALIBRATION_FINAL_SHA256,
        "calibration_scene_count": 4,
        "held_out_scene_count": 2,
        "category_count": 9,
        "region_count": 35,
        "skipped_underfilled_region_count": 0,
        "summarized_region_voxel_count": 2060,
        "selective_token_row_count": 12,
        "loaded_parameter_count": 1,
        "cpu_only": True,
        "local_files_only": True,
        "raw_map_write_count": 0,
        "raw_maps_preserved": True,
        "question_dependent_selection": False,
        "qa_update_authorized": True,
        "bridge_written": True,
        "bridge_sha256": EXPECTED_CALIBRATION_BRIDGE_SHA256,
    }.items():
        _equal(audit.get(field), expected, f"V26 calibration {field}")
    _equal(
        supervision.calibration_scene_ids,
        CALIBRATION_SCENES,
        "V26 report calibration split",
    )
    _equal(supervision.held_out_scene_ids, HELD_OUT_SCENES, "V26 report held-out split")
    training = _mapping(audit.get("training"), "V26 calibration training")
    for field, expected in {
        "learning_rate": warmup.learning_rate,
        "weight_decay": warmup.weight_decay,
        "delta_mse_regularization_weight": warmup.delta_rms_regularization_weight,
        "maximum_optimizer_steps": warmup.max_optimizer_steps,
        "optimizer_steps": 13,
        "stopped_at_first_pass": True,
        "calibration_passed": True,
        "final_state_sha256": EXPECTED_CALIBRATION_FINAL_SHA256,
    }.items():
        _equal(training.get(field), expected, f"V26 calibration training.{field}")
    history = _sequence(training.get("history"), "V26 calibration history")
    _equal(len(history), 13, "V26 calibration history length")
    _equal(
        _canonical_sha256(history),
        EXPECTED_CALIBRATION_HISTORY_SHA256,
        "V26 calibration history SHA-256",
    )
    for expected_step, row_raw in enumerate(history, start=1):
        row = _mapping(row_raw, f"V26 calibration history {expected_step}")
        _equal(row.get("optimizer_step"), expected_step, "V26 calibration optimizer step")
        _equal(
            row.get("passed"),
            expected_step == 13,
            "V26 calibration first-pass boundary",
        )
    final = _mapping(history[-1], "V26 calibration final history row")
    if _finite_float(final.get("top1_accuracy"), "V26 final accuracy") < 1.0:
        _fail("V26 final accuracy relaxed below 1.0")
    if _finite_float(final.get("minimum_cosine_margin"), "V26 final margin") < 0.10:
        _fail("V26 final margin relaxed below 0.10")
    if _finite_float(final.get("delta_rms"), "V26 final delta RMS") > 1.0:
        _fail("V26 final delta RMS exceeds 1.0")
    if _finite_float(final.get("delta_abs_max"), "V26 final delta absolute max") > 3.5:
        _fail("V26 final delta absolute max exceeds 3.5")
    held_out = _mapping(audit.get("held_out_localization"), "V26 held-out localization")
    _equal(
        _canonical_sha256(held_out),
        EXPECTED_HELD_OUT_LOCALIZATION_SHA256,
        "V26 held-out localization SHA-256",
    )
    _equal(held_out.get("passed"), True, "V26 held-out localization pass")
    access = _mapping(audit.get("scene_access_audit"), "V26 scene access audit")
    _equal(
        _canonical_sha256(access),
        EXPECTED_SCENE_ACCESS_AUDIT_SHA256,
        "V26 scene access audit SHA-256",
    )
    for field, expected in {
        "map_access_count": 6,
        "oracle_access_count": 10,
        "calibration_map_access_count": 4,
        "calibration_oracle_access_count": 8,
        "held_out_map_access_count": 2,
        "held_out_oracle_access_count": 2,
        "forbidden_map_access_count": 0,
        "forbidden_oracle_access_count": 0,
        "qa_final_test_map_access_count": 0,
        "qa_final_test_oracle_access_count": 0,
        "forbidden_zero_access": True,
        "qa_final_test_zero_access": True,
    }.items():
        _equal(access.get(field), expected, f"V26 scene access {field}")
    bridge = _bridge_audit(bridge_path)
    decision = {
        "schema_version": 1,
        "audit_type": "v26_dense_alignment_calibration_authorization",
        "decision": "exact_selected_calibration_passed_stage1_authorized",
        "calibration_authorized": True,
        "paired_qa_stage_authorized": True,
        "terminal_stop": False,
        "thresholds_preserved": True,
        "threshold_relaxation_permitted": False,
        "optimizer_steps": 13,
        "maximum_optimizer_steps": 40,
        "held_out_localization_passed": True,
        "forbidden_scene_access_count": 0,
        "final_qa_test_scene_access_count": 0,
        "final_qa_test_untouched": True,
        "qa_run_count": 0,
        "greedy_audit_authorized": False,
        "static_chat_authorized": False,
        "embodied_phase_authorized": False,
        "model_loaded": False,
        "oracle_loaded_by_controller": False,
        "question_files_loaded_by_controller": False,
        "report_only": True,
        "config_sha256": EXPECTED_CONFIG_SHA256,
        "contract_sha256": EXPECTED_CONTRACT_SHA256,
        "preflight_sha256": _file_sha256(_regular_file(preflight_path, "V26 preflight")),
        "source_report": str(calibration_path),
        "source_report_sha256": report_sha,
        "final_state_sha256": EXPECTED_CALIBRATION_FINAL_SHA256,
        "history_sha256": EXPECTED_CALIBRATION_HISTORY_SHA256,
        "held_out_localization_sha256": EXPECTED_HELD_OUT_LOCALIZATION_SHA256,
        "scene_access_audit_sha256": EXPECTED_SCENE_ACCESS_AUDIT_SHA256,
        "bridge": bridge,
        "source_preflight_contract_sha256": preflight.get("contract_sha256"),
    }
    if output is not None:
        destination = _write_json(decision, output)
        decision["output"] = str(destination)
    return decision


def _warmup_gate(audit: Mapping[str, Any], config: Mapping[str, Any]) -> dict[str, Any]:
    """Revalidate the immutable V26 calibration embedded in training metadata."""

    _require_numeric_hash_only(audit, field="V26 checkpoint calibration")
    warmup = dense_alignment_warmup_settings(config)
    training = _mapping(audit.get("training"), "V26 checkpoint calibration training")
    history = _sequence(training.get("history"), "V26 checkpoint calibration history")
    _equal(len(history), 13, "V26 checkpoint calibration history length")
    _equal(
        _canonical_sha256(history),
        EXPECTED_CALIBRATION_HISTORY_SHA256,
        "V26 checkpoint calibration history SHA-256",
    )
    final = _mapping(history[-1], "V26 checkpoint calibration final row")
    for expected_step, row_raw in enumerate(history, start=1):
        row = _mapping(row_raw, f"V26 checkpoint calibration row {expected_step}")
        _equal(row.get("optimizer_step"), expected_step, "V26 warm-up optimizer step")
        _equal(row.get("passed"), expected_step == 13, "V26 warm-up first pass")
    checks = {
        "schema_version": audit.get("schema_version") == 1,
        "config_sha256": audit.get("config_sha256") == EXPECTED_CONFIG_SHA256,
        "calibration_split": audit.get("calibration_split_sha256")
        == _canonical_sha256(list(CALIBRATION_SCENES)),
        "held_out_split": audit.get("held_out_split_sha256")
        == _canonical_sha256(list(HELD_OUT_SCENES)),
        "calibration_scene_count": audit.get("calibration_scene_count") == 4,
        "held_out_scene_count": audit.get("held_out_scene_count") == 2,
        "category_count": audit.get("category_count") == 9,
        "region_count": audit.get("region_count") == 35,
        "skipped_regions": audit.get("skipped_underfilled_region_count") == 0,
        "loaded_parameter_count": audit.get("loaded_parameter_count") == 1,
        "cpu_only": audit.get("cpu_only") is True,
        "local_files_only": audit.get("local_files_only") is True,
        "raw_map_write_count": audit.get("raw_map_write_count") == 0,
        "raw_maps_preserved": audit.get("raw_maps_preserved") is True,
        "question_independent": audit.get("question_dependent_selection") is False,
        "learning_rate": training.get("learning_rate") == warmup.learning_rate,
        "weight_decay": training.get("weight_decay") == warmup.weight_decay,
        "regularizer": training.get("delta_mse_regularization_weight")
        == warmup.delta_rms_regularization_weight,
        "maximum_steps": training.get("maximum_optimizer_steps")
        == warmup.max_optimizer_steps,
        "optimizer_steps": training.get("optimizer_steps") == 13,
        "stopped_at_first_pass": training.get("stopped_at_first_pass") is True,
        "calibration_passed": training.get("calibration_passed") is True,
        "final_passed": final.get("passed") is True,
        "top1_accuracy": _finite_float(final.get("top1_accuracy"), "V26 warm-up accuracy")
        >= 1.0,
        "minimum_margin": _finite_float(
            final.get("minimum_cosine_margin"), "V26 warm-up margin"
        )
        >= 0.10,
        "delta_rms": _finite_float(final.get("delta_rms"), "V26 warm-up delta RMS")
        <= 1.0,
        "delta_abs_max": _finite_float(
            final.get("delta_abs_max"), "V26 warm-up delta absolute max"
        )
        <= 3.5,
        "qa_update_authorized": audit.get("qa_update_authorized") is True,
        "pair_optimizer_empty_before": audit.get(
            "pair_optimizer_state_empty_before_warmup"
        )
        is True,
        "pair_optimizer_rebuilt": audit.get("pair_optimizer_rebuilt_after_warmup")
        is True,
        "pair_optimizer_empty_after": audit.get(
            "pair_optimizer_state_empty_after_warmup"
        )
        is True,
        "pair_optimizer_steps_before_qa": audit.get("pair_optimizer_steps_before_qa")
        == 0,
        "held_out_no_gradient": audit.get("held_out_scene_gradient_access") is False,
        "no_text_prototypes": audit.get("category_text_prototypes_serialized") is False,
        "no_oracle_retained": audit.get("oracle_payload_retained") is False,
        "nested_final_state": training.get("final_state_sha256")
        == EXPECTED_CALIBRATION_FINAL_SHA256,
        "top_level_final_state": audit.get("final_state_sha256")
        == EXPECTED_CALIBRATION_FINAL_SHA256,
        "initial_state": audit.get("initial_state_sha256")
        == EXPECTED_DENSE_INITIAL_SHA256,
    }
    held_out = _mapping(
        audit.get("held_out_localization"), "V26 checkpoint held-out localization"
    )
    checks.update(
        {
            "held_out_hash": _canonical_sha256(held_out)
            == EXPECTED_HELD_OUT_LOCALIZATION_SHA256,
            "held_out_passed": held_out.get("passed") is True,
            "held_out_all_hit": held_out.get("all_target_hit_at_k") is True,
            "held_out_precision": _finite_float(
                held_out.get("minimum_precision_at_k"), "V26 held-out precision"
            )
            >= 0.10,
            "held_out_region_margin": _finite_float(
                held_out.get("minimum_region_margin"), "V26 held-out region margin"
            )
            > 0.0,
            "held_out_query_margin": _finite_float(
                held_out.get("minimum_correct_vs_distractor_margin"),
                "V26 held-out query margin",
            )
            > 0.0,
            "held_out_mirror_error": _finite_float(
                held_out.get("maximum_mirror_centroid_error_m"),
                "V26 held-out mirror error",
            )
            <= 0.15,
        }
    )
    access = _mapping(audit.get("scene_access_audit"), "V26 checkpoint access audit")
    checks.update(
        {
            "access_hash": _canonical_sha256(access)
            == EXPECTED_SCENE_ACCESS_AUDIT_SHA256,
            "forbidden_map_zero": access.get("forbidden_map_access_count") == 0,
            "forbidden_oracle_zero": access.get("forbidden_oracle_access_count") == 0,
            "qa_final_map_zero": access.get("qa_final_test_map_access_count") == 0,
            "qa_final_oracle_zero": access.get("qa_final_test_oracle_access_count") == 0,
            "forbidden_zero_access": access.get("forbidden_zero_access") is True,
            "qa_final_zero_access": access.get("qa_final_test_zero_access") is True,
        }
    )
    if not all(checks.values()):
        failed = sorted(key for key, passed in checks.items() if not passed)
        _fail(f"V26 embedded calibration gate failed: {failed}")
    return {
        "passed": True,
        "optimizer_steps": 13,
        "initial_state_sha256": EXPECTED_DENSE_INITIAL_SHA256,
        "final_state_sha256": EXPECTED_CALIBRATION_FINAL_SHA256,
        "history_sha256": EXPECTED_CALIBRATION_HISTORY_SHA256,
        "held_out_localization_sha256": EXPECTED_HELD_OUT_LOCALIZATION_SHA256,
        "scene_access_audit_sha256": EXPECTED_SCENE_ACCESS_AUDIT_SHA256,
        "final_qa_test_untouched": True,
        "checks": checks,
    }


def _teacher_forced_gate(metadata: Mapping[str, Any]) -> dict[str, Any]:
    try:
        return v25._teacher_forced_gate(metadata)
    except ValueError as error:
        _fail(f"V26 teacher-forced gate failed: {error}")


def _training_metadata_contract(
    metadata: Mapping[str, Any], config: Mapping[str, Any]
) -> dict[str, Any]:
    """Validate the trainer, source, split, and optimizer contract fail closed."""

    _equal(metadata.get("schema_version"), 3, "V26 trainer metadata schema")
    _equal(metadata.get("config_hash"), config_hash(dict(config)), "V26 trainer config hash")
    provenance = _clean_provenance(
        _mapping(metadata.get("source_provenance"), "V26 training source provenance"),
        "V26 training source provenance",
    )
    _equal(
        tuple(_sequence(metadata.get("scene_ids"), "V26 scene IDs")),
        PAIRED_QA_TRAIN_SCENES,
        "V26 loaded scene IDs",
    )
    _equal(
        tuple(_sequence(metadata.get("train_scene_ids"), "V26 train scene IDs")),
        PAIRED_QA_TRAIN_SCENES,
        "V26 train scene IDs",
    )
    _equal(
        tuple(_sequence(metadata.get("validation_scene_ids"), "V26 validation scene IDs")),
        (),
        "V26 validation scene IDs",
    )
    _equal(
        tuple(_sequence(metadata.get("test_scene_ids"), "V26 test scene IDs")),
        FINAL_QA_TEST_SCENES,
        "V26 final QA test scene IDs",
    )
    loaded_training_scenes = {str(value) for value in metadata["train_scene_ids"]}
    _equal(
        sorted(loaded_training_scenes & set(FINAL_QA_TEST_SCENES)),
        [],
        "V26 loaded training/final-test overlap",
    )
    _equal(metadata.get("gradient_accumulation"), 12, "V26 gradient accumulation")
    for field, expected in {
        "counterfactual_pair_unit_count": 12,
        "counterfactual_pair_unit_selection_sha256": (
            EXPECTED_PAIR_UNIT_SELECTION_SHA256
        ),
        "training_counterfactual_pair_count": 2,
        "training_counterfactual_pair_membership_sha256": (
            EXPECTED_PAIR_MEMBERSHIP_SHA256
        ),
    }.items():
        _equal(metadata.get(field), expected, f"V26 QA selection {field}")
    _equal(
        dict(_mapping(metadata.get("dense_alignment_optimizer"), "V26 dense optimizer")),
        {"name": "AdamW", "learning_rate": 0.0003, "weight_decay": 0.0},
        "V26 dense optimizer",
    )
    pair = _mapping(metadata.get("pair_curriculum"), "V26 pair curriculum")
    for field, expected in {
        "enabled": True,
        "pair_only": True,
        "pair_only_scene_ids": list(PAIRED_QA_TRAIN_SCENES),
        "steps_per_epoch": 12,
        "gate_enabled": True,
    }.items():
        _equal(pair.get(field), expected, f"V26 pair curriculum {field}")

    source_metadata = _load_json(
        v25.SOURCE_CHECKPOINT / TRAINING_METADATA_FILENAME,
        "V24 immutable source metadata",
    )
    initialization = _mapping(
        metadata.get("initialization_provenance"), "V26 initialization provenance"
    )
    expected_initialization = {
        "schema_version": 7,
        "mode": "frozen_named_lora_scene_stack_plus_zero_output_dense_alignment",
        "checkpoint": str(v25.SOURCE_CHECKPOINT),
        "adapter_sha256": v25.EXPECTED_SOURCE_ARTIFACTS["adapter_sha256"],
        "metadata_sha256": v25.EXPECTED_SOURCE_ARTIFACTS["metadata_sha256"],
        "expected_adapter_sha256": v25.EXPECTED_SOURCE_ARTIFACTS["adapter_sha256"],
        "expected_metadata_sha256": v25.EXPECTED_SOURCE_ARTIFACTS["metadata_sha256"],
        "checkpoint_epoch": 1,
        "checkpoint_output_namespace": "gemma4_v24_shared_query",
        "checkpoint_config_hash": source_metadata.get("config_hash"),
        "checkpoint_source_provenance": source_metadata.get("source_provenance"),
        "initialize_named_lora_freeze_for_dense_alignment_transition": True,
        "optimizer_state_loaded": False,
        "history_loaded": False,
        "source_lora_bank_state_sha256": {
            name: EXPECTED_FROZEN_HASHES[name] for name in FROZEN_BANKS
        },
        "source_scene_state_sha256": EXPECTED_FROZEN_HASHES["scene"],
        "expected_source_scene_state_sha256": EXPECTED_FROZEN_HASHES["scene"],
        "source_global_scene_residual_state_sha256": EXPECTED_FROZEN_HASHES["global"],
        "expected_source_global_scene_residual_state_sha256": EXPECTED_FROZEN_HASHES[
            "global"
        ],
        "source_signed_x_scene_residual_state_sha256": EXPECTED_FROZEN_HASHES[
            "signed_x"
        ],
        "expected_source_signed_x_scene_residual_state_sha256": EXPECTED_FROZEN_HASHES[
            "signed_x"
        ],
        "all_source_modules_frozen": True,
        "dense_alignment_initial_state_sha256": EXPECTED_DENSE_INITIAL_SHA256,
        "expected_dense_alignment_initial_state_sha256": EXPECTED_DENSE_INITIAL_SHA256,
        "dense_alignment_zero_output": True,
        "source_checkpoint_loaded_dense_alignment": False,
        "dense_alignment_calibration_authorized": True,
        "dense_alignment_calibration_final_state_sha256": (
            EXPECTED_CALIBRATION_FINAL_SHA256
        ),
        "pair_optimizer_rebuilt_after_dense_alignment_calibration": True,
    }
    for field, expected in expected_initialization.items():
        _equal(initialization.get(field), expected, f"V26 initialization {field}")

    _equal(
        metadata.get("dense_alignment"),
        dense_alignment_settings(config).contract(),
        "V26 dense architecture contract",
    )
    _equal(
        metadata.get("dense_alignment_initial_state_sha256"),
        EXPECTED_DENSE_INITIAL_SHA256,
        "V26 dense initial state",
    )
    _equal(
        metadata.get("frozen_global_scene_residual_state_sha256"),
        EXPECTED_FROZEN_HASHES["global"],
        "V26 frozen global attestation",
    )
    _equal(
        metadata.get("frozen_signed_x_scene_residual_state_sha256"),
        EXPECTED_FROZEN_HASHES["signed_x"],
        "V26 frozen signed-X attestation",
    )
    _equal(
        metadata.get("question_dependent_scene_processing"),
        False,
        "V26 question-independent scene processing",
    )
    _equal(metadata.get("all_voxels_transformed"), True, "V26 all-voxel processing")
    return {
        "schema_version": 3,
        "config_hash": config_hash(dict(config)),
        "source_provenance_sha256": _canonical_sha256(provenance),
        "train_scene_ids": list(PAIRED_QA_TRAIN_SCENES),
        "test_scene_ids": list(FINAL_QA_TEST_SCENES),
        "training_corpus_sha256": EXPECTED_QA_TRAIN_SHA256,
        "training_map_sha256": dict(EXPECTED_QA_TRAIN_MAP_SHA256),
        "counterfactual_pair_unit_selection_sha256": (
            EXPECTED_PAIR_UNIT_SELECTION_SHA256
        ),
        "training_counterfactual_pair_membership_sha256": (
            EXPECTED_PAIR_MEMBERSHIP_SHA256
        ),
        "final_qa_test_training_access_count": 0,
        "optimizer": {
            "name": "AdamW",
            "learning_rate": 0.0003,
            "weight_decay": 0.0,
            "gradient_accumulation": 12,
            "steps_per_epoch": 12,
        },
        "source_checkpoint": str(v25.SOURCE_CHECKPOINT),
        "source_artifacts": dict(v25.EXPECTED_SOURCE_ARTIFACTS),
    }


def verify_update1_metadata(
    metadata: Mapping[str, Any], config: Mapping[str, Any]
) -> dict[str, Any]:
    """Verify calibration provenance, frozen surfaces, and update-one gate."""

    metadata_contract = _training_metadata_contract(metadata, config)
    _equal(metadata.get("epoch"), 1, "V26 update-1 epoch")
    _equal(metadata.get("optimizer_step"), 1, "V26 update-1 optimizer step")
    _equal(metadata.get("output_namespace"), PRIMARY_NAMESPACE, "V26 output namespace")
    _equal(metadata.get("train_dense_alignment_only"), True, "V26 dense-only mode")
    _equal(metadata.get("freeze_scene_adapter"), True, "V26 frozen scene adapter")
    _equal(
        metadata.get("dense_alignment_parameter_count"),
        EXPECTED_DENSE_PARAMETER_COUNT,
        "V26 dense parameter count",
    )
    calibration = _mapping(
        metadata.get("dense_alignment_calibration"), "V26 embedded calibration"
    )
    calibration_gate = _warmup_gate(calibration, config)
    initialization = _mapping(
        metadata.get("initialization_provenance"), "V26 initialization provenance"
    )
    _equal(
        initialization.get("dense_alignment_calibration_authorized"),
        True,
        "V26 calibration initialization authorization",
    )
    _equal(
        initialization.get("dense_alignment_calibration_final_state_sha256"),
        EXPECTED_CALIBRATION_FINAL_SHA256,
        "V26 immutable calibration-final state",
    )
    _equal(
        initialization.get("pair_optimizer_rebuilt_after_dense_alignment_calibration"),
        True,
        "V26 paired optimizer calibration boundary",
    )
    current_dense_hash = metadata.get("dense_alignment_state_sha256")
    if not isinstance(current_dense_hash, str) or _SHA256.fullmatch(current_dense_hash) is None:
        _fail("V26 update-1 current dense state hash is invalid")
    if current_dense_hash in {
        EXPECTED_DENSE_INITIAL_SHA256,
        EXPECTED_CALIBRATION_FINAL_SHA256,
    }:
        _fail("V26 update-1 did not produce a distinct post-QA dense state")
    _equal(
        metadata.get("frozen_scene_state_sha256"),
        EXPECTED_FROZEN_HASHES["scene"],
        "V26 frozen scene state",
    )
    for field, key in (
        ("global_scene_residual_state_sha256", "global"),
        ("signed_x_scene_residual_state_sha256", "signed_x"),
    ):
        _equal(metadata.get(field), EXPECTED_FROZEN_HASHES[key], f"V26 frozen {key} state")
    bank_hashes = _mapping(metadata.get("lora_bank_state_sha256"), "V26 LoRA hashes")
    _equal(
        dict(bank_hashes),
        {name: EXPECTED_FROZEN_HASHES[name] for name in FROZEN_BANKS},
        "V26 frozen LoRA hashes",
    )
    teacher = _teacher_forced_gate(metadata)
    if not bool(teacher["stage_2_passed"]):
        _fail("V26 update 1 did not pass the preregistered stage-2 gate")
    return {
        "match": True,
        "training_metadata_contract": metadata_contract,
        "calibration_gate": calibration_gate,
        "teacher_forced_gate": teacher,
        "paired_qa_optimizer_step": 1,
        "all_source_surfaces_frozen": True,
        "stage_2_authorized": True,
        "greedy_audit_authorized": False,
        "final_qa_test_untouched": True,
    }


_RUNTIME_FORBIDDEN_SUBSTRINGS = (
    "dense_alignment_calibration",
    "pair_candidate_gate",
    "history",
    "question_id",
    "question_ids",
    "train_scene_ids",
    "validation_scene_ids",
    "test_scene_ids",
    "optimizer",
    "canonical_training_targets",
    "oracle",
    "category",
    "caption",
    "prototype",
    "segmentation",
    "instance_id",
    "target_instance",
)


def _runtime_sidecar_audit(
    metadata: Mapping[str, Any], runtime: Mapping[str, Any]
) -> dict[str, Any]:
    try:
        validate_runtime_checkpoint_metadata(runtime)
        expected = runtime_checkpoint_metadata(metadata)
    except (TypeError, ValueError, RuntimeError) as error:
        _fail(f"V26 runtime sidecar validation failed: {error}")
    _equal(dict(runtime), expected, "V26 runtime sidecar sanitized projection")
    serialized = json.dumps(runtime, sort_keys=True, separators=(",", ":")).casefold()
    present = sorted(
        value for value in _RUNTIME_FORBIDDEN_SUBSTRINGS if value in serialized
    )
    _equal(present, [], "V26 runtime sidecar training-only payload")
    _equal(
        runtime.get("dense_alignment_state_sha256"),
        metadata.get("dense_alignment_state_sha256"),
        "V26 runtime/full dense state",
    )
    _equal(
        runtime.get("question_dependent_scene_processing"),
        False,
        "V26 runtime question-independent processing",
    )
    _equal(runtime.get("all_voxels_transformed"), True, "V26 runtime all-voxel path")
    return {
        "field_count": len(runtime),
        "sanitized_projection_exact": True,
        "training_only_field_count": 0,
        "calibration_payload_present": False,
        "history_present": False,
        "question_ids_present": False,
        "scene_split_present": False,
        "optimizer_payload_present": False,
        "environmental_text_payload_present": False,
        "question_dependent_scene_processing": False,
        "all_voxels_transformed": True,
    }


def _adapter_audit(
    config: Mapping[str, Any], adapter_path: Path, metadata: Mapping[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    source = _regular_file(adapter_path, "V26 checkpoint adapter")
    immutable_source = _regular_file(
        v25.SOURCE_CHECKPOINT / "adapter.safetensors", "V24 immutable source adapter"
    )
    _equal(
        _file_sha256(immutable_source),
        v25.EXPECTED_SOURCE_ARTIFACTS["adapter_sha256"],
        "V24 immutable source adapter SHA-256",
    )
    with safe_open(immutable_source, framework="pt", device="cpu") as handle:
        source_keys = set(handle.keys())
    _equal(
        len(source_keys),
        EXPECTED_SOURCE_ADAPTER_KEY_COUNT,
        "V24 source adapter tensor count",
    )
    _equal(
        _key_inventory_sha256(source_keys),
        EXPECTED_SOURCE_ADAPTER_KEYS_SHA256,
        "V24 source adapter key inventory",
    )
    with safe_open(source, framework="pt", device="cpu") as handle:
        keys = list(handle.keys())
        prohibited = sorted(
            key
            for key in keys
            if any(part in key.casefold() for part in _PROHIBITED_TENSOR_PARTS)
        )
        _equal(prohibited, [], "V26 adapter prohibited tensor keys")
        state = {key: handle.get_tensor(key) for key in keys}
        dense_state = {
            key.removeprefix("dense_aligner."): value
            for key, value in state.items()
            if key.startswith("dense_aligner.")
        }
    expected_keys = source_keys | _BRIDGE_KEYS
    _equal(set(keys), expected_keys, "V26 complete adapter tensor inventory")
    _equal(len(keys), EXPECTED_ADAPTER_KEY_COUNT, "V26 adapter tensor count")
    _equal(
        _key_inventory_sha256(set(keys)),
        EXPECTED_ADAPTER_KEYS_SHA256,
        "V26 adapter key inventory SHA-256",
    )
    scene_state = {
        key: value
        for key, value in state.items()
        if key.startswith(("scene_model.", "composer.", "grounding."))
    }
    global_state = {
        key: value for key, value in state.items() if key.startswith("global_scene_residual.")
    }
    signed_x_state = {
        key: value
        for key, value in state.items()
        if key.startswith("signed_x_scene_residual.")
    }
    lora_states = {
        bank: {
            key.removeprefix(f"lora_banks.{bank}."): value
            for key, value in state.items()
            if key.startswith(f"lora_banks.{bank}.")
        }
        for bank in FROZEN_BANKS
    }
    classified = (
        set(scene_state)
        | set(global_state)
        | set(signed_x_state)
        | {
            key
            for bank in FROZEN_BANKS
            for key in state
            if key.startswith(f"lora_banks.{bank}.")
        }
        | _BRIDGE_KEYS
    )
    _equal(classified, set(keys), "V26 classified adapter tensor inventory")
    observed_frozen_hashes = {
        "scene": tensor_state_sha256(scene_state),
        "global": tensor_state_sha256(global_state),
        "signed_x": tensor_state_sha256(signed_x_state),
        **{bank: tensor_state_sha256(lora_states[bank]) for bank in FROZEN_BANKS},
    }
    _equal(
        observed_frozen_hashes,
        EXPECTED_FROZEN_HASHES,
        "V26 adapter frozen tensor states",
    )
    _equal(
        metadata.get("frozen_scene_state_sha256"),
        observed_frozen_hashes["scene"],
        "V26 adapter/metadata frozen scene state",
    )
    _equal(
        metadata.get("global_scene_residual_state_sha256"),
        observed_frozen_hashes["global"],
        "V26 adapter/metadata global state",
    )
    _equal(
        metadata.get("signed_x_scene_residual_state_sha256"),
        observed_frozen_hashes["signed_x"],
        "V26 adapter/metadata signed-X state",
    )
    _equal(
        dict(_mapping(metadata.get("lora_bank_state_sha256"), "V26 metadata LoRA hashes")),
        {bank: observed_frozen_hashes[bank] for bank in FROZEN_BANKS},
        "V26 adapter/metadata LoRA states",
    )
    _equal(
        set(dense_state),
        {"architecture_marker", "scaling", "alignment_a", "alignment_b"},
        "V26 checkpoint dense tensor keys",
    )
    module = construct_dense_alignment(config, semantic_dim=3072)
    if not isinstance(module, DenseAlignmentResidual):
        _fail("V26 checkpoint dense module could not be constructed")
    module.load_state_dict(dense_state, strict=True)
    dense_audit = validate_dense_alignment_state(
        module,
        expected_parameter_count=EXPECTED_DENSE_PARAMETER_COUNT,
        context="V26 checkpoint",
    )
    _equal(
        dense_audit.get("state_sha256"),
        metadata.get("dense_alignment_state_sha256"),
        "V26 adapter/full metadata dense state",
    )
    if dense_audit.get("state_sha256") == EXPECTED_DENSE_INITIAL_SHA256:
        _fail("V26 checkpoint retained the exact-zero initial bridge")
    return dense_audit, {
        "tensor_key_count": len(keys),
        "tensor_key_inventory_sha256": EXPECTED_ADAPTER_KEYS_SHA256,
        "source_tensor_key_count": len(source_keys),
        "source_tensor_key_inventory_sha256": EXPECTED_SOURCE_ADAPTER_KEYS_SHA256,
        "frozen_state_sha256": observed_frozen_hashes,
        "prohibited_tensor_key_count": 0,
        "tensor_payload_only": True,
    }


def _validate_calibration_decision(
    path: Path, *, preflight_path: Path, config_path: Path
) -> dict[str, Any]:
    decision = _load_json(path, "V26 calibration authorization")
    preflight = validate_preflight(preflight_path, config_path)
    for field, expected in {
        "schema_version": 1,
        "audit_type": "v26_dense_alignment_calibration_authorization",
        "decision": "exact_selected_calibration_passed_stage1_authorized",
        "calibration_authorized": True,
        "paired_qa_stage_authorized": True,
        "thresholds_preserved": True,
        "threshold_relaxation_permitted": False,
        "optimizer_steps": 13,
        "final_qa_test_scene_access_count": 0,
        "final_qa_test_untouched": True,
        "qa_run_count": 0,
        "greedy_audit_authorized": False,
        "config_sha256": EXPECTED_CONFIG_SHA256,
        "contract_sha256": EXPECTED_CONTRACT_SHA256,
        "final_state_sha256": EXPECTED_CALIBRATION_FINAL_SHA256,
        "history_sha256": EXPECTED_CALIBRATION_HISTORY_SHA256,
        "held_out_localization_sha256": EXPECTED_HELD_OUT_LOCALIZATION_SHA256,
        "scene_access_audit_sha256": EXPECTED_SCENE_ACCESS_AUDIT_SHA256,
        "source_preflight_contract_sha256": EXPECTED_CONTRACT_SHA256,
    }.items():
        _equal(decision.get(field), expected, f"V26 calibration decision {field}")
    _equal(
        decision.get("source_report_sha256"),
        EXPECTED_CALIBRATION_REPORT_SHA256,
        "V26 calibration decision source report",
    )
    _equal(
        decision.get("preflight_sha256"),
        _file_sha256(_regular_file(preflight_path, "V26 preflight")),
        "V26 calibration decision preflight",
    )
    _equal(
        decision.get("source_preflight_contract_sha256"),
        preflight.get("contract_sha256"),
        "V26 calibration/preflight contract",
    )
    _equal(
        dict(_mapping(decision.get("bridge"), "V26 calibration decision bridge")),
        _bridge_audit(CALIBRATION_BRIDGE_PATH),
        "V26 calibration decision bridge",
    )
    return decision


def verify_update1(
    *,
    config_path: Path,
    preflight_path: Path,
    calibration_decision_path: Path,
    checkpoint: Path,
    output: Path | None = None,
) -> dict[str, Any]:
    """Bind update one to full metadata, runtime sidecar, and adapter bytes."""

    config = load_config(config_path)
    _validate_contract(config)
    preflight = validate_preflight(preflight_path, config_path)
    calibration_decision = _validate_calibration_decision(
        calibration_decision_path,
        preflight_path=preflight_path,
        config_path=config_path,
    )
    qa_data_audit = _qa_split_audit(config)
    checkpoint_path = _resolve(checkpoint)
    metadata_path = _regular_file(
        checkpoint_path / TRAINING_METADATA_FILENAME, "V26 update-1 full metadata"
    )
    runtime_path = _regular_file(
        checkpoint_path / RUNTIME_METADATA_FILENAME, "V26 update-1 runtime metadata"
    )
    adapter_path = _regular_file(
        checkpoint_path / "adapter.safetensors", "V26 update-1 adapter"
    )
    optimizer_path = _regular_file(
        checkpoint_path / "optimizer.pt", "V26 update-1 optimizer"
    )
    metadata = _load_json(metadata_path, "V26 update-1 full metadata")
    runtime = _load_json(runtime_path, "V26 update-1 runtime metadata")
    verification = verify_update1_metadata(metadata, config)
    runtime_audit = _runtime_sidecar_audit(metadata, runtime)
    dense_audit, payload_audit = _adapter_audit(config, adapter_path, metadata)
    report = {
        "schema_version": 1,
        "audit_type": "v26_dense_alignment_update1_verification",
        "match": True,
        "stage_2_authorized": True,
        "greedy_audit_authorized": False,
        "report_only": True,
        "model_loaded": False,
        "oracle_loaded": False,
        "question_files_loaded": False,
        "question_dependent_scene_processing": False,
        "final_qa_test_untouched": True,
        "config_sha256": EXPECTED_CONFIG_SHA256,
        "contract_sha256": EXPECTED_CONTRACT_SHA256,
        "preflight_sha256": _file_sha256(_regular_file(preflight_path, "V26 preflight")),
        "preflight_contract_sha256": preflight["contract_sha256"],
        "calibration_decision_sha256": _file_sha256(
            _regular_file(calibration_decision_path, "V26 calibration authorization")
        ),
        "calibration_chain": {
            "source_report_sha256": calibration_decision["source_report_sha256"],
            "final_state_sha256": calibration_decision["final_state_sha256"],
            "bridge_sha256": _mapping(
                calibration_decision["bridge"], "V26 calibration bridge"
            )["bridge_sha256"],
            "preflight_sha256": calibration_decision["preflight_sha256"],
            "contract_sha256": calibration_decision["contract_sha256"],
        },
        "checkpoint": str(checkpoint),
        "artifact_hashes": {
            "adapter_sha256": _file_sha256(adapter_path),
            "metadata_sha256": _file_sha256(metadata_path),
            "runtime_metadata_sha256": _file_sha256(runtime_path),
            "optimizer_sha256": _file_sha256(optimizer_path),
        },
        "qa_data_artifacts": qa_data_audit,
        "verification": verification,
        "dense_alignment": dense_audit,
        "adapter_payload": payload_audit,
        "runtime_sidecar": runtime_audit,
    }
    if output is not None:
        destination = _write_json(report, output)
        report["output"] = str(destination)
    return report


def _epoch_audit(
    metadata: Mapping[str, Any], config: Mapping[str, Any], *, expected_epoch: int
) -> dict[str, Any]:
    metadata_contract = _training_metadata_contract(metadata, config)
    _equal(metadata.get("epoch"), expected_epoch, f"V26 epoch {expected_epoch}")
    _equal(
        metadata.get("optimizer_step"),
        expected_epoch,
        f"V26 epoch {expected_epoch} optimizer step",
    )
    namespace = metadata.get("output_namespace")
    _equal(namespace, PRIMARY_NAMESPACE, f"V26 epoch {expected_epoch} primary namespace")
    _equal(metadata.get("train_dense_alignment_only"), True, "V26 dense-only mode")
    _equal(metadata.get("freeze_scene_adapter"), True, "V26 frozen scene adapter")
    _equal(
        metadata.get("dense_alignment_parameter_count"),
        EXPECTED_DENSE_PARAMETER_COUNT,
        "V26 dense parameter count",
    )
    calibration_gate = _warmup_gate(
        _mapping(metadata.get("dense_alignment_calibration"), "V26 calibration audit"),
        config,
    )
    initialization = _mapping(
        metadata.get("initialization_provenance"), "V26 initialization provenance"
    )
    _equal(
        initialization.get("dense_alignment_calibration_final_state_sha256"),
        EXPECTED_CALIBRATION_FINAL_SHA256,
        "V26 immutable calibration-final state",
    )
    _equal(
        metadata.get("frozen_scene_state_sha256"),
        EXPECTED_FROZEN_HASHES["scene"],
        "V26 frozen scene state",
    )
    _equal(
        metadata.get("global_scene_residual_state_sha256"),
        EXPECTED_FROZEN_HASHES["global"],
        "V26 frozen global state",
    )
    _equal(
        metadata.get("signed_x_scene_residual_state_sha256"),
        EXPECTED_FROZEN_HASHES["signed_x"],
        "V26 frozen signed-X state",
    )
    _equal(
        dict(_mapping(metadata.get("lora_bank_state_sha256"), "V26 LoRA hashes")),
        {name: EXPECTED_FROZEN_HASHES[name] for name in FROZEN_BANKS},
        "V26 frozen LoRA hashes",
    )
    dense_hash = metadata.get("dense_alignment_state_sha256")
    if not isinstance(dense_hash, str) or _SHA256.fullmatch(dense_hash) is None:
        _fail("V26 epoch dense-alignment state hash is invalid")
    if dense_hash in {
        EXPECTED_DENSE_INITIAL_SHA256,
        EXPECTED_CALIBRATION_FINAL_SHA256,
    }:
        _fail("V26 epoch lacks a distinct post-QA dense state")
    teacher = _teacher_forced_gate(metadata)
    return {
        "epoch": expected_epoch,
        "optimizer_step": expected_epoch,
        "output_namespace": namespace,
        "dense_alignment_state_sha256": dense_hash,
        "calibration_final_state_sha256": calibration_gate["final_state_sha256"],
        "calibration_optimizer_steps": calibration_gate["optimizer_steps"],
        "final_qa_test_untouched": True,
        "training_metadata_contract": metadata_contract,
        "source_provenance_sha256": metadata_contract["source_provenance_sha256"],
        "teacher_forced_gate": teacher,
    }


def select_epoch_metadata(
    epochs: Mapping[int, Mapping[str, Any]], config: Mapping[str, Any]
) -> dict[str, Any]:
    """Apply only the primary four-update screen; extension is a separate protocol."""

    observed = sorted(epochs)
    if observed != list(range(1, 5)):
        _fail("V26 primary selection requires exactly epochs 1--4")
    audits = [
        _epoch_audit(epochs[epoch], config, expected_epoch=epoch) for epoch in observed
    ]
    _equal(
        len({str(value["calibration_final_state_sha256"]) for value in audits}),
        1,
        "V26 calibration state across epochs",
    )
    _equal(
        len({str(value["source_provenance_sha256"]) for value in audits}),
        1,
        "V26 training source provenance across epochs",
    )
    dense_hashes = [str(value["dense_alignment_state_sha256"]) for value in audits]
    _equal(
        len(set(dense_hashes)),
        len(dense_hashes),
        "V26 unique post-QA dense states",
    )
    eligible = [
        value for value in audits if bool(value["teacher_forced_gate"]["stage_2_passed"])
    ]
    full = [
        value
        for value in audits
        if bool(value["teacher_forced_gate"]["full_teacher_gate_passed"])
    ]
    if full:
        selected = min(full, key=lambda value: int(value["epoch"]))
        decision = "full_teacher_gate_passed_greedy_audit_authorized"
        greedy = True
        conditional = False
    elif eligible:
        selected = max(
            eligible,
            key=lambda value: (
                int(value["teacher_forced_gate"]["mirror"]["full_vocab_sides"]),
                int(value["teacher_forced_gate"]["mirror"]["full_vocab_units"]),
                float(
                    value["teacher_forced_gate"]["mirror"]["minimum_full_vocab_margin"]
                ),
                float(value["teacher_forced_gate"]["mirror"]["minimum_candidate_margin"]),
                -int(value["epoch"]),
            ),
        )
        greedy = False
        conditional = False
        decision = "primary_screen_complete_extension_requires_separate_authorization"
    else:
        selected = None
        decision = "no_eligible_epoch_stop"
        greedy = False
        conditional = False
    ranking = sorted(
        audits,
        key=lambda value: (
            bool(value["teacher_forced_gate"]["full_teacher_gate_passed"]),
            bool(value["teacher_forced_gate"]["stage_2_passed"]),
            int(value["teacher_forced_gate"]["mirror"]["full_vocab_sides"]),
            int(value["teacher_forced_gate"]["mirror"]["full_vocab_units"]),
            float(value["teacher_forced_gate"]["mirror"]["minimum_full_vocab_margin"]),
            -int(value["epoch"]),
        ),
        reverse=True,
    )
    return {
        "schema_version": 1,
        "audit_type": "v26_dense_alignment_bounded_epoch_selector",
        "decision": decision,
        "evaluated_optimizer_updates": len(audits),
        "hard_optimizer_update_limit": 4,
        "primary_screen_complete": True,
        "conditional_limit_reached": False,
        "conditional_extension_authorized": conditional,
        "extension_controller_required": not bool(full) and bool(eligible),
        "full_teacher_gate_passed": bool(full),
        "full_teacher_first_pass_epoch": (
            None if not full else min(int(value["epoch"]) for value in full)
        ),
        "greedy_audit_authorized": greedy,
        "static_chat_authorized": False,
        "embodied_phase_authorized": False,
        "runtime_eligible": False,
        "final_qa_test_untouched": True,
        "selected_epoch": None if selected is None else selected["epoch"],
        "ranking": [value["epoch"] for value in ranking],
        "epochs": audits,
    }


def _checkpoint_epoch_artifacts(
    config: Mapping[str, Any], checkpoint: Path, *, expected_epoch: int
) -> tuple[dict[str, Any], dict[str, str], dict[str, Any]]:
    root = _resolve(checkpoint)
    metadata_path = _regular_file(
        root / TRAINING_METADATA_FILENAME, f"V26 epoch {expected_epoch} full metadata"
    )
    runtime_path = _regular_file(
        root / RUNTIME_METADATA_FILENAME, f"V26 epoch {expected_epoch} runtime metadata"
    )
    adapter_path = _regular_file(
        root / "adapter.safetensors", f"V26 epoch {expected_epoch} adapter"
    )
    optimizer_path = _regular_file(
        root / "optimizer.pt", f"V26 epoch {expected_epoch} optimizer"
    )
    metadata = _load_json(metadata_path, f"V26 epoch {expected_epoch} full metadata")
    runtime = _load_json(runtime_path, f"V26 epoch {expected_epoch} runtime metadata")
    runtime_audit = _runtime_sidecar_audit(metadata, runtime)
    _adapter_audit(config, adapter_path, metadata)
    return metadata, {
        "adapter_sha256": _file_sha256(adapter_path),
        "metadata_sha256": _file_sha256(metadata_path),
        "runtime_metadata_sha256": _file_sha256(runtime_path),
        "optimizer_sha256": _file_sha256(optimizer_path),
    }, runtime_audit


def _parse_epoch_binding(value: str) -> tuple[int, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("epoch binding must be EPOCH=CHECKPOINT_DIR")
    epoch_text, path_text = value.split("=", 1)
    try:
        epoch = int(epoch_text)
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "epoch binding must be EPOCH=CHECKPOINT_DIR"
        ) from error
    if epoch < 1 or not path_text:
        raise argparse.ArgumentTypeError("epoch binding must be EPOCH=CHECKPOINT_DIR")
    return epoch, Path(path_text)


def select_epochs(
    *,
    config_path: Path,
    update1_report_path: Path,
    epoch_bindings: Sequence[tuple[int, Path]],
    output: Path | None = None,
) -> dict[str, Any]:
    config = load_config(config_path)
    _validate_contract(config)
    update1 = _load_json(update1_report_path, "V26 update-1 verification")
    for field, expected in {
        "schema_version": 1,
        "audit_type": "v26_dense_alignment_update1_verification",
        "match": True,
        "stage_2_authorized": True,
        "greedy_audit_authorized": False,
        "report_only": True,
        "model_loaded": False,
        "oracle_loaded": False,
        "question_files_loaded": False,
        "question_dependent_scene_processing": False,
        "final_qa_test_untouched": True,
        "config_sha256": EXPECTED_CONFIG_SHA256,
        "contract_sha256": EXPECTED_CONTRACT_SHA256,
    }.items():
        _equal(update1.get(field), expected, f"V26 update-1 report {field}")
    for field in ("preflight_sha256", "calibration_decision_sha256"):
        value = update1.get(field)
        if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
            _fail(f"V26 update-1 report {field} is invalid")
    _equal(
        update1.get("preflight_contract_sha256"),
        EXPECTED_CONTRACT_SHA256,
        "V26 update-1 preflight contract",
    )
    calibration_chain = _mapping(
        update1.get("calibration_chain"), "V26 update-1 calibration chain"
    )
    _equal(
        dict(calibration_chain),
        {
            "source_report_sha256": EXPECTED_CALIBRATION_REPORT_SHA256,
            "final_state_sha256": EXPECTED_CALIBRATION_FINAL_SHA256,
            "bridge_sha256": EXPECTED_CALIBRATION_BRIDGE_SHA256,
            "preflight_sha256": update1["preflight_sha256"],
            "contract_sha256": EXPECTED_CONTRACT_SHA256,
        },
        "V26 update-1 calibration chain",
    )
    current_qa_data = _qa_split_audit(config)
    _equal(
        update1.get("qa_data_artifacts"),
        current_qa_data,
        "V26 update-1/current QA data artifacts",
    )
    if len({epoch for epoch, _path in epoch_bindings}) != len(epoch_bindings):
        _fail("V26 epoch bindings contain duplicate epochs")
    metadata: dict[int, Mapping[str, Any]] = {}
    artifact_hashes: dict[str, dict[str, str]] = {}
    runtime_audits: dict[str, Any] = {}
    paths: dict[int, str] = {}
    for epoch, path in epoch_bindings:
        loaded, hashes, runtime_audit = _checkpoint_epoch_artifacts(
            config, path, expected_epoch=epoch
        )
        metadata[epoch] = loaded
        artifact_hashes[str(epoch)] = hashes
        runtime_audits[str(epoch)] = runtime_audit
        paths[epoch] = str(path)
    update1_artifacts = dict(
        _mapping(update1.get("artifact_hashes"), "V26 update-1 artifact hashes")
    )
    _equal(
        set(update1_artifacts),
        {
            "adapter_sha256",
            "metadata_sha256",
            "runtime_metadata_sha256",
            "optimizer_sha256",
        },
        "V26 update-1 artifact inventory",
    )
    for field, value in update1_artifacts.items():
        if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
            _fail(f"V26 update-1 artifact hash {field} is invalid")
    _equal(
        artifact_hashes.get("1"),
        update1_artifacts,
        "V26 selector epoch-1/update-1 artifact binding",
    )
    if 1 not in paths:
        _fail("V26 selector is missing epoch 1")
    _equal(
        _resolve(Path(paths[1])),
        _resolve(Path(str(update1.get("checkpoint")))),
        "V26 selector epoch-1/update-1 checkpoint binding",
    )
    update1_dense = _mapping(
        update1.get("dense_alignment"), "V26 update-1 dense audit"
    ).get("state_sha256")
    _equal(
        metadata[1].get("dense_alignment_state_sha256"),
        update1_dense,
        "V26 selector epoch-1/update-1 dense state",
    )
    report = select_epoch_metadata(metadata, config)
    report.update(
        {
            "config_sha256": EXPECTED_CONFIG_SHA256,
            "contract_sha256": EXPECTED_CONTRACT_SHA256,
            "update1_report_sha256": _file_sha256(
                _regular_file(update1_report_path, "V26 update-1 verification")
            ),
            "epoch_artifact_sha256": artifact_hashes,
            "runtime_sidecar_audits": runtime_audits,
            "selected_checkpoint": (
                None
                if report["selected_epoch"] is None
                else paths[int(report["selected_epoch"])]
            ),
            "model_loaded": False,
            "oracle_loaded": False,
            "question_files_loaded": False,
            "report_only": True,
            "question_dependent_scene_processing": False,
        }
    )
    if output is not None:
        destination = _write_json(report, output)
        report["output"] = str(destination)
    return report


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    preflight = commands.add_parser("preflight")
    preflight.add_argument("--config", type=Path, default=CONFIG_PATH)
    preflight.add_argument("--output", type=Path, required=True)
    validate = commands.add_parser("validate-preflight")
    validate.add_argument("--config", type=Path, default=CONFIG_PATH)
    validate.add_argument("--preflight", type=Path, required=True)
    calibration = commands.add_parser("verify-calibration")
    calibration.add_argument("--config", type=Path, default=CONFIG_PATH)
    calibration.add_argument("--preflight", type=Path, required=True)
    calibration.add_argument("--calibration", type=Path, required=True)
    calibration.add_argument("--bridge", type=Path, required=True)
    calibration.add_argument("--output", type=Path, required=True)
    update1 = commands.add_parser("verify-update1")
    update1.add_argument("--config", type=Path, default=CONFIG_PATH)
    update1.add_argument("--preflight", type=Path, required=True)
    update1.add_argument("--calibration-decision", type=Path, required=True)
    update1.add_argument("--checkpoint", type=Path, required=True)
    update1.add_argument("--output", type=Path, required=True)
    select = commands.add_parser("select")
    select.add_argument("--config", type=Path, default=CONFIG_PATH)
    select.add_argument("--update1-report", type=Path, required=True)
    select.add_argument(
        "--epoch",
        type=_parse_epoch_binding,
        action="append",
        required=True,
        dest="epoch_bindings",
    )
    select.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.command == "preflight":
        report = run_preflight(args.config, args.output)
    elif args.command == "validate-preflight":
        report = validate_preflight(args.preflight, args.config)
    elif args.command == "verify-calibration":
        report = verify_calibration_report(
            config_path=args.config,
            preflight_path=args.preflight,
            calibration_path=args.calibration,
            bridge_path=args.bridge,
            output=args.output,
        )
    elif args.command == "verify-update1":
        report = verify_update1(
            config_path=args.config,
            preflight_path=args.preflight,
            calibration_decision_path=args.calibration_decision,
            checkpoint=args.checkpoint,
            output=args.output,
        )
    elif args.command == "select":
        report = select_epochs(
            config_path=args.config,
            update1_report_path=args.update1_report,
            epoch_bindings=args.epoch_bindings,
            output=args.output,
        )
    else:  # pragma: no cover - argparse enforces the command set
        raise AssertionError(args.command)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":  # pragma: no cover
    main()
