"""Seal V41 retry1's failed update-8 train-only gate.

This audit is deliberately offline and read-only.  It authenticates the exact
retry1 update-zero and update-eight checkpoint envelopes, replays tensor,
optimizer, projected-gradient-history, gate, and retry-provenance evidence,
and authorizes only one bounded V42 train-only *no-step* diagnostic screen.
It does not load Gemma, QA, maps, validation, oracle, or final-scene inputs.
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
from semantic_3d_chat.training.train_projected_gradient_v41 import (
    replay_v41_gates,
    v41_contract,
    validate_v41_projection_history,
)

DEFAULT_CONFIG = Path(
    "configs/experiments/gemma4_diverse28_projected_gradient_v41_retry1.yaml"
)
DEFAULT_CHECKPOINT_ROOT = Path(
    "data_gemma4/checkpoints/"
    "gemma4_v41_retry1_diverse28_projected_gradient_l14_query"
)
DEFAULT_OUTPUT = Path(
    "reports/gemma4/metrics/v41_retry1_update8_terminal_gate.json"
)
RETRY1_TERMINAL = Path(
    "reports/gemma4/metrics/v41_update1_conversion_terminal_gate.json"
)
PROTECTED_REPORT = Path(
    "reports/gemma4/metrics/"
    "training_selection_gemma4_color_mirror_full_vocab_v11_resume36.json"
)
V42_DIAGNOSTIC_OUTPUT = Path(
    "reports/gemma4/metrics/v42_v41_retry1_update8_no_step_diagnostic.json"
)

_V42_ALPHA_GRID = [-1.0, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 1.0, 1.25]
_V42_CANDIDATE_HASHES = {
    "-1": {
        "target": "7ec6142103fa90964e2490637cf39ef17341a707cf246f07014ea0ea4f01ee60",
        "full": "816aad8ba5ddba78222c0e076fecdb4fdd9827e7a4dc432faafd2296df27b266",
    },
    "-0.5": {
        "target": "d875b680fbfe8a7b1575868831023cb5745c1d90bd20877fa26715e62bc1b47e",
        "full": "2c65a26b3a601fcd01db058d49d32f1794392ec3c526529ad822b03eb7aab41c",
    },
    "-0.25": {
        "target": "dce0054992a1db06adaa91ce1e643368c9f217b3e55102364cf9da022454102d",
        "full": "50f2cb5e9ad7e9ad801caf279b8456985a569414823d2f35821eca69cc651641",
    },
    "0": {
        "target": "d0834cc588ee2a9edf08aabedfd01e0a6d2b01c6b6ae7e3a3d764eaddf58cc3e",
        "full": "7b951c6d7ae4f7b50603159f0bc4dfb4d50b5b40f9325134d78d1de1dae87fc0",
    },
    "0.25": {
        "target": "e349de6baa085166af7718492c28e0bbfe4da349b3c0fc456f230d162b269e8e",
        "full": "4e04766b7bf3511d532944356463095ce19d92898e1bb5d94012b9c2669d6ed6",
    },
    "0.5": {
        "target": "8f106bea1daa766b8b454145365393827a618ecdc3de4d33e1d832a633234541",
        "full": "2e14c18278a097fe0fb8c470235cceaa2100d7763f74abbc99b72af739c00591",
    },
    "0.75": {
        "target": "e82cbeb7cf406c9c6a4f2bed92a7dec7b078c63336d296c47f48452806a938db",
        "full": "ba2b64d25476635accf9fc9813ff2e5624384b76d4024bbf97e278fc1bbd9477",
    },
    "1": {
        "target": "2d6a8cdd1c67cf6405b17ea4d8b9eb6d48121ffb6630f7910fe43447872702fd",
        "full": "5ebc17795c35cb15c0e47f1c3d2d15a74e65e277519829d26308eb3b9fd34ce4",
    },
    "1.25": {
        "target": "0b2a1d8a6001d9f08aa7c77c5708a9dc885c2f83cb3c48df4356e3424fee8f84",
        "full": "601960feae0876c554fee633f9ec49cf0c326d6b9b9a60a0d5f09c4f7eca132c",
    },
}

_CONFIG_SHA256 = "4e3adb9b375d0e3ebd4c0936fba62e34a05ad88e015d073d43e55428de0b90c7"
_CONFIG_HASH = "535d7a3935ce"
_RETRY1_TERMINAL_SHA256 = (
    "cefe759791e1d97557f0a230d6605a1ae079d5f32be7fac3e2b806adaf82eef8"
)
_PROTECTED_SHA256 = (
    "c0086f66edbb8854a7938e09c57535bfd47100adbaf3b3c95eeb4b08014ce2f8"
)
_FILES = {
    "update_000/adapter.safetensors": (
        "b0bdceb7699e9d97467915c69186f433d3b0fac2b09144d38c3078afe1f70cb0"
    ),
    "update_000/metadata.json": (
        "331cda3f2ebc1539e8ee27ebbae398be5e19f3fd77d0aa20dde635d569e29d6d"
    ),
    "update_000/runtime_metadata.json": (
        "690e790b612e0b75323c1f27f7e9afe87243ccc1564c8cc690e86a442cffbfcd"
    ),
    "update_008/adapter.safetensors": (
        "db622c4069a8bcc546172c61d027af9c0d0570ceb9952d4f011b9ad82fd60d7a"
    ),
    "update_008/metadata.json": (
        "034c9e21bbd270832e7cf3f146f71e762ea2ae709a0131071fb187100c04ee28"
    ),
    "update_008/optimizer.pt": (
        "08da3ce96b8e6dcaba774a92757a0cde8a20961c94c884305e9e9b1468a40f00"
    ),
    "update_008/optimizer_audit.json": (
        "c6336fd9cc61444b573d0076a2761b08359c629a1725e72216ca91d2cb7791be"
    ),
    "update_008/runtime_metadata.json": (
        "e11cbf647f32c6db961620b2be03d8d9b2e6b84fc8c18c8d2c7a8856a5dc27b7"
    ),
}
_TARGET = "lora_banks.extension_v28_stage_b_query.adapters.1.lora_b"
_V28_PREFIX = "lora_banks.extension_v28_stage_b_query."
_FULL_U0_SHA256 = (
    "7b951c6d7ae4f7b50603159f0bc4dfb4d50b5b40f9325134d78d1de1dae87fc0"
)
_FULL_U8_SHA256 = (
    "5ebc17795c35cb15c0e47f1c3d2d15a74e65e277519829d26308eb3b9fd34ce4"
)
_TARGET_U0_SHA256 = (
    "d0834cc588ee2a9edf08aabedfd01e0a6d2b01c6b6ae7e3a3d764eaddf58cc3e"
)
_TARGET_U8_SHA256 = (
    "2d6a8cdd1c67cf6405b17ea4d8b9eb6d48121ffb6630f7910fe43447872702fd"
)
_V28_U0_SHA256 = (
    "cc9dfa838bb87f32e2922d675658af4a1085d53a84ccdca6d5bacc6f7097217b"
)
_V28_U8_SHA256 = (
    "9750a2fa250363daa4f198bc344201bd0761ecb79ff7d83de68b7fbd0bdea76a"
)
_FROZEN_SHA256 = (
    "cec01bc088bb87c6bb44e0659eb03aa766f951ddeee706ca9a70edaa080dea5e"
)
_SOURCE_BROAD_NLL = 2.9013306349515915
_UPDATE8_BROAD_NLL = 2.914914463957151
_SOURCE_PRIORITY_DEFICIT = 31.113729119300842
_UPDATE8_PRIORITY_DEFICIT = 31.69202709197998
_UPDATE8_PRIORITY_IMPROVEMENT = -0.5782979726791382


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


def _canonical_sha256(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _authenticate_inventory(root: Path) -> dict[str, Any]:
    if root.is_symlink() or not root.is_dir():
        raise FileNotFoundError("V41 retry1 root must be a real directory")
    if sorted(path.name for path in root.iterdir()) != ["update_000", "update_008"]:
        raise ValueError("V41 retry1 root inventory changed")
    expected_by_arm = {
        "update_000": [
            "adapter.safetensors",
            "metadata.json",
            "runtime_metadata.json",
        ],
        "update_008": [
            "adapter.safetensors",
            "metadata.json",
            "optimizer.pt",
            "optimizer_audit.json",
            "runtime_metadata.json",
        ],
    }
    for arm, expected in expected_by_arm.items():
        directory = root / arm
        if directory.is_symlink() or not directory.is_dir():
            raise FileNotFoundError(f"V41 retry1 {arm} must be a real directory")
        if sorted(path.name for path in directory.iterdir()) != expected:
            raise ValueError(f"V41 retry1 {arm} inventory changed")
    for relative, expected in _FILES.items():
        _locked_file(root / relative, expected, f"V41 retry1 {relative}")
    return {
        "root": _relative(root),
        "root_entries": ["update_000", "update_008"],
        "update_000_entries": expected_by_arm["update_000"],
        "update_008_entries": expected_by_arm["update_008"],
        "file_sha256": dict(_FILES),
        "manifest_sha256": _canonical_sha256(_FILES),
        "no_update_after_eight_persisted": True,
    }


def _authenticate_runtime(
    root: Path, metadata: Mapping[str, Any], *, step: int
) -> dict[str, Any]:
    runtime = _read_json(
        root / f"update_{step:03d}/runtime_metadata.json",
        _FILES[f"update_{step:03d}/runtime_metadata.json"],
        f"V41 retry1 update-{step} runtime metadata",
    )
    validate_runtime_checkpoint_metadata(runtime)
    if runtime != runtime_checkpoint_metadata(metadata):
        raise ValueError(f"V41 retry1 update-{step} runtime sanitization changed")
    return {
        "optimizer_step": step,
        "sanitized_runtime_exact": True,
        "environmental_training_history_absent": True,
        "runtime_metadata_sha256": _FILES[
            f"update_{step:03d}/runtime_metadata.json"
        ],
    }


def _authenticate_tensors(root: Path) -> dict[str, Any]:
    update0 = load_file(root / "update_000/adapter.safetensors", device="cpu")
    update8 = load_file(root / "update_008/adapter.safetensors", device="cpu")
    if len(update0) != 179 or set(update0) != set(update8):
        raise ValueError("V41 retry1 tensor inventory changed")
    if any(not torch.isfinite(value).all() for value in (*update0.values(), *update8.values())):
        raise ValueError("V41 retry1 adapter contains a nonfinite tensor")
    changed = sorted(
        name for name in update0 if not torch.equal(update0[name], update8[name])
    )
    if changed != [_TARGET]:
        raise ValueError(f"V41 retry1 changed an unauthorized tensor: {changed}")
    full = (tensor_state_sha256(update0), tensor_state_sha256(update8))
    target = (
        tensor_state_sha256({"lora_b": update0[_TARGET]}),
        tensor_state_sha256({"lora_b": update8[_TARGET]}),
    )
    frozen0 = {name: value for name, value in update0.items() if name != _TARGET}
    frozen8 = {name: value for name, value in update8.items() if name != _TARGET}
    frozen = (tensor_state_sha256(frozen0), tensor_state_sha256(frozen8))
    v28 = []
    for tensors in (update0, update8):
        bank = {
            name.removeprefix(_V28_PREFIX): value
            for name, value in tensors.items()
            if name.startswith(_V28_PREFIX)
        }
        v28.append(tensor_state_sha256(bank))
    if (
        full != (_FULL_U0_SHA256, _FULL_U8_SHA256)
        or target != (_TARGET_U0_SHA256, _TARGET_U8_SHA256)
        or frozen != (_FROZEN_SHA256, _FROZEN_SHA256)
        or tuple(v28) != (_V28_U0_SHA256, _V28_U8_SHA256)
        or tuple(update0[_TARGET].shape) != (4096, 4)
        or update0[_TARGET].dtype != torch.float32
        or update8[_TARGET].dtype != torch.float32
    ):
        raise ValueError("V41 retry1 tensor-state transition changed")
    return {
        "tensor_count": 179,
        "changed_tensor_names": changed,
        "changed_tensor_count": 1,
        "changed_parameter_count": 16_384,
        "target_shape": [4096, 4],
        "target_dtype": "torch.float32",
        "full_tensor_state_sha256": {"update_000": full[0], "update_008": full[1]},
        "target_lora_b_state_sha256": {
            "update_000": target[0],
            "update_008": target[1],
        },
        "complete_v28_bank_state_sha256": {
            "update_000": v28[0],
            "update_008": v28[1],
        },
        "frozen_excluding_target_state_sha256": {
            "update_000": frozen[0],
            "update_008": frozen[1],
        },
        "only_existing_layer14_q_proj_lora_b_changed": True,
        "every_other_tensor_bit_exact": True,
        "all_tensors_finite": True,
    }


def _authenticate_optimizer(root: Path) -> dict[str, Any]:
    manifest = _read_json(
        root / "update_008/optimizer_audit.json",
        _FILES["update_008/optimizer_audit.json"],
        "V41 retry1 update-eight optimizer audit",
    )
    expected_manifest = {
        "schema_version": 1,
        "artifact": "v41_optimizer_integrity_manifest",
        "optimizer_step": 8,
        "optimizer_filename": "optimizer.pt",
        "optimizer_sha256": _FILES["update_008/optimizer.pt"],
    }
    if dict(manifest) != expected_manifest:
        raise ValueError("V41 retry1 optimizer manifest changed")
    payload = torch.load(
        root / "update_008/optimizer.pt", map_location="cpu", weights_only=True
    )
    expected_group = {
        "name": "lora_banks.extension_v28_stage_b_query.adapters.1",
        "parameter_names": [_TARGET],
        "lr": 0.003,
        "weight_decay": 0.0,
        "momentum": 0.0,
        "dampening": 0.0,
        "nesterov": False,
        "maximize": False,
        "foreach": False,
        "differentiable": False,
        "fused": False,
        "params": [0],
    }
    if (
        not isinstance(payload, Mapping)
        or set(payload) != {"state", "param_groups"}
        or payload.get("state") != {}
        or payload.get("param_groups") != [expected_group]
        or (root / "update_000/optimizer.pt").exists()
    ):
        raise ValueError("V41 retry1 optimizer payload changed")
    return {
        "optimizer_step": 8,
        "optimizer_sha256": _FILES["update_008/optimizer.pt"],
        "manifest_sha256": _FILES["update_008/optimizer_audit.json"],
        "implementation": "torch.optim.SGD",
        "stateless_momentum_free_payload": True,
        "state_entry_count": 0,
        "parameter_group_count": 1,
        "parameter_names": [_TARGET],
        "learning_rate": 0.003,
        "weight_decay": 0.0,
        "momentum": 0.0,
        "foreach": False,
        "fused": False,
        "update_zero_optimizer_absent": True,
    }


def _authenticate_retry_provenance(
    *,
    root: Path,
    update0: Mapping[str, Any],
    update8: Mapping[str, Any],
    retry_terminal: Mapping[str, Any],
) -> dict[str, Any]:
    stage0 = _mapping(update0.get("v41_projected_gradient"), "V41 retry1 u0 stage")
    stage8 = _mapping(update8.get("v41_projected_gradient"), "V41 retry1 u8 stage")
    expected_gate = {
        "path": str(_resolve(RETRY1_TERMINAL)),
        "sha256": _RETRY1_TERMINAL_SHA256,
    }
    authorization = _mapping(
        retry_terminal.get("conditional_successor_authorization"),
        "V41 retry1 terminal authorization",
    )
    predecessor0 = _mapping(
        stage0.get("retry1_predecessor_attestation"), "V41 retry1 u0 predecessor"
    )
    predecessor8 = _mapping(
        stage8.get("retry1_predecessor_attestation"), "V41 retry1 u8 predecessor"
    )
    expected_root = str(root)
    if (
        retry_terminal.get("artifact") != "v41_update1_conversion_terminal_gate"
        or retry_terminal.get("passed") is not True
        or retry_terminal.get(
            "v41_retry1_train_only_projected_gradient_continuation_authorized"
        )
        is not True
        or stage0.get("conditional_v41_retry1_terminal_gate") != expected_gate
        or stage8.get("conditional_v41_retry1_terminal_gate") != expected_gate
        or stage0.get("retry1_conditional_authorization") != authorization
        or stage8.get("retry1_conditional_authorization") != authorization
        or predecessor0 != predecessor8
        or stage0.get("authorized_output_root") != expected_root
        or stage8.get("authorized_output_root") != expected_root
        or authorization.get("authorized_output_root")
        != _relative(_resolve(DEFAULT_CHECKPOINT_ROOT))
        or authorization.get("cpu_first_mps_conversion_required") is not True
        or authorization.get("objective_schedule_gates_unchanged") is not True
        or any(
            authorization.get(field) is not False
            for field in (
                "validation_access_authorized",
                "oracle_access_authorized",
                "final_test_access_authorized",
                "selector_execution_authorized",
                "chat_or_runtime_promotion_authorized",
            )
        )
        or predecessor0.get("target_and_frozen_state_unchanged") is not True
        or predecessor0.get("validation_qa_loaded") is not False
        or predecessor0.get("oracle_environment_files_loaded") is not False
        or predecessor0.get("final_test_scenes_loaded") is not False
    ):
        raise ValueError("V41 retry1 authorization provenance changed")
    return {
        "retry1_terminal_report": _relative(_resolve(RETRY1_TERMINAL)),
        "retry1_terminal_report_sha256": _RETRY1_TERMINAL_SHA256,
        "authorization_id": authorization.get("authorization_id"),
        "authorization_revision": authorization.get("authorization_revision"),
        "authorized_output_root": _relative(root),
        "cpu_first_mps_conversion_required": True,
        "predecessor_attestation_sha256": _canonical_sha256(predecessor0),
        "same_authorization_and_predecessor_persisted_at_u0_and_u8": True,
        "objective_schedule_gates_unchanged": True,
        "restricted_access_remained_denied": True,
    }


def _authenticate_history_and_gate(
    update8: Mapping[str, Any], config: Mapping[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    contract = v41_contract(config)
    history = _sequence(update8.get("history"), "V41 retry1 update-eight history")
    if len(history) != 9 or [row.get("optimizer_update") for row in history] != list(
        range(9)
    ):
        raise ValueError("V41 retry1 history is not updates zero through eight")
    projection = validate_v41_projection_history(history, contract)
    stage = _mapping(update8.get("v41_projected_gradient"), "V41 retry1 u8 stage")
    if projection != stage.get("projection_history_attestation"):
        raise ValueError("V41 retry1 projection-history attestation changed")
    gate8, gate16, gate41 = replay_v41_gates(update8, contract)
    row8 = _mapping(history[8], "V41 retry1 update-eight row")
    if gate8 != row8.get("update8_train_only_gate") or gate8 != stage.get(
        "update8_train_only_gate"
    ):
        raise ValueError("V41 retry1 update-eight gate replay changed")
    gate = _mapping(gate8, "V41 retry1 replayed update-eight gate")
    checks = _mapping(gate.get("checks"), "V41 retry1 gate checks")
    expected_failed = {
        "priority_teacher_deficit_improved_at_least_0_5",
        "teacher_positive_sides_at_least_34",
        "teacher_cross_complete_units_at_least_17",
    }
    failed = {name for name, value in checks.items() if value is not True}
    pair = _mapping(row8.get("training_pair_metrics"), "V41 retry1 u8 pair metrics")
    if (
        gate.get("passed") is not False
        or failed != expected_failed
        or pair.get("complete_units") != 9
        or pair.get("positive_sides") != 33
        or pair.get("cross_prefix_complete_units") != 16
        or float(row8.get("training_broad_nll")) != _UPDATE8_BROAD_NLL
        or float(stage.get("source_broad_train_nll")) != _SOURCE_BROAD_NLL
        or float(gate.get("source_priority_teacher_side_deficit"))
        != _SOURCE_PRIORITY_DEFICIT
        or float(gate.get("priority_teacher_side_deficit"))
        != _UPDATE8_PRIORITY_DEFICIT
        or float(gate.get("priority_teacher_side_deficit_improvement"))
        != _UPDATE8_PRIORITY_IMPROVEMENT
        or stage.get("update16_train_only_gate") is not None
        or stage.get("update41_train_only_gate") is not None
        or gate16 is not None
        or gate41 is not None
        or row8.get("scene_prefix_and_residual_exact") is not True
        or row8.get("frozen_excluding_query_state_sha256") != _FROZEN_SHA256
        or row8.get("query_bank_state_sha256") != _TARGET_U8_SHA256
    ):
        raise ValueError("V41 retry1 failed update-eight gate evidence changed")
    masks = [
        int(_mapping(row.get("projected_gradient_attestation"), "projection")["selected_mask"])
        for row in history[1:]
    ]
    active_counts = [
        int(
            _mapping(row.get("projected_gradient_attestation"), "projection")[
                "active_constraint_count"
            ]
        )
        for row in history[1:]
    ]
    return (
        {
            **projection,
            "optimizer_steps": list(range(1, 9)),
            "selected_masks": masks,
            "active_constraint_counts": active_counts,
            "projected_steps": [index for index, mask in enumerate(masks, 1) if mask],
            "all_target_and_frozen_hash_chains_authenticated": True,
            "all_device_cast_and_clip_attestations_authenticated": True,
        },
        {
            "passed": False,
            "replayed_exactly": True,
            "failed_checks": sorted(failed),
            "passing_checks": sorted(set(checks) - failed),
            "complete_units": 9,
            "positive_sides": 33,
            "cross_prefix_complete_units": 16,
            "source_broad_nll": _SOURCE_BROAD_NLL,
            "update8_broad_nll": _UPDATE8_BROAD_NLL,
            "broad_nll_delta": float(gate["broad_nll_delta_from_update_zero"]),
            "source_priority_side_deficit": _SOURCE_PRIORITY_DEFICIT,
            "update8_priority_side_deficit": _UPDATE8_PRIORITY_DEFICIT,
            "priority_side_deficit_improvement": _UPDATE8_PRIORITY_IMPROVEMENT,
            "stopped_before_update_nine": True,
            "update16_gate_absent": True,
            "update41_gate_absent": True,
        },
    )


def audit_v41_retry1_update8(
    config_path: str | Path = DEFAULT_CONFIG,
    checkpoint_root: str | Path = DEFAULT_CHECKPOINT_ROOT,
) -> dict[str, Any]:
    """Replay the immutable retry1 terminal evidence without environment access."""

    config_file = _resolve(config_path)
    root = _resolve(checkpoint_root)
    _locked_file(config_file, _CONFIG_SHA256, "V41 retry1 config")
    config = load_config(config_file)
    if config_hash(config) != _CONFIG_HASH:
        raise ValueError("V41 retry1 normalized config hash changed")
    inventory = _authenticate_inventory(root)
    retry_terminal = _read_json(
        _resolve(RETRY1_TERMINAL),
        _RETRY1_TERMINAL_SHA256,
        "V41 retry1 launch terminal",
    )
    _locked_file(
        _resolve(PROTECTED_REPORT), _PROTECTED_SHA256, "protected selection report"
    )
    update0 = _read_json(
        root / "update_000/metadata.json",
        _FILES["update_000/metadata.json"],
        "V41 retry1 update-zero metadata",
    )
    update8 = _read_json(
        root / "update_008/metadata.json",
        _FILES["update_008/metadata.json"],
        "V41 retry1 update-eight metadata",
    )
    if (
        update0.get("schema_version") != 1
        or update0.get("config_hash") != _CONFIG_HASH
        or update0.get("optimizer_step") != 0
        or update0.get("epoch") != 0
        or len(_sequence(update0.get("history"), "V41 retry1 u0 history")) != 1
        or update8.get("schema_version") != 1
        or update8.get("config_hash") != _CONFIG_HASH
        or update8.get("optimizer_step") != 8
        or update8.get("epoch") != 8
    ):
        raise ValueError("V41 retry1 checkpoint identity changed")
    runtime = {
        "update_000": _authenticate_runtime(root, update0, step=0),
        "update_008": _authenticate_runtime(root, update8, step=8),
    }
    tensors = _authenticate_tensors(root)
    optimizer = _authenticate_optimizer(root)
    provenance = _authenticate_retry_provenance(
        root=root,
        update0=update0,
        update8=update8,
        retry_terminal=retry_terminal,
    )
    projection, gate = _authenticate_history_and_gate(update8, config)
    stage0 = _mapping(update0.get("v41_projected_gradient"), "V41 retry1 u0 stage")
    stage8 = _mapping(update8.get("v41_projected_gradient"), "V41 retry1 u8 stage")
    for stage in (stage0, stage8):
        if (
            stage.get("validation_qa_loaded") is not False
            or stage.get("oracle_environment_files_loaded") is not False
            or stage.get("deferred_final_scene_ids_loaded") != []
            or stage.get("question_dependent_scene_processing") is not False
            or stage.get("question_dependent_retrieval") is not False
        ):
            raise ValueError("V41 retry1 crossed its train-only data boundary")

    authorization = {
        "schema_version": 1,
        "authorization_id": "v42_v41_retry1_update8_no_step_diagnostic_screen",
        "authorized": True,
        "only_exact_action": "one_train_only_no_step_diagnostic_screen",
        "source_checkpoint_root": _relative(root),
        "source_optimizer_step": 8,
        "source_update8_adapter_sha256": _FILES[
            "update_008/adapter.safetensors"
        ],
        "source_update8_tensor_state_sha256": _FULL_U8_SHA256,
        "source_update8_target_lora_b_state_sha256": _TARGET_U8_SHA256,
        "source_frozen_excluding_target_state_sha256": _FROZEN_SHA256,
        "authorized_output": str(V42_DIAGNOSTIC_OUTPUT),
        "diagnostic_scope": {
            "single_report_only": True,
            "exact_training_qa_and_maps_read_only": True,
            "training_scene_scope_only": True,
            "read_only_u0_u8_checkpoint_comparison": True,
            "forward_only_candidate_diagnostics_allowed": True,
            "gradient_measurement_authorized": False,
            "parameter_mutation_authorized": False,
            "persistent_parameter_mutation_authorized": False,
            "temporary_target_b_substitution_authorized": True,
            "temporary_substitution_must_restore_exact_u0_after_each_candidate": True,
            "temporary_substitution_formula": (
                "float32(float64(B0) + alpha * (float64(B8) - float64(B0)))"
            ),
            "fixed_alpha_grid": list(_V42_ALPHA_GRID),
            "fixed_candidate_state_sha256": dict(_V42_CANDIDATE_HASHES),
            "adaptive_candidate_refinement_authorized": False,
            "optimizer_construction_authorized": False,
            "optimizer_deserialization_authorized": False,
            "optimizer_step_authorized": False,
            "checkpoint_write_authorized": False,
            "resume_training_authorized": False,
            "candidate_training_authorized": False,
        },
        "required_diagnostics": [
            "all_25_changed_train_units_for_every_fixed_alpha",
            "all_25_per_unit_nll_diagnostics_for_every_fixed_alpha",
            "fixed_48_row_broad_nll_for_every_fixed_alpha",
            "alpha_zero_exact_update_zero_endpoint_replay",
            "alpha_one_exact_failed_update_eight_endpoint_replay",
            "deterministic_teacher_only_candidate_ranking",
            "selected_candidate_exact_replay_if_any",
            "optional_selected_candidate_greedy_audit_not_used_for_selection",
            "exact_update_zero_restoration_after_every_candidate",
            "evidence_based_next_experiment_recommendation_without_authorization",
        ],
        "validation_access_authorized": False,
        "oracle_access_authorized": False,
        "final_test_access_authorized": False,
        "selector_execution_authorized": False,
        "chat_or_runtime_promotion_authorized": False,
        "embodied_agent_promotion_authorized": False,
        "new_terminal_seal_required_after_diagnostic": True,
    }
    return {
        "schema_version": 1,
        "seal_revision": 2,
        "artifact": "v41_retry1_update8_terminal_gate",
        "passed": True,
        "terminal_conclusion": "update8_train_only_gate_failed_stop_is_final",
        "checkpoint_inventory": inventory,
        "tensor_transition": tensors,
        "optimizer_integrity": optimizer,
        "projection_history_replay": projection,
        "update8_gate_replay": gate,
        "retry1_provenance": provenance,
        "runtime_metadata_audit": runtime,
        "execution_conclusion": {
            "updates_one_through_eight_executed": True,
            "update8_checkpoint_persisted": True,
            "update8_gate_passed": False,
            "update_nine_executed": False,
            "no_checkpoint_after_update_eight": True,
            "frozen_state_exact": True,
            "validation_loaded": False,
            "oracle_loaded": False,
            "final_test_loaded": False,
            "selector_executed": False,
        },
        "conditional_successor_authorization": authorization,
        "only_exact_successor_authorized": (
            "v42_train_only_no_step_diagnostic_screen"
        ),
        "v42_train_only_no_step_diagnostic_screen_authorized": True,
        "arbitrary_training_authorized": False,
        "resume_v41_training_authorized": False,
        "parameter_update_authorized": False,
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
            "optimizer_deserialized_on_cpu_for_integrity_audit": True,
            "optimizer_step_executed": False,
            "protected_report_access": "bytes_hashed_only",
        },
        "input_integrity": {
            "config_sha256": _CONFIG_SHA256,
            "normalized_config_hash": _CONFIG_HASH,
            "retry1_terminal_sha256": _RETRY1_TERMINAL_SHA256,
            "protected_report_sha256": _PROTECTED_SHA256,
            "checkpoint_manifest_sha256": inventory["manifest_sha256"],
        },
    }


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
        delete=False,
    ) as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
        temporary = Path(handle.name)
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def write_report(
    output: str | Path = DEFAULT_OUTPUT,
    *,
    config_path: str | Path = DEFAULT_CONFIG,
    checkpoint_root: str | Path = DEFAULT_CHECKPOINT_ROOT,
) -> dict[str, Any]:
    report = audit_v41_retry1_update8(config_path, checkpoint_root)
    _atomic_json(_resolve(output), report)
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


__all__ = ["audit_v41_retry1_update8", "write_report"]
