"""Canonical AdamW evidence for V19's one-matrix optimizer surface."""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping
from typing import Any

import torch

from semantic_3d_chat.language.lora import tensor_state_sha256

V19_ADAMW_STATE_SCHEMA_VERSION = 1
V19_SIGNED_X_OPTIMIZER_GROUP_NAME = "signed_x_output_projection"
V19_SIGNED_X_PARAMETER_NAME = "output_projection.weight"
V19_SIGNED_X_PARAMETER_SHAPE = (1536, 128)

_STATE_KEYS = {"step", "exp_avg", "exp_avg_sq"}
_PARAM_GROUP_KEYS = {
    "name",
    "lr",
    "weight_decay",
    "betas",
    "eps",
    "amsgrad",
    "maximize",
    "foreach",
    "capturable",
    "differentiable",
    "fused",
    "decoupled_weight_decay",
    "params",
}
_SHA256 = re.compile(r"[0-9a-f]{64}")


class V19AdamWStateViolation(ValueError):
    """The state is not the exact one-parameter V19 AdamW contract."""


def _fail(message: str) -> None:
    raise V19AdamWStateViolation(message)


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _mapping(value: Any, field: str) -> Mapping[Any, Any]:
    if not isinstance(value, Mapping):
        _fail(f"{field} must be an object")
    return value


def _strict_keys(value: Mapping[Any, Any], expected: set[Any], field: str) -> None:
    if set(value) != expected:
        _fail(
            f"{field} keys mismatch: missing={sorted(expected - set(value), key=str)} "
            f"unknown={sorted(set(value) - expected, key=str)}"
        )


def _finite_equal(value: Any, expected: float, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        _fail(f"{field} must be numeric")
    observed = float(value)
    if not math.isfinite(observed) or observed != expected:
        _fail(f"{field} mismatch: expected={expected!r} observed={value!r}")
    return observed


def _boolean(value: Any, expected: bool, field: str) -> bool:
    if type(value) is not bool or value is not expected:
        _fail(f"{field} mismatch: expected={expected!r} observed={value!r}")
    return value


def _group_manifest(
    value: Mapping[str, Any], optimizer_contract: Mapping[str, Any]
) -> dict[str, Any]:
    _strict_keys(value, _PARAM_GROUP_KEYS, "AdamW param_groups[0]")
    if value["name"] != V19_SIGNED_X_OPTIMIZER_GROUP_NAME:
        _fail(
            "AdamW group name mismatch: "
            f"expected={V19_SIGNED_X_OPTIMIZER_GROUP_NAME!r} observed={value['name']!r}"
        )
    if value["params"] != [0] or any(type(item) is not int for item in value["params"]):
        _fail("AdamW parameter order must be exactly [0]")
    betas = value["betas"]
    if not isinstance(betas, tuple) or len(betas) != 2:
        _fail("AdamW betas must be an exact two-value tuple")
    expected_betas = tuple(float(item) for item in optimizer_contract["betas"])
    normalized_betas = [
        _finite_equal(observed, expected, f"AdamW beta[{index}]")
        for index, (observed, expected) in enumerate(zip(betas, expected_betas, strict=True))
    ]
    return {
        "name": value["name"],
        "lr": _finite_equal(value["lr"], float(optimizer_contract["learning_rate"]), "AdamW lr"),
        "weight_decay": _finite_equal(
            value["weight_decay"],
            float(optimizer_contract["weight_decay"]),
            "AdamW weight_decay",
        ),
        "betas": normalized_betas,
        "eps": _finite_equal(value["eps"], float(optimizer_contract["epsilon"]), "AdamW eps"),
        "amsgrad": _boolean(value["amsgrad"], bool(optimizer_contract["amsgrad"]), "AdamW amsgrad"),
        "maximize": _boolean(
            value["maximize"], bool(optimizer_contract["maximize"]), "AdamW maximize"
        ),
        "foreach": _boolean(value["foreach"], bool(optimizer_contract["foreach"]), "AdamW foreach"),
        "capturable": _boolean(
            value["capturable"],
            bool(optimizer_contract["capturable"]),
            "AdamW capturable",
        ),
        "differentiable": _boolean(value["differentiable"], False, "AdamW differentiable"),
        "fused": _boolean(value["fused"], bool(optimizer_contract["fused"]), "AdamW fused"),
        "decoupled_weight_decay": _boolean(
            value["decoupled_weight_decay"], True, "AdamW decoupled_weight_decay"
        ),
        "params": [0],
    }


def _tensor_manifest(
    value: Any,
    *,
    expected_shape: tuple[int, ...],
    field: str,
    expected_value: float | None = None,
) -> tuple[dict[str, Any], torch.Tensor]:
    if not isinstance(value, torch.Tensor):
        _fail(f"{field} must be a tensor")
    tensor = value.detach().to(device="cpu").contiguous()
    if tensor.dtype != torch.float32:
        _fail(f"{field} dtype mismatch: expected=float32 observed={tensor.dtype}")
    if tuple(tensor.shape) != expected_shape:
        _fail(f"{field} shape mismatch: expected={expected_shape} observed={tuple(tensor.shape)}")
    if not bool(torch.isfinite(tensor).all()):
        _fail(f"{field} contains non-finite values")
    manifest: dict[str, Any] = {
        "shape": list(expected_shape),
        "dtype": "float32",
        "finite": True,
        "sha256": tensor_state_sha256({field: tensor}),
    }
    if expected_value is not None:
        observed = float(tensor.item())
        if observed != expected_value:
            _fail(f"{field} mismatch: expected={expected_value} observed={observed}")
        manifest["value"] = observed
    return manifest, tensor


def canonical_v19_adamw_state(
    state_dict: Mapping[str, Any],
    optimizer_contract: Mapping[str, Any],
) -> tuple[dict[str, Any], str]:
    """Validate and hash every V19 optimizer option, moment, and step tensor."""

    root = _mapping(state_dict, "AdamW state_dict")
    _strict_keys(root, {"state", "param_groups"}, "AdamW state_dict")
    groups = root["param_groups"]
    if not isinstance(groups, list) or len(groups) != 1:
        _fail("AdamW state_dict must contain exactly one parameter group")
    group = _group_manifest(_mapping(groups[0], "AdamW param_groups[0]"), optimizer_contract)
    states = _mapping(root["state"], "AdamW state")
    if set(states) != {0} or any(type(key) is not int for key in states):
        _fail("AdamW state parameter IDs must be exactly {0}")
    entry = _mapping(states[0], "AdamW state[0]")
    _strict_keys(entry, _STATE_KEYS, "AdamW state[0]")
    tensors: dict[str, torch.Tensor] = {}
    manifests: dict[str, Any] = {}
    for name, shape, expected in (
        ("step", (), float(optimizer_contract["step_index"])),
        ("exp_avg", V19_SIGNED_X_PARAMETER_SHAPE, None),
        ("exp_avg_sq", V19_SIGNED_X_PARAMETER_SHAPE, None),
    ):
        field = f"state.0.{name}"
        manifest, tensor = _tensor_manifest(
            entry[name],
            expected_shape=shape,
            field=field,
            expected_value=expected,
        )
        manifests[name] = manifest
        tensors[field] = tensor
    manifest = {
        "schema_version": V19_ADAMW_STATE_SCHEMA_VERSION,
        "optimizer": "AdamW",
        "state_parameter_count": 1,
        "parameter_order": [
            {
                "parameter_id": 0,
                "name": V19_SIGNED_X_PARAMETER_NAME,
                "shape": list(V19_SIGNED_X_PARAMETER_SHAPE),
                "dtype": "float32",
            }
        ],
        "param_groups": [group],
        "states": [
            {
                "parameter_id": 0,
                "parameter_name": V19_SIGNED_X_PARAMETER_NAME,
                "state": manifests,
            }
        ],
        "all_state_tensors_sha256": tensor_state_sha256(tensors),
    }
    return manifest, _canonical_sha256(manifest)


def validate_v19_adamw_state_manifest(
    value: Mapping[str, Any], optimizer_contract: Mapping[str, Any]
) -> str:
    """Validate a JSON preflight manifest before optimizer deserialization."""

    manifest = _mapping(value, "AdamW state manifest")
    _strict_keys(
        manifest,
        {
            "schema_version",
            "optimizer",
            "state_parameter_count",
            "parameter_order",
            "param_groups",
            "states",
            "all_state_tensors_sha256",
        },
        "AdamW state manifest",
    )
    if manifest["schema_version"] != V19_ADAMW_STATE_SCHEMA_VERSION:
        _fail("AdamW state manifest schema mismatch")
    if manifest["optimizer"] != "AdamW" or manifest["state_parameter_count"] != 1:
        _fail("AdamW state manifest root contract mismatch")
    expected_order = [
        {
            "parameter_id": 0,
            "name": V19_SIGNED_X_PARAMETER_NAME,
            "shape": list(V19_SIGNED_X_PARAMETER_SHAPE),
            "dtype": "float32",
        }
    ]
    if manifest["parameter_order"] != expected_order:
        _fail("AdamW state manifest parameter order mismatch")
    groups = manifest["param_groups"]
    if not isinstance(groups, list) or len(groups) != 1:
        _fail("AdamW state manifest must contain exactly one group")
    raw_group = dict(_mapping(groups[0], "AdamW state manifest group"))
    if not isinstance(raw_group.get("betas"), list):
        _fail("AdamW state manifest betas must be a list")
    raw_group["betas"] = tuple(raw_group["betas"])
    _group_manifest(raw_group, optimizer_contract)
    states = manifest["states"]
    if not isinstance(states, list) or len(states) != 1:
        _fail("AdamW state manifest must contain one parameter state")
    state = _mapping(states[0], "AdamW state manifest state[0]")
    _strict_keys(state, {"parameter_id", "parameter_name", "state"}, "manifest state[0]")
    if state["parameter_id"] != 0 or state["parameter_name"] != V19_SIGNED_X_PARAMETER_NAME:
        _fail("AdamW state manifest parameter identity mismatch")
    entries = _mapping(state["state"], "manifest state[0].state")
    _strict_keys(entries, _STATE_KEYS, "manifest state[0].state")
    for name, shape in (
        ("step", ()),
        ("exp_avg", V19_SIGNED_X_PARAMETER_SHAPE),
        ("exp_avg_sq", V19_SIGNED_X_PARAMETER_SHAPE),
    ):
        entry = _mapping(entries[name], f"manifest state[0].{name}")
        expected_keys = {"shape", "dtype", "finite", "sha256"}
        if name == "step":
            expected_keys.add("value")
        _strict_keys(entry, expected_keys, f"manifest state[0].{name}")
        if entry["shape"] != list(shape) or entry["dtype"] != "float32":
            _fail(f"AdamW state manifest tensor metadata mismatch for {name}")
        if entry["finite"] is not True or not isinstance(entry["sha256"], str):
            _fail(f"AdamW state manifest tensor evidence invalid for {name}")
        if _SHA256.fullmatch(entry["sha256"]) is None:
            _fail(f"AdamW state manifest digest invalid for {name}")
        if name == "step" and entry["value"] != float(optimizer_contract["step_index"]):
            _fail("AdamW state manifest step mismatch")
    combined = manifest["all_state_tensors_sha256"]
    if not isinstance(combined, str) or _SHA256.fullmatch(combined) is None:
        _fail("AdamW state manifest combined tensor digest is invalid")
    return _canonical_sha256(dict(manifest))


__all__ = [
    "V19_ADAMW_STATE_SCHEMA_VERSION",
    "V19_SIGNED_X_OPTIMIZER_GROUP_NAME",
    "V19_SIGNED_X_PARAMETER_NAME",
    "V19_SIGNED_X_PARAMETER_SHAPE",
    "V19AdamWStateViolation",
    "canonical_v19_adamw_state",
    "validate_v19_adamw_state_manifest",
]
