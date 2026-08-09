"""Canonical, fail-closed AdamW state evidence for the V18 residual."""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping, Sequence
from typing import Any

import torch

from semantic_3d_chat.language.lora import tensor_state_sha256

V18_ADAMW_STATE_SCHEMA_VERSION = 1
V18_RESIDUAL_OPTIMIZER_GROUP_NAME = "global_scene_residual"
V18_RESIDUAL_PARAMETER_SPECS: tuple[tuple[str, tuple[int, ...]], ...] = (
    ("scene_norm.weight", (1536,)),
    ("scene_norm.bias", (1536,)),
    ("scene_projection.weight", (128, 1536)),
    ("scene_projection.bias", (128,)),
    ("position_projection.weight", (128, 27)),
    ("position_projection.bias", (128,)),
    ("content_gate_projection.weight", (1, 128)),
    ("output_projection.weight", (1536, 128)),
)

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


class V18AdamWStateViolation(ValueError):
    """An optimizer state cannot be the exact V18 update-one state."""


def _fail(message: str) -> None:
    raise V18AdamWStateViolation(message)


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


def _finite_number(value: Any, expected: float, field: str) -> float:
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


def _optimizer_group_manifest(
    group: Mapping[str, Any],
    optimizer_contract: Mapping[str, Any],
    *,
    parameter_count: int,
) -> dict[str, Any]:
    _strict_keys(group, _PARAM_GROUP_KEYS, "AdamW param_groups[0]")
    if group["name"] != V18_RESIDUAL_OPTIMIZER_GROUP_NAME:
        _fail(
            "AdamW param-group name mismatch: "
            f"expected={V18_RESIDUAL_OPTIMIZER_GROUP_NAME!r} observed={group['name']!r}"
        )
    params = group["params"]
    if not isinstance(params, list):
        _fail("AdamW param_groups[0].params must be a list")
    expected_ids = list(range(parameter_count))
    if params != expected_ids or any(type(value) is not int for value in params):
        _fail(f"AdamW parameter order mismatch: expected={expected_ids} observed={params!r}")
    betas = group["betas"]
    if not isinstance(betas, tuple) or len(betas) != 2:
        _fail("AdamW param_groups[0].betas must be the exact two-value tuple")
    expected_betas = tuple(float(value) for value in optimizer_contract["betas"])
    observed_betas = tuple(
        _finite_number(value, expected, f"AdamW beta[{index}]")
        for index, (value, expected) in enumerate(zip(betas, expected_betas, strict=True))
    )
    normalized: dict[str, Any] = {
        "name": group["name"],
        "lr": _finite_number(group["lr"], float(optimizer_contract["learning_rate"]), "AdamW lr"),
        "weight_decay": _finite_number(
            group["weight_decay"],
            float(optimizer_contract["weight_decay"]),
            "AdamW weight_decay",
        ),
        "betas": list(observed_betas),
        "eps": _finite_number(group["eps"], float(optimizer_contract["epsilon"]), "AdamW eps"),
        "amsgrad": _boolean(group["amsgrad"], bool(optimizer_contract["amsgrad"]), "AdamW amsgrad"),
        "maximize": _boolean(
            group["maximize"],
            bool(optimizer_contract["maximize"]),
            "AdamW maximize",
        ),
        "foreach": _boolean(group["foreach"], bool(optimizer_contract["foreach"]), "AdamW foreach"),
        "capturable": _boolean(
            group["capturable"],
            bool(optimizer_contract["capturable"]),
            "AdamW capturable",
        ),
        "differentiable": _boolean(group["differentiable"], False, "AdamW differentiable"),
        "fused": _boolean(group["fused"], bool(optimizer_contract["fused"]), "AdamW fused"),
        "decoupled_weight_decay": _boolean(
            group["decoupled_weight_decay"], True, "AdamW decoupled_weight_decay"
        ),
        "params": list(params),
    }
    return normalized


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
        _fail(f"{field} dtype mismatch: expected=torch.float32 observed={tensor.dtype}")
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


def canonical_v18_adamw_state(
    state_dict: Mapping[str, Any],
    optimizer_contract: Mapping[str, Any],
    *,
    parameter_specs: Sequence[tuple[str, tuple[int, ...]]] = V18_RESIDUAL_PARAMETER_SPECS,
) -> tuple[dict[str, Any], str]:
    """Validate and canonically hash every tensor and option in V18 AdamW state."""

    root = _mapping(state_dict, "AdamW state_dict")
    _strict_keys(root, {"state", "param_groups"}, "AdamW state_dict")
    groups = root["param_groups"]
    if not isinstance(groups, list) or len(groups) != 1:
        _fail("AdamW state_dict must contain exactly one param group")
    group = _mapping(groups[0], "AdamW param_groups[0]")
    specs = tuple((str(name), tuple(shape)) for name, shape in parameter_specs)
    if not specs:
        _fail("AdamW parameter specification cannot be empty")
    group_manifest = _optimizer_group_manifest(
        group, optimizer_contract, parameter_count=len(specs)
    )

    states = _mapping(root["state"], "AdamW state")
    expected_ids = set(range(len(specs)))
    if set(states) != expected_ids or any(type(value) is not int for value in states):
        _fail(
            "AdamW state parameter IDs mismatch: "
            f"expected={sorted(expected_ids)} observed={sorted(states, key=str)}"
        )

    tensor_payload: dict[str, torch.Tensor] = {}
    state_manifest: list[dict[str, Any]] = []
    parameter_manifest: list[dict[str, Any]] = []
    for parameter_id, (name, shape) in enumerate(specs):
        parameter_manifest.append(
            {
                "parameter_id": parameter_id,
                "name": name,
                "shape": list(shape),
                "dtype": "float32",
            }
        )
        entry = _mapping(states[parameter_id], f"AdamW state[{parameter_id}]")
        _strict_keys(entry, _STATE_KEYS, f"AdamW state[{parameter_id}]")
        tensor_manifests: dict[str, Any] = {}
        for state_name, expected_shape, expected_value in (
            ("step", (), float(optimizer_contract["step_index"])),
            ("exp_avg", shape, None),
            ("exp_avg_sq", shape, None),
        ):
            field = f"state.{parameter_id}.{state_name}"
            tensor_manifest, tensor = _tensor_manifest(
                entry[state_name],
                expected_shape=expected_shape,
                expected_value=expected_value,
                field=field,
            )
            tensor_manifests[state_name] = tensor_manifest
            tensor_payload[field] = tensor
        state_manifest.append(
            {
                "parameter_id": parameter_id,
                "parameter_name": name,
                "state": tensor_manifests,
            }
        )

    manifest = {
        "schema_version": V18_ADAMW_STATE_SCHEMA_VERSION,
        "optimizer": "AdamW",
        "state_parameter_count": len(state_manifest),
        "parameter_order": parameter_manifest,
        "param_groups": [group_manifest],
        "states": state_manifest,
        "all_state_tensors_sha256": tensor_state_sha256(tensor_payload),
    }
    return manifest, _canonical_sha256(manifest)


def validate_v18_adamw_state_manifest(
    value: Mapping[str, Any], optimizer_contract: Mapping[str, Any]
) -> str:
    """Validate a JSON manifest before comparing it with deserialized state."""

    manifest = _mapping(value, "AdamW state manifest")
    expected_root = {
        "schema_version",
        "optimizer",
        "state_parameter_count",
        "parameter_order",
        "param_groups",
        "states",
        "all_state_tensors_sha256",
    }
    _strict_keys(manifest, expected_root, "AdamW state manifest")
    if manifest["schema_version"] != V18_ADAMW_STATE_SCHEMA_VERSION:
        _fail("AdamW state manifest schema mismatch")
    if manifest["optimizer"] != "AdamW":
        _fail("AdamW state manifest optimizer mismatch")
    if manifest["state_parameter_count"] != len(V18_RESIDUAL_PARAMETER_SPECS):
        _fail("AdamW state manifest parameter count mismatch")
    if not isinstance(manifest["parameter_order"], list):
        _fail("AdamW state manifest parameter_order must be a list")
    expected_order = [
        {
            "parameter_id": index,
            "name": name,
            "shape": list(shape),
            "dtype": "float32",
        }
        for index, (name, shape) in enumerate(V18_RESIDUAL_PARAMETER_SPECS)
    ]
    if manifest["parameter_order"] != expected_order:
        _fail("AdamW state manifest parameter order mismatch")
    groups = manifest["param_groups"]
    if not isinstance(groups, list) or len(groups) != 1:
        _fail("AdamW state manifest must contain exactly one param group")
    group = _mapping(groups[0], "AdamW state manifest param_groups[0]")
    # The canonical manifest stores betas as JSON lists; restore the exact raw
    # tuple form expected by the shared option validator.
    raw_group = dict(group)
    betas = raw_group.get("betas")
    if not isinstance(betas, list):
        _fail("AdamW state manifest betas must be a list")
    raw_group["betas"] = tuple(betas)
    _optimizer_group_manifest(
        raw_group,
        optimizer_contract,
        parameter_count=len(V18_RESIDUAL_PARAMETER_SPECS),
    )

    states = manifest["states"]
    if not isinstance(states, list) or len(states) != len(V18_RESIDUAL_PARAMETER_SPECS):
        _fail("AdamW state manifest states length mismatch")
    tensor_hashes: list[str] = []
    for parameter_id, ((name, shape), state_value) in enumerate(
        zip(V18_RESIDUAL_PARAMETER_SPECS, states, strict=True)
    ):
        state = _mapping(state_value, f"AdamW state manifest states[{parameter_id}]")
        _strict_keys(
            state, {"parameter_id", "parameter_name", "state"}, f"manifest state {parameter_id}"
        )
        if state["parameter_id"] != parameter_id or state["parameter_name"] != name:
            _fail(f"AdamW state manifest state order mismatch at parameter {parameter_id}")
        entries = _mapping(state["state"], f"manifest state {parameter_id}.state")
        _strict_keys(entries, _STATE_KEYS, f"manifest state {parameter_id}.state")
        for state_name, expected_shape in (("step", ()), ("exp_avg", shape), ("exp_avg_sq", shape)):
            entry = _mapping(entries[state_name], f"manifest state {parameter_id}.{state_name}")
            expected_keys = {"shape", "dtype", "finite", "sha256"}
            if state_name == "step":
                expected_keys.add("value")
            _strict_keys(entry, expected_keys, f"manifest state {parameter_id}.{state_name}")
            if entry["shape"] != list(expected_shape) or entry["dtype"] != "float32":
                _fail(
                    f"AdamW state manifest tensor metadata mismatch at {parameter_id}.{state_name}"
                )
            if entry["finite"] is not True:
                _fail(f"AdamW state manifest finiteness mismatch at {parameter_id}.{state_name}")
            digest = entry["sha256"]
            if not isinstance(digest, str) or _SHA256.fullmatch(digest) is None:
                _fail(f"AdamW state manifest digest invalid at {parameter_id}.{state_name}")
            tensor_hashes.append(digest)
            if state_name == "step" and entry["value"] != float(optimizer_contract["step_index"]):
                _fail(f"AdamW state manifest step mismatch at parameter {parameter_id}")
    combined = manifest["all_state_tensors_sha256"]
    if not isinstance(combined, str) or _SHA256.fullmatch(combined) is None:
        _fail("AdamW state manifest combined tensor digest is invalid")
    if len(tensor_hashes) != 3 * len(V18_RESIDUAL_PARAMETER_SPECS):
        _fail("AdamW state manifest tensor count mismatch")
    return _canonical_sha256(dict(manifest))


__all__ = [
    "V18_ADAMW_STATE_SCHEMA_VERSION",
    "V18_RESIDUAL_OPTIMIZER_GROUP_NAME",
    "V18_RESIDUAL_PARAMETER_SPECS",
    "V18AdamWStateViolation",
    "canonical_v18_adamw_state",
    "validate_v18_adamw_state_manifest",
]
