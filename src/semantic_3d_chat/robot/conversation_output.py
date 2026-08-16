"""Small human-facing renderer for embodied conversation results.

The machine-readable JSON surface remains the canonical audit protocol.  This
module deliberately renders only answers, numeric pose/action state, hashes,
and protocol status; it never invents or decodes an environmental description.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any


def _short_hash(value: object) -> str:
    return str(value)[:12] if value else "unavailable"


def _number(value: object, *, digits: int = 2) -> str:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return "?"
    return f"{float(value):.{digits}f}"


def _position(receipt: Mapping[str, Any]) -> str:
    value = receipt.get("position_m")
    if not isinstance(value, list) or len(value) < 2:
        return "[?, ?]"
    return f"[{_number(value[0])}, {_number(value[1])}] m"


def _action_line(index: int, step: Mapping[str, Any]) -> str:
    command = str(step.get("command", "unknown"))
    receipts = step.get("action_receipts")
    receipt: Mapping[str, Any] = {}
    if isinstance(receipts, list) and receipts and isinstance(receipts[-1], Mapping):
        receipt = receipts[-1]
    details: list[str] = []
    if command in {"turn", "look"}:
        details.append(f"yaw={_number(receipt.get('body_yaw_degrees'))}°")
    if command.startswith("move"):
        details.append(f"position={_position(receipt)}")
        details.append(f"moved={_number(receipt.get('distance_moved'))} m")
    if command == "scan":
        details.append(f"map_version={receipt.get('map_version', '?')}")
        details.append(f"valid_depth={receipt.get('valid_depth_pixels', '?')}")
    if receipt.get("collision") is True:
        details.append("collision=true")
    status = "ok" if step.get("success") is True else "failed"
    suffix = f" ({', '.join(details)})" if details else ""
    return f"  {index}. {command}: {status}{suffix}"


def render_startup(payload: Mapping[str, Any]) -> str:
    """Render an authenticated startup record without environmental prose."""

    binding = payload.get("prefix_binding")
    binding = binding if isinstance(binding, Mapping) else {}
    policy = payload.get("llm_tool_policy")
    policy = policy if isinstance(policy, Mapping) else {}
    navigation = payload.get("navigation_policy")
    navigation = navigation if isinstance(navigation, Mapping) else {}
    lines = [
        "Embodied Semantic 3D Chat is ready.",
        f"  scene: {payload.get('scene_id', 'unknown')}",
        f"  fixed scene prefix: {_short_hash(binding.get('scene_prefix_sha256'))}",
        f"  source voxels: {binding.get('source_voxels', '?')}",
        f"  action backend: {policy.get('backend', 'disabled')}",
        f"  task-trained controller: {bool(navigation.get('task_trained'))}",
        "  environmental text/oracle inputs: none",
    ]
    return "\n".join(lines)


def render_turn(payload: Mapping[str, Any]) -> str:
    """Render one answer or bounded action sequence for a person."""

    kind = payload.get("kind")
    if kind == "answer":
        xyz = payload.get("grounding_xyz_m")
        coordinate = json.dumps(xyz) if isinstance(xyz, list) else "unavailable"
        return "\n".join(
            [
                f"Assistant> {payload.get('answer', 'unknown')}",
                (
                    "  grounding: "
                    f"xyz={coordinate}, confidence="
                    f"{_number(payload.get('grounding_confidence'), digits=3)}"
                ),
                f"  fixed prefix: {_short_hash(payload.get('prefix_hash'))}",
            ]
        )

    if kind == "learned_navigation_closed_loop":
        steps = payload.get("steps")
        final_receipt: Mapping[str, Any] = {}
        rendered = [
            "Robot action sequence: "
            + ("completed" if payload.get("success") is True else "stopped without success"),
            f"  termination: {payload.get('termination_reason', 'unknown')}",
        ]
        if isinstance(steps, list):
            rendered.extend(
                _action_line(index, step)
                for index, step in enumerate(steps, start=1)
                if isinstance(step, Mapping)
            )
            for step in reversed(steps):
                if not isinstance(step, Mapping):
                    continue
                receipts = step.get("action_receipts")
                if isinstance(receipts, list) and receipts and isinstance(
                    receipts[-1], Mapping
                ):
                    final_receipt = receipts[-1]
                    break
        binding = payload.get("prefix_binding")
        if isinstance(binding, Mapping):
            rendered.append(f"  final position: {_position(final_receipt)}")
            rendered.append(
                "  refreshed active prefix: "
                f"{_short_hash(binding.get('active_prefix_sha256'))}"
            )
        rendered.append(
            "  continuous grounding: "
            f"{len(payload.get('continuous_grounding_attestations', []))} refreshed step(s)"
        )
        return "\n".join(rendered)

    if kind == "navigation":
        return "\n".join(
            [
                "Robot action: "
                + ("completed" if payload.get("success") is True else "failed"),
                _action_line(1, payload),
            ]
        )

    return json.dumps(dict(payload), sort_keys=True, allow_nan=False)


__all__ = ["render_startup", "render_turn"]
