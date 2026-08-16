"""Co-resident V96 continuous-prefix policy driving the official MCP surface.

The action policy and robot runtime intentionally share one process so the
policy can consume the actual 738+4 tensor, rather than trying to reconstruct
continuous memory from hashes returned over a subprocess transport.  Every
action is nevertheless dispatched through the public ``MCPServer.call_tool``
API with the same registered tools, Pydantic schemas, and numeric-only result
model used by the stdio server.  Independent tests cover the stdio transport.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Final, Protocol

from mcp.server.mcpserver import MCPServer

from semantic_3d_chat.language.prefix_injection import prefix_sha256
from semantic_3d_chat.mcp_server.server import build_server
from semantic_3d_chat.robot.llm_tool_policy import (
    GeneratedToolProposal,
    validate_tool_call_text,
)
from semantic_3d_chat.robot.mcp_stdio_runtime import validate_numeric_tool_receipt
from semantic_3d_chat.robot.tools import TOOL_ARGUMENTS
from semantic_3d_chat.robot.v96_release_action import (
    ACTIVE_TOKEN_COUNT,
    HIDDEN_SIZE,
    ROBOT_TOKEN_COUNT,
    V96_SCENE_TOKEN_COUNT,
)

TRANSPORT_MODE: Final[str] = "official_python_mcp_sdk_in_process_dispatch"
_AUTO_SCAN_ACTIONS: Final[frozenset[str]] = frozenset(
    {"look", "turn", "move_forward", "move_backward", "move_to"}
)


class V96ActionProposalBackend(Protocol):
    last_v96_context_audit: Mapping[str, Any] | None

    def generate(
        self,
        instruction: str,
        *,
        correction_code: str | None,
    ) -> GeneratedToolProposal: ...


@dataclass(frozen=True)
class V96MCPActionStep:
    """One context-bound proposal, MCP dispatch, and numeric receipt."""

    index: int
    instruction_sha256: str
    call: dict[str, Any]
    call_sha256: str
    proposal_sha256: str
    before_receipt: dict[str, Any]
    before_binding: dict[str, Any]
    after_binding: dict[str, Any]
    receipt: dict[str, Any]
    policy_context_audit: dict[str, Any]
    rgbd_observation_expected: bool
    map_refresh_verified_before_next_decision: bool
    transport: str = TRANSPORT_MODE

    def as_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "instruction_sha256": self.instruction_sha256,
            "call": dict(self.call),
            "call_sha256": self.call_sha256,
            "proposal_sha256": self.proposal_sha256,
            "before_receipt": dict(self.before_receipt),
            "before_binding": dict(self.before_binding),
            "after_binding": dict(self.after_binding),
            "receipt": dict(self.receipt),
            "policy_context_audit": dict(self.policy_context_audit),
            "rgbd_observation_expected": self.rgbd_observation_expected,
            "map_refresh_verified_before_next_decision": (
                self.map_refresh_verified_before_next_decision
            ),
            "transport": self.transport,
            "numeric_tool_output_only": True,
            "environmental_text_inputs": [],
            "oracle_inputs_at_runtime": False,
        }


@dataclass(frozen=True)
class V96MCPNavigationResult:
    instruction_sha256: str
    termination_reason: str
    steps: tuple[V96MCPActionStep, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": "semantic_3d_chat.v96_co_resident_mcp_navigation.v1",
            "instruction_sha256": self.instruction_sha256,
            "termination_reason": self.termination_reason,
            "step_count": len(self.steps),
            "steps": [step.as_dict() for step in self.steps],
            "transport": TRANSPORT_MODE,
            "policy_consumed_738_scene_tokens_every_decision": all(
                step.policy_context_audit.get(
                    "policy_consumed_738_scene_tokens"
                )
                is True
                for step in self.steps
            ),
            "policy_consumed_4_robot_tokens_every_decision": all(
                step.policy_context_audit.get("policy_consumed_4_robot_tokens")
                is True
                for step in self.steps
            ),
            "numeric_tool_outputs_only": True,
            "successful_rgbd_refreshes_verified_before_next_decision": all(
                step.receipt.get("success") is not True
                or not step.rgbd_observation_expected
                or step.map_refresh_verified_before_next_decision
                for step in self.steps
            ),
            "environmental_text_inputs": [],
            "oracle_inputs_at_runtime": False,
            "held_out_navigation_claim": False,
        }


def _sha256_json(value: object) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _binding_subset(receipt: Mapping[str, Any]) -> dict[str, Any]:
    fields = (
        "schema",
        "scene_id",
        "scene_version",
        "map_version",
        "map_sha256",
        "scene_prefix_sha256",
        "scene_control_signature_sha256",
        "active_prefix_sha256",
        "robot_state_sha256",
        "robot_tokens_sha256",
        "active_binding_sha256",
    )
    return {field: receipt[field] for field in fields}


def _verify_successful_rgbd_refresh(
    before: Mapping[str, Any],
    after: Mapping[str, Any],
) -> None:
    """Fail closed unless one successful capture changed map and scene memory."""

    if (
        after.get("scene_id") != before.get("scene_id")
        or int(after["map_version"]) != int(before["map_version"]) + 1
        or after.get("scene_version") != after.get("map_version")
        or int(after["scan_count"]) != int(before["scan_count"]) + 1
        or after.get("map_sha256") == before.get("map_sha256")
        or after.get("scene_control_signature_sha256")
        == before.get("scene_control_signature_sha256")
        or after.get("active_prefix_sha256") == before.get("active_prefix_sha256")
        or not isinstance(after.get("observation_id"), str)
        or int(after["valid_depth_pixels"]) < 1
        or int(after["visible_voxels"]) < 1
    ):
        raise RuntimeError(
            "Successful RGB-D action did not refresh map/version and complete scene memory"
        )


def _validate_policy_context(
    audit: object,
    *,
    proposal: GeneratedToolProposal,
    before: Mapping[str, Any],
) -> dict[str, Any]:
    if not isinstance(audit, Mapping):
        raise TypeError("V96 action backend did not expose a context audit")
    value = dict(audit)
    if (
        value.get("active_prefix_shape") != [1, ACTIVE_TOKEN_COUNT, HIDDEN_SIZE]
        or value.get("scene_tokens_consumed") != V96_SCENE_TOKEN_COUNT
        or value.get("robot_tokens_consumed") != ROBOT_TOKEN_COUNT
        or value.get("policy_consumed_738_scene_tokens") is not True
        or value.get("policy_consumed_4_robot_tokens") is not True
        or value.get("complete_scene_memory_used") is not True
        or value.get("question_dependent_scene_retrieval") is not False
        or value.get("source_policy_was_retrained_on_v96") is not False
        or value.get("environmental_text_inputs") != []
        or value.get("oracle_inputs_at_runtime") is not False
        or value.get("active_prefix_sha256") != before.get("active_prefix_sha256")
        or value.get("full_scene_memory_sha256")
        != before.get("scene_control_signature_sha256")
        or value.get("robot_tokens_sha256") != before.get("robot_tokens_sha256")
        or proposal.active_prefix_sha256 != before.get("active_prefix_sha256")
        or proposal.scene_prefix_sha256 != before.get("scene_prefix_sha256")
        or proposal.robot_tokens_sha256 != before.get("robot_tokens_sha256")
        or proposal.local_inference is not True
        or proposal.used_continuous_scene_prefix is not True
        or proposal.used_continuous_robot_tokens is not True
    ):
        raise RuntimeError("V96 proposal is not bound to the complete active context")
    return value


class V96CoResidentMCPAgent:
    """Select bounded actions from 738+4 tensors and execute via MCP."""

    def __init__(
        self,
        runtime: Any,
        backend: V96ActionProposalBackend,
        config: Mapping[str, Any],
        *,
        server: MCPServer[None] | None = None,
    ) -> None:
        if not callable(getattr(runtime, "active_prefix_snapshot", None)):
            raise TypeError("Co-resident V96 runtime lacks an active-prefix snapshot")
        if not callable(getattr(backend, "generate", None)):
            raise TypeError("Co-resident V96 action backend is invalid")
        self.runtime = runtime
        self.backend = backend
        self.config = dict(config)
        robot_config = self.config.get("robot")
        if not isinstance(robot_config, Mapping):
            raise TypeError("Co-resident V96 config lacks robot settings")
        auto_scan_after_motion = robot_config.get("auto_scan_after_motion", False)
        if not isinstance(auto_scan_after_motion, bool):
            raise TypeError("robot.auto_scan_after_motion must be boolean")
        self.auto_scan_after_motion = auto_scan_after_motion
        self.server = server or build_server(runtime)
        self.steps: list[V96MCPActionStep] = []
        prefix, binding = runtime.active_prefix_snapshot()
        if (
            tuple(prefix.shape) != (1, ACTIVE_TOKEN_COUNT, HIDDEN_SIZE)
            or prefix_sha256(prefix) != binding.get("active_prefix_sha256")
            or not isinstance(binding.get("scene_control_signature_sha256"), str)
            or not isinstance(binding.get("robot_tokens_sha256"), str)
        ):
            raise ValueError("Co-resident V96 runtime lacks the exact 738+4 binding")

    async def _require_tool_inventory(self) -> None:
        tools = await self.server.list_tools()
        names = {tool.name for tool in tools}
        if names != set(TOOL_ARGUMENTS) or len(tools) != 9:
            raise RuntimeError("Co-resident MCP tool inventory changed")

    async def _call_mcp(
        self,
        name: str,
        arguments: Mapping[str, Any],
    ) -> dict[str, Any]:
        result = await self.server.call_tool(name, dict(arguments))
        if getattr(result, "is_error", False):
            raise RuntimeError(f"MCP tool dispatch failed: {name}")
        structured = getattr(result, "structured_content", None)
        if not isinstance(structured, Mapping):
            raise TypeError("MCP tool did not return structured numeric content")
        receipt = dict(structured)
        validate_numeric_tool_receipt(receipt, require_continuous_binding=True)
        return receipt

    async def step(self, instruction: str) -> V96MCPActionStep:
        if not isinstance(instruction, str) or not instruction.strip():
            raise ValueError("Navigation instruction must be non-empty text")
        if len(instruction) > 4096:
            raise ValueError("Navigation instruction is too long")
        await self._require_tool_inventory()
        before = await self._call_mcp("get_robot_state", {})
        proposal = self.backend.generate(
            instruction.strip(),
            correction_code=None,
        )
        if not isinstance(proposal, GeneratedToolProposal):
            raise TypeError("V96 action backend returned an invalid proposal")
        policy_audit = _validate_policy_context(
            self.backend.last_v96_context_audit,
            proposal=proposal,
            before=before,
        )
        validation = validate_tool_call_text(
            proposal.text,
            self.config,
            robot_state=before,
        )
        if validation.call is None or validation.error_code is not None:
            raise RuntimeError(
                f"V96 policy proposal failed strict validation: "
                f"{validation.error_code or 'E_SCHEMA'}"
            )
        call = validation.call
        canonical_proposal = json.dumps(
            call.as_dict(),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
        if proposal.text != canonical_proposal:
            raise RuntimeError(
                "V96 policy proposal must be the exact canonical numeric tool call"
            )
        receipt = await self._call_mcp(call.name, call.arguments)
        rgbd_observation_expected = call.name == "scan" or (
            self.auto_scan_after_motion and call.name in _AUTO_SCAN_ACTIONS
        )
        map_refresh_verified = False
        if receipt.get("success") is True and rgbd_observation_expected:
            _verify_successful_rgbd_refresh(before, receipt)
            map_refresh_verified = True
        after_prefix, after_binding = self.runtime.active_prefix_snapshot()
        if (
            prefix_sha256(after_prefix) != receipt.get("active_prefix_sha256")
            or receipt.get("active_prefix_sha256")
            != after_binding.get("active_prefix_sha256")
            or receipt.get("scene_control_signature_sha256")
            != after_binding.get("scene_control_signature_sha256")
            or receipt.get("robot_tokens_sha256")
            != after_binding.get("robot_tokens_sha256")
        ):
            raise RuntimeError("MCP receipt differs from the post-action continuous context")
        index = len(self.steps)
        step = V96MCPActionStep(
            index=index,
            instruction_sha256=hashlib.sha256(
                instruction.strip().encode("utf-8")
            ).hexdigest(),
            call=call.as_dict(),
            call_sha256=call.call_sha256,
            proposal_sha256=hashlib.sha256(
                canonical_proposal.encode("utf-8")
            ).hexdigest(),
            before_receipt=before,
            before_binding=_binding_subset(before),
            after_binding=_binding_subset(receipt),
            receipt=receipt,
            policy_context_audit=policy_audit,
            rgbd_observation_expected=rgbd_observation_expected,
            map_refresh_verified_before_next_decision=map_refresh_verified,
        )
        self.steps.append(step)
        return step

    async def run_instruction(
        self,
        instruction: str,
        *,
        max_steps: int = 24,
    ) -> V96MCPNavigationResult:
        if isinstance(max_steps, bool) or not isinstance(max_steps, int):
            raise TypeError("max_steps must be an integer")
        if not 1 <= max_steps <= 128:
            raise ValueError("max_steps must be in [1,128]")
        instruction_hash = hashlib.sha256(instruction.strip().encode("utf-8")).hexdigest()
        start = len(self.steps)
        termination = "max_steps"
        for _ in range(max_steps):
            step = await self.step(instruction)
            if step.receipt.get("success") is not True:
                termination = "action_rejected"
                break
            if step.call.get("tool") == "stop" or step.receipt.get("stopped") is True:
                termination = "stop"
                break
        selected: Sequence[V96MCPActionStep] = self.steps[start:]
        return V96MCPNavigationResult(
            instruction_sha256=instruction_hash,
            termination_reason=termination,
            steps=tuple(selected),
        )


__all__ = [
    "TRANSPORT_MODE",
    "V96ActionProposalBackend",
    "V96CoResidentMCPAgent",
    "V96MCPActionStep",
    "V96MCPNavigationResult",
]
