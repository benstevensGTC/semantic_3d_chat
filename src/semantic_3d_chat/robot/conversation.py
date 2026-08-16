"""Conversational embodied-camera policy over continuous semantic geometry.

The parser sees only the user's instruction. Environmental grounding reads the
complete numeric voxel map and matching local text embeddings. It never accepts
an object inventory, label, caption, scene graph, segmentation, or oracle path.
Bounded actions execute through the refreshing runtime so every arrival scan is
fused before the next chat turn.
"""

from __future__ import annotations

import hashlib
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Protocol

import numpy as np

from semantic_3d_chat.evaluation.gemma4_semantic_sanity import (
    GEMMA4_PROJECTED_DIM,
    GEMMA4_PROJECTED_START,
)
from semantic_3d_chat.robot.llm_tool_policy import (
    LocalGemmaToolPolicy,
    execute_validated_tool_call,
)
from semantic_3d_chat.robot.semantic_agent import (
    ContinuousSemanticGrounding,
    ContinuousSemanticTargetGrounder,
    ContinuousTextEncoder,
    LabelFreeSemanticNavigator,
)

CommandKind = Literal[
    "approach",
    "face",
    "between",
    "turn",
    "move_forward",
    "move_backward",
    "scan",
    "stop",
]


class RefreshingConversationRuntime(Protocol):
    simulator: Any
    map_updater: Any

    def turn(self, angle_degrees: float) -> dict[str, Any]: ...

    def move_forward(self, distance_meters: float) -> dict[str, Any]: ...

    def move_backward(self, distance_meters: float) -> dict[str, Any]: ...

    def move_to(self, x: float, y: float) -> dict[str, Any]: ...

    def scan(self) -> dict[str, Any]: ...

    def stop(self) -> dict[str, Any]: ...

    def answer(self, question: str) -> Any: ...

    def prefix_binding(self) -> dict[str, Any]: ...


@dataclass(frozen=True)
class NavigationCommand:
    kind: CommandKind
    targets: tuple[str, ...] = ()
    value: float | None = None


_SPACE = re.compile(r"\s+")
_TRAILING = re.compile(r"[.!?\s]+$")
_ARTICLE = re.compile(r"^(?:the|a|an)\s+", re.IGNORECASE)


def _target(value: str) -> str:
    result = _TRAILING.sub("", _SPACE.sub(" ", value.strip()))
    result = _ARTICLE.sub("", result).strip()
    if not result:
        raise ValueError("Navigation target is empty")
    return result


def parse_navigation_instruction(text: str) -> NavigationCommand | None:
    """Parse a deliberately small, deterministic imperative action language."""

    if not isinstance(text, str) or not text.strip():
        raise ValueError("Conversation input must be non-empty text")
    normalized = _SPACE.sub(" ", text.strip())
    lower = normalized.casefold()

    if re.fullmatch(r"stop[.!]?", lower):
        return NavigationCommand("stop")
    if re.fullmatch(r"(?:scan|scan the room|look around)[.!]?", lower):
        return NavigationCommand("scan")

    numeric = re.fullmatch(
        r"turn\s+(?:(left|right)\s+)?(-?\d+(?:\.\d+)?)\s*(?:degrees?|deg)?[.!]?",
        lower,
    )
    if numeric:
        value = float(numeric.group(2))
        if numeric.group(1) == "left":
            value = -abs(value)
        elif numeric.group(1) == "right":
            value = abs(value)
        return NavigationCommand("turn", value=value)

    numeric = re.fullmatch(
        r"move\s+(forward|backward)\s+(-?\d+(?:\.\d+)?)\s*(?:meters?|m)?[.!]?",
        lower,
    )
    if numeric:
        kind: CommandKind = (
            "move_forward" if numeric.group(1) == "forward" else "move_backward"
        )
        return NavigationCommand(kind, value=float(numeric.group(2)))

    between = re.fullmatch(
        r"(?:stop|move|go)\s+between\s+(?:the\s+)?(.+?)\s+and\s+(?:the\s+)?(.+?)[.!]?",
        normalized,
        flags=re.IGNORECASE,
    )
    if between:
        return NavigationCommand(
            "between",
            targets=(_target(between.group(1)), _target(between.group(2))),
        )

    around = re.fullmatch(
        r"go\s+around\s+.+?\s+and\s+stop\s+(?:beside|near)\s+(?:the\s+)?(.+?)[.!]?",
        normalized,
        flags=re.IGNORECASE,
    )
    if around:
        return NavigationCommand("approach", targets=(_target(around.group(1)),))

    until_ahead = re.fullmatch(
        r"move\s+until\s+(?:the\s+)?(.+?)\s+is\s+directly\s+ahead[.!]?",
        normalized,
        flags=re.IGNORECASE,
    )
    if until_ahead:
        return NavigationCommand("approach", targets=(_target(until_ahead.group(1)),))

    approach = re.fullmatch(
        r"(?:move\s+closer\s+to|walk\s+toward|move\s+toward|go\s+to|approach)\s+"
        r"(?:the\s+)?(.+?)[.!]?",
        normalized,
        flags=re.IGNORECASE,
    )
    if approach:
        return NavigationCommand("approach", targets=(_target(approach.group(1)),))

    face = re.fullmatch(
        r"(?:turn\s+toward|turn\s+to\s+face|face|look\s+at)\s+(?:the\s+)?(.+?)[.!]?",
        normalized,
        flags=re.IGNORECASE,
    )
    if face:
        return NavigationCommand("face", targets=(_target(face.group(1)),))
    return None


def should_offer_llm_tool_policy(text: str) -> bool:
    """Conservatively identify turns that explicitly request robot action."""

    if not isinstance(text, str) or not text.strip():
        raise ValueError("Conversation input must be non-empty text")
    normalized = _SPACE.sub(" ", text.strip()).casefold()
    return bool(
        re.match(
            r"^(?:approach|face|get\s+(?:the\s+)?robot\s+state|go|look|move|reset|scan|stop|turn|walk)\b",
            normalized,
        )
    )


def _request_sha256(text: str) -> str:
    return hashlib.sha256(text.strip().encode("utf-8")).hexdigest()


def _grounding_payload(value: ContinuousSemanticGrounding) -> dict[str, object]:
    return value.as_dict()


class ConversationalEmbodiedAgent:
    """Route natural instructions or questions through one refreshed runtime."""

    def __init__(
        self,
        runtime: RefreshingConversationRuntime,
        text_encoder: ContinuousTextEncoder,
        *,
        room_size_m: tuple[float, float, float] | list[float],
        feature_start: int = GEMMA4_PROJECTED_START,
        feature_dim: int = GEMMA4_PROJECTED_DIM,
        tool_policy: LocalGemmaToolPolicy | None = None,
    ) -> None:
        self.runtime = runtime
        self.text_encoder = text_encoder
        self.room_size_m = tuple(float(value) for value in room_size_m)
        if len(self.room_size_m) != 3 or any(value <= 0 for value in self.room_size_m):
            raise ValueError("room_size_m must contain three positive values")
        self.feature_start = int(feature_start)
        self.feature_dim = int(feature_dim)
        self.tool_policy = tool_policy

    def _active_map_path(self) -> Path:
        updater = self.runtime.map_updater
        persistent = Path(updater.persistent_map_path)
        return persistent if persistent.is_file() else Path(updater.base_map_path)

    def _grounder(self) -> ContinuousSemanticTargetGrounder:
        return ContinuousSemanticTargetGrounder(
            self._active_map_path(),
            self.text_encoder,
            room_size_m=self.room_size_m,
            feature_start=self.feature_start,
            feature_dim=self.feature_dim,
        )

    def _face_xy(self, target_xy: np.ndarray) -> tuple[list[dict[str, Any]], bool]:
        simulator = self.runtime.simulator
        delta = target_xy - simulator.state.position_xy_m
        if float(np.linalg.norm(delta)) <= 1e-8:
            return [], True
        desired = math.degrees(math.atan2(-float(delta[0]), float(delta[1])))
        remaining = (desired - simulator.state.body_yaw_degrees + 180.0) % 360.0 - 180.0
        maximum = float(simulator.settings["max_turn_degrees"])
        receipts: list[dict[str, Any]] = []
        while abs(remaining) > 1e-7:
            step = max(-maximum, min(maximum, remaining))
            receipt = self.runtime.turn(step)
            receipts.append(receipt)
            if not receipt.get("success"):
                return receipts, False
            remaining -= step
        return receipts, True

    def _move_direct(self, target_xy: np.ndarray) -> tuple[list[dict[str, Any]], bool]:
        simulator = self.runtime.simulator
        start = simulator.state.position_xy_m.astype(np.float64)
        delta = target_xy.astype(np.float64) - start
        distance = float(np.linalg.norm(delta))
        maximum = float(simulator.settings["max_move_to_m"])
        pieces = max(1, math.ceil(distance / maximum))
        receipts: list[dict[str, Any]] = []
        for step in range(1, pieces + 1):
            point = start + delta * (step / pieces)
            receipt = self.runtime.move_to(float(point[0]), float(point[1]))
            receipts.append(receipt)
            if not receipt.get("success"):
                return receipts, False
        return receipts, True

    def _result(
        self,
        text: str,
        command: NavigationCommand,
        *,
        success: bool,
        receipts: list[dict[str, Any]],
        groundings: list[ContinuousSemanticGrounding] | None = None,
        navigation: dict[str, object] | None = None,
    ) -> dict[str, Any]:
        return {
            "kind": "navigation",
            "command": command.kind,
            "success": bool(success),
            "request_sha256": _request_sha256(text),
            "target_count": len(command.targets),
            "groundings": [
                _grounding_payload(value) for value in (groundings or [])
            ],
            "navigation": navigation,
            "action_receipts": receipts,
            "prefix_binding": self.runtime.prefix_binding(),
            "environmental_text_inputs": [],
            "question_dependent_scene_retrieval": False,
        }

    def _handle_parsed(
        self,
        text: str,
        command: NavigationCommand | None,
    ) -> dict[str, Any]:
        if command is None:
            answer = self.runtime.answer(text)
            return {
                "kind": "answer",
                "request_sha256": _request_sha256(text),
                "answer": answer.answer,
                "grounding_xyz_m": list(answer.grounding_xyz_m),
                "grounding_confidence": answer.grounding_confidence,
                "prefix_hash": answer.prefix_hash,
                "prefix_binding": self.runtime.prefix_binding(),
                "environmental_text_inputs": [],
            }

        if command.kind in {"turn", "move_forward", "move_backward"}:
            assert command.value is not None
            method = getattr(self.runtime, command.kind)
            receipt = method(command.value)
            return self._result(
                text,
                command,
                success=bool(receipt.get("success")),
                receipts=[receipt],
            )
        if command.kind in {"scan", "stop"}:
            receipt = getattr(self.runtime, command.kind)()
            return self._result(
                text,
                command,
                success=bool(receipt.get("success")),
                receipts=[receipt],
            )

        grounder = self._grounder()
        if command.kind == "approach":
            navigation = LabelFreeSemanticNavigator(
                self.runtime.simulator,
                grounder,
                action_surface=self.runtime,
            ).navigate(command.targets[0], scan_on_arrival=True)
            return self._result(
                text,
                command,
                success=navigation.success,
                receipts=[],
                groundings=[navigation.grounding],
                navigation=navigation.as_dict(),
            )
        if command.kind == "face":
            grounding = grounder.ground(command.targets[0])
            receipts, success = self._face_xy(
                np.asarray(grounding.target_xyz_m[:2], dtype=np.float64)
            )
            if success:
                scan = self.runtime.scan()
                receipts.append(scan)
                success = bool(scan.get("success"))
            return self._result(
                text,
                command,
                success=success,
                receipts=receipts,
                groundings=[grounding],
            )
        if command.kind == "between":
            first = grounder.ground(command.targets[0])
            second = grounder.ground(command.targets[1])
            midpoint = (
                np.asarray(first.target_xyz_m[:2], dtype=np.float64)
                + np.asarray(second.target_xyz_m[:2], dtype=np.float64)
            ) / 2.0
            receipts, success = self._move_direct(midpoint)
            if success:
                scan = self.runtime.scan()
                receipts.append(scan)
                success = bool(scan.get("success"))
            return self._result(
                text,
                command,
                success=success,
                receipts=receipts,
                groundings=[first, second],
            )
        raise RuntimeError(f"Unhandled navigation command: {command.kind}")

    def handle(self, text: str) -> dict[str, Any]:
        command = parse_navigation_instruction(text)
        if self.tool_policy is None or not should_offer_llm_tool_policy(text):
            return self._handle_parsed(text, command)

        decision = self.tool_policy.select(text)
        audit = decision.audit_payload()
        if decision.call is not None:
            receipt = execute_validated_tool_call(
                self.runtime,
                decision.call,
                config=self.tool_policy.config,
            )
            return {
                "kind": "navigation",
                "command": decision.call.name,
                "success": bool(receipt.get("success")),
                "request_sha256": _request_sha256(text),
                "target_count": 0,
                "groundings": [],
                "navigation": None,
                "action_receipts": [receipt],
                "prefix_binding": self.runtime.prefix_binding(),
                "tool_selection": audit,
                "tool_selection_fallback_used": False,
                "environmental_text_inputs": [],
                "question_dependent_scene_retrieval": False,
            }

        if decision.fallback_policy == "deterministic_parser" and command is not None:
            result = self._handle_parsed(text, command)
            result["tool_selection"] = audit
            result["tool_selection_fallback_used"] = True
            return result

        return {
            "kind": "navigation",
            "command": "local_gemma_tool_policy",
            "success": False,
            "error_code": "E_TOOL_POLICY_REJECTED",
            "request_sha256": _request_sha256(text),
            "target_count": 0,
            "groundings": [],
            "navigation": None,
            "action_receipts": [],
            "prefix_binding": self.runtime.prefix_binding(),
            "tool_selection": audit,
            "tool_selection_fallback_used": False,
            "environmental_text_inputs": [],
            "question_dependent_scene_retrieval": False,
        }

    def answer_question(self, text: str) -> dict[str, Any]:
        """Answer through continuous memory without exposing any action parser.

        The production model-only rover calls this narrower entry point for a
        turn classified as a scene question.  It deliberately supplies
        ``command=None`` so a missed or adversarial phrasing can never reach a
        deterministic motion primitive through the legacy conversational
        parser.
        """

        if not isinstance(text, str) or not text.strip():
            raise ValueError("Conversation question must be non-empty text")
        return self._handle_parsed(text, None)


__all__ = [
    "ConversationalEmbodiedAgent",
    "NavigationCommand",
    "RefreshingConversationRuntime",
    "parse_navigation_instruction",
    "should_offer_llm_tool_policy",
]
