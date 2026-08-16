"""Ask Gemma questions about the room it perceived, and let it drive.

Both capabilities share one idea: the model is given the *metric* scene it
reconstructed -- names it chose, coordinates it measured, floor it can occupy --
and reasons in that coordinate frame.  For questions it answers in prose; for
navigation it answers with numbers a deterministic executor can check.

Nothing here is trained.  The model's spatial competence comes from reading a
compact metric description of its own perception.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from typing import Any

from semantic_3d_chat.spatial_lens.scene_graph import SceneGraph

QA_SYSTEM = (
    "You are the spatial reasoning system of a small indoor robot. You are "
    "given a metric map of a room that you built yourself by looking at it. "
    "Answer questions about the room using the coordinates and sizes given. "
    "Reason about geometry: distances, directions, what is between what, what "
    "fits where. Be concise and concrete, and give numbers in metres when they "
    "are useful. If the map does not contain something, say so plainly rather "
    "than guessing."
)

NAV_SYSTEM = (
    "You are the navigation system of a small indoor robot. You are given a "
    "metric map of the room that the robot built by looking at it, the robot's "
    "pose, and a goal. You decide where the robot goes.\n"
    "\n"
    "Reply with ONE json object and nothing else:\n"
    '{"reasoning": "<one short sentence>", "action": "<ACTION>", '
    '"x": <metres>, "y": <metres>, "yaw_degrees": <degrees>}\n'
    "\n"
    "Actions:\n"
    '  MOVE_TOWARD  with "x","y": head for that point. The robot automatically '
    "advances only as far as it is allowed to in one move, so you may name a "
    "destination that is far away. Prefer this action.\n"
    '  MOVE_TO      with "x","y": go exactly there. Only legal within one step.\n'
    '  FACE         with "yaw_degrees": turn on the spot.\n'
    "  STOP         no arguments: the goal is achieved.\n"
    "\n"
    "Each object lists a 'stand at' point, which is free floor beside it. To "
    "reach an object, MOVE_TOWARD its stand-at point, not its centre -- the "
    "centre is inside the object and is not drivable. If a move is rejected as "
    "blocked, pick a different intermediate point that goes around the "
    "obstruction. Emit STOP only when the goal is actually achieved.\n"
    "\n"
    "The prompt lists every direction that is clear from the robot's current "
    "spot, with the exact position each one would reach. When the straight line "
    "to your target is blocked, do NOT keep retrying it. Instead read that list, "
    "work out which listed position ends up closest to your target, and reply "
    "with MOVE_TO and that exact position. Repeat until you arrive."
)

_JSON = re.compile(r"\{.*\}", re.DOTALL)


def answer_question(chat: Any, graph: SceneGraph, question: str, *, max_new_tokens: int = 220) -> str:
    """Answer one spatial question about the perceived room."""

    prompt = (
        f"{graph.describe()}\n\n"
        f"QUESTION\n  {question.strip()}\n\n"
        "Answer using the map above."
    )
    return chat.ask_text(prompt, system=QA_SYSTEM, max_new_tokens=max_new_tokens)


@dataclass(frozen=True)
class RoverPose:
    x_m: float
    y_m: float
    yaw_degrees: float

    def as_dict(self) -> dict[str, float]:
        return {
            "x_m": round(self.x_m, 4),
            "y_m": round(self.y_m, 4),
            "yaw_degrees": round(self.yaw_degrees, 3),
        }


@dataclass(frozen=True)
class NavDecision:
    action: str
    x_m: float | None
    y_m: float | None
    yaw_degrees: float | None
    reasoning: str
    raw: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "action": self.action,
            "x_m": self.x_m,
            "y_m": self.y_m,
            "yaw_degrees": self.yaw_degrees,
            "reasoning": self.reasoning,
        }


def heading_to(origin: tuple[float, float], target: tuple[float, float]) -> float:
    """Yaw that points from origin to target, in the project convention.

    Yaw 0 faces +Y and yaw -90 faces +X, matching the rover executor.
    """

    dx = target[0] - origin[0]
    dy = target[1] - origin[1]
    return math.degrees(math.atan2(-dx, dy))


def normalize_degrees(value: float) -> float:
    return math.degrees(math.atan2(math.sin(math.radians(value)), math.cos(math.radians(value))))


def parse_decision(text: str) -> NavDecision:
    """Read the model's JSON action, tolerating chatter around it."""

    match = _JSON.search(text or "")
    if match is None:
        raise ValueError(f"No JSON object in model reply: {text[:200]!r}")
    payload = json.loads(match.group(0))
    action = str(payload.get("action", "")).strip().upper()
    if action not in {"MOVE_TO", "MOVE_TOWARD", "FACE", "STOP"}:
        raise ValueError(f"Unsupported action: {action!r}")

    def number(key: str) -> float | None:
        value = payload.get(key)
        if value is None or isinstance(value, bool):
            return None
        if isinstance(value, (int, float)) and math.isfinite(float(value)):
            return float(value)
        return None

    return NavDecision(
        action=action,
        x_m=number("x"),
        y_m=number("y"),
        yaw_degrees=number("yaw_degrees"),
        reasoning=str(payload.get("reasoning", "")).strip()[:300],
        raw=text,
    )


def navigation_prompt(
    graph: SceneGraph,
    pose: RoverPose,
    goal: str,
    history: list[str],
    *,
    max_step_m: float,
    approach_points: dict[str, tuple[float, float]] | None = None,
    open_directions: list[tuple[float, tuple[float, float], bool]] | None = None,
) -> str:
    """Describe the room *relative to where the robot is standing right now*.

    A 2B model reliably fails to derive "how far away is the lamp, and which way
    is it" from two coordinate pairs.  Those quantities are measured here and
    handed over, so the model spends its reasoning on the decision -- which
    object, which side, how to get round the table -- rather than on trigonometry.
    """

    width, depth, _height = graph.room_size_m
    lines = [
        "ROOM",
        f"  {width:.2f} m (X) by {depth:.2f} m (Y), origin at the centre",
        "  yaw 0 faces +Y, yaw -90 faces +X, yaw +90 faces -X, yaw 180 faces -Y",
        "",
        "ROBOT",
        (
            f"  standing at ({pose.x_m:+.2f}, {pose.y_m:+.2f}), facing yaw "
            f"{pose.yaw_degrees:+.1f}"
        ),
        f"  one move covers at most {max_step_m:.2f} m",
        "",
        "OBJECTS, as seen from where the robot is standing",
    ]
    for item in sorted(
        graph.objects,
        key=lambda o: math.dist((pose.x_m, pose.y_m), o.center_m[:2]),
    ):
        target = (item.center_m[0], item.center_m[1])
        distance = math.dist((pose.x_m, pose.y_m), target)
        bearing = heading_to((pose.x_m, pose.y_m), target)
        turn = normalize_degrees(bearing - pose.yaw_degrees)
        entry = (
            f"  {item.name}: {distance:.2f} m away, bearing yaw {bearing:+.0f} "
            f"(turn {turn:+.0f}), centre ({target[0]:+.2f}, {target[1]:+.2f}), "
            f"occupies X [{item.bbox_min_m[0]:+.2f}, {item.bbox_max_m[0]:+.2f}] "
            f"Y [{item.bbox_min_m[1]:+.2f}, {item.bbox_max_m[1]:+.2f}]"
        )
        stand = (approach_points or {}).get(item.name)
        if stand is not None:
            entry += f", stand at ({stand[0]:+.2f}, {stand[1]:+.2f})"
        lines.append(entry)
    if open_directions:
        lines.extend(
            [
                "",
                "WHERE THE ROBOT CAN GO FROM HERE (one move, straight line)",
            ]
        )
        for label, point, clear in open_directions:
            if clear:
                lines.append(
                    f"  yaw {label:+4.0f}: clear, would reach "
                    f"({point[0]:+.2f}, {point[1]:+.2f})"
                )
            else:
                lines.append(f"  yaw {label:+4.0f}: blocked")
    lines.extend(["", f"GOAL\n  {goal.strip()}"])
    if history:
        lines.extend(["", "WHAT HAS HAPPENED SO FAR"])
        lines.extend(
            f"  {index + 1}. {item}" for index, item in enumerate(history[-6:])
        )
    lines.extend(["", "Choose the next action as one JSON object."])
    return "\n".join(lines)


__all__ = [
    "NAV_SYSTEM",
    "QA_SYSTEM",
    "NavDecision",
    "RoverPose",
    "answer_question",
    "heading_to",
    "navigation_prompt",
    "normalize_degrees",
    "parse_decision",
]
