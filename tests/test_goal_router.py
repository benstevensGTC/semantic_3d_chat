from __future__ import annotations

import pytest

from semantic_3d_chat.robot.goal_router import parse_semantic_goal


@pytest.mark.parametrize(
    "text",
    (
        "do a lap around the room",
        "Please patrol the environment.",
        "circle the room",
        "explore the space",
        "make a circuit of the room",
    ),
)
def test_room_scale_outcomes_route_to_lap(text: str) -> None:
    goal = parse_semantic_goal(text)

    assert goal is not None
    assert goal.kind == "lap"
    assert goal.targets == ()


@pytest.mark.parametrize(
    ("text", "kind", "targets"),
    (
        ("go to the thing someone could sit on", "approach", ("thing someone could sit on",)),
        ("move close to the bowl", "approach", ("bowl",)),
        ("park beside the red object", "approach", ("red object",)),
        ("orient toward whatever is hanging on the wall", "face", ("whatever is hanging on the wall",)),
        ("stand between the nearest seat and the table", "between", ("nearest seat", "table")),
    ),
)
def test_targets_are_copied_from_user_without_an_inventory(
    text: str,
    kind: str,
    targets: tuple[str, ...],
) -> None:
    goal = parse_semantic_goal(text)

    assert goal is not None
    assert goal.kind == kind
    assert goal.targets == targets
    assert len(goal.request_sha256) == 64


@pytest.mark.parametrize(
    "text",
    (
        "turn left 30 degrees",
        "move forward three meters",
        "what is in this room?",
        "what is beside the table?",
        "hello",
    ),
)
def test_low_level_motor_commands_and_dialogue_are_not_semantic_goals(text: str) -> None:
    assert parse_semantic_goal(text) is None
