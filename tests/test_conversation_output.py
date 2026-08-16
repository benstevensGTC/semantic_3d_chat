from semantic_3d_chat.robot.conversation_output import render_startup, render_turn


def test_human_startup_exposes_only_numeric_binding_and_protocol_state() -> None:
    rendered = render_startup(
        {
            "scene_id": "scene_000001",
            "prefix_binding": {
                "scene_prefix_sha256": "a" * 64,
                "source_voxels": 74699,
            },
            "llm_tool_policy": {"backend": "learned_navigation_v3"},
            "navigation_policy": {"task_trained": True},
        }
    )
    assert "scene_000001" in rendered
    assert "aaaaaaaaaaaa" in rendered
    assert "74699" in rendered
    assert "environmental text/oracle inputs: none" in rendered
    assert "chair" not in rendered.casefold()


def test_human_navigation_summary_is_short_and_numeric() -> None:
    rendered = render_turn(
        {
            "kind": "learned_navigation_closed_loop",
            "success": True,
            "termination_reason": "stop",
            "steps": [
                {
                    "command": "turn",
                    "success": True,
                    "action_receipts": [
                        {
                            "body_yaw_degrees": 64.358,
                            "position_m": [0.0, 0.0, 0.0],
                            "collision": False,
                        }
                    ],
                },
                {
                    "command": "stop",
                    "success": True,
                    "action_receipts": [{"position_m": [0.0, 0.0, 0.0]}],
                },
            ],
            "prefix_binding": {
                "position_m": [0.0, 0.0, 0.0],
                "active_prefix_sha256": "b" * 64,
            },
            "continuous_grounding_attestations": [{}, {}],
        }
    )
    assert "Robot action sequence: completed" in rendered
    assert "yaw=64.36°" in rendered
    assert "collision" not in rendered
    assert "bbbbbbbbbbbb" in rendered
    assert "final position: [0.00, 0.00] m" in rendered
    assert len(rendered) < 500


def test_human_answer_preserves_answer_grounding_and_prefix_hash() -> None:
    rendered = render_turn(
        {
            "kind": "answer",
            "answer": "left",
            "grounding_xyz_m": [1.0, 2.0, 0.5],
            "grounding_confidence": 0.75,
            "prefix_hash": "c" * 64,
        }
    )
    assert "Assistant> left" in rendered
    assert "[1.0, 2.0, 0.5]" in rendered
    assert "0.750" in rendered
    assert "cccccccccccc" in rendered
