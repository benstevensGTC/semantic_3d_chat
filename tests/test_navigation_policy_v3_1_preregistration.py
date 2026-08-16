from __future__ import annotations

import json

from semantic_3d_chat.evaluation.navigation_policy_v3_1_preregistration import (
    ACCEPTANCE_GATES,
    RUNTIME_VERSION,
    authenticate_preregistration,
    build_preregistration,
)


def test_v3_1_preregistration_freezes_narrow_runtime_change_and_gates(tmp_path) -> None:
    payload = build_preregistration()
    destination = tmp_path / "preregistration.json"
    destination.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )

    authenticated = authenticate_preregistration(destination)

    assert authenticated["runtime_version"] == RUNTIME_VERSION == "v3.1"
    assert authenticated["acceptance_gates"] == ACCEPTANCE_GATES
    assert authenticated["parent_result"]["success_count"] == 5
    assert authenticated["parent_result"]["failed_task_id"] == "nav_005"
    assert authenticated["authorized_change"]["learned_checkpoint_changed"] is False
    assert authenticated["authorized_change"]["collision_interlock_changed"] is False
    assert authenticated["authorized_change"]["environmental_text_inputs"] == []
    assert authenticated["authorized_change"]["oracle_inputs_at_runtime"] is False
    assert authenticated["benchmark_rerun_completed"] is False
    assert authenticated["runtime_promotion_authorized"] is False
