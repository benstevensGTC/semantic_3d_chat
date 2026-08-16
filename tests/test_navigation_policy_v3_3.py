from __future__ import annotations

import json
import sys
from importlib import import_module
from types import SimpleNamespace

import numpy as np
import pytest
from test_navigation_policy_v3_2 import _DiagnosticTextEncoder, _write_numeric_map

from semantic_3d_chat.config import PROJECT_ROOT, load_config
from semantic_3d_chat.evaluation.llm_navigation_benchmark import (
    NavigationTask,
    _policy_instruction,
    file_sha256,
)
from semantic_3d_chat.robot.llm_tool_policy import GeneratedToolProposal
from semantic_3d_chat.robot.navigation_policy_v3 import (
    SemanticGroundedActionBackendV3,
    load_navigation_policy_v3_checkpoint,
)
from semantic_3d_chat.robot.navigation_policy_v3_3 import (
    SemanticGroundedActionBackendV33,
)
from semantic_3d_chat.robot.semantic_mapping import semantic_map_content_hash


def test_exact_live_runner_routes_enveloped_sequence_through_v3_3_planner(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Run the wrapper-installed class and real V3 controller construction."""

    scripts = PROJECT_ROOT / "scripts"
    monkeypatch.syspath_prepend(str(scripts))
    sys.modules.pop("run_llm_navigation_inference_v3_3", None)
    wrapper = import_module("run_llm_navigation_inference_v3_3")
    # Register the original base-module binding for pytest teardown before the
    # wrapper performs the same direct assignment used by its production main.
    monkeypatch.setattr(
        wrapper.base,
        "SemanticGroundedActionBackendV3",
        wrapper.base.SemanticGroundedActionBackendV3,
    )
    monkeypatch.setattr(wrapper.base, "_run_contract", wrapper.base._run_contract)
    routed_class = wrapper.install_v3_3_routing()
    assert routed_class is SemanticGroundedActionBackendV33
    assert wrapper.base.SemanticGroundedActionBackendV3 is routed_class

    monkeypatch.setattr(
        wrapper,
        "_ORIGINAL_RUN_CONTRACT",
        lambda **_kwargs: {"inherited_contract": True},
    )
    contract = wrapper._v3_3_run_contract(navigation_policy_version=3)
    assert contract["inherited_contract"] is True
    assert contract["navigation_runtime_interlock_version"] == "v3.3"
    assert contract["compound_scan_approach_numeric_planner"] is True
    assert contract["live_protocol_envelope_unwrapped_before_action_grammar"] is True
    assert contract["inference_source_sha256"] == file_sha256(
        PROJECT_ROOT / "scripts/run_llm_navigation_inference_v3_3.py"
    )
    assert contract["navigation_policy_source_sha256"] == file_sha256(
        PROJECT_ROOT / "src/semantic_3d_chat/robot/navigation_policy_v3_3.py"
    )
    assert contract["navigation_policy_parent_source_sha256"] == file_sha256(
        PROJECT_ROOT / "src/semantic_3d_chat/robot/navigation_policy_v3_2.py"
    )

    map_path = tmp_path / "maps" / "scene_314159" / "voxel_map.npz"
    _write_numeric_map(map_path)
    state = SimpleNamespace(
        position_xy_m=np.asarray([-1.4, 0.0], dtype=np.float64),
        scene_version=0,
        scan_count=0,
    )
    simulator = SimpleNamespace(state=state, collision_map=None)
    runtime = SimpleNamespace(
        simulator=simulator,
        map_updater=SimpleNamespace(
            persistent_map_path=tmp_path / "absent.npz",
            base_map_path=map_path,
        ),
        prefix_refresher=SimpleNamespace(
            runtime=SimpleNamespace(language=SimpleNamespace(hidden_size=1536))
        ),
    )
    controller, metadata = load_navigation_policy_v3_checkpoint(
        PROJECT_ROOT / "data_gemma4/checkpoints/navigation_policy_v3",
        expected_hidden_size=1536,
        expected_model_id="google/gemma-4-E2B-it",
        expected_model_revision="3e22461f65e89153144f8adb70e3b8c2cc9845a7",
        device="cpu",
    )
    config = load_config(PROJECT_ROOT / "configs/runtime/embodied_navigation_v2.yaml")
    backend = routed_class(
        runtime,
        controller,
        metadata,
        config,
        text_encoder=_DiagnosticTextEncoder(),
    )
    map_sha256 = semantic_map_content_hash(map_path)

    def diagnostic_parent_generate(self, instruction, *, correction_code):
        del instruction, correction_code
        self.last_grounding = {
            "target_available": True,
            "target_xyz_m": [1.4, 0.0, 0.8],
            "map_sha256": map_sha256,
            "scored_voxels": 22,
        }
        call = (
            {"tool": "scan", "arguments": {}}
            if self.runtime.simulator.state.scan_count < 1
            else {"tool": "stop", "arguments": {}}
        )
        return GeneratedToolProposal(
            text=json.dumps(call, sort_keys=True, separators=(",", ":")),
            active_prefix_sha256="a" * 64,
            scene_prefix_sha256="b" * 64,
            robot_tokens_sha256="c" * 64,
            local_inference=True,
            used_continuous_scene_prefix=True,
            used_continuous_robot_tokens=True,
            training_status=("supervised_continuous_semantic_grounded_navigation_policy_v3"),
        )

    monkeypatch.setattr(
        SemanticGroundedActionBackendV3,
        "generate",
        diagnostic_parent_generate,
    )
    task = NavigationTask(
        task_id="nav_314",
        family="update_after_scan",
        instruction=("Scan the room, then move closer to arbitrary-alpha and stop."),
        max_steps=12,
    )
    first_prompt = _policy_instruction(task, None)
    first = json.loads(backend.generate(first_prompt, correction_code=None).text)
    assert first["tool"] == "scan"
    state.scan_count = 1
    state.scene_version = 1
    prompt = _policy_instruction(task, {"scan_count": 1, "success": True})
    inherited = json.loads(diagnostic_parent_generate(backend, prompt, correction_code=None).text)
    second = json.loads(backend.generate(prompt, correction_code=None).text)
    assert inherited["tool"] == "stop"
    assert second["tool"] == "move_to"
    planner = backend.last_grounding["numeric_compound_approach_planner"]
    assert planner["runtime_interlock_version"] == "v3.3"
    assert planner["environmental_text_inputs"] == []
    assert planner["oracle_inputs_at_runtime"] is False

    sequence = ["scan", "move_to"]
    call = second
    for _ in range(11):
        if call["tool"] == "move_to":
            state.position_xy_m = np.asarray(
                [call["arguments"]["x"], call["arguments"]["y"]],
                dtype=np.float64,
            )
        call = json.loads(backend.generate(prompt, correction_code=None).text)
        sequence.append(call["tool"])
        if call["tool"] == "stop":
            break

    assert sequence[-1] == "stop"
    assert "move_to" in sequence[1:-1]
    assert sequence != ["scan", "stop"]
    completion = backend.last_grounding["numeric_compound_approach_planner"]
    assert completion["completion_satisfied"] is True
    assert completion["completion_mode"] == ("collision_capped_numeric_waypoint_standoff")
