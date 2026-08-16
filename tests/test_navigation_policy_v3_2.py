from __future__ import annotations

import inspect
import json
import sys
from importlib import import_module
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from semantic_3d_chat.config import PROJECT_ROOT, load_config
from semantic_3d_chat.evaluation.llm_navigation_benchmark import (
    NavigationTask,
    _policy_instruction,
    file_sha256,
)
from semantic_3d_chat.mapping.voxel_map import SparseVoxelMap
from semantic_3d_chat.robot.llm_tool_policy import GeneratedToolProposal
from semantic_3d_chat.robot.navigation_policy_v3 import (
    SemanticGroundedActionBackendV3,
    load_navigation_policy_v3_checkpoint,
)
from semantic_3d_chat.robot.navigation_policy_v3_2 import (
    RUNTIME_INTERLOCK_VERSION,
    SemanticGroundedActionBackendV32,
    is_compound_scan_approach_instruction,
    literal_navigation_instruction,
)
from semantic_3d_chat.robot.semantic_mapping import semantic_map_content_hash


@pytest.mark.parametrize(
    "instruction",
    [
        "Scan, then approach arbitrary-alpha and stop.",
        "Scan the room, then move closer to opaque 917, then stop.",
        "Look around, then walk toward any target phrase and stop.",
        "Look around then move toward q-z-4, then stop.",
    ],
)
def test_v3_2_compound_grammar_accepts_arbitrary_target_phrases(
    instruction: str,
) -> None:
    assert is_compound_scan_approach_instruction(instruction) is True


def test_v3_2_unwraps_exact_live_policy_envelope_and_ignores_numeric_receipt() -> None:
    literal = "Scan the room, then move closer to opaque 917 and stop."
    prompt = (
        f"User navigation instruction: {literal}\n"
        "Issue exactly one bounded action now. Continue toward the same instruction after "
        "each numeric result, and issue stop only when the instruction is complete.\n"
        'Numeric result from the preceding bounded action: {"scan_count":1}'
    )

    assert literal_navigation_instruction(prompt) == literal
    assert is_compound_scan_approach_instruction(prompt) is True


@pytest.mark.parametrize(
    "instruction",
    [
        "Scan the room, then stop.",
        "Move closer to opaque 917, then stop.",
        "Scan the room, then move closer to opaque 917.",
        "Face opaque 917, then stop.",
    ],
)
def test_v3_2_calibration_does_not_expand_beyond_compound_protocol(
    instruction: str,
) -> None:
    assert is_compound_scan_approach_instruction(instruction) is False


def _write_numeric_map(path: Path) -> None:
    barrier = np.stack(
        (
            np.zeros(21, dtype=np.float32),
            np.linspace(-0.5, 0.5, 21, dtype=np.float32),
            np.full(21, 0.8, dtype=np.float32),
        ),
        axis=1,
    )
    target_surface = np.asarray([[1.4, 0.0, 0.8]], dtype=np.float32)
    points = np.concatenate((barrier, target_surface))
    features = np.ones((len(points), 4), dtype=np.float32)
    voxel_map = SparseVoxelMap(0.05, feature_dim=4)
    voxel_map.add_observations(
        points,
        features,
        rgb=np.full((len(points), 3), 96.0, dtype=np.float32),
        frame_id="f_000001",
    )
    voxel_map.save(path, metadata={"scene_id": "scene_314159"})


def test_v3_2_planner_uses_bound_numeric_map_and_releases_safe_bounded_waypoint(
    tmp_path: Path,
) -> None:
    map_path = tmp_path / "maps" / "scene_314159" / "voxel_map.npz"
    _write_numeric_map(map_path)
    state = SimpleNamespace(
        position_xy_m=np.asarray([-1.4, 0.0], dtype=np.float64),
        scene_version=1,
    )
    simulator = SimpleNamespace(state=state, collision_map=None)
    runtime = SimpleNamespace(
        simulator=simulator,
        map_updater=SimpleNamespace(
            persistent_map_path=tmp_path / "absent.npz",
            base_map_path=map_path,
        ),
    )
    backend = object.__new__(SemanticGroundedActionBackendV32)
    backend.runtime = runtime
    backend.last_grounding = {
        "target_available": True,
        "target_xyz_m": [1.4, 0.0, 0.8],
        "map_sha256": semantic_map_content_hash(map_path),
    }
    backend._v32_config = {
        "scene": {"room_size_m": [6.0, 5.0, 3.0]},
        "robot": {
            "radius_m": 0.16,
            "surface_padding_m": 0.02,
            "collision_z_min_m": 0.12,
            "collision_z_max_m": 1.80,
            "max_move_to_m": 0.50,
        },
    }
    backend.compound_approach_standoff_m = 0.35
    backend.planner_grid_resolution_m = 0.10
    backend.planner_standoff_tolerance_m = 0.20
    backend._v32_plan = None
    backend._v32_waypoints = []
    backend._v32_scene_version = None
    backend._v32_target_sha256 = None
    backend._v32_collision_map_sha256 = None
    backend._v32_collision_map = None

    result = backend._planner_action(
        "Scan, then approach arbitrary-alpha and stop."
    )

    assert result is not None
    call, audit = result
    assert call["tool"] == "move_to"
    endpoint = np.asarray([call["arguments"]["x"], call["arguments"]["y"]])
    start = state.position_xy_m
    assert np.linalg.norm(endpoint - start) <= 0.50 + 1e-9
    assert simulator.collision_map.segment_check(start, endpoint).collision is False
    assert audit["runtime_interlock_version"] == RUNTIME_INTERLOCK_VERSION == "v3.2"
    assert audit["calibrated_semantic_standoff_m"] == pytest.approx(0.35)
    assert audit["numeric_map_only"] is True
    assert audit["all_map_voxels_scored_for_grounding"] is True
    assert audit["environmental_text_inputs"] == []
    assert audit["oracle_inputs_at_runtime"] is False


def test_v3_2_runtime_source_has_no_benchmark_task_or_label_dependency() -> None:
    source = inspect.getsource(
        __import__(
            "semantic_3d_chat.robot.navigation_policy_v3_2",
            fromlist=["dummy"],
        )
    ).casefold()
    for prohibited in (
        "nav_005",
        "scene_000001",
        "configs/benchmarks",
        "data/oracle",
        "target_instance_id",
        "floor lamp",
        "chair",
        "table",
        "bowl",
    ):
        assert prohibited not in source


class _DiagnosticTextEncoder:
    output_dim = 1536

    def encode_queries(self, queries):
        output = np.zeros((len(queries), self.output_dim), dtype=np.float32)
        output[:, 0] = 1.0
        return output


def test_exact_live_runner_routes_enveloped_sequence_through_v3_2_planner(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Exercise the class installation and construction used by live main."""

    scripts = PROJECT_ROOT / "scripts"
    monkeypatch.syspath_prepend(str(scripts))
    sys.modules.pop("run_llm_navigation_inference_v3_2", None)
    wrapper = import_module("run_llm_navigation_inference_v3_2")
    routed_class = wrapper.install_v3_2_routing()
    assert routed_class is SemanticGroundedActionBackendV32
    assert wrapper.base.SemanticGroundedActionBackendV3 is routed_class

    contract_base = {"existing": True}
    monkeypatch.setattr(
        wrapper,
        "_ORIGINAL_RUN_CONTRACT",
        lambda **_kwargs: dict(contract_base),
    )
    contract = wrapper._v3_2_run_contract(navigation_policy_version=3)
    assert contract["existing"] is True
    assert contract["navigation_runtime_interlock_version"] == "v3.2"
    assert contract["compound_scan_approach_numeric_planner"] is True
    assert contract["inference_source_sha256"] == file_sha256(
        PROJECT_ROOT / "scripts/run_llm_navigation_inference_v3_2.py"
    )
    assert contract["navigation_policy_source_sha256"] == file_sha256(
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
            training_status=(
                "supervised_continuous_semantic_grounded_navigation_policy_v3"
            ),
        )

    monkeypatch.setattr(
        SemanticGroundedActionBackendV3,
        "generate",
        diagnostic_parent_generate,
    )
    task = NavigationTask(
        task_id="nav_314",
        family="update_after_scan",
        instruction=(
            "Scan the room, then move closer to arbitrary-alpha and stop."
        ),
        max_steps=12,
    )
    first_prompt = _policy_instruction(task, None)
    first = json.loads(backend.generate(first_prompt, correction_code=None).text)
    assert first["tool"] == "scan"
    state.scan_count = 1
    state.scene_version = 1
    prior = {"scan_count": 1, "success": True}
    second_prompt = _policy_instruction(task, prior)
    inherited = json.loads(
        diagnostic_parent_generate(
            backend,
            second_prompt,
            correction_code=None,
        ).text
    )
    second = json.loads(backend.generate(second_prompt, correction_code=None).text)

    assert inherited["tool"] == "stop"
    assert second["tool"] == "move_to"
    assert backend.last_grounding["numeric_compound_approach_planner"][
        "runtime_interlock_version"
    ] == "v3.2"
    assert backend.last_grounding["numeric_compound_approach_planner"][
        "environmental_text_inputs"
    ] == []

    sequence = ["scan", "move_to"]
    call = second
    for _ in range(11):
        if call["tool"] == "move_to":
            state.position_xy_m = np.asarray(
                [call["arguments"]["x"], call["arguments"]["y"]],
                dtype=np.float64,
            )
        call = json.loads(backend.generate(second_prompt, correction_code=None).text)
        sequence.append(call["tool"])
        if call["tool"] == "stop":
            break

    assert sequence[-1] == "stop"
    assert "move_to" in sequence[1:-1]
    assert sequence != ["scan", "stop"]
    completion = backend.last_grounding["numeric_compound_approach_planner"]
    assert completion["completion_satisfied"] is True
    assert completion["completion_mode"] == (
        "collision_capped_numeric_waypoint_standoff"
    )
