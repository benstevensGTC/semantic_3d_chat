from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
from types import ModuleType

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ISOLATION_SCRIPT = PROJECT_ROOT / "scripts/verify_gemma_waypoint_oracle_isolation.py"
LIVE_SCRIPT = PROJECT_ROOT / "scripts/verify_gemma_waypoint_rover_live.py"


def _load(path: Path, name: str) -> ModuleType:
    specification = importlib.util.spec_from_file_location(name, path)
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def _checkpoint(root: Path, weight_name: str) -> None:
    root.mkdir(parents=True)
    (root / weight_name).write_bytes(b"weights")
    (root / "runtime_metadata.json").write_text("{}", encoding="utf-8")


def test_source_oracle_is_atomically_hidden_and_restored(tmp_path: Path) -> None:
    module = _load(ISOLATION_SCRIPT, "gemma_waypoint_physical_isolation_success_test")
    project = tmp_path / "project"
    oracle = project / "data/oracle"
    payload = oracle / "scene_000001/oracle.json"
    payload.parent.mkdir(parents=True)
    payload.write_bytes(b"sealed-oracle")

    with module._source_oracle_temporarily_unavailable(project) as state:
        hidden = Path(state["hidden"])
        assert state["renamed"] is True
        assert state["unavailable_during_child"] is True
        assert state["restored"] is False
        assert not oracle.exists()
        assert hidden.is_dir()
        assert (hidden / "scene_000001/oracle.json").read_bytes() == b"sealed-oracle"

    assert state["restored"] is True
    assert payload.read_bytes() == b"sealed-oracle"
    assert not hidden.exists()


def test_source_oracle_is_restored_when_isolated_child_fails(tmp_path: Path) -> None:
    module = _load(ISOLATION_SCRIPT, "gemma_waypoint_physical_isolation_failure_test")
    project = tmp_path / "project"
    oracle = project / "data/oracle"
    payload = oracle / "scene_000001/oracle.json"
    payload.parent.mkdir(parents=True)
    payload.write_bytes(b"must-survive")
    state: dict[str, object] | None = None

    with (
        pytest.raises(RuntimeError, match="simulated child failure"),
        module._source_oracle_temporarily_unavailable(project) as isolated,
    ):
        state = isolated
        assert not oracle.exists()
        raise RuntimeError("simulated child failure")

    assert state is not None and state["restored"] is True
    assert payload.read_bytes() == b"must-survive"
    assert not Path(str(state["hidden"])).exists()


def test_source_oracle_isolation_rejects_symlink_target(tmp_path: Path) -> None:
    module = _load(ISOLATION_SCRIPT, "gemma_waypoint_physical_isolation_symlink_test")
    project = tmp_path / "project"
    data = project / "data"
    outside = tmp_path / "outside_oracle"
    data.mkdir(parents=True)
    outside.mkdir()
    (data / "oracle").symlink_to(outside, target_is_directory=True)

    with (
        pytest.raises(ValueError, match="exact real"),
        module._source_oracle_temporarily_unavailable(project),
    ):
        raise AssertionError("symlink target must never be hidden")

    assert (data / "oracle").is_symlink()
    assert outside.is_dir()


def test_materialized_runtime_copy_contains_only_sanitized_inputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load(ISOLATION_SCRIPT, "gemma_waypoint_isolation_materialize_test")
    source = tmp_path / "source"
    (source / "configs/runtime").mkdir(parents=True)
    (source / "configs/runtime/embodied_live.yaml").write_text("scene: {}\n", encoding="utf-8")
    (source / "configs/runtime/control.yaml").write_text("runtime: {}\n", encoding="utf-8")
    (source / "blender").mkdir()
    (source / "blender/render_runtime_observation.py").write_text("pass\n", encoding="utf-8")
    map_path = source / "data_gemma4/maps/scene_000001/voxel_map.npz"
    map_path.parent.mkdir(parents=True)
    map_path.write_bytes(b"continuous-map")
    asset = source / "data/runtime_assets/scene_000001/s_000001.blend"
    asset.parent.mkdir(parents=True)
    asset.write_bytes(b"opaque-blender")
    asset.with_suffix(".json").write_text("{}", encoding="utf-8")
    _checkpoint(source / "checkpoints/base", "adapter.safetensors")
    _checkpoint(source / "checkpoints/control", "control.safetensors")
    _checkpoint(source / "checkpoints/robot", "state.safetensors")
    _checkpoint(source / "checkpoints/navigation", "policy.safetensors")
    forbidden = source / "data/oracle/scene_000001/oracle.json"
    forbidden.parent.mkdir(parents=True)
    forbidden.write_text(json.dumps({"category": "chair"}), encoding="utf-8")
    monkeypatch.setattr(module, "SOURCE_PROJECT_ROOT", source)

    args = argparse.Namespace(
        scene="scene_000001",
        config="configs/runtime/embodied_live.yaml",
        control_config="configs/runtime/control.yaml",
        base_checkpoint="checkpoints/base",
        control_checkpoint="checkpoints/control",
        robot_state_checkpoint="checkpoints/robot",
        navigation_checkpoint="checkpoints/navigation",
        map=None,
        runtime_asset=None,
        check=False,
        question=["Question one?", "Question two?"],
        decision_instruction="Face something.",
        decision_steps=1,
    )
    isolated = tmp_path / "isolated"
    request = module._materialize_runtime_copy(args, isolated)

    assert Path(request["runtime_root"]) == isolated
    assert (isolated / "data_gemma4/maps/scene_000001/voxel_map.npz").read_bytes() == b"continuous-map"
    assert (isolated / "data/runtime_assets/scene_000001/s_000001.blend").read_bytes() == b"opaque-blender"
    assert not any(path.exists() for path in module._absence_paths(isolated))
    assert not (isolated / "data/oracle/scene_000001/oracle.json").exists()


def _valid_live_report(module: ModuleType) -> dict[str, object]:
    prefix = "a" * 64
    active_prefix = "b" * 64
    loaded = ["/runtime/configs/runtime.yaml", "/runtime/data_gemma4/maps/scene_000001/map.npz"]
    return {
        "schema": "semantic_3d_chat.gemma_waypoint_oracle_isolation.v1",
        "passed": True,
        "mode": "live",
        "runtime_copy_oracle_and_qa_absent": True,
        "source_oracle_directory_mutated": False,
        "scene_prefix_built_before_questions": True,
        "startup_scene_prefix_sha256": prefix,
        "startup_active_prefix_sha256": active_prefix,
        "question_receipts": [
            {
                "scene_prefix_sha256": prefix,
                "active_prefix_sha256": active_prefix,
            },
            {
                "scene_prefix_sha256": prefix,
                "active_prefix_sha256": active_prefix,
            },
        ],
        "decision_receipts": [{"scene_prefix_sha256": prefix}],
        "all_question_scene_prefix_hashes_identical": True,
        "all_question_active_prefix_hashes_identical": True,
        "all_decision_scene_prefix_hashes_identical": True,
        "final_scene_prefix_sha256": prefix,
        "whole_process_read_audit": {
            "started_before_production_runtime_import": True,
            "loaded_files": loaded,
            "forbidden_roots": [
                "/runtime/data/oracle",
                "/runtime/data/qa",
                "/runtime/data_gemma4/training",
            ],
            "forbidden_accesses": [],
            "forbidden_access_count": 0,
            "passed": True,
            "loaded_file_inventory_sha256": module._sha256_text("\n".join(loaded)),
        },
    }


def test_isolation_report_requires_prefix_invariance_and_no_forbidden_inventory() -> None:
    module = _load(ISOLATION_SCRIPT, "gemma_waypoint_isolation_validation_test")
    valid = _valid_live_report(module)
    module._validate_child_report(valid, expected_mode="live")

    changed_prefix = json.loads(json.dumps(valid))
    changed_prefix["decision_receipts"][0]["scene_prefix_sha256"] = "b" * 64
    with pytest.raises(AssertionError, match="static scene prefix"):
        module._validate_child_report(changed_prefix, expected_mode="live")

    leaked = json.loads(json.dumps(valid))
    leaked["whole_process_read_audit"]["loaded_files"].append(
        "/runtime/data/oracle/scene_000001/oracle.json"
    )
    with pytest.raises(AssertionError, match="oracle/QA isolation"):
        module._validate_child_report(leaked, expected_mode="live")

    renamed_leak = json.loads(json.dumps(valid))
    renamed_leak["whole_process_read_audit"]["loaded_files"].append(
        "/source/data/.oracle-unavailable-waypoint-123-deadbeef/scene_000001/oracle.json"
    )
    with pytest.raises(AssertionError, match="oracle/QA isolation"):
        module._validate_child_report(renamed_leak, expected_mode="live")


def test_live_verifier_requires_every_decision_to_bind_the_startup_scene_prefix() -> None:
    module = _load(LIVE_SCRIPT, "gemma_waypoint_live_prefix_test")
    prefix = "c" * 64
    decision = {
        "step": 1,
        "model_action": "stop",
        "accepted": True,
        "executed": True,
        "scene_prefix_sha256": prefix,
        "actual_gemma_causal_forward": True,
        "model_selected_every_waypoint_and_heading": True,
        "deterministic_route_planner_used": False,
        "substitution_applied": False,
        "synthetic_stop_applied": False,
    }
    assert module._verify_decisions(
        {"model_decisions": [decision]},
        expected_scene_prefix_sha256=prefix,
    ) == [decision]

    changed = dict(decision, scene_prefix_sha256="d" * 64)
    with pytest.raises(AssertionError, match="failed provenance"):
        module._verify_decisions(
            {"model_decisions": [changed]},
            expected_scene_prefix_sha256=prefix,
        )
