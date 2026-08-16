from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import torch
from safetensors.torch import save_file

from semantic_3d_chat.chat.file_audit import FileAccessAudit
from semantic_3d_chat.robot.navigation_policy import (
    ACTION_NAMES,
    ContinuousNavigationActionController,
    load_navigation_policy_checkpoint,
    normalized_argument_for_action,
    save_navigation_policy_checkpoint,
    split_active_prefix,
    tool_call_from_prediction,
)
from semantic_3d_chat.training.navigation_trace_generator import (
    generate_navigation_trace_dataset,
    load_navigation_trace_dataset,
)
from semantic_3d_chat.training.train_navigation_policy import (
    _evaluate,
    _PreparedSamples,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _runtime_metadata() -> dict[str, object]:
    return {
        "scene_token_count": 6,
        "robot_token_count": 2,
        "model_id": "local/test-model",
        "model_revision": "1" * 40,
        "max_turn_degrees": 45.0,
        "max_move_m": 0.5,
        "task_trained": True,
        "training_dataset_sha256": "2" * 64,
        "train_scene_count": 2,
        "validation_scene_count": 1,
        "scene_splits_disjoint": True,
        "complete_scene_prefix_required": True,
        "question_independent_scene_prefix_required": True,
        "every_scene_token_processed": True,
        "numeric_robot_tokens_required": True,
        "environmental_text_inputs": [],
        "oracle_inputs_at_runtime": False,
        "collision_interlock_required": True,
    }


def test_controller_processes_every_scene_token_and_robot_tokens() -> None:
    torch.manual_seed(7)
    controller = ContinuousNavigationActionController(16, model_dim=8)
    scene = torch.randn(2, 7, 16, requires_grad=True)
    robot = torch.randn(2, 3, 16, requires_grad=True)
    instruction = torch.randn(2, 16, requires_grad=True)
    logits, argument = controller(scene, robot, instruction)
    assert logits.shape == (2, len(ACTION_NAMES))
    assert argument.shape == (2, len(ACTION_NAMES))
    (logits.square().sum() + argument.square().sum()).backward()
    assert scene.grad is not None
    assert robot.grad is not None
    assert torch.all(scene.grad.abs().sum(dim=-1) > 0)
    assert torch.all(robot.grad.abs().sum(dim=-1) > 0)


def test_evaluation_preserves_original_sample_order_across_scene_batches() -> None:
    class OrderController(torch.nn.Module):
        def eval(self):
            return self

        def forward(
            self,
            scene_prefix,
            robot_tokens,
            instruction_embedding,
            *,
            scene_batch_indices,
        ):
            del scene_prefix, instruction_embedding, scene_batch_indices
            targets = robot_tokens[:, 0, 0].long()
            logits = torch.full((len(targets), len(ACTION_NAMES)), -10.0)
            logits.scatter_(1, targets[:, None], 10.0)
            arguments = torch.zeros_like(logits)
            return logits, arguments

    actions = torch.tensor([2, 0, 3, 1, 4], dtype=torch.long)
    samples = _PreparedSamples(
        scene_indices=torch.tensor([1, 0, 1, 0, 1], dtype=torch.long),
        robot_tokens=actions.float().reshape(-1, 1, 1),
        instruction_embeddings=torch.zeros(5, 1),
        action_targets=actions,
        argument_targets=torch.zeros(5),
        families=("test",) * 5,
        scene_ids=("scene_000001",) * 5,
    )
    metrics = _evaluate(
        OrderController(),
        torch.zeros(2, 3, 1),
        samples,
        batch_size=2,
    )
    assert metrics["action_accuracy"] == 1.0


def test_active_prefix_split_and_bounded_action_decode() -> None:
    scene = torch.arange(6 * 4, dtype=torch.float32).reshape(1, 6, 4)
    robot = torch.full((1, 2, 4), 99.0)
    active = torch.cat((scene[:, :-1], robot, scene[:, -1:]), dim=1)
    recovered_scene, recovered_robot = split_active_prefix(
        active, scene_token_count=6, robot_token_count=2
    )
    assert torch.equal(recovered_scene, scene)
    assert torch.equal(recovered_robot, robot)
    assert tool_call_from_prediction(
        ACTION_NAMES.index("turn"),
        -2.0,
        max_turn_degrees=45.0,
        max_move_m=0.5,
    ) == {"tool": "turn", "arguments": {"angle_degrees": -45.0}}
    movement = tool_call_from_prediction(
        ACTION_NAMES.index("move_forward"),
        2.0,
        max_turn_degrees=45.0,
        max_move_m=0.5,
    )
    assert movement == {"tool": "move_forward", "arguments": {"distance_meters": 0.5}}
    assert normalized_argument_for_action(
        "move_forward", 0.25, max_turn_degrees=45.0, max_move_m=0.5
    ) == pytest.approx(0.0)
    with pytest.raises(ValueError, match="exceeds"):
        normalized_argument_for_action(
            "turn", 46.0, max_turn_degrees=45.0, max_move_m=0.5
        )


def test_sanitized_checkpoint_loads_with_oracle_absent_and_is_audited(
    tmp_path: Path,
) -> None:
    controller = ContinuousNavigationActionController(16, model_dim=8)
    checkpoint = tmp_path / "checkpoints" / "policy"
    saved = save_navigation_policy_checkpoint(
        checkpoint, controller, runtime_metadata=_runtime_metadata()
    )
    oracle = tmp_path / "data" / "oracle"
    oracle.mkdir(parents=True)
    (oracle / "forbidden.json").write_text('{"category":"chair"}')
    (oracle / "forbidden.json").unlink()
    oracle.rmdir()
    audit = FileAccessAudit([oracle], forbidden_component_names={"oracle"}, block_forbidden=True)
    with audit:
        loaded, metadata = load_navigation_policy_checkpoint(
            checkpoint,
            expected_hidden_size=16,
            expected_model_id="local/test-model",
            expected_model_revision="1" * 40,
            audit=audit,
        )
    audit.assert_clean()
    assert metadata["oracle_inputs_at_runtime"] is False
    assert metadata["environmental_text_inputs"] == []
    assert metadata["weights_sha256"] == saved["weights_sha256"]
    assert {Path(path).name for path in audit.unique_paths} == {
        "policy.safetensors",
        "runtime_metadata.json",
    }
    assert not loaded.training
    assert not any(parameter.requires_grad for parameter in loaded.parameters())


def test_checkpoint_rejects_training_tree_and_semantic_metadata(tmp_path: Path) -> None:
    controller = ContinuousNavigationActionController(16, model_dim=8)
    with pytest.raises(ValueError, match="blocked"):
        save_navigation_policy_checkpoint(
            tmp_path / "training" / "policy",
            controller,
            runtime_metadata=_runtime_metadata(),
        )
    checkpoint = tmp_path / "checkpoints" / "policy"
    save_navigation_policy_checkpoint(
        checkpoint, controller, runtime_metadata=_runtime_metadata()
    )
    metadata_path = checkpoint / "runtime_metadata.json"
    metadata = json.loads(metadata_path.read_text())
    metadata["category_names"] = ["chair"]
    metadata_path.write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="fields changed"):
        load_navigation_policy_checkpoint(
            checkpoint,
            expected_hidden_size=16,
            expected_model_id="local/test-model",
            expected_model_revision="1" * 40,
        )


def _write_scene_sources(root: Path, scene_id: str) -> None:
    oracle_dir = root / "oracle" / scene_id
    map_dir = root / "maps" / scene_id
    prefix_dir = root / "prefixes"
    oracle_dir.mkdir(parents=True)
    map_dir.mkdir(parents=True)
    prefix_dir.mkdir(parents=True, exist_ok=True)
    instances = []
    for index, (category, center) in enumerate(
        {
            "chair": (-1.0, 0.6, 0.6),
            "table": (0.9, 0.6, 0.4),
            "bowl": (-1.3, -1.0, 0.1),
            "floor lamp": (-2.0, 1.5, 1.1),
        }.items()
    ):
        instances.append(
            {
                "instance_id": f"i_{100 + index:06d}",
                "category": category,
                "expected_center_xyz_m": list(center),
            }
        )
    (oracle_dir / "oracle.json").write_text(
        json.dumps({"scene_id": scene_id, "instances": instances})
    )
    # Anonymous high wall samples make a nonempty collision map while leaving
    # the room center and all planned routes free in this contract test.
    centers = np.asarray(
        [
            [-2.9, -2.4, 1.0],
            [-2.9, 2.4, 1.0],
            [2.9, -2.4, 1.0],
            [2.9, 2.4, 1.0],
        ],
        dtype=np.float32,
    )
    np.savez(map_dir / "voxel_map.npz", centers_world=centers)
    prefix_path = prefix_dir / f"{scene_id}.safetensors"
    save_file({"scene_prefix": torch.randn(1, 6, 16)}, str(prefix_path))


def test_oracle_trace_generator_is_physically_isolated_and_bounded(
    tmp_path: Path,
) -> None:
    for scene_id in ("scene_000001", "scene_000002"):
        _write_scene_sources(tmp_path, scene_id)
    prefix_dir = tmp_path / "prefixes"
    scenes = {}
    for scene_id in ("scene_000001", "scene_000002"):
        prefix_path = prefix_dir / f"{scene_id}.safetensors"
        scenes[scene_id] = {
            "filename": prefix_path.name,
            "file_sha256": _sha256(prefix_path),
            "prefix_sha256": "3" * 64,
        }
    (prefix_dir / "manifest.json").write_text(json.dumps({"scenes": scenes}))
    config = {
        "seed": 7,
        "scene": {"room_size_m": [6.0, 5.0, 3.0]},
        "robot": {
            "radius_m": 0.25,
            "max_turn_degrees": 45.0,
            "max_move_m": 0.5,
            "collision_z_min_m": 0.12,
            "collision_z_max_m": 1.8,
            "surface_padding_m": 0.035,
        },
        "navigation_policy": {
            "train_scene_ids": ["scene_000001"],
            "validation_scene_ids": ["scene_000002"],
            "oracle_root": str(tmp_path / "oracle"),
            "map_root": str(tmp_path / "maps"),
            "prefix_cache_root": str(prefix_dir),
            "start_pose_count": 1,
            "initial_yaw_degrees": [0.0],
            "planner_grid_resolution_m": 0.20,
            "planner_standoff_m": 0.60,
            "planner_standoff_tolerance_m": 0.10,
            "planner_angular_samples": 24,
        },
    }
    destination = tmp_path / "training" / "navigation"
    manifest = generate_navigation_trace_dataset(config, destination)
    loaded, rows = load_navigation_trace_dataset(destination)
    assert loaded["dataset_sha256"] == manifest["dataset_sha256"]
    assert loaded["scene_splits_disjoint"] is True
    assert loaded["collision_checked_movement_targets"] is True
    assert {row["split"] for row in rows} == {"train", "validation"}
    assert all(-1.0 <= row["argument_target_normalized"] <= 1.0 for row in rows)
    assert all(row["oracle_available_at_runtime"] is False for row in rows)
    with pytest.raises(ValueError, match="training tree"):
        generate_navigation_trace_dataset(config, tmp_path / "runtime" / "bad")
