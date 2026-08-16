from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

import semantic_3d_chat.robot.runtime_refresh as runtime_refresh_module
from semantic_3d_chat.chat.file_audit import FileAccessAudit
from semantic_3d_chat.chat.runtime import ChatAnswer
from semantic_3d_chat.language.prefix_injection import prefix_sha256
from semantic_3d_chat.mapping.voxel_map import SparseVoxelMap
from semantic_3d_chat.robot.runtime_refresh import (
    build_refreshing_embodied_runtime,
    robot_state_encoder_sha256,
)
from semantic_3d_chat.robot.state_encoder import RobotStateEncoder
from semantic_3d_chat.scene_encoder.map_io import MapTensorData
from semantic_3d_chat.vision.patch_features import DensePatchFeatures


class CountingDenseEncoder:
    def __init__(self, component_dim: int = 2) -> None:
        self.component_dim = component_dim
        self.calls = 0

    def encode_image(self, image: np.ndarray) -> DensePatchFeatures:
        self.calls += 1
        assert image.shape == (8, 8, 3)
        streams = [
            torch.full((2, 2, self.component_dim), 1.0 + offset)
            for offset in range(3)
        ]
        return DensePatchFeatures(*streams)


class FakeSceneRuntime:
    def __init__(
        self,
        config: dict,
        map_data: MapTensorData,
        *,
        scene_dtype: torch.dtype = torch.float32,
    ) -> None:
        self.config = config
        self.scene_id = "scene_000001"
        self.checkpoint_metadata = {"semantic_dim": map_data.feature_dim}
        self.map_data = map_data
        signal = float(map_data.semantic.float().sum() + map_data.observation_count.sum())
        values = torch.linspace(signal, signal + 1.0, 32).reshape(1, 4, 8).to(
            scene_dtype
        )
        self.scene_prefix = values
        self.scene_prefix_hash = prefix_sha256(values)
        self.questions_answered = 0

    def current_prefix_hash(self) -> str:
        return prefix_sha256(self.scene_prefix)

    def answer(self, question: str) -> ChatAnswer:
        self.questions_answered += 1
        return ChatAnswer(
            question=question,
            answer="numeric",
            grounding_xyz_m=(0.0, 0.0, 0.0),
            grounding_confidence=0.5,
            grounding_support_distance_m=0.0,
            prefix_hash=self.scene_prefix_hash,
            generated_tokens=1,
            elapsed_seconds=0.0,
        )


class CountingRuntimeBuilder:
    def __init__(self, *, fail: bool = False) -> None:
        self.calls = 0
        self.fail = fail

    def __call__(self, previous: FakeSceneRuntime, map_data: MapTensorData) -> FakeSceneRuntime:
        self.calls += 1
        if self.fail:
            raise RuntimeError("synthetic full-scene tokenization failure")
        return FakeSceneRuntime(previous.config, map_data)


class CountingSignatureControl:
    """Tiny stand-in for the shared V3/V4/V5/V6 signature interface."""

    def __init__(self) -> None:
        self.encode_calls = 0

    def encode_scene(self, scene_prefix: torch.Tensor) -> torch.Tensor:
        self.encode_calls += 1
        return torch.stack(
            (
                scene_prefix[:, 1:-1].mean(dim=1),
                scene_prefix[:, 1:-1].square().mean(dim=1),
            ),
            dim=1,
        ).detach()

    def forward_from_signature(
        self,
        signature: torch.Tensor,
        question_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        del question_embeddings
        return signature


class SignatureControlledRuntime:
    """Stub wrapper with the production question-control cache protocol."""

    def __init__(
        self,
        base: FakeSceneRuntime,
        control: CountingSignatureControl,
        control_metadata: dict,
    ) -> None:
        self.base = base
        self.control = control
        self.control_metadata = dict(control_metadata)
        self.scene_prefix_hash = base.scene_prefix_hash
        self._scene_control_signature = control.encode_scene(base.scene_prefix.float())

    def answer(self, question: str) -> ChatAnswer:
        return self.base.answer(question)


class CountingControlledRuntimeBuilder:
    def __init__(self, *, stale_signature: bool = False) -> None:
        self.calls = 0
        self.stale_signature = stale_signature

    def __call__(
        self,
        previous: SignatureControlledRuntime,
        map_data: MapTensorData,
    ) -> SignatureControlledRuntime:
        self.calls += 1
        candidate = SignatureControlledRuntime(
            FakeSceneRuntime(previous.base.config, map_data),
            previous.control,
            previous.control_metadata,
        )
        if self.stale_signature:
            candidate._scene_control_signature = previous._scene_control_signature
        return candidate


class CountingDenseControl:
    """Tiny stand-in for the V74/V75 pre-question dense scene cache."""

    def __init__(self) -> None:
        self.encode_calls = 0

    def encode_scene(
        self,
        scene_prefix: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        self.encode_calls += 1
        environment = scene_prefix[:, 1:-1]
        return environment.detach(), environment.square().detach()

    def forward_encoded(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
        question_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        del question_embeddings
        return key + value


class DenseControlledRuntime:
    """Stub wrapper matching the production V74/V75 dense-cache protocol."""

    def __init__(
        self,
        base: FakeSceneRuntime,
        control: CountingDenseControl,
        control_metadata: dict,
    ) -> None:
        self.base = base
        self.control = control
        self.control_metadata = dict(control_metadata)
        self.scene_prefix_hash = base.scene_prefix_hash
        self._scene_control_signature = None
        key, value = control.encode_scene(base.scene_prefix.float())
        self._scene_control_key = key.detach().clone()
        self._scene_control_value = value.detach().clone()

    def answer(self, question: str) -> ChatAnswer:
        return self.base.answer(question)


class CountingDenseRuntimeBuilder:
    def __init__(self, *, stale_cache: bool = False) -> None:
        self.calls = 0
        self.stale_cache = stale_cache

    def __call__(
        self,
        previous: DenseControlledRuntime,
        map_data: MapTensorData,
    ) -> DenseControlledRuntime:
        self.calls += 1
        candidate = DenseControlledRuntime(
            FakeSceneRuntime(previous.base.config, map_data),
            previous.control,
            previous.control_metadata,
        )
        if self.stale_cache:
            candidate._scene_control_key = previous._scene_control_key
            candidate._scene_control_value = previous._scene_control_value
        return candidate


def _base_map(path: Path) -> None:
    points = np.asarray(
        [
            [-0.30, 1.00, 1.00],
            [-0.10, 1.00, 1.10],
            [0.10, 1.00, 1.20],
            [0.30, 1.00, 1.30],
            [-0.25, 1.25, 1.35],
            [0.00, 1.25, 1.20],
            [0.25, 1.25, 1.05],
        ],
        dtype=np.float32,
    )
    features = np.arange(1, 43, dtype=np.float32).reshape(7, 6)
    rgb = np.tile(np.asarray([[40.0, 130.0, 210.0]], dtype=np.float32), (7, 1))
    voxel_map = SparseVoxelMap(0.05, feature_dim=6)
    voxel_map.add_observations(points, features, rgb=rgb, frame_id="f_000000")
    voxel_map.save(path, metadata={"scene_id": "scene_000001"})


def _config(tmp_path: Path) -> dict:
    maps = tmp_path / "maps"
    _base_map(maps / "scene_000001" / "voxel_map.npz")
    return {
        "seed": 7,
        "paths": {
            "data_root": str(tmp_path / "runtime"),
            "maps_root": str(maps),
        },
        "scene": {"room_size_m": [6.0, 5.0, 3.0]},
        "scene_encoder": {"input_voxel_size_m": None},
        "render": {"resolution": [8, 8], "horizontal_fov_degrees": 72.0},
        "mapping": {
            "depth_min_m": 0.1,
            "depth_max_m": 6.0,
            "pixel_stride": 1,
            "confidence_distance_scale_m": 6.0,
        },
        "robot": {
            "radius_m": 0.20,
            "camera_height_m": 1.20,
            "max_move_m": 0.50,
            "max_move_to_m": 1.0,
            "max_turn_degrees": 45.0,
            "max_look_delta_degrees": 30.0,
            "max_camera_yaw_offset_degrees": 60.0,
            "max_pitch_degrees": 45.0,
            "collision_z_min_m": 0.12,
            "collision_z_max_m": 1.80,
            "surface_padding_m": 0.02,
            "scan_depth_min_m": 0.10,
            "scan_depth_max_m": 6.0,
            "initial_position_xy_m": [0.0, 0.0],
            "history_length": 16,
        },
    }


def _runtime(
    tmp_path: Path,
    *,
    dense_dim: int = 2,
    builder: CountingRuntimeBuilder | None = None,
    state_encoder: RobotStateEncoder | None = None,
    auto_scan_after_motion: bool | None = None,
    scene_dtype: torch.dtype = torch.float32,
):
    config = _config(tmp_path)
    if auto_scan_after_motion is not None:
        config["robot"]["auto_scan_after_motion"] = auto_scan_after_motion
    base_data = MapTensorData(
        semantic=torch.arange(1, 43, dtype=torch.float32).reshape(7, 6),
        xyz=torch.tensor(
            [
                [-0.30, 1.00, 1.00],
                [-0.10, 1.00, 1.10],
                [0.10, 1.00, 1.20],
                [0.30, 1.00, 1.30],
                [-0.25, 1.25, 1.35],
                [0.00, 1.25, 1.20],
                [0.25, 1.25, 1.05],
            ]
        ),
        rgb=torch.zeros(7, 3),
        normal=torch.zeros(7, 3),
        confidence=torch.ones(7),
        observation_count=torch.ones(7),
        room_min=torch.tensor([-3.0, -2.5, 0.0]),
        room_max=torch.tensor([3.0, 2.5, 3.0]),
        source_voxel_count=7,
        input_voxel_size_m=None,
    )
    chat = FakeSceneRuntime(config, base_data, scene_dtype=scene_dtype)
    active_builder = builder or CountingRuntimeBuilder()
    state_hash = None if state_encoder is None else robot_state_encoder_sha256(state_encoder)
    result = build_refreshing_embodied_runtime(
        config,
        "scene_000001",
        checkpoint="unused",
        chat_runtime=chat,
        vision_encoder=CountingDenseEncoder(dense_dim),
        persistent_map_path=tmp_path / "persistent" / "semantic_map.npz",
        runtime_builder=active_builder,
        robot_state_encoder=state_encoder,
        robot_state_encoder_sha256=state_hash,
    )
    return result, active_builder


def _controlled_runtime(
    tmp_path: Path,
    *,
    builder: CountingControlledRuntimeBuilder | None = None,
    state_encoder: RobotStateEncoder | None = None,
    audit: FileAccessAudit | None = None,
):
    config = _config(tmp_path)
    base_data = MapTensorData(
        semantic=torch.arange(1, 43, dtype=torch.float32).reshape(7, 6),
        xyz=torch.tensor(
            [
                [-0.30, 1.00, 1.00],
                [-0.10, 1.00, 1.10],
                [0.10, 1.00, 1.20],
                [0.30, 1.00, 1.30],
                [-0.25, 1.25, 1.35],
                [0.00, 1.25, 1.20],
                [0.25, 1.25, 1.05],
            ]
        ),
        rgb=torch.zeros(7, 3),
        normal=torch.zeros(7, 3),
        confidence=torch.ones(7),
        observation_count=torch.ones(7),
        room_min=torch.tensor([-3.0, -2.5, 0.0]),
        room_max=torch.tensor([3.0, 2.5, 3.0]),
        source_voxel_count=7,
        input_voxel_size_m=None,
    )
    control = CountingSignatureControl()
    chat = SignatureControlledRuntime(
        FakeSceneRuntime(config, base_data),
        control,
        {"architecture": "stub_shared_scene_signature_v6"},
    )
    active_builder = builder or CountingControlledRuntimeBuilder()
    state_hash = None if state_encoder is None else robot_state_encoder_sha256(state_encoder)
    result = build_refreshing_embodied_runtime(
        config,
        "scene_000001",
        checkpoint="unused",
        chat_runtime=chat,
        vision_encoder=CountingDenseEncoder(),
        persistent_map_path=tmp_path / "persistent" / "semantic_map.npz",
        runtime_builder=active_builder,
        robot_state_encoder=state_encoder,
        robot_state_encoder_sha256=state_hash,
        audit=audit,
    )
    return result, active_builder, control


def _dense_controlled_runtime(
    tmp_path: Path,
    *,
    builder: CountingDenseRuntimeBuilder | None = None,
):
    config = _config(tmp_path)
    base_data = MapTensorData(
        semantic=torch.arange(1, 43, dtype=torch.float32).reshape(7, 6),
        xyz=torch.tensor(
            [
                [-0.30, 1.00, 1.00],
                [-0.10, 1.00, 1.10],
                [0.10, 1.00, 1.20],
                [0.30, 1.00, 1.30],
                [-0.25, 1.25, 1.35],
                [0.00, 1.25, 1.20],
                [0.25, 1.25, 1.05],
            ]
        ),
        rgb=torch.zeros(7, 3),
        normal=torch.zeros(7, 3),
        confidence=torch.ones(7),
        observation_count=torch.ones(7),
        room_min=torch.tensor([-3.0, -2.5, 0.0]),
        room_max=torch.tensor([3.0, 2.5, 3.0]),
        source_voxel_count=7,
        input_voxel_size_m=None,
    )
    control = CountingDenseControl()
    chat = DenseControlledRuntime(
        FakeSceneRuntime(config, base_data),
        control,
        {"architecture": "stub_dense_full_scene_control_v75"},
    )
    active_builder = builder or CountingDenseRuntimeBuilder()
    result = build_refreshing_embodied_runtime(
        config,
        "scene_000001",
        checkpoint="unused",
        chat_runtime=chat,
        vision_encoder=CountingDenseEncoder(),
        persistent_map_path=tmp_path / "persistent" / "semantic_map.npz",
        runtime_builder=active_builder,
    )
    return result, active_builder, control


def test_successful_scan_atomically_changes_complete_prefix_binding(tmp_path: Path) -> None:
    runtime, builder = _runtime(tmp_path)
    before = runtime.prefix_binding()

    result = runtime.scan()

    assert result["success"] is True
    assert result["map_version"] == result["scene_version"] == 1
    assert result["map_sha256"] != before["map_sha256"]
    assert result["scene_prefix_sha256"] != before["scene_prefix_sha256"]
    assert result["processed_voxels"] == result["source_voxels"]
    assert result["binding_sha256"] == runtime.prefix_binding()["binding_sha256"]
    assert builder.calls == 1
    encoded = json.dumps(result)
    for prohibited in ("oracle", "category", "caption", "relationship", "object_name"):
        assert prohibited not in encoded


@pytest.mark.parametrize(
    ("method", "arguments", "receipt_field", "receipt_value"),
    [
        ("look", (10.0, -5.0), "turn_degrees", 10.0),
        ("turn", (15.0,), "turn_degrees", 15.0),
        ("move_forward", (0.10,), "distance_moved", 0.10),
        ("move_backward", (0.10,), "distance_moved", 0.10),
        ("move_to", (0.10, 0.0), "distance_moved", 0.10),
    ],
)
def test_auto_scan_after_successful_motion_refreshes_map_and_prefix(
    tmp_path: Path,
    method: str,
    arguments: tuple[float, ...],
    receipt_field: str,
    receipt_value: float,
) -> None:
    runtime, builder = _runtime(tmp_path, auto_scan_after_motion=True)
    before = runtime.prefix_binding()
    encoder = runtime.map_updater.encoder

    result = getattr(runtime, method)(*arguments)

    assert result["success"] is True
    assert result["scene_version"] == result["map_version"] == 1
    assert result["scan_count"] == 1
    assert result["observation_id"] == "o_000001"
    assert result["map_sha256"] != before["map_sha256"]
    assert result["scene_prefix_sha256"] != before["scene_prefix_sha256"]
    assert result[receipt_field] == pytest.approx(receipt_value)
    assert encoder.calls == 1
    assert builder.calls == 1
    encoded = json.dumps(result)
    for prohibited in ("oracle", "category", "caption", "relationship", "object_name"):
        assert prohibited not in encoded


def test_auto_scan_does_not_run_after_rejected_motion(tmp_path: Path) -> None:
    runtime, builder = _runtime(tmp_path, auto_scan_after_motion=True)
    before = runtime.prefix_binding()
    encoder = runtime.map_updater.encoder

    result = runtime.move_to(0.0, 0.85)

    assert result["success"] is False
    assert result["error_code"] == "E_COLLISION"
    assert result["scene_version"] == result["map_version"] == 0
    assert result["scan_count"] == 0
    assert result["observation_id"] is None
    assert runtime.prefix_binding() == before
    assert encoder.calls == 0
    assert builder.calls == 0


def test_prefix_is_invariant_for_questions_within_one_map_version(tmp_path: Path) -> None:
    runtime, _builder = _runtime(tmp_path)
    runtime.scan()
    binding = runtime.prefix_binding()

    first = runtime.answer("first question")
    second = runtime.answer("second question")

    assert first.prefix_hash == second.prefix_hash == binding["active_prefix_sha256"]
    assert runtime.prefix_binding() == binding


def test_refreshing_runtime_reset_rejects_cross_scene_without_prefix_bypass(
    tmp_path: Path,
) -> None:
    runtime, _builder = _runtime(tmp_path)
    runtime.scan()
    binding = runtime.prefix_binding()
    state = runtime.get_robot_state()

    result = runtime.reset_scene("scene_000002", 99)

    assert result["success"] is False
    assert result["error_code"] == "E_SCENE_UNAVAILABLE"
    assert result["scene_id"] == state["scene_id"]
    assert result["scene_version"] == state["scene_version"]
    assert runtime.prefix_binding() == binding


def test_refreshing_runtime_same_scene_reset_restores_base_map_and_prefix(
    tmp_path: Path,
) -> None:
    runtime, _builder = _runtime(tmp_path)
    base_binding = runtime.prefix_binding()
    runtime.turn(15.0)
    scanned = runtime.scan()
    assert scanned["success"] is True
    assert scanned["map_version"] == 1
    assert runtime.map_updater.persistent_map_path.is_file()

    result = runtime.reset_scene("scene_000001", 99)

    assert result["success"] is True
    assert result["error_code"] is None
    assert result["seed"] == 99
    assert result["scene_version"] == result["map_version"] == 0
    assert result["scan_count"] == 0
    assert result["body_yaw_degrees"] == 0.0
    assert result["scene_prefix_sha256"] == base_binding["scene_prefix_sha256"]
    assert result["map_sha256"] == base_binding["map_sha256"]
    assert not runtime.map_updater.persistent_map_path.exists()
    assert runtime.prefix_binding()["scene_prefix_sha256"] == base_binding[
        "scene_prefix_sha256"
    ]


def test_refreshing_runtime_reset_failure_restores_scanner_and_prefix(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime, _builder = _runtime(tmp_path)
    assert runtime.scan()["success"] is True
    binding = runtime.prefix_binding()
    scanner_counts = runtime.simulator.scanner.observation_count.copy()
    state = runtime.get_robot_state()

    def fail_reset() -> None:
        raise OSError("synthetic reset failure")

    monkeypatch.setattr(runtime.map_updater, "reset_to_base", fail_reset)
    result = runtime.reset_scene("scene_000001", 99)

    assert result["success"] is False
    assert result["error_code"] == "E_MAP_RESET"
    assert result["scene_version"] == state["scene_version"]
    assert runtime.prefix_binding() == binding
    assert np.array_equal(runtime.simulator.scanner.observation_count, scanner_counts)


@pytest.mark.parametrize(
    ("scene_id", "seed", "error_code"),
    [
        ("chair", 1, "E_SCENE_ID"),
        ("scene_000001", -1, "E_NUMERIC"),
        ("scene_000001", True, "E_NUMERIC"),
    ],
)
def test_refreshing_runtime_reset_rejects_invalid_request_without_state_change(
    tmp_path: Path,
    scene_id: str,
    seed: object,
    error_code: str,
) -> None:
    runtime, _builder = _runtime(tmp_path)
    before = runtime.prefix_binding()

    result = runtime.reset_scene(scene_id, seed)

    assert result["success"] is False
    assert result["error_code"] == error_code
    assert runtime.prefix_binding() == before


def test_failed_fusion_rolls_back_map_version_and_prefix(tmp_path: Path) -> None:
    runtime, builder = _runtime(tmp_path, dense_dim=3)
    before = runtime.prefix_binding()

    result = runtime.scan()

    assert result["success"] is False and result["error_code"] == "E_MAP_UPDATE"
    assert result["scene_version"] == 0
    assert runtime.prefix_binding() == before
    assert builder.calls == 0
    assert not (tmp_path / "persistent" / "semantic_map.npz").exists()


def test_failed_prefix_build_rolls_back_staged_numeric_map(tmp_path: Path) -> None:
    builder = CountingRuntimeBuilder(fail=True)
    runtime, _ = _runtime(tmp_path, builder=builder)
    before = runtime.prefix_binding()

    result = runtime.scan()

    assert result["success"] is False and result["error_code"] == "E_MAP_UPDATE"
    assert runtime.prefix_binding() == before
    assert builder.calls == 1
    assert not (tmp_path / "persistent" / "semantic_map.npz").exists()


def test_checkpoint_bound_robot_tokens_are_cached_before_questions(tmp_path: Path) -> None:
    torch.manual_seed(19)
    state_encoder = RobotStateEncoder(8, hidden_dim=12, token_count=2)
    runtime, _builder = _runtime(tmp_path, state_encoder=state_encoder)
    initial = runtime.prefix_binding()
    assert initial["active_prefix_sha256"] != initial["scene_prefix_sha256"]
    assert initial["robot_state_encoder_sha256"] == robot_state_encoder_sha256(state_encoder)

    first = runtime.answer("first")
    second = runtime.answer("second")
    assert first.prefix_hash == second.prefix_hash == initial["active_prefix_sha256"]
    assert runtime.prefix_binding() == initial

    turned = runtime.turn(15.0)
    assert turned["scene_prefix_sha256"] == initial["scene_prefix_sha256"]
    assert turned["active_prefix_sha256"] != initial["active_prefix_sha256"]
    assert turned["robot_state_sha256"] != initial["robot_state_sha256"]


def test_action_prefix_snapshot_is_an_immutable_scene_plus_robot_copy(tmp_path: Path) -> None:
    torch.manual_seed(29)
    state_encoder = RobotStateEncoder(8, hidden_dim=12, token_count=2)
    runtime, _builder = _runtime(tmp_path, state_encoder=state_encoder)

    snapshot, binding = runtime.active_prefix_snapshot()

    assert prefix_sha256(snapshot) == binding["active_prefix_sha256"]
    assert binding["scene_prefix_sha256"] != binding["active_prefix_sha256"]
    assert binding["robot_tokens_sha256"] is not None
    snapshot.zero_()
    assert runtime.prefix_binding()["active_prefix_sha256"] == binding["active_prefix_sha256"]
    assert prefix_sha256(runtime.active_prefix_snapshot()[0]) == binding["active_prefix_sha256"]

    runtime.turn(15.0)
    moved_snapshot, moved_binding = runtime.active_prefix_snapshot()
    assert moved_binding["scene_prefix_sha256"] == binding["scene_prefix_sha256"]
    assert moved_binding["active_prefix_sha256"] != binding["active_prefix_sha256"]
    assert prefix_sha256(moved_snapshot) == moved_binding["active_prefix_sha256"]


def test_robot_token_binding_hashes_exact_bfloat16_active_slice(tmp_path: Path) -> None:
    """The published robot digest must bind the bytes inserted for Gemma."""

    torch.manual_seed(31)
    state_encoder = RobotStateEncoder(8, hidden_dim=12, token_count=2)
    runtime, _builder = _runtime(
        tmp_path,
        state_encoder=state_encoder,
        scene_dtype=torch.bfloat16,
    )

    active, binding = runtime.active_prefix_snapshot()
    robot = active[:, 3:5]

    assert active.dtype is torch.bfloat16
    assert prefix_sha256(robot) == binding["robot_tokens_sha256"]


def test_question_control_scan_refreshes_prefix_and_cached_signature(tmp_path: Path) -> None:
    runtime, builder, control = _controlled_runtime(tmp_path)
    before = runtime.prefix_binding()

    result = runtime.scan()

    assert result["success"] is True
    assert result["scene_prefix_sha256"] != before["scene_prefix_sha256"]
    assert result["scene_control_signature_sha256"] is not None
    assert (
        result["scene_control_signature_sha256"]
        != before["scene_control_signature_sha256"]
    )
    assert builder.calls == 1
    # Initial/candidate construction and binding verification each encode once.
    assert control.encode_calls == 4


def test_dense_question_control_scan_refreshes_prefix_and_cached_key_value(
    tmp_path: Path,
) -> None:
    runtime, builder, control = _dense_controlled_runtime(tmp_path)
    before = runtime.prefix_binding()

    result = runtime.scan()

    assert result["success"] is True
    assert result["scene_prefix_sha256"] != before["scene_prefix_sha256"]
    assert result["scene_control_signature_sha256"] is not None
    assert (
        result["scene_control_signature_sha256"]
        != before["scene_control_signature_sha256"]
    )
    assert builder.calls == 1
    # Initial/candidate construction and binding verification each encode once.
    assert control.encode_calls == 4

    encode_calls = control.encode_calls
    first = runtime.answer("one wording")
    second = runtime.answer("an unrelated wording")
    assert first.prefix_hash == second.prefix_hash == result["scene_prefix_sha256"]
    assert control.encode_calls == encode_calls


def test_stale_dense_question_control_cache_rolls_back_map_and_runtime(
    tmp_path: Path,
) -> None:
    builder = CountingDenseRuntimeBuilder(stale_cache=True)
    runtime, _builder, _control = _dense_controlled_runtime(
        tmp_path,
        builder=builder,
    )
    before = runtime.prefix_binding()

    result = runtime.scan()

    assert result["success"] is False
    assert result["error_code"] == "E_MAP_UPDATE"
    assert runtime.prefix_binding() == before
    assert builder.calls == 1
    assert not (tmp_path / "persistent" / "semantic_map.npz").exists()


def test_default_rebuilder_rewraps_any_shared_signature_runtime(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime, _builder, control = _controlled_runtime(tmp_path)
    previous = runtime.prefix_refresher.runtime
    map_data = previous.base.map_data
    replacement = MapTensorData(
        semantic=map_data.semantic * 1.5,
        xyz=map_data.xyz,
        rgb=map_data.rgb,
        normal=map_data.normal,
        confidence=map_data.confidence,
        observation_count=map_data.observation_count + 1.0,
        room_min=map_data.room_min,
        room_max=map_data.room_max,
        source_voxel_count=map_data.source_voxel_count,
        input_voxel_size_m=map_data.input_voxel_size_m,
    )
    prior_signature = prefix_sha256(previous._scene_control_signature)
    monkeypatch.setattr(
        runtime_refresh_module,
        "_rebuild_static_base",
        lambda base, candidate_map: FakeSceneRuntime(base.config, candidate_map),
    )

    candidate = runtime_refresh_module._rebuild_runtime(previous, replacement)

    assert type(candidate) is type(previous)
    assert candidate.control is control
    assert candidate.base is not previous.base
    assert prefix_sha256(candidate._scene_control_signature) != prior_signature


def test_question_control_signature_is_question_independent(tmp_path: Path) -> None:
    runtime, _builder, control = _controlled_runtime(tmp_path)
    runtime.scan()
    binding = runtime.prefix_binding()
    encode_calls = control.encode_calls

    first = runtime.answer("one wording")
    second = runtime.answer("an unrelated wording")

    assert first.prefix_hash == second.prefix_hash == binding["scene_prefix_sha256"]
    assert runtime.prefix_binding() == binding
    assert control.encode_calls == encode_calls


def test_stale_question_control_signature_rolls_back_map_and_runtime(tmp_path: Path) -> None:
    builder = CountingControlledRuntimeBuilder(stale_signature=True)
    runtime, _builder, _control = _controlled_runtime(tmp_path, builder=builder)
    before = runtime.prefix_binding()

    result = runtime.scan()

    assert result["success"] is False
    assert result["error_code"] == "E_MAP_UPDATE"
    assert runtime.prefix_binding() == before
    assert builder.calls == 1
    assert not (tmp_path / "persistent" / "semantic_map.npz").exists()


def test_question_control_robot_state_remains_a_separate_numeric_seam(
    tmp_path: Path,
) -> None:
    torch.manual_seed(23)
    state_encoder = RobotStateEncoder(8, hidden_dim=12, token_count=2)
    runtime, _builder, control = _controlled_runtime(
        tmp_path,
        state_encoder=state_encoder,
    )
    initial = runtime.prefix_binding()
    signature = initial["scene_control_signature_sha256"]
    encode_calls = control.encode_calls

    turned = runtime.turn(15.0)
    answer = runtime.answer("numeric state seam")

    assert turned["scene_prefix_sha256"] == initial["scene_prefix_sha256"]
    assert turned["scene_control_signature_sha256"] == signature
    assert turned["active_prefix_sha256"] != initial["active_prefix_sha256"]
    assert answer.prefix_hash == turned["active_prefix_sha256"]
    assert control.encode_calls == encode_calls


def test_question_control_refresh_records_only_sanitized_numeric_maps(
    tmp_path: Path,
) -> None:
    audit = FileAccessAudit(
        [tmp_path / "oracle", tmp_path / "qa"],
        forbidden_component_names={"oracle", "qa"},
        block_forbidden=True,
    )
    with audit:
        runtime, _builder, _control = _controlled_runtime(tmp_path, audit=audit)
        result = runtime.scan()

    assert result["success"] is True
    assert audit.forbidden_accesses() == []
    loaded = "\n".join(audit.unique_paths)
    assert "voxel_map.npz" in loaded
    assert "pending.npz" in loaded
    assert "/oracle/" not in loaded and "/qa/" not in loaded


def test_oracle_path_is_rejected_before_map_or_prefix_use(tmp_path: Path) -> None:
    config = _config(tmp_path)
    base_data = MapTensorData(
        semantic=torch.ones(1, 6),
        xyz=torch.zeros(1, 3),
        rgb=torch.zeros(1, 3),
        normal=torch.zeros(1, 3),
        confidence=torch.ones(1),
        observation_count=torch.ones(1),
        room_min=torch.tensor([-3.0, -2.5, 0.0]),
        room_max=torch.tensor([3.0, 2.5, 3.0]),
        source_voxel_count=1,
        input_voxel_size_m=None,
    )
    with pytest.raises(ValueError, match="oracle or QA"):
        build_refreshing_embodied_runtime(
            config,
            "scene_000001",
            checkpoint="unused",
            chat_runtime=FakeSceneRuntime(config, base_data),
            vision_encoder=CountingDenseEncoder(),
            persistent_map_path=tmp_path / "oracle" / "semantic_map.npz",
            runtime_builder=CountingRuntimeBuilder(),
        )
