from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import torch

from semantic_3d_chat.evaluation.control_predict import (
    BuiltControlRuntime,
    SharedControlRuntimeFactory,
    _zero_runtime_scene_memory,
    apply_map_control,
    deterministic_wrong_scene_sources,
    run_control_suite,
)
from semantic_3d_chat.evaluation.prediction_artifacts import PredictionProvenance
from semantic_3d_chat.evaluation.question_manifest import build_question_manifest
from semantic_3d_chat.language.prefix_injection import prefix_sha256
from semantic_3d_chat.scene_encoder.map_io import MapTensorData
from semantic_3d_chat.scene_encoder.projector import SceneTokenizerOutput


def _map_data() -> MapTensorData:
    count = 7
    return MapTensorData(
        semantic=torch.arange(count * 5, dtype=torch.float32).reshape(count, 5),
        xyz=torch.arange(count * 3, dtype=torch.float32).reshape(count, 3) / 10,
        rgb=torch.linspace(0.0, 1.0, count * 3).reshape(count, 3),
        normal=torch.eye(3).repeat(3, 1)[:count],
        confidence=torch.linspace(0.4, 1.0, count),
        observation_count=torch.arange(1, count + 1, dtype=torch.float32),
        room_min=torch.tensor([-3.0, -2.5, 0.0]),
        room_max=torch.tensor([3.0, 2.5, 3.0]),
        source_voxel_count=70,
        input_voxel_size_m=0.15,
    )


def test_map_controls_change_only_declared_inputs_and_keep_all_rows() -> None:
    source = _map_data()
    original = _map_data()

    semantic, semantic_meta = apply_map_control(
        source, "semantic_shuffle", seed=17, scene_id="scene_000001"
    )
    assert semantic.voxel_count == source.voxel_count
    assert torch.equal(semantic.xyz, source.xyz)
    assert not torch.equal(semantic.semantic, source.semantic)
    assert sorted(semantic.semantic[:, 0].tolist()) == sorted(source.semantic[:, 0].tolist())
    assert semantic_meta["permutation_sha256"]

    positions, _ = apply_map_control(
        source, "position_shuffle", seed=17, scene_id="scene_000001"
    )
    assert torch.equal(positions.semantic, source.semantic)
    assert not torch.equal(positions.xyz, source.xyz)
    assert sorted(positions.xyz[:, 0].tolist()) == sorted(source.xyz[:, 0].tolist())

    geometry, _ = apply_map_control(
        source, "geometry_only", seed=17, scene_id="scene_000001"
    )
    assert torch.count_nonzero(geometry.semantic) == 0
    assert torch.equal(geometry.xyz, source.xyz)
    assert torch.equal(geometry.rgb, source.rgb)
    assert torch.equal(geometry.normal, source.normal)

    no_xyz, _ = apply_map_control(
        source, "semantics_without_xyz", seed=17, scene_id="scene_000001"
    )
    expected_center = (source.room_min + source.room_max) * 0.5
    assert torch.equal(no_xyz.semantic, source.semantic)
    assert torch.allclose(no_xyz.xyz, expected_center.expand_as(no_xyz.xyz))

    no_rgb, _ = apply_map_control(
        source, "remove_rgb", seed=17, scene_id="scene_000001"
    )
    no_normals, _ = apply_map_control(
        source, "remove_normals", seed=17, scene_id="scene_000001"
    )
    assert torch.count_nonzero(no_rgb.rgb) == 0
    assert torch.count_nonzero(no_normals.normal) == 0

    # Controls clone their tensors; none may corrupt the cached primary map.
    assert torch.equal(source.semantic, original.semantic)
    assert torch.equal(source.xyz, original.xyz)
    assert torch.equal(source.rgb, original.rgb)
    assert torch.equal(source.normal, original.normal)


def test_wrong_scene_sources_are_a_deterministic_derangement() -> None:
    scenes = ["scene_000003", "scene_000001", "scene_000002"]
    first = deterministic_wrong_scene_sources(scenes)
    second = deterministic_wrong_scene_sources(reversed(scenes))
    assert first == second
    assert set(first) == set(scenes)
    assert set(first.values()) == set(scenes)
    assert all(target != source for target, source in first.items())


def test_shared_control_factory_preserves_complete_trained_scene_stack(
    monkeypatch,
) -> None:
    from semantic_3d_chat.evaluation import control_predict

    dense = object()
    sidecar_adapter = object()
    block_cross_residual = object()
    global_residual = object()
    signed_residual = object()
    captured: dict[str, object] = {}

    class FakeStaticRuntime:
        @classmethod
        def load(cls, config, scene_id, checkpoint, local_files_only):
            del config, scene_id, checkpoint, local_files_only
            runtime = SimpleNamespace(
                checkpoint_path=Path("/checkpoint"),
                checkpoint_metadata={"semantic_dim": 5},
                language=object(),
                scene_model=object(),
                dense_aligner=dense,
                dense_sidecar_adapter=sidecar_adapter,
                block_cross_residual=block_cross_residual,
                global_scene_residual=global_residual,
                signed_x_scene_residual=signed_residual,
                composer=object(),
                grounding=object(),
                warnings=[],
                map_data=_map_data(),
                scene_prefix=torch.zeros(1, 4, 3),
            )
            runtime.scene_prefix_hash = prefix_sha256(runtime.scene_prefix)
            runtime.assert_prefix_unchanged = lambda: None
            return runtime

        def __init__(self, **kwargs):
            captured.update(kwargs)
            self.map_data = kwargs["map_data"]
            self.scene_prefix = torch.zeros(1, 4, 3)
            self.scene_prefix_hash = prefix_sha256(self.scene_prefix)

        def assert_prefix_unchanged(self) -> None:
            return None

    monkeypatch.setattr(control_predict, "StaticChatRuntime", FakeStaticRuntime)
    factory = SharedControlRuntimeFactory(
        {"scene": {"room_size_m": [6.0, 5.0, 3.0]}, "scene_encoder": {}},
        "/checkpoint",
        "scene_000001",
    )
    monkeypatch.setattr(factory, "_load_map", lambda _scene_id: _map_data())

    built = factory.build("primary", "scene_000002", "scene_000002")

    assert built.runtime is not None
    assert captured["dense_aligner"] is dense
    assert captured["dense_sidecar_adapter"] is sidecar_adapter
    assert captured["block_cross_residual"] is block_cross_residual
    assert captured["global_scene_residual"] is global_residual
    assert captured["signed_x_scene_residual"] is signed_residual


def test_zero_prefix_control_removes_language_and_grounding_scene_signal() -> None:
    prefix = torch.randn(1, 6, 8)
    start = prefix[:, :1].clone()
    end = prefix[:, -1:].clone()
    runtime = SimpleNamespace(
        scene_prefix=prefix,
        scene_prefix_hash=prefix_sha256(prefix),
        scene_output=SceneTokenizerOutput(
            scene_tokens=torch.randn(1, 4, 8),
            native_latents=torch.randn(1, 4, 6),
            block_tokens=torch.randn(3, 6),
            audit={"processed_voxels": torch.tensor(12)},
        ),
    )
    _zero_runtime_scene_memory(runtime)
    assert torch.equal(runtime.scene_prefix[:, :1], start)
    assert torch.equal(runtime.scene_prefix[:, -1:], end)
    assert torch.count_nonzero(runtime.scene_prefix[:, 1:-1]) == 0
    assert torch.count_nonzero(runtime.scene_output.scene_tokens) == 0
    assert torch.count_nonzero(runtime.scene_output.native_latents) == 0
    assert torch.count_nonzero(runtime.scene_output.block_tokens) == 0
    assert runtime.scene_prefix_hash == prefix_sha256(runtime.scene_prefix)


class _FakeRuntime:
    def __init__(self, prefix_key: str, events: list[str]) -> None:
        digest = torch.tensor(list(prefix_key.encode()), dtype=torch.float32).sum()
        self.scene_prefix = torch.full((1, 4, 3), digest.item())
        self.scene_prefix_hash = prefix_sha256(self.scene_prefix)
        self._initial_hash = self.scene_prefix_hash
        self._events = events

    def assert_prefix_unchanged(self) -> None:
        assert prefix_sha256(self.scene_prefix) == self._initial_hash

    def answer(self, question: str):
        self._events.append(f"answer:{question}")
        self.assert_prefix_unchanged()
        return SimpleNamespace(
            answer="yes",
            grounding_xyz_m=(1.0, 2.0, 0.5),
            grounding_confidence=0.75,
            prefix_hash=self.scene_prefix_hash,
            generated_tokens=1,
            elapsed_seconds=0.01,
        )


def test_control_runner_builds_one_prefix_before_questions_and_writes_metric_jsonl(
    tmp_path: Path,
) -> None:
    questions = [
        {
            "scene_id": scene,
            "question_id": f"q_{index:06d}",
            "question": f"question {index}",
        }
        for scene in ("scene_000001", "scene_000002")
        for index in range(2)
    ]
    question_manifest = build_question_manifest(
        questions, source_qa_sha256="0" * 64
    )
    events: list[str] = []
    builds: list[tuple[str, str, str]] = []

    def builder(condition: str, target: str, source: str) -> BuiltControlRuntime:
        builds.append((condition, target, source))
        events.append(f"build:{condition}:{target}:{source}")
        return BuiltControlRuntime(
            runtime=_FakeRuntime(f"{condition}:{source}", events),
            prefix_source_scene_id=source,
            metadata={"question_dependent_selection": False},
        )

    result = run_control_suite(
        question_manifest,
        runtime_builder=builder,
        output_directory=tmp_path / "controls",
        conditions=("primary", "empty_scene_prefix", "wrong_scene_prefix"),
    )

    assert len(builds) == 3 * 2
    for build_index, event in enumerate(events):
        if not event.startswith("build:"):
            continue
        next_build = next(
            (index for index in range(build_index + 1, len(events)) if events[index].startswith("build:")),
            len(events),
        )
        assert any(item.startswith("answer:") for item in events[build_index + 1 : next_build])

    wrong_builds = [item for item in builds if item[0] == "wrong_scene_prefix"]
    assert wrong_builds == [
        ("wrong_scene_prefix", "scene_000001", "scene_000002"),
        ("wrong_scene_prefix", "scene_000002", "scene_000001"),
    ]
    assert result["one_prefix_per_scene_condition"] is True
    assert result["question_dependent_retrieval"] is False

    for condition in ("primary", "empty_scene_prefix", "wrong_scene_prefix"):
        path = tmp_path / "controls" / f"{condition}.jsonl"
        records = [json.loads(line) for line in path.read_text().splitlines()]
        assert len(records) == 4
        assert all(record["condition"] == condition for record in records)
        assert all("predicted_answer" in record for record in records)
        assert all("grounding_xyz" in record for record in records)
        assert all("answer" not in record for record in records)
        for scene_id in ("scene_000001", "scene_000002"):
            hashes = {r["prefix_hash"] for r in records if r["scene_id"] == scene_id}
            assert len(hashes) == 1


def test_control_runner_resumes_only_hash_compatible_question_records(tmp_path: Path) -> None:
    question_manifest = build_question_manifest(
        [
            {
                "scene_id": "scene_000001",
                "question_id": "q_000001",
                "question": "Is anything present?",
            }
        ],
        source_qa_sha256="1" * 64,
    )
    provenance = PredictionProvenance(
        config_path="/config.yaml",
        config_sha256="2" * 64,
        config_file_sha256="3" * 64,
        checkpoint_path="/checkpoint",
        checkpoint_sha256="4" * 64,
        checkpoint_files=(),
        references_path="/questions.json",
        references_sha256="5" * 64,
        scene_map_manifest_sha256="6" * 64,
        scene_map_manifest={
            "scene_000001": {
                "voxel_map_sha256": "7" * 64,
                "voxel_map_size_bytes": 1,
            }
        },
        split="test",
        run_kind="continuous_scene_control",
        condition="primary",
    )
    events: list[str] = []

    def builder(condition: str, target: str, source: str) -> BuiltControlRuntime:
        events.append("build")
        return BuiltControlRuntime(
            runtime=_FakeRuntime(f"{condition}:{source}", events),
            prefix_source_scene_id=source,
            metadata={"question_dependent_selection": False},
        )

    first = run_control_suite(
        question_manifest,
        runtime_builder=builder,
        output_directory=tmp_path / "controls",
        conditions=("primary",),
        prediction_provenance={"primary": provenance},
    )
    second = run_control_suite(
        question_manifest,
        runtime_builder=builder,
        output_directory=tmp_path / "controls",
        conditions=("primary",),
        prediction_provenance={"primary": provenance},
    )

    assert first["conditions"]["primary"]["new_prediction_count"] == 1
    assert second["conditions"]["primary"]["resumed_prediction_count"] == 1
    assert second["conditions"]["primary"]["new_prediction_count"] == 0
    assert events.count("answer:Is anything present?") == 1
