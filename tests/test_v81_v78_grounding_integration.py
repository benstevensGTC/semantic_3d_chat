from __future__ import annotations

import os
import subprocess
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch

import scripts.check_v81_scene_memory_demo as demo_check
import semantic_3d_chat.chat.v81_scene_memory_runtime as runtime_module
from semantic_3d_chat.chat.grounding_sidecar_v78_runtime import (
    V78GroundingOutput,
    V78GroundingSidecarRuntime,
)
from semantic_3d_chat.chat.v81_scene_memory_runtime import (
    V81SceneMemoryChatRuntime,
)
from semantic_3d_chat.language.prefix_injection import prefix_sha256
from semantic_3d_chat.language.v81_structured_dense_atlas_sidecar import (
    FIXED_PREFIX_TOKENS,
    HIDDEN_SIZE,
    reconstruct_base_v54_prefix_v81,
)
from semantic_3d_chat.scene_encoder.v81_scene_memory_artifact import (
    LoadedV81SceneMemory,
    build_v81_scene_memory_metadata,
)

ROOT = Path(__file__).parents[1]
BASE_SHA = "a" * 64
RUNTIME_SHA = "b" * 64
CONTROL_SHA = "c" * 64
PROBE_SHA = "d" * 64


def _memory(seed: int = 8178) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    return (
        torch.randn(
            1,
            FIXED_PREFIX_TOKENS,
            HIDDEN_SIZE,
            generator=generator,
            dtype=torch.float32,
        )
        * 0.01
    ).to(torch.bfloat16)


def _loaded(memory: torch.Tensor, root: Path) -> LoadedV81SceneMemory:
    metadata = build_v81_scene_memory_metadata(
        memory,
        scene_id="scene_000001",
        tensor_file_sha256="e" * 64,
        source_base_checkpoint_sha256=BASE_SHA,
        runtime_config_sha256=RUNTIME_SHA,
        source_control_checkpoint_sha256=CONTROL_SHA,
        source_probe_tensor_sha256=PROBE_SHA,
    )
    return LoadedV81SceneMemory(root=root, memory=memory, metadata=metadata)


class _Tokenizer:
    def __call__(self, *_args: object, **_kwargs: object) -> dict[str, torch.Tensor]:
        return {"input_ids": torch.tensor([[2, 3, 4]], dtype=torch.long)}

    def apply_chat_template(self, *_args: object, **_kwargs: object) -> torch.Tensor:
        return torch.tensor([[1, 5, 6]], dtype=torch.long)

    def decode(self, *_args: object, **_kwargs: object) -> str:
        return "right"


class _Backend:
    def __init__(self) -> None:
        self.generation_prefix: torch.Tensor | None = None
        self.controls: torch.Tensor | None = None

    def prepare(
        self,
        scene_prefix: torch.Tensor,
        prompt_ids: torch.Tensor,
        **kwargs: object,
    ) -> SimpleNamespace:
        controls = kwargs["control_tokens"]
        assert isinstance(controls, torch.Tensor)
        self.generation_prefix = scene_prefix
        self.controls = controls
        total = scene_prefix.shape[1] + prompt_ids.shape[1] + controls.shape[1]
        inputs = torch.zeros(1, total, HIDDEN_SIZE, dtype=scene_prefix.dtype)
        inputs[:, -controls.shape[1] :] = controls
        per_layer = torch.zeros_like(inputs)
        return SimpleNamespace(
            scene_prefix_length=scene_prefix.shape[1],
            inputs_embeds=inputs,
            per_layer_inputs=per_layer,
            mm_token_type_ids=torch.zeros(1, total, dtype=torch.long),
        )

    def padding_values(
        self,
        batch_size: int,
        token_count: int,
        *,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        shape = (batch_size, token_count, HIDDEN_SIZE)
        zeros = torch.zeros(shape, dtype=torch.bfloat16, device=device)
        return (
            zeros,
            zeros.clone(),
            torch.zeros(batch_size, token_count, dtype=torch.long, device=device),
        )

    def generate(self, *_args: object, **_kwargs: object) -> torch.Tensor:
        return torch.tensor([[7]], dtype=torch.long)


class _GroundingSidecar:
    def __init__(self, expected_prefix: torch.Tensor) -> None:
        self.expected_prefix = expected_prefix.detach().clone()
        self.expected_hash = prefix_sha256(expected_prefix)
        self.predict_kwargs: dict[str, torch.Tensor] | None = None

    def assert_prefix_unchanged(self, scene_prefix: torch.Tensor) -> None:
        if prefix_sha256(scene_prefix) != self.expected_hash:
            raise RuntimeError("V78 grounding full-scene prefix changed")

    def startup_audit(self) -> dict[str, Any]:
        return {
            "enabled": True,
            "full_prefix_sha256": self.expected_hash,
            "all_scene_tokens_scored": True,
        }

    def predict(
        self,
        _question_embeddings: torch.Tensor,
        **kwargs: torch.Tensor,
    ) -> V78GroundingOutput:
        self.predict_kwargs = kwargs
        self.assert_prefix_unchanged(kwargs["scene_prefix"])
        return V78GroundingOutput(
            xyz_m=(0.25, -0.5, 1.0),
            confidence=0.75,
            support_distance_m=0.125,
            audit={
                "all_scene_tokens_scored": True,
                "minimum_attention_weight": 1e-6,
            },
        )


class _BaseRuntime:
    def __init__(self, scene_prefix: torch.Tensor) -> None:
        self.config = {
            "language": {
                "system_prompt": "Use continuous scene memory.",
                "max_question_tokens": 16,
                "max_answer_tokens": 4,
                "scene_prefix_after_bos": False,
                "scene_boundary_mode": "learned",
            }
        }
        self.scene_id = "scene_000001"
        self.scene_prefix = scene_prefix
        self.scene_prefix_hash = prefix_sha256(scene_prefix)
        embedding = torch.nn.Embedding(16, HIDDEN_SIZE)
        embedding.requires_grad_(False)
        self.backend = _Backend()
        self.language = SimpleNamespace(
            device=torch.device("cpu"),
            tokenizer=_Tokenizer(),
            model=SimpleNamespace(get_input_embeddings=lambda: embedding),
            prefix_backend=self.backend,
        )
        self.map_data = SimpleNamespace(
            xyz=torch.tensor([[0.0, 0.0, 0.0], [1.0, 2.0, 0.5]]),
            confidence=torch.tensor([0.4, 0.9]),
            room_min=torch.tensor([-3.0, -2.5, 0.0]),
            room_max=torch.tensor([3.0, 2.5, 3.0]),
        )

    def assert_prefix_unchanged(self) -> None:
        if prefix_sha256(self.scene_prefix) != self.scene_prefix_hash:
            raise RuntimeError("base prefix changed")

    def startup_summary(self) -> dict[str, Any]:
        return {"base_runtime_ready": True}

    def _eos_token_ids(self) -> int:
        return 1

    def _predict_grounding(
        self, _embeddings: torch.Tensor
    ) -> tuple[tuple[float, float, float], float, float]:
        return (9.0, 9.0, 9.0), 0.1, 9.0


def test_v81_load_binds_v78_to_exact_sealed_base_reconstruction(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    memory = _memory()
    reconstructed = reconstruct_base_v54_prefix_v81(memory)
    base = _BaseRuntime(reconstructed.clone())
    loaded = _loaded(memory, tmp_path / "scene_000001")
    captured: dict[str, Any] = {}

    monkeypatch.setattr(
        runtime_module.StaticChatRuntime,
        "load",
        lambda *_args, **_kwargs: base,
    )
    monkeypatch.setattr(runtime_module, "checkpoint_fingerprint", lambda *_: (BASE_SHA, []))
    monkeypatch.setattr(
        runtime_module,
        "effective_runtime_config_sha256",
        lambda *_: RUNTIME_SHA,
    )
    monkeypatch.setattr(runtime_module, "load_v81_scene_memory", lambda *_args, **_kwargs: loaded)

    def fake_grounding_load(_checkpoint: object, **kwargs: Any) -> _GroundingSidecar:
        captured.update(kwargs)
        return _GroundingSidecar(kwargs["scene_prefix"])

    monkeypatch.setattr(V78GroundingSidecarRuntime, "load", fake_grounding_load)
    config = {
        "_runtime_safe_config": True,
        "language": {"model_id": "google/gemma-4-E2B-it", "revision": "revision"},
    }
    runtime = V81SceneMemoryChatRuntime.load(
        config,
        "scene_000001",
        base_checkpoint=tmp_path / "base",
        scene_memory=tmp_path / "memory",
        grounding_checkpoint=tmp_path / "grounding",
    )

    assert torch.equal(captured["scene_prefix"], reconstructed)
    assert captured["scene_prefix"].data_ptr() != base.scene_prefix.data_ptr()
    assert captured["room_min"] is base.map_data.room_min
    assert captured["room_max"] is base.map_data.room_max
    assert torch.equal(runtime._grounding_scene_prefix, reconstructed)


def test_v81_answer_passes_exact_prefix_and_both_numeric_grounding_map_tensors(
    tmp_path: Path,
) -> None:
    memory = _memory()
    reconstructed = reconstruct_base_v54_prefix_v81(memory)
    base = _BaseRuntime(reconstructed.clone())
    sidecar = _GroundingSidecar(reconstructed)
    runtime = V81SceneMemoryChatRuntime(
        base,  # type: ignore[arg-type]
        _loaded(memory, tmp_path / "scene_000001"),
        grounding_sidecar=sidecar,  # type: ignore[arg-type]
    )

    startup = runtime.startup_summary()
    assert startup["optional_v78_grounding_enabled"] is True
    assert startup["v78_grounding_uses_exact_reconstructed_base_prefix"] is True
    assert startup["v78_grounding_numeric_map_inputs"] == ["xyz", "confidence"]
    assert startup["optional_v78_grounding"]["full_prefix_sha256"] == prefix_sha256(reconstructed)

    answer = runtime.answer("Where is the chair?")

    assert answer.answer == "right"
    assert answer.grounding_xyz_m == (0.25, -0.5, 1.0)
    assert sidecar.predict_kwargs is not None
    assert sidecar.predict_kwargs["scene_prefix"] is runtime._grounding_scene_prefix
    assert torch.equal(sidecar.predict_kwargs["scene_prefix"], reconstructed)
    assert sidecar.predict_kwargs["map_xyz"] is base.map_data.xyz
    assert sidecar.predict_kwargs["map_confidence"] is base.map_data.confidence
    assert base.backend.generation_prefix is runtime.scene_prefix
    assert runtime.last_grounding_audit is not None
    assert runtime.last_grounding_audit["exact_reconstructed_base_scene_prefix"] is True
    assert runtime.last_grounding_audit["full_base_prefix_sha256"] == prefix_sha256(reconstructed)
    assert runtime.last_grounding_audit["numeric_map_inputs"] == ["xyz", "confidence"]
    assert runtime.last_grounding_audit["map_xyz_shape"] == [2, 3]
    assert runtime.last_grounding_audit["map_confidence_shape"] == [2]


def test_v81_v78_binding_fails_closed_when_grounding_prefix_mutates(
    tmp_path: Path,
) -> None:
    memory = _memory()
    reconstructed = reconstruct_base_v54_prefix_v81(memory)
    runtime = V81SceneMemoryChatRuntime(
        _BaseRuntime(reconstructed.clone()),  # type: ignore[arg-type]
        _loaded(memory, tmp_path / "scene_000001"),
        grounding_sidecar=_GroundingSidecar(reconstructed),  # type: ignore[arg-type]
    )
    assert runtime._grounding_scene_prefix is not None
    runtime._grounding_scene_prefix[0, 7, 11] += 1.0

    with pytest.raises(RuntimeError, match="prefix changed"):
        runtime.assert_prefix_unchanged()


def test_v81_model_free_check_authenticates_optional_v78_checkpoint(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    map_path = tmp_path / "maps" / "scene_000001" / "voxel_map.npz"
    map_path.parent.mkdir(parents=True)
    map_path.write_bytes(b"numeric map")
    memory = _memory()
    loaded = _loaded(memory, tmp_path / "memory")
    grounding_path = tmp_path / "grounding"
    captured: dict[str, Any] = {}
    config = {
        "paths": {"maps_root": str(tmp_path / "maps")},
        "language": {"model_id": "model", "revision": "revision"},
    }

    monkeypatch.setattr(demo_check, "load_runtime_config", lambda *_: config)
    monkeypatch.setattr(demo_check, "checkpoint_fingerprint", lambda *_: (BASE_SHA, ["a"]))
    monkeypatch.setattr(demo_check, "effective_runtime_config_sha256", lambda *_: RUNTIME_SHA)
    monkeypatch.setattr(demo_check, "load_v81_scene_memory", lambda *_args, **_kwargs: loaded)

    def fake_authenticate(checkpoint: object, **kwargs: Any) -> dict[str, Any]:
        captured["checkpoint"] = checkpoint
        captured.update(kwargs)
        return {"passed": True, "gemma_model_loaded": False}

    monkeypatch.setattr(demo_check, "authenticate_v78_grounding_checkpoint", fake_authenticate)
    report = demo_check.validate_v81_scene_memory_demo_inputs(
        config_path=tmp_path / "runtime.yaml",
        scene_id="scene_000001",
        base_checkpoint=tmp_path / "base",
        scene_memory=tmp_path / "memory",
        grounding_checkpoint=grounding_path,
    )

    assert report["passed"] is True
    assert report["gemma_loaded"] is False
    assert report["optional_v78_grounding_supplied"] is True
    assert report["optional_v78_grounding_checkpoint_authentication"] == {
        "passed": True,
        "gemma_model_loaded": False,
    }
    assert captured == {
        "checkpoint": grounding_path,
        "base_checkpoint_sha256": BASE_SHA,
        "base_runtime_config_sha256": RUNTIME_SHA,
        "model_id": "model",
        "model_revision": "revision",
    }


def test_v81_launcher_derives_every_report_path_after_scene_selection(
    tmp_path: Path,
) -> None:
    fake_python = tmp_path / "python"
    invocation_log = tmp_path / "invocations.log"
    fake_python.write_text(
        '#!/usr/bin/env bash\nprintf "%s\\n" "$*" >> "$V81_DEMO_TEST_INVOCATIONS"\n',
        encoding="utf-8",
    )
    fake_python.chmod(0o755)
    environment = dict(os.environ)
    for name in (
        "V81_DEMO_AUDIT_LOG",
        "V81_DEMO_CHAT_LOG",
        "V81_DEMO_LEAKAGE_REPORT",
        "V81_DEMO_COMPILE_AUDIT",
    ):
        environment.pop(name, None)
    environment.update(
        {
            "V81_DEMO_PYTHON": str(fake_python),
            "V81_DEMO_TEST_INVOCATIONS": str(invocation_log),
        }
    )
    launcher = ROOT / "scripts" / "run_v81_scene_memory_demo.sh"

    subprocess.run(
        [str(launcher), "--scene", "scene_000031", "--question", "numeric?"],
        cwd=ROOT,
        env=environment,
        check=True,
    )
    subprocess.run(
        [str(launcher), "--leakage", "--scene", "scene_000031"],
        cwd=ROOT,
        env=environment,
        check=True,
    )
    missing_memory = tmp_path / "missing-memory"
    subprocess.run(
        [
            str(launcher),
            "--compile",
            "--scene",
            "scene_000031",
            "--scene-memory",
            str(missing_memory),
        ],
        cwd=ROOT,
        env=environment,
        check=True,
    )
    subprocess.run(
        [str(launcher), "--verify-compile", "--scene", "scene_000031"],
        cwd=ROOT,
        env=environment,
        check=True,
    )

    invocations = invocation_log.read_text(encoding="utf-8").splitlines()
    assert any(
        "--audit-log reports/gemma4/metrics/v81_chat_access_scene_000031.json" in line
        and "--chat-log reports/gemma4/examples/v81_chat_scene_000031.jsonl" in line
        for line in invocations
    )
    assert any(
        "--output reports/gemma4/metrics/v81_scene_memory_leakage_scene_000031.json" in line
        for line in invocations
    )
    assert any(
        "--audit-report reports/gemma4/metrics/v81_compile_access_scene_000031.json" in line
        for line in invocations
    )
    assert any(
        "--verify-existing" in line
        and "--audit-report reports/gemma4/metrics/v81_compile_access_scene_000031.json" in line
        for line in invocations
    )
    check_calls = [line for line in invocations if "check_v81_scene_memory_demo.py" in line]
    assert check_calls
    assert all("--grounding-checkpoint" in line for line in check_calls)
