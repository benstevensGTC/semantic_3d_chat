from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch
from torch import nn

from semantic_3d_chat.language.gemma4_tool_decoder_v2 import (
    CLEARANCE_TOKEN_COUNT,
    HIDDEN_SIZE,
    LORA_ALPHA,
    LORA_PARAMETER_COUNT,
    LORA_RANK,
    MODEL_ID,
    MODEL_REVISION,
    PROJECTOR_PARAMETER_COUNT,
    TARGET_PROJECTION,
    TARGET_TOKEN_COUNT,
    TOTAL_TRAINABLE_PARAMETER_COUNT,
)
from semantic_3d_chat.language.gemma4_tool_decoder_v2_checkpoint import (
    ARCHITECTURE,
    METADATA_FILENAME,
    TRAINING_STATUS,
    WEIGHTS_FILENAME,
)
from semantic_3d_chat.language.lora import LoRALinear, LoRASettings, install_lora_adapters
from semantic_3d_chat.robot import conversation_cli
from semantic_3d_chat.robot import gemma4_tool_decoder_v2_integration as integration
from semantic_3d_chat.robot.gemma4_tool_decoder_v2_integration import (
    ScopedGemmaToolDecoderBackendV2,
    inspect_promoted_gemma_tool_decoder_v2,
    load_promoted_gemma_tool_decoder_v2,
)
from semantic_3d_chat.robot.llm_tool_policy import (
    GeneratedToolProposal,
    LocalGemmaToolPolicy,
)
from semantic_3d_chat.robot.navigation_policy import ACTION_NAMES


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _promoted_checkpoint(tmp_path: Path) -> tuple[Path, Path, dict[str, Any]]:
    base = tmp_path / "numeric_base"
    base.mkdir()
    base_adapter = base / "adapter.safetensors"
    base_adapter.write_bytes(b"frozen V54 adapter fixture")

    checkpoint = tmp_path / "tool_decoder"
    checkpoint.mkdir()
    weights = checkpoint / WEIGHTS_FILENAME
    weights.write_bytes(b"trained tool decoder fixture")
    digest = "1" * 64
    metadata: dict[str, Any] = {
        "schema_version": 2,
        "architecture": ARCHITECTURE,
        "training_status": TRAINING_STATUS,
        "status": "promoted_runtime",
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "base_checkpoint_sha256": _sha256(base_adapter),
        "preregistration_sha256": digest,
        "cpu_preflight_sha256": digest,
        "training_authorization_sha256": digest,
        "clearance_cache_sha256": digest,
        "prefix_inventory_sha256": digest,
        "target_module": TARGET_PROJECTION,
        "lora_rank": LORA_RANK,
        "lora_alpha": LORA_ALPHA,
        "lora_parameter_count": LORA_PARAMETER_COUNT,
        "numeric_projector_parameter_count": PROJECTOR_PARAMETER_COUNT,
        "total_trainable_parameter_count": TOTAL_TRAINABLE_PARAMETER_COUNT,
        "lora_state_sha256": digest,
        "numeric_projector_state_sha256": digest,
        "weights_sha256": _sha256(weights),
        "scene_prefix_tokens": 258,
        "robot_tokens": 4,
        "target_tokens": TARGET_TOKEN_COUNT,
        "clearance_tokens": CLEARANCE_TOKEN_COUNT,
        "hidden_size": HIDDEN_SIZE,
        "max_new_tokens": 24,
        "tool_vocabulary": list(ACTION_NAMES),
        "task_trained": True,
        "promotion_gates_passed": True,
        "saved_runtime_execution_gate_passed": True,
        "complete_scene_prefix_required": True,
        "question_independent_static_scene_prefix_required": True,
        "numeric_robot_tokens_required": True,
        "continuous_target_tokens_required": True,
        "numeric_clearance_tokens_required": True,
        "all_map_voxels_scored_for_target_grounding": True,
        "collision_interlock_required": True,
        "environmental_text_inputs": [],
        "oracle_inputs_at_runtime": False,
        "runtime_required_files": [WEIGHTS_FILENAME, METADATA_FILENAME],
    }
    (checkpoint / METADATA_FILENAME).write_text(
        json.dumps(metadata, sort_keys=True),
        encoding="utf-8",
    )
    return checkpoint, base, metadata


def _proposal(text: str = '{"tool":"stop","arguments":{}}') -> GeneratedToolProposal:
    return GeneratedToolProposal(
        text=text,
        active_prefix_sha256="a" * 64,
        scene_prefix_sha256="b" * 64,
        robot_tokens_sha256="c" * 64,
        local_inference=True,
        used_continuous_scene_prefix=True,
        used_continuous_robot_tokens=True,
        training_status=TRAINING_STATUS,
    )


def test_promoted_checkpoint_inspection_fails_closed_on_gate_or_hash_change(
    tmp_path: Path,
) -> None:
    checkpoint, base, metadata = _promoted_checkpoint(tmp_path)
    assert inspect_promoted_gemma_tool_decoder_v2(
        checkpoint,
        base_checkpoint=base,
    ) == metadata

    missing = dict(metadata)
    missing.pop("lora_rank")
    (checkpoint / METADATA_FILENAME).write_text(json.dumps(missing), encoding="utf-8")
    with pytest.raises(ValueError, match="unpromoted"):
        inspect_promoted_gemma_tool_decoder_v2(checkpoint, base_checkpoint=base)

    metadata["saved_runtime_execution_gate_passed"] = False
    (checkpoint / METADATA_FILENAME).write_text(json.dumps(metadata), encoding="utf-8")
    with pytest.raises(ValueError, match="unpromoted"):
        inspect_promoted_gemma_tool_decoder_v2(checkpoint, base_checkpoint=base)

    metadata["saved_runtime_execution_gate_passed"] = True
    (checkpoint / METADATA_FILENAME).write_text(json.dumps(metadata), encoding="utf-8")
    (checkpoint / WEIGHTS_FILENAME).write_bytes(b"tampered")
    with pytest.raises(ValueError, match="weights hash changed"):
        inspect_promoted_gemma_tool_decoder_v2(checkpoint, base_checkpoint=base)


class _TinyDecoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.decoder = nn.Module()
        self.decoder.projection = nn.Linear(3, 2, bias=False)


class _ScaleCheckingBackend:
    def __init__(self, model: _TinyDecoder, *, fail: bool = False) -> None:
        self.model = model
        self.fail = fail
        self.last_context: dict[str, Any] | None = None
        self.active_output: torch.Tensor | None = None

    def generate(
        self,
        instruction: str,
        *,
        correction_code: str | None,
    ) -> GeneratedToolProposal:
        del instruction, correction_code
        assert isinstance(self.model.decoder.projection, LoRALinear)
        assert self.model.decoder.projection.scaling > 0.0
        self.active_output = self.model.decoder.projection(torch.ones(1, 3))
        if self.fail:
            raise RuntimeError("generation failed")
        self.last_context = {"target_available": False, "scored_voxels": None}
        return _proposal()


@pytest.mark.parametrize("fail", [False, True])
def test_scoped_tool_lora_preserves_static_qa_exactly_even_after_failure(
    monkeypatch: pytest.MonkeyPatch,
    fail: bool,
) -> None:
    model = _TinyDecoder().eval()
    inputs = torch.ones(1, 3)
    baseline = model.decoder.projection(inputs).detach().clone()
    settings = LoRASettings(
        enabled=True,
        rank=1,
        alpha=2.0,
        dropout=0.0,
        target_modules=("decoder.projection",),
    )
    installation = install_lora_adapters(model, settings)
    assert installation is not None
    original = installation.adapters[0].base
    adapter_calls = 0

    def record_adapter_call(
        module: nn.Module,
        inputs: tuple[torch.Tensor, ...],
        output: torch.Tensor,
    ) -> None:
        nonlocal adapter_calls
        del module, inputs, output
        adapter_calls += 1

    installation.adapters[0].register_forward_hook(record_adapter_call)
    with torch.no_grad():
        installation.adapters[0].lora_a.fill_(1.0)
        installation.adapters[0].lora_b.fill_(0.5)
    monkeypatch.setattr(integration, "TARGET_PROJECTION", "decoder.projection")
    delegate = _ScaleCheckingBackend(model, fail=fail)
    scoped = ScopedGemmaToolDecoderBackendV2(  # type: ignore[arg-type]
        delegate,
        installation,
        model,
    )

    assert model.decoder.projection is original
    outside = model.decoder.projection(inputs)
    assert outside.detach().numpy().tobytes() == baseline.numpy().tobytes()
    assert adapter_calls == 0
    if fail:
        with pytest.raises(RuntimeError, match="generation failed"):
            scoped.generate("stop", correction_code=None)
    else:
        assert scoped.generate("stop", correction_code=None) == _proposal()
    assert delegate.active_output is not None
    assert not torch.equal(delegate.active_output, baseline)
    assert adapter_calls == 1
    assert model.decoder.projection is original
    after = model.decoder.projection(inputs)
    assert after.detach().numpy().tobytes() == baseline.numpy().tobytes()
    assert adapter_calls == 1


class _TinyGemma(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = nn.Module()
        self.model.language_model = nn.Module()
        layers = [nn.Module() for _ in range(35)]
        self.model.language_model.layers = nn.ModuleList(layers)
        layers[34].mlp = nn.Module()
        layers[34].mlp.down_proj = nn.Linear(3, 2, bias=False)


def _fake_runtime(base: Path) -> tuple[SimpleNamespace, _TinyGemma]:
    model = _TinyGemma()
    language = SimpleNamespace(
        backend_name="gemma4",
        prefix_backend=object(),
        hidden_size=HIDDEN_SIZE,
        model=model,
        device=torch.device("cpu"),
    )
    static = SimpleNamespace(
        language=language,
        config={
            "language": {
                "model_id": MODEL_ID,
                "revision": MODEL_REVISION,
            }
        },
        checkpoint_path=base,
    )
    runtime = SimpleNamespace(prefix_refresher=SimpleNamespace(runtime=static))
    return runtime, model


def test_production_loader_installs_only_after_authentication_and_deactivates(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    checkpoint, base, metadata = _promoted_checkpoint(tmp_path)
    runtime, model = _fake_runtime(base)
    original = model.get_submodule(TARGET_PROJECTION)
    calls: dict[str, Any] = {}

    monkeypatch.setattr(
        integration,
        "validate_decoder_surface_v2",
        lambda loaded: loaded.get_submodule(TARGET_PROJECTION),
    )

    def load_checkpoint(
        path: str | Path,
        installation: Any,
        projector: Any,
        *,
        expected_provenance: dict[str, str],
        require_promoted: bool,
    ) -> dict[str, Any]:
        calls.update(
            path=path,
            installation=installation,
            projector=projector,
            provenance=expected_provenance,
            require_promoted=require_promoted,
        )
        return metadata

    class FakeBackend:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            calls["backend"] = (args, kwargs)
            self.last_context = None

    monkeypatch.setattr(integration, "load_runtime_checkpoint_v2", load_checkpoint)
    monkeypatch.setattr(integration, "ContinuousGemmaToolDecoderBackendV2", FakeBackend)
    scoped, loaded = load_promoted_gemma_tool_decoder_v2(
        runtime,
        {},
        checkpoint,
        text_encoder=SimpleNamespace(output_dim=HIDDEN_SIZE),
    )

    assert loaded == metadata
    assert calls["require_promoted"] is True
    assert calls["provenance"]["base_checkpoint_sha256"] == metadata[
        "base_checkpoint_sha256"
    ]
    installed = model.get_submodule(TARGET_PROJECTION)
    assert installed is original
    assert isinstance(scoped.installation.adapters[0], LoRALinear)
    assert scoped.installation.adapters[0].base is original
    assert scoped.installation.parameters()
    assert all(not parameter.requires_grad for parameter in model.parameters())
    assert all(not parameter.requires_grad for parameter in scoped.installation.parameters())


def test_production_loader_restores_original_projection_when_strict_load_fails(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    checkpoint, base, _metadata = _promoted_checkpoint(tmp_path)
    runtime, model = _fake_runtime(base)
    original = model.get_submodule(TARGET_PROJECTION)
    monkeypatch.setattr(
        integration,
        "validate_decoder_surface_v2",
        lambda loaded: loaded.get_submodule(TARGET_PROJECTION),
    )

    def reject(*args: Any, **kwargs: Any) -> dict[str, Any]:
        del args, kwargs
        raise ValueError("strict checkpoint rejection")

    monkeypatch.setattr(integration, "load_runtime_checkpoint_v2", reject)
    with pytest.raises(ValueError, match="strict checkpoint rejection"):
        load_promoted_gemma_tool_decoder_v2(runtime, {}, checkpoint)
    assert model.get_submodule(TARGET_PROJECTION) is original


def test_policy_accepts_promoted_gemma_training_attestation() -> None:
    assert LocalGemmaToolPolicy._context_error(_proposal()) is None


def test_cli_exposes_mutually_exclusive_promoted_decoder_and_reports_it() -> None:
    required = [
        "--base-checkpoint",
        "numeric_base",
        "--runtime-asset",
        "opaque.blend",
        "--robot-state-checkpoint",
        "numeric_robot",
    ]
    parsed = conversation_cli._parser().parse_args(
        [*required, "--gemma-tool-decoder-checkpoint", "promoted_tool_decoder"]
    )
    assert parsed.gemma_tool_decoder_checkpoint == "promoted_tool_decoder"
    with pytest.raises(SystemExit):
        conversation_cli._parser().parse_args(
            [
                *required,
                "--gemma-tool-decoder-checkpoint",
                "promoted_tool_decoder",
                "--navigation-policy-checkpoint",
                "navigation",
            ]
        )

    base_runtime = SimpleNamespace(startup_summary=lambda: {"device": "cpu"})
    runtime = SimpleNamespace(
        prefix_refresher=SimpleNamespace(runtime=base_runtime),
        prefix_binding=lambda: {"active_prefix_sha256": "a" * 64},
    )
    startup = conversation_cli._startup(
        runtime,
        "scene_000031",
        gemma_tool_decoder_checkpoint="promoted_tool_decoder",
        gemma_tool_decoder_metadata={
            "training_status": TRAINING_STATUS,
            "task_trained": True,
            "status": "promoted_runtime",
            "saved_runtime_execution_gate_passed": True,
            "complete_scene_prefix_required": True,
            "question_independent_static_scene_prefix_required": True,
            "numeric_robot_tokens_required": True,
            "continuous_target_tokens_required": True,
            "numeric_clearance_tokens_required": True,
            "collision_interlock_required": True,
            "environmental_text_inputs": [],
            "oracle_inputs_at_runtime": False,
        },
    )
    assert startup["llm_tool_policy"]["backend"] == "trained_gemma_tool_decoder_v2"
    assert startup["llm_tool_policy"]["training_status"] == TRAINING_STATUS
    assert startup["gemma_tool_decoder"]["promoted_runtime"] is True
    assert startup["learned_navigation_closed_loop"]["enabled"] is True


def test_cli_preflight_authenticates_decoder_without_loading_gemma(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    metadata = {
        "training_status": TRAINING_STATUS,
        "status": "promoted_runtime",
        "task_trained": True,
    }
    calls: dict[str, Any] = {}

    def inspect(path: str | Path, **kwargs: Any) -> dict[str, Any]:
        calls["path"] = path
        calls.update(kwargs)
        return metadata

    monkeypatch.setattr(
        conversation_cli,
        "inspect_promoted_gemma_tool_decoder_v2",
        inspect,
    )
    config = {
        "scene_encoder": {"language_aligned_tail_dim": HIDDEN_SIZE},
        "language": {"model_id": MODEL_ID, "revision": MODEL_REVISION},
    }
    audit = conversation_cli._runtime_file_audit()
    result = conversation_cli._preflight(
        config,
        None,
        3,
        gemma_tool_decoder_checkpoint="promoted_tool_decoder",
        base_checkpoint="numeric_base",
        audit=audit,
    )
    assert result["ready"] is True
    assert result["loads_language_model"] is False
    assert result["gemma_tool_decoder"]["promoted_runtime"] is True
    assert calls["expected_model_id"] == MODEL_ID
    assert calls["expected_model_revision"] == MODEL_REVISION
    assert calls["base_checkpoint"] == "numeric_base"
    assert calls["audit"] is audit

    config["scene_encoder"]["language_aligned_tail_dim"] = 1024
    with pytest.raises(ValueError, match="1536-wide"):
        conversation_cli._preflight(
            config,
            None,
            3,
            gemma_tool_decoder_checkpoint="promoted_tool_decoder",
            base_checkpoint="numeric_base",
            audit=audit,
        )
