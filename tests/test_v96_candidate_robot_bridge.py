from __future__ import annotations

import ast
import asyncio
import inspect
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest
import torch

import semantic_3d_chat.robot.runtime_refresh as refresh
from semantic_3d_chat.chat import v96_explicit_candidate_cli as candidate_cli
from semantic_3d_chat.chat.file_audit import FileAccessAudit
from semantic_3d_chat.evaluation import v96_candidate_mcp_live_smoke as live_smoke
from semantic_3d_chat.language.prefix_injection import prefix_sha256
from semantic_3d_chat.mcp_server import server as mcp_server
from semantic_3d_chat.robot import semantic_mapping
from semantic_3d_chat.robot import v96_candidate_refresh as bridge
from semantic_3d_chat.scene_encoder.fixed_prefix_atlas import tensor_sha256
from semantic_3d_chat.scene_encoder.question_control_v75 import (
    DenseFullSceneContinuousControlV75,
)


def _digest(character: str) -> str:
    return character * 64


def _promoted_release_receipt() -> dict[str, Any]:
    return {
        "phase": "v96_strict_runtime_release_verified",
        "passed": True,
        "check_count": len(bridge._REQUIRED_RELEASE_CHECKS),
        "deferred_final_binding_exact": True,
        "runtime_smoke_binding_exact": True,
        "promoted_runtime_release_verified": True,
        "candidate_fingerprint_sha256": _digest("1"),
        "deferred_final_evidence_sha256": _digest("2"),
        "runtime_smoke_sha256": _digest("3"),
        "release_checkpoint_sha256": _digest("4"),
        "release_adapter_sha256": _digest("5"),
        "v95_state_sha256": _digest("6"),
        "v96_state_sha256": _digest("7"),
        "runtime_implementation_inventory_sha256": _digest("8"),
        "scene_ids": list(live_smoke.SCENE_IDS),
    }


class _FixedMemoryRuntime:
    def __init__(self) -> None:
        base_prefix = torch.arange(32, dtype=torch.float32).reshape(1, 4, 8)
        full_memory = torch.arange(48, dtype=torch.float32).reshape(1, 6, 8)
        self.base = SimpleNamespace(
            scene_prefix=base_prefix,
            scene_prefix_hash=prefix_sha256(base_prefix),
            current_prefix_hash=lambda: prefix_sha256(base_prefix),
        )
        self.scene_prefix = base_prefix.detach().clone()
        self.base_scene_prefix_hash = prefix_sha256(base_prefix)
        self.fixed_scene_memory = full_memory
        self.scene_prefix_hash = prefix_sha256(full_memory)

    def current_prefix_hash(self) -> str:
        return prefix_sha256(self.fixed_scene_memory)

    def assert_prefix_unchanged(self) -> None:
        if self.current_prefix_hash() != self.scene_prefix_hash:
            raise RuntimeError("synthetic fixed memory changed")


def test_fixed_memory_is_bound_as_numeric_mcp_control_signature() -> None:
    runtime = _FixedMemoryRuntime()

    assert refresh._static_base(runtime) is runtime.base
    assert refresh._control_signature_sha256(
        runtime, verify_against_prefix=True
    ) == prefix_sha256(runtime.fixed_scene_memory)


def test_fixed_memory_refuses_legacy_258_token_default_rebuilder() -> None:
    runtime = _FixedMemoryRuntime()

    with pytest.raises(RuntimeError, match="question-free memory compiler"):
        refresh._rebuild_runtime(runtime, cast(Any, object()))


def test_fixed_memory_signature_fails_closed_after_tensor_mutation() -> None:
    runtime = _FixedMemoryRuntime()
    runtime.fixed_scene_memory[0, 0, 0] += 1

    with pytest.raises(RuntimeError, match="fixed memory changed"):
        refresh._control_signature_sha256(runtime)


def test_v96_bridge_hook_is_explicit_numeric_only_and_unpromoted() -> None:
    hook = bridge.load_v96_candidate_mcp_hook(
        "configs/runtime/gemma4_v96_explicit_candidate_mcp_bridge.yaml"
    )

    assert hook.candidate_hook.name == "gemma4_v96_explicit_candidate_hook.yaml"
    assert hook.atlas_control_checkpoint.name == "gemma4_v75_nll_control_release_v1"
    assert hook.atlas_probe_bank.name == "probe_bank"


def test_v96_embodied_builder_requires_explicit_candidate_before_authentication() -> None:
    with pytest.raises(ValueError, match="explicit acknowledgement"):
        bridge.V96CandidateRuntimeBuilder(
            cast(Any, object()),
            cast(Any, object()),
            cast(Any, object()),
            allow_explicit_candidate=False,
        )


def _candidate_server_arguments(*extra: str) -> Any:
    return mcp_server.parser().parse_args(
        [
            "--config",
            "configs/runtime/embodied_live.yaml",
            "--scene",
            "scene_000001",
            "--checkpoint",
            "reports/gemma4/artifacts/v85_strict_runtime_candidate",
            "--runtime-asset",
            "data/runtime_scenes/scene_000001/s_000001.blend",
            "--robot-state-checkpoint",
            "data_gemma4/checkpoints/robot_state_numeric_v1",
            *extra,
        ]
    )


def test_mcp_server_candidate_mode_is_off_by_default() -> None:
    args = _candidate_server_arguments()

    assert args.v96_candidate_bridge_hook is None
    assert args.v96_scene_memory is None
    assert args.allow_explicit_v96_candidate is False
    assert mcp_server._validate_arguments(args) is True


def test_generic_mcp_response_omits_null_only_v96_receipts() -> None:
    payload: dict[str, Any] = {
        "success": True,
        "error_code": None,
        "scene_id": "scene_000001",
        "seed": 0,
        "scene_version": 0,
        "position_m": [0.0, 0.0, 0.0],
        "camera_position_m": [0.0, 0.0, 1.0],
        "body_yaw_degrees": 0.0,
        "camera_yaw_degrees": 0.0,
        "pitch_degrees": 0.0,
        "linear_velocity_xy_m": [0.0, 0.0],
        "angular_velocity_degrees": 0.0,
        "collision": False,
        "last_movement_delta_m": [0.0, 0.0, 0.0],
        "distance_moved": 0.0,
        "turn_degrees": 0.0,
        "scan_coverage": 0.0,
        "scan_count": 0,
        "visible_voxels": 0,
        "valid_depth_pixels": 0,
        "observation_id": None,
        "clearance_m": None,
        "action_count": 0,
        "stopped": False,
    }
    receipt_fields = {
        "map_sha256",
        "schema",
        "map_version",
        "scene_prefix_sha256",
        "scene_control_signature_sha256",
        "source_voxels",
        "processed_voxels",
        "binding_sha256",
        "active_prefix_sha256",
        "robot_state_sha256",
        "robot_tokens_sha256",
        "robot_state_encoder_sha256",
        "active_binding_sha256",
    }

    generic = mcp_server._response(payload).model_dump(by_alias=True)
    assert receipt_fields.isdisjoint(generic)

    v96_payload = {
        **payload,
        "map_sha256": _digest("1"),
        "schema": "semantic_3d_chat.scene_prefix_binding.v2",
        "map_version": 1,
        "scene_prefix_sha256": _digest("2"),
        "scene_control_signature_sha256": _digest("3"),
        "source_voxels": 100,
        "processed_voxels": 100,
        "binding_sha256": _digest("4"),
        "active_prefix_sha256": _digest("5"),
        "robot_state_sha256": _digest("6"),
        "robot_tokens_sha256": _digest("7"),
        "robot_state_encoder_sha256": _digest("8"),
        "active_binding_sha256": _digest("9"),
    }
    v96 = mcp_server._response(v96_payload).model_dump(by_alias=True)
    assert receipt_fields <= set(v96)


@pytest.mark.parametrize(
    "extra",
    (
        ("--v96-candidate-bridge-hook", "configs/runtime/gemma4_v96_explicit_candidate_mcp_bridge.yaml"),
        ("--allow-explicit-v96-candidate",),
        ("--v96-scene-memory", "sanitized/memory"),
    ),
)
def test_mcp_server_candidate_mode_requires_both_explicit_gates(
    extra: tuple[str, ...],
) -> None:
    args = _candidate_server_arguments(*extra)

    with pytest.raises(SystemExit):
        mcp_server._validate_arguments(args)


def test_mcp_server_candidate_mode_rejects_legacy_question_controller() -> None:
    args = _candidate_server_arguments(
        "--v96-candidate-bridge-hook",
        "configs/runtime/gemma4_v96_explicit_candidate_mcp_bridge.yaml",
        "--allow-explicit-v96-candidate",
        "--control-checkpoint",
        "data_gemma4/runtime/checkpoints/gemma4_v75_nll_control_release_v1",
        "--control-runtime-config",
        "configs/runtime/gemma4_v56_question_control.yaml",
    )

    with pytest.raises(SystemExit, match="cannot be combined"):
        mcp_server._validate_arguments(args)


def test_mcp_server_candidate_mode_accepts_explicit_unpromoted_pair() -> None:
    args = _candidate_server_arguments(
        "--v96-candidate-bridge-hook",
        "configs/runtime/gemma4_v96_explicit_candidate_mcp_bridge.yaml",
        "--allow-explicit-v96-candidate",
    )

    assert mcp_server._validate_arguments(args) is True


def test_mcp_candidate_preflight_authenticates_pass_before_checkpoint_or_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class AuthorizationBoundaryReached(RuntimeError):
        pass

    def stop_before_runtime(*_args: Any, **_kwargs: Any) -> None:
        raise AuthorizationBoundaryReached

    monkeypatch.setattr(
        candidate_cli,
        "run_isolated_v96_authorization",
        stop_before_runtime,
    )
    args = _candidate_server_arguments(
        "--v96-candidate-bridge-hook",
        "configs/runtime/gemma4_v96_explicit_candidate_mcp_bridge.yaml",
        "--allow-explicit-v96-candidate",
    )
    audit = FileAccessAudit([], block_forbidden=True)

    with audit, pytest.raises(AuthorizationBoundaryReached):
        mcp_server._v96_candidate_server_contract(args, audit=audit)

    loaded = "\n".join(audit.unique_paths)
    assert "gemma4_v96_explicit_candidate_mcp_bridge.yaml" in loaded
    assert "gemma4_v96_explicit_candidate_hook.yaml" in loaded
    assert "adapter.safetensors" not in loaded
    assert "memory.safetensors" not in loaded


def test_mcp_candidate_rejects_known_development_only_before_checkpoint_or_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class DeferredReleaseRequired(RuntimeError):
        pass

    monkeypatch.setattr(
        candidate_cli,
        "run_isolated_v96_authorization",
        lambda *_args, **_kwargs: object(),
    )

    def reject_known_development_only(*_args: Any, **_kwargs: Any) -> None:
        raise DeferredReleaseRequired

    monkeypatch.setattr(
        bridge,
        "run_isolated_v96_release_verification",
        reject_known_development_only,
    )
    args = _candidate_server_arguments(
        "--v96-candidate-bridge-hook",
        "configs/runtime/gemma4_v96_explicit_candidate_mcp_bridge.yaml",
        "--allow-explicit-v96-candidate",
    )
    audit = FileAccessAudit([], block_forbidden=True)

    with audit, pytest.raises(DeferredReleaseRequired):
        mcp_server._v96_candidate_server_contract(args, audit=audit)

    loaded = "\n".join(audit.unique_paths)
    assert "adapter.safetensors" not in loaded
    assert "memory.safetensors" not in loaded


def test_isolated_release_verifier_requires_exact_deferred_and_smoke_checks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checks = {name: True for name in bridge._REQUIRED_RELEASE_CHECKS}

    def completed(payload: dict[str, Any]) -> SimpleNamespace:
        return SimpleNamespace(
            returncode=0,
            stdout=json.dumps(payload),
            stderr="",
        )

    valid = {
        "phase": bridge._RELEASE_VERIFY_PHASE,
        "checks": checks,
        "passed": True,
        "candidate_fingerprint_sha256": _digest("1"),
        "candidate_checkpoint_sha256": _digest("2"),
        "candidate_adapter_sha256": _digest("3"),
        "deferred_final_evidence_sha256": _digest("4"),
        "runtime_smoke_sha256": _digest("5"),
        "release_report_sha256": _digest("6"),
        "release_checkpoint_sha256": _digest("7"),
        "release_adapter_sha256": _digest("8"),
        "v95_state_sha256": _digest("9"),
        "v96_state_sha256": _digest("a"),
        "runtime_implementation_inventory_sha256": _digest("b"),
        "scene_ids": [f"scene_{index:06d}" for index in range(25, 31)],
    }
    monkeypatch.setattr(bridge.subprocess, "run", lambda *_args, **_kwargs: completed(valid))
    receipt = bridge.run_isolated_v96_release_verification()
    assert receipt["deferred_final_binding_exact"] is True
    assert receipt["runtime_smoke_binding_exact"] is True
    assert receipt["promoted_runtime_release_verified"] is True

    known_development_only = {
        **valid,
        "checks": {"known_development_gate_passed": True},
    }
    monkeypatch.setattr(
        bridge.subprocess,
        "run",
        lambda *_args, **_kwargs: completed(known_development_only),
    )
    with pytest.raises(ValueError, match="deferred-final"):
        bridge.run_isolated_v96_release_verification()


def test_v96_bridge_source_has_no_evaluator_or_training_imports() -> None:
    tree = ast.parse(inspect.getsource(bridge))
    modules = {
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module is not None
    }

    assert not any(
        module.startswith(("semantic_3d_chat.evaluation", "semantic_3d_chat.training"))
        for module in modules
    )


@pytest.mark.parametrize("component", ("oracle", "qa", "training", "predictions"))
def test_v96_compiler_rejects_protected_paths_before_open(
    tmp_path: Path, component: str
) -> None:
    with pytest.raises(ValueError, match="forbidden runtime data"):
        bridge._safe_runtime_path(
            tmp_path / component / "candidate", purpose="synthetic protected path"
        )


@pytest.mark.parametrize(
    "component", ("questions", "predictions", "scorer", "features", "rendered")
)
def test_actual_mcp_memory_and_map_guards_reject_protected_paths(
    tmp_path: Path, component: str
) -> None:
    candidate = tmp_path / component / "numeric.bin"
    with pytest.raises(ValueError, match="environmental supervision"):
        mcp_server._safe_runtime_path(candidate, purpose="synthetic V96 memory")
    with pytest.raises(ValueError, match="protected runtime path"):
        refresh._safe_map_path(candidate, purpose="synthetic V96 persistent map")
    with pytest.raises(ValueError, match="protected runtime path"):
        semantic_mapping._safe_numeric_path(
            candidate, purpose="synthetic V96 persistent map"
        )


@pytest.mark.parametrize("component", ("questions", "predictions", "scorer"))
def test_mcp_lifetime_audit_blocks_held_out_and_scorer_reads(
    tmp_path: Path, component: str
) -> None:
    protected = tmp_path / component / "held.json"
    protected.parent.mkdir()
    protected.write_text("{}", encoding="utf-8")
    audit = mcp_server._runtime_audit()

    with audit, pytest.raises(PermissionError, match="Blocked forbidden"):
        protected.read_text(encoding="utf-8")

    assert audit.forbidden_accesses() == [str(protected.resolve())]


def test_v96_make_targets_isolate_candidate_map_scans_and_audits() -> None:
    makefile = Path("Makefile").read_text(encoding="utf-8")
    check = makefile.split("\nv96-explicit-candidate-embodied-mcp-check:", 1)[1].split(
        "\n\n", 1
    )[0]
    live = makefile.split("\nv96-explicit-candidate-embodied-mcp:", 1)[1].split(
        "\n\n", 1
    )[0]

    for body in (check, live):
        assert "--persistent-map" in body
        assert "--scan-output-directory" in body
        assert "--audit-report" in body


def test_v96_make_exposes_finite_promoted_release_robot_smoke() -> None:
    makefile = Path("Makefile").read_text(encoding="utf-8")
    target = makefile.split(
        "\nv96-explicit-candidate-embodied-mcp-live-smoke:", 1
    )[1].split("\n\n", 1)[0]

    assert "v96-release-verify" in target
    assert "scene_000025" in target
    assert "v95_deferred_final/memory_cache/$(SCENE)" in target
    assert "scripts/run_v96_candidate_mcp_live_smoke.py" in target
    assert "--output" in target
    assert "semantic_3d_chat.mcp_server.server" not in target


def test_v96_finite_live_smoke_command_is_numeric_and_explicit() -> None:
    parameters = live_smoke.v96_server_parameters(
        python_executable=".venv-gemma4/bin/python",
        config="configs/runtime/embodied_live.yaml",
        scene_id="scene_000025",
        base_checkpoint="reports/gemma4/artifacts/v85_strict_runtime_candidate",
        bridge_hook="configs/runtime/gemma4_v96_explicit_candidate_mcp_bridge.yaml",
        scene_memory=(
            "reports/gemma4/artifacts/v95_deferred_final/memory_cache/scene_000025"
        ),
        runtime_asset="data/runtime_assets/scene_000025/s_000025.blend",
        robot_state_checkpoint="data_gemma4/checkpoints/robot_state_numeric_v1",
        persistent_map="reports/gemma4/artifacts/synthetic/map.npz",
        scan_output_directory="reports/gemma4/artifacts/synthetic/scans",
        audit_report="reports/gemma4/metrics/synthetic_access.json",
    )

    assert "--v96-candidate-bridge-hook" in parameters.args
    assert "--v96-scene-memory" in parameters.args
    assert "--allow-explicit-v96-candidate" in parameters.args
    assert "--persistent-map" in parameters.args
    assert "--scan-output-directory" in parameters.args
    assert parameters.args[-2:] == ["--transport", "stdio"]
    assert "--control-checkpoint" not in parameters.args
    assert "--control-runtime-config" not in parameters.args
    assert parameters.env is not None
    assert parameters.env["HF_HUB_OFFLINE"] == "1"
    assert parameters.env["TRANSFORMERS_OFFLINE"] == "1"


def test_v96_finite_live_smoke_refuses_before_transport_without_release(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def reject_release() -> dict[str, Any]:
        raise RuntimeError("promoted release unavailable")

    def transport_must_not_start(*_args: Any, **_kwargs: Any) -> None:
        raise AssertionError("MCP transport started before release authentication")

    monkeypatch.setattr(live_smoke, "run_isolated_v96_release_verification", reject_release)
    monkeypatch.setattr(live_smoke, "stdio_client", transport_must_not_start)

    with pytest.raises(RuntimeError, match="promoted release unavailable"):
        asyncio.run(
            live_smoke.run_v96_candidate_mcp_live_smoke(
                config="missing/config.yaml",
                scene_id="scene_000025",
                base_checkpoint="missing/checkpoint",
                bridge_hook="missing/hook.yaml",
                scene_memory="missing/memory",
                runtime_asset="missing/asset.blend",
                robot_state_checkpoint="missing/robot",
                audit_report="missing/audit.json",
            )
        )


def test_v96_finite_live_smoke_report_does_not_claim_answer_token_support() -> None:
    report = live_smoke.build_v96_live_smoke_report(
        release_receipt=_promoted_release_receipt(),
        scene_id="scene_000025",
        base_checkpoint="reports/gemma4/artifacts/v85_strict_runtime_candidate",
        scene_memory=(
            "reports/gemma4/artifacts/v95_deferred_final/memory_cache/scene_000025"
        ),
        runtime_asset="data/runtime_assets/scene_000025/s_000025.blend",
        audit_report="reports/gemma4/metrics/synthetic_access.json",
        audit_report_sha256=_digest("9"),
        protocol_version="synthetic",
        server_name="semantic-3d-chat",
        server_version="synthetic",
        loaded_file_count=1,
        elapsed_seconds=1.0,
        evidence={"numeric_refresh_validated": True},
    )

    assert live_smoke.LIVE_TOOL_SEQUENCE == ("get_robot_state", "scan", "turn")
    assert report["promoted_runtime_release_verified_before_transport"] is True
    assert report["audit_report_sha256"] == _digest("9")
    assert report["numeric_tool_outputs_only"] is True
    assert report["question_free_full_memory_refresh"] is True
    assert report["language_questions_asked"] == 0
    assert report["v96_answer_generation_exercised"] is False
    assert report["direct_v96_answer_robot_tokens_authenticated"] is False


def test_question_free_compiler_builds_complete_bfloat16_memory_without_text() -> None:
    torch.manual_seed(96)
    basis = torch.eye(1536, dtype=torch.float32)[:2].contiguous()
    controller = DenseFullSceneContinuousControlV75(
        1536,
        basis,
        environment_latents=256,
        query_count=4,
        model_dimension=2,
        coefficient_decoder_hidden_dimension=2,
        uniform_floor_mass=0.05,
        maximum_control_rms=0.25,
    )
    for parameter in controller.parameters():
        parameter.data.zero_()
    probes = torch.randn(96, 1536, dtype=torch.float32)
    probe_hash = tensor_sha256(probes)
    compiler = bridge.V75QuestionFreeV96MemoryCompiler(
        controller,
        probes,
        source_control_checkpoint_sha256=_digest("a"),
        source_probe_tensor_sha256=probe_hash,
    )
    base_prefix = torch.randn(1, 258, 1536, dtype=torch.bfloat16)
    base = SimpleNamespace(scene_id="scene_000039", scene_prefix=base_prefix)
    metadata = {
        "scene_id": "scene_000039",
        "source_base_checkpoint_sha256": _digest("b"),
        "runtime_config_sha256": _digest("c"),
        "source_control_checkpoint_sha256": _digest("a"),
        "source_probe_tensor_sha256": probe_hash,
        "environmental_text_inputs": [],
        "questions_or_answers_serialized": False,
        "oracle_loaded": False,
    }

    loaded = compiler.compile(cast(Any, base), prior_metadata=metadata)

    assert loaded.memory.shape == (1, 738, 1536)
    assert loaded.memory.dtype == torch.bfloat16
    assert loaded.metadata["base_prefix_sha256"] == prefix_sha256(base_prefix)
    assert loaded.metadata["canonical_prefix_sha256"] == prefix_sha256(
        loaded.memory
    )
    assert loaded.metadata["compiled_before_user_question"] is True
    assert loaded.metadata["question_inputs_used_for_compilation"] is False
    assert loaded.metadata["question_dependent_retrieval"] is False
    assert loaded.metadata["environmental_text_inputs"] == []
    assert loaded.metadata["questions_or_answers_serialized"] is False
    assert loaded.metadata["oracle_loaded"] is False


def test_question_free_compiler_rejects_changed_source_binding() -> None:
    torch.manual_seed(97)
    probes = torch.randn(96, 1536, dtype=torch.float32)
    controller = DenseFullSceneContinuousControlV75(
        1536,
        torch.eye(1536, dtype=torch.float32)[:2].contiguous(),
        environment_latents=256,
        query_count=4,
        model_dimension=2,
        coefficient_decoder_hidden_dimension=2,
    )
    compiler = bridge.V75QuestionFreeV96MemoryCompiler(
        controller,
        probes,
        source_control_checkpoint_sha256=_digest("d"),
        source_probe_tensor_sha256=tensor_sha256(probes),
    )
    base = SimpleNamespace(
        scene_id="scene_000039",
        scene_prefix=torch.zeros(1, 258, 1536, dtype=torch.bfloat16),
    )
    metadata = {
        "scene_id": "scene_000039",
        "source_control_checkpoint_sha256": _digest("e"),
        "source_probe_tensor_sha256": tensor_sha256(probes),
        "environmental_text_inputs": [],
        "questions_or_answers_serialized": False,
        "oracle_loaded": False,
    }

    with pytest.raises(ValueError, match="sources differ"):
        compiler.compile(cast(Any, base), prior_metadata=metadata)
