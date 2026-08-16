from __future__ import annotations

import asyncio
import copy
import hashlib
import json
import shutil
from collections.abc import Mapping
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch

from semantic_3d_chat.evaluation import v96_embodied_heldout as heldout
from semantic_3d_chat.language.prefix_injection import prefix_sha256
from semantic_3d_chat.robot import v96_release_action as release_action
from semantic_3d_chat.robot import v96_release_embodied as release_embodied
from semantic_3d_chat.robot.llm_tool_policy import GeneratedToolProposal
from semantic_3d_chat.robot.mcp_stdio_runtime import (
    MCPActionTransportError,
    validate_numeric_tool_receipt,
)
from semantic_3d_chat.robot.navigation_policy import split_active_prefix
from semantic_3d_chat.robot.v96_co_resident_mcp_agent import (
    V96CoResidentMCPAgent,
)
from semantic_3d_chat.robot.v96_release_action import (
    ACTIVE_TOKEN_COUNT,
    HIDDEN_SIZE,
    ROBOT_TOKEN_COUNT,
    SOURCE_SCENE_TOKEN_COUNT,
    V96_SCENE_TOKEN_COUNT,
    load_v96_sequence_length_transfer,
)


def _digest(character: str) -> str:
    return character * 64


def _release_receipt() -> dict[str, Any]:
    return {
        "phase": "v96_strict_runtime_release_verified",
        "passed": True,
        "check_count": 15,
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
        "scene_ids": [f"scene_{index:06d}" for index in range(25, 31)],
    }


def _canonical_sha256(payload: object) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _path_inventory(paths: tuple[str, ...], character: str) -> dict[str, Any]:
    rows = [
        {"path": path, "sha256": _digest(character), "size_bytes": 1}
        for path in sorted(paths)
    ]
    return {"files": rows, "inventory_sha256": _canonical_sha256(rows)}


def _artifact_inventory(expected: Mapping[str, str]) -> dict[str, Any]:
    rows = [
        {"name": name, "sha256": expected[name], "size_bytes": 1}
        for name in sorted(expected)
    ]
    return {"files": rows, "inventory_sha256": _canonical_sha256(rows)}


def _dependency_contract() -> dict[str, Any]:
    navigation = {
        **_artifact_inventory(
            {
                "policy.safetensors": release_action.V3_POLICY_WEIGHTS_SHA256,
                "runtime_metadata.json": release_action.V3_POLICY_METADATA_SHA256,
            }
        ),
        "training_dataset_sha256": release_action.V3_TRAINING_DATASET_SHA256,
        "training_manifest_file_sha256": heldout.NAVIGATION_MANIFEST_SHA256,
        "training_traces_file_sha256": heldout.NAVIGATION_TRACES_SHA256,
        "training_experiment_config_sha256": (
            heldout.NAVIGATION_EXPERIMENT_CONFIG_SHA256
        ),
        "train_scene_ids": list(heldout.NAVIGATION_TRAIN_SCENES),
        "validation_scene_ids": list(heldout.NAVIGATION_VALIDATION_SCENES),
    }
    assets: dict[str, Any] = {}
    for scene_id in release_embodied.RELEASE_SCENE_IDS:
        suffix = scene_id.removeprefix("scene_")
        asset = {
            "scene_id": scene_id,
            "asset_path": f"data/runtime_assets/{scene_id}/s_{suffix}.blend",
            "asset_sha256": _digest("a"),
            "asset_size_bytes": 1,
            "manifest_path": f"data/runtime_assets/{scene_id}/s_{suffix}.json",
            "manifest_file_sha256": _digest("b"),
            "manifest_contract_sha256": _digest("c"),
        }
        asset["contract_sha256"] = _canonical_sha256(asset)
        assets[scene_id] = asset
    value = {
        "schema": "semantic_3d_chat.v96_embodied_dependency_contract.v1",
        "navigation_policy": navigation,
        "robot_state_checkpoint": _artifact_inventory(
            release_embodied.ROBOT_STATE_FILE_SHA256S
        ),
        "question_free_compiler_checkpoint": _artifact_inventory(
            release_embodied.V75_CONTROL_FILE_SHA256S
        ),
        "numeric_probe_bank": _artifact_inventory(
            release_embodied.V75_PROBE_FILE_SHA256S
        ),
        "runtime_config_inventory": _path_inventory(
            heldout._RUNTIME_CONFIG_PATHS, "d"
        ),
        "runtime_source_inventory": _path_inventory(
            heldout._RUNTIME_SOURCE_PATHS, "e"
        ),
        "implementation_source_inventory": _path_inventory(
            heldout._IMPLEMENTATION_SOURCE_PATHS, "f"
        ),
        "runtime_assets": assets,
        "runtime_asset_inventory_sha256": _canonical_sha256(assets),
        "policy_transfer": release_action.TRANSFER_MODE,
        "source_policy_retrained_on_v96": False,
    }
    value["contract_sha256"] = _canonical_sha256(value)
    return value


@contextmanager
def _patched_dependency_contract(value: Mapping[str, Any]) -> Any:
    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setattr(
            heldout,
            "build_embodied_dependency_contract",
            lambda **_kwargs: copy.deepcopy(dict(value)),
        )
        yield


def _test_preregistration_payload(
    receipt: Mapping[str, Any],
    dependencies: Mapping[str, Any],
) -> dict[str, Any]:
    with _patched_dependency_contract(dependencies):
        return heldout.build_preregistration_payload(receipt)


def _rehash_runtime_evidence(value: dict[str, Any]) -> None:
    value["navigation_sha256"] = _canonical_sha256(value["navigation"])
    value["runtime_access_log_sha256"] = _canonical_sha256(
        value["runtime_access_log"]
    )
    identity = dict(value)
    identity.pop("evidence_identity_sha256", None)
    value["evidence_identity_sha256"] = _canonical_sha256(identity)


def _valid_scan_evidence(
    tmp_path: Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], Path, dict[str, Any]]:
    receipt = _release_receipt()
    dependencies = _dependency_contract()
    preregistration = tmp_path / "preregistration.json"
    with _patched_dependency_contract(dependencies):
        written = heldout.write_preregistration(
            preregistration,
            release_verifier=lambda: receipt,
        )
    expected = next(
        row for row in written["payload"]["tasks"] if row["key"] == "scan_only"
    )
    runtime = _SyntheticV96Runtime()
    backend = _SyntheticV96Backend(runtime, "scan")
    agent = V96CoResidentMCPAgent(runtime, backend, runtime.config)
    navigation = asyncio.run(
        agent.run_instruction(expected["instruction"], max_steps=1)
    ).as_dict()
    asset = dependencies["runtime_assets"][expected["scene_id"]]
    required_reads = [
        str((heldout.PROJECT_ROOT / asset["asset_path"]).resolve()),
        str((heldout.PROJECT_ROOT / asset["manifest_path"]).resolve()),
        str(
            (
                heldout.NAVIGATION_CHECKPOINT / "policy.safetensors"
            ).resolve()
        ),
        str(
            (
                heldout.NAVIGATION_CHECKPOINT / "runtime_metadata.json"
            ).resolve()
        ),
    ]
    evidence = heldout.build_runtime_evidence(
        task_id=expected["task_id"],
        scene_id=expected["scene_id"],
        navigation_result=navigation,
        preregistration_sha256=written["sha256"],
        release_receipt_sha256=written["payload"]["release_receipt_sha256"],
        dependency_contract_sha256=dependencies["contract_sha256"],
        runtime_config_inventory_sha256=dependencies["runtime_config_inventory"][
            "inventory_sha256"
        ],
        runtime_source_inventory_sha256=dependencies["runtime_source_inventory"][
            "inventory_sha256"
        ],
        implementation_source_inventory_sha256=dependencies[
            "implementation_source_inventory"
        ]["inventory_sha256"],
        runtime_asset_contract_sha256=asset["contract_sha256"],
        runtime_task_input_sha256=written["payload"][
            "runtime_task_input_sha256_by_scene"
        ][expected["scene_id"]],
        runtime_access_log=heldout.build_runtime_access_log(required_reads),
    )
    return evidence, expected, written, preregistration, runtime.config


def test_promoted_receipt_rejects_weaker_or_incomplete_gate() -> None:
    valid = _release_receipt()
    assert release_embodied.validate_promoted_v96_release_receipt(valid) == valid

    incomplete = dict(valid)
    incomplete.pop("runtime_smoke_sha256")
    with pytest.raises(ValueError, match="fields changed"):
        release_embodied.validate_promoted_v96_release_receipt(incomplete)

    unpromoted = dict(valid)
    unpromoted["promoted_runtime_release_verified"] = False
    with pytest.raises(ValueError, match="exact promoted release PASS"):
        release_embodied.validate_promoted_v96_release_receipt(unpromoted)


def test_release_gate_runs_before_any_config_or_model_load(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []

    def blocked() -> Mapping[str, Any]:
        calls.append("verifier")
        raise RuntimeError("release unavailable")

    def forbidden_load(*_args: Any, **_kwargs: Any) -> Any:
        calls.append("config")
        raise AssertionError("config was opened before release gate")

    monkeypatch.setattr(release_embodied, "load_runtime_config", forbidden_load)
    with pytest.raises(RuntimeError, match="release unavailable"):
        release_embodied.build_promoted_v96_embodied_runtime(
            "scene_000025",
            runtime_asset="does_not_exist.blend",
            release_verifier=blocked,
        )
    assert calls == ["verifier"]


def test_downstream_make_targets_are_release_gated_and_not_evaluator_prerequisites() -> None:
    makefile = Path("Makefile").read_text(encoding="utf-8")

    assert "v96-embodied-check:" in makefile
    assert "v96-embodied-heldout-preflight:" in makefile
    assert "v96-embodied-heldout-preregister: v96-release-verify" in makefile
    assert (
        "v96-embodied-heldout-run: v96-embodied-heldout-authenticate" in makefile
    )
    assert (
        "v96-embodied-heldout-score: v96-embodied-heldout-authenticate" in makefile
    )
    assert "v96-embodied-live: v96-release-verify" in makefile
    assert "v96-handoff-check: v96-report-check v96-demo-check" in makefile
    assert "v96-handoff-check: v96-embodied" not in makefile


def test_promoted_runtime_builder_accepts_only_promoted_strict_runtime(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeStrict:
        def __init__(self, *, promoted: bool = True) -> None:
            self.scene_id = "scene_000025"
            self.runtime_package_mode = "promoted" if promoted else "candidate"
            self.runtime_promotion_authorized = promoted
            self.release_provenance = {
                "promotion_decision": (
                    release_embodied.PROMOTED_DECISION if promoted else "pending"
                ),
                "candidate_fingerprint_sha256": _digest("1"),
            }
            self.v95_state_sha256 = _digest("6")
            self.v96_state_sha256 = _digest("7")
            self.fixed_scene_memory = torch.zeros(1, 738, 1536, dtype=torch.float16)
            self.scene_prefix_hash = _digest("9")
            self.questions_answered = 0
            self.scene_memory_metadata = {
                "source_control_checkpoint_sha256": _digest("a"),
                "source_probe_tensor_sha256": _digest("b"),
            }

        def assert_prefix_unchanged(self) -> None:
            return None

        def current_prefix_hash(self) -> str:
            return self.scene_prefix_hash

    compiler = SimpleNamespace(
        authenticated_control_sha256s=frozenset({_digest("a")}),
        source_probe_tensor_sha256=_digest("b"),
    )
    monkeypatch.setattr(
        release_embodied,
        "V96StrictMultisceneChatRuntime",
        FakeStrict,
    )
    builder = release_embodied.PromotedV96RuntimeBuilder(
        _release_receipt(), compiler, FakeStrict()
    )
    assert builder.release_receipt["promoted_runtime_release_verified"] is True
    with pytest.raises(ValueError, match="contract changed"):
        release_embodied.PromotedV96RuntimeBuilder(
            _release_receipt(), compiler, FakeStrict(promoted=False)
        )


@pytest.fixture(scope="module")
def transferred_controller() -> Any:
    controller, metadata, contract = load_v96_sequence_length_transfer(
        "data_gemma4/checkpoints/navigation_policy_v3",
        expected_model_id="google/gemma-4-E2B-it",
        expected_model_revision="3e22461f65e89153144f8adb70e3b8c2cc9845a7",
        device="cpu",
    )
    return controller, metadata, contract


def test_exact_v3_weights_transfer_from_258_to_738_without_metadata_mutation(
    transferred_controller: Any,
) -> None:
    controller, metadata, contract = transferred_controller
    source = json.loads(
        Path("data_gemma4/checkpoints/navigation_policy_v3/runtime_metadata.json").read_text(
            encoding="utf-8"
        )
    )

    assert source["scene_token_count"] == SOURCE_SCENE_TOKEN_COUNT
    assert metadata["scene_token_count"] == V96_SCENE_TOKEN_COUNT
    assert metadata["robot_token_count"] == ROBOT_TOKEN_COUNT
    assert contract.as_dict()["weights_changed"] is False
    assert contract.as_dict()["retrained_on_v96"] is False
    assert next(controller.parameters()).requires_grad is False


def test_transferred_controller_consumes_all_738_plus_4_and_rejects_bad_layout(
    transferred_controller: Any,
) -> None:
    controller, _metadata, _contract = transferred_controller
    generator = torch.Generator().manual_seed(96)
    scene = torch.randn(1, 738, 1536, generator=generator)
    robot = torch.randn(1, 4, 1536, generator=generator)
    instruction = torch.randn(1, 1536, generator=generator)
    target = torch.randn(1, 10, generator=generator)
    logits, arguments = controller(scene, robot, instruction, target)
    first_audit = dict(controller.last_forward_audit)

    changed_scene = scene.clone()
    changed_scene[0, 737, 17] += 2.0
    changed_logits, changed_arguments = controller(
        changed_scene, robot, instruction, target
    )
    second_audit = dict(controller.last_forward_audit)

    assert logits.shape == arguments.shape == (1, 5)
    assert second_audit["scene_tokens_processed"] == 738
    assert second_audit["robot_tokens_processed"] == 4
    assert first_audit["scene_prefix_sha256"] != second_audit["scene_prefix_sha256"]
    assert not (
        torch.equal(logits, changed_logits)
        and torch.equal(arguments, changed_arguments)
    )

    with pytest.raises(ValueError, match=r"\[1,738,1536\]"):
        controller(scene[:, :-1], robot, instruction, target)
    invalid = robot.clone()
    invalid[0, 0, 0] = float("nan")
    with pytest.raises(ValueError, match="NaN or infinity"):
        controller(scene, invalid, instruction, target)


class _SyntheticV96Runtime:
    def __init__(self, *, auto_scan_after_motion: bool = False) -> None:
        self.config: dict[str, Any] = {
            "scene": {"room_size_m": [6.0, 5.0, 3.0]},
            "robot": {
                "auto_scan_after_motion": auto_scan_after_motion,
                "max_move_m": 0.5,
                "max_move_to_m": 1.0,
                "max_turn_degrees": 45.0,
                "max_look_delta_degrees": 45.0,
                "max_camera_yaw_offset_degrees": 90.0,
                "max_pitch_degrees": 45.0,
            },
        }
        self.scene = torch.linspace(
            -1.0, 1.0, V96_SCENE_TOKEN_COUNT * HIDDEN_SIZE
        ).reshape(1, V96_SCENE_TOKEN_COUNT, HIDDEN_SIZE)
        self.robot = torch.zeros(1, ROBOT_TOKEN_COUNT, HIDDEN_SIZE)
        self.yaw = 0.0
        self.version = 0
        self.action_count = 0
        self.stopped = False
        self._refresh()

    def _refresh(self) -> None:
        self.active = torch.cat((self.scene[:, :-1], self.robot, self.scene[:, -1:]), dim=1)
        scene_identity = {
            "schema": "semantic_3d_chat.scene_prefix_binding.v2",
            "scene_id": "scene_000025",
            "map_version": self.version,
            "map_sha256": f"{(self.version + 1) % 16:x}" * 64,
            "scene_prefix_sha256": _digest("a"),
            "scene_control_signature_sha256": prefix_sha256(self.scene),
            "source_voxels": 32,
            "processed_voxels": 32,
        }
        binding_sha = _canonical_sha256(scene_identity)
        active_identity = {
            **scene_identity,
            "binding_sha256": binding_sha,
            "active_prefix_sha256": prefix_sha256(self.active),
            "robot_state_sha256": f"{(self.action_count + 2) % 16:x}" * 64,
            "robot_tokens_sha256": prefix_sha256(self.robot),
            "robot_state_encoder_sha256": _digest("f"),
        }
        self.binding = {
            **active_identity,
            "active_binding_sha256": _canonical_sha256(active_identity),
        }

    def active_prefix_snapshot(self) -> tuple[torch.Tensor, dict[str, Any]]:
        return self.active.clone(), dict(self.binding)

    def _receipt(
        self,
        *,
        success: bool = True,
        collision: bool = False,
        observation_id: str | None = None,
        turn_degrees: float = 0.0,
    ) -> dict[str, Any]:
        return {
            "success": success,
            "error_code": None if success else "E_COLLISION",
            "scene_id": "scene_000025",
            "seed": heldout.EVALUATION_RESET_SEED,
            "scene_version": self.version,
            "position_m": [0.0, 0.0, 0.0],
            "camera_position_m": [0.0, 0.0, 1.2],
            "body_yaw_degrees": self.yaw,
            "camera_yaw_degrees": self.yaw,
            "pitch_degrees": 0.0,
            "linear_velocity_xy_m": [0.0, 0.0],
            "angular_velocity_degrees": turn_degrees,
            "collision": collision,
            "last_movement_delta_m": [0.0, 0.0, 0.0],
            "distance_moved": 0.0,
            "turn_degrees": turn_degrees,
            "scan_coverage": min(1.0, self.version / 10.0),
            "scan_count": self.version,
            "visible_voxels": 32 if self.version else 0,
            "valid_depth_pixels": 64 if self.version else 0,
            "observation_id": observation_id,
            "clearance_m": None,
            "action_count": self.action_count,
            "stopped": self.stopped,
            **self.binding,
        }

    def get_robot_state(self) -> dict[str, Any]:
        return self._receipt()

    def turn(self, angle_degrees: float) -> dict[str, Any]:
        self.yaw += float(angle_degrees)
        self.action_count += 1
        self.robot = self.robot.clone()
        self.robot[0, 0, 0] += float(angle_degrees) / 45.0
        self._refresh()
        return self._receipt(turn_degrees=float(angle_degrees))

    def scan(self) -> dict[str, Any]:
        self.version += 1
        self.action_count += 1
        self.scene = self.scene.clone()
        self.scene[0, 100, 3] += 0.25
        self.robot = self.robot.clone()
        self.robot[0, 1, 0] += 1.0
        self._refresh()
        return self._receipt(observation_id=f"o_{self.version:06d}")

    def look(self, _yaw: float, _pitch: float) -> dict[str, Any]:
        self.action_count += 1
        self._refresh()
        return self._receipt()

    def move_forward(self, _distance: float) -> dict[str, Any]:
        self.action_count += 1
        self._refresh()
        return self._receipt()

    def move_backward(self, _distance: float) -> dict[str, Any]:
        self.action_count += 1
        self._refresh()
        return self._receipt()

    def move_to(self, _x: float, _y: float) -> dict[str, Any]:
        self.action_count += 1
        self._refresh()
        return self._receipt()

    def stop(self) -> dict[str, Any]:
        self.action_count += 1
        self.stopped = True
        self._refresh()
        return self._receipt()

    def reset_scene(self, _scene_id: str, _seed: int) -> dict[str, Any]:
        self.__init__()
        return self._receipt()


class _SyntheticV96Backend:
    def __init__(self, runtime: _SyntheticV96Runtime, tool: str) -> None:
        self.runtime = runtime
        self.tool = tool
        self.forward_calls = 0
        self.last_v96_context_audit: dict[str, Any] | None = None

    def generate(
        self, instruction: str, *, correction_code: str | None
    ) -> GeneratedToolProposal:
        del instruction, correction_code
        active, binding = self.runtime.active_prefix_snapshot()
        scene, robot = split_active_prefix(
            active,
            scene_token_count=V96_SCENE_TOKEN_COUNT,
            robot_token_count=ROBOT_TOKEN_COUNT,
        )
        call = (
            {"tool": "turn", "arguments": {"angle_degrees": 15.0}}
            if self.tool == "turn"
            else {"tool": self.tool, "arguments": {}}
        )
        self.forward_calls += 1
        transfer = release_action.V96NavigationTransferContract(
            source_scene_token_count=SOURCE_SCENE_TOKEN_COUNT,
            target_scene_token_count=V96_SCENE_TOKEN_COUNT,
            robot_token_count=ROBOT_TOKEN_COUNT,
            hidden_size=HIDDEN_SIZE,
            source_weights_sha256=release_action.V3_POLICY_WEIGHTS_SHA256,
            source_metadata_sha256=release_action.V3_POLICY_METADATA_SHA256,
            source_training_dataset_sha256=(
                release_action.V3_TRAINING_DATASET_SHA256
            ),
            source_training_status=(
                "supervised_continuous_semantic_grounded_navigation_policy_v3"
            ),
        ).as_dict()
        forward = {
            "schema": "semantic_3d_chat.v96_navigation_transfer_forward.v1",
            "forward_call": self.forward_calls,
            "scene_shape": [1, 738, 1536],
            "robot_shape": [1, 4, 1536],
            "scene_prefix_sha256": prefix_sha256(scene),
            "robot_tokens_sha256": prefix_sha256(robot),
            "scene_tokens_processed": 738,
            "robot_tokens_processed": 4,
            "hidden_size": 1536,
            "all_scene_tokens_enter_attention_keys_and_values": True,
            "all_scene_tokens_enter_global_mean": True,
            "robot_tokens_enter_robot_value_mean": True,
            "question_dependent_scene_selection": False,
            "top_k_scene_selection": False,
            "environmental_text_inputs": [],
            "oracle_inputs_at_runtime": False,
            "transfer": transfer,
        }
        self.last_v96_context_audit = {
            "schema": "semantic_3d_chat.v96_release_action_context.v1",
            "active_prefix_shape": [1, ACTIVE_TOKEN_COUNT, HIDDEN_SIZE],
            "active_prefix_sha256": prefix_sha256(active),
            "full_scene_memory_sha256": prefix_sha256(scene),
            "base_scene_prefix_sha256": binding["scene_prefix_sha256"],
            "robot_tokens_sha256": prefix_sha256(robot),
            "map_sha256": binding["map_sha256"],
            "scene_tokens_consumed": 738,
            "robot_tokens_consumed": 4,
            "policy_consumed_738_scene_tokens": True,
            "policy_consumed_4_robot_tokens": True,
            "complete_scene_memory_used": True,
            "question_dependent_scene_retrieval": False,
            "target_grounding_used": False,
            "all_active_map_voxels_scored_for_target_grounding": None,
            "grounding_scored_voxels": None,
            "source_policy_was_retrained_on_v96": False,
            "transfer": transfer,
            "forward": forward,
            "environmental_text_inputs": [],
            "oracle_inputs_at_runtime": False,
        }
        return GeneratedToolProposal(
            text=json.dumps(call, sort_keys=True, separators=(",", ":")),
            active_prefix_sha256=binding["active_prefix_sha256"],
            scene_prefix_sha256=binding["scene_prefix_sha256"],
            robot_tokens_sha256=binding["robot_tokens_sha256"],
            local_inference=True,
            used_continuous_scene_prefix=True,
            used_continuous_robot_tokens=True,
            training_status="supervised_continuous_semantic_grounded_navigation_policy_v3",
        )


def test_co_resident_policy_dispatches_through_official_mcp_and_rebinds_robot() -> None:
    runtime = _SyntheticV96Runtime()
    backend = _SyntheticV96Backend(runtime, "turn")
    agent = V96CoResidentMCPAgent(runtime, backend, runtime.config)

    step = asyncio.run(agent.step("Turn toward the target."))

    assert step.call == {"tool": "turn", "arguments": {"angle_degrees": 15.0}}
    assert step.transport == "official_python_mcp_sdk_in_process_dispatch"
    assert step.receipt["body_yaw_degrees"] == 15.0
    assert step.before_binding["active_prefix_sha256"] != step.after_binding[
        "active_prefix_sha256"
    ]
    assert step.policy_context_audit["policy_consumed_738_scene_tokens"] is True
    assert step.policy_context_audit["policy_consumed_4_robot_tokens"] is True
    serialized = json.dumps(step.as_dict(), sort_keys=True)
    assert "Turn toward the target." not in serialized
    assert "chair" not in serialized

    leaked = dict(step.receipt)
    leaked["object_name"] = "chair"
    with pytest.raises(MCPActionTransportError, match="numeric schema"):
        validate_numeric_tool_receipt(leaked, require_continuous_binding=True)


def test_co_resident_scan_refreshes_map_and_complete_v96_memory() -> None:
    runtime = _SyntheticV96Runtime()
    backend = _SyntheticV96Backend(runtime, "scan")
    agent = V96CoResidentMCPAgent(runtime, backend, runtime.config)

    step = asyncio.run(agent.step("Scan."))

    assert step.receipt["map_version"] == 1
    assert step.before_binding["map_sha256"] != step.after_binding["map_sha256"]
    assert step.before_binding["scene_control_signature_sha256"] != step.after_binding[
        "scene_control_signature_sha256"
    ]
    assert step.rgbd_observation_expected is True
    assert step.map_refresh_verified_before_next_decision is True


def test_co_resident_rejects_motion_that_promised_rgbd_without_map_refresh() -> None:
    runtime = _SyntheticV96Runtime(auto_scan_after_motion=True)
    backend = _SyntheticV96Backend(runtime, "turn")
    agent = V96CoResidentMCPAgent(runtime, backend, runtime.config)

    with pytest.raises(RuntimeError, match="Successful RGB-D action"):
        asyncio.run(agent.step("Turn."))
    assert agent.steps == []


def test_absent_grounding_never_claims_all_voxel_grounding() -> None:
    absent = release_action._grounding_coverage_audit(
        None,
        available_voxels=32,
        map_sha256=_digest("a"),
    )
    assert absent == {
        "target_grounding_used": False,
        "all_active_map_voxels_scored_for_target_grounding": None,
        "grounding_scored_voxels": None,
    }

    with pytest.raises(RuntimeError, match="complete active-map coverage"):
        release_action._grounding_coverage_audit(
            {
                "target_available": True,
                "scored_voxels": 31,
                "map_sha256": _digest("a"),
            },
            available_voxels=32,
            map_sha256=_digest("a"),
        )


def test_heldout_preregistration_is_fixed_release_gated_and_navigation_disjoint(
    tmp_path: Path,
) -> None:
    receipt = _release_receipt()
    dependencies = _dependency_contract()
    payload = _test_preregistration_payload(receipt, dependencies)

    assert len(payload["scene_ids"]) == 6
    assert len(payload["tasks"]) == 36
    assert set(payload["scene_ids"]).isdisjoint(payload["navigation_train_scene_ids"])
    assert set(payload["scene_ids"]).isdisjoint(
        payload["navigation_validation_scene_ids"]
    )
    assert payload["navigation_held_out"] is True
    assert payload["static_unseen_claim"] is False

    destination = tmp_path / "preregistration.json"
    with _patched_dependency_contract(dependencies):
        written = heldout.write_preregistration(
            destination,
            release_verifier=lambda: receipt,
        )
        authenticated = heldout.authenticate_preregistration(
            destination,
            release_receipt=receipt,
        )
    assert written["sha256"] == authenticated["sha256"]
    with _patched_dependency_contract(dependencies), pytest.raises(FileExistsError):
        heldout.write_preregistration(
            destination,
            release_verifier=lambda: receipt,
        )


def test_heldout_default_preregistration_is_absent_and_release_failure_writes_nothing(
    tmp_path: Path,
) -> None:
    destination = tmp_path / "must_not_exist.json"

    def unavailable() -> Mapping[str, Any]:
        raise RuntimeError("static V96 release is not promoted")

    assert not heldout.DEFAULT_PREREGISTRATION.exists()
    with pytest.raises(RuntimeError, match="not promoted"):
        heldout.write_preregistration(destination, release_verifier=unavailable)
    assert not destination.exists()


def test_heldout_scorer_rejects_incomplete_runtime_inventory_before_oracle_open(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    receipt = _release_receipt()
    dependencies = _dependency_contract()
    prereg = tmp_path / "prereg.json"
    with _patched_dependency_contract(dependencies):
        heldout.write_preregistration(
            prereg,
            release_verifier=lambda: receipt,
        )
    oracle_reads: list[Path] = []
    original = heldout._read_object

    def recording_read(path: Path) -> dict[str, Any]:
        if path.name == "oracle.json":
            oracle_reads.append(path)
        return original(path)

    monkeypatch.setattr(heldout, "_read_object", recording_read)
    with _patched_dependency_contract(dependencies), pytest.raises(
        ValueError, match="result inventory"
    ):
        heldout.score_heldout_results(
            prereg,
            {},
            oracle_root=tmp_path / "oracle",
            release_verifier=lambda: receipt,
        )
    assert oracle_reads == []


def test_v96_transfer_rejects_a_byte_identical_caller_selected_copy(
    tmp_path: Path,
) -> None:
    copied = tmp_path / "copied_policy"
    shutil.copytree(release_action.V3_NAVIGATION_CHECKPOINT, copied)

    with pytest.raises(ValueError, match="exact immutable"):
        load_v96_sequence_length_transfer(
            copied,
            expected_model_id="google/gemma-4-E2B-it",
            expected_model_revision="3e22461f65e89153144f8adb70e3b8c2cc9845a7",
            device="cpu",
        )


def test_dependency_contract_rejects_self_consistent_nested_tampering() -> None:
    tampered = copy.deepcopy(_dependency_contract())
    rows = tampered["navigation_policy"]["files"]
    rows[0]["sha256"] = _digest("0")
    tampered["navigation_policy"]["inventory_sha256"] = _canonical_sha256(rows)
    tampered.pop("contract_sha256")
    tampered["contract_sha256"] = _canonical_sha256(tampered)

    with pytest.raises(ValueError, match="navigation policy"):
        heldout.validate_embodied_dependency_contract(tampered)


def test_runtime_task_inputs_expose_only_scene_task_instruction_and_limit(
    tmp_path: Path,
) -> None:
    receipt = _release_receipt()
    dependencies = _dependency_contract()
    payload = _test_preregistration_payload(receipt, dependencies)
    runtime_payload = heldout.runtime_task_input_payload(
        "scene_000025",
        payload["tasks"],
    )
    assert set(runtime_payload) == {"schema", "scene_id", "tasks"}
    assert all(
        set(row) == {"scene_id", "task_id", "instruction", "max_steps"}
        for row in runtime_payload["tasks"]
    )
    serialized = json.dumps(runtime_payload, sort_keys=True)
    for forbidden in (
        "oracle_category",
        "target_xyz",
        "expected_answer",
        "answer_type",
        '"kind"',
        '"key"',
    ):
        assert forbidden not in serialized

    root = tmp_path / "runtime_inputs"
    manifest = heldout.write_runtime_task_inputs(
        root,
        payload,
        preregistration_sha256=_digest("9"),
    )
    assert len(manifest["scenes"]) == 6
    assert heldout._runtime_input_manifest(
        root,
        payload,
        preregistration_sha256=_digest("9"),
    ) == manifest


def test_runtime_child_has_no_scorer_import_or_category_mapping() -> None:
    source = Path(
        "src/semantic_3d_chat/robot/v96_embodied_task_runner.py"
    ).read_text(encoding="utf-8")
    assert "from semantic_3d_chat.evaluation" not in source
    assert "import semantic_3d_chat.evaluation" not in source
    assert "_SCORER_ORACLE_CATEGORY_BY_KEY" not in source
    assert "oracle_category" not in source


def test_runtime_source_contract_closes_first_party_import_graph_without_scorer() -> None:
    paths = set(heldout._RUNTIME_SOURCE_PATHS)
    assert len(paths) >= 110
    assert {
        "blender/render_runtime_observation.py",
        "blender/runtime_scene_contract.py",
        "blender/scene_utils.py",
        "src/semantic_3d_chat/language/gemma4_backend.py",
        "src/semantic_3d_chat/mapping/fusion.py",
        "src/semantic_3d_chat/robot/v96_embodied_task_runner.py",
        "src/semantic_3d_chat/robot/v96_runtime_source_contract.py",
        "src/semantic_3d_chat/scene_encoder/map_io.py",
        "src/semantic_3d_chat/vision/gemma4_encoder.py",
    }.issubset(paths)
    assert (
        "src/semantic_3d_chat/evaluation/v96_embodied_heldout.py" not in paths
    )


def test_parent_runtime_command_cannot_select_navigation_checkpoint() -> None:
    receipt = _release_receipt()
    dependencies = _dependency_contract()
    preregistration = _test_preregistration_payload(receipt, dependencies)
    command = heldout._runtime_child_command(
        scene_id="scene_000025",
        task_input=Path("/tmp/t.json"),
        output=Path("/tmp/output"),
        scratch=Path("/tmp/scratch"),
        preregistration=preregistration,
        preregistration_sha256=_digest("9"),
    )
    assert "--navigation-checkpoint" not in command
    assert "oracle" not in " ".join(command).casefold()


def test_strict_runtime_evidence_recomputes_scan_transition_and_bindings(
    tmp_path: Path,
) -> None:
    evidence, expected, written, _preregistration, runtime_config = (
        _valid_scan_evidence(tmp_path)
    )
    path = tmp_path / "evidence.json"
    path.write_text(json.dumps(evidence, sort_keys=True), encoding="utf-8")

    validated = heldout._validate_runtime_evidence(
        path,
        expected,
        preregistration_payload=written["payload"],
        preregistration_sha256=written["sha256"],
        runtime_config=runtime_config,
    )
    assert validated["successful_scan_count"] == 1
    assert validated["refreshed_scan_count"] == 1


@pytest.mark.parametrize(
    "mutation",
    [
        "instruction",
        "proposal",
        "call",
        "forward",
        "access_log",
        "access_log_type",
        "receipt_type",
        "release_binding",
        "legacy_schema",
    ],
)
def test_strict_runtime_evidence_rejects_self_consistent_forgery(
    tmp_path: Path,
    mutation: str,
) -> None:
    evidence, expected, written, _preregistration, runtime_config = (
        _valid_scan_evidence(tmp_path)
    )
    value = copy.deepcopy(evidence)
    step = value["navigation"]["steps"][0]
    if mutation == "instruction":
        value["navigation"]["instruction_sha256"] = _digest("0")
    elif mutation == "proposal":
        step["proposal_sha256"] = _digest("0")
    elif mutation == "call":
        step["call"] = {"tool": "stop", "arguments": {}}
        canonical = json.dumps(
            step["call"], sort_keys=True, separators=(",", ":")
        )
        digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
        step["call_sha256"] = digest
        step["proposal_sha256"] = digest
    elif mutation == "forward":
        step["policy_context_audit"]["forward"]["scene_tokens_processed"] = 737
    elif mutation == "access_log":
        loaded = value["runtime_access_log"]["loaded_files"][:-1]
        value["runtime_access_log"]["loaded_files"] = loaded
        value["runtime_access_log"]["loaded_file_count"] = len(loaded)
        value["runtime_access_log"]["loaded_file_inventory_sha256"] = (
            _canonical_sha256(loaded)
        )
    elif mutation == "access_log_type":
        value["runtime_access_log"]["loaded_file_count"] = float(
            value["runtime_access_log"]["loaded_file_count"]
        )
    elif mutation == "receipt_type":
        step["before_receipt"]["action_count"] = False
    elif mutation == "release_binding":
        value["release_receipt_sha256"] = _digest("0")
    elif mutation == "legacy_schema":
        value["schema"] = "semantic_3d_chat.v96_embodied_navigation_runtime_evidence.v1"
    else:  # pragma: no cover - parameterization is exhaustive
        raise AssertionError(mutation)
    _rehash_runtime_evidence(value)
    path = tmp_path / f"forged_{mutation}.json"
    path.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")

    with pytest.raises((TypeError, ValueError)):
        heldout._validate_runtime_evidence(
            path,
            expected,
            preregistration_payload=written["payload"],
            preregistration_sha256=written["sha256"],
            runtime_config=runtime_config,
        )
