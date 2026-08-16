from __future__ import annotations

import ast
import copy
import inspect
import json
from typing import Any

import pytest
import yaml

from semantic_3d_chat.chat import (
    v89_strict_scene1_runtime,
    v91_strict_scene1_cli,
    v91_strict_scene1_runtime,
)
from semantic_3d_chat.chat.runtime import ChatAnswer
from semantic_3d_chat.chat.v83_direct_scene_memory_runtime import (
    V83DirectSceneMemoryChatRuntime,
)
from semantic_3d_chat.chat.v91_strict_scene1_cli import _forbidden_roots, _parser, _run
from semantic_3d_chat.chat.v91_strict_scene1_runtime import (
    EXPECTED_ADAPTER_PARAMETER_COUNT,
    EXPECTED_BANKS,
    PROMOTION_DECISION,
    V90_BANK,
    V90_PARAMETER_COUNT,
    V90_STATE_SHA256,
    V90_TARGET,
    V91_ALPHA,
    V91_BANK,
    V91_PARAMETER_COUNT,
    V91_RANK,
    V91_TARGET,
    V91StrictScene1ChatRuntime,
    validate_v91_runtime_contract,
)
from semantic_3d_chat.config import PROJECT_ROOT


def _runtime_surfaces(state: str = "a" * 64) -> tuple[dict[str, Any], dict[str, Any]]:
    config = copy.deepcopy(
        yaml.safe_load(
            (PROJECT_ROOT / "configs/runtime/gemma4_v89_strict_scene1.yaml").read_text(
                encoding="utf-8"
            )
        )
    )
    metadata = copy.deepcopy(
        json.loads(
            (
                PROJECT_ROOT
                / "data_gemma4/runtime/checkpoints/gemma4_v89_strict_scene1_release_v1"
                / "runtime_metadata.json"
            ).read_text(encoding="utf-8")
        )
    )
    additions = (
        (V90_BANK, V90_TARGET, 8, 16.0, V90_PARAMETER_COUNT, V90_STATE_SHA256),
        (V91_BANK, V91_TARGET, V91_RANK, V91_ALPHA, V91_PARAMETER_COUNT, state),
    )
    for name, target, rank, alpha, count, digest in additions:
        config["language"]["lora_banks"][name] = {
            "trainable": False,
            "rank": rank,
            "alpha": alpha,
            "dropout": 0.0,
            "initialization_algorithm": "checkpoint_overwrite",
            "initialization_seed": None,
            "expected_initial_state_sha256": digest,
            "target_modules": [target],
        }
        metadata["lora"]["banks"].append(
            {
                "name": name,
                "trainable": False,
                "rank": rank,
                "alpha": alpha,
                "dropout": 0.0,
                "initialization_algorithm": "checkpoint_overwrite",
                "initialization_seed": None,
                "expected_initial_state_sha256": digest,
                "target_modules": [target],
                "adapter_parameter_count": count,
            }
        )
        metadata["lora_bank_state_sha256"][name] = digest
        metadata["lora_bank_wrapped_modules"][name] = [target]
        metadata["lora_bank_parameter_counts"][name] = {target: count}
    metadata["lora"]["adapter_parameter_count"] = EXPECTED_ADAPTER_PARAMETER_COUNT
    metadata["lora"]["trainable_adapter_parameter_count"] = 0
    metadata["lora_parameter_count"] = EXPECTED_ADAPTER_PARAMETER_COUNT
    metadata["lora_trainable_parameter_count"] = 0
    metadata["initialization_provenance"]["v91_strict_runtime_release"] = {
        "schema_version": 91,
        "experiment_config_sha256": "1" * 64,
        "preregistration_sha256": "2" * 64,
        "cpu_preflight_sha256": "3" * 64,
        "training_report_sha256": "4" * 64,
        "model_gate_report_sha256": "5" * 64,
        "evaluation_predictions_sha256": "6" * 64,
        "v90_bridge_state_sha256": V90_STATE_SHA256,
        "v91_bridge_state_sha256": state,
        "promotion_decision": PROMOTION_DECISION,
        "runtime_promotion_authorized": True,
        "model_acceptance_gate_passed": True,
        "model_gate_report_authenticated": True,
        "held_out_generalization_claim": False,
    }
    return config, metadata


def _direct_imports(module: object) -> set[str]:
    tree = ast.parse(inspect.getsource(module))
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module)
    return names


def test_v91_runtime_is_exact_v89_plus_frozen_v90_plus_dynamic_repair() -> None:
    config, metadata = _runtime_surfaces()
    result = validate_v91_runtime_contract(
        scene_id="scene_000001",
        runtime_config=config,
        checkpoint_metadata=metadata,
    )

    assert EXPECTED_BANKS[:11] == v89_strict_scene1_runtime.EXPECTED_BANKS
    assert EXPECTED_BANKS[-2:] == (V90_BANK, V91_BANK)
    assert len(EXPECTED_BANKS) == 13
    assert EXPECTED_ADAPTER_PARAMETER_COUNT == 1_122_304
    assert (V91_TARGET, V91_RANK, V91_ALPHA, V91_PARAMETER_COUNT) == (
        "model.language_model.layers.33.mlp.down_proj",
        16,
        32.0,
        221_184,
    )
    assert result["v91_bridge_state_sha256"] == "a" * 64
    assert result["frozen_lora_bank_count"] == 13
    assert result["runtime_promotion_authorized"] is True
    assert issubclass(V91StrictScene1ChatRuntime, V83DirectSceneMemoryChatRuntime)


def test_v91_runtime_rejects_v90_drift_and_unpromoted_v91() -> None:
    config, metadata = _runtime_surfaces()
    metadata["lora_bank_state_sha256"][V90_BANK] = "b" * 64
    with pytest.raises(ValueError, match="exact failed V90"):
        validate_v91_runtime_contract(
            scene_id="scene_000001",
            runtime_config=config,
            checkpoint_metadata=metadata,
        )

    config, metadata = _runtime_surfaces()
    metadata["initialization_provenance"]["v91_strict_runtime_release"][
        "promotion_decision"
    ] = "candidate"
    with pytest.raises(ValueError, match="promoted conversational gate"):
        validate_v91_runtime_contract(
            scene_id="scene_000001",
            runtime_config=config,
            checkpoint_metadata=metadata,
        )


def test_v91_runtime_reuses_one_exact_environment_input_hash(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fixed_answer(self: object, question: str) -> ChatAnswer:
        self.last_prepared_layout_audit = {"question_derived_environmental_tokens": 0}
        return ChatAnswer(
            question=question,
            answer="ok",
            grounding_xyz_m=(0.0, 0.0, 0.0),
            grounding_confidence=0.0,
            grounding_support_distance_m=0.0,
            prefix_hash=self.scene_prefix_hash,
            generated_tokens=1,
            elapsed_seconds=0.0,
        )

    monkeypatch.setattr(V83DirectSceneMemoryChatRuntime, "answer", fixed_answer)
    runtime = object.__new__(V91StrictScene1ChatRuntime)
    runtime.scene_prefix_hash = "c" * 64
    runtime.environment_conditioned_input_hashes = []
    runtime.last_prepared_layout_audit = None
    runtime.answer("first")
    runtime.answer("second")
    assert runtime.environment_conditioned_input_hashes == ["c" * 64, "c" * 64]


def test_v91_runtime_and_cli_import_no_training_or_evaluation_surface() -> None:
    for module in (v91_strict_scene1_runtime, v91_strict_scene1_cli):
        imports = _direct_imports(module)
        assert not any(
            name.startswith(("semantic_3d_chat.training", "semantic_3d_chat.evaluation"))
            for name in imports
        )


def test_v91_cli_is_release_only_and_blocks_both_offline_bridges() -> None:
    defaults = _parser().parse_args([])
    assert defaults.config == "configs/runtime/gemma4_v91_strict_scene1.yaml"
    assert defaults.base_checkpoint.endswith("gemma4_v91_strict_scene1_release_v1")
    assert defaults.scene_memory.endswith("runtime/scene_memories/v91/scene_000001")
    with pytest.raises(ValueError, match="only scene_000001"):
        _run(["--scene", "scene_000039", "--question", "anything"])
    forbidden = {path.relative_to(PROJECT_ROOT).as_posix() for path in _forbidden_roots()}
    assert "reports/gemma4/artifacts/v90_scene1_conversational_final" in forbidden
    assert (
        "reports/gemma4/artifacts/v91_scene1_conversational_repair_final_v2"
        in forbidden
    )
