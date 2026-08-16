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
    v90_strict_scene1_cli,
    v90_strict_scene1_runtime,
)
from semantic_3d_chat.chat.runtime import ChatAnswer
from semantic_3d_chat.chat.v83_direct_scene_memory_runtime import (
    V83DirectSceneMemoryChatRuntime,
)
from semantic_3d_chat.chat.v90_strict_scene1_cli import _forbidden_roots, _parser, _run
from semantic_3d_chat.chat.v90_strict_scene1_runtime import (
    EXPECTED_ADAPTER_PARAMETER_COUNT,
    EXPECTED_BANKS,
    PROMOTION_DECISION,
    V90_ALPHA,
    V90_BANK,
    V90_PARAMETER_COUNT,
    V90_RANK,
    V90_TARGET,
    V90StrictScene1ChatRuntime,
    validate_v90_runtime_contract,
)
from semantic_3d_chat.config import PROJECT_ROOT


def _runtime_surfaces(state: str = "a" * 64) -> tuple[dict[str, Any], dict[str, Any]]:
    config_path = PROJECT_ROOT / "configs/runtime/gemma4_v89_strict_scene1.yaml"
    metadata_path = (
        PROJECT_ROOT
        / "data_gemma4/runtime/checkpoints/gemma4_v89_strict_scene1_release_v1"
        / "runtime_metadata.json"
    )
    config = copy.deepcopy(yaml.safe_load(config_path.read_text(encoding="utf-8")))
    metadata = copy.deepcopy(json.loads(metadata_path.read_text(encoding="utf-8")))
    config["language"]["lora_banks"][V90_BANK] = {
        "trainable": False,
        "rank": V90_RANK,
        "alpha": V90_ALPHA,
        "dropout": 0.0,
        "initialization_algorithm": "checkpoint_overwrite",
        "initialization_seed": None,
        "expected_initial_state_sha256": state,
        "target_modules": [V90_TARGET],
    }
    metadata["lora"]["banks"].append(
        {
            "name": V90_BANK,
            "trainable": False,
            "rank": V90_RANK,
            "alpha": V90_ALPHA,
            "dropout": 0.0,
            "initialization_algorithm": "checkpoint_overwrite",
            "initialization_seed": None,
            "expected_initial_state_sha256": state,
            "target_modules": [V90_TARGET],
            "adapter_parameter_count": V90_PARAMETER_COUNT,
        }
    )
    metadata["lora"]["adapter_parameter_count"] = EXPECTED_ADAPTER_PARAMETER_COUNT
    metadata["lora"]["trainable_adapter_parameter_count"] = 0
    metadata["lora_parameter_count"] = EXPECTED_ADAPTER_PARAMETER_COUNT
    metadata["lora_trainable_parameter_count"] = 0
    metadata["lora_bank_state_sha256"][V90_BANK] = state
    metadata["lora_bank_wrapped_modules"][V90_BANK] = [V90_TARGET]
    metadata["lora_bank_parameter_counts"][V90_BANK] = {
        V90_TARGET: V90_PARAMETER_COUNT
    }
    metadata["initialization_provenance"]["v90_strict_runtime_release"] = {
        "schema_version": 90,
        "experiment_config_sha256": "1" * 64,
        "preregistration_sha256": "2" * 64,
        "cpu_preflight_sha256": "3" * 64,
        "training_report_sha256": "4" * 64,
        "model_gate_report_sha256": "5" * 64,
        "evaluation_predictions_sha256": "6" * 64,
        "v90_bridge_state_sha256": state,
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


def test_v90_runtime_contract_is_exact_v89_parent_plus_frozen_layer28_bank() -> None:
    config, metadata = _runtime_surfaces()
    result = validate_v90_runtime_contract(
        scene_id="scene_000001",
        runtime_config=config,
        checkpoint_metadata=metadata,
    )

    assert EXPECTED_BANKS[:-1] == v89_strict_scene1_runtime.EXPECTED_BANKS
    assert EXPECTED_BANKS[-1] == V90_BANK
    assert len(EXPECTED_BANKS) == 12
    assert EXPECTED_ADAPTER_PARAMETER_COUNT == 901_120
    assert (V90_TARGET, V90_RANK, V90_ALPHA, V90_PARAMETER_COUNT) == (
        "model.language_model.layers.28.self_attn.o_proj",
        8,
        16.0,
        28_672,
    )
    assert result["v90_bridge_state_sha256"] == "a" * 64
    assert result["frozen_lora_bank_count"] == 12
    assert result["runtime_promotion_authorized"] is True
    assert issubclass(V90StrictScene1ChatRuntime, V83DirectSceneMemoryChatRuntime)


def test_v90_final_state_is_dynamic_lowercase_and_bound_identically() -> None:
    for state in ("a" * 64, "b" * 64):
        config, metadata = _runtime_surfaces(state)
        result = validate_v90_runtime_contract(
            scene_id="scene_000001",
            runtime_config=config,
            checkpoint_metadata=metadata,
        )
        assert result["v90_bridge_state_sha256"] == state

    config, metadata = _runtime_surfaces()
    metadata["lora_bank_state_sha256"][V90_BANK] = "A" * 64
    with pytest.raises(ValueError, match="lowercase SHA-256"):
        validate_v90_runtime_contract(
            scene_id="scene_000001",
            runtime_config=config,
            checkpoint_metadata=metadata,
        )

    config, metadata = _runtime_surfaces()
    metadata["initialization_provenance"]["v90_strict_runtime_release"][
        "v90_bridge_state_sha256"
    ] = "b" * 64
    with pytest.raises(ValueError, match="promoted conversational gate"):
        validate_v90_runtime_contract(
            scene_id="scene_000001",
            runtime_config=config,
            checkpoint_metadata=metadata,
        )


def test_v90_rejects_reordering_target_drift_and_unpromoted_provenance() -> None:
    config, metadata = _runtime_surfaces()
    banks = config["language"]["lora_banks"]
    config["language"]["lora_banks"] = {
        name: banks[name] for name in reversed(tuple(banks))
    }
    with pytest.raises(ValueError, match="ordered frozen 12-bank"):
        validate_v90_runtime_contract(
            scene_id="scene_000001",
            runtime_config=config,
            checkpoint_metadata=metadata,
        )

    config, metadata = _runtime_surfaces()
    config["language"]["lora_banks"][V90_BANK]["target_modules"] = [
        "model.language_model.layers.28.self_attn.q_proj"
    ]
    with pytest.raises(ValueError, match=V90_BANK):
        validate_v90_runtime_contract(
            scene_id="scene_000001",
            runtime_config=config,
            checkpoint_metadata=metadata,
        )

    config, metadata = _runtime_surfaces()
    metadata["initialization_provenance"]["v90_strict_runtime_release"][
        "promotion_decision"
    ] = "candidate"
    with pytest.raises(ValueError, match="promoted conversational gate"):
        validate_v90_runtime_contract(
            scene_id="scene_000001",
            runtime_config=config,
            checkpoint_metadata=metadata,
        )


def test_v90_runtime_reuses_one_exact_environment_input_hash(
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
    runtime = object.__new__(V90StrictScene1ChatRuntime)
    runtime.scene_prefix_hash = "c" * 64
    runtime.environment_conditioned_input_hashes = []
    runtime.last_prepared_layout_audit = None

    runtime.answer("first")
    runtime.answer("second")
    assert runtime.environment_conditioned_input_hashes == ["c" * 64, "c" * 64]

    def changed_answer(self: object, question: str) -> ChatAnswer:
        result = fixed_answer(self, question)
        return ChatAnswer(**{**result.to_dict(), "grounding_xyz_m": (0.0, 0.0, 0.0), "prefix_hash": "d" * 64})

    monkeypatch.setattr(V83DirectSceneMemoryChatRuntime, "answer", changed_answer)
    with pytest.raises(RuntimeError, match="environment-conditioned input changed"):
        runtime.answer("third")


def test_v90_runtime_and_cli_have_no_direct_training_or_evaluation_imports() -> None:
    for module in (v90_strict_scene1_runtime, v90_strict_scene1_cli):
        imports = _direct_imports(module)
        assert not any(
            name.startswith(
                ("semantic_3d_chat.training", "semantic_3d_chat.evaluation")
            )
            for name in imports
        )
    runtime_source = inspect.getsource(v90_strict_scene1_runtime).casefold()
    for forbidden in (
        "train_v90_scene1_conversational",
        "evaluate_v90_scene1_conversational",
        "v90_scene1_conversational_preflight",
        "reference_answer",
        "target_instance",
    ):
        assert forbidden not in runtime_source


def test_v90_cli_defaults_to_future_promoted_release_and_blocks_offline_artifacts() -> None:
    defaults = _parser().parse_args([])

    assert defaults.config == "configs/runtime/gemma4_v90_strict_scene1.yaml"
    assert defaults.base_checkpoint.endswith("gemma4_v90_strict_scene1_release_v1")
    assert defaults.scene_memory.endswith("runtime/scene_memories/v90/scene_000001")
    with pytest.raises(ValueError, match="only scene_000001"):
        _run(["--scene", "scene_000039", "--question", "anything"])

    forbidden = {path.relative_to(PROJECT_ROOT).as_posix() for path in _forbidden_roots()}
    assert {
        "configs/experiments/gemma4_v90_scene1_conversational.yaml",
        "data_gemma4/checkpoints/v90_scene1_conversational_work",
        "reports/gemma4/artifacts/v90_scene1_conversational_final",
        "reports/gemma4/predictions/gemma4_v90_scene1_conversational_evaluation.json",
        "reports/gemma4/metrics/gemma4_v90_scene1_conversational_preregistration.json",
        "reports/gemma4/metrics/gemma4_v90_scene1_conversational_cpu_preflight.json",
        "reports/gemma4/metrics/gemma4_v90_scene1_conversational_training.json",
        "reports/gemma4/metrics/gemma4_v90_scene1_conversational_evaluation.json",
    } <= forbidden
