from __future__ import annotations

import ast
import copy
import inspect
from pathlib import Path

import pytest
import torch

from semantic_3d_chat.chat import v94_strict_multiscene_cli as cli
from semantic_3d_chat.chat import v94_strict_multiscene_runtime as runtime
from semantic_3d_chat.chat.v83_direct_scene_memory_runtime import (
    V83DirectSceneMemoryChatRuntime,
)
from semantic_3d_chat.scene_encoder.v81_scene_memory_artifact import (
    LoadedV81SceneMemory,
)


def _valid_contract(*, promoted: bool = False) -> tuple[dict, dict]:
    config_banks: dict[str, dict] = {}
    metadata_banks: list[dict] = []
    states: dict[str, str] = {}
    modules: dict[str, list[str]] = {}
    counts: dict[str, dict[str, int]] = {}
    for spec in runtime._BANK_SPECS:
        row = {
            "trainable": False,
            "rank": spec.rank,
            "alpha": spec.alpha,
            "dropout": 0.0,
            "initialization_algorithm": "checkpoint_overwrite",
            "initialization_seed": None,
            "expected_initial_state_sha256": spec.state_sha256,
            "target_modules": list(spec.targets),
        }
        config_banks[spec.name] = dict(row)
        metadata_banks.append(
            {
                "name": spec.name,
                **row,
                "adapter_parameter_count": spec.parameter_count,
            }
        )
        states[spec.name] = spec.state_sha256
        modules[spec.name] = list(spec.targets)
        counts[spec.name] = {target: 1 for target in spec.targets}
        counts[spec.name][spec.targets[0]] += spec.parameter_count - len(spec.targets)
    decision = (
        runtime.PROMOTED_DECISION if promoted else runtime.CANDIDATE_DECISION
    )
    config = {"language": {"lora_banks": config_banks}}
    metadata = {
        "lora": {
            "schema_version": 2,
            "enabled": True,
            "banks": metadata_banks,
            "adapter_parameter_count": runtime.EXPECTED_ADAPTER_PARAMETER_COUNT,
            "trainable_adapter_parameter_count": 0,
        },
        "lora_parameter_count": runtime.EXPECTED_ADAPTER_PARAMETER_COUNT,
        "lora_trainable_parameter_count": 0,
        "lora_bank_state_sha256": states,
        "lora_bank_wrapped_modules": modules,
        "lora_bank_parameter_counts": counts,
        "question_dependent_scene_processing": False,
        "initialization_provenance": {
            "v94_strict_runtime_release": {
                "schema_version": 94,
                "source_v94_evidence_sha256": "1" * 64,
                "source_v94_score_sha256": "2" * 64,
                "v94_bridge_state_sha256": runtime.V94_STATE_SHA256,
                "model_acceptance_gate_passed": True,
                "model_gate_report_authenticated": True,
                "promotion_decision": decision,
                "runtime_promotion_authorized": promoted,
                "smoke_report_sha256": "3" * 64 if promoted else None,
                "held_out_generalization_claim": True,
            }
        },
    }
    return config, metadata


def _loaded_memory(scene_id: str = "scene_000057") -> LoadedV81SceneMemory:
    return LoadedV81SceneMemory(
        root=Path("numeric-memory"),
        memory=torch.zeros((1, 738, 1536), dtype=torch.bfloat16),
        metadata={
            "scene_id": scene_id,
            "shape": [1, 738, 1536],
            "fixed_memory_tokens": 738,
            "hidden_size": 1536,
            "compiled_before_user_question": True,
            "question_inputs_used_for_compilation": False,
            "question_dependent_scene_processing": False,
            "question_dependent_retrieval": False,
            "semantic_or_spatial_top_k_selection": False,
            "environmental_text_inputs": [],
            "questions_or_answers_serialized": False,
            "oracle_loaded": False,
        },
    )


def test_v94_contract_accepts_only_explicit_candidate_or_promoted_provenance() -> None:
    candidate = runtime.validate_v94_runtime_contract(
        runtime_config=_valid_contract()[0],
        checkpoint_metadata=_valid_contract()[1],
    )
    promoted_config, promoted_metadata = _valid_contract(promoted=True)
    promoted = runtime.validate_v94_runtime_contract(
        runtime_config=promoted_config,
        checkpoint_metadata=promoted_metadata,
    )

    assert candidate["runtime_package_mode"] == "candidate"
    assert candidate["runtime_promotion_authorized"] is False
    assert promoted["runtime_package_mode"] == "promoted"
    assert promoted["runtime_promotion_authorized"] is True
    assert promoted["frozen_lora_bank_count"] == 8
    assert promoted["adapter_parameter_count"] == 675_840


@pytest.mark.parametrize(
    "mutation",
    ("reordered", "trainable", "wrong_state", "wrong_count", "bad_provenance"),
)
def test_v94_contract_rejects_stack_or_release_drift(mutation: str) -> None:
    config, metadata = _valid_contract()
    if mutation == "reordered":
        metadata["lora"]["banks"][0], metadata["lora"]["banks"][1] = (
            metadata["lora"]["banks"][1],
            metadata["lora"]["banks"][0],
        )
    elif mutation == "trainable":
        metadata["lora"]["banks"][-1]["trainable"] = True
    elif mutation == "wrong_state":
        metadata["lora_bank_state_sha256"][runtime.V94_BANK] = "4" * 64
    elif mutation == "wrong_count":
        metadata["lora_parameter_count"] -= 1
    else:
        release = metadata["initialization_provenance"]["v94_strict_runtime_release"]
        release["runtime_promotion_authorized"] = True

    with pytest.raises(ValueError):
        runtime.validate_v94_runtime_contract(
            runtime_config=config, checkpoint_metadata=metadata
        )


def test_v94_numeric_memory_contract_accepts_complete_text_free_memory() -> None:
    runtime.validate_v94_scene_memory_contract(
        scene_id="scene_000057", loaded=_loaded_memory()
    )


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("question_inputs_used_for_compilation", True),
        ("question_dependent_retrieval", True),
        ("semantic_or_spatial_top_k_selection", True),
        ("environmental_text_inputs", ["forbidden"]),
        ("questions_or_answers_serialized", True),
        ("oracle_loaded", True),
    ),
)
def test_v94_numeric_memory_rejects_leakage_or_question_selection(
    field: str, value: object
) -> None:
    loaded = _loaded_memory()
    loaded.metadata[field] = value
    with pytest.raises(ValueError, match="oracle-free"):
        runtime.validate_v94_scene_memory_contract(
            scene_id="scene_000057", loaded=loaded
        )


def test_v94_cli_selects_candidate_only_through_explicit_flag() -> None:
    defaults = cli._parser().parse_args([])
    candidate = cli._parser().parse_args(["--allow-candidate"])
    default_checkpoint, default_memory = cli._selected_paths(defaults)
    candidate_checkpoint, candidate_memory = cli._selected_paths(candidate)

    assert defaults.allow_candidate is False
    assert default_checkpoint.as_posix().endswith(cli.RELEASE_CHECKPOINT)
    assert default_memory.as_posix().endswith(
        f"{cli.RELEASE_MEMORY_ROOT}/{cli.DEFAULT_SCENE}"
    )
    assert candidate.allow_candidate is True
    assert candidate_checkpoint.as_posix().endswith(cli.CANDIDATE_CHECKPOINT)
    assert candidate_memory.as_posix().endswith(
        f"{cli.CANDIDATE_MEMORY_ROOT}/{cli.DEFAULT_SCENE}"
    )


def test_v94_runtime_is_direct_v83_and_has_no_offline_import_surface() -> None:
    assert issubclass(
        runtime.V94StrictMultisceneChatRuntime, V83DirectSceneMemoryChatRuntime
    )
    tree = ast.parse(inspect.getsource(runtime))
    imported = {
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module is not None
    }
    assert not any(
        module.startswith(
            ("semantic_3d_chat.evaluation", "semantic_3d_chat.training")
        )
        for module in imported
    )
    source = inspect.getsource(runtime).casefold()
    assert "data_diverse52" not in source
    assert "train.jsonl" not in source
    assert "validation.jsonl" not in source


def test_v94_forbidden_roots_cover_oracle_qa_training_and_scorer() -> None:
    collapsed = "\n".join(path.as_posix().casefold() for path in cli._forbidden_roots())

    assert "/oracle" in collapsed
    assert "/qa" in collapsed
    assert "v94_strict_multiscene_full40_training.json" in collapsed
    assert "/scorer_only" in collapsed
    assert "/predictions" in collapsed


def test_v94_promoted_provenance_requires_authenticated_smoke_hash() -> None:
    config, metadata = _valid_contract(promoted=True)
    broken = copy.deepcopy(metadata)
    broken["initialization_provenance"]["v94_strict_runtime_release"][
        "smoke_report_sha256"
    ] = None

    with pytest.raises(ValueError, match="authenticated release gate"):
        runtime.validate_v94_runtime_contract(
            runtime_config=config, checkpoint_metadata=broken
        )
