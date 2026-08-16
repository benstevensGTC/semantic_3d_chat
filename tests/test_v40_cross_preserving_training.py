from __future__ import annotations

import ast
import copy
import json
from collections import Counter
from pathlib import Path

import pytest
import torch

from semantic_3d_chat.config import PROJECT_ROOT, load_config
from semantic_3d_chat.training import train_cross_preserving_v40 as v40

CONFIG_PATH = PROJECT_ROOT / v40.DEFAULT_CONFIG


def _config() -> dict:
    return load_config(CONFIG_PATH)


def test_v40_contract_is_exact_b_only_sgd() -> None:
    config = _config()
    settings = v40.v40_settings(config)
    contract = v40.v40_contract(config)
    assert settings.learning_rate == 0.003
    assert settings.cross_prefix_flip_weight == 56.0
    assert settings.broad_nll_weight == 1.0
    assert settings.pair_correct_nll_weight == 0.5
    assert contract.saved_optimizer_steps == (0, 8, 16, 24, 32, 40, 41)
    assert contract.diagnostic_steps == (0, 8, 16, 41)
    assert contract.query_source_state_sha256 == v40._QUERY_SOURCE_STATE_SHA256
    assert v40._QUERY_PARAMETER_NAMES == (
        "lora_banks.extension_v28_stage_b_query.adapters.1.lora_b",
    )
    assert v40._QUERY_SHAPES == ((4096, 4),)


def test_v40_construction_copy_restores_v28_source_pin_without_serializing_it() -> None:
    config = _config()
    assert (
        config["language"]["lora_banks"]["extension_v28_stage_b_query"][
            "expected_initial_state_sha256"
        ]
        is None
    )
    loader = v40.v40_loader_config(config)
    assert loader is not config
    assert (
        loader["language"]["lora_banks"]["extension_v28_stage_b_query"][
            "expected_initial_state_sha256"
        ]
        == v40._V28_BANK_STATE_SHA256
    )
    assert (
        config["language"]["lora_banks"]["extension_v28_stage_b_query"][
            "expected_initial_state_sha256"
        ]
        is None
    )


def test_v40_terminal_and_no_model_preflight_are_exact() -> None:
    config = _config()
    terminal = v40.require_v39_terminal_gate(config)
    report = v40.preflight_v40(config)
    assert terminal["sha256"] == (
        "fcf0494c18ed13c3f1fe54eb109a51391183a4eeb14abb9dbd2ad0ad0ca448c3"
    )
    assert report["passed"] is True
    assert report["gemma_loaded"] is False
    assert report["scene_maps_loaded"] is False
    assert report["validation_qa_loaded"] is False
    assert report["exact_trainable_tensor_count"] == 1
    assert report["exact_trainable_parameter_count"] == 16_384


def test_v40_source_authenticates_b_and_all_178_frozen_tensors() -> None:
    tensors, _metadata, audit = v40.require_exact_v40_sources(_config())
    assert len(tensors) == 179
    assert audit["source_tensor_hashes"] == {
        "full": v40._V38_FULL_STATE_SHA256,
        "target": v40._QUERY_SOURCE_STATE_SHA256,
        "v28_bank": v40._V28_BANK_STATE_SHA256,
        "v23_bank": v40._HYBRID_V23_STATE_SHA256,
        "block_core": v40._CORE_STATE_SHA256,
        "frozen_excluding_target": v40._FROZEN_STATE_SHA256,
    }
    assert len(v40._frozen_excluding_query(tensors)) == 178


def test_v40_rejects_draft_update4_schedule() -> None:
    config = _config()
    config["v40_cross_preserving"]["saved_optimizer_steps"] = [
        0,
        4,
        8,
        16,
        24,
        32,
        40,
        41,
    ]
    with pytest.raises(ValueError, match="contract changed"):
        v40.v40_contract(config)


def test_v40_rejects_adam_like_objective_or_output_alias(tmp_path: Path) -> None:
    config = _config()
    changed = copy.deepcopy(config)
    changed["training"]["v40_cross_preserving"]["learning_rate"] = 2e-5
    with pytest.raises(ValueError, match="terminal lock"):
        v40.v40_settings(changed)
    with pytest.raises(ValueError, match="output root"):
        v40.run_v40(config=config, output=tmp_path / "alias")


def test_component_gradient_guard_accepts_shared_direction() -> None:
    components = {
        "broad": (torch.tensor([1.0, 0.0]),),
        "answer": (torch.tensor([0.5, 0.0]),),
        "side": (torch.tensor([0.25, 0.0]),),
        "cross": (torch.tensor([0.75, 0.0]),),
    }
    total, audit = v40.component_gradient_guard(components)
    assert torch.equal(total[0], torch.tensor([2.5, 0.0]))
    assert audit["raw_guard_passed"] is True
    assert all(
        row["strictly_positive_if_nonzero"]
        for row in audit["directional_checks"].values()
    )


def test_gradient_audit_promotes_to_float64_only_after_leaving_mps() -> None:
    if not torch.backends.mps.is_built() or not torch.backends.mps.is_available():
        pytest.skip("MPS is unavailable")
    gradient = torch.tensor([1.0, 2.0], device="mps")
    vector = v40._gradient_vector((gradient,))
    assert vector.device.type == "cpu"
    assert vector.dtype == torch.float64
    assert torch.equal(vector, torch.tensor([1.0, 2.0], dtype=torch.float64))


def test_component_gradient_guard_fails_before_mutation_on_conflict() -> None:
    components = {
        "broad": (torch.tensor([-1.0, 0.0]),),
        "answer": (torch.tensor([1.0, 0.0]),),
        "side": (torch.tensor([1.0, 0.0]),),
        "cross": (torch.tensor([1.0, 0.0]),),
    }
    with pytest.raises(v40.V40GradientGuardFailure) as caught:
        v40.component_gradient_guard(components)
    assert caught.value.audit["raw_guard_passed"] is False
    assert (
        caught.value.audit["directional_checks"]["broad"][
            "strictly_positive_if_nonzero"
        ]
        is False
    )


def test_nonfinite_and_zero_guard_failures_are_persistable() -> None:
    components = {
        "broad": (torch.tensor([float("nan")]),),
        "answer": (torch.tensor([0.0]),),
        "side": (torch.tensor([0.0]),),
        "cross": (torch.tensor([0.0]),),
    }
    with pytest.raises(v40.V40GradientGuardFailure) as caught:
        v40.component_gradient_guard(components)
    assert caught.value.audit["guard_stage"] == "raw_component_direction"
    assert caught.value.audit["component_norms"]["broad"] is None
    json.dumps(caught.value.audit, allow_nan=False)

    parameter = torch.nn.Parameter(torch.zeros(1))
    zeros = {name: (torch.zeros(1),) for name in ("broad", "answer", "side", "cross")}
    with pytest.raises(v40.V40GradientGuardFailure) as clip_failure:
        v40.clip_direction_attestation(
            parameters=(parameter,),
            raw_total=(torch.zeros(1),),
            components=zeros,
            clip_norm=1.0,
        )
    assert clip_failure.value.audit["guard_stage"] == "scalar_global_clip"
    json.dumps(clip_failure.value.audit, allow_nan=False)


def test_scalar_clip_preserves_all_positive_directions() -> None:
    parameter = torch.nn.Parameter(torch.zeros(2))
    components = {
        "broad": (torch.tensor([3.0, 0.0]),),
        "answer": (torch.tensor([2.0, 0.0]),),
        "side": (torch.tensor([1.0, 0.0]),),
        "cross": (torch.tensor([4.0, 0.0]),),
    }
    total, _guard = v40.component_gradient_guard(components)
    audit = v40.clip_direction_attestation(
        parameters=(parameter,),
        raw_total=total,
        components=components,
        clip_norm=1.0,
    )
    assert audit["scalar_clip_direction_preserved"] is True
    assert audit["raw_total_norm"] == 10.0
    assert audit["clipped_total_norm"] == pytest.approx(1.0)
    assert audit["observed_scalar"] == pytest.approx(0.1)


def test_guard_failure_artifact_is_atomic_and_fail_stop(tmp_path: Path) -> None:
    path = v40.persist_gradient_guard_failure(
        tmp_path,
        optimizer_step=3,
        audit={"raw_guard_passed": False},
    )
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["optimizer_step_not_executed"] == 3
    assert payload["optimizer_step_executed"] is False
    assert payload["checkpoint_written"] is False
    with pytest.raises(FileExistsError):
        v40.persist_gradient_guard_failure(
            tmp_path,
            optimizer_step=3,
            audit={"raw_guard_passed": False},
        )
    with pytest.raises(RuntimeError, match="forbids resume"):
        v40.latest_v40_resume_checkpoint(tmp_path, v40.v40_contract(_config()))


def test_sgd_payload_has_one_stateless_parameter_and_exact_defaults() -> None:
    parameter = torch.nn.Parameter(torch.zeros(4096, 4))
    optimizer = torch.optim.SGD(
        [
            {
                "name": "lora_banks.extension_v28_stage_b_query.adapters.1",
                "params": [parameter],
                "parameter_names": list(v40._QUERY_PARAMETER_NAMES),
                "lr": 0.003,
                "weight_decay": 0.0,
                "momentum": 0.0,
                "dampening": 0.0,
                "nesterov": False,
            }
        ],
        foreach=False,
        fused=False,
    )
    audit = v40._optimizer_payload_audit(
        optimizer.state_dict(),
        expected_step=8,
        tensors={v40._QUERY_PARAMETER_NAMES[0]: parameter.detach()},
    )
    assert audit["moment_tensor_count"] == 0
    assert audit["momentum_free_stateless_sgd_verified"] is True


def test_v40_module_has_unique_top_level_definitions_and_no_stale_surface() -> None:
    source_path = Path(v40.__file__)
    source = source_path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    names = [
        node.name
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
    ]
    duplicates = {name for name, count in Counter(names).items() if count > 1}
    assert duplicates == set()
    assert "131_072" not in source
    assert "131072" not in source
    assert "torch.optim.Adam" not in source
    assert "_adam" not in source.lower()
