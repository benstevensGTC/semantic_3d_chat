from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest
import torch

from semantic_3d_chat.config import PROJECT_ROOT, load_config
from semantic_3d_chat.training.train_projected_gradient_v41 import (
    V41GradientGuardFailure,
    clip_direction_attestation,
    persist_gradient_guard_failure,
    project_gradient_to_feasible_descent,
    raw_component_gradient_diagnostic,
    require_exact_v41_sources,
    require_v40_terminal_gate,
    v41_contract,
    validate_v41_projection_history,
)

CONFIG = PROJECT_ROOT / "configs/experiments/gemma4_diverse28_projected_gradient_v41.yaml"
TRAINER = (
    PROJECT_ROOT
    / "src/semantic_3d_chat/training/train_projected_gradient_v41.py"
)


def _components(*values: list[float], dtype: torch.dtype = torch.float64):
    names = ("broad", "answer", "side", "cross")
    return {
        name: (torch.tensor(value, dtype=dtype),)
        for name, value in zip(names, values, strict=True)
    }


def _conflicting_components(dtype: torch.dtype = torch.float64):
    return _components(
        [0.8217994362420474, -2.1159867824727114, 0.2359260754036501, 0.37107376188412144],
        [-1.0630841059676626, 0.7593422824832046, -1.4079728190401823, -1.727048416644511],
        [-0.14521085747635995, 0.16181963346027795, -0.07510561393558862, -1.2844032367410247],
        [-2.1386006282012957, -0.0196275299103969, -0.22622194132593756, 0.6829122498052708],
        dtype=dtype,
    )


def test_v41_terminal_and_exact_v40_update_zero_are_pinned() -> None:
    config = load_config(CONFIG)
    contract = v41_contract(config)
    terminal = require_v40_terminal_gate(config)
    tensors, _metadata, audit = require_exact_v41_sources(config)
    assert terminal["sha256"] == (
        "d4c30be9e4f685697478b6e5a37f4f55d6e99962484b1cbae5c3c3214c24b35e"
    )
    assert contract.source_checkpoint.name == "update_000"
    assert len(tensors) == 179
    assert audit["source_tensor_hashes"]["full"] == contract.source_tensor_state_sha256
    assert audit["source_optimizer_file_opened"] is False


def test_raw_feasible_direction_is_returned_bit_exact_after_double_solve() -> None:
    components = _components([1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [1.0, 0.5])
    raw, diagnostic = raw_component_gradient_diagnostic(components)
    projected, audit = project_gradient_to_feasible_descent(components)
    assert torch.equal(projected[0], raw[0])
    assert audit["selected_mask"] == 0
    assert audit["raw_solver_feasible"] is True
    assert audit["raw_feasible_returned_bit_exact"] is True
    assert audit["active_solve_safety_delta"] == 0.0
    assert audit["double_solve_replay"]["selected_lambdas_bit_exact"] is True
    assert diagnostic["raw_direction_already_feasible"] is True


def test_conflicting_raw_gradient_is_projected_to_nearest_safe_face() -> None:
    components = _conflicting_components()
    raw, diagnostic = raw_component_gradient_diagnostic(components)
    projected, audit = project_gradient_to_feasible_descent(components)
    assert diagnostic["conflicting_directions"] == ["broad"]
    assert audit["selected_mask"] == 1
    assert len(audit["candidate_audits"]) == 16
    assert audit["enumerated_mask_order"] == list(range(16))
    assert audit["active_solve_safety_delta"] == pytest.approx(1e-10)
    broad = components["broad"][0]
    unit = broad / broad.norm()
    raw_vector = raw[0]
    solve_beta = audit["active_solve_beta"]
    expected = raw_vector + (solve_beta - torch.dot(unit, raw_vector)) * unit
    assert torch.allclose(projected[0], expected, atol=1e-12, rtol=1e-12)
    assert audit["cpu_solution_safety_passed"] is True
    assert audit["post_device_cast_safety_passed"] is True
    assert audit["cpu_correction_ratio"] <= 0.25
    assert audit["cpu_projected_raw_cosine"] >= 0.95
    again, repeated = project_gradient_to_feasible_descent(components)
    assert torch.equal(projected[0], again[0])
    assert audit == repeated


def test_dependent_subsets_are_rejected_but_all_masks_are_audited() -> None:
    components = _components([1.0, 0.0], [2.0, 0.0], [0.0, 1.0], [0.0, 2.0])
    _projected, audit = project_gradient_to_feasible_descent(components)
    assert len(audit["candidate_audits"]) == 16
    assert any(
        row["rejection_reason"] == "linearly_dependent_active_subset"
        for row in audit["candidate_audits"]
    )


@pytest.mark.parametrize(
    "components",
    [
        _components([1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [0.0, 1.0]),
        _components([1.0, 0.0], [float("nan"), 0.0], [0.0, 1.0], [1.0, 1.0]),
        _components([1.0, 0.0], [0.0, 1.0], [float("nan"), 0.0], [1.0, 1.0]),
        _components([5e-13, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]),
        _components([1e-12, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]),
    ],
)
def test_infeasible_or_nonfinite_component_fails_before_mutation(components) -> None:
    with pytest.raises(V41GradientGuardFailure) as captured:
        project_gradient_to_feasible_descent(components)
    assert captured.value.audit["failure_reason"] in {
        "closed_halfspace_qp_has_no_feasible_candidate",
        "nonfinite_component_or_too_small_raw_total",
    }


@pytest.mark.parametrize(
    ("components", "active", "inactive", "mask_count"),
    [
        (
            _components([1.0, 0.0], [0.0, 0.0], [0.0, 1.0], [1.0, 1.0]),
            ["broad", "scene", "cross"],
            ["answer"],
            8,
        ),
        (
            _components([1.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]),
            ["broad"],
            ["answer", "scene", "cross"],
            2,
        ),
        (
            _components([1.0, 0.0], [0.0, 1.0], [0.0, 0.0], [0.0, 0.0]),
            ["broad", "answer"],
            ["scene", "cross"],
            4,
        ),
        (
            _components(
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, -1.0],
                [0.0, 0.0, 1.0],
            ),
            ["broad", "answer", "cross"],
            ["scene"],
            8,
        ),
    ],
)
def test_exact_zero_constraints_are_inactive_and_dynamically_enumerated(
    components, active, inactive, mask_count
) -> None:
    projected, audit = project_gradient_to_feasible_descent(components)
    assert projected
    assert audit["active_constraint_direction_names"] == active
    assert audit["inactive_constraint_direction_names"] == inactive
    assert audit["enumerated_subset_count"] == mask_count
    assert len(audit["candidate_audits"]) == mask_count
    for name in inactive:
        assert audit["cpu_directional_attestation"][name]["passed"] is True
        assert audit["cpu_directional_attestation"][name]["active"] is False
    parameter = torch.nn.Parameter(torch.zeros_like(projected[0]))
    clip = clip_direction_attestation(
        parameters=(parameter,),
        projected_total=projected,
        components=components,
        projection_attestation=audit,
        clip_norm=1.0,
    )
    assert clip["constraint_activity_linked"] is True
    assert clip["active_constraint_direction_names"] == active
    assert clip["inactive_constraint_direction_names"] == inactive


def test_float32_roundtrip_and_scalar_clip_are_hash_linked() -> None:
    components = _conflicting_components(dtype=torch.float32)
    projected, projection = project_gradient_to_feasible_descent(components)
    parameter = torch.nn.Parameter(torch.zeros_like(projected[0]))
    clip = clip_direction_attestation(
        parameters=(parameter,),
        projected_total=projected,
        components=components,
        projection_attestation=projection,
        clip_norm=1.0,
    )
    assert projection["post_device_cast_safety_passed"] is True
    assert clip["projection_input_linked"] is True
    assert clip["projected_input_state_sha256"] == projection[
        "post_device_cast_state_sha256"
    ]
    assert clip["projected_to_clipped_cosine"] >= 0.9999999
    assert clip["scalar_clip_direction_preserved"] is True


def test_full_projection_moves_mps_gradients_to_cpu_before_float64() -> None:
    if not torch.backends.mps.is_built() or not torch.backends.mps.is_available():
        pytest.skip("MPS is unavailable")
    cpu_components = _components(
        [1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [1.0, 0.5], dtype=torch.float32
    )
    components = {
        name: (values[0].to("mps"),) for name, values in cpu_components.items()
    }
    projected, projection = project_gradient_to_feasible_descent(components)
    assert projected[0].device.type == "mps"
    assert torch.isfinite(projected[0]).all()
    assert projection["selected_mask"] == 0
    assert projection["projection_applied"] is False
    assert projection["raw_feasible_returned_bit_exact"] is True
    assert projection["post_device_cast_safety_passed"] is True
    assert projection["applied_vs_source_dtype_raw_safety_passed"] is True
    assert projection["component_finite"] == {
        "broad": True,
        "answer": True,
        "side": True,
        "cross": True,
        "scene": True,
        "raw_total": True,
    }
    parameter = torch.nn.Parameter(torch.zeros_like(projected[0]))
    clip = clip_direction_attestation(
        parameters=(parameter,),
        projected_total=projected,
        components=components,
        projection_attestation=projection,
        clip_norm=1.0,
    )
    assert clip["scalar_clip_direction_preserved"] is True


def test_conflicting_mps_gradient_roundtrips_projected_direction_and_clip() -> None:
    if not torch.backends.mps.is_built() or not torch.backends.mps.is_available():
        pytest.skip("MPS is unavailable")
    small_components = _conflicting_components(dtype=torch.float32)
    cpu_components = {
        name: (values[0].reshape(1, 4).repeat(4096, 1).contiguous(),)
        for name, values in small_components.items()
    }
    components = {
        name: (values[0].to("mps"),) for name, values in cpu_components.items()
    }
    assert components["broad"][0].shape == (4096, 4)
    assert components["broad"][0].numel() == 16_384
    projected, projection = project_gradient_to_feasible_descent(components)
    assert projected[0].device.type == "mps"
    assert torch.isfinite(projected[0]).all()
    assert projection["projection_applied"] is True
    assert projection["selected_mask"] == 1
    assert projection["post_device_cast_safety_passed"] is True
    assert projection["applied_vs_source_dtype_raw_safety_passed"] is True
    parameter = torch.nn.Parameter(torch.zeros_like(projected[0]))
    clip = clip_direction_attestation(
        parameters=(parameter,),
        projected_total=projected,
        components=components,
        projection_attestation=projection,
        clip_norm=1.0,
    )
    assert clip["projection_input_linked"] is True
    assert clip["scalar_clip_direction_preserved"] is True


def test_guard_failure_is_atomic_and_json_finite(tmp_path: Path) -> None:
    path = persist_gradient_guard_failure(
        tmp_path,
        optimizer_step=3,
        audit={"guard_stage": "test", "value": 1.0},
    )
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["optimizer_step_executed"] is False
    assert payload["checkpoint_written"] is False
    assert list(tmp_path.iterdir()) == [path]


def test_projection_history_validator_authenticates_hash_chain_and_u3() -> None:
    contract = v41_contract(load_config(CONFIG))
    components = _components(
        [1.0, 0.0],
        [0.0, 0.0],
        [0.0, 1.0],
        [1.0, 1.0],
        dtype=torch.float32,
    )
    raw_total, raw = raw_component_gradient_diagnostic(components)
    projected, projection = project_gradient_to_feasible_descent(components)
    parameter = torch.nn.Parameter(torch.zeros_like(projected[0]))
    clip = clip_direction_attestation(
        parameters=(parameter,),
        projected_total=projected,
        components=components,
        projection_attestation=projection,
        clip_norm=1.0,
    )
    assert raw_total
    replay_hash = (
        "8c50aa3a5975f450c3c95fb00dbf077a33285bf22ac3208f5d745cd617bd8d48"
    )
    hashes = [contract.query_source_state_sha256, "1" * 64, replay_hash, "3" * 64]
    history: list[dict[str, object]] = [
        {
            "optimizer_update": 0,
            "query_bank_state_sha256": hashes[0],
            "validation_qa_loaded": False,
            "oracle_environment_files_loaded": False,
        }
    ]
    for step in range(1, 4):
        transient = None
        if step == 3:
            transient = {
                "observed_target_state_sha256": replay_hash,
                "expected_target_state_sha256": replay_hash,
                "exact_replay_of_v40_transient_steps_one_and_two": True,
                "checked_before_component_gradients_clip_and_optimizer_step": True,
            }
        history.append(
            {
                "optimizer_update": step,
                "true_optimizer_step": True,
                "raw_component_gradient_diagnostic": raw,
                "projected_gradient_attestation": projection,
                "clip_direction_attestation": clip,
                "transient_pre_update3_replay_attestation": transient,
                "target_hash_before": hashes[step - 1],
                "target_hash_after": hashes[step],
                "query_bank_state_sha256": hashes[step],
                "frozen_excluding_b_hash_before": contract.frozen_state_sha256,
                "frozen_excluding_b_hash_after": contract.frozen_state_sha256,
                "frozen_excluding_query_state_sha256": contract.frozen_state_sha256,
                "validation_qa_loaded": False,
                "oracle_environment_files_loaded": False,
            }
        )
    audit = validate_v41_projection_history(history, contract)
    assert audit["validated_optimizer_steps"] == 3
    history[2]["projected_gradient_attestation"] = {
        **projection,
        "enumerated_mask_order": list(reversed(range(8))),
    }
    with pytest.raises(ValueError, match="update 2"):
        validate_v41_projection_history(history, contract)


def test_run_loop_calls_projection_and_contains_exact_transient_gate() -> None:
    source = TRAINER.read_text(encoding="utf-8")
    tree = ast.parse(source)
    calls = {
        node.func.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    assert "raw_component_gradient_diagnostic" in calls
    assert "project_gradient_to_feasible_descent" in calls
    assert "clip_direction_attestation" in calls
    assert "component_gradient_guard" not in calls
    assert "8c50aa3a5975f450c3c95fb00dbf077a33285bf22ac3208f5d745cd617bd8d48" in source
    assert "d4c30be9e4f685697478b6e5a37f4f55d6e99962484b1cbae5c3c3214c24b35e" in source
