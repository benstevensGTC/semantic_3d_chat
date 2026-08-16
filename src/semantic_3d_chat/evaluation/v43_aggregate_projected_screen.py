"""Aggregate train-only projected-gradient response screen for V43.

The screen measures one globally aggregated gradient direction at exact V41
update zero, fixes every candidate tensor hash before evaluating candidates,
and writes no checkpoint.  Only training QA/maps are reachable.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch

from semantic_3d_chat.chat.file_audit import FileAccessAudit
from semantic_3d_chat.config import PROJECT_ROOT, load_config
from semantic_3d_chat.evaluation import v42_delta_line_screen as v42
from semantic_3d_chat.language.lora import tensor_state_sha256
from semantic_3d_chat.training.checkpointing import module_collection_state_sha256
from semantic_3d_chat.training.pair_curriculum import build_exact_question_pair_units
from semantic_3d_chat.training.train_block_cross_v35 import (
    broad_answer_nll,
    build_v35_schedule,
    current_scene_tokens,
    load_v35_train_qa_records,
    paired_cross_prefix_objective,
    residual_rms_diagnostics,
    v35_settings,
)
from semantic_3d_chat.training.train_environmental_sidecar_v33 import (
    assert_deferred_final_scenes_absent,
)
from semantic_3d_chat.training.train_joint_block_cross_v36 import (
    training_broad_nll,
    training_greedy_metrics,
    v36_broad_calibration_records,
)
from semantic_3d_chat.training.train_joint_pair_v30 import require_approved_v29_source
from semantic_3d_chat.training.train_joint_pair_v31 import v31_contract
from semantic_3d_chat.training.train_projected_gradient_v41 import (
    _prefix_replay_attestation,
    _target_parameters,
    build_v41_schedule,
    cache_v41_train_scenes,
    clip_direction_attestation,
    frozen_v41_state_sha256,
    load_v41_bundle,
    priority_side_deficit,
    project_gradient_to_feasible_descent,
    raw_component_gradient_diagnostic,
    require_exact_v41_sources,
    target_v41_state_sha256,
    training_pair_gate_diagnostics,
    v41_contract,
    v41_loader_config,
    v41_settings,
    v41_update8_gate,
)
from semantic_3d_chat.training.train_scene_ingress_kv_v37 import (
    validate_v37_training_cache_boundary,
)

DEFAULT_CONFIG = v42.DEFAULT_CONFIG
DEFAULT_TERMINAL = Path("reports/gemma4/metrics/v42_delta_line_terminal_gate.json")
DEFAULT_OUTPUT = Path(
    "reports/gemma4/metrics/v43_aggregate_projected_no_step_diagnostic.json"
)
_TERMINAL_SHA256 = "1f4f73c782813fd47d1ea8fd659df3545dffe8143bbcacc0d47c9d40baea59e8"
_STEPS = (-0.008, -0.004, 0.0, 0.002, 0.004, 0.008, 0.012, 0.016)
_TOLERANCE = 1e-6


def _resolve(path: str | Path) -> Path:
    value = Path(path)
    return value.resolve() if value.is_absolute() else (PROJECT_ROOT / value).resolve()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _mapping(value: object, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{field} must be a mapping")
    return value


def require_terminal() -> dict[str, Any]:
    path = _resolve(DEFAULT_TERMINAL)
    if path.is_symlink() or not path.is_file() or _sha256(path) != _TERMINAL_SHA256:
        raise ValueError("V43 requires the exact V42 terminal")
    report = json.loads(path.read_text(encoding="utf-8"))
    authorization = _mapping(
        _mapping(report, "V42 terminal").get("conditional_successor_authorization"),
        "V43 authorization",
    )
    gradient = _mapping(authorization.get("gradient_surface"), "V43 gradient surface")
    projection = _mapping(authorization.get("projection"), "V43 projection")
    scope = _mapping(authorization.get("diagnostic_scope"), "V43 scope")
    required = {
        "artifact": report.get("artifact") == "v42_delta_line_terminal_gate",
        "passed": report.get("passed") is True,
        "successor": report.get("only_exact_successor_authorized")
        == "v43_aggregate_projected_train_only_no_step_screen",
        "id": authorization.get("authorization_id")
        == "v43_aggregate_projected_train_only_no_step_screen",
        "output": authorization.get("authorized_output") == str(DEFAULT_OUTPUT),
        "target": authorization.get("source_target_state_sha256") == v42._TARGET_U0_SHA256,
        "full": authorization.get("source_full_state_sha256") == v42._FULL_U0_SHA256,
        "frozen": authorization.get("source_frozen_state_sha256") == v42._FROZEN_SHA256,
        "broad": gradient.get("broad_component") == "mean_48_unchanged_rows_times_1",
        "answer": gradient.get("answer_component")
        == "mean_8_priority_pair_answer_nll_times_0.5",
        "side": gradient.get("side_component")
        == "mean_8_priority_pair_side_hinge_times_8",
        "cross": gradient.get("cross_component")
        == "mean_all_25_pair_cross_hinge_times_56",
        "autograd": gradient.get("autograd_api") == "torch.autograd.grad",
        "qp": projection.get("implementation") == "v41_cpu_float64_active_set_qp",
        "steps": projection.get("fixed_scalar_steps") == list(_STEPS),
        "prehash": projection.get(
            "candidate_hashes_must_be_fixed_before_candidate_forward_evaluation"
        )
        is True,
        "temporary": scope.get("temporary_target_substitution_authorized") is True,
        "restore": scope.get("exact_u0_restoration_after_every_candidate") is True,
        "no_optimizer": scope.get("optimizer_construction_or_load_authorized") is False,
        "no_step": scope.get("optimizer_step_authorized") is False,
        "no_checkpoint": scope.get("checkpoint_write_authorized") is False,
        "no_validation": scope.get("validation_access_authorized") is False,
        "no_oracle": scope.get("oracle_access_authorized") is False,
        "no_final": scope.get("final_test_access_authorized") is False,
    }
    if not all(required.values()):
        raise ValueError(f"V43 authorization changed: {required}")
    return {"path": str(path), "sha256": _TERMINAL_SHA256, "authorization": dict(authorization)}


def candidate_from_direction(
    source: torch.Tensor, clipped_direction: torch.Tensor, scalar_step: float
) -> torch.Tensor:
    if scalar_step not in _STEPS:
        raise ValueError("V43 scalar step is outside the fixed grid")
    if source.dtype != torch.float32 or clipped_direction.dtype != torch.float32:
        raise TypeError("V43 source and direction must be float32")
    if source.shape != (4096, 4) or clipped_direction.shape != source.shape:
        raise ValueError("V43 source or direction shape changed")
    if scalar_step == 0.0:
        candidate = source.clone()
    else:
        candidate = (
            source.double() - scalar_step * clipped_direction.double()
        ).float()
    if not torch.isfinite(candidate).all():
        raise ValueError("V43 candidate is nonfinite")
    return candidate


def candidate_rank_key(row: Mapping[str, Any]) -> tuple[Any, ...]:
    """Rank eligible V43 candidates without depending on another step grid."""

    metrics = _mapping(row.get("pair_metrics"), "V43 candidate metrics")
    families = _mapping(
        metrics.get("complete_units_by_family"), "V43 family metrics"
    )
    family_count = sum(
        int(int(families.get(name, 0)) >= 1)
        for name in ("book_support", "mirror_lr", "picture_support")
    )
    step = float(row["scalar_step"])
    if step not in _STEPS:
        raise ValueError("V43 candidate step is outside the fixed grid")
    return (
        -int(metrics["complete_units"]),
        -family_count,
        -int(metrics["positive_sides"]),
        -int(metrics["cross_prefix_complete_units"]),
        float(row["priority_side_deficit"]),
        float(row["broad_nll"]),
        abs(step),
        list(_STEPS).index(step),
    )


def bundle_state_attestation(
    bundle: Any,
    *,
    expected_target_sha256: str,
    expected_full_sha256: str,
    expected_frozen_sha256: str,
) -> dict[str, Any]:
    """Measure exact mutable/frozen state and prove no gradients remain."""

    result = {
        "target_state_sha256": target_v41_state_sha256(bundle),
        "full_state_sha256": module_collection_state_sha256(
            bundle.checkpoint_modules
        ),
        "frozen_state_sha256": frozen_v41_state_sha256(bundle),
        "all_gradients_absent": not any(
            parameter.grad is not None
            for parameter in bundle.language.model.parameters()
        ),
    }
    result["passed"] = bool(
        result["target_state_sha256"] == expected_target_sha256
        and result["full_state_sha256"] == expected_full_sha256
        and result["frozen_state_sha256"] == expected_frozen_sha256
        and result["all_gradients_absent"] is True
    )
    return result


def _aggregate_cpu_gradients(
    gradients: Sequence[torch.Tensor], *, expected_count: int, name: str
) -> torch.Tensor:
    if len(gradients) != expected_count:
        raise RuntimeError(f"V43 {name} gradient count changed")
    result = torch.stack([value.detach().cpu().double() for value in gradients]).mean(0)
    if result.shape != (4096, 4) or not torch.isfinite(result).all():
        raise RuntimeError(f"V43 {name} aggregate gradient is invalid")
    return result.float()


def _objective_gradient(
    objective: torch.Tensor,
    target: torch.nn.Parameter,
    *,
    retain_graph: bool,
) -> torch.Tensor:
    if target.grad is not None:
        raise RuntimeError("V43 found an accumulated parameter gradient")
    (gradient,) = torch.autograd.grad(
        objective,
        (target,),
        retain_graph=retain_graph,
        create_graph=False,
        allow_unused=False,
    )
    if gradient.shape != target.shape or not torch.isfinite(gradient).all():
        raise RuntimeError("V43 objective gradient is invalid")
    if target.grad is not None:
        raise RuntimeError("V43 autograd.grad accumulated into parameter.grad")
    return gradient.detach()


def _screen_preflight(config_path: str | Path = DEFAULT_CONFIG) -> dict[str, Any]:
    terminal = require_terminal()
    base = v42.preflight(config_path)
    return {
        "schema_version": 1,
        "artifact": "v43_aggregate_projected_preflight",
        "passed": base["passed"] is True,
        "terminal": terminal,
        "fixed_scalar_steps": list(_STEPS),
        "train_question_count": base["train_question_count"],
        "changed_train_unit_count": base["changed_train_unit_count"],
        "gemma_loaded": False,
        "scene_maps_loaded": False,
        "optimizer_loaded_or_constructed": False,
        "validation_qa_loaded": False,
        "oracle_loaded": False,
        "final_test_scenes_touched": False,
        "forbidden_file_accesses": base["forbidden_file_accesses"],
    }


def run_screen(config_path: str | Path = DEFAULT_CONFIG) -> dict[str, Any]:
    terminal = require_terminal()
    config_path = _resolve(config_path)
    if _sha256(config_path) != v42._CONFIG_SHA256:
        raise ValueError("V43 config bytes changed")
    config = load_config(config_path)
    audit = FileAccessAudit(
        v42._forbidden_roots(config),
        forbidden_component_names={"oracle"},
        block_forbidden=True,
    )
    with audit:
        source_state, _stopped_state, _metadata8, stage8 = v42._endpoint_evidence()
        source_target = source_state[v42._TARGET]
        loader = v41_loader_config(config)
        assert_deferred_final_scenes_absent(loader)
        records, qa_audit = load_v35_train_qa_records(loader)
        units = build_exact_question_pair_units(records)
        schedule41, schedule41_audit = build_v41_schedule(
            records, units, seed=int(config["seed"])
        )
        priority_units = [item.pair_unit for item in schedule41[:8]]
        inherited_schedule, _ = build_v35_schedule(
            records,
            units,
            settings=v35_settings(loader),
            seed=int(config["seed"]),
        )
        broad_records = v36_broad_calibration_records(inherited_schedule)
        if len(units) != 25 or len(priority_units) != 8 or len(broad_records) != 48:
            raise RuntimeError("V43 aggregate inventory changed")

        contract = v41_contract(config)
        settings = v41_settings(config)
        hybrid, pinned_metadata, source_audit = require_exact_v41_sources(config)
        approved = require_approved_v29_source(loader)
        bundle, block_core, loaded_metadata, transition = load_v41_bundle(
            config, approved, contract.source_checkpoint, hybrid
        )
        if loaded_metadata != pinned_metadata:
            raise RuntimeError("V43 loaded source metadata changed")
        split = v31_contract(loader)
        manifest_ids = (*split.train_scene_ids, *split.validation_scene_ids)
        caches, cache_audit = cache_v41_train_scenes(
            config=loader,
            bundle=bundle,
            source_metadata=pinned_metadata,
            scene_ids=split.train_scene_ids,
            manifest_scene_ids=manifest_ids,
        )
        cache_audit.update(
            {
                "scene_scope": "training_only",
                "authenticated_manifest_scene_count": len(manifest_ids),
                "authenticated_manifest_train_subset_count": len(split.train_scene_ids),
                "validation_scene_ids_loaded": [],
                "validation_environment_maps_loaded": False,
                "deferred_final_scene_ids_loaded": [],
            }
        )
        cache_boundary = validate_v37_training_cache_boundary(
            cache_audit=cache_audit,
            caches=caches,
            config=loader,
            train_scene_ids=split.train_scene_ids,
            validation_scene_ids=split.validation_scene_ids,
        )
        prefix_evidence = _prefix_replay_attestation(
            caches=caches,
            block_cross_residual=block_core,
            bundle=bundle,
            expected_scene_ids=split.train_scene_ids,
        )
        residual = residual_rms_diagnostics(
            caches=caches,
            block_cross_residual=block_core,
            device=bundle.language.device,
        )
        if residual != stage8["source_residual_diagnostics"]:
            raise RuntimeError("V43 frozen scene residual changed")

        model_dtype = next(bundle.language.model.parameters()).dtype
        with torch.inference_mode():
            scene_tokens = {
                scene_id: current_scene_tokens(
                    caches[scene_id], block_core, device=bundle.language.device
                ).detach().to(model_dtype)
                for scene_id in split.train_scene_ids
            }
        target = _target_parameters(bundle)[0]
        if target_v41_state_sha256(bundle) != v42._TARGET_U0_SHA256:
            raise RuntimeError("V43 did not start from exact update zero")

        broad_gradients: list[torch.Tensor] = []
        for record in broad_records:
            loss = settings.broad_nll_weight * broad_answer_nll(
                scene_tokens=scene_tokens[record.scene_id], record=record, bundle=bundle
            )
            broad_gradients.append(
                _objective_gradient(loss, target, retain_graph=False)
            )
            del loss
            if bundle.language.device.type == "mps":
                torch.mps.empty_cache()

        priority_keys = {unit.question_key for unit in priority_units}
        answer_gradients: list[torch.Tensor] = []
        side_gradients: list[torch.Tensor] = []
        cross_gradients: list[torch.Tensor] = []
        gradient_loss_rows: list[dict[str, Any]] = []
        for unit in sorted(units, key=lambda value: (value.pair_id, value.question_key)):
            tokens = {scene_id: scene_tokens[scene_id] for scene_id in unit.scene_ids}
            correct, side, cross, diagnostics = paired_cross_prefix_objective(
                unit=unit,
                scene_tokens=tokens,
                bundle=bundle,
                side_margin=settings.side_hinge_margin,
                cross_prefix_margin=settings.cross_prefix_flip_margin,
            )
            is_priority = unit.question_key in priority_keys
            if is_priority:
                answer_gradients.append(
                    _objective_gradient(
                        settings.pair_correct_nll_weight * correct,
                        target,
                        retain_graph=True,
                    )
                )
                side_gradients.append(
                    _objective_gradient(
                        settings.side_hinge_weight * side,
                        target,
                        retain_graph=True,
                    )
                )
            cross_gradients.append(
                _objective_gradient(
                    settings.cross_prefix_flip_weight * cross,
                    target,
                    retain_graph=False,
                )
            )
            gradient_loss_rows.append(
                {
                    "question_key": unit.question_key,
                    "priority": is_priority,
                    "correct_answer_nll": float(correct.detach().cpu()),
                    "side_hinge": float(side.detach().cpu()),
                    "cross_hinge": float(cross.detach().cpu()),
                    "side_margin_mean": float(
                        diagnostics["side_margins"].detach().float().mean().cpu()
                    ),
                    "cross_margin_mean": float(
                        diagnostics["cross_prefix_margins"]
                        .detach()
                        .float()
                        .mean()
                        .cpu()
                    ),
                }
            )
            del correct, side, cross, diagnostics
            if bundle.language.device.type == "mps":
                torch.mps.empty_cache()

        component_cpu = {
            "broad": _aggregate_cpu_gradients(
                broad_gradients, expected_count=48, name="broad"
            ),
            "answer": _aggregate_cpu_gradients(
                answer_gradients, expected_count=8, name="answer"
            ),
            "side": _aggregate_cpu_gradients(
                side_gradients, expected_count=8, name="side"
            ),
            "cross": _aggregate_cpu_gradients(
                cross_gradients, expected_count=25, name="cross"
            ),
        }
        components = {
            name: (value.to(device=target.device, dtype=target.dtype),)
            for name, value in component_cpu.items()
        }
        _raw, raw_audit = raw_component_gradient_diagnostic(components)
        projected, projection_audit = project_gradient_to_feasible_descent(components)
        clip_audit = clip_direction_attestation(
            parameters=(target,),
            projected_total=projected,
            components=components,
            projection_attestation=projection_audit,
            clip_norm=1.0,
        )
        if target.grad is None:
            raise RuntimeError("V43 clip did not materialize its diagnostic direction")
        clipped_direction = target.grad.detach().cpu().float().clone()
        target.grad = None
        direction_hash = tensor_state_sha256({"clipped_projected_gradient": clipped_direction})
        component_hashes = {
            name: tensor_state_sha256({"gradient": value})
            for name, value in component_cpu.items()
        }
        gradient_source_state = bundle_state_attestation(
            bundle,
            expected_target_sha256=v42._TARGET_U0_SHA256,
            expected_full_sha256=v42._FULL_U0_SHA256,
            expected_frozen_sha256=v42._FROZEN_SHA256,
        )
        if gradient_source_state["passed"] is not True:
            raise RuntimeError("V43 gradient measurement changed its source state")

        candidates: dict[float, torch.Tensor] = {}
        candidate_inventory: list[dict[str, Any]] = []
        for step in _STEPS:
            candidate = candidate_from_direction(source_target, clipped_direction, step)
            full = dict(source_state)
            full[v42._TARGET] = candidate
            row = {
                "scalar_step": step,
                "target_state_sha256": tensor_state_sha256({"lora_b": candidate}),
                "full_state_sha256": tensor_state_sha256(full),
            }
            candidates[step] = candidate
            candidate_inventory.append(row)
        inventory_hash = hashlib.sha256(
            json.dumps(candidate_inventory, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()

        source_pair = _mapping(stage8["source_pair_metrics"], "V43 source pair")
        source_nll = stage8["source_per_unit_nll_diagnostics"]
        source_broad = float(stage8["source_broad_train_nll"])
        source_deficit = float(priority_side_deficit(source_pair)["combined"])
        rows: list[dict[str, Any]] = []
        restorations: list[dict[str, Any]] = []

        def restore_source(step: float) -> None:
            with torch.no_grad():
                target.copy_(source_target.to(device=target.device, dtype=target.dtype))
            target.grad = None
            passed = bool(
                target_v41_state_sha256(bundle) == v42._TARGET_U0_SHA256
                and module_collection_state_sha256(bundle.checkpoint_modules)
                == v42._FULL_U0_SHA256
                and frozen_v41_state_sha256(bundle) == v42._FROZEN_SHA256
            )
            restorations.append({"scalar_step": step, "passed": passed})
            if not passed:
                raise RuntimeError("V43 failed to restore update zero")

        target.requires_grad_(False)
        for index, step in enumerate(_STEPS):
            candidate_spec = candidate_inventory[index]
            try:
                with torch.no_grad():
                    target.copy_(candidates[step].to(device=target.device, dtype=target.dtype))
                if (
                    target_v41_state_sha256(bundle)
                    != candidate_spec["target_state_sha256"]
                    or module_collection_state_sha256(bundle.checkpoint_modules)
                    != candidate_spec["full_state_sha256"]
                    or frozen_v41_state_sha256(bundle) != v42._FROZEN_SHA256
                ):
                    raise RuntimeError("V43 candidate state hash changed")
                pair_metrics, nll_rows = training_pair_gate_diagnostics(
                    units=units,
                    caches=caches,
                    block_cross_residual=block_core,
                    bundle=bundle,
                    settings=settings,
                )
                broad_nll = training_broad_nll(
                    records=broad_records,
                    caches=caches,
                    block_cross_residual=block_core,
                    bundle=bundle,
                )
                gate = v41_update8_gate(
                    pair_metrics=pair_metrics,
                    broad_nll=broad_nll,
                    source_broad_nll=source_broad,
                    source_priority_deficit=source_deficit,
                    query_state_sha256=candidate_spec["target_state_sha256"],
                    frozen_state_sha256=v42._FROZEN_SHA256,
                    scene_state_exact=True,
                    per_unit_nll_diagnostics=nll_rows,
                    contract=contract,
                )
                deficit = float(priority_side_deficit(pair_metrics)["combined"])
                row = {
                    "alpha": step,
                    "scalar_step": step,
                    **candidate_spec,
                    "pair_metrics": dict(pair_metrics),
                    "per_unit_nll_diagnostics": [dict(value) for value in nll_rows],
                    "broad_nll": broad_nll,
                    "priority_side_deficit": deficit,
                    "unchanged_v41_update8_gate": gate,
                    "teacher_eligible": gate["passed"] is True,
                }
                rows.append(row)
                print(
                    json.dumps(
                        {
                            "phase": "v43_aggregate_candidate",
                            "scalar_step": step,
                            "complete_units": pair_metrics["complete_units"],
                            "positive_sides": pair_metrics["positive_sides"],
                            "cross_prefix_complete_units": pair_metrics[
                                "cross_prefix_complete_units"
                            ],
                            "priority_side_deficit": deficit,
                            "broad_nll": broad_nll,
                            "teacher_eligible": gate["passed"],
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
            finally:
                restore_source(step)

        zero = rows[list(_STEPS).index(0.0)]
        endpoint = {
            "pair_metrics": v42._numeric_close(zero["pair_metrics"], source_pair),
            "per_unit_nll": v42._numeric_close(
                zero["per_unit_nll_diagnostics"], source_nll
            ),
            "broad_nll": math.isclose(
                zero["broad_nll"], source_broad, rel_tol=0.0, abs_tol=_TOLERANCE
            ),
        }
        endpoint["passed"] = all(endpoint.values())
        if endpoint["passed"] is not True:
            raise RuntimeError("V43 update-zero endpoint replay failed")

        eligible = [row for row in rows if row["teacher_eligible"] is True]
        selected = min(eligible, key=candidate_rank_key) if eligible else None
        selected_replay: dict[str, Any] | None = None
        greedy: dict[str, Any] | None = None
        strong_greedy = False
        if selected is not None:
            step = float(selected["scalar_step"])
            try:
                with torch.no_grad():
                    target.copy_(candidates[step].to(device=target.device, dtype=target.dtype))
                replay_identity = bundle_state_attestation(
                    bundle,
                    expected_target_sha256=str(selected["target_state_sha256"]),
                    expected_full_sha256=str(selected["full_state_sha256"]),
                    expected_frozen_sha256=v42._FROZEN_SHA256,
                )
                if replay_identity["passed"] is not True:
                    raise RuntimeError("V43 selected replay identity changed")
                replay_pair, replay_nll = training_pair_gate_diagnostics(
                    units=units,
                    caches=caches,
                    block_cross_residual=block_core,
                    bundle=bundle,
                    settings=settings,
                )
                replay_broad = training_broad_nll(
                    records=broad_records,
                    caches=caches,
                    block_cross_residual=block_core,
                    bundle=bundle,
                )
                selected_replay = {
                    "scalar_step": step,
                    "identity_before_forward": replay_identity,
                    "pair_metrics_identical": v42._numeric_close(
                        replay_pair, selected["pair_metrics"]
                    ),
                    "per_unit_nll_identical": v42._numeric_close(
                        replay_nll, selected["per_unit_nll_diagnostics"]
                    ),
                    "broad_nll_identical": math.isclose(
                        replay_broad,
                        float(selected["broad_nll"]),
                        rel_tol=0.0,
                        abs_tol=_TOLERANCE,
                    ),
                }
                selected_replay["passed"] = all(
                    selected_replay[key] is True
                    for key in (
                        "pair_metrics_identical",
                        "per_unit_nll_identical",
                        "broad_nll_identical",
                    )
                )
                if selected_replay["passed"] is not True:
                    raise RuntimeError("V43 selected candidate replay was not exact")
                greedy = training_greedy_metrics(
                    units=units,
                    broad_records=broad_records,
                    caches=caches,
                    block_cross_residual=block_core,
                    bundle=bundle,
                    config=loader,
                )
                family = _mapping(greedy["complete_units_by_family"], "V43 greedy")
                strong_greedy = bool(
                    int(greedy["complete_units"]) >= 6
                    and all(
                        int(family.get(name, 0)) >= 1
                        for name in ("book_support", "mirror_lr", "picture_support")
                    )
                    and int(greedy["broad_exact_correct"]) >= 23
                    and int(greedy["broad_row_count"]) == 48
                )
            finally:
                restore_source(step)

        target.requires_grad_(False)
        final_state = {
            "target_state_sha256": target_v41_state_sha256(bundle),
            "full_state_sha256": module_collection_state_sha256(bundle.checkpoint_modules),
            "frozen_state_sha256": frozen_v41_state_sha256(bundle),
            "all_gradients_absent": not any(
                parameter.grad is not None for parameter in bundle.language.model.parameters()
            ),
            "all_requires_grad_false": not any(
                parameter.requires_grad for parameter in bundle.language.model.parameters()
            ),
        }
        final_state["restored_exact"] = bool(
            final_state["target_state_sha256"] == v42._TARGET_U0_SHA256
            and final_state["full_state_sha256"] == v42._FULL_U0_SHA256
            and final_state["frozen_state_sha256"] == v42._FROZEN_SHA256
            and final_state["all_gradients_absent"] is True
            and final_state["all_requires_grad_false"] is True
            and all(row["passed"] is True for row in restorations)
        )
        if final_state["restored_exact"] is not True:
            raise RuntimeError("V43 final source restoration failed")

    audit.assert_clean()
    loaded_maps = sorted(
        path for path in audit.unique_paths if "/maps/" in path and path.endswith(".npz")
    )
    if loaded_maps != sorted(cache_audit["loaded_environment_files"]):
        raise RuntimeError("V43 map reads differ from its train cache")
    return {
        "schema_version": 1,
        "artifact": "v43_aggregate_projected_no_step_diagnostic",
        "screen_integrity_passed": True,
        "terminal": terminal,
        "gradient_inventory": {
            "broad_row_count": 48,
            "priority_unit_count": 8,
            "cross_unit_count": 25,
            "component_state_sha256": component_hashes,
            "raw_diagnostic": raw_audit,
            "projection_attestation": projection_audit,
            "clip_attestation": clip_audit,
            "clipped_direction_state_sha256": direction_hash,
            "gradient_loss_rows": gradient_loss_rows,
            "parameter_grad_accumulation": False,
            "optimizer_constructed_or_loaded": False,
            "source_state_after_gradient_measurement": gradient_source_state,
            "source_state_unchanged_during_gradient_measurement":
                gradient_source_state["passed"] is True,
        },
        "candidate_inventory": {
            "formula": "float32(B0 - scalar_step * clipped_projected_gradient)",
            "fixed_scalar_steps": list(_STEPS),
            "candidate_hashes_fixed_before_forward_evaluation": True,
            "candidate_inventory_sha256": inventory_hash,
            "candidates": candidate_inventory,
        },
        "candidate_results": rows,
        "update_zero_endpoint_replay": endpoint,
        "teacher_eligible_candidate_found": selected is not None,
        "selected_scalar_step": None if selected is None else selected["scalar_step"],
        "selected_target_sha256": None
        if selected is None
        else selected["target_state_sha256"],
        "selected_candidate": selected,
        "selected_candidate_replay": selected_replay,
        "optional_selected_candidate_greedy_audit": greedy,
        "optional_greedy_audit_passed": strong_greedy,
        "restoration_audit": restorations,
        "final_state": final_state,
        "schedule": schedule41_audit,
        "source_audit": source_audit,
        "loader_transition": transition,
        "cache_boundary": cache_boundary,
        "scene_prefix_evidence": prefix_evidence,
        "qa_audit": qa_audit,
        "model_loaded_once": True,
        "optimizer_constructed_or_loaded": False,
        "candidate_checkpoint_written": False,
        "validation_qa_loaded": False,
        "oracle_loaded": False,
        "final_test_scenes_touched": False,
        "selector_execution_authorized": False,
        "training_authorized": False,
        "runtime_promotion_authorized": False,
        "loaded_files": audit.unique_paths,
        "forbidden_file_accesses": audit.forbidden_accesses(),
    }


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    os.close(descriptor)
    temporary_path = Path(temporary)
    try:
        temporary_path.write_text(
            json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def write_report(
    output: str | Path = DEFAULT_OUTPUT, *, config_path: str | Path = DEFAULT_CONFIG
) -> dict[str, Any]:
    path = _resolve(output)
    authorized = _resolve(DEFAULT_OUTPUT)
    if path != authorized:
        raise ValueError(f"V43 output is pinned to {authorized}")
    if path.is_symlink() or path.exists():
        raise FileExistsError(f"V43 is one-shot and will not overwrite {path}")
    report = run_screen(config_path)
    _atomic_json(path, report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    result = _screen_preflight(args.config) if args.preflight_only else write_report(
        args.output, config_path=args.config
    )
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "bundle_state_attestation",
    "candidate_from_direction",
    "candidate_rank_key",
    "require_terminal",
    "run_screen",
    "write_report",
]
