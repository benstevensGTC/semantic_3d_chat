"""Fixed train-only response screen along the stopped V41 update direction.

This diagnostic constructs no optimizer and saves no candidate checkpoint.  It
temporarily substitutes nine pre-hashed values for the single layer-14 LoRA-B
tensor, evaluates only authenticated training scenes, and restores exact V41
update zero after every candidate.  Validation, final-scene, and oracle inputs
are blocked by a process-wide file-access audit.
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
from safetensors.torch import load_file

from semantic_3d_chat.chat.file_audit import FileAccessAudit
from semantic_3d_chat.config import PROJECT_ROOT, artifact_root, load_config
from semantic_3d_chat.language.lora import tensor_state_sha256
from semantic_3d_chat.training.checkpointing import module_collection_state_sha256
from semantic_3d_chat.training.pair_curriculum import build_exact_question_pair_units
from semantic_3d_chat.training.train_block_cross_v35 import (
    build_v35_schedule,
    load_v35_train_qa_records,
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
    cache_v41_train_scenes,
    frozen_v41_state_sha256,
    load_v41_bundle,
    priority_side_deficit,
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

DEFAULT_CONFIG = Path(
    "configs/experiments/gemma4_diverse28_projected_gradient_v41_retry1.yaml"
)
DEFAULT_TERMINAL = Path(
    "reports/gemma4/metrics/v41_retry1_update8_terminal_gate.json"
)
DEFAULT_OUTPUT = Path(
    "reports/gemma4/metrics/v42_v41_retry1_update8_no_step_diagnostic.json"
)
CHECKPOINT_ROOT = Path(
    "data_gemma4/checkpoints/"
    "gemma4_v41_retry1_diverse28_projected_gradient_l14_query"
)

_CONFIG_SHA256 = "4e3adb9b375d0e3ebd4c0936fba62e34a05ad88e015d073d43e55428de0b90c7"
_TERMINAL_SHA256 = "16cd37d91ceb911904737d8b306a308c9a12984fd13d78ee10842f36c8b771fd"
_U0_ADAPTER_SHA256 = "b0bdceb7699e9d97467915c69186f433d3b0fac2b09144d38c3078afe1f70cb0"
_U8_ADAPTER_SHA256 = "db622c4069a8bcc546172c61d027af9c0d0570ceb9952d4f011b9ad82fd60d7a"
_U8_METADATA_SHA256 = "034c9e21bbd270832e7cf3f146f71e762ea2ae709a0131071fb187100c04ee28"
_TARGET = "lora_banks.extension_v28_stage_b_query.adapters.1.lora_b"
_TARGET_U0_SHA256 = "d0834cc588ee2a9edf08aabedfd01e0a6d2b01c6b6ae7e3a3d764eaddf58cc3e"
_TARGET_U8_SHA256 = "2d6a8cdd1c67cf6405b17ea4d8b9eb6d48121ffb6630f7910fe43447872702fd"
_FULL_U0_SHA256 = "7b951c6d7ae4f7b50603159f0bc4dfb4d50b5b40f9325134d78d1de1dae87fc0"
_FULL_U8_SHA256 = "5ebc17795c35cb15c0e47f1c3d2d15a74e65e277519829d26308eb3b9fd34ce4"
_FROZEN_SHA256 = "cec01bc088bb87c6bb44e0659eb03aa766f951ddeee706ca9a70edaa080dea5e"
_ALPHAS = (-1.0, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 1.0, 1.25)
_CANDIDATE_HASHES: Mapping[str, Mapping[str, str]] = {
    "-1": {"target": "7ec6142103fa90964e2490637cf39ef17341a707cf246f07014ea0ea4f01ee60", "full": "816aad8ba5ddba78222c0e076fecdb4fdd9827e7a4dc432faafd2296df27b266"},
    "-0.5": {"target": "d875b680fbfe8a7b1575868831023cb5745c1d90bd20877fa26715e62bc1b47e", "full": "2c65a26b3a601fcd01db058d49d32f1794392ec3c526529ad822b03eb7aab41c"},
    "-0.25": {"target": "dce0054992a1db06adaa91ce1e643368c9f217b3e55102364cf9da022454102d", "full": "50f2cb5e9ad7e9ad801caf279b8456985a569414823d2f35821eca69cc651641"},
    "0": {"target": _TARGET_U0_SHA256, "full": _FULL_U0_SHA256},
    "0.25": {"target": "e349de6baa085166af7718492c28e0bbfe4da349b3c0fc456f230d162b269e8e", "full": "4e04766b7bf3511d532944356463095ce19d92898e1bb5d94012b9c2669d6ed6"},
    "0.5": {"target": "8f106bea1daa766b8b454145365393827a618ecdc3de4d33e1d832a633234541", "full": "2e14c18278a097fe0fb8c470235cceaa2100d7763f74abbc99b72af739c00591"},
    "0.75": {"target": "e82cbeb7cf406c9c6a4f2bed92a7dec7b078c63336d296c47f48452806a938db", "full": "ba2b64d25476635accf9fc9813ff2e5624384b76d4024bbf97e278fc1bbd9477"},
    "1": {"target": _TARGET_U8_SHA256, "full": _FULL_U8_SHA256},
    "1.25": {"target": "0b2a1d8a6001d9f08aa7c77c5708a9dc885c2f83cb3c48df4356e3424fee8f84", "full": "601960feae0876c554fee633f9ec49cf0c326d6b9b9a60a0d5f09c4f7eca132c"},
}
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


def alpha_key(alpha: float) -> str:
    return f"{alpha:g}"


def candidate_tensor(
    source: torch.Tensor, stopped: torch.Tensor, alpha: float
) -> torch.Tensor:
    """Construct a preregistered candidate with one final float32 cast."""

    if source.dtype != torch.float32 or stopped.dtype != torch.float32:
        raise TypeError("V42 endpoints must be float32")
    if source.shape != (4096, 4) or stopped.shape != source.shape:
        raise ValueError("V42 endpoint shape changed")
    if alpha not in _ALPHAS:
        raise ValueError("V42 alpha is outside the sealed grid")
    if alpha == 0.0:
        result = source.clone()
    elif alpha == 1.0:
        result = stopped.clone()
    else:
        result = (source.double() + alpha * (stopped.double() - source.double())).float()
    if not torch.isfinite(result).all():
        raise ValueError("V42 candidate is nonfinite")
    expected = _CANDIDATE_HASHES[alpha_key(alpha)]["target"]
    if tensor_state_sha256({"lora_b": result}) != expected:
        raise ValueError("V42 candidate target hash differs from preregistration")
    return result


def build_candidate_inventory(
    source_state: Mapping[str, torch.Tensor], stopped_state: Mapping[str, torch.Tensor]
) -> tuple[dict[float, torch.Tensor], dict[str, Any]]:
    if set(source_state) != set(stopped_state) or _TARGET not in source_state:
        raise ValueError("V42 endpoint tensor inventories differ")
    changed = [
        name for name in source_state if not torch.equal(source_state[name], stopped_state[name])
    ]
    if changed != [_TARGET]:
        raise ValueError("V42 endpoints differ outside the target LoRA-B tensor")
    candidates: dict[float, torch.Tensor] = {}
    rows: list[dict[str, Any]] = []
    for alpha in _ALPHAS:
        candidate = candidate_tensor(source_state[_TARGET], stopped_state[_TARGET], alpha)
        full = dict(source_state)
        full[_TARGET] = candidate
        expected = _CANDIDATE_HASHES[alpha_key(alpha)]
        observed_full = tensor_state_sha256(full)
        if observed_full != expected["full"]:
            raise ValueError("V42 candidate full-envelope hash changed")
        candidates[alpha] = candidate
        rows.append(
            {
                "alpha": alpha,
                "target_state_sha256": expected["target"],
                "full_state_sha256": expected["full"],
            }
        )
    return candidates, {
        "formula": "float32(float64(B0) + alpha * (float64(B8) - float64(B0)))",
        "fixed_alpha_grid": list(_ALPHAS),
        "candidate_rows": rows,
        "adaptive_refinement": False,
        "endpoint_zero_exact_clone": torch.equal(candidates[0.0], source_state[_TARGET]),
        "endpoint_one_exact_clone": torch.equal(candidates[1.0], stopped_state[_TARGET]),
    }


def require_terminal() -> dict[str, Any]:
    path = _resolve(DEFAULT_TERMINAL)
    if path.is_symlink() or not path.is_file() or _sha256(path) != _TERMINAL_SHA256:
        raise ValueError("V42 requires the exact V41 retry1 terminal revision 2")
    report = json.loads(path.read_text(encoding="utf-8"))
    authorization = _mapping(
        _mapping(report, "V41 retry terminal").get("conditional_successor_authorization"),
        "V42 authorization",
    )
    scope = _mapping(authorization.get("diagnostic_scope"), "V42 diagnostic scope")
    required = {
        "artifact": report.get("artifact") == "v41_retry1_update8_terminal_gate",
        "revision": report.get("seal_revision") == 2,
        "passed": report.get("passed") is True,
        "successor": report.get("only_exact_successor_authorized")
        == "v42_train_only_no_step_diagnostic_screen",
        "authorization": authorization.get("authorization_id")
        == "v42_v41_retry1_update8_no_step_diagnostic_screen",
        "output": authorization.get("authorized_output") == str(DEFAULT_OUTPUT),
        "temporary": scope.get("temporary_target_b_substitution_authorized") is True,
        "restore": scope.get(
            "temporary_substitution_must_restore_exact_u0_after_each_candidate"
        )
        is True,
        "grid": scope.get("fixed_alpha_grid") == list(_ALPHAS),
        "hashes": scope.get("fixed_candidate_state_sha256") == dict(_CANDIDATE_HASHES),
        "no_adaptive": scope.get("adaptive_candidate_refinement_authorized") is False,
        "no_gradient": scope.get("gradient_measurement_authorized") is False,
        "no_optimizer": scope.get("optimizer_construction_authorized") is False,
        "no_step": scope.get("optimizer_step_authorized") is False,
        "no_checkpoint": scope.get("checkpoint_write_authorized") is False,
        "no_validation": authorization.get("validation_access_authorized") is False,
        "no_oracle": authorization.get("oracle_access_authorized") is False,
        "no_final": authorization.get("final_test_access_authorized") is False,
        "no_selector": authorization.get("selector_execution_authorized") is False,
    }
    if not all(required.values()):
        raise ValueError(f"V42 exact terminal authorization changed: {required}")
    return {
        "path": str(path),
        "sha256": _TERMINAL_SHA256,
        "authorization": dict(authorization),
    }


def _endpoint_evidence() -> tuple[
    dict[str, torch.Tensor], dict[str, torch.Tensor], dict[str, Any], dict[str, Any]
]:
    root = _resolve(CHECKPOINT_ROOT)
    u0_path = root / "update_000/adapter.safetensors"
    u8_path = root / "update_008/adapter.safetensors"
    metadata_path = root / "update_008/metadata.json"
    expected = {
        u0_path: _U0_ADAPTER_SHA256,
        u8_path: _U8_ADAPTER_SHA256,
        metadata_path: _U8_METADATA_SHA256,
    }
    for path, digest in expected.items():
        if path.is_symlink() or not path.is_file() or _sha256(path) != digest:
            raise ValueError(f"V42 endpoint file changed: {path}")
    source = load_file(u0_path, device="cpu")
    stopped = load_file(u8_path, device="cpu")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    stage = dict(_mapping(metadata.get("v41_projected_gradient"), "V42 u8 stage"))
    history = metadata.get("history")
    if not isinstance(history, list) or len(history) != 9:
        raise ValueError("V42 stopped history changed")
    if tensor_state_sha256(source) != _FULL_U0_SHA256 or tensor_state_sha256(stopped) != _FULL_U8_SHA256:
        raise ValueError("V42 endpoint tensor state changed")
    return source, stopped, metadata, stage


def _forbidden_roots(config: Mapping[str, Any]) -> list[Path]:
    loader = v41_loader_config(config)
    split = v31_contract(loader)
    qa_root = artifact_root(loader, "qa").resolve()
    maps_root = artifact_root(loader, "maps").resolve()
    roots = [
        artifact_root(loader, "oracle").resolve(),
        artifact_root(loader, "rendered").resolve(),
        artifact_root(loader, "features").resolve(),
        qa_root / "validation.jsonl",
        qa_root / "test.jsonl",
        _resolve(CHECKPOINT_ROOT) / "update_008/optimizer.pt",
        _resolve(CHECKPOINT_ROOT) / "update_008/optimizer_audit.json",
    ]
    allowed = set(split.train_scene_ids)
    if maps_root.is_dir():
        roots.extend(path for path in maps_root.iterdir() if path.name not in allowed)
    roots.extend(PROJECT_ROOT.rglob("optimizer.pt"))
    return [path.resolve() for path in roots]


def _numeric_close(first: object, second: object, *, tolerance: float = _TOLERANCE) -> bool:
    if isinstance(first, Mapping) and isinstance(second, Mapping):
        return set(first) == set(second) and all(
            _numeric_close(first[key], second[key], tolerance=tolerance) for key in first
        )
    if isinstance(first, Sequence) and not isinstance(first, (str, bytes)):
        return isinstance(second, Sequence) and not isinstance(second, (str, bytes)) and len(first) == len(second) and all(
            _numeric_close(a, b, tolerance=tolerance)
            for a, b in zip(first, second, strict=True)
        )
    if isinstance(first, bool) or isinstance(second, bool):
        return first is second
    if isinstance(first, (int, float)) and isinstance(second, (int, float)):
        return math.isfinite(float(first)) and math.isfinite(float(second)) and math.isclose(
            float(first), float(second), rel_tol=0.0, abs_tol=tolerance
        )
    return first == second


def candidate_rank_key(row: Mapping[str, Any]) -> tuple[Any, ...]:
    metrics = _mapping(row.get("pair_metrics"), "V42 candidate metrics")
    families = _mapping(metrics.get("complete_units_by_family"), "V42 family metrics")
    family_count = sum(
        int(int(families.get(name, 0)) >= 1)
        for name in ("book_support", "mirror_lr", "picture_support")
    )
    return (
        -int(metrics["complete_units"]),
        -family_count,
        -int(metrics["positive_sides"]),
        -int(metrics["cross_prefix_complete_units"]),
        float(row["priority_side_deficit"]),
        float(row["broad_nll"]),
        abs(float(row["alpha"])),
        list(_ALPHAS).index(float(row["alpha"])),
    )


def _candidate_summary(
    *,
    alpha: float,
    expected_hashes: Mapping[str, str],
    pair_metrics: Mapping[str, Any],
    nll_rows: Sequence[Mapping[str, Any]],
    broad_nll: float,
    gate: Mapping[str, Any],
) -> dict[str, Any]:
    deficit = float(priority_side_deficit(pair_metrics)["combined"])
    families = _mapping(pair_metrics.get("complete_units_by_family"), "V42 families")
    return {
        "alpha": alpha,
        "target_state_sha256": expected_hashes["target"],
        "full_state_sha256": expected_hashes["full"],
        "pair_metrics": dict(pair_metrics),
        "per_unit_nll_diagnostics": [dict(row) for row in nll_rows],
        "broad_nll": broad_nll,
        "priority_side_deficit": deficit,
        "priority_family_complete_count": sum(
            int(int(families.get(name, 0)) >= 1)
            for name in ("book_support", "mirror_lr", "picture_support")
        ),
        "unchanged_v41_update8_gate": dict(gate),
        "teacher_eligible": gate.get("passed") is True,
    }


def preflight(config_path: str | Path = DEFAULT_CONFIG) -> dict[str, Any]:
    terminal = require_terminal()
    config_path = _resolve(config_path)
    if _sha256(config_path) != _CONFIG_SHA256:
        raise ValueError("V42 config bytes changed")
    config = load_config(config_path)
    audit = FileAccessAudit(
        _forbidden_roots(config), forbidden_component_names={"oracle"}, block_forbidden=True
    )
    with audit:
        source, stopped, _metadata, _stage = _endpoint_evidence()
        candidates, inventory = build_candidate_inventory(source, stopped)
        loader = v41_loader_config(config)
        assert_deferred_final_scenes_absent(loader)
        records, qa_audit = load_v35_train_qa_records(loader)
        units = build_exact_question_pair_units(records)
    audit.assert_clean()
    if len(units) != 25 or len(candidates) != 9:
        raise RuntimeError("V42 preflight inventory changed")
    return {
        "schema_version": 1,
        "artifact": "v42_delta_line_screen_preflight",
        "passed": True,
        "terminal": terminal,
        "candidate_inventory": inventory,
        "train_question_count": len(records),
        "changed_train_unit_count": len(units),
        "qa_audit": qa_audit,
        "gemma_loaded": False,
        "scene_maps_loaded": False,
        "validation_qa_loaded": False,
        "oracle_loaded": False,
        "final_test_scenes_touched": False,
        "optimizer_loaded_or_constructed": False,
        "output_already_exists": _resolve(DEFAULT_OUTPUT).exists(),
        "loaded_files": audit.unique_paths,
        "forbidden_file_accesses": audit.forbidden_accesses(),
    }


def run_screen(config_path: str | Path = DEFAULT_CONFIG) -> dict[str, Any]:
    terminal = require_terminal()
    config_path = _resolve(config_path)
    if _sha256(config_path) != _CONFIG_SHA256:
        raise ValueError("V42 config bytes changed")
    config = load_config(config_path)
    audit = FileAccessAudit(
        _forbidden_roots(config), forbidden_component_names={"oracle"}, block_forbidden=True
    )
    with audit:
        source, stopped, metadata8, stage8 = _endpoint_evidence()
        candidates, inventory = build_candidate_inventory(source, stopped)
        loader = v41_loader_config(config)
        assert_deferred_final_scenes_absent(loader)
        records, qa_audit = load_v35_train_qa_records(loader)
        units = build_exact_question_pair_units(records)
        if len(units) != 25:
            raise RuntimeError("V42 requires the exact 25 changed train units")
        inherited_schedule, _schedule_audit = build_v35_schedule(
            records,
            units,
            settings=v35_settings(loader),
            seed=int(config["seed"]),
        )
        broad_records = v36_broad_calibration_records(inherited_schedule)
        if len(broad_records) != 48:
            raise RuntimeError("V42 requires the exact 48-row broad calibration")

        contract = v41_contract(config)
        settings = v41_settings(config)
        hybrid, pinned_metadata, source_audit = require_exact_v41_sources(config)
        approved = require_approved_v29_source(loader)
        bundle, block_core, loaded_metadata, transition = load_v41_bundle(
            config, approved, contract.source_checkpoint, hybrid
        )
        if loaded_metadata != pinned_metadata:
            raise RuntimeError("V42 loaded source metadata changed")
        for module in bundle.checkpoint_modules.values():
            module.requires_grad_(False).eval()
        bundle.language.model.requires_grad_(False).eval()
        if any(parameter.requires_grad or parameter.grad is not None for parameter in bundle.language.model.parameters()):
            raise RuntimeError("V42 model is not gradient-free")

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
            raise RuntimeError("V42 frozen residual differs from authenticated source")

        target = _target_parameters(bundle)[0]
        source_target = source[_TARGET]
        if target_v41_state_sha256(bundle) != _TARGET_U0_SHA256:
            raise RuntimeError("V42 live target did not start at exact update zero")
        source_pair = _mapping(stage8["source_pair_metrics"], "V42 source pair metrics")
        source_nll = stage8["source_per_unit_nll_diagnostics"]
        source_broad = float(stage8["source_broad_train_nll"])
        source_deficit = float(priority_side_deficit(source_pair)["combined"])
        history8 = _mapping(metadata8["history"][8], "V42 persisted update-eight row")
        stopped_pair = _mapping(history8["training_pair_metrics"], "V42 u8 metrics")
        stopped_nll = history8["per_unit_nll_diagnostics"]
        stopped_broad = float(history8["training_broad_nll"])

        rows: list[dict[str, Any]] = []
        restoration_rows: list[dict[str, Any]] = []

        def restore_source(alpha: float) -> None:
            with torch.no_grad():
                target.copy_(source_target.to(device=target.device, dtype=target.dtype))
            restored = {
                "alpha_after": alpha,
                "target_state_sha256": target_v41_state_sha256(bundle),
                "full_state_sha256": module_collection_state_sha256(bundle.checkpoint_modules),
                "frozen_state_sha256": frozen_v41_state_sha256(bundle),
            }
            restored["passed"] = (
                restored["target_state_sha256"] == _TARGET_U0_SHA256
                and restored["full_state_sha256"] == _FULL_U0_SHA256
                and restored["frozen_state_sha256"] == _FROZEN_SHA256
            )
            restoration_rows.append(restored)
            if restored["passed"] is not True:
                raise RuntimeError("V42 failed to restore exact update zero")

        for alpha in _ALPHAS:
            expected = _CANDIDATE_HASHES[alpha_key(alpha)]
            try:
                with torch.no_grad():
                    target.copy_(candidates[alpha].to(device=target.device, dtype=target.dtype))
                if (
                    target_v41_state_sha256(bundle) != expected["target"]
                    or module_collection_state_sha256(bundle.checkpoint_modules) != expected["full"]
                    or frozen_v41_state_sha256(bundle) != _FROZEN_SHA256
                ):
                    raise RuntimeError("V42 live candidate state differs from preregistration")
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
                    query_state_sha256=expected["target"],
                    frozen_state_sha256=_FROZEN_SHA256,
                    scene_state_exact=True,
                    per_unit_nll_diagnostics=nll_rows,
                    contract=contract,
                )
                row = _candidate_summary(
                    alpha=alpha,
                    expected_hashes=expected,
                    pair_metrics=pair_metrics,
                    nll_rows=nll_rows,
                    broad_nll=broad_nll,
                    gate=gate,
                )
                rows.append(row)
                print(
                    json.dumps(
                        {
                            "phase": "v42_delta_candidate",
                            "alpha": alpha,
                            "complete_units": pair_metrics["complete_units"],
                            "positive_sides": pair_metrics["positive_sides"],
                            "cross_prefix_complete_units": pair_metrics[
                                "cross_prefix_complete_units"
                            ],
                            "priority_side_deficit": row["priority_side_deficit"],
                            "broad_nll": broad_nll,
                            "teacher_eligible": row["teacher_eligible"],
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
            finally:
                restore_source(alpha)

        endpoint_zero = rows[list(_ALPHAS).index(0.0)]
        endpoint_one = rows[list(_ALPHAS).index(1.0)]
        endpoint_checks = {
            "alpha_zero_pair_metrics": _numeric_close(endpoint_zero["pair_metrics"], source_pair),
            "alpha_zero_nll": _numeric_close(endpoint_zero["per_unit_nll_diagnostics"], source_nll),
            "alpha_zero_broad": math.isclose(endpoint_zero["broad_nll"], source_broad, rel_tol=0.0, abs_tol=_TOLERANCE),
            "alpha_one_pair_metrics": _numeric_close(endpoint_one["pair_metrics"], stopped_pair),
            "alpha_one_nll": _numeric_close(endpoint_one["per_unit_nll_diagnostics"], stopped_nll),
            "alpha_one_broad": math.isclose(endpoint_one["broad_nll"], stopped_broad, rel_tol=0.0, abs_tol=_TOLERANCE),
        }
        if not all(endpoint_checks.values()):
            raise RuntimeError(f"V42 endpoint replay failed: {endpoint_checks}")

        eligible = [row for row in rows if row["teacher_eligible"] is True]
        selected = min(eligible, key=candidate_rank_key) if eligible else None
        selected_replay: dict[str, Any] | None = None
        greedy: dict[str, Any] | None = None
        strong_greedy = False
        if selected is not None:
            alpha = float(selected["alpha"])
            expected = _CANDIDATE_HASHES[alpha_key(alpha)]
            try:
                with torch.no_grad():
                    target.copy_(candidates[alpha].to(device=target.device, dtype=target.dtype))
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
                    "alpha": alpha,
                    "target_state_sha256": expected["target"],
                    "pair_metrics_identical": _numeric_close(replay_pair, selected["pair_metrics"]),
                    "per_unit_nll_identical": _numeric_close(replay_nll, selected["per_unit_nll_diagnostics"]),
                    "broad_nll_identical": math.isclose(replay_broad, float(selected["broad_nll"]), rel_tol=0.0, abs_tol=_TOLERANCE),
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
                    raise RuntimeError("V42 selected-candidate replay changed")
                greedy = training_greedy_metrics(
                    units=units,
                    broad_records=broad_records,
                    caches=caches,
                    block_cross_residual=block_core,
                    bundle=bundle,
                    config=loader,
                )
                family = _mapping(greedy["complete_units_by_family"], "V42 greedy families")
                strong_greedy = bool(
                    int(greedy["complete_units"]) >= 6
                    and all(int(family.get(name, 0)) >= 1 for name in ("book_support", "mirror_lr", "picture_support"))
                    and int(greedy["broad_exact_correct"]) >= 23
                    and int(greedy["broad_row_count"]) == 48
                )
            finally:
                restore_source(alpha)

        final_state = {
            "target_state_sha256": target_v41_state_sha256(bundle),
            "full_state_sha256": module_collection_state_sha256(bundle.checkpoint_modules),
            "frozen_state_sha256": frozen_v41_state_sha256(bundle),
            "all_parameters_require_grad_false": not any(
                parameter.requires_grad for parameter in bundle.language.model.parameters()
            ),
            "all_parameter_gradients_absent": not any(
                parameter.grad is not None for parameter in bundle.language.model.parameters()
            ),
        }
        final_state["restored_exact"] = (
            final_state["target_state_sha256"] == _TARGET_U0_SHA256
            and final_state["full_state_sha256"] == _FULL_U0_SHA256
            and final_state["frozen_state_sha256"] == _FROZEN_SHA256
            and final_state["all_parameters_require_grad_false"] is True
            and final_state["all_parameter_gradients_absent"] is True
            and all(row["passed"] is True for row in restoration_rows)
        )
        if final_state["restored_exact"] is not True:
            raise RuntimeError("V42 final update-zero restoration failed")

    audit.assert_clean()
    loaded_maps = sorted(
        path for path in audit.unique_paths if "/maps/" in path and path.endswith(".npz")
    )
    if loaded_maps != sorted(cache_audit["loaded_environment_files"]):
        raise RuntimeError("V42 observed map reads differ from its train cache")
    return {
        "schema_version": 1,
        "artifact": "v42_v41_retry1_update8_no_step_diagnostic",
        "screen_integrity_passed": True,
        "terminal": terminal,
        "candidate_inventory": inventory,
        "candidate_results": rows,
        "endpoint_replay": {**endpoint_checks, "passed": all(endpoint_checks.values())},
        "teacher_eligible_candidate_found": selected is not None,
        "selected_alpha": None if selected is None else selected["alpha"],
        "selected_target_sha256": None if selected is None else selected["target_state_sha256"],
        "selected_candidate": selected,
        "selected_candidate_replay": selected_replay,
        "optional_selected_candidate_greedy_audit": greedy,
        "optional_greedy_audit_passed": strong_greedy,
        "restoration_audit": restoration_rows,
        "final_state": final_state,
        "source_audit": source_audit,
        "loader_transition": transition,
        "cache_boundary": cache_boundary,
        "scene_prefix_evidence": prefix_evidence,
        "qa_audit": qa_audit,
        "model_loaded_once": True,
        "optimizer_constructed_or_loaded": False,
        "gradient_measurement_performed": False,
        "candidate_checkpoint_written": False,
        "validation_qa_loaded": False,
        "oracle_loaded": False,
        "final_test_scenes_touched": False,
        "selector_execution_authorized": False,
        "validation_access_authorized": False,
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
    if path.exists():
        raise FileExistsError(f"V42 is a one-shot screen and will not overwrite {path}")
    report = run_screen(config_path)
    _atomic_json(path, report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    result = preflight(args.config) if args.preflight_only else write_report(
        args.output, config_path=args.config
    )
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "build_candidate_inventory",
    "candidate_rank_key",
    "candidate_tensor",
    "preflight",
    "run_screen",
    "write_report",
]
