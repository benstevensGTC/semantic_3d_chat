"""One released V6.1 MPS smoke with bounded numerical equivalence gates."""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from semantic_3d_chat.chat.file_audit import FileAccessAudit
from semantic_3d_chat.config import PROJECT_ROOT
from semantic_3d_chat.evaluation.fixed_prefix_decoder_reader_v6_1_release import (
    ARTIFACT,
    GRADIENT_EQUIVALENCE_THRESHOLDS,
    MPS_SMOKE_ATTEMPT,
    MPS_SMOKE_RELEASE,
    MPS_SMOKE_REPORT,
    OBJECTIVE_EQUIVALENCE_THRESHOLDS,
    V6_PREREGISTRATION_SHA256,
    V6_RELEASE_SHA256,
    V6_TERMINAL_FAILURE_SHA256,
    claim_v6_1_mps_smoke_attempt,
    gradient_equivalence_passes,
    objective_equivalence_passes,
    sha256_file,
)
from semantic_3d_chat.evaluation.fixed_prefix_decoder_reader_v6_preregistration import (
    INITIAL_STATE_SHA256,
    INITIALIZATION_SEED,
    TARGET_MODULES,
    answer_varying_wrong_prefixes,
    build_v6_schedule,
    decoder_reader_lora_settings_v6,
    structural_preflight,
)
from semantic_3d_chat.language.gemma4_answer_tail import (
    answer_tail_forward,
    answer_tail_model_kwargs,
    reference_answer_tail_from_full_logits,
)
from semantic_3d_chat.language.gemma4_tool_decoder_v2 import (
    INITIAL_LORA_STATE_SHA256 as TOOL_INITIAL_LORA_STATE_SHA256,
)
from semantic_3d_chat.language.gemma4_tool_decoder_v2 import (
    PROJECTOR_INITIALIZATION_SEED,
    tool_decoder_lora_settings_v2,
    validate_decoder_surface_v2,
)
from semantic_3d_chat.language.lora import (
    initialize_lora_adapter_state,
    install_lora_adapters,
    tensor_state_sha256,
)
from semantic_3d_chat.training import smoke_fixed_prefix_decoder_reader_v6 as v6
from semantic_3d_chat.training import train_fixed_prefix_ple_v54 as v1
from semantic_3d_chat.training import train_fixed_prefix_ple_v54_v4 as v4

_SCENE = "scene_000011"


class V61GateFailure(RuntimeError):
    """A terminal gate failure carrying all already-computed diagnostics."""

    def __init__(self, message: str, *, stage: str, metrics: dict[str, Any]) -> None:
        super().__init__(message)
        self.stage = stage
        self.metrics = metrics


def _resolve(path: str | Path) -> Path:
    value = Path(path).expanduser()
    return (value if value.is_absolute() else PROJECT_ROOT / value).resolve()


def _tensor_sha256(tensor: torch.Tensor) -> str:
    value = tensor.detach().contiguous().cpu()
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode())
    digest.update(json.dumps(list(value.shape), separators=(",", ":")).encode())
    digest.update(value.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _path_inventory_sha256(paths: list[str]) -> str:
    payload = json.dumps(paths, separators=(",", ":"), ensure_ascii=False).encode()
    return hashlib.sha256(payload).hexdigest()


def _atomic_create_report(value: dict[str, Any]) -> None:
    destination = _resolve(MPS_SMOKE_REPORT)
    if destination.exists() or destination.is_symlink():
        raise FileExistsError("V6.1 MPS smoke report already exists")
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = (
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode()
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def _prepared_hashes(prepared: Any) -> dict[str, dict[str, Any]]:
    fields: dict[str, dict[str, Any]] = {}
    for name in (
        "inputs_embeds",
        "attention_mask",
        "per_layer_inputs",
        "mm_token_type_ids",
        "labels",
    ):
        value = getattr(prepared, name, None)
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"V6.1 prepared batch lacks tensor {name}")
        fields[name] = {
            "sha256": _tensor_sha256(value),
            "shape": list(value.shape),
            "dtype": str(value.dtype),
        }
    return fields


def _postsoftcap_logits(model: Any, hidden_states: torch.Tensor) -> torch.Tensor:
    logits = model.lm_head(hidden_states)
    text = model.config.get_text_config()
    cap = getattr(text, "final_logit_softcapping", None)
    if cap is not None:
        logits = torch.tanh(logits / cap) * cap
    return logits


def _prediction_rank_metrics(
    reference: torch.Tensor,
    selected: torch.Tensor,
    targets: torch.Tensor,
) -> dict[str, Any]:
    top1_reference = reference.argmax(dim=-1)
    top1_selected = selected.argmax(dim=-1)
    top5_reference = reference.topk(5, dim=-1).indices
    top5_selected = selected.topk(5, dim=-1).indices
    top10_reference = reference.topk(10, dim=-1).indices
    top10_selected = selected.topk(10, dim=-1).indices
    overlap: list[float] = []
    reference_membership: list[bool] = []
    selected_membership: list[bool] = []
    reference_ranks: list[int] = []
    selected_ranks: list[int] = []
    reference_strict: list[int] = []
    selected_strict: list[int] = []
    confined = True
    deltas: list[float] = []
    tie_bands: list[float] = []
    maximum_crossed_gaps: list[float] = []
    for index, target in enumerate(targets.tolist()):
        ref_row = reference[index].double()
        sel_row = selected[index].double()
        ref_target = ref_row[target]
        sel_target = sel_row[target]
        delta = float((ref_row - sel_row).abs().max())
        tie_band = (
            float(OBJECTIVE_EQUIVALENCE_THRESHOLDS["rank_tie_band_multiplier"])
            * delta
        )
        deltas.append(delta)
        tie_bands.append(tie_band)
        ref_above = ref_row > ref_target
        sel_above = sel_row > sel_target
        reference_ranks.append(int(ref_above.sum()) + 1)
        selected_ranks.append(int(sel_above.sum()) + 1)
        reference_strict.append(int((ref_row > ref_target + tie_band).sum()) + 1)
        selected_strict.append(int((sel_row > sel_target + tie_band).sum()) + 1)
        changed = torch.logical_xor(ref_above, sel_above)
        if bool(changed.any()):
            # A rank may change only when the crossed token was inside the
            # preregistered 2*delta band around the reference target logit.
            ref_gaps = (ref_row[changed] - ref_target).abs()
            maximum_crossed_gap = float(ref_gaps.max())
            confined = confined and maximum_crossed_gap <= tie_band
        else:
            maximum_crossed_gap = 0.0
        maximum_crossed_gaps.append(maximum_crossed_gap)
        overlap.append(
            len(set(top5_reference[index].tolist()) & set(top5_selected[index].tolist()))
            / 5.0
        )
        reference_membership.append(bool((top10_reference[index] == target).any()))
        selected_membership.append(bool((top10_selected[index] == target).any()))
    strict_exact = reference_strict == selected_strict
    return {
        "reference_top1_token_ids": top1_reference.tolist(),
        "selected_top1_token_ids": top1_selected.tolist(),
        "top1_exact": torch.equal(top1_reference, top1_selected),
        "top5_overlap_fraction_by_token": overlap,
        "minimum_top5_overlap_fraction": min(overlap),
        "reference_target_top10_membership": reference_membership,
        "selected_target_top10_membership": selected_membership,
        "target_top10_membership_exact": reference_membership == selected_membership,
        "reference_target_ranks": reference_ranks,
        "selected_target_ranks": selected_ranks,
        "per_token_max_vocabulary_abs_logit_difference": deltas,
        "per_token_rank_tie_bands": tie_bands,
        "maximum_crossed_reference_target_gap_by_token": maximum_crossed_gaps,
        "target_rank_changes_confined_to_tie_band": confined,
        "reference_strict_above_band_ranks": reference_strict,
        "selected_strict_above_band_ranks": selected_strict,
        "strict_above_band_rank_exact": strict_exact,
    }


def _distribution_metrics(
    reference: torch.Tensor,
    selected: torch.Tensor,
    targets: torch.Tensor,
) -> dict[str, Any]:
    divergences: list[float] = []
    gradient_max = 0.0
    gradient_dot = 0.0
    reference_squared = 0.0
    selected_squared = 0.0
    for index, target in enumerate(targets.tolist()):
        ref_log = torch.log_softmax(reference[index].double(), dim=-1)
        sel_log = torch.log_softmax(selected[index].double(), dim=-1)
        ref_probability = ref_log.exp()
        sel_probability = sel_log.exp()
        mixture_log = torch.logaddexp(ref_log, sel_log) - math.log(2.0)
        divergence = 0.5 * (
            (ref_probability * (ref_log - mixture_log)).sum()
            + (sel_probability * (sel_log - mixture_log)).sum()
        )
        divergences.append(max(0.0, float(divergence)))
        ref_gradient = ref_probability
        sel_gradient = sel_probability
        ref_gradient[target] -= 1.0
        sel_gradient[target] -= 1.0
        gradient_max = max(
            gradient_max, float((ref_gradient - sel_gradient).abs().max())
        )
        gradient_dot += float(torch.dot(ref_gradient, sel_gradient))
        reference_squared += float(torch.dot(ref_gradient, ref_gradient))
        selected_squared += float(torch.dot(sel_gradient, sel_gradient))
    cosine = gradient_dot / math.sqrt(reference_squared * selected_squared)
    return {
        "js_divergence_by_token": divergences,
        "maximum_js_divergence": max(divergences),
        "softmax_ce_gradient_max_abs_difference": gradient_max,
        "softmax_ce_gradient_cosine_similarity": cosine,
    }


def _raw_logit_sufficient_statistics(
    reference: torch.Tensor, selected: torch.Tensor
) -> dict[str, Any]:
    """Return compact evidence sufficient to recompute every raw-logit metric."""

    if reference.shape != selected.shape or reference.ndim != 2:
        raise ValueError("V6.1 raw-logit sufficient-stat shapes differ")
    rows: list[dict[str, float | int]] = []
    for reference_row, selected_row in zip(reference, selected, strict=True):
        reference_double = reference_row.double()
        selected_double = selected_row.double()
        difference = reference_double - selected_double
        rows.append(
            {
                "vocabulary_count": int(difference.numel()),
                "difference_sum_abs": float(difference.abs().sum()),
                "difference_sum_squares": float(difference.square().sum()),
                "difference_max_abs": float(difference.abs().max()),
                "reference_selected_dot": float(
                    torch.dot(reference_double, selected_double)
                ),
                "reference_sum_squares": float(reference_double.square().sum()),
                "selected_sum_squares": float(selected_double.square().sum()),
            }
        )
    return {
        "reference_logits_sha256": _tensor_sha256(reference),
        "selected_logits_sha256": _tensor_sha256(selected),
        "per_token": rows,
    }


def _raw_logit_metrics(
    reference: torch.Tensor, selected: torch.Tensor
) -> dict[str, Any]:
    evidence = _raw_logit_sufficient_statistics(reference, selected)
    rows = evidence["per_token"]
    element_count = sum(int(row["vocabulary_count"]) for row in rows)
    sum_abs = sum(float(row["difference_sum_abs"]) for row in rows)
    sum_squares = sum(float(row["difference_sum_squares"]) for row in rows)
    per_token_cosine = [
        float(row["reference_selected_dot"])
        / math.sqrt(
            float(row["reference_sum_squares"])
            * float(row["selected_sum_squares"])
        )
        for row in rows
    ]
    return {
        "sufficient_statistics": evidence,
        "byte_exact": (
            evidence["reference_logits_sha256"]
            == evidence["selected_logits_sha256"]
        ),
        "max_abs_difference": max(
            float(row["difference_max_abs"]) for row in rows
        ),
        "rms_difference": math.sqrt(sum_squares / element_count),
        "mean_abs_difference": sum_abs / element_count,
        "per_token_cosine_similarity": per_token_cosine,
        "minimum_per_token_cosine_similarity": min(per_token_cosine),
    }


def _objective_equivalence(
    bundle: v1.ReaderBundle, row: v1.ReaderRecord
) -> tuple[dict[str, Any], torch.Tensor]:
    prepared = v1._prepared_batch(bundle, bundle.prefixes[row.scene_id], row)
    reference_prepared = _prepared_hashes(prepared)
    captures: list[torch.Tensor] = []
    norm = bundle.language.model.get_submodule("model.language_model.norm")

    def capture_hidden(_module: Any, _inputs: Any, output: Any) -> None:
        if not isinstance(output, torch.Tensor):
            raise TypeError("V6.1 final decoder norm hook did not receive a tensor")
        captures.append(output.detach().clone())

    hook = norm.register_forward_hook(capture_hidden)
    try:
        with torch.inference_mode():
            tail_kwargs, _positions = answer_tail_model_kwargs(prepared)
            full_kwargs = dict(tail_kwargs)
            full_kwargs.pop("logits_to_keep")
            full_kwargs["labels"] = prepared.labels
            full = bundle.language.model(**full_kwargs)
            if len(captures) != 1:
                raise RuntimeError("V6.1 full decoder hidden hook count changed")
            full_hidden = captures.pop()
            selected_prepared = _prepared_hashes(prepared)
            tail = answer_tail_forward(bundle.language, prepared)
            if len(captures) != 1:
                raise RuntimeError("V6.1 tail decoder hidden hook count changed")
            tail_hidden = captures.pop()
    finally:
        hook.remove()
    final_prepared = _prepared_hashes(prepared)
    prepared_fields: dict[str, dict[str, Any]] = {}
    for name, reference_value in reference_prepared.items():
        selected_value = selected_prepared[name]
        final_value = final_prepared[name]
        exact = (
            reference_value["sha256"]
            == selected_value["sha256"]
            == final_value["sha256"]
        )
        prepared_fields[name] = {
            "reference_sha256": reference_value["sha256"],
            "selected_sha256": final_value["sha256"],
            "shape": reference_value["shape"],
            "dtype": reference_value["dtype"],
            "exact": exact,
        }

    if full.loss is None:
        raise RuntimeError("V6.1 full batch-1 forward did not return HF loss")
    reference_tail = reference_answer_tail_from_full_logits(
        full.logits.float(), prepared.labels
    )
    causal = reference_tail.causal_positions
    full_selected_hidden = full_hidden[:, causal]
    tail_selected_hidden = tail_hidden[:, causal]
    with torch.inference_mode():
        common_reference = _postsoftcap_logits(
            bundle.language.model, full_selected_hidden
        )
        common_selected = _postsoftcap_logits(
            bundle.language.model, tail_selected_hidden
        )
        common_reference_nll = F.cross_entropy(
            common_reference[0].float(), reference_tail.targets, reduction="none"
        )
        common_selected_nll = F.cross_entropy(
            common_selected[0].float(), tail.targets, reduction="none"
        )

    reference_logits = reference_tail.logits.detach().float().cpu().contiguous()
    selected_logits = tail.logits.detach().float().cpu().contiguous()
    reference_matrix = reference_logits[0]
    selected_matrix = selected_logits[0]
    targets = reference_tail.targets.detach().cpu()
    raw_metrics = _raw_logit_metrics(reference_matrix, selected_matrix)
    reference_nll = reference_tail.per_token_nll.detach().float().cpu()
    selected_nll = tail.per_token_nll.detach().float().cpu()
    reference_nll_values = reference_nll.tolist()
    selected_nll_values = selected_nll.tolist()
    nll_max_difference = max(
        abs(float(left) - float(right))
        for left, right in zip(
            reference_nll_values, selected_nll_values, strict=True
        )
    )
    reference_nll_mean = sum(reference_nll_values) / len(reference_nll_values)
    selected_nll_mean = sum(selected_nll_values) / len(selected_nll_values)
    hf_loss = float(full.loss.detach().float().cpu())
    manual_ce = reference_nll_mean
    metrics: dict[str, Any] = {
        "contract_version": OBJECTIVE_EQUIVALENCE_THRESHOLDS["contract_version"],
        "thresholds": OBJECTIVE_EQUIVALENCE_THRESHOLDS,
        "token_count": int(targets.numel()),
        "vocabulary_size": int(reference_matrix.shape[-1]),
        "prepared_identity": {
            "fields": prepared_fields,
            "all_exact": all(value["exact"] for value in prepared_fields.values()),
        },
        "index_identity": {
            "target_token_ids": targets.tolist(),
            "label_positions": reference_tail.label_positions.detach().cpu().tolist(),
            "causal_positions": causal.detach().cpu().tolist(),
            "targets_exact": torch.equal(
                reference_tail.targets.detach().cpu(), tail.targets.detach().cpu()
            ),
            "label_positions_exact": torch.equal(
                reference_tail.label_positions.detach().cpu(),
                tail.label_positions.detach().cpu(),
            ),
            "causal_positions_exact": torch.equal(
                causal.detach().cpu(), tail.causal_positions.detach().cpu()
            ),
            "reference_targets_sha256": _tensor_sha256(reference_tail.targets),
            "selected_targets_sha256": _tensor_sha256(tail.targets),
            "reference_label_positions_sha256": _tensor_sha256(
                reference_tail.label_positions
            ),
            "selected_label_positions_sha256": _tensor_sha256(tail.label_positions),
            "reference_causal_positions_sha256": _tensor_sha256(causal),
            "selected_causal_positions_sha256": _tensor_sha256(
                tail.causal_positions
            ),
        },
        "hidden_identity": {
            "hook_module": "model.language_model.norm",
            "accessible": True,
            "entire_exact": torch.equal(full_hidden, tail_hidden),
            "selected_exact": torch.equal(full_selected_hidden, tail_selected_hidden),
            "entire_shape": list(full_hidden.shape),
            "selected_shape": list(full_selected_hidden.shape),
            "reference_entire_sha256": _tensor_sha256(full_hidden),
            "selected_entire_sha256": _tensor_sha256(tail_hidden),
            "reference_selected_sha256": _tensor_sha256(full_selected_hidden),
            "selected_selected_sha256": _tensor_sha256(tail_selected_hidden),
        },
        "common_shape_reprojection": {
            "shape": list(common_reference.shape),
            "logits_exact": torch.equal(common_reference, common_selected),
            "nll_exact": torch.equal(common_reference_nll, common_selected_nll),
            "reference_logits_sha256": _tensor_sha256(common_reference),
            "selected_logits_sha256": _tensor_sha256(common_selected),
            "reference_per_token_nll": common_reference_nll.detach().cpu().tolist(),
            "selected_per_token_nll": common_selected_nll.detach().cpu().tolist(),
        },
        "hf_loss_manual_ce": {
            "hf_batch1_loss": hf_loss,
            "manual_full_fp32_ce": manual_ce,
            "absolute_difference": abs(hf_loss - manual_ce),
            "passed": abs(hf_loss - manual_ce)
            <= OBJECTIVE_EQUIVALENCE_THRESHOLDS[
                "hf_loss_vs_manual_full_fp32_ce_abs"
            ],
        },
        "raw_postsoftcap_logits": {
            "reference_shape": list(reference_logits.shape),
            "selected_shape": list(selected_logits.shape),
            **raw_metrics,
        },
        "nll": {
            "reference_per_token": reference_nll_values,
            "selected_per_token": selected_nll_values,
            "max_abs_difference": nll_max_difference,
            "reference_mean": reference_nll_mean,
            "selected_mean": selected_nll_mean,
            "mean_absolute_difference": abs(
                reference_nll_mean - selected_nll_mean
            ),
        },
        "distribution": _distribution_metrics(
            reference_matrix, selected_matrix, targets
        ),
        "predictions_and_ranks": _prediction_rank_metrics(
            reference_matrix, selected_matrix, targets
        ),
        "passed": True,
    }
    if not objective_equivalence_passes(metrics):
        metrics["passed"] = False
    return metrics, selected_logits


def _gradient_snapshot(reader: Any) -> dict[str, Any]:
    tensors: dict[str, torch.Tensor] = {}
    b_norms: dict[str, float] = {}
    a_norms: dict[str, float] = {}
    a_exact_zero = True
    coverage: list[str] = []
    vectors: list[torch.Tensor] = []
    for target, adapter in zip(TARGET_MODULES, reader.adapters, strict=True):
        for suffix, parameter in (("lora_a", adapter.lora_a), ("lora_b", adapter.lora_b)):
            gradient = parameter.grad
            value = (
                torch.zeros_like(parameter, device="cpu", dtype=torch.float32)
                if gradient is None
                else gradient.detach().float().cpu().contiguous().clone()
            )
            tensors[f"{target}.{suffix}"] = value
            vectors.append(value.flatten())
            if suffix == "lora_a":
                a_norms[target] = float(value.double().norm())
                a_exact_zero = a_exact_zero and bool(torch.count_nonzero(value) == 0)
            else:
                norm = float(value.double().norm())
                b_norms[target] = norm
                if math.isfinite(norm) and norm > 0.0:
                    coverage.append(target)
    vector = torch.cat(vectors)
    return {
        "tensors": tensors,
        "vector": vector,
        "b_norms": b_norms,
        "a_norms": a_norms,
        "a_exact_zero": a_exact_zero,
        "coverage": coverage,
    }


def _clean_branch_backward(
    bundle: v1.ReaderBundle,
    reader: Any,
    prepared: Any,
    *,
    execution: str,
) -> tuple[float, dict[str, Any]]:
    bundle.language.model.zero_grad(set_to_none=True)
    if execution == "full":
        kwargs, _positions = answer_tail_model_kwargs(prepared)
        kwargs.pop("logits_to_keep")
        kwargs["labels"] = prepared.labels
        output = bundle.language.model(**kwargs)
        if output.loss is None:
            raise RuntimeError("V6.1 clean full branch returned no loss")
        loss = output.loss
    elif execution == "tail":
        loss = answer_tail_forward(bundle.language, prepared).mean_nll
    else:
        raise ValueError(f"Unknown V6.1 branch execution: {execution}")
    loss.backward()
    scalar = float(loss.detach().cpu())
    snapshot = _gradient_snapshot(reader)
    bundle.language.model.zero_grad(set_to_none=True)
    return scalar, snapshot


def _compare_gradient_snapshots(
    full: dict[str, Any], tail: dict[str, Any]
) -> dict[str, Any]:
    full_vector = full["vector"].double()
    tail_vector = tail["vector"].double()
    full_norm = float(full_vector.norm())
    tail_norm = float(tail_vector.norm())
    cosine = float(torch.dot(full_vector, tail_vector) / (full_norm * tail_norm))
    relative_l2 = float(
        (full_vector - tail_vector).norm() / max(full_norm, tail_norm)
    )
    ratio = tail_norm / full_norm
    thresholds = GRADIENT_EQUIVALENCE_THRESHOLDS
    coverage_exact = full["coverage"] == tail["coverage"] == list(TARGET_MODULES)
    full_tail_dot = float(torch.dot(full_vector, tail_vector))
    difference_squared = float((full_vector - tail_vector).square().sum())
    passed = (
        math.isfinite(cosine)
        and cosine >= thresholds["gradient_cosine_min"]
        and relative_l2 <= thresholds["gradient_relative_l2_max"]
        and thresholds["gradient_norm_ratio_min"]
        <= ratio
        <= thresholds["gradient_norm_ratio_max"]
        and full["a_exact_zero"]
        and tail["a_exact_zero"]
        and coverage_exact
    )
    return {
        "full_norm": full_norm,
        "tail_norm": tail_norm,
        "cosine_similarity": cosine,
        "relative_l2": relative_l2,
        "norm_ratio": ratio,
        "full_lora_b_gradient_l2_by_target": full["b_norms"],
        "tail_lora_b_gradient_l2_by_target": tail["b_norms"],
        "full_lora_a_gradient_l2_by_target": full["a_norms"],
        "tail_lora_a_gradient_l2_by_target": tail["a_norms"],
        "full_lora_a_exact_zero": full["a_exact_zero"],
        "tail_lora_a_exact_zero": tail["a_exact_zero"],
        "full_coverage": full["coverage"],
        "tail_coverage": tail["coverage"],
        "coverage_exact": coverage_exact,
        "sufficient_statistics": {
            "element_count": int(full_vector.numel()),
            "full_vector_sha256": _tensor_sha256(full["vector"]),
            "tail_vector_sha256": _tensor_sha256(tail["vector"]),
            "full_sum_squares": full_norm * full_norm,
            "tail_sum_squares": tail_norm * tail_norm,
            "full_tail_dot": full_tail_dot,
            "difference_sum_squares": difference_squared,
        },
        "passed": passed,
    }


def _aggregate_snapshots(
    snapshots: dict[str, dict[str, Any]], *, hinge_active: bool
) -> dict[str, Any]:
    weights = {
        "correct": 4.5 if hinge_active else 0.5,
        "wrong": -4.0 if hinge_active else 0.0,
        "broad": 0.5,
    }
    keys = tuple(snapshots["correct"]["tensors"])
    tensors = {
        key: sum(
            (weights[name] * snapshots[name]["tensors"][key] for name in weights),
            start=torch.zeros_like(snapshots["correct"]["tensors"][key]),
        )
        for key in keys
    }
    vector = torch.cat([tensors[key].flatten() for key in keys])
    b_norms: dict[str, float] = {}
    coverage: list[str] = []
    a_exact_zero = True
    for target in TARGET_MODULES:
        a_value = tensors[f"{target}.lora_a"]
        b_value = tensors[f"{target}.lora_b"]
        a_exact_zero = a_exact_zero and bool(torch.count_nonzero(a_value) == 0)
        norm = float(b_value.double().norm())
        b_norms[target] = norm
        if math.isfinite(norm) and norm > 0:
            coverage.append(target)
    return {
        "tensors": tensors,
        "vector": vector,
        "b_norms": b_norms,
        "a_norms": {
            target: float(tensors[f"{target}.lora_a"].double().norm())
            for target in TARGET_MODULES
        },
        "a_exact_zero": a_exact_zero,
        "coverage": coverage,
    }


def _assign_snapshot_gradients(reader: Any, snapshot: dict[str, Any]) -> None:
    for target, adapter in zip(TARGET_MODULES, reader.adapters, strict=True):
        for suffix, parameter in (("lora_a", adapter.lora_a), ("lora_b", adapter.lora_b)):
            value = snapshot["tensors"][f"{target}.{suffix}"]
            parameter.grad = value.to(parameter.device, dtype=parameter.dtype)


def _isolated_retention_backward(
    bundle: v1.ReaderBundle,
    reader: Any,
    retention_row: dict[str, str],
    retention_teacher: torch.Tensor,
    memory_sampler: Any,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Measure retention from zero gradients and leave gradients cleared."""

    bundle.language.model.zero_grad(set_to_none=True)
    retention = v1.retention_kl_loss(bundle, retention_row, retention_teacher)
    memory_sampler.sample("after_retention_forward")
    (0.5 * retention).backward()
    memory_sampler.sample("after_retention_backward")
    snapshot = _gradient_snapshot(reader)
    bundle.language.model.zero_grad(set_to_none=True)
    return retention, snapshot


def _clean_gradient_equivalence(
    bundle: v1.ReaderBundle,
    reader: Any,
    varying: v1.ReaderRecord,
    wrong_scene: str,
    broad_row: v1.ReaderRecord,
    retention_row: dict[str, str],
    retention_teacher: torch.Tensor,
    memory_sampler: Any,
) -> tuple[dict[str, Any], dict[str, float], dict[str, float]]:
    prepared = {
        "correct": v1._prepared_batch(
            bundle, bundle.prefixes[varying.scene_id], varying
        ),
        "wrong": v1._prepared_batch(bundle, bundle.prefixes[wrong_scene], varying),
        "broad": v1._prepared_batch(
            bundle, bundle.prefixes[broad_row.scene_id], broad_row
        ),
    }
    values: dict[str, dict[str, float]] = {"full": {}, "tail": {}}
    snapshots: dict[str, dict[str, dict[str, Any]]] = {"full": {}, "tail": {}}
    for name in ("correct", "wrong"):
        for execution in ("full", "tail"):
            value, snapshot = _clean_branch_backward(
                bundle, reader, prepared[name], execution=execution
            )
            values[execution][name] = value
            snapshots[execution][name] = snapshot
            torch.mps.empty_cache()
    memory_sampler.sample("after_contrastive_forwards")
    memory_sampler.sample("after_contrastive_backward")
    for execution in ("full", "tail"):
        value, snapshot = _clean_branch_backward(
            bundle, reader, prepared["broad"], execution=execution
        )
        values[execution]["broad"] = value
        snapshots[execution]["broad"] = snapshot
        torch.mps.empty_cache()
    memory_sampler.sample("after_broad_forward")
    memory_sampler.sample("after_broad_backward")

    full_margin = values["full"]["wrong"] - values["full"]["correct"]
    tail_margin = values["tail"]["wrong"] - values["tail"]["correct"]
    full_hinge = full_margin < 0.5
    tail_hinge = tail_margin < 0.5
    full_composite = (
        0.5 * values["full"]["correct"]
        + 4.0 * max(0.0, 0.5 - full_margin)
        + 0.5 * values["full"]["broad"]
    )
    tail_composite = (
        0.5 * values["tail"]["correct"]
        + 4.0 * max(0.0, 0.5 - tail_margin)
        + 0.5 * values["tail"]["broad"]
    )
    full_aggregate = _aggregate_snapshots(
        snapshots["full"], hinge_active=full_hinge
    )
    tail_aggregate = _aggregate_snapshots(
        snapshots["tail"], hinge_active=tail_hinge
    )
    comparisons = {
        name: _compare_gradient_snapshots(
            snapshots["full"][name], snapshots["tail"][name]
        )
        for name in ("correct", "wrong", "broad")
    }
    comparisons["aggregate"] = _compare_gradient_snapshots(
        full_aggregate, tail_aggregate
    )

    # Retention is measured as its own freshly-zeroed branch.  It must never
    # contaminate the clean zero-B first-schedule gradient whose LoRA-A
    # derivative is mathematically exact zero at initialization.
    retention, retention_snapshot = _isolated_retention_backward(
        bundle, reader, retention_row, retention_teacher, memory_sampler
    )
    _assign_snapshot_gradients(reader, tail_aggregate)
    metrics: dict[str, Any] = {
        "contract_version": GRADIENT_EQUIVALENCE_THRESHOLDS["contract_version"],
        "thresholds": GRADIENT_EQUIVALENCE_THRESHOLDS,
        "objective_values": {
            "full_correct_nll": values["full"]["correct"],
            "tail_correct_nll": values["tail"]["correct"],
            "correct_nll_abs_difference": abs(
                values["full"]["correct"] - values["tail"]["correct"]
            ),
            "full_wrong_nll": values["full"]["wrong"],
            "tail_wrong_nll": values["tail"]["wrong"],
            "wrong_nll_abs_difference": abs(
                values["full"]["wrong"] - values["tail"]["wrong"]
            ),
            "full_broad_nll": values["full"]["broad"],
            "tail_broad_nll": values["tail"]["broad"],
            "broad_nll_abs_difference": abs(
                values["full"]["broad"] - values["tail"]["broad"]
            ),
            "full_margin": full_margin,
            "tail_margin": tail_margin,
            "margin_abs_difference": abs(full_margin - tail_margin),
            "full_composite": full_composite,
            "tail_composite": tail_composite,
            "composite_abs_difference": abs(full_composite - tail_composite),
            "full_hinge_active": full_hinge,
            "tail_hinge_active": tail_hinge,
        },
        "gradient_comparisons": comparisons,
        "retention_self_kl": float(retention.detach().cpu()),
        "retention_gradient": {
            "measured_from_freshly_zeroed_gradients": True,
            "lora_a_exact_zero": retention_snapshot["a_exact_zero"],
            "lora_b_gradient_l2_by_target": retention_snapshot["b_norms"],
        },
        "passed": True,
    }
    if not gradient_equivalence_passes(metrics):
        metrics["passed"] = False
    return metrics, tail_aggregate["b_norms"], {
        target: float(tail_aggregate["tensors"][f"{target}.lora_a"].norm())
        for target in TARGET_MODULES
    }


def _execute_released_smoke(
    *, release_sha: str, attempt_sha: str, started: float
) -> dict[str, Any]:
    if not torch.backends.mps.is_available():
        raise RuntimeError("V6.1 released smoke requires available PyTorch MPS")
    if structural_preflight()["passed"] is not True:
        raise RuntimeError("V6.1 structural preflight failed after release")
    software_versions = v6._software_versions()
    memory_sampler = v6._MPSMemorySampler()
    memory_sampler.sample("before_model_load")
    torch.manual_seed(INITIALIZATION_SEED)
    bundle = v6._load_base_bundle()
    memory_sampler.sample("after_model_load_and_prefix_cache")
    train = v1.load_training_records()
    schedule = build_v6_schedule(train)
    varying = schedule[0].contrastive[0]
    broad_row = schedule[0].broad[0]
    wrong_scene = answer_varying_wrong_prefixes(train)[
        (varying.scene_id, varying.question_id)
    ]
    retention_row = v1.load_retention_corpus()[0]

    objective_metrics, frozen_logits = _objective_equivalence(bundle, varying)
    memory_sampler.sample("after_full_vs_tail_equivalence")
    with torch.inference_mode():
        retention_teacher = v4.bounded_retention_logits(
            bundle, retention_row["prompt"]
        ).detach().cpu()
    memory_sampler.sample("after_retention_teacher")
    if objective_metrics["passed"] is not True:
        raise V61GateFailure(
            "V6.1 bounded numerical objective equivalence failed",
            stage="objective_equivalence",
            metrics={"objective_equivalence": objective_metrics},
        )
    torch.mps.empty_cache()
    memory_sampler.sample("after_full_logit_cache_clear")

    reader = install_lora_adapters(
        bundle.language.model, decoder_reader_lora_settings_v6()
    )
    if reader is None:
        raise RuntimeError("V6.1 full-model reader adapter was not installed")
    initialize_lora_adapter_state(reader, seed=INITIALIZATION_SEED)
    if reader.state_sha256() != INITIAL_STATE_SHA256:
        raise ValueError("V6.1 full-model initial adapter state changed")
    reader.assert_only_lora_trainable(bundle.language.model)
    bundle.installation = reader
    memory_sampler.sample("after_v6_reader_install")
    zero_logits = v6._selected_logits(bundle, varying)
    memory_sampler.sample("after_v6_zero_output_forward")
    zero_noop = torch.equal(frozen_logits, zero_logits)
    if not zero_noop:
        raise V61GateFailure(
            "V6.1 zero-output adapter changed real answer logits",
            stage="v6_zero_output",
            metrics={
                "objective_equivalence": objective_metrics,
                "v6_zero_output_exact_noop": False,
            },
        )

    gradient_metrics, b_gradients, a_gradients = _clean_gradient_equivalence(
        bundle,
        reader,
        varying,
        wrong_scene,
        broad_row,
        retention_row,
        retention_teacher,
        memory_sampler,
    )
    if gradient_metrics["passed"] is not True:
        raise V61GateFailure(
            "V6.1 first-schedule objective or gradient equivalence failed",
            stage="gradient_equivalence",
            metrics={
                "objective_equivalence": objective_metrics,
                "gradient_equivalence": gradient_metrics,
            },
        )
    gradient_by_module = {
        target: {
            "lora_a": a_gradients[target],
            "lora_b": b_gradients[target],
            "total_l2": math.hypot(a_gradients[target], b_gradients[target]),
        }
        for target in TARGET_MODULES
    }
    gradient_total_l2 = math.sqrt(
        sum(
            a_gradients[target] ** 2 + b_gradients[target] ** 2
            for target in TARGET_MODULES
        )
    )
    both_nonzero = all(
        math.isfinite(value) and value > 0.0 for value in b_gradients.values()
    )
    if not both_nonzero or any(value != 0.0 for value in a_gradients.values()):
        raise V61GateFailure(
            "V6.1 real gradient did not reach both exact-zero-B adapters",
            stage="v6_gradient_validation",
            metrics={
                "objective_equivalence": objective_metrics,
                "gradient_equivalence": gradient_metrics,
                "v6_lora_b_gradient_l2_by_target": b_gradients,
                "v6_lora_a_gradient_l2_expected_zero_by_target": a_gradients,
            },
        )
    reader.validate_state()
    memory_sampler.sample("after_v6_gradient_validation")

    bundle.language.model.zero_grad(set_to_none=True)
    for parameter in reader.parameters():
        parameter.requires_grad_(False)
    validate_decoder_surface_v2(bundle.language.model)
    prepared_tool, projector = v6._tool_runtime_inputs(bundle)
    memory_sampler.sample("after_numeric_robot_tool_inputs")
    with torch.inference_mode():
        reader_only_tool = (
            answer_tail_forward(bundle.language, prepared_tool)
            .logits.detach()
            .cpu()
            .contiguous()
        )
    memory_sampler.sample("after_reader_only_tool_forward")
    tool = install_lora_adapters(
        bundle.language.model, tool_decoder_lora_settings_v2()
    )
    if tool is None:
        raise RuntimeError("V6.1 joint smoke tool adapter was not installed")
    initialize_lora_adapter_state(tool, seed=PROJECTOR_INITIALIZATION_SEED)
    if tool.state_sha256() != TOOL_INITIAL_LORA_STATE_SHA256:
        raise ValueError("V6.1 joint smoke tool adapter state changed")
    tool.assert_only_lora_trainable(bundle.language.model)
    memory_sampler.sample("after_zero_output_tool_install")
    with torch.inference_mode():
        joint_tool = (
            answer_tail_forward(bundle.language, prepared_tool)
            .logits.detach()
            .cpu()
            .contiguous()
        )
    memory_sampler.sample("after_joint_zero_output_tool_forward")
    joint_noop = torch.equal(reader_only_tool, joint_tool)
    if not joint_noop:
        raise V61GateFailure(
            "V6.1 plus zero-output tool adapter changed real tool logits",
            stage="joint_zero_output",
            metrics={
                "objective_equivalence": objective_metrics,
                "gradient_equivalence": gradient_metrics,
                "joint_zero_output_exact_noop": False,
            },
        )
    for parameter in tool.parameters():
        parameter.requires_grad_(False)
    roundtrip = v6._joint_state_roundtrip(reader, tool)
    memory_sampler.sample("after_joint_state_roundtrip")
    if any(parameter.requires_grad for parameter in bundle.language.model.parameters()):
        raise RuntimeError("V6.1 joint runtime smoke did not finish fully frozen")
    if any(parameter.requires_grad for parameter in projector.parameters()):
        raise RuntimeError("V6.1 numeric projector did not finish frozen")

    memory = memory_sampler.report()
    if memory["mps_driver_sample_count"] != 19:
        raise RuntimeError("V6.1 MPS smoke did not preserve all 19 memory phases")
    if memory["mps_driver_allocated_bytes_sampled_peak"] > 25_000_000_000:
        raise RuntimeError("V6.1 real MPS smoke exceeded its driver-memory gate")
    objective_values = gradient_metrics["objective_values"]
    return {
        "schema_version": 1,
        "artifact": f"{ARTIFACT}_real_mps_smoke",
        "status": "passed",
        "passed": True,
        "authorization_sha256": release_sha,
        "attempt_sha256": attempt_sha,
        "v6_parent_preregistration_sha256": V6_PREREGISTRATION_SHA256,
        "v6_parent_release_sha256": V6_RELEASE_SHA256,
        "v6_parent_terminal_failure_sha256": V6_TERMINAL_FAILURE_SHA256,
        "device": "mps",
        "software_versions": software_versions,
        "full_model_loaded": True,
        "mps_used": True,
        "optimizer_constructed": False,
        "optimizer_steps": 0,
        "training_executed": False,
        "checkpoint_published": False,
        "objective_equivalence": objective_metrics,
        "gradient_equivalence": gradient_metrics,
        "v6_zero_output_exact_noop": zero_noop,
        "v6_initial_state_sha256": reader.state_sha256(),
        "v6_gradient_l2": gradient_total_l2,
        "v6_gradient_by_module": gradient_by_module,
        "v6_lora_b_gradient_l2_by_target": b_gradients,
        "v6_lora_a_gradient_l2_expected_zero_by_target": a_gradients,
        "both_v6_adapter_gradients_nonzero": both_nonzero,
        "contrastive_correct_nll": objective_values["tail_correct_nll"],
        "contrastive_wrong_nll": objective_values["tail_wrong_nll"],
        "contrastive_margin": objective_values["tail_margin"],
        "broad_nll": objective_values["tail_broad_nll"],
        "retention_self_kl": gradient_metrics["retention_self_kl"],
        "joint_zero_output_structural_runtime_coexistence_passed": True,
        "joint_nonzero_semantic_or_tool_behavior_proven": False,
        "joint_zero_output_exact_noop": joint_noop,
        "tool_numeric_projector_state_sha256": tensor_state_sha256(
            projector.state_dict()
        ),
        "joint_state_roundtrip": roundtrip,
        "scene_prefix_shape": list(bundle.prefixes[_SCENE].shape),
        "question_dependent_scene_retrieval": False,
        "environmental_text_inputs": [],
        "memory": memory,
        "elapsed_seconds": time.perf_counter() - started,
    }


def run_released_full_model_mps_smoke_v6_1() -> dict[str, Any]:
    """Consume exactly one V6.1 attempt; terminalize every claimed outcome."""

    if any(_resolve(path).exists() for path in (MPS_SMOKE_ATTEMPT, MPS_SMOKE_REPORT)):
        raise FileExistsError("V6.1 MPS smoke attempt was already consumed")
    audit = FileAccessAudit(v6._forbidden_evaluation_roots(), block_forbidden=True)
    started = time.perf_counter()
    attempt_claimed = False
    attempt_sha: str | None = None
    release_sha: str | None = None
    core: dict[str, Any] | None = None
    failure: BaseException | None = None
    with audit:
        try:
            _attempt_path, attempt_sha = claim_v6_1_mps_smoke_attempt()
            attempt_claimed = True
            release_sha = sha256_file(MPS_SMOKE_RELEASE)
            core = _execute_released_smoke(
                release_sha=release_sha,
                attempt_sha=attempt_sha,
                started=started,
            )
            audit.assert_clean()
        except Exception as error:  # noqa: BLE001 - terminalize consumed attempt
            failure = error
    loaded = audit.unique_paths
    audit_summary = {
        "file_access_audit_active_for_entire_execution": True,
        "loaded_files": loaded,
        "loaded_file_count": len(loaded),
        "loaded_file_inventory_sha256": _path_inventory_sha256(loaded),
        "forbidden_file_accesses": audit.forbidden_accesses(),
        "deferred_or_final_qa_accessed": bool(audit.forbidden_accesses()),
    }
    if failure is not None:
        if attempt_claimed:
            _atomic_create_report(
                {
                    "schema_version": 1,
                    "artifact": f"{ARTIFACT}_real_mps_smoke",
                    "status": "failed_terminal_attempt_consumed",
                    "passed": False,
                    "authorization_sha256": release_sha,
                    "attempt_sha256": attempt_sha,
                    "optimizer_constructed": False,
                    "optimizer_steps": 0,
                    "training_executed": False,
                    "checkpoint_published": False,
                    "failure_type": type(failure).__name__,
                    "failure_message": str(failure),
                    "failure_stage": getattr(failure, "stage", "unclassified"),
                    "failure_metrics": getattr(failure, "metrics", {}),
                    **audit_summary,
                    "elapsed_seconds": time.perf_counter() - started,
                }
            )
        raise failure
    if core is None or attempt_sha is None or release_sha is None:
        raise RuntimeError("V6.1 MPS smoke ended without a result or failure")
    report = {**core, **audit_summary}
    _atomic_create_report(report)
    return report


__all__ = [
    "V61GateFailure",
    "run_released_full_model_mps_smoke_v6_1",
]
