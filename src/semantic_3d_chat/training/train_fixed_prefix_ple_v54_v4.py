"""Execute streamed answer-tail PLE-V54 V4 with exact loss equivalence."""

from __future__ import annotations

import argparse
import json
import math
import time
from collections import defaultdict
from collections.abc import Sequence
from typing import Any

import torch
import torch.nn.functional as F

from semantic_3d_chat.evaluation.fixed_prefix_ple_v54_v4_preregistration import (
    ARTIFACT,
    OUTPUT_CHECKPOINT,
    PREREGISTRATION,
    RESULT_REPORT,
    SMOKE_REPORT,
    authenticate_preregistration,
)
from semantic_3d_chat.training import train_fixed_prefix_ple_v54 as v1
from semantic_3d_chat.training.train_fixed_prefix_ple_v54_v3 import (
    canonical_tuple_mapping_hash,
)

_LEGACY_FULL_SEQUENCE_ANSWER_NLLS = v1.answer_nlls


def selected_answer_nll_from_full_logits(
    logits: torch.Tensor,
    labels: torch.Tensor,
) -> torch.Tensor:
    """Reference-select exact causal logits preceding supervised answer labels."""

    if logits.ndim != 3 or labels.ndim != 2 or logits.shape[:2] != labels.shape:
        raise ValueError("PLE-V54 V4 reference logits and labels must align")
    values: list[torch.Tensor] = []
    for index in range(labels.shape[0]):
        label_positions = torch.nonzero(labels[index].ne(-100), as_tuple=False).flatten()
        if label_positions.numel() < 1 or torch.any(label_positions <= 0):
            raise ValueError("PLE-V54 V4 answer labels require preceding causal positions")
        expected = torch.arange(
            int(label_positions[0]),
            int(label_positions[-1]) + 1,
            device=label_positions.device,
        )
        if not torch.equal(label_positions, expected):
            raise ValueError("PLE-V54 V4 answer labels must form one contiguous suffix")
        causal_positions = label_positions - 1
        selected = logits[index, causal_positions].float()
        targets = labels[index, label_positions]
        # Match ``token_normalized_nll``'s explicit per-token loss, FP32 sum,
        # and integer token-count division rather than relying on a reduction
        # kernel whose accumulation order can differ by a few ulps.
        losses = F.cross_entropy(selected, targets, reduction="none")
        values.append((losses.sum() / label_positions.numel()).reshape(1))
    return torch.cat(values)


def _single_streamed_answer_nll(
    bundle: v1.ReaderBundle,
    prefix: torch.Tensor,
    row: v1.ReaderRecord,
) -> torch.Tensor:
    """Forward one row and materialize vocab logits only at supervised positions."""

    batch = v1._prepared_batch(bundle, prefix, row)
    if batch.labels is None or batch.labels.shape[0] != 1:
        raise ValueError("PLE-V54 V4 requires one answer-labeled example")
    label_positions = torch.nonzero(batch.labels[0].ne(-100), as_tuple=False).flatten()
    if label_positions.numel() < 1 or torch.any(label_positions <= 0):
        raise ValueError("PLE-V54 V4 answer labels lack causal predecessor positions")
    expected = torch.arange(
        int(label_positions[0]),
        int(label_positions[-1]) + 1,
        device=label_positions.device,
    )
    if not torch.equal(label_positions, expected):
        raise ValueError("PLE-V54 V4 answer labels must be a contiguous suffix")
    causal_positions = (label_positions - 1).to(dtype=torch.long)
    kwargs: dict[str, Any] = {
        "inputs_embeds": batch.inputs_embeds,
        "attention_mask": batch.attention_mask,
        "use_cache": False,
        "labels": None,
        "logits_to_keep": causal_positions,
        "return_dict": True,
    }
    if bundle.language.prefix_backend is not None:
        if batch.per_layer_inputs is None or batch.mm_token_type_ids is None:
            raise ValueError("PLE-V54 V4 Gemma row lacks PLE or modality metadata")
        kwargs["per_layer_inputs"] = batch.per_layer_inputs
        kwargs["mm_token_type_ids"] = batch.mm_token_type_ids
    output = bundle.language.model(**kwargs)
    expected_shape = (1, label_positions.numel())
    if output.logits.shape[:2] != expected_shape:
        raise RuntimeError(
            "PLE-V54 V4 selected-logit shape changed: "
            f"{tuple(output.logits.shape[:2])} != {expected_shape}"
        )
    targets = batch.labels[0, label_positions]
    losses = F.cross_entropy(output.logits[0].float(), targets, reduction="none")
    nll = (losses.sum() / label_positions.numel()).reshape(1)
    if not torch.isfinite(nll).all():
        raise RuntimeError("PLE-V54 V4 streamed answer NLL is nonfinite")
    return nll


def streamed_answer_nlls(
    bundle: v1.ReaderBundle,
    examples: Sequence[tuple[torch.Tensor, v1.ReaderRecord]],
) -> torch.Tensor:
    """Stream examples independently while retaining each scalar autograd graph."""

    if not examples:
        raise ValueError("PLE-V54 V4 cannot score an empty example sequence")
    values = [
        _single_streamed_answer_nll(bundle, prefix, row)
        for prefix, row in examples
    ]
    result = torch.cat(values)
    if result.shape != (len(examples),) or not torch.isfinite(result).all():
        raise RuntimeError("PLE-V54 V4 streamed answer NLL vector is invalid")
    return result


def bounded_retention_logits(bundle: v1.ReaderBundle, prompt: str) -> torch.Tensor:
    """Keep only the single next-token logit vector used by retention gates."""

    ids = v1._retention_ids(bundle, prompt)
    output = bundle.language.model(
        input_ids=ids,
        use_cache=False,
        logits_to_keep=1,
        return_dict=True,
    )
    logits = output.logits[:, -1].float()
    if logits.ndim != 2 or logits.shape[0] != 1 or not torch.isfinite(logits).all():
        raise RuntimeError("PLE-V54 V4 bounded retention logits are invalid")
    return logits


@torch.inference_mode()
def evaluate_teacher_forcing_streamed(
    bundle: v1.ReaderBundle,
    rows: Sequence[v1.ReaderRecord],
) -> dict[str, Any]:
    """Exact V3 metrics with an explicit one-row evaluation microbatch."""

    bundle.installation.eval()
    correct: dict[tuple[str, str], float] = {}
    for row in rows:
        value = streamed_answer_nlls(
            bundle, ((bundle.prefixes[row.scene_id], row),)
        )[0]
        correct[(row.scene_id, row.question_id)] = float(value.cpu())
    changed = [row for row in rows if row.changed]
    wrong: dict[tuple[str, str], float] = {}
    for row in changed:
        if row.paired_scene_id is None:
            raise ValueError("PLE-V54 V4 validation changed row lacks paired scene")
        value = streamed_answer_nlls(
            bundle,
            ((bundle.prefixes[row.paired_scene_id], row),),
        )[0]
        wrong[(row.scene_id, row.question_id)] = float(value.cpu())
    margins = {key: wrong[key] - correct[key] for key in wrong}
    units: defaultdict[tuple[str, str], list[float]] = defaultdict(list)
    for row in changed:
        assert row.pair_id is not None and row.pair_question_key is not None
        units[(row.pair_id, row.pair_question_key)].append(
            margins[(row.scene_id, row.question_id)]
        )
    if len(units) != 26 or any(len(values) != 2 for values in units.values()):
        raise ValueError("PLE-V54 V4 validation changed-unit inventory changed")
    positive = sum(value > 0.0 for value in margins.values())
    complete = sum(all(value > 0.0 for value in values) for values in units.values())
    return {
        "answer_nll_mean": sum(correct.values()) / len(correct),
        "answer_nll_count": len(correct),
        "changed_margin_mean": sum(margins.values()) / len(margins),
        "changed_positive_margin_sides": positive,
        "changed_side_count": len(margins),
        "changed_positive_margin_rate": positive / len(margins),
        "changed_complete_units": complete,
        "changed_unit_count": len(units),
        "correct_nll_sha256": canonical_tuple_mapping_hash(correct),
        "changed_margin_sha256": canonical_tuple_mapping_hash(margins),
        "evaluation_microbatch_size": 1,
        "answer_logit_positions_only": True,
    }


def _activate_v4() -> None:
    v1.ARTIFACT = ARTIFACT
    v1.PREREGISTRATION = PREREGISTRATION
    v1.SMOKE_REPORT = SMOKE_REPORT
    v1.RESULT_REPORT = RESULT_REPORT
    v1.OUTPUT_CHECKPOINT = OUTPUT_CHECKPOINT
    v1.authenticate_preregistration = authenticate_preregistration
    v1.answer_nlls = streamed_answer_nlls
    v1.evaluate_teacher_forcing = evaluate_teacher_forcing_streamed
    v1.retention_logits = bounded_retention_logits


def structural_preflight() -> dict[str, Any]:
    _activate_v4()
    result = v1.structural_preflight()
    result["artifact"] = ARTIFACT
    result["v4_resource_contract"] = {
        "evaluation_microbatch_size": 1,
        "answer_logit_positions_only": True,
        "retention_logits_to_keep": 1,
        "numeric_objective_changed": False,
    }
    return result


def _synthetic_equivalence() -> dict[str, Any]:
    # Uniform deterministic logits yield an exactly representable shared CE
    # output under both kernel shapes, making this a byte-exact indexing and
    # normalization test. The real Gemma smoke below separately measures the
    # nonuniform full-vs-tail path to a preregistered 1e-6 tolerance.
    logits = torch.zeros(3, 11, 17, dtype=torch.float32)
    labels = torch.full((3, 11), -100, dtype=torch.long)
    labels[0, -2:] = torch.tensor([3, 5])
    labels[1, -4:] = torch.tensor([1, 7, 9, 2])
    labels[2, -1:] = torch.tensor([6])
    from semantic_3d_chat.training.pair_curriculum import token_normalized_nll

    # V4 streams one example at a time.  Compare against the legacy full-logit
    # function under that exact per-example execution contract; a multi-row CE
    # kernel may choose a different (mathematically equivalent) accumulation
    # order by a few ulps.
    full = torch.cat(
        [
            token_normalized_nll(logits[index : index + 1], labels[index : index + 1])
            for index in range(logits.shape[0])
        ]
    )
    selected = selected_answer_nll_from_full_logits(logits, labels)
    return {
        "full": full.tolist(),
        "selected": selected.tolist(),
        "maximum_absolute_difference": float((full - selected).abs().max()),
        "exact": torch.equal(full, selected),
    }


def equivalence_gradient_smoke() -> dict[str, Any]:
    _activate_v4()
    if v1._resolve(SMOKE_REPORT).exists():
        raise FileExistsError("PLE-V54 V4 smoke report already exists")
    preflight = structural_preflight()
    if preflight["passed"] is not True:
        raise RuntimeError("PLE-V54 V4 structural preflight failed")
    synthetic = _synthetic_equivalence()
    if synthetic["exact"] is not True:
        raise RuntimeError("PLE-V54 V4 synthetic answer-NLL equivalence failed")
    started = time.perf_counter()
    bundle = v1._load_bundle(gradient_checkpointing=True)
    train = v1.load_training_records()
    one = train[0]
    with torch.inference_mode():
        legacy = _LEGACY_FULL_SEQUENCE_ANSWER_NLLS(
            bundle, ((bundle.prefixes[one.scene_id], one),)
        ).detach()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    with torch.inference_mode():
        streamed = streamed_answer_nlls(
            bundle, ((bundle.prefixes[one.scene_id], one),)
        ).detach()
    real_difference = float((legacy - streamed).abs().max().cpu())
    changed = next(row for row in train if row.changed)
    corpus = v1.load_retention_corpus()
    teachers = v1.retention_baseline(bundle, corpus[:1])
    bundle.installation.train()
    bundle.language.model.zero_grad(set_to_none=True)
    loss, diagnostics = v1.changed_side_loss(bundle, changed)
    retention = v1.retention_kl_loss(bundle, corpus[0], teachers[0])
    total = loss + 0.2 * retention
    total.backward()
    gradients = bundle.installation.gradient_norms()
    bundle.installation.validate_state()
    memory = v1.memory_metrics()
    driver = memory["mps_driver_allocated_bytes"]
    retention_value = float(retention.detach().cpu())
    passed = (
        synthetic["exact"] is True
        and real_difference <= 1e-06
        and bool(torch.isfinite(total).item())
        and float(gradients["total_l2"]) > 0.0
        and math.isfinite(diagnostics["margin"])
        and abs(retention_value) <= 1e-05
        and (driver is None or driver <= 25_000_000_000)
    )
    report = {
        "schema_version": 1,
        "artifact": f"{ARTIFACT}_equivalence_gradient_smoke",
        "status": "passed" if passed else "failed",
        "passed": bool(passed),
        "synthetic_equivalence": synthetic,
        "real_one_row_full_nll": float(legacy[0].cpu()),
        "real_one_row_tail_nll": float(streamed[0].cpu()),
        "real_one_row_absolute_difference": real_difference,
        "real_one_row_tolerance": 1e-06,
        "tail_gradient_l2": gradients["total_l2"],
        "tail_gradient_by_module": gradients["by_module"],
        "retention_self_kl": retention_value,
        "loss": float(total.detach().cpu()),
        "changed_margin": diagnostics["margin"],
        "memory": memory,
        "elapsed_seconds": time.perf_counter() - started,
        "trainable_parameter_count": bundle.installation.parameter_count,
        "evaluation_microbatch_size": 1,
        "answer_logit_positions_only": True,
        "retention_logits_to_keep": 1,
        "question_dependent_scene_retrieval": False,
        "environmental_text_inputs": [],
        "preregistration_sha256": v1.sha256_file(PREREGISTRATION),
    }
    v1._atomic_create_json(SMOKE_REPORT, report)
    return report


def train_and_gate() -> dict[str, Any]:
    _activate_v4()
    return v1.train_and_gate()


def authenticate_result() -> dict[str, Any]:
    _activate_v4()
    return v1.authenticate_result()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "mode", choices=("preflight", "smoke", "train", "authenticate")
    )
    mode = parser.parse_args(argv).mode
    result = {
        "preflight": structural_preflight,
        "smoke": equivalence_gradient_smoke,
        "train": train_and_gate,
        "authenticate": authenticate_result,
    }[mode]()
    print(json.dumps(result, sort_keys=True, allow_nan=False), flush=True)
    return 0 if result.get("passed", True) else 1


if __name__ == "__main__":
    raise SystemExit(main())
