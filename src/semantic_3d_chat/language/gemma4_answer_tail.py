"""Memory-bounded causal answer-tail loss for Gemma-4 continuous prefixes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch.nn import functional as F


@dataclass(frozen=True)
class Gemma4AnswerTail:
    logits: torch.Tensor
    targets: torch.Tensor
    label_positions: torch.Tensor
    causal_positions: torch.Tensor
    per_token_nll: torch.Tensor
    mean_nll: torch.Tensor


def answer_tail_positions(labels: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return contiguous supervised positions and their causal predecessors."""

    if labels.ndim != 2 or labels.shape[0] != 1:
        raise ValueError("Gemma answer-tail execution requires labels with shape [1,L]")
    positions = torch.nonzero(labels[0].ne(-100), as_tuple=False).flatten()
    if positions.numel() < 1 or torch.any(positions <= 0):
        raise ValueError("Gemma answer-tail labels require causal predecessor positions")
    expected = torch.arange(
        int(positions[0]), int(positions[-1]) + 1, device=positions.device
    )
    if not torch.equal(positions, expected):
        raise ValueError("Gemma answer-tail labels must form one contiguous suffix")
    return positions, (positions - 1).to(dtype=torch.long)


def answer_tail_model_kwargs(prepared: Any) -> tuple[dict[str, Any], torch.Tensor]:
    """Build an explicit labels=None selected-logit Gemma call."""

    labels = getattr(prepared, "labels", None)
    if not isinstance(labels, torch.Tensor):
        raise TypeError("Gemma answer-tail inputs have no supervised labels")
    label_positions, causal_positions = answer_tail_positions(labels)
    inputs_embeds = getattr(prepared, "inputs_embeds", None)
    attention_mask = getattr(prepared, "attention_mask", None)
    per_layer_inputs = getattr(prepared, "per_layer_inputs", None)
    mm_token_type_ids = getattr(prepared, "mm_token_type_ids", None)
    if not all(
        isinstance(value, torch.Tensor)
        for value in (
            inputs_embeds,
            attention_mask,
            per_layer_inputs,
            mm_token_type_ids,
        )
    ):
        raise ValueError("Gemma answer-tail inputs lack PLE or modality tensors")
    return (
        {
            "inputs_embeds": inputs_embeds,
            "per_layer_inputs": per_layer_inputs,
            "attention_mask": attention_mask,
            "mm_token_type_ids": mm_token_type_ids,
            "use_cache": False,
            "labels": None,
            "logits_to_keep": causal_positions,
            "return_dict": True,
        },
        label_positions,
    )


def answer_tail_forward(language: Any, prepared: Any) -> Gemma4AnswerTail:
    """Materialize vocabulary logits only at answer-predicting positions."""

    if getattr(language, "backend_name", None) != "gemma4":
        raise ValueError("Gemma answer-tail execution requires the Gemma-4 backend")
    kwargs, label_positions = answer_tail_model_kwargs(prepared)
    output = language.model(**kwargs)
    logits = output.logits
    expected_shape = (1, int(label_positions.numel()))
    if not isinstance(logits, torch.Tensor) or logits.shape[:2] != expected_shape:
        raise RuntimeError(
            "Gemma answer-tail selected-logit shape changed: "
            f"{tuple(logits.shape[:2])} != {expected_shape}"
        )
    targets = prepared.labels[0, label_positions]
    losses = F.cross_entropy(logits[0].float(), targets, reduction="none")
    mean = losses.sum() / label_positions.numel()
    if not torch.isfinite(losses).all() or not torch.isfinite(mean):
        raise RuntimeError("Gemma answer-tail NLL is nonfinite")
    return Gemma4AnswerTail(
        logits=logits,
        targets=targets,
        label_positions=label_positions,
        causal_positions=label_positions - 1,
        per_token_nll=losses,
        mean_nll=mean,
    )


def reference_answer_tail_from_full_logits(
    logits: torch.Tensor,
    labels: torch.Tensor,
) -> Gemma4AnswerTail:
    """Select the mathematically identical answer tail from full logits."""

    if logits.ndim != 3 or labels.ndim != 2 or logits.shape[:2] != labels.shape:
        raise ValueError("Full Gemma logits and labels must align")
    label_positions, causal_positions = answer_tail_positions(labels)
    selected = logits[:, causal_positions]
    targets = labels[0, label_positions]
    losses = F.cross_entropy(selected[0].float(), targets, reduction="none")
    mean = losses.sum() / label_positions.numel()
    if not torch.isfinite(losses).all() or not torch.isfinite(mean):
        raise RuntimeError("Full-logit answer-tail reference is nonfinite")
    return Gemma4AnswerTail(
        logits=selected,
        targets=targets,
        label_positions=label_positions,
        causal_positions=causal_positions,
        per_token_nll=losses,
        mean_nll=mean,
    )


__all__ = [
    "Gemma4AnswerTail",
    "answer_tail_forward",
    "answer_tail_model_kwargs",
    "answer_tail_positions",
    "reference_answer_tail_from_full_logits",
]
