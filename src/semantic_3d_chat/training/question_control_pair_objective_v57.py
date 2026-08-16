"""Delta-sensitive training objective for the V1 full-scene question controller.

V56 optimized the two sides of a counterfactual unit with independent answer
cross entropy.  That objective admits a question-only saddle: the controller can
emit almost identical tokens for both physical scenes and settle on the more
probable answer.  This module keeps the runtime architecture unchanged and adds
training-only evidence that explicitly compares the two continuous scene paths.

No labels or answer embeddings are stored in the runtime checkpoint.  They are
used only inside supervised training, while inference still receives the room
through the complete, immutable continuous scene prefix.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F

from semantic_3d_chat.data.dataset import QARecord
from semantic_3d_chat.language.local_lm import prompt_token_ids, question_token_ids
from semantic_3d_chat.language.prefix_injection import (
    PrefixBatch,
    scene_boundary_mode_setting,
    scene_prefix_after_bos_setting,
    stack_prefix_batches,
)
from semantic_3d_chat.scene_encoder.question_control import FullSceneQuestionControl
from semantic_3d_chat.training.pair_curriculum import (
    differing_answer_token_masks,
    pair_ranking_hinge,
    restrict_labels_to_answer_mask,
    single_differing_answer_token,
    token_normalized_nll,
)
from semantic_3d_chat.training.train_adapter import forward_prefix_batch, tokenize_answer
from semantic_3d_chat.training.train_question_control_v56 import (
    _prepared_to_prefix_batch,
    assert_answer_only_labels,
)


@dataclass(frozen=True)
class V57PairObjectiveSettings:
    """Weights and margins for one atomic changed-scene optimizer step."""

    answer_nll_weight: float = 1.0
    side_hinge_weight: float = 0.5
    side_margin: float = 0.5
    cross_prefix_hinge_weight: float = 1.0
    cross_prefix_margin: float = 0.1
    control_delta_weight: float = 8.0
    minimum_relative_control_delta: float = 0.03
    attention_entropy_weight: float = 2.0
    minimum_normalized_attention_entropy: float = 0.55
    attention_logit_spread_weight: float = 1.0
    maximum_attention_logit_rms: float = 1.0
    answer_alignment_weight: float = 2.0
    answer_alignment_margin: float = 0.1
    answer_absolute_alignment_weight: float = 1.0
    answer_delta_alignment_weight: float = 2.0

    def __post_init__(self) -> None:
        values = {
            "answer_nll_weight": self.answer_nll_weight,
            "side_hinge_weight": self.side_hinge_weight,
            "side_margin": self.side_margin,
            "cross_prefix_hinge_weight": self.cross_prefix_hinge_weight,
            "cross_prefix_margin": self.cross_prefix_margin,
            "control_delta_weight": self.control_delta_weight,
            "minimum_relative_control_delta": self.minimum_relative_control_delta,
            "attention_entropy_weight": self.attention_entropy_weight,
            "minimum_normalized_attention_entropy": (
                self.minimum_normalized_attention_entropy
            ),
            "attention_logit_spread_weight": self.attention_logit_spread_weight,
            "maximum_attention_logit_rms": self.maximum_attention_logit_rms,
            "answer_alignment_weight": self.answer_alignment_weight,
            "answer_alignment_margin": self.answer_alignment_margin,
            "answer_absolute_alignment_weight": (
                self.answer_absolute_alignment_weight
            ),
            "answer_delta_alignment_weight": self.answer_delta_alignment_weight,
        }
        for field, value in values.items():
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or float(value) < 0.0
            ):
                raise ValueError(f"V57 {field} must be finite and nonnegative")
        if self.minimum_normalized_attention_entropy > 1.0:
            raise ValueError("V57 normalized attention-entropy floor cannot exceed one")
        if self.answer_nll_weight == 0.0 and not any(
            value > 0.0
            for value in (
                self.side_hinge_weight,
                self.cross_prefix_hinge_weight,
                self.control_delta_weight,
                self.answer_alignment_weight,
                self.answer_absolute_alignment_weight,
                self.answer_delta_alignment_weight,
            )
        ):
            raise ValueError("V57 pair objective enables no answer-sensitive loss")

    def contract(self) -> dict[str, float]:
        return {
            field: float(getattr(self, field))
            for field in self.__dataclass_fields__
        }


def _full_scene_control_components(
    control: FullSceneQuestionControl,
    scene_prefix: torch.Tensor,
    question_embeddings: torch.Tensor,
    question_attention_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run the exact V1 controller while retaining differentiable attention.

    The production controller deliberately stores only a detached audit copy.
    V57 needs the live attention probabilities for its train-only entropy floor,
    so this function mirrors that small public-module computation.  Tests prove
    numerical identity with :meth:`FullSceneQuestionControl.forward`.
    """

    if scene_prefix.ndim != 3 or question_embeddings.ndim != 3:
        raise ValueError("V57 scene and question tensors must have shape [B,L,H]")
    if scene_prefix.shape[0] != question_embeddings.shape[0]:
        raise ValueError("V57 scene and question batch sizes must match")
    if (
        scene_prefix.shape[-1] != control.hidden_size
        or question_embeddings.shape[-1] != control.hidden_size
    ):
        raise ValueError("V57 controller hidden dimension mismatch")
    if scene_prefix.shape[1] < 1 or question_embeddings.shape[1] < 1:
        raise ValueError("V57 controller inputs must be nonempty")
    if not torch.isfinite(scene_prefix).all() or not torch.isfinite(
        question_embeddings
    ).all():
        raise ValueError("V57 controller inputs must be finite")

    if question_attention_mask is None:
        pooled_question = question_embeddings.mean(dim=1)
    else:
        if question_attention_mask.shape != question_embeddings.shape[:2]:
            raise ValueError("V57 question mask must have shape [B,Q]")
        weights = question_attention_mask.to(question_embeddings).clamp(0.0, 1.0)
        counts = weights.sum(dim=1, keepdim=True)
        if torch.any(counts <= 0.0):
            raise ValueError("V57 every question requires an unmasked token")
        pooled_question = torch.sum(
            question_embeddings * weights.unsqueeze(-1), dim=1
        ) / counts

    normalized_scene = control.scene_norm(scene_prefix)
    normalized_question = control.question_norm(pooled_question)
    query = control.query(normalized_question).reshape(
        scene_prefix.shape[0], control.control_token_count, control.attention_dim
    )
    query = query + control.control_identity
    keys = control.key(normalized_scene)
    values = control.value(normalized_scene)
    logits = torch.einsum("bcd,bld->bcl", query, keys) / math.sqrt(
        control.attention_dim
    )
    learned = torch.softmax(logits.float(), dim=-1).to(logits.dtype)
    attention = (1.0 - control.uniform_floor) * learned + control.uniform_floor / float(
        scene_prefix.shape[1]
    )
    context = torch.einsum("bcl,bld->bcd", attention, values)
    output = control.output(context + query) * control.output_scale
    if not torch.isfinite(output).all() or not torch.isfinite(attention).all():
        raise RuntimeError("V57 controller output contains NaN or infinity")
    return output, attention, logits


def full_scene_control_with_attention(
    control: FullSceneQuestionControl,
    scene_prefix: torch.Tensor,
    question_embeddings: torch.Tensor,
    question_attention_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Public two-result wrapper used to prove V1 runtime equivalence."""

    output, attention, _logits = _full_scene_control_components(
        control,
        scene_prefix,
        question_embeddings,
        question_attention_mask,
    )
    return output, attention


def normalized_attention_entropy(attention: torch.Tensor) -> torch.Tensor:
    """Return entropy in [0,1] for every batch/control-token distribution."""

    if attention.ndim != 3 or attention.shape[-1] < 2:
        raise ValueError("V57 attention must have shape [B,C,L] with L >= 2")
    probabilities = attention.float()
    if (
        not torch.isfinite(probabilities).all()
        or torch.any(probabilities <= 0.0)
        or not torch.allclose(
            probabilities.sum(dim=-1),
            torch.ones_like(probabilities.sum(dim=-1)),
            atol=2e-5,
            rtol=2e-5,
        )
    ):
        raise ValueError("V57 attention must contain normalized positive probabilities")
    entropy = -(probabilities * probabilities.log()).sum(dim=-1)
    return entropy / math.log(probabilities.shape[-1])


def attention_entropy_hinge(
    attention: torch.Tensor,
    *,
    minimum_normalized_entropy: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    entropy = normalized_attention_entropy(attention)
    loss = torch.relu(float(minimum_normalized_entropy) - entropy).mean()
    return loss, entropy


def attention_logit_spread_penalty(
    logits: torch.Tensor,
    *,
    maximum_rms: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Penalize saturated attention through pre-softmax logits.

    Entropy gradients become very small after softmax has already collapsed.
    Centered logit RMS remains directly differentiable in that regime and is
    used only as a train-time guardrail; it does not alter runtime attention.
    """

    if logits.ndim != 3 or logits.shape[-1] < 2:
        raise ValueError("V57 attention logits must have shape [B,C,L] with L >= 2")
    values = logits.float()
    if not torch.isfinite(values).all():
        raise ValueError("V57 attention logits must be finite")
    centered = values - values.mean(dim=-1, keepdim=True)
    rms = centered.square().mean(dim=-1).sqrt()
    excess = torch.relu(rms - float(maximum_rms))
    return excess.square().mean(), rms


def relative_control_delta(control_tokens: torch.Tensor) -> torch.Tensor:
    """Measure scene-side separation relative to the two control magnitudes."""

    if control_tokens.ndim != 3 or control_tokens.shape[0] != 2:
        raise ValueError("V57 changed pair control tokens must have shape [2,C,H]")
    values = control_tokens.float()
    if not torch.isfinite(values).all():
        raise ValueError("V57 control tokens must be finite")
    numerator = torch.linalg.vector_norm(values[0] - values[1])
    denominator = 0.5 * (
        torch.linalg.vector_norm(values[0]) + torch.linalg.vector_norm(values[1])
    )
    return numerator / denominator.clamp_min(torch.finfo(values.dtype).eps)


def control_delta_hinge(
    control_tokens: torch.Tensor,
    *,
    minimum_relative_delta: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    delta = relative_control_delta(control_tokens)
    return torch.relu(float(minimum_relative_delta) - delta), delta


def answer_alignment_hinge(
    control_tokens: torch.Tensor,
    answer_targets: torch.Tensor,
    *,
    margin: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Prefer each scene control over the paired answer's frozen embedding.

    This supervision is train-only.  The frozen answer embeddings and QA text
    are neither checkpointed nor available to the inference process.
    """

    if control_tokens.ndim != 3 or control_tokens.shape[0] != 2:
        raise ValueError("V57 answer alignment requires control shape [2,C,H]")
    if answer_targets.shape != (2, control_tokens.shape[-1]):
        raise ValueError("V57 answer targets must have shape [2,H]")
    controls = F.normalize(control_tokens.float().mean(dim=1), dim=-1, eps=1e-8)
    targets = F.normalize(answer_targets.detach().float(), dim=-1, eps=1e-8)
    similarities = controls @ targets.transpose(0, 1)
    margins = torch.stack(
        (similarities[0, 0] - similarities[0, 1], similarities[1, 1] - similarities[1, 0])
    )
    return torch.relu(float(margin) - margins).mean(), margins, similarities


def answer_delta_alignment_loss(
    control_tokens: torch.Tensor,
    answer_targets: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Align the explicit scene-side control delta with the answer delta."""

    if control_tokens.ndim != 3 or control_tokens.shape[0] != 2:
        raise ValueError("V57 delta alignment requires control shape [2,C,H]")
    if answer_targets.shape != (2, control_tokens.shape[-1]):
        raise ValueError("V57 delta answer targets must have shape [2,H]")
    control_means = control_tokens.float().mean(dim=1)
    control_delta = control_means[0] - control_means[1]
    target_delta = answer_targets.detach().float()[0] - answer_targets.detach().float()[1]
    control_norm = torch.linalg.vector_norm(control_delta)
    target_norm = torch.linalg.vector_norm(target_delta)
    if not torch.isfinite(control_norm) or not torch.isfinite(target_norm):
        raise ValueError("V57 answer-delta alignment contains nonfinite values")
    if float(target_norm.detach().cpu()) <= 1e-8:
        raise ValueError("V57 paired answer embeddings have zero delta")
    cosine = F.cosine_similarity(
        control_delta.unsqueeze(0), target_delta.unsqueeze(0), dim=-1, eps=1e-8
    ).squeeze(0)
    return 1.0 - cosine, cosine


def answer_absolute_alignment_loss(
    alignment_similarities: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pull each control mean directly toward its own frozen answer embedding."""

    if alignment_similarities.shape != (2, 2):
        raise ValueError("V57 answer similarities must have shape [2,2]")
    if not torch.isfinite(alignment_similarities).all():
        raise ValueError("V57 answer similarities must be finite")
    own_similarities = torch.diagonal(alignment_similarities)
    return (1.0 - own_similarities).mean(), own_similarities


def pair_and_cross_prefix_hinges(
    *,
    correct_rank_nll: torch.Tensor,
    swapped_rank_nll: torch.Tensor,
    side_margin: float,
    cross_prefix_margin: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compare both candidate answers within and across the two scene prefixes."""

    if correct_rank_nll.shape != (2,) or swapped_rank_nll.shape != (2,):
        raise ValueError("V57 pair rank score vectors must each have shape [2]")
    side_hinge, side_margins = pair_ranking_hinge(
        correct_rank_nll,
        swapped_rank_nll,
        margin=side_margin,
    )
    cross_margins = torch.stack(
        (
            swapped_rank_nll[1] - correct_rank_nll[0],
            swapped_rank_nll[0] - correct_rank_nll[1],
        )
    )
    cross_hinge = torch.relu(float(cross_prefix_margin) - cross_margins).mean()
    return side_hinge, side_margins, cross_hinge, cross_margins


def _aligned_candidate_nll(
    logits: torch.Tensor,
    labels: torch.Tensor,
    first_answer_ids: torch.Tensor,
    second_answer_ids: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Score a one-token answer change from one two-scene LM forward pass."""

    offset, first_token, second_token = single_differing_answer_token(
        first_answer_ids, second_answer_ids
    )
    if logits.ndim != 3 or labels.ndim != 2 or logits.shape[:2] != labels.shape:
        raise ValueError("V57 logits and labels are not aligned")
    if logits.shape[0] != 2:
        raise ValueError("V57 candidate scoring requires exactly two scene rows")
    prediction_positions: list[int] = []
    for row, expected in enumerate((first_answer_ids, second_answer_ids)):
        supervised = labels[row].ne(-100).nonzero(as_tuple=False).flatten()
        expected_flat = expected.reshape(-1).to(labels.device)
        if supervised.numel() != expected_flat.numel() or not torch.equal(
            labels[row, supervised], expected_flat
        ):
            raise ValueError("V57 answer label alignment changed")
        label_position = int(supervised[offset].item())
        if label_position < 1:
            raise ValueError("V57 answer token lacks a causal prediction position")
        prediction_positions.append(label_position - 1)
    rows = torch.arange(2, device=logits.device)
    positions = torch.tensor(prediction_positions, device=logits.device)
    log_probabilities = torch.log_softmax(logits[rows, positions].float(), dim=-1)
    correct_ids = torch.tensor((first_token, second_token), device=logits.device)
    swapped_ids = torch.tensor((second_token, first_token), device=logits.device)
    return (
        -log_probabilities.gather(1, correct_ids[:, None]).squeeze(1),
        -log_probabilities.gather(1, swapped_ids[:, None]).squeeze(1),
    )


def _compose_batch(
    *,
    runtime: Any,
    scene_prefix: torch.Tensor,
    record: QARecord,
    answer: str,
    control_tokens: torch.Tensor,
) -> tuple[PrefixBatch, torch.Tensor]:
    language = runtime.language
    backend = language.prefix_backend
    if backend is None or language.backend_name != "gemma4":
        raise RuntimeError("V57 pair training requires the Gemma prefix backend")
    prompt_ids = prompt_token_ids(
        language.tokenizer,
        str(runtime.config["language"]["system_prompt"]),
        record.question,
        language.device,
    )
    answer_ids = tokenize_answer(language.tokenizer, answer, language.device)
    prepared = backend.prepare(
        scene_prefix,
        prompt_ids,
        answer_ids,
        scene_prefix_after_bos=scene_prefix_after_bos_setting(runtime.config),
        scene_boundary_mode=scene_boundary_mode_setting(runtime.config),
        control_tokens=control_tokens.to(scene_prefix),
    )
    assert_answer_only_labels(prepared.labels, answer_ids)
    return _prepared_to_prefix_batch(prepared), answer_ids


def _canonical_answer_target(
    embedding_layer: torch.nn.Module,
    answer_ids: torch.Tensor,
    *,
    eos_token_id: int | None,
) -> torch.Tensor:
    ids = answer_ids.reshape(-1)
    keep = torch.ones(ids.shape, dtype=torch.bool, device=ids.device)
    if eos_token_id is not None:
        keep &= ids.ne(int(eos_token_id))
    if not keep.any():
        keep[:] = True
    with torch.no_grad():
        embeddings = embedding_layer(answer_ids).detach().float().squeeze(0)
    if embeddings.ndim != 2 or embeddings.shape[0] != ids.numel():
        raise ValueError("V57 frozen answer-embedding shape changed")
    return embeddings[keep].mean(dim=0)


def paired_question_control_objective(
    *,
    runtime: Any,
    control: FullSceneQuestionControl,
    prefixes: dict[str, torch.Tensor],
    records: tuple[QARecord, QARecord],
    settings: V57PairObjectiveSettings,
) -> tuple[torch.Tensor, dict[str, torch.Tensor | bool]]:
    """Compute one delta-sensitive loss for a locked two-scene changed unit."""

    if len(records) != 2:
        raise ValueError("V57 changed unit requires exactly two records")
    first, second = records
    if (
        first.scene_id == second.scene_id
        or first.question != second.question
        or first.answer == second.answer
        or first.counterfactual_expected_change is not True
        or second.counterfactual_expected_change is not True
    ):
        raise ValueError("V57 changed unit is not a same-question, changed-answer pair")
    language = runtime.language
    embedding_layer = language.model.get_input_embeddings()
    model_dtype = next(language.model.parameters()).dtype
    try:
        scene_prefix = torch.cat(
            tuple(
                prefixes[record.scene_id].to(
                    device=language.device,
                    dtype=model_dtype,
                )
                for record in records
            ),
            dim=0,
        )
    except KeyError as exc:
        raise KeyError(f"V57 missing immutable prefix for {exc.args[0]}") from exc
    question_ids = question_token_ids(
        language.tokenizer, first.question, language.device
    )
    with torch.no_grad():
        question_embeddings = embedding_layer(question_ids).detach().float().expand(
            2, -1, -1
        )
    continuous_control, attention, attention_logits = _full_scene_control_components(
        control, scene_prefix.float(), question_embeddings
    )

    correct_parts: list[PrefixBatch] = []
    answer_ids: list[torch.Tensor] = []
    for index, record in enumerate(records):
        batch, ids = _compose_batch(
            runtime=runtime,
            scene_prefix=scene_prefix[index : index + 1],
            record=record,
            answer=record.answer,
            control_tokens=continuous_control[index : index + 1],
        )
        correct_parts.append(batch)
        answer_ids.append(ids)
    correct = stack_prefix_batches(
        correct_parts,
        language.device,
        prefix_backend=language.prefix_backend,
    )
    correct_output = forward_prefix_batch(language, correct)
    if correct.labels is None:
        raise RuntimeError("V57 correct pair batch lacks answer labels")
    correct_answer_nll = token_normalized_nll(correct_output.logits, correct.labels)

    used_single_forward_candidate_scoring = True
    try:
        correct_rank_nll, swapped_rank_nll = _aligned_candidate_nll(
            correct_output.logits,
            correct.labels,
            answer_ids[0],
            answer_ids[1],
        )
    except ValueError:
        used_single_forward_candidate_scoring = False
        first_mask, second_mask = differing_answer_token_masks(
            answer_ids[0], answer_ids[1]
        )
        correct_rank_labels = correct.labels.clone()
        restrict_labels_to_answer_mask(correct_rank_labels, 0, first_mask)
        restrict_labels_to_answer_mask(correct_rank_labels, 1, second_mask)
        correct_rank_nll = token_normalized_nll(
            correct_output.logits, correct_rank_labels
        )
        swapped_parts = [
            _compose_batch(
                runtime=runtime,
                scene_prefix=scene_prefix[index : index + 1],
                record=record,
                answer=records[1 - index].answer,
                control_tokens=continuous_control[index : index + 1],
            )[0]
            for index, record in enumerate(records)
        ]
        swapped = stack_prefix_batches(
            swapped_parts,
            language.device,
            prefix_backend=language.prefix_backend,
        )
        swapped_output = forward_prefix_batch(language, swapped)
        if swapped.labels is None:
            raise RuntimeError("V57 swapped pair batch lacks answer labels")
        swapped_rank_labels = swapped.labels.clone()
        restrict_labels_to_answer_mask(swapped_rank_labels, 0, second_mask)
        restrict_labels_to_answer_mask(swapped_rank_labels, 1, first_mask)
        swapped_rank_nll = token_normalized_nll(
            swapped_output.logits, swapped_rank_labels
        )

    side_hinge, side_margins, cross_hinge, cross_margins = (
        pair_and_cross_prefix_hinges(
            correct_rank_nll=correct_rank_nll,
            swapped_rank_nll=swapped_rank_nll,
            side_margin=settings.side_margin,
            cross_prefix_margin=settings.cross_prefix_margin,
        )
    )
    delta_hinge, relative_delta = control_delta_hinge(
        continuous_control,
        minimum_relative_delta=settings.minimum_relative_control_delta,
    )
    entropy_hinge, entropies = attention_entropy_hinge(
        attention,
        minimum_normalized_entropy=settings.minimum_normalized_attention_entropy,
    )
    logit_spread_penalty, logit_rms = attention_logit_spread_penalty(
        attention_logits,
        maximum_rms=settings.maximum_attention_logit_rms,
    )
    eos_token_id = getattr(language.tokenizer, "eos_token_id", None)
    answer_targets = torch.stack(
        tuple(
            _canonical_answer_target(
                embedding_layer,
                ids,
                eos_token_id=eos_token_id,
            )
            for ids in answer_ids
        )
    )
    alignment_hinge, alignment_margins, alignment_similarities = (
        answer_alignment_hinge(
            continuous_control,
            answer_targets,
            margin=settings.answer_alignment_margin,
        )
    )
    absolute_alignment_loss, own_answer_similarities = (
        answer_absolute_alignment_loss(alignment_similarities)
    )
    delta_alignment_loss, delta_alignment_cosine = answer_delta_alignment_loss(
        continuous_control,
        answer_targets,
    )
    correct_nll = correct_answer_nll.mean()
    total = (
        settings.answer_nll_weight * correct_nll
        + settings.side_hinge_weight * side_hinge
        + settings.cross_prefix_hinge_weight * cross_hinge
        + settings.control_delta_weight * delta_hinge
        + settings.attention_entropy_weight * entropy_hinge
        + settings.attention_logit_spread_weight * logit_spread_penalty
        + settings.answer_alignment_weight * alignment_hinge
        + settings.answer_absolute_alignment_weight * absolute_alignment_loss
        + settings.answer_delta_alignment_weight * delta_alignment_loss
    )
    finite = (
        total,
        correct_nll,
        side_hinge,
        cross_hinge,
        delta_hinge,
        entropy_hinge,
        logit_spread_penalty,
        alignment_hinge,
        absolute_alignment_loss,
        delta_alignment_loss,
        relative_delta,
    )
    if total.ndim != 0 or not all(torch.isfinite(value) for value in finite):
        raise RuntimeError("V57 pair objective is nonfinite or nonscalar")
    return total, {
        "correct_answer_nll": correct_answer_nll,
        "correct_ranking_nll": correct_rank_nll,
        "swapped_ranking_nll": swapped_rank_nll,
        "side_hinge": side_hinge,
        "side_margins": side_margins,
        "cross_prefix_hinge": cross_hinge,
        "cross_prefix_margins": cross_margins,
        "control_delta_hinge": delta_hinge,
        "relative_control_delta": relative_delta,
        "attention_entropy_hinge": entropy_hinge,
        "normalized_attention_entropy": entropies,
        "attention_logit_spread_penalty": logit_spread_penalty,
        "attention_logit_rms": logit_rms,
        "answer_alignment_hinge": alignment_hinge,
        "answer_alignment_margins": alignment_margins,
        "answer_alignment_similarities": alignment_similarities,
        "answer_absolute_alignment_loss": absolute_alignment_loss,
        "own_answer_similarities": own_answer_similarities,
        "answer_delta_alignment_loss": delta_alignment_loss,
        "answer_delta_alignment_cosine": delta_alignment_cosine,
        "single_forward_candidate_scoring": used_single_forward_candidate_scoring,
    }


__all__ = [
    "V57PairObjectiveSettings",
    "answer_absolute_alignment_loss",
    "answer_alignment_hinge",
    "answer_delta_alignment_loss",
    "attention_entropy_hinge",
    "attention_logit_spread_penalty",
    "control_delta_hinge",
    "full_scene_control_with_attention",
    "normalized_attention_entropy",
    "pair_and_cross_prefix_hinges",
    "paired_question_control_objective",
    "relative_control_delta",
]
