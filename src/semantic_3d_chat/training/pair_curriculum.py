"""Training-only counterfactual pair construction, scheduling, and metrics.

This module consumes QA annotations exclusively inside the supervised training
process.  It is intentionally not imported by the chat runtime.  A pair unit
contains the exact same question asked about two scenes whose canonical answer
changes; the environment still reaches the language model only through each
scene's continuous prefix.
"""

from __future__ import annotations

import math
import random
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Literal, TypeVar

import torch
import torch.nn.functional as F

from semantic_3d_chat.data.dataset import QARecord

T = TypeVar("T")


@dataclass(frozen=True)
class CounterfactualPairUnit:
    """One changed fact represented by the same question in two scenes."""

    pair_id: str
    question_key: str
    reference: QARecord
    counterfactual: QARecord

    @property
    def records(self) -> tuple[QARecord, QARecord]:
        return self.reference, self.counterfactual

    @property
    def scene_ids(self) -> tuple[str, str]:
        return self.reference.scene_id, self.counterfactual.scene_id


@dataclass(frozen=True)
class CurriculumBatch:
    """One standard single-scene batch or one paired two-scene batch."""

    kind: Literal["standard", "pair"]
    records: tuple[QARecord, ...] = ()
    scene_id: str | None = None
    pair_id: str | None = None
    pair_units: tuple[CounterfactualPairUnit, ...] = ()

    def __post_init__(self) -> None:
        if self.kind == "standard":
            if self.scene_id is None or not self.records or self.pair_units:
                raise ValueError("A standard curriculum batch needs one scene and records")
        elif self.kind == "pair":
            if self.pair_id is None or not self.pair_units or self.records:
                raise ValueError("A pair curriculum batch needs a pair ID and pair units")
            if any(unit.pair_id != self.pair_id for unit in self.pair_units):
                raise ValueError("A pair curriculum batch cannot mix scene pairs")
        else:  # pragma: no cover - Literal protects typed callers
            raise ValueError(f"Unknown curriculum batch kind: {self.kind}")


@dataclass(frozen=True)
class PairCurriculumSettings:
    """Validated counterfactual curriculum and candidate-gate configuration."""

    ranking_weight: float
    ranking_margin: float
    ranking_mode: Literal["nll", "candidate_logit"]
    batch_fraction: float
    units_per_batch: int
    steps_per_epoch: int | None
    pair_only: bool
    pair_only_scene_ids: tuple[str, ...]
    gate_enabled: bool
    gate_every_epochs: int
    stop_when_gate_passes: bool
    first_answer_token_top1_accuracy_threshold: float | None
    changed_unit_accuracy_threshold: float
    prediction_flip_threshold: float
    wrong_prefix_flip_threshold: float

    @property
    def enabled(self) -> bool:
        return self.ranking_weight > 0 or self.pair_only


def pair_curriculum_settings(config: Mapping[str, object]) -> PairCurriculumSettings:
    """Resolve opt-in settings while preserving old zero-ranking configurations."""

    raw_training = config.get("training")
    if not isinstance(raw_training, Mapping):
        raise TypeError("Config is missing a training mapping")
    ranking_weight = float(raw_training.get("pair_ranking_weight", 0.0))
    pair_only = bool(raw_training.get("pair_only_mode", False))
    enabled = ranking_weight > 0 or pair_only
    default_fraction = 1.0 if pair_only else (0.5 if enabled else 0.0)
    batch_size = int(raw_training.get("batch_size", 1))
    scene_ids_value = raw_training.get("pair_only_scene_ids", ())
    if isinstance(scene_ids_value, str) or not isinstance(scene_ids_value, Sequence):
        raise TypeError("pair_only_scene_ids must be a sequence of opaque scene IDs")
    steps_value = raw_training.get("pair_steps_per_epoch")
    first_token_threshold_value = raw_training.get("pair_gate_first_answer_token_top1_accuracy")
    ranking_mode = str(raw_training.get("pair_ranking_mode", "nll"))
    if ranking_mode not in {"nll", "candidate_logit"}:
        raise ValueError("pair_ranking_mode must be 'nll' or 'candidate_logit'")
    settings = PairCurriculumSettings(
        ranking_weight=ranking_weight,
        ranking_margin=float(raw_training.get("pair_ranking_margin", 0.5)),
        ranking_mode=ranking_mode,
        batch_fraction=float(raw_training.get("pair_batch_fraction", default_fraction)),
        units_per_batch=int(raw_training.get("pair_units_per_batch", max(1, batch_size // 2))),
        steps_per_epoch=None if steps_value is None else int(steps_value),
        pair_only=pair_only,
        pair_only_scene_ids=tuple(str(value) for value in scene_ids_value),
        gate_enabled=bool(raw_training.get("pair_gate_enabled", pair_only)),
        gate_every_epochs=int(raw_training.get("pair_gate_every_epochs", 1)),
        stop_when_gate_passes=bool(raw_training.get("pair_gate_stop_when_passed", pair_only)),
        first_answer_token_top1_accuracy_threshold=(
            None if first_token_threshold_value is None else float(first_token_threshold_value)
        ),
        changed_unit_accuracy_threshold=float(
            raw_training.get("pair_gate_changed_unit_accuracy", 0.95)
        ),
        prediction_flip_threshold=float(raw_training.get("pair_gate_prediction_flip_rate", 1.0)),
        wrong_prefix_flip_threshold=float(
            raw_training.get("pair_gate_wrong_prefix_flip_rate", 1.0)
        ),
    )
    if settings.ranking_weight < 0:
        raise ValueError("pair_ranking_weight cannot be negative")
    if settings.ranking_margin < 0:
        raise ValueError("pair_ranking_margin cannot be negative")
    if settings.enabled and settings.ranking_weight == 0:
        raise ValueError("Pair-only mode requires a positive pair_ranking_weight")
    if settings.enabled and not 0.5 <= settings.batch_fraction <= 1.0:
        raise ValueError("An enabled pair curriculum requires pair_batch_fraction >= 0.5")
    if not settings.enabled and settings.batch_fraction != 0.0:
        raise ValueError("pair_batch_fraction requires pair_ranking_weight > 0")
    if settings.units_per_batch < 1:
        raise ValueError("pair_units_per_batch must be positive")
    if settings.steps_per_epoch is not None and settings.steps_per_epoch < 1:
        raise ValueError("pair_steps_per_epoch must be positive")
    if settings.pair_only and len(settings.pair_only_scene_ids) < 2:
        raise ValueError("Pair-only mode requires at least two pair_only_scene_ids")
    if settings.gate_enabled and not settings.enabled:
        raise ValueError("pair_gate_enabled requires an enabled pair curriculum")
    if settings.gate_every_epochs < 1:
        raise ValueError("pair_gate_every_epochs must be positive")
    for name, value in (
        ("pair_gate_changed_unit_accuracy", settings.changed_unit_accuracy_threshold),
        ("pair_gate_prediction_flip_rate", settings.prediction_flip_threshold),
        ("pair_gate_wrong_prefix_flip_rate", settings.wrong_prefix_flip_threshold),
    ):
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"{name} must be in [0, 1]")
    if (
        settings.first_answer_token_top1_accuracy_threshold is not None
        and not 0.0 <= settings.first_answer_token_top1_accuracy_threshold <= 1.0
    ):
        raise ValueError("pair_gate_first_answer_token_top1_accuracy must be in [0, 1]")
    if (
        settings.first_answer_token_top1_accuracy_threshold is not None
        and not settings.gate_enabled
    ):
        raise ValueError("pair_gate_first_answer_token_top1_accuracy requires pair_gate_enabled")
    return settings


def build_exact_question_pair_units(
    records: Sequence[QARecord],
) -> list[CounterfactualPairUnit]:
    """Build complete changed-answer units strictly from the supplied records.

    Incomplete annotations are rejected instead of looking in another dataset
    split.  Questions must be byte-for-byte equal, and changed units must have
    distinct answers so that answer swapping defines a meaningful ranking task.
    """

    grouped: defaultdict[tuple[str, str], list[QARecord]] = defaultdict(list)
    pair_scenes: defaultdict[str, set[str]] = defaultdict(set)
    for record in records:
        if record.counterfactual_expected_change is not True:
            continue
        if not record.counterfactual_pair_id or not record.counterfactual_question_key:
            raise ValueError(
                f"Changed record {record.question_id} is missing counterfactual pair metadata"
            )
        grouped[(record.counterfactual_pair_id, record.counterfactual_question_key)].append(record)
        pair_scenes[record.counterfactual_pair_id].add(record.scene_id)

    for pair_id, scene_ids in pair_scenes.items():
        if len(scene_ids) != 2:
            raise ValueError(
                f"Counterfactual pair {pair_id} is incomplete in the supplied training records"
            )

    units: list[CounterfactualPairUnit] = []
    for (pair_id, question_key), members in sorted(grouped.items()):
        by_role = {record.counterfactual_role: record for record in members}
        if len(members) != 2 or set(by_role) != {"reference", "counterfactual"}:
            raise ValueError(
                "Each changed pair unit must contain exactly one reference and one "
                f"counterfactual record: {(pair_id, question_key)}"
            )
        reference = by_role["reference"]
        counterfactual = by_role["counterfactual"]
        if reference.scene_id == counterfactual.scene_id:
            raise ValueError(f"Pair unit {(pair_id, question_key)} repeats one scene")
        if reference.question != counterfactual.question:
            raise ValueError(
                f"Pair unit {(pair_id, question_key)} does not use the exact same question"
            )
        if reference.answer.strip() == counterfactual.answer.strip():
            raise ValueError(f"Changed pair unit {(pair_id, question_key)} has identical answers")
        units.append(
            CounterfactualPairUnit(
                pair_id=pair_id,
                question_key=question_key,
                reference=reference,
                counterfactual=counterfactual,
            )
        )
    return units


def select_pair_only_records(
    records: Sequence[QARecord], scene_ids: Sequence[str]
) -> list[QARecord]:
    """Select complete changed units whose two members are in ``scene_ids``."""

    allowed = set(scene_ids)
    if not allowed:
        raise ValueError("pair_only_scene_ids cannot be empty")
    unknown = allowed - {record.scene_id for record in records}
    if unknown:
        raise ValueError(f"pair_only_scene_ids are absent from training: {sorted(unknown)}")
    candidate_records = [record for record in records if record.scene_id in allowed]
    units = build_exact_question_pair_units(candidate_records)
    if not units:
        raise ValueError("Pair-only selection contains no changed counterfactual units")
    selected_ids = {
        (record.scene_id, record.question_id) for unit in units for record in unit.records
    }
    return [
        record
        for record in candidate_records
        if (record.scene_id, record.question_id) in selected_ids
    ]


def _cycled_sample(items: Sequence[T], count: int, rng: random.Random) -> list[T]:
    if count <= 0:
        return []
    if not items:
        raise ValueError("Cannot sample from an empty curriculum")
    result: list[T] = []
    while len(result) < count:
        cycle = list(items)
        rng.shuffle(cycle)
        result.extend(cycle[: count - len(result)])
    return result


def build_epoch_curriculum(
    records_by_scene: Mapping[str, Sequence[QARecord]],
    pair_units: Sequence[CounterfactualPairUnit],
    *,
    standard_batch_size: int,
    pair_units_per_batch: int,
    pair_batch_fraction: float,
    pair_only: bool,
    seed: int,
    steps_per_epoch: int | None = None,
) -> list[CurriculumBatch]:
    """Create a deterministic interleaved curriculum with paired scene forwards.

    Pair units are grouped by pair ID, so a pair step requires exactly two scene
    encoder forwards regardless of how many units it contains.  The scheduler
    samples batches directly; it never traverses all questions for scene A and
    then all questions for scene B to construct a counterfactual update.
    """

    if standard_batch_size < 1 or pair_units_per_batch < 1:
        raise ValueError("Curriculum batch sizes must be positive")
    if not 0.0 <= pair_batch_fraction <= 1.0:
        raise ValueError("pair_batch_fraction must be in [0, 1]")
    if pair_only and pair_batch_fraction != 1.0:
        raise ValueError("Pair-only mode requires pair_batch_fraction=1.0")
    if pair_batch_fraction > 0 and pair_batch_fraction < 0.5:
        raise ValueError("An enabled pair curriculum requires pair_batch_fraction >= 0.5")
    if steps_per_epoch is not None and steps_per_epoch < 1:
        raise ValueError("pair_steps_per_epoch must be positive")

    rng = random.Random(seed)
    standard_batches: list[CurriculumBatch] = []
    if not pair_only:
        for scene_id in sorted(records_by_scene):
            scene_records = list(records_by_scene[scene_id])
            rng.shuffle(scene_records)
            standard_batches.extend(
                CurriculumBatch(
                    kind="standard",
                    scene_id=scene_id,
                    records=tuple(scene_records[offset : offset + standard_batch_size]),
                )
                for offset in range(0, len(scene_records), standard_batch_size)
            )

    units_by_pair: defaultdict[str, list[CounterfactualPairUnit]] = defaultdict(list)
    for unit in pair_units:
        units_by_pair[unit.pair_id].append(unit)
    pair_batches: list[CurriculumBatch] = []
    for pair_id in sorted(units_by_pair):
        units = list(units_by_pair[pair_id])
        rng.shuffle(units)
        pair_batches.extend(
            CurriculumBatch(
                kind="pair",
                pair_id=pair_id,
                pair_units=tuple(units[offset : offset + pair_units_per_batch]),
            )
            for offset in range(0, len(units), pair_units_per_batch)
        )

    if pair_batch_fraction == 0.0:
        rng.shuffle(standard_batches)
        return standard_batches
    if not pair_batches:
        raise ValueError("The pair curriculum is enabled but no complete pair units exist")

    if pair_only:
        target_steps = steps_per_epoch or len(pair_batches)
        return _cycled_sample(pair_batches, target_steps, rng)

    target_steps = steps_per_epoch or max(len(standard_batches), len(pair_batches))
    pair_count = math.ceil(target_steps * pair_batch_fraction)
    standard_count = target_steps - pair_count
    selected = [
        *_cycled_sample(pair_batches, pair_count, rng),
        *_cycled_sample(standard_batches, standard_count, rng),
    ]
    rng.shuffle(selected)
    return selected


def token_normalized_nll(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Return one causal next-token NLL per sequence, normalized by answer length."""

    if logits.ndim != 3 or labels.ndim != 2 or logits.shape[:2] != labels.shape:
        raise ValueError("Expected logits [B,L,V] and aligned labels [B,L]")
    shift_logits = logits[:, :-1].float()
    shift_labels = labels[:, 1:]
    valid = shift_labels.ne(-100)
    token_counts = valid.sum(dim=1)
    if torch.any(token_counts == 0):
        raise ValueError("Every ranking sequence must contain at least one answer token")
    losses = F.cross_entropy(
        shift_logits.reshape(-1, shift_logits.shape[-1]),
        shift_labels.reshape(-1),
        ignore_index=-100,
        reduction="none",
    ).reshape_as(shift_labels)
    return (losses * valid).sum(dim=1) / token_counts


def differing_answer_token_masks(
    first_answer_ids: torch.Tensor,
    second_answer_ids: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Select answer-token positions that discriminate two candidates.

    Canonical answer sequences commonly end in the same EOS token. Including
    that shared suffix in a two-token candidate halves the scene-conditioned
    log-odds margin and can leave a balanced pair at the hinge's cancellation
    saddle. Full answer NLL remains the language objective; only the pairwise
    ranking term excludes tokens that are identical at the same answer offset.
    """

    first = first_answer_ids.reshape(-1)
    second = second_answer_ids.reshape(-1)
    if first.numel() == 0 or second.numel() == 0:
        raise ValueError("Candidate answers cannot be empty")
    first_mask = torch.ones(first.shape, dtype=torch.bool, device=first.device)
    second_mask = torch.ones(second.shape, dtype=torch.bool, device=second.device)
    common_length = min(first.numel(), second.numel())
    equal_positions = first[:common_length].eq(second[:common_length].to(first.device))
    first_mask[:common_length] = ~equal_positions
    second_mask[:common_length] = ~equal_positions.to(second.device)
    if not first_mask.any() and not second_mask.any():
        raise ValueError("Counterfactual answer candidates have no differing tokens")
    # If one tokenization is a strict prefix of the other, retain its terminal
    # token so both candidate NLLs remain defined and length-normalized.
    if not first_mask.any():
        first_mask[-1] = True
    if not second_mask.any():
        second_mask[-1] = True
    return first_mask, second_mask


def single_differing_answer_token(
    first_answer_ids: torch.Tensor,
    second_answer_ids: torch.Tensor,
) -> tuple[int, int, int]:
    """Validate and locate an exact one-token counterfactual answer change.

    Candidate-logit ranking compares both candidates in the *same* next-token
    distribution.  That is valid only when the two tokenized answers have the
    same length and differ at exactly one aligned offset.  All tokens before
    that offset are consequently identical, which proves that the candidate
    token is predicted from the same textual context on either answer path.
    """

    first = first_answer_ids.reshape(-1)
    second = second_answer_ids.reshape(-1)
    if first.numel() == 0 or second.numel() == 0:
        raise ValueError("Candidate answers cannot be empty")
    if first.numel() != second.numel():
        raise ValueError(
            "candidate_logit ranking requires equal-length tokenized answers; "
            f"received {first.numel()} and {second.numel()} tokens"
        )
    differing = first.ne(second.to(first.device)).nonzero(as_tuple=False).flatten()
    if differing.numel() != 1:
        raise ValueError(
            "candidate_logit ranking requires exactly one differing answer token "
            f"at an aligned position; found {differing.numel()}"
        )
    offset = int(differing.item())
    return offset, int(first[offset].item()), int(second[offset].item())


def restrict_labels_to_answer_mask(
    labels: torch.Tensor,
    row: int,
    answer_mask: torch.Tensor,
) -> None:
    """Mask shared answer tokens in one padded teacher-forcing label row."""

    supervised = labels[row].ne(-100).nonzero(as_tuple=False).flatten()
    if supervised.numel() != answer_mask.numel():
        raise ValueError(
            "Answer mask length does not match the supervised token count: "
            f"{answer_mask.numel()} != {supervised.numel()}"
        )
    shared_positions = supervised[~answer_mask.to(device=labels.device)]
    labels[row, shared_positions] = -100


def pair_ranking_hinge(
    correct_nll: torch.Tensor,
    swapped_nll: torch.Tensor,
    *,
    margin: float = 0.5,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Penalize a swapped answer unless its NLL exceeds the correct NLL."""

    if correct_nll.shape != swapped_nll.shape:
        raise ValueError("Correct and swapped NLL tensors must have the same shape")
    if correct_nll.numel() == 0:
        raise ValueError("Pair ranking tensors cannot be empty")
    if not torch.isfinite(correct_nll).all() or not torch.isfinite(swapped_nll).all():
        raise ValueError("Pair ranking NLL contains NaN or infinity")
    if margin < 0:
        raise ValueError("Pair ranking margin cannot be negative")
    correct_vs_swapped_margin = swapped_nll - correct_nll
    return ranking_margin_hinge(correct_vs_swapped_margin, margin=margin)


def ranking_margin_hinge(
    margins: torch.Tensor,
    *,
    margin: float = 0.5,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply the shared pair hinge to a positive-is-correct score margin."""

    if margins.numel() == 0:
        raise ValueError("Pair ranking margins cannot be empty")
    if not torch.isfinite(margins).all():
        raise ValueError("Pair ranking margin contains NaN or infinity")
    if margin < 0:
        raise ValueError("Pair ranking margin cannot be negative")
    return F.relu(float(margin) - margins).mean(), margins


def candidate_logit_margins(
    logits: torch.Tensor,
    labels: torch.Tensor,
    candidate_specs: Sequence[tuple[int, int, int]],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Score own and alternate tokens from each correct forward distribution.

    Each candidate spec is ``(answer_offset, own_token_id, alternate_token_id)``.
    The logit row immediately before that answer label is the common next-token
    distribution.  No swapped-answer sequence and no cross-entropy subtraction
    is involved.
    """

    if logits.ndim != 3 or labels.ndim != 2 or logits.shape[:2] != labels.shape:
        raise ValueError("Expected logits [B,L,V] and aligned labels [B,L]")
    if len(candidate_specs) != logits.shape[0]:
        raise ValueError(
            "Candidate specification count must match the logit batch: "
            f"{len(candidate_specs)} != {logits.shape[0]}"
        )
    own_scores: list[torch.Tensor] = []
    alternate_scores: list[torch.Tensor] = []
    vocabulary_size = logits.shape[-1]
    for row, (answer_offset, own_token_id, alternate_token_id) in enumerate(candidate_specs):
        supervised = labels[row].ne(-100).nonzero(as_tuple=False).flatten()
        if not 0 <= answer_offset < supervised.numel():
            raise ValueError(
                f"Candidate answer offset {answer_offset} is outside row {row}'s "
                f"{supervised.numel()} supervised answer tokens"
            )
        target_position = int(supervised[answer_offset].item())
        if target_position == 0:
            raise ValueError("A candidate answer token requires a preceding causal position")
        if int(labels[row, target_position].item()) != own_token_id:
            raise ValueError(
                f"Candidate own token {own_token_id} does not match row {row}'s label "
                f"{int(labels[row, target_position].item())}"
            )
        for token_id in (own_token_id, alternate_token_id):
            if not 0 <= token_id < vocabulary_size:
                raise ValueError(f"Candidate token ID {token_id} is outside the vocabulary")
        distribution = logits[row, target_position - 1].float()
        own_scores.append(distribution[own_token_id])
        alternate_scores.append(distribution[alternate_token_id])
    own = torch.stack(own_scores)
    alternate = torch.stack(alternate_scores)
    margins = own - alternate
    if not torch.isfinite(margins).all():
        raise ValueError("Candidate-logit margin contains NaN or infinity")
    return margins, own, alternate


def first_answer_token_full_vocab_margins(
    logits: torch.Tensor,
    labels: torch.Tensor,
) -> torch.Tensor:
    """Compare each target's first answer token with the full vocabulary.

    The target and strongest non-target scores come from the same causal
    next-token distribution already used by teacher forcing. A strictly
    positive margin means the target is the unique top-1 token; ties fail the
    deterministic gate. No additional model forward is required.
    """

    if logits.ndim != 3 or labels.ndim != 2 or logits.shape[:2] != labels.shape:
        raise ValueError("Expected logits [B,L,V] and aligned labels [B,L]")
    if logits.shape[0] == 0:
        raise ValueError("Full-vocabulary diagnostics require at least one row")
    if logits.shape[-1] < 2:
        raise ValueError("Full-vocabulary diagnostics require at least two tokens")
    margins: list[torch.Tensor] = []
    for row in range(logits.shape[0]):
        supervised = labels[row].ne(-100).nonzero(as_tuple=False).flatten()
        if supervised.numel() == 0:
            raise ValueError(f"Row {row} has no supervised answer token")
        target_position = int(supervised[0].item())
        if target_position == 0:
            raise ValueError("A first answer token requires a preceding causal position")
        target_token_id = int(labels[row, target_position].item())
        vocabulary_size = logits.shape[-1]
        if not 0 <= target_token_id < vocabulary_size:
            raise ValueError(f"Target token ID {target_token_id} is outside the vocabulary")
        distribution = logits[row, target_position - 1].float()
        if not torch.isfinite(distribution).all():
            raise ValueError("Full-vocabulary answer logits contain NaN or infinity")
        non_target = distribution.clone()
        non_target[target_token_id] = -torch.inf
        margins.append(distribution[target_token_id] - non_target.max())
    return torch.stack(margins)


def pair_gate_metrics(
    margins: torch.Tensor | Sequence[Sequence[float]],
    *,
    changed_unit_accuracy_threshold: float = 0.95,
    prediction_flip_threshold: float = 1.0,
    wrong_prefix_flip_threshold: float = 1.0,
    ranking_margin: float = 0.5,
    ranking_mode: Literal["nll", "candidate_logit"] = "nll",
    first_answer_token_full_vocab_margins: (torch.Tensor | Sequence[Sequence[float]] | None) = None,
    first_answer_token_top1_accuracy_threshold: float | None = None,
) -> dict[str, float | int | bool | str | None]:
    """Score a two-answer candidate gate from per-unit, per-prefix margins.

    A positive margin means that a scene prefix prefers its own answer.  In NLL
    mode this is swapped minus correct answer NLL. In candidate-logit mode this
    is own minus alternate token logit from one shared next-token distribution.
    The deterministic gate is useful during overfit training; it is explicitly
    not a substitute for the separate free-generation evaluation.
    """

    if ranking_mode not in {"nll", "candidate_logit"}:
        raise ValueError("ranking_mode must be 'nll' or 'candidate_logit'")
    values = torch.as_tensor(margins, dtype=torch.float32)
    if values.ndim != 2 or values.shape[1] != 2 or values.shape[0] == 0:
        raise ValueError("Pair gate margins must have shape [unit_count, 2]")
    if not torch.isfinite(values).all():
        raise ValueError("Pair gate margins contain NaN or infinity")
    if not 0.0 <= changed_unit_accuracy_threshold <= 1.0:
        raise ValueError("changed_unit_accuracy_threshold must be in [0, 1]")
    if not 0.0 <= prediction_flip_threshold <= 1.0:
        raise ValueError("prediction_flip_threshold must be in [0, 1]")
    if not 0.0 <= wrong_prefix_flip_threshold <= 1.0:
        raise ValueError("wrong_prefix_flip_threshold must be in [0, 1]")
    if (
        first_answer_token_top1_accuracy_threshold is not None
        and not 0.0 <= first_answer_token_top1_accuracy_threshold <= 1.0
    ):
        raise ValueError("first_answer_token_top1_accuracy_threshold must be in [0, 1]")

    own_answer_selected = values > 0
    unit_correct = own_answer_selected.all(dim=1)
    # With exactly two candidates, the two prefixes choose different answers
    # when their own-answer decisions agree (both correct or both swapped).
    prediction_flips = own_answer_selected[:, 0].eq(own_answer_selected[:, 1])
    # Asking the identical question with the partner prefix should select the
    # partner answer.  Both directions follow the injected (wrong) prefix only
    # when both scene-specific rankings are correct.
    wrong_prefix_follows = unit_correct
    changed_unit_accuracy = float(unit_correct.float().mean())
    prediction_flip_rate = float(prediction_flips.float().mean())
    wrong_prefix_flip_rate = float(wrong_prefix_follows.float().mean())
    ranking_hinge = float(F.relu(float(ranking_margin) - values).mean())
    pairwise_passed = (
        changed_unit_accuracy >= changed_unit_accuracy_threshold
        and prediction_flip_rate >= prediction_flip_threshold
        and wrong_prefix_flip_rate >= wrong_prefix_flip_threshold
    )
    full_vocab_values: torch.Tensor | None = None
    if first_answer_token_full_vocab_margins is not None:
        full_vocab_values = torch.as_tensor(
            first_answer_token_full_vocab_margins, dtype=torch.float32
        )
        if full_vocab_values.shape != values.shape:
            raise ValueError(
                "First-answer-token full-vocabulary margins must match pair gate "
                f"shape {tuple(values.shape)}"
            )
        if not torch.isfinite(full_vocab_values).all():
            raise ValueError("First-answer-token full-vocabulary margins contain NaN or infinity")
    if first_answer_token_top1_accuracy_threshold is not None and full_vocab_values is None:
        raise ValueError("A first-answer-token top-1 threshold requires full-vocabulary margins")
    full_vocab_accuracy = (
        None if full_vocab_values is None else float(full_vocab_values.gt(0).float().mean())
    )
    full_vocab_gate_passed = (
        None
        if first_answer_token_top1_accuracy_threshold is None
        else bool(full_vocab_accuracy >= first_answer_token_top1_accuracy_threshold)
    )
    full_vocab_requirement_satisfied = first_answer_token_top1_accuracy_threshold is None or bool(
        full_vocab_gate_passed
    )
    passed = pairwise_passed and full_vocab_requirement_satisfied
    result: dict[str, float | int | bool | str | None] = {
        "evaluation_type": (
            "teacher_forced_same_distribution_candidate_logit_ranking"
            if ranking_mode == "candidate_logit"
            else "teacher_forced_discriminative_answer_candidate_ranking"
        ),
        "ranking_mode": ranking_mode,
        "free_generation_evaluated": False,
        "shared_candidate_tokens_excluded": True,
        "same_next_token_distribution": ranking_mode == "candidate_logit",
        "unit_count": int(values.shape[0]),
        "side_count": int(values.numel()),
        "changed_unit_accuracy": changed_unit_accuracy,
        "side_accuracy": float(own_answer_selected.float().mean()),
        "prediction_flip_rate": prediction_flip_rate,
        "wrong_prefix_flip_rate": wrong_prefix_flip_rate,
        "pairwise_passed": pairwise_passed,
        "first_answer_token_full_vocab_evaluated": full_vocab_values is not None,
        "first_answer_token_top1_gate_enabled": (
            first_answer_token_top1_accuracy_threshold is not None
        ),
        "first_answer_token_top1_accuracy_threshold": (first_answer_token_top1_accuracy_threshold),
        "first_answer_token_top1_gate_passed": full_vocab_gate_passed,
        "mean_ranking_margin": float(values.mean()),
        "minimum_ranking_margin": float(values.min()),
        "ranking_hinge_at_configured_margin": ranking_hinge,
        "changed_unit_accuracy_threshold": changed_unit_accuracy_threshold,
        "prediction_flip_threshold": prediction_flip_threshold,
        "wrong_prefix_flip_threshold": wrong_prefix_flip_threshold,
        "passed": passed,
    }
    if full_vocab_values is not None:
        full_vocab_top1 = full_vocab_values.gt(0)
        result.update(
            {
                "first_answer_token_top1_accuracy": float(full_vocab_top1.float().mean()),
                "first_answer_token_top1_unit_accuracy": float(
                    full_vocab_top1.all(dim=1).float().mean()
                ),
                "mean_first_answer_token_target_vs_best_other_logit_margin": float(
                    full_vocab_values.mean()
                ),
                "minimum_first_answer_token_target_vs_best_other_logit_margin": float(
                    full_vocab_values.min()
                ),
                "first_answer_token_target_vs_best_other_hinge": float(
                    F.relu(-full_vocab_values).mean()
                ),
            }
        )
    if ranking_mode == "candidate_logit":
        result["mean_own_vs_alternate_candidate_logit_margin"] = float(values.mean())
        result["minimum_own_vs_alternate_candidate_logit_margin"] = float(values.min())
    else:
        result["mean_correct_vs_swapped_nll_margin"] = float(values.mean())
        result["minimum_correct_vs_swapped_nll_margin"] = float(values.min())
    return result
