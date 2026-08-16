"""Training-only numeric answer prototypes for the V66 all-row adapter.

The authenticated V62 teacher cache covers 21 of the 28 answer classes.  V66
reuses a deterministic teacher medoid for every supported class.  The primary
experiment refuses a class without an eligible verified teacher.  A separate,
explicit ablation can instead create a deterministic prompt from the frozen
local language model's native input embeddings.  That fallback is never a
promotion gate.  Answer text and the codebook are never serialized into a
runtime checkpoint.

For pair-held-out cross-validation callers must pass only the other eleven
pairs.  The builder rejects any row or teacher from the forbidden pair, and a
held answer is vocabulary-supported only when the same canonical class occurs
outside that pair.
"""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F

from semantic_3d_chat.evaluation.metrics import normalize_answer
from semantic_3d_chat.training.soft_prompt_teacher_v62 import PROMPT_SHAPE
from semantic_3d_chat.training.train_question_control_v63 import V63Row


@dataclass(frozen=True)
class HybridAnswerPrototypeCodebookV66:
    prototypes: dict[str, torch.Tensor]
    targets: dict[tuple[str, str], torch.Tensor]
    class_by_key: dict[tuple[str, str], str]
    manifest: dict[str, Any]
    sha256: str


def _canonical_bytes(value: object) -> bytes:
    return (
        json.dumps(
            value,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("utf-8")


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _tensor_sha256(value: torch.Tensor) -> str:
    tensor = value.detach().cpu().float().contiguous()
    digest = hashlib.sha256()
    digest.update(str(tuple(tensor.shape)).encode())
    digest.update(str(tensor.dtype).encode())
    digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def answer_class_id_v66(answer: str) -> str:
    normalized = normalize_answer(answer)
    if not normalized:
        raise ValueError("V66 canonical training answer normalizes to empty")
    return f"answer_{hashlib.sha256(normalized.encode('utf-8')).hexdigest()[:20]}"


def _token_ids(tokenizer: Any, canonical_answer: str) -> torch.Tensor:
    encoded = tokenizer(
        canonical_answer,
        add_special_tokens=False,
        return_tensors="pt",
    )
    if isinstance(encoded, Mapping):
        ids = encoded.get("input_ids")
    else:
        ids = getattr(encoded, "input_ids", None)
    if not isinstance(ids, torch.Tensor) or ids.ndim != 2 or ids.shape[0] != 1:
        raise TypeError("V66 tokenizer must return input_ids with shape [1,T]")
    if ids.shape[1] < 1:
        raise ValueError("V66 canonical answer produced no language-model tokens")
    return ids


@torch.inference_mode()
def lm_native_answer_prototype(
    canonical_answer: str,
    *,
    tokenizer: Any,
    embedding_layer: torch.nn.Module,
    control_tokens: int = PROMPT_SHAPE[1],
    target_rms: float = 0.10,
) -> torch.Tensor:
    """Turn one training answer into a fixed native-width continuous prompt.

    Short answers are repeated cyclically to preserve their token directions;
    longer answers are deterministically average-pooled to the fixed prompt
    length.  Every output token is rescaled to exactly ``target_rms`` without
    changing direction.  No token ID or answer string is returned.
    """

    if not isinstance(canonical_answer, str) or not canonical_answer.strip():
        raise ValueError("V66 canonical answer must be nonempty")
    if isinstance(control_tokens, bool) or not isinstance(control_tokens, int) or control_tokens < 1:
        raise ValueError("V66 control_tokens must be a positive integer")
    if (
        isinstance(target_rms, bool)
        or not isinstance(target_rms, (int, float))
        or not 0.0 < float(target_rms) <= 1.0
    ):
        raise ValueError("V66 target_rms must be in (0,1]")
    normalized = normalize_answer(canonical_answer)
    if not normalized:
        raise ValueError("V66 canonical answer normalizes to empty")
    ids = _token_ids(tokenizer, normalized)
    weight = getattr(embedding_layer, "weight", None)
    device = weight.device if isinstance(weight, torch.Tensor) else torch.device("cpu")
    embedded = embedding_layer(ids.to(device)).detach().cpu().float()
    if embedded.ndim != 3 or embedded.shape[:2] != ids.shape:
        raise ValueError("V66 language-model answer embeddings have an invalid shape")
    sequence = embedded[0]
    if not torch.isfinite(sequence).all():
        raise ValueError("V66 language-model answer embeddings are nonfinite")
    if sequence.shape[0] <= control_tokens:
        indexes = torch.arange(control_tokens) % sequence.shape[0]
        prompt = sequence[indexes]
    else:
        prompt = F.adaptive_avg_pool1d(
            sequence.T.unsqueeze(0), control_tokens
        ).squeeze(0).T
    rms = prompt.square().mean(dim=-1, keepdim=True).sqrt()
    if torch.any(rms <= 1e-8):
        raise ValueError("V66 language-model answer prototype contains a zero token")
    prompt = prompt * (float(target_rms) / rms)
    expected_shape = (1, control_tokens, sequence.shape[-1])
    result = prompt.unsqueeze(0).contiguous()
    if tuple(result.shape) != expected_shape or not torch.isfinite(result).all():
        raise RuntimeError("V66 numeric answer prototype construction failed")
    return result


def _deterministic_teacher_medoid(
    members: Sequence[tuple[tuple[str, str], torch.Tensor]],
) -> tuple[tuple[str, str], torch.Tensor, dict[str, float]]:
    if not members:
        raise ValueError("V66 cannot select a teacher medoid from an empty class")
    ordered = sorted(members, key=lambda item: item[0])
    normalized: list[torch.Tensor] = []
    for _key, value in ordered:
        if tuple(value.shape) != PROMPT_SHAPE or not torch.isfinite(value).all():
            raise ValueError("V66 verified teacher shape or finiteness changed")
        flat = value.detach().cpu().float().flatten()
        if float(flat.square().sum()) <= 1e-12:
            raise ValueError("V66 verified teacher must be nonzero")
        normalized.append(F.normalize(flat, dim=0))
    stack = torch.stack(normalized)
    similarities = stack @ stack.T
    means = similarities.mean(dim=1)
    best = float(means.max())
    candidates = [
        index for index, score in enumerate(means.tolist()) if abs(score - best) <= 1e-8
    ]
    selected = min(candidates, key=lambda index: ordered[index][0])
    return (
        ordered[selected][0],
        ordered[selected][1].detach().cpu().float().clone(),
        {
            "selected_mean_flat_prompt_cosine": float(means[selected]),
            "class_mean_pairwise_flat_prompt_cosine": float(similarities.mean()),
            "class_minimum_pairwise_flat_prompt_cosine": float(similarities.min()),
        },
    )


def build_hybrid_answer_prototype_codebook_v66(
    rows: Sequence[V63Row],
    verified_teachers: Mapping[tuple[str, str], torch.Tensor],
    *,
    native_prototype_provider: Callable[[str], torch.Tensor] | None = None,
    allow_unverified_native_fallback: bool = False,
    expected_class_count: int | None = None,
    scope: str,
    forbidden_pair_id: str | None = None,
) -> HybridAnswerPrototypeCodebookV66:
    """Build all-row numeric targets without serializing answer strings."""

    if not rows:
        raise ValueError("V66 codebook requires training rows")
    if not scope or any(character.isspace() for character in scope):
        raise ValueError("V66 codebook scope must be one opaque token")
    if type(allow_unverified_native_fallback) is not bool:
        raise TypeError("V66 native-fallback flag must be boolean")
    if allow_unverified_native_fallback and native_prototype_provider is None:
        raise ValueError("V66 native-fallback mode requires a numeric provider")
    if forbidden_pair_id is not None and any(row.pair_id == forbidden_pair_id for row in rows):
        raise AssertionError("V66 held pair reached its fold-local codebook")
    by_key = {row.key: row for row in rows}
    if len(by_key) != len(rows):
        raise ValueError("V66 rows contain duplicate opaque keys")
    if not set(verified_teachers).issubset(by_key):
        raise AssertionError("V66 held or foreign teacher reached the codebook")
    # The original cache contains changed-side teachers, while the optional
    # V66 supplemental cache deliberately uses unchanged rows for the seven
    # classes absent from that selection.  Cache loaders authenticate those
    # sources before this pure codebook builder is called; here we enforce the
    # decisive fold boundary by requiring every source key to be in ``rows``.

    grouped_rows: defaultdict[str, list[V63Row]] = defaultdict(list)
    grouped_teachers: defaultdict[
        str, list[tuple[tuple[str, str], torch.Tensor]]
    ] = defaultdict(list)
    normalized_by_class: dict[str, str] = {}
    class_by_key: dict[tuple[str, str], str] = {}
    for row in rows:
        normalized = normalize_answer(row.answer)
        class_id = answer_class_id_v66(row.answer)
        previous = normalized_by_class.setdefault(class_id, normalized)
        if previous != normalized:
            raise RuntimeError("V66 answer-class digest collision")
        grouped_rows[class_id].append(row)
        class_by_key[row.key] = class_id
    for key, teacher in verified_teachers.items():
        grouped_teachers[class_by_key[key]].append((key, teacher))
    if expected_class_count is not None and len(grouped_rows) != expected_class_count:
        raise ValueError(
            f"V66 requires {expected_class_count} supported answer classes; "
            f"observed={len(grouped_rows)}"
        )

    prototypes: dict[str, torch.Tensor] = {}
    records: list[dict[str, Any]] = []
    for class_id in sorted(grouped_rows):
        class_rows = grouped_rows[class_id]
        teacher_members = grouped_teachers.get(class_id, [])
        if teacher_members:
            selected_key, prototype, diagnostics = _deterministic_teacher_medoid(
                teacher_members
            )
            source_mode = "verified_teacher_medoid"
            selected_identity: dict[str, Any] = {
                "selected_source_scene_id": selected_key[0],
                "selected_source_question_id": selected_key[1],
                **diagnostics,
            }
        else:
            if not allow_unverified_native_fallback:
                raise ValueError(
                    "V66 primary codebook requires a verified numeric teacher for "
                    f"every supported class; missing={class_id}"
                )
            assert native_prototype_provider is not None
            prototype = native_prototype_provider(normalized_by_class[class_id])
            source_mode = "frozen_lm_native_answer_embedding"
            selected_identity = {}
        if tuple(prototype.shape) != PROMPT_SHAPE or not torch.isfinite(prototype).all():
            raise ValueError("V66 numeric prototype shape or finiteness changed")
        prototypes[class_id] = prototype.detach().cpu().float().clone()
        records.append(
            {
                "answer_class_id": class_id,
                "source_mode": source_mode,
                "row_count": len(class_rows),
                "source_pair_ids": sorted({row.pair_id for row in class_rows}),
                "verified_teacher_count": len(teacher_members),
                "prototype_sha256": _tensor_sha256(prototypes[class_id]),
                "prototype_shape": list(prototypes[class_id].shape),
                **selected_identity,
            }
        )
    targets = {
        row.key: prototypes[class_by_key[row.key]]
        for row in rows
    }
    if len(targets) != len(rows) or any(
        _tensor_sha256(targets[row.key])
        != _tensor_sha256(prototypes[class_by_key[row.key]])
        for row in rows
    ):
        raise RuntimeError("V66 all-row target assignment changed a class prototype")

    manifest = {
        "schema_version": 1,
        "artifact": "v66_training_only_hybrid_answer_numeric_codebook_v1",
        "scope": scope,
        "fold_local": forbidden_pair_id is not None,
        "forbidden_pair_id": forbidden_pair_id,
        "forbidden_pair_absent": (
            forbidden_pair_id is None
            or all(row.pair_id != forbidden_pair_id for row in rows)
        ),
        "answer_class_count": len(records),
        "all_row_target_count": len(targets),
        "verified_teacher_prototype_count": sum(
            record["source_mode"] == "verified_teacher_medoid" for record in records
        ),
        "lm_native_prototype_count": sum(
            record["source_mode"] == "frozen_lm_native_answer_embedding"
            for record in records
        ),
        "prompt_shape": list(PROMPT_SHAPE),
        "answer_text_used_training_only": True,
        "unverified_native_fallback_enabled": allow_unverified_native_fallback,
        "answer_strings_serialized": False,
        "runtime_load_permitted": False,
        "environmental_text_runtime_inputs": [],
        "held_pair_rows_or_teachers_used": False,
        "records": records,
    }
    return HybridAnswerPrototypeCodebookV66(
        prototypes=prototypes,
        targets=targets,
        class_by_key=class_by_key,
        manifest=manifest,
        sha256=_canonical_sha256(manifest),
    )


__all__ = [
    "HybridAnswerPrototypeCodebookV66",
    "answer_class_id_v66",
    "build_hybrid_answer_prototype_codebook_v66",
    "lm_native_answer_prototype",
]
