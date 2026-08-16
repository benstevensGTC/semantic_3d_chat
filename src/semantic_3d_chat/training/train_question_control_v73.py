"""Train-only V73 full-scene numeric causality screen.

The command has no language-model generation, checkpoint publication, atlas,
validation, test, deferred-final, or oracle path.  It uses only the 40-scene
training pool, frozen question/token embeddings from the pinned local Gemma 4
snapshot, immutable pre-question V54 prefixes, and LM-native numeric answer
prototypes.  The decisive split is pair-ID-disjoint: twelve historical pairs
fit the readers, and eight later replicated-family pairs are opened only after
optimization for a single fixed numeric comparison.

Two readers receive the exact same training protocol:

* V73: two positive-floor attention hops over every one of 256 scene latents.
* DCT40: the same stack after the duplicated first-8 + first-32 DCT bottleneck
  used to diagnose the V71 information bottleneck.

No model output is a runtime artifact.  A later command may be designed only
if this numeric-only screen passes its locked causal gates.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import signal
import tempfile
import time
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Final

import torch
import torch.nn.functional as F
import yaml
from safetensors import safe_open
from safetensors.torch import load_file
from torch import nn
from transformers import AutoTokenizer

from semantic_3d_chat.config import PROJECT_ROOT
from semantic_3d_chat.evaluation.metrics import normalize_answer
from semantic_3d_chat.scene_encoder.question_control_v73 import (
    DCT40QuestionControlBaselineV73,
    FullSceneSetAttentionQuestionControlV73,
)


EXPECTED_ROWS: Final[int] = 960
EXPECTED_SCENES: Final[int] = 40
EXPECTED_PAIRS: Final[int] = 20
EXPECTED_PREFIX_SHAPE: Final[tuple[int, int, int]] = (1, 258, 1536)
EXPECTED_HIDDEN_SIZE: Final[int] = 1536
EXPECTED_CONTROL_TOKENS: Final[int] = 4

TRAIN_PAIR_IDS: Final[tuple[str, ...]] = (
    "pair_000005",
    "pair_000006",
    "pair_000007",
    "pair_000008",
    "pair_000009",
    "pair_000010",
    "pair_000011",
    "pair_000015",
    "pair_000016",
    "pair_000017",
    "pair_000018",
    "pair_000026",
)
HELD_PAIR_IDS: Final[tuple[str, ...]] = (
    "pair_000019",
    "pair_000020",
    "pair_000021",
    "pair_000022",
    "pair_000023",
    "pair_000024",
    "pair_000025",
    "pair_000027",
)
EXPECTED_TRAIN_ROWS: Final[int] = 576
EXPECTED_HELD_ROWS: Final[int] = 384
EXPECTED_TRAIN_SCENES: Final[int] = 24
EXPECTED_HELD_SCENES: Final[int] = 16
EXPECTED_HELD_CHANGED_SIDES: Final[int] = 52
EXPECTED_HELD_CHANGED_UNITS: Final[int] = 26


@dataclass(frozen=True)
class ScreenGatesV73:
    causal_margin_gain: float = 0.03
    causal_margin_gain_bootstrap_lower_bound: float = 0.0
    additional_positive_sides: int = 8
    additional_complete_units: int = 6
    prediction_change_units: int = 13
    improved_families: int = 6
    minimum_family_margin_gain: float = -0.02
    maximum_broad_accuracy_drop: float = 0.02
    correct_over_wrong_scene_margin: float = 0.02
    zero_scene_maximum_absolute_control: float = 0.0


LOCKED_GATES: Final[ScreenGatesV73] = ScreenGatesV73()


@dataclass(frozen=True)
class AbsoluteReaderGatesV73:
    supported_accuracy: float = 0.80
    changed_supported_accuracy: float = 0.65
    complete_class_units: int = 8
    prediction_change_units: int = 13
    positive_own_over_opposite_sides: int = 34
    mean_own_over_opposite_margin: float = 0.20
    correct_over_wrong_scene_margin: float = 0.02
    zero_scene_maximum_absolute_control: float = 0.0


LOCKED_ABSOLUTE_GATES: Final[AbsoluteReaderGatesV73] = AbsoluteReaderGatesV73()


@dataclass(frozen=True)
class RowV73:
    scene_id: str
    question_id: str
    question: str
    answer: str
    answer_class: str
    answer_type: str
    pair_id: str
    paired_scene_id: str
    question_key: str
    change_type: str
    expected_change: bool

    @property
    def key(self) -> tuple[str, str]:
        return self.scene_id, self.question_id


@dataclass(frozen=True)
class ChangedUnitV73:
    pair_id: str
    question_key: str
    change_type: str
    left: RowV73
    right: RowV73


@dataclass(frozen=True)
class EmbeddingAssetsV73:
    questions: dict[str, torch.Tensor]
    answers: dict[str, torch.Tensor]
    tokenizer_files: tuple[str, ...]
    model_file: str
    model_file_sha256: str
    embedding_tensor_name: str
    embedding_shape: tuple[int, int]
    embedding_dtype: str


@dataclass(frozen=True)
class PrototypeBankV73:
    class_ids: tuple[str, ...]
    prototypes: torch.Tensor
    class_index: dict[str, int]
    output_basis: torch.Tensor


class WallTimeExceededV73(RuntimeError):
    """Raised when the train-only numeric screen exceeds its locked budget."""


class _WallTimerV73:
    def __init__(self, seconds: int) -> None:
        self.seconds = seconds
        self.previous: Any = None

    @staticmethod
    def _raise(_signal: int, _frame: Any) -> None:
        raise WallTimeExceededV73("V73 locked wall-time budget exceeded")

    def __enter__(self) -> "_WallTimerV73":
        if self.seconds > 0 and hasattr(signal, "SIGALRM"):
            self.previous = signal.signal(signal.SIGALRM, self._raise)
            signal.alarm(self.seconds)
        return self

    def __exit__(self, *_error: object) -> None:
        if self.seconds > 0 and hasattr(signal, "SIGALRM"):
            signal.alarm(0)
            signal.signal(signal.SIGALRM, self.previous)


def _resolve(path: str | Path) -> Path:
    value = Path(path).expanduser()
    return (value if value.is_absolute() else PROJECT_ROOT / value).resolve()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _class_id(answer: str) -> str:
    normalized = normalize_answer(answer)
    if not normalized:
        raise ValueError("V73 answer normalizes to empty")
    return "answer_" + hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:20]


def _guard_training_path(path: str | Path, *, kind: str) -> Path:
    source = _resolve(path)
    forbidden = {"oracle", "validation", "validate", "test", "deferred", "final"}
    lowered = {part.lower() for part in source.parts}
    if lowered & forbidden:
        raise ValueError(f"V73 {kind} path crosses a forbidden boundary")
    if not source.exists() or source.is_symlink():
        raise FileNotFoundError(f"V73 {kind} is unavailable or symlinked: {source}")
    return source


def load_config_v73(path: str | Path) -> dict[str, Any]:
    source = _guard_training_path(path, kind="config")
    payload = yaml.safe_load(source.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping) or set(payload) != {"v73"}:
        raise ValueError("V73 config must contain exactly one v73 mapping")
    config = payload["v73"]
    if not isinstance(config, Mapping):
        raise ValueError("V73 config payload changed")
    split = config.get("split")
    if not isinstance(split, Mapping):
        raise ValueError("V73 split configuration is missing")
    if tuple(split.get("train_pair_ids", ())) != TRAIN_PAIR_IDS:
        raise ValueError("V73 historical training-pair split changed")
    if tuple(split.get("held_pair_ids", ())) != HELD_PAIR_IDS:
        raise ValueError("V73 replicated-family held-pair split changed")
    if set(TRAIN_PAIR_IDS) & set(HELD_PAIR_IDS):
        raise RuntimeError("V73 pair split overlaps")
    if config.get("gates") != asdict(LOCKED_GATES):
        raise ValueError("V73 numeric screen gates changed")
    if config.get("absolute_reader_gates") != asdict(LOCKED_ABSOLUTE_GATES):
        raise ValueError("V73 absolute reader gates changed")
    scope = config.get("scope")
    expected_scope = {
        "training_pool_only": True,
        "official_validation_loaded": False,
        "official_test_loaded": False,
        "deferred_final_loaded": False,
        "oracle_loaded": False,
        "gemma_generation_used": False,
        "checkpoint_published": False,
    }
    if scope != expected_scope:
        raise ValueError("V73 scope contract changed")
    if config.get("wall_time_budget_seconds") != 600:
        raise ValueError("V73 wall-time budget changed")
    return dict(config)


def load_training_rows_v73(path: str | Path) -> tuple[RowV73, ...]:
    source = _guard_training_path(path, kind="training QA")
    rows: list[RowV73] = []
    for line_number, line in enumerate(source.read_text(encoding="utf-8").splitlines(), 1):
        try:
            raw = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"V73 invalid training JSON at line {line_number}") from error
        required = {
            "scene_id",
            "question_id",
            "question",
            "answer",
            "answer_type",
            "counterfactual_pair_id",
            "counterfactual_paired_scene_id",
            "counterfactual_question_key",
            "counterfactual_change_type",
            "counterfactual_expected_change",
        }
        if not isinstance(raw, Mapping) or not required <= set(raw):
            raise ValueError(f"V73 training row fields changed at line {line_number}")
        string_fields = (
            "scene_id",
            "question_id",
            "question",
            "answer",
            "answer_type",
            "counterfactual_pair_id",
            "counterfactual_paired_scene_id",
            "counterfactual_question_key",
            "counterfactual_change_type",
        )
        if any(not isinstance(raw[field], str) or not raw[field] for field in string_fields):
            raise ValueError(f"V73 training string field changed at line {line_number}")
        if type(raw["counterfactual_expected_change"]) is not bool:
            raise ValueError("V73 expected-change field must be boolean")
        rows.append(
            RowV73(
                scene_id=raw["scene_id"],
                question_id=raw["question_id"],
                question=raw["question"],
                answer=normalize_answer(raw["answer"]),
                answer_class=_class_id(raw["answer"]),
                answer_type=raw["answer_type"],
                pair_id=raw["counterfactual_pair_id"],
                paired_scene_id=raw["counterfactual_paired_scene_id"],
                question_key=raw["counterfactual_question_key"],
                change_type=raw["counterfactual_change_type"],
                expected_change=raw["counterfactual_expected_change"],
            )
        )
    if len(rows) != EXPECTED_ROWS:
        raise ValueError(f"V73 requires exactly {EXPECTED_ROWS} training-pool rows")
    if len({row.key for row in rows}) != len(rows):
        raise ValueError("V73 training row keys are not unique")
    scenes = {row.scene_id for row in rows}
    pairs = {row.pair_id for row in rows}
    if len(scenes) != EXPECTED_SCENES or len(pairs) != EXPECTED_PAIRS:
        raise ValueError("V73 training scene or pair inventory changed")
    if pairs != set(TRAIN_PAIR_IDS) | set(HELD_PAIR_IDS):
        raise ValueError("V73 pair inventory differs from the locked split")
    per_pair: dict[str, list[RowV73]] = defaultdict(list)
    for row in rows:
        per_pair[row.pair_id].append(row)
    for pair_id, members in per_pair.items():
        if len(members) != 48 or len({row.scene_id for row in members}) != 2:
            raise ValueError(f"V73 pair layout changed: {pair_id}")
        if any(row.paired_scene_id not in {item.scene_id for item in members} for row in members):
            raise ValueError(f"V73 paired scene link escaped pair: {pair_id}")
    return tuple(rows)


def split_rows_v73(
    rows: Sequence[RowV73],
) -> tuple[tuple[RowV73, ...], tuple[RowV73, ...]]:
    train = tuple(row for row in rows if row.pair_id in TRAIN_PAIR_IDS)
    held = tuple(row for row in rows if row.pair_id in HELD_PAIR_IDS)
    if len(train) != EXPECTED_TRAIN_ROWS or len(held) != EXPECTED_HELD_ROWS:
        raise ValueError("V73 pair-disjoint row counts changed")
    if len({row.scene_id for row in train}) != EXPECTED_TRAIN_SCENES:
        raise ValueError("V73 historical training scene count changed")
    if len({row.scene_id for row in held}) != EXPECTED_HELD_SCENES:
        raise ValueError("V73 replicated-family held scene count changed")
    if {row.scene_id for row in train} & {row.scene_id for row in held}:
        raise ValueError("V73 train and held scenes overlap")
    return train, held


def changed_units_v73(rows: Sequence[RowV73]) -> tuple[ChangedUnitV73, ...]:
    grouped: dict[tuple[str, str], list[RowV73]] = defaultdict(list)
    for row in rows:
        if row.expected_change:
            grouped[(row.pair_id, row.question_key)].append(row)
    result: list[ChangedUnitV73] = []
    for (pair_id, question_key), members in sorted(grouped.items()):
        if len(members) != 2 or len({row.scene_id for row in members}) != 2:
            raise ValueError("V73 changed unit must contain exactly two scene sides")
        left, right = sorted(members, key=lambda row: row.scene_id)
        if left.question != right.question or left.answer_class == right.answer_class:
            raise ValueError("V73 changed unit question/answer contrast changed")
        result.append(
            ChangedUnitV73(pair_id, question_key, left.change_type, left, right)
        )
    return tuple(result)


def load_prefixes_v73(
    path: str | Path, scene_ids: Iterable[str]
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    root = _guard_training_path(path, kind="prefix cache")
    if not root.is_dir():
        raise ValueError("V73 prefix cache must be a directory")
    requested = tuple(sorted(set(scene_ids)))
    manifest_path = root / "manifest.json"
    if not manifest_path.is_file() or manifest_path.is_symlink():
        raise FileNotFoundError("V73 prefix manifest is unavailable")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        not isinstance(manifest, Mapping)
        or manifest.get("artifact") != "question_independent_scene_prefix_cache_v1"
        or manifest.get("scene_count") != EXPECTED_SCENES
        or manifest.get("question_inputs_used") is not False
        or manifest.get("question_dependent_scene_retrieval") is not False
        or manifest.get("complete_scene_prefixes") is not True
        or manifest.get("environmental_text_inputs") != []
    ):
        raise ValueError("V73 prefix cache contract changed")
    entries = manifest.get("scenes")
    if not isinstance(entries, Mapping) or not set(requested) <= set(entries):
        raise ValueError("V73 prefix cache lacks requested scenes")
    prefixes: dict[str, torch.Tensor] = {}
    for scene_id in requested:
        entry = entries[scene_id]
        if not isinstance(entry, Mapping) or entry.get("filename") != f"{scene_id}.safetensors":
            raise ValueError("V73 prefix entry filename is not opaque")
        source = root / entry["filename"]
        if (
            not source.is_file()
            or source.is_symlink()
            or source.stat().st_size != entry.get("file_size_bytes")
            or _sha256_file(source) != entry.get("file_sha256")
        ):
            raise ValueError(f"V73 prefix file changed: {scene_id}")
        state = load_file(str(source), device="cpu")
        if set(state) != {"scene_prefix"}:
            raise ValueError("V73 prefix file must contain only scene_prefix")
        prefix = state["scene_prefix"].detach().float().contiguous()
        if tuple(prefix.shape) != EXPECTED_PREFIX_SHAPE or not torch.isfinite(prefix).all():
            raise ValueError(f"V73 prefix tensor changed: {scene_id}")
        prefixes[scene_id] = prefix
    return prefixes, dict(manifest)


def _resolved_snapshot_asset(snapshot: Path, name: str) -> Path:
    logical = snapshot / name
    if not logical.exists():
        raise FileNotFoundError(f"V73 pinned snapshot asset is missing: {name}")
    resolved = logical.resolve(strict=True)
    blob_root = snapshot.parent.parent / "blobs"
    try:
        resolved.relative_to(blob_root.resolve(strict=True))
    except ValueError as error:
        raise ValueError("V73 snapshot asset escaped the pinned HF blob store") from error
    if not resolved.is_file() or resolved.is_symlink():
        raise ValueError("V73 resolved snapshot asset is not a regular blob")
    return resolved


def _tokenizer_inventory(snapshot: Path) -> tuple[Path, ...]:
    names = (
        "tokenizer.json",
        "tokenizer_config.json",
        "processor_config.json",
        "chat_template.jinja",
    )
    result = tuple(
        _resolved_snapshot_asset(snapshot, name)
        for name in names
        if (snapshot / name).exists()
    )
    if not result:
        raise ValueError("V73 pinned tokenizer inventory is unavailable")
    return result


def load_embedding_assets_v73(
    snapshot_path: str | Path,
    questions: Iterable[str],
    answers_by_class: Mapping[str, str],
) -> EmbeddingAssetsV73:
    snapshot = _guard_training_path(snapshot_path, kind="Gemma snapshot")
    if not snapshot.is_dir():
        raise ValueError("V73 Gemma snapshot must be a directory")
    tokenizer_files = _tokenizer_inventory(snapshot)
    tokenizer = AutoTokenizer.from_pretrained(
        str(snapshot), local_files_only=True, use_fast=True
    )
    question_ids: dict[str, torch.Tensor] = {}
    answer_ids: dict[str, torch.Tensor] = {}
    for question in sorted(set(questions)):
        encoded = tokenizer(question, add_special_tokens=False, return_tensors="pt")
        ids = encoded["input_ids"][0].long()
        if ids.numel() < 1 or ids.numel() > 64:
            raise ValueError("V73 question token length is outside the fixed bound")
        question_ids[question] = ids
    for class_id, answer in sorted(answers_by_class.items()):
        ids = tokenizer(answer, add_special_tokens=False, return_tensors="pt")[
            "input_ids"
        ][0].long()
        if ids.numel() < 1 or ids.numel() > 32:
            raise ValueError("V73 answer token length is outside the fixed bound")
        answer_ids[class_id] = ids

    model_file = _resolved_snapshot_asset(snapshot, "model.safetensors")
    tensor_name = "model.language_model.embed_tokens.weight"
    with safe_open(str(model_file), framework="pt", device="cpu") as handle:
        if tensor_name not in handle.keys():
            raise ValueError("V73 Gemma input embedding tensor is unavailable")
        embedding = handle.get_tensor(tensor_name)
    if (
        embedding.ndim != 2
        or embedding.shape[1] != EXPECTED_HIDDEN_SIZE
        or not torch.isfinite(embedding).all()
    ):
        raise ValueError("V73 Gemma input embedding tensor changed")
    questions_embedded = {
        text: embedding[ids].detach().float().contiguous()
        for text, ids in question_ids.items()
    }
    answers_embedded = {
        class_id: embedding[ids].detach().float().contiguous()
        for class_id, ids in answer_ids.items()
    }
    shape = tuple(int(value) for value in embedding.shape)
    dtype = str(embedding.dtype)
    del embedding, tokenizer
    return EmbeddingAssetsV73(
        questions=questions_embedded,
        answers=answers_embedded,
        tokenizer_files=tuple(str(path) for path in tokenizer_files),
        model_file=str(model_file),
        model_file_sha256=_sha256_file(model_file),
        embedding_tensor_name=tensor_name,
        embedding_shape=shape,
        embedding_dtype=dtype,
    )


def _four_token_prototype(value: torch.Tensor, target_rms: float) -> torch.Tensor:
    if value.ndim != 2 or value.shape[1] != EXPECTED_HIDDEN_SIZE:
        raise ValueError("V73 answer embedding shape changed")
    if value.shape[0] <= EXPECTED_CONTROL_TOKENS:
        indices = torch.arange(EXPECTED_CONTROL_TOKENS) % value.shape[0]
        result = value[indices]
    else:
        result = F.adaptive_avg_pool1d(
            value.T.unsqueeze(0), EXPECTED_CONTROL_TOKENS
        ).squeeze(0).T
    rms = result.square().mean(dim=-1, keepdim=True).sqrt()
    if bool((rms <= 1e-8).any()):
        raise ValueError("V73 answer prototype contains a zero token")
    return (result * (target_rms / rms)).float().contiguous()


def build_prototype_bank_v73(
    train_rows: Sequence[RowV73],
    answer_embeddings: Mapping[str, torch.Tensor],
    *,
    target_rms: float,
    basis_rank: int,
) -> PrototypeBankV73:
    class_ids = tuple(sorted({row.answer_class for row in train_rows}))
    if set(class_ids) != set(answer_embeddings):
        raise ValueError("V73 answer embedding inventory must be training-fold-only")
    prototypes = torch.stack(
        [_four_token_prototype(answer_embeddings[class_id], target_rms) for class_id in class_ids]
    )
    flattened = prototypes.reshape(-1, EXPECTED_HIDDEN_SIZE)
    _u, _s, vh = torch.linalg.svd(flattened, full_matrices=False)
    rank = min(int(basis_rank), vh.shape[0])
    basis = vh[:rank].float().contiguous()
    if not torch.allclose(
        basis @ basis.T, torch.eye(rank), atol=2e-4, rtol=2e-4
    ):
        raise RuntimeError("V73 SVD output basis lost orthonormality")
    return PrototypeBankV73(
        class_ids=class_ids,
        prototypes=prototypes,
        class_index={class_id: index for index, class_id in enumerate(class_ids)},
        output_basis=basis,
    )


def _question_batch(
    rows: Sequence[RowV73],
    questions: Mapping[str, torch.Tensor],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    values = [questions[row.question] for row in rows]
    length = max(value.shape[0] for value in values)
    batch = torch.zeros(len(values), length, EXPECTED_HIDDEN_SIZE, dtype=torch.float32)
    mask = torch.zeros(len(values), length, dtype=torch.bool)
    for index, value in enumerate(values):
        batch[index, : value.shape[0]] = value
        mask[index, : value.shape[0]] = True
    return batch.to(device), mask.to(device)


def _prefix_batch(
    rows: Sequence[RowV73],
    prefixes: Mapping[str, torch.Tensor],
    device: torch.device,
    *,
    wrong_scene: bool = False,
) -> torch.Tensor:
    scene_ids = [row.paired_scene_id if wrong_scene else row.scene_id for row in rows]
    return torch.cat([prefixes[scene_id] for scene_id in scene_ids]).to(device)


def _prototype_logits(
    output: torch.Tensor, prototypes: torch.Tensor, *, temperature: float
) -> torch.Tensor:
    flat = output.reshape(output.shape[0], -1)
    bank = prototypes.reshape(prototypes.shape[0], -1)
    bank = F.normalize(bank, dim=-1, eps=1e-8)
    return flat @ bank.T / float(temperature)


def _targets(
    rows: Sequence[RowV73], bank: PrototypeBankV73, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    ids = torch.tensor(
        [bank.class_index[row.answer_class] for row in rows],
        device=device,
        dtype=torch.long,
    )
    return bank.prototypes.to(device)[ids], ids


def _fit_reader_v73(
    model: nn.Module,
    train_rows: Sequence[RowV73],
    *,
    prefixes: Mapping[str, torch.Tensor],
    questions: Mapping[str, torch.Tensor],
    bank: PrototypeBankV73,
    config: Mapping[str, Any],
    seed: int,
    device: torch.device,
) -> dict[str, float | int]:
    fit = config["fit"]
    epochs = int(fit["epochs"])
    batch_size = int(fit["batch_size"])
    pair_batch_size = int(fit["pair_batch_size"])
    generator = torch.Generator().manual_seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    model.to(device).train()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(fit["learning_rate"]),
        weight_decay=float(fit["weight_decay"]),
    )
    units = changed_units_v73(train_rows)
    maximum_preclip = 0.0
    optimizer_steps = 0
    last_loss = math.inf
    started = time.perf_counter()
    target_scale = float(fit["prototype_rms"]) ** 2

    for _epoch in range(epochs):
        order = torch.randperm(len(train_rows), generator=generator).tolist()
        for offset in range(0, len(order), batch_size):
            batch = [train_rows[index] for index in order[offset : offset + batch_size]]
            question, mask = _question_batch(batch, questions, device)
            output = model(
                _prefix_batch(batch, prefixes, device), question, mask
            ).control_tokens
            target, class_ids = _targets(batch, bank, device)
            logits = _prototype_logits(
                output, bank.prototypes.to(device), temperature=float(fit["temperature"])
            )
            value = F.mse_loss(output, target) / target_scale
            classification = F.cross_entropy(logits, class_ids)
            loss = float(fit["value_weight"]) * value + float(
                fit["classification_weight"]
            ) * classification
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            preclip = float(
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), float(fit["gradient_clip_norm"])
                )
            )
            if not math.isfinite(preclip):
                raise RuntimeError("V73 row gradient became nonfinite")
            maximum_preclip = max(maximum_preclip, preclip)
            optimizer.step()
            optimizer_steps += 1
            last_loss = float(loss.detach().cpu())

        unit_order = torch.randperm(len(units), generator=generator).tolist()
        for offset in range(0, len(unit_order), pair_batch_size):
            selected = [units[index] for index in unit_order[offset : offset + pair_batch_size]]
            batch = [row for unit in selected for row in (unit.left, unit.right)]
            question, mask = _question_batch(batch, questions, device)
            output = model(
                _prefix_batch(batch, prefixes, device), question, mask
            ).control_tokens
            target, class_ids = _targets(batch, bank, device)
            pair_output = output.reshape(len(selected), 2, *output.shape[1:])
            pair_target = target.reshape(len(selected), 2, *target.shape[1:])
            delta_loss = F.mse_loss(
                pair_output[:, 1] - pair_output[:, 0],
                pair_target[:, 1] - pair_target[:, 0],
            ) / target_scale
            own = (pair_output - pair_target).square().mean(dim=(2, 3)) / target_scale
            opposite_target = pair_target.flip(1)
            opposite = (
                (pair_output - opposite_target).square().mean(dim=(2, 3)) / target_scale
            )
            opposite_loss = F.relu(
                float(fit["opposite_margin"]) + own - opposite
            ).mean()
            class_loss = F.cross_entropy(
                _prototype_logits(
                    output,
                    bank.prototypes.to(device),
                    temperature=float(fit["temperature"]),
                ),
                class_ids,
            )
            loss = (
                float(fit["pair_delta_weight"]) * delta_loss
                + float(fit["opposite_weight"]) * opposite_loss
                + float(fit["pair_classification_weight"]) * class_loss
                + float(fit["pair_value_weight"]) * own.mean()
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            preclip = float(
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), float(fit["gradient_clip_norm"])
                )
            )
            if not math.isfinite(preclip):
                raise RuntimeError("V73 pair gradient became nonfinite")
            maximum_preclip = max(maximum_preclip, preclip)
            optimizer.step()
            optimizer_steps += 1
            last_loss = float(loss.detach().cpu())

    return {
        "optimizer_steps": optimizer_steps,
        "elapsed_seconds": time.perf_counter() - started,
        "last_loss": last_loss,
        "maximum_preclip_gradient_norm": maximum_preclip,
    }


@torch.inference_mode()
def _predict_v73(
    model: nn.Module,
    rows: Sequence[RowV73],
    *,
    prefixes: Mapping[str, torch.Tensor],
    questions: Mapping[str, torch.Tensor],
    batch_size: int,
    device: torch.device,
    wrong_scene: bool = False,
    zero_scene: bool = False,
) -> torch.Tensor:
    model.eval().to(device)
    outputs: list[torch.Tensor] = []
    for offset in range(0, len(rows), batch_size):
        batch = rows[offset : offset + batch_size]
        question, mask = _question_batch(batch, questions, device)
        prefix = _prefix_batch(batch, prefixes, device, wrong_scene=wrong_scene)
        if zero_scene:
            prefix = torch.zeros_like(prefix)
        outputs.append(model(prefix, question, mask).control_tokens.detach().cpu())
    return torch.cat(outputs)


def _cosine_to_class(output: torch.Tensor, prototypes: torch.Tensor) -> torch.Tensor:
    return F.normalize(output.flatten(1), dim=-1, eps=1e-8) @ F.normalize(
        prototypes.flatten(1), dim=-1, eps=1e-8
    ).T


def _bootstrap_lower_bound(values: Sequence[float], seed: int) -> float:
    if not values:
        raise ValueError("V73 bootstrap population is empty")
    tensor = torch.tensor(values, dtype=torch.float64)
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randint(
        len(values), (2000, len(values)), generator=generator
    )
    means = tensor[indices].mean(dim=1).sort().values
    return float(means[int(0.025 * len(means))])


def evaluate_reader_v73(
    model: nn.Module,
    held_rows: Sequence[RowV73],
    *,
    prefixes: Mapping[str, torch.Tensor],
    questions: Mapping[str, torch.Tensor],
    bank: PrototypeBankV73,
    batch_size: int,
    device: torch.device,
) -> tuple[dict[str, Any], dict[tuple[str, str], float]]:
    outputs = _predict_v73(
        model,
        held_rows,
        prefixes=prefixes,
        questions=questions,
        batch_size=batch_size,
        device=device,
    )
    wrong = _predict_v73(
        model,
        held_rows,
        prefixes=prefixes,
        questions=questions,
        batch_size=batch_size,
        device=device,
        wrong_scene=True,
    )
    zero = _predict_v73(
        model,
        held_rows[: min(16, len(held_rows))],
        prefixes=prefixes,
        questions=questions,
        batch_size=batch_size,
        device=device,
        zero_scene=True,
    )
    similarities = _cosine_to_class(outputs, bank.prototypes)
    wrong_similarities = _cosine_to_class(wrong, bank.prototypes)
    prediction = similarities.argmax(dim=-1)
    index_by_key = {row.key: index for index, row in enumerate(held_rows)}
    supported = [row.answer_class in bank.class_index for row in held_rows]
    correct = [
        bool(supported[index] and int(prediction[index]) == bank.class_index[row.answer_class])
        for index, row in enumerate(held_rows)
    ]
    units = changed_units_v73(held_rows)
    if len(units) != EXPECTED_HELD_CHANGED_UNITS:
        raise ValueError("V73 held changed-unit count changed")
    margin_by_side: dict[tuple[str, str], float] = {}
    wrong_scene_drops: list[float] = []
    complete_units = 0
    prediction_change_units = 0
    positive_sides = 0
    positive_pair_delta = 0
    pair_delta_cosines: list[float] = []
    family_margins: dict[str, list[float]] = defaultdict(list)
    supported_changed_sides = 0
    for unit in units:
        indexes = [index_by_key[unit.left.key], index_by_key[unit.right.key]]
        if not all(supported[index] for index in indexes):
            continue
        supported_changed_sides += 2
        class_ids = [
            bank.class_index[unit.left.answer_class],
            bank.class_index[unit.right.answer_class],
        ]
        if all(correct[index] for index in indexes):
            complete_units += 1
        if int(prediction[indexes[0]]) != int(prediction[indexes[1]]):
            prediction_change_units += 1
        for side, (index, own, opposite) in enumerate(
            zip(indexes, class_ids, reversed(class_ids), strict=True)
        ):
            margin = float(similarities[index, own] - similarities[index, opposite])
            margin_by_side[(unit.pair_id, f"{unit.question_key}:{side}")] = margin
            family_margins[unit.change_type].append(margin)
            positive_sides += int(margin > 0.0)
            wrong_margin = float(
                wrong_similarities[index, own] - wrong_similarities[index, opposite]
            )
            wrong_scene_drops.append(margin - wrong_margin)
        predicted_delta = outputs[indexes[1]] - outputs[indexes[0]]
        target_delta = (
            bank.prototypes[class_ids[1]] - bank.prototypes[class_ids[0]]
        )
        pair_cosine = float(
            F.cosine_similarity(predicted_delta.flatten(), target_delta.flatten(), dim=0)
        )
        pair_delta_cosines.append(pair_cosine)
        positive_pair_delta += int(pair_cosine > 0.0)
    supported_total = sum(supported)
    exact = sum(correct)
    changed_correct = sum(
        correct[index] for index, row in enumerate(held_rows) if row.expected_change
    )
    return (
        {
            "inventory_total": len(held_rows),
            "supported_total": supported_total,
            "unsupported_total": len(held_rows) - supported_total,
            "supported_class_exact": exact,
            "supported_accuracy": exact / max(supported_total, 1),
            "changed_supported_sides": supported_changed_sides,
            "changed_class_exact": changed_correct,
            "changed_supported_accuracy": changed_correct
            / max(supported_changed_sides, 1),
            "complete_class_units": complete_units,
            "complete_unit_total": len(units),
            "prediction_change_units": prediction_change_units,
            "positive_own_over_opposite_sides": positive_sides,
            "mean_own_over_opposite_margin": sum(margin_by_side.values())
            / max(len(margin_by_side), 1),
            "positive_pair_delta_units": positive_pair_delta,
            "mean_pair_delta_cosine": sum(pair_delta_cosines)
            / max(len(pair_delta_cosines), 1),
            "mean_correct_over_wrong_scene_margin": sum(wrong_scene_drops)
            / max(len(wrong_scene_drops), 1),
            "zero_scene_maximum_absolute_control": float(zero.abs().max()),
            "family_mean_margins": {
                family: sum(values) / len(values)
                for family, values in sorted(family_margins.items())
            },
            "question_or_answer_text_serialized": False,
            "environmental_text_inputs": 0,
        },
        margin_by_side,
    )


def compare_readers_v73(
    full: Mapping[str, Any],
    dct: Mapping[str, Any],
    full_margins: Mapping[tuple[str, str], float],
    dct_margins: Mapping[tuple[str, str], float],
    *,
    seed: int,
) -> dict[str, Any]:
    if set(full_margins) != set(dct_margins):
        raise ValueError("V73 and DCT held side inventories differ")
    gains = [full_margins[key] - dct_margins[key] for key in sorted(full_margins)]
    family_gains = {
        family: full["family_mean_margins"][family]
        - dct["family_mean_margins"][family]
        for family in sorted(full["family_mean_margins"])
    }
    comparison = {
        "causal_margin_gain": sum(gains) / len(gains),
        "causal_margin_gain_bootstrap_lower_bound": _bootstrap_lower_bound(
            gains, seed
        ),
        "additional_positive_sides": full["positive_own_over_opposite_sides"]
        - dct["positive_own_over_opposite_sides"],
        "additional_complete_units": full["complete_class_units"]
        - dct["complete_class_units"],
        "prediction_change_units": full["prediction_change_units"],
        "improved_families": sum(value > 0.0 for value in family_gains.values()),
        "minimum_family_margin_gain": min(family_gains.values()),
        "broad_accuracy_drop": dct["supported_accuracy"]
        - full["supported_accuracy"],
        "correct_over_wrong_scene_margin": full[
            "mean_correct_over_wrong_scene_margin"
        ],
        "zero_scene_maximum_absolute_control": full[
            "zero_scene_maximum_absolute_control"
        ],
        "family_margin_gains": family_gains,
    }
    gates = asdict(LOCKED_GATES)
    checks = {
        "causal_margin_gain": comparison["causal_margin_gain"]
        >= gates["causal_margin_gain"],
        "causal_margin_gain_bootstrap_lower_bound": comparison[
            "causal_margin_gain_bootstrap_lower_bound"
        ]
        > gates["causal_margin_gain_bootstrap_lower_bound"],
        "additional_positive_sides": comparison["additional_positive_sides"]
        >= gates["additional_positive_sides"],
        "additional_complete_units": comparison["additional_complete_units"]
        >= gates["additional_complete_units"],
        "prediction_change_units": comparison["prediction_change_units"]
        >= gates["prediction_change_units"],
        "improved_families": comparison["improved_families"]
        >= gates["improved_families"],
        "minimum_family_margin_gain": comparison["minimum_family_margin_gain"]
        >= gates["minimum_family_margin_gain"],
        "broad_accuracy_retention": comparison["broad_accuracy_drop"]
        <= gates["maximum_broad_accuracy_drop"],
        "wrong_scene_control": comparison["correct_over_wrong_scene_margin"]
        >= gates["correct_over_wrong_scene_margin"],
        "zero_scene_control": comparison["zero_scene_maximum_absolute_control"]
        == gates["zero_scene_maximum_absolute_control"],
    }
    return {"metrics": comparison, "checks": checks, "passed": all(checks.values())}


def absolute_reader_gate_v73(metrics: Mapping[str, Any]) -> dict[str, Any]:
    gates = asdict(LOCKED_ABSOLUTE_GATES)
    checks = {
        "supported_accuracy": metrics["supported_accuracy"]
        >= gates["supported_accuracy"],
        "changed_supported_accuracy": metrics["changed_supported_accuracy"]
        >= gates["changed_supported_accuracy"],
        "complete_class_units": metrics["complete_class_units"]
        >= gates["complete_class_units"],
        "prediction_change_units": metrics["prediction_change_units"]
        >= gates["prediction_change_units"],
        "positive_own_over_opposite_sides": metrics[
            "positive_own_over_opposite_sides"
        ]
        >= gates["positive_own_over_opposite_sides"],
        "mean_own_over_opposite_margin": metrics["mean_own_over_opposite_margin"]
        >= gates["mean_own_over_opposite_margin"],
        "wrong_scene_control": metrics["mean_correct_over_wrong_scene_margin"]
        >= gates["correct_over_wrong_scene_margin"],
        "zero_scene_control": metrics["zero_scene_maximum_absolute_control"]
        == gates["zero_scene_maximum_absolute_control"],
    }
    return {"thresholds": gates, "checks": checks, "passed": all(checks.values())}


def _select_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    if name == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("V73 requested MPS but it is unavailable")
    if name not in {"cpu", "mps"}:
        raise ValueError("V73 device must be auto, cpu, or mps")
    return torch.device(name)


def _build_model(
    kind: str, bank: PrototypeBankV73, architecture: Mapping[str, Any]
) -> nn.Module:
    cls: type[nn.Module]
    if kind == "full_scene":
        cls = FullSceneSetAttentionQuestionControlV73
    elif kind == "dct40":
        cls = DCT40QuestionControlBaselineV73
    else:
        raise ValueError("V73 reader kind is unsupported")
    return cls(
        EXPECTED_HIDDEN_SIZE,
        bank.output_basis,
        expected_environment_latents=256,
        control_token_count=4,
        model_dimension=int(architecture["model_dimension"]),
        head_count=int(architecture["head_count"]),
        feedforward_dimension=int(architecture["feedforward_dimension"]),
        scene_encoder_layers=int(architecture["scene_encoder_layers"]),
        scene_cross_attention_layers=int(architecture["scene_cross_attention_layers"]),
        internal_reader_slots=int(architecture["internal_reader_slots"]),
        uniform_floor_mass=float(architecture["uniform_floor_mass"]),
        maximum_control_rms=float(architecture["maximum_control_rms"]),
        initial_control_rms=float(architecture["initial_control_rms"]),
    )


def build_preflight_v73(config: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    rows = load_training_rows_v73(config["training_qa"])
    train, held = split_rows_v73(rows)
    train_units = changed_units_v73(train)
    held_units = changed_units_v73(held)
    if len(held_units) != EXPECTED_HELD_CHANGED_UNITS:
        raise ValueError("V73 held unit inventory changed")
    prefix_root = _guard_training_path(config["prefix_cache"], kind="prefix cache")
    prefix_manifest = json.loads((prefix_root / "manifest.json").read_text())
    snapshot = _guard_training_path(config["gemma_snapshot"], kind="Gemma snapshot")
    model_file = _resolved_snapshot_asset(snapshot, "model.safetensors")
    preflight = {
        "artifact": "v73_fullscene_numeric_preflight_v1",
        "training_qa_sha256": _sha256_file(_resolve(config["training_qa"])),
        "prefix_manifest_sha256": _sha256_file(prefix_root / "manifest.json"),
        "base_checkpoint_sha256": prefix_manifest.get("base_checkpoint_sha256"),
        "gemma_model_file_sha256": _sha256_file(model_file),
        "row_count": len(rows),
        "scene_count": len({row.scene_id for row in rows}),
        "pair_count": len({row.pair_id for row in rows}),
        "training_rows": len(train),
        "training_scenes": len({row.scene_id for row in train}),
        "training_pairs": list(TRAIN_PAIR_IDS),
        "training_changed_units": len(train_units),
        "held_rows": len(held),
        "held_scenes": len({row.scene_id for row in held}),
        "held_pairs": list(HELD_PAIR_IDS),
        "held_changed_sides": sum(row.expected_change for row in held),
        "held_changed_units": len(held_units),
        "train_held_pair_disjoint": True,
        "train_held_scene_disjoint": True,
        "official_validation_loaded": False,
        "official_test_loaded": False,
        "oracle_loaded": False,
        "gemma_generation_used": False,
        "checkpoint_published": False,
        "numeric_gates": asdict(LOCKED_GATES),
        "absolute_reader_gates": asdict(LOCKED_ABSOLUTE_GATES),
    }
    return preflight, {
        "rows": rows,
        "train": train,
        "held": held,
    }


def run_screen_v73(config: Mapping[str, Any]) -> dict[str, Any]:
    with _WallTimerV73(int(config["wall_time_budget_seconds"])):
        return _run_screen_inside_budget_v73(config)


def _run_screen_inside_budget_v73(config: Mapping[str, Any]) -> dict[str, Any]:
    started = time.perf_counter()
    preflight, state = build_preflight_v73(config)
    rows: tuple[RowV73, ...] = state["rows"]
    train: tuple[RowV73, ...] = state["train"]
    held: tuple[RowV73, ...] = state["held"]
    prefixes, _manifest = load_prefixes_v73(
        config["prefix_cache"], {row.scene_id for row in rows}
    )
    train_answer_text: dict[str, str] = {}
    for row in train:
        prior = train_answer_text.setdefault(row.answer_class, row.answer)
        if prior != row.answer:
            raise ValueError("V73 answer class hash collision detected")
    assets = load_embedding_assets_v73(
        config["gemma_snapshot"],
        {row.question for row in rows},
        train_answer_text,
    )
    bank = build_prototype_bank_v73(
        train,
        assets.answers,
        target_rms=float(config["fit"]["prototype_rms"]),
        basis_rank=int(config["architecture"]["output_basis_rank"]),
    )
    device = _select_device(str(config["device"]))
    seed = int(config["seed"])
    torch.manual_seed(seed)
    full = _build_model("full_scene", bank, config["architecture"])
    torch.manual_seed(seed)
    dct = _build_model("dct40", bank, config["architecture"])
    if not torch.equal(full.coefficient_output.weight, dct.coefficient_output.weight):
        raise RuntimeError("V73 readers lost their identical zero-output initialization")
    full_initial_zero = float(
        full(
            torch.zeros(1, *EXPECTED_PREFIX_SHAPE[1:]),
            torch.ones(1, 2, EXPECTED_HIDDEN_SIZE),
        ).control_tokens.abs().max()
    )
    if full_initial_zero != 0.0:
        raise RuntimeError("V73 initial zero-scene control is not exact zero")
    full_fit = _fit_reader_v73(
        full,
        train,
        prefixes=prefixes,
        questions=assets.questions,
        bank=bank,
        config=config,
        seed=seed,
        device=device,
    )
    dct_fit = _fit_reader_v73(
        dct,
        train,
        prefixes=prefixes,
        questions=assets.questions,
        bank=bank,
        config=config,
        seed=seed,
        device=device,
    )
    full_metrics, full_margins = evaluate_reader_v73(
        full,
        held,
        prefixes=prefixes,
        questions=assets.questions,
        bank=bank,
        batch_size=int(config["fit"]["evaluation_batch_size"]),
        device=device,
    )
    dct_metrics, dct_margins = evaluate_reader_v73(
        dct,
        held,
        prefixes=prefixes,
        questions=assets.questions,
        bank=bank,
        batch_size=int(config["fit"]["evaluation_batch_size"]),
        device=device,
    )
    comparison = compare_readers_v73(
        full_metrics, dct_metrics, full_margins, dct_margins, seed=seed
    )
    full_absolute = absolute_reader_gate_v73(full_metrics)
    dct_absolute = absolute_reader_gate_v73(dct_metrics)
    numeric_causality_demonstrated = bool(
        full_absolute["passed"] or dct_absolute["passed"]
    )
    selected_reader = max(
        ("full_scene", "dct40"),
        key=lambda name: (
            (full_metrics if name == "full_scene" else dct_metrics)[
                "complete_class_units"
            ],
            (full_metrics if name == "full_scene" else dct_metrics)[
                "prediction_change_units"
            ],
            (full_metrics if name == "full_scene" else dct_metrics)[
                "supported_accuracy"
            ],
        ),
    )
    full_cpu = full.cpu().eval()
    sample = held[0]
    question, mask = _question_batch([sample], assets.questions, torch.device("cpu"))
    full_cpu(prefixes[sample.scene_id], question, mask, return_traces=True)
    audit = asdict(full_cpu.audit())
    result = {
        "schema_version": 1,
        "artifact": "v73_fullscene_pair_disjoint_numeric_screen_v1",
        "passed": numeric_causality_demonstrated,
        "numeric_causality_demonstrated": numeric_causality_demonstrated,
        "full_scene_advantage_demonstrated": comparison["passed"],
        # One held row has a class absent from the historical fit.  Numeric
        # continuation can be earned, but runtime promotion fails closed until
        # a training-only continuous teacher for that opaque class is verified.
        "promotion_eligible": False,
        "promotion_blocker": "unverified_held_only_continuous_answer_class",
        "selected_reader_for_continuation": selected_reader,
        "checkpoint_published": False,
        "gemma_generation_used": False,
        "preflight": preflight,
        "architecture": {
            "full_scene": audit,
            "full_scene_trainable_parameters": full.trainable_parameter_count,
            "dct40_trainable_parameters": dct.trainable_parameter_count,
            "output_basis_rank": bank.output_basis.shape[0],
            "native_output_shape": [4, 1536],
            "immutable_full_prefix_retained_separately": True,
            "question_only_output_path_exists": False,
        },
        "fit": {"full_scene": full_fit, "dct40": dct_fit},
        "held_metrics": {"full_scene": full_metrics, "dct40": dct_metrics},
        "absolute_reader_gates": {
            "full_scene": full_absolute,
            "dct40": dct_absolute,
        },
        "comparison": comparison,
        "scope": config["scope"],
        "provenance": {
            "embedding_tensor_name": assets.embedding_tensor_name,
            "embedding_shape": list(assets.embedding_shape),
            "embedding_dtype": assets.embedding_dtype,
            "gemma_model_file_sha256": assets.model_file_sha256,
            "question_or_answer_text_serialized": False,
        },
        "total_wall_time_seconds": time.perf_counter() - started,
    }
    return result


def _write_new_json(path: Path, payload: Mapping[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"V73 result already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    data = (
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    descriptor, name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("preflight", "screen"))
    parser.add_argument(
        "--config",
        default="configs/experiments/gemma4_v73_fullscene_controller.yaml",
    )
    parser.add_argument("--output")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    config = load_config_v73(args.config)
    if args.mode == "preflight":
        preflight, _state = build_preflight_v73(config)
        print(json.dumps(preflight, indent=2, sort_keys=True, allow_nan=False))
        return 0
    if not args.output:
        raise ValueError("V73 screen mode requires --output")
    result = run_screen_v73(config)
    _write_new_json(_resolve(args.output), result)
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0 if result["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "EXPECTED_HELD_CHANGED_UNITS",
    "HELD_PAIR_IDS",
    "LOCKED_GATES",
    "ScreenGatesV73",
    "TRAIN_PAIR_IDS",
    "build_preflight_v73",
    "build_prototype_bank_v73",
    "changed_units_v73",
    "compare_readers_v73",
    "evaluate_reader_v73",
    "load_config_v73",
    "load_prefixes_v73",
    "load_training_rows_v73",
    "split_rows_v73",
]
