"""Train and gate one rank-4 Gemma-4 reader over fixed V54 scene prefixes."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import resource
import shutil
import tempfile
import time
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import torch
import torch.nn.functional as F
import yaml
from safetensors.torch import load_file, save_file

from semantic_3d_chat.chat.runtime import StaticChatRuntime
from semantic_3d_chat.chat.runtime_config import load_runtime_config
from semantic_3d_chat.config import PROJECT_ROOT
from semantic_3d_chat.evaluation.fixed_prefix_ple_v54_preregistration import (
    ARTIFACT,
    BASE_CHECKPOINT,
    BASELINE_PREDICTIONS,
    CONFIG,
    OUTPUT_CHECKPOINT,
    PREFIX_CACHE,
    PREREGISTRATION,
    RESULT_REPORT,
    RETENTION,
    SMOKE_REPORT,
    TRAIN_QA,
    TRAIN_SCENES,
    VALIDATION_QUESTIONS,
    VALIDATION_REFERENCES,
    VALIDATION_SCENES,
    authenticate_preregistration,
    build_preregistration,
    sha256_file,
    validate_objective,
)
from semantic_3d_chat.evaluation.metrics import normalize_answer
from semantic_3d_chat.evaluation.prediction_artifacts import checkpoint_fingerprint
from semantic_3d_chat.evaluation.v55_development_score import (
    canonical_type_specific_match,
)
from semantic_3d_chat.language.generation import generate_from_embeddings
from semantic_3d_chat.language.local_lm import prompt_token_ids
from semantic_3d_chat.language.lora import (
    LoRAInstallation,
    initialize_lora_adapter_state,
    install_lora_adapters,
)
from semantic_3d_chat.language.prefix_injection import (
    PrefixBatch,
    prefix_sha256,
    scene_boundary_mode_setting,
    scene_prefix_after_bos_setting,
    stack_prefix_batches,
)
from semantic_3d_chat.training.pair_curriculum import token_normalized_nll
from semantic_3d_chat.training.train_adapter import forward_prefix_batch, tokenize_answer
from semantic_3d_chat.training.train_question_control_v56 import assert_answer_only_labels

_EXPECTED_BASE_FINGERPRINT: Final[str] = (
    "3e128b40c1b73bb32750285679cda6b1bea364e67465e986a94a81dfc95e81e8"
)
_EXPECTED_RUNTIME_EFFECTIVE_SHA256: Final[str] = (
    "714c60ce9ccb1dff69c72f6618f8afb6f31bc60a830b5ee0fb794fedaa8a321e"
)
_MAX_UPDATES: Final[int] = 40
_GRADIENT_ACCUMULATION: Final[int] = 4
_SEED: Final[int] = 720054


@dataclass(frozen=True)
class ReaderRecord:
    scene_id: str
    question_id: str
    question: str
    answer: str
    answer_type: str
    pair_id: str | None
    pair_question_key: str | None
    paired_scene_id: str | None
    changed: bool
    role: str | None


@dataclass
class ReaderBundle:
    runtime: StaticChatRuntime
    installation: LoRAInstallation
    prefixes: dict[str, torch.Tensor]
    config: dict[str, Any]

    @property
    def language(self) -> Any:
        return self.runtime.language


def _resolve(path: str | Path) -> Path:
    value = Path(path).expanduser()
    return (value if value.is_absolute() else PROJECT_ROOT / value).resolve()


def _read_json(path: str | Path) -> Any:
    source = _resolve(path)
    if not source.is_file() or source.is_symlink():
        raise FileNotFoundError(f"PLE-V54 JSON input is missing or unsafe: {source}")
    return json.loads(source.read_text(encoding="utf-8"))


def _canonical_hash(value: object) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def _atomic_create_json(path: str | Path, value: Mapping[str, Any]) -> tuple[Path, str]:
    destination = _resolve(path)
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"PLE-V54 create-once artifact exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = (
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False)
        + "\n"
    ).encode()
    descriptor, name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination, hashlib.sha256(payload).hexdigest()


def _config() -> dict[str, Any]:
    source = _resolve(CONFIG)
    value = yaml.safe_load(source.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError("PLE-V54 experiment config must be a mapping")
    return value


def _record(value: Mapping[str, Any]) -> ReaderRecord:
    required = ("scene_id", "question_id", "question", "answer", "answer_type")
    if any(not isinstance(value.get(key), str) or not value[key] for key in required):
        raise TypeError("PLE-V54 QA row has invalid required text fields")
    paired = value.get("counterfactual_paired_scene_id")
    pair_id = value.get("counterfactual_pair_id")
    pair_key = value.get("counterfactual_question_key")
    role = value.get("counterfactual_role")
    for item, label in (
        (paired, "paired scene"),
        (pair_id, "pair ID"),
        (pair_key, "pair question key"),
        (role, "pair role"),
    ):
        if item is not None and (not isinstance(item, str) or not item):
            raise TypeError(f"PLE-V54 {label} must be a nonempty string or null")
    changed = value.get("counterfactual_expected_change") is True
    return ReaderRecord(
        scene_id=str(value["scene_id"]),
        question_id=str(value["question_id"]),
        question=str(value["question"]),
        answer=str(value["answer"]),
        answer_type=str(value["answer_type"]),
        pair_id=pair_id,
        pair_question_key=pair_key,
        paired_scene_id=paired,
        changed=changed,
        role=role,
    )


def load_training_records() -> list[ReaderRecord]:
    rows = [
        _record(json.loads(line))
        for line in _resolve(TRAIN_QA).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(rows) != 576 or {row.scene_id for row in rows} != set(TRAIN_SCENES):
        raise ValueError("PLE-V54 training inventory changed")
    if sum(row.changed for row in rows) != 80:
        raise ValueError("PLE-V54 changed training-side inventory changed")
    keys = {(row.scene_id, row.question_id) for row in rows}
    if len(keys) != len(rows):
        raise ValueError("PLE-V54 training rows contain duplicate opaque keys")
    if any(
        row.changed
        and (
            row.paired_scene_id not in TRAIN_SCENES
            or row.paired_scene_id == row.scene_id
            or row.pair_id is None
            or row.pair_question_key is None
        )
        for row in rows
    ):
        raise ValueError("PLE-V54 changed training rows lack a paired training scene")
    return rows


def load_validation_records() -> list[ReaderRecord]:
    questions = _read_json(VALIDATION_QUESTIONS)
    references = _read_json(VALIDATION_REFERENCES)
    question_rows = questions.get("questions") if isinstance(questions, Mapping) else None
    reference_rows = references.get("records") if isinstance(references, Mapping) else None
    if not isinstance(question_rows, list) or not isinstance(reference_rows, list):
        raise TypeError("PLE-V54 validation manifests are invalid")
    question_index = {
        (str(row["scene_id"]), str(row["question_id"])): str(row["question"])
        for row in question_rows
    }
    rows: list[ReaderRecord] = []
    for reference in reference_rows:
        if not isinstance(reference, Mapping):
            raise TypeError("PLE-V54 validation reference row is invalid")
        key = (str(reference["scene_id"]), str(reference["question_id"]))
        if key not in question_index:
            raise ValueError(f"PLE-V54 validation question missing: {key}")
        rows.append(
            _record(
                {
                    **reference,
                    "question": question_index[key],
                    "counterfactual_expected_change": bool(reference.get("route_label")),
                }
            )
        )
    if (
        len(rows) != 384
        or {row.scene_id for row in rows} != set(VALIDATION_SCENES)
        or sum(row.changed for row in rows) != 52
        or set(question_index) != {(row.scene_id, row.question_id) for row in rows}
    ):
        raise ValueError("PLE-V54 validation inventory changed")
    if not set(TRAIN_SCENES).isdisjoint({row.scene_id for row in rows}):
        raise ValueError("PLE-V54 training and validation scenes overlap")
    return rows


def load_prefixes() -> dict[str, torch.Tensor]:
    manifest = _read_json(f"{PREFIX_CACHE}/manifest.json")
    prereg = build_preregistration()
    if not isinstance(manifest, Mapping) or manifest.get("scene_count") != 40:
        raise ValueError("PLE-V54 prefix-cache manifest changed")
    result: dict[str, torch.Tensor] = {}
    for scene_id in (*TRAIN_SCENES, *VALIDATION_SCENES):
        entry = manifest["scenes"][scene_id]
        source = _resolve(PREFIX_CACHE) / str(entry["filename"])
        if (
            source.is_symlink()
            or not source.is_file()
            or source.stat().st_size != int(entry["file_size_bytes"])
            or sha256_file(source) != entry["file_sha256"]
        ):
            raise ValueError(f"PLE-V54 cached prefix bytes changed: {scene_id}")
        state = load_file(str(source), device="cpu")
        if set(state) != {"scene_prefix"}:
            raise ValueError("PLE-V54 prefix cache contains unexpected tensors")
        prefix = state["scene_prefix"].detach().contiguous()
        if (
            tuple(prefix.shape) != (1, 258, 1536)
            or prefix.dtype != torch.bfloat16
            or prefix_sha256(prefix) != entry["prefix_sha256"]
            or not torch.isfinite(prefix).all()
        ):
            raise ValueError(f"PLE-V54 cached prefix tensor changed: {scene_id}")
        result[scene_id] = prefix
    if prereg["runtime_contract"]["question_dependent_retrieval"] is not False:
        raise AssertionError("PLE-V54 preregistration unexpectedly allows retrieval")
    return result


def load_retention_corpus() -> list[dict[str, str]]:
    value = _read_json(RETENTION)
    examples = value.get("examples") if isinstance(value, Mapping) else None
    if (
        value.get("environmental_scene_content") is not False
        or not isinstance(examples, list)
        or len(examples) != 16
    ):
        raise ValueError("PLE-V54 retention corpus contract changed")
    result: list[dict[str, str]] = []
    for row in examples:
        if (
            not isinstance(row, Mapping)
            or set(row) != {"prompt", "continuation"}
            or not all(isinstance(row[key], str) and row[key] for key in row)
        ):
            raise TypeError("PLE-V54 retention example is invalid")
        result.append({"prompt": row["prompt"], "continuation": row["continuation"]})
    return result


def _load_bundle(*, gradient_checkpointing: bool) -> ReaderBundle:
    experiment = _config()
    runtime_config = load_runtime_config(_resolve(experiment["experiment"]["base_runtime_config"]))
    base_fingerprint, _ = checkpoint_fingerprint(_resolve(BASE_CHECKPOINT))
    if base_fingerprint != _EXPECTED_BASE_FINGERPRINT:
        raise ValueError("PLE-V54 base checkpoint fingerprint changed")
    runtime = StaticChatRuntime.load(
        runtime_config,
        TRAIN_SCENES[0],
        checkpoint=_resolve(BASE_CHECKPOINT),
        local_files_only=True,
    )
    prefixes = load_prefixes()
    if runtime.scene_prefix_hash != prefix_sha256(prefixes[TRAIN_SCENES[0]]):
        raise ValueError("PLE-V54 cached prefix differs from a fresh frozen V54 runtime")
    runtime.language.model.requires_grad_(False)
    from semantic_3d_chat.evaluation.ple_reader_preregistration import (
        reader_lora_settings,
        validate_projection_surface,
    )

    validate_projection_surface(runtime.language.model)
    installation = install_lora_adapters(runtime.language.model, reader_lora_settings())
    if installation is None:
        raise RuntimeError("PLE-V54 reader LoRA was not installed")
    initialize_lora_adapter_state(installation, seed=_SEED)
    installation.assert_only_lora_trainable(runtime.language.model)
    if installation.parameter_count != 41_984:
        raise ValueError("PLE-V54 trainable parameter count changed")
    if gradient_checkpointing:
        runtime.language.enable_decoder_gradient_checkpointing()
    return ReaderBundle(runtime, installation, prefixes, experiment)


def _prepared_batch(bundle: ReaderBundle, prefix: torch.Tensor, row: ReaderRecord) -> PrefixBatch:
    language = bundle.language
    backend = language.prefix_backend
    if backend is None or language.backend_name != "gemma4":
        raise RuntimeError("PLE-V54 requires the local Gemma-4 prefix backend")
    model_dtype = next(language.model.parameters()).dtype
    scene_prefix = prefix.to(device=language.device, dtype=model_dtype)
    prompt_ids = prompt_token_ids(
        language.tokenizer,
        str(bundle.runtime.config["language"]["system_prompt"]),
        row.question,
        language.device,
    )
    answer_ids = tokenize_answer(language.tokenizer, row.answer, language.device)
    prepared = backend.prepare(
        scene_prefix,
        prompt_ids,
        answer_ids,
        scene_prefix_after_bos=scene_prefix_after_bos_setting(bundle.runtime.config),
        scene_boundary_mode=scene_boundary_mode_setting(bundle.runtime.config),
    )
    assert_answer_only_labels(prepared.labels, answer_ids)
    return PrefixBatch(
        inputs_embeds=prepared.inputs_embeds,
        attention_mask=prepared.attention_mask,
        labels=prepared.labels,
        scene_prefix_length=prepared.scene_prefix_length,
        per_layer_inputs=prepared.per_layer_inputs,
        mm_token_type_ids=prepared.mm_token_type_ids,
    )


def answer_nlls(
    bundle: ReaderBundle,
    examples: Sequence[tuple[torch.Tensor, ReaderRecord]],
) -> torch.Tensor:
    batches = [_prepared_batch(bundle, prefix, row) for prefix, row in examples]
    stacked = stack_prefix_batches(
        batches,
        bundle.language.device,
        prefix_backend=bundle.language.prefix_backend,
    )
    output = forward_prefix_batch(bundle.language, stacked)
    if stacked.labels is None:
        raise RuntimeError("PLE-V54 teacher forcing lost its labels")
    nll = token_normalized_nll(output.logits, stacked.labels)
    if nll.shape != (len(examples),) or not torch.isfinite(nll).all():
        raise RuntimeError("PLE-V54 answer NLL is invalid")
    return nll


def changed_side_loss(bundle: ReaderBundle, row: ReaderRecord) -> tuple[torch.Tensor, dict[str, float]]:
    if not row.changed or row.paired_scene_id is None:
        raise ValueError("PLE-V54 changed-side loss requires a paired changed row")
    nll = answer_nlls(
        bundle,
        (
            (bundle.prefixes[row.scene_id], row),
            (bundle.prefixes[row.paired_scene_id], row),
        ),
    )
    loss, diagnostics = validate_objective(nll[:1], nll[1:])
    return loss, {
        "correct_nll": float(nll[0].detach().cpu()),
        "wrong_nll": float(nll[1].detach().cpu()),
        "margin": float(diagnostics["wrong_prefix_margins"][0].detach().cpu()),
        "hinge": float(diagnostics["wrong_prefix_hinge"].detach().cpu()),
    }


def broad_loss(bundle: ReaderBundle, row: ReaderRecord) -> torch.Tensor:
    return answer_nlls(bundle, ((bundle.prefixes[row.scene_id], row),)).mean()


def _retention_ids(bundle: ReaderBundle, prompt: str) -> torch.Tensor:
    encoded = bundle.language.tokenizer(prompt, return_tensors="pt")
    ids = encoded.input_ids if hasattr(encoded, "input_ids") else encoded["input_ids"]
    if ids.ndim != 2 or ids.shape[1] < 1:
        raise ValueError("PLE-V54 retention prompt tokenized to an empty sequence")
    return ids.to(bundle.language.device)


def _retention_target_id(bundle: ReaderBundle, continuation: str) -> int:
    encoded = bundle.language.tokenizer(
        continuation, add_special_tokens=False, return_tensors="pt"
    )
    ids = encoded.input_ids if hasattr(encoded, "input_ids") else encoded["input_ids"]
    if ids.ndim != 2 or ids.shape[1] < 1:
        raise ValueError("PLE-V54 retention continuation tokenized to an empty sequence")
    return int(ids[0, 0].item())


def retention_logits(bundle: ReaderBundle, prompt: str) -> torch.Tensor:
    ids = _retention_ids(bundle, prompt)
    output = bundle.language.model(input_ids=ids, use_cache=False, logits_to_keep=1)
    logits = output.logits[:, -1].float()
    if logits.ndim != 2 or logits.shape[0] != 1 or not torch.isfinite(logits).all():
        raise RuntimeError("PLE-V54 retention logits are invalid")
    return logits


@torch.inference_mode()
def retention_baseline(bundle: ReaderBundle, corpus: Sequence[Mapping[str, str]]) -> list[torch.Tensor]:
    return [retention_logits(bundle, row["prompt"]).detach().cpu() for row in corpus]


def retention_kl_loss(bundle: ReaderBundle, row: Mapping[str, str], teacher: torch.Tensor) -> torch.Tensor:
    current = retention_logits(bundle, row["prompt"])
    teacher_probabilities = torch.softmax(teacher.to(current.device).float(), dim=-1)
    loss = F.kl_div(
        torch.log_softmax(current, dim=-1),
        teacher_probabilities,
        reduction="batchmean",
    )
    if loss.ndim != 0 or not torch.isfinite(loss):
        raise RuntimeError("PLE-V54 retention KL is invalid")
    return loss


def memory_metrics() -> dict[str, int | None]:
    rss = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    # ru_maxrss is bytes on macOS and KiB on Linux.
    if os.uname().sysname != "Darwin":
        rss *= 1024
    current = None
    driver = None
    if torch.backends.mps.is_available():
        try:
            current = int(torch.mps.current_allocated_memory())
            driver_method = getattr(torch.mps, "driver_allocated_memory", None)
            driver = None if driver_method is None else int(driver_method())
        except RuntimeError:
            pass
    return {
        "peak_process_rss_bytes": rss,
        "mps_current_allocated_bytes": current,
        "mps_driver_allocated_bytes": driver,
    }


def structural_preflight() -> dict[str, Any]:
    prereg = authenticate_preregistration(PREREGISTRATION)
    train = load_training_records()
    validation = load_validation_records()
    retention = load_retention_corpus()
    manifest = _read_json(f"{PREFIX_CACHE}/manifest.json")
    output = _resolve(OUTPUT_CHECKPOINT)
    result = _resolve(RESULT_REPORT)
    return {
        "artifact": ARTIFACT,
        "status": "structural_preflight_passed",
        "preregistration": prereg,
        "train_rows": len(train),
        "train_scenes": len({row.scene_id for row in train}),
        "validation_rows": len(validation),
        "validation_scenes": len({row.scene_id for row in validation}),
        "scene_disjoint": set(TRAIN_SCENES).isdisjoint(VALIDATION_SCENES),
        "changed_train_sides": sum(row.changed for row in train),
        "changed_validation_sides": sum(row.changed for row in validation),
        "retention_examples": len(retention),
        "prefix_cache_scene_count": manifest["scene_count"],
        "output_checkpoint_absent": not output.exists(),
        "terminal_result_absent": not result.exists(),
        "mps_available": torch.backends.mps.is_available(),
        "passed": not output.exists() and not result.exists(),
    }


def gradient_smoke() -> dict[str, Any]:
    if _resolve(SMOKE_REPORT).exists():
        raise FileExistsError("PLE-V54 smoke report already exists")
    preflight = structural_preflight()
    if preflight["passed"] is not True:
        raise RuntimeError("PLE-V54 structural preflight failed before gradient smoke")
    started = time.perf_counter()
    bundle = _load_bundle(gradient_checkpointing=True)
    train = load_training_records()
    row = next(record for record in train if record.changed)
    corpus = load_retention_corpus()
    teachers = retention_baseline(bundle, corpus[:1])
    bundle.installation.train()
    bundle.language.model.zero_grad(set_to_none=True)
    loss, diagnostics = changed_side_loss(bundle, row)
    retention = retention_kl_loss(bundle, corpus[0], teachers[0])
    total = loss + 0.2 * retention
    total.backward()
    gradients = bundle.installation.gradient_norms()
    bundle.installation.validate_state()
    passed = (
        torch.isfinite(total).item()
        and float(gradients["total_l2"]) > 0.0
        and math.isfinite(diagnostics["margin"])
        and abs(float(retention.detach().cpu())) <= 1e-6
    )
    report = {
        "schema_version": 1,
        "artifact": f"{ARTIFACT}_gradient_smoke",
        "status": "passed" if passed else "failed",
        "passed": bool(passed),
        "device": str(bundle.language.device),
        "base_dtype": str(next(bundle.language.model.parameters()).dtype),
        "trainable_parameter_count": bundle.installation.parameter_count,
        "initial_adapter_state_sha256": bundle.installation.state_sha256(),
        "loss": float(total.detach().cpu()),
        "correct_answer_nll": diagnostics["correct_nll"],
        "wrong_answer_nll": diagnostics["wrong_nll"],
        "initial_wrong_prefix_margin": diagnostics["margin"],
        "initial_retention_kl": float(retention.detach().cpu()),
        "gradient_l2": gradients["total_l2"],
        "gradient_by_module": gradients["by_module"],
        "elapsed_seconds": time.perf_counter() - started,
        "memory": memory_metrics(),
        "prefix_shape": [1, 258, 1536],
        "question_dependent_scene_retrieval": False,
        "environmental_text_inputs": [],
        "preregistration_sha256": sha256_file(PREREGISTRATION),
    }
    _atomic_create_json(SMOKE_REPORT, report)
    return report


def build_schedule(records: Sequence[ReaderRecord]) -> list[tuple[ReaderRecord, ReaderRecord, ReaderRecord, ReaderRecord]]:
    grouped: defaultdict[tuple[str, str], list[ReaderRecord]] = defaultdict(list)
    broad: list[ReaderRecord] = []
    for row in records:
        if row.changed:
            if row.pair_id is None or row.pair_question_key is None:
                raise ValueError("PLE-V54 changed row lacks a pair key")
            grouped[(row.pair_id, row.pair_question_key)].append(row)
        else:
            broad.append(row)
    units: list[tuple[ReaderRecord, ReaderRecord]] = []
    for key, sides in grouped.items():
        if (
            len(sides) != 2
            or len({row.scene_id for row in sides}) != 2
            or len({row.answer for row in sides}) != 2
            or len({row.question for row in sides}) != 1
        ):
            raise ValueError(f"PLE-V54 changed unit is invalid: {key}")
        units.append(tuple(sorted(sides, key=lambda row: (row.role or "", row.scene_id))))
    if len(units) != _MAX_UPDATES or len(broad) != 496:
        raise ValueError("PLE-V54 schedule inventory changed")
    rng = random.Random(_SEED)
    rng.shuffle(units)
    rng.shuffle(broad)
    return [
        (first, second, broad[2 * index], broad[2 * index + 1])
        for index, (first, second) in enumerate(units)
    ]


@torch.inference_mode()
def evaluate_teacher_forcing(
    bundle: ReaderBundle,
    rows: Sequence[ReaderRecord],
) -> dict[str, Any]:
    installation = bundle.installation
    installation.eval()
    correct: dict[tuple[str, str], float] = {}
    batch_size = 2
    for offset in range(0, len(rows), batch_size):
        selected = rows[offset : offset + batch_size]
        nlls = answer_nlls(
            bundle,
            tuple((bundle.prefixes[row.scene_id], row) for row in selected),
        )
        for row, value in zip(selected, nlls.tolist(), strict=True):
            correct[(row.scene_id, row.question_id)] = float(value)
    changed = [row for row in rows if row.changed]
    wrong: dict[tuple[str, str], float] = {}
    for offset in range(0, len(changed), batch_size):
        selected = changed[offset : offset + batch_size]
        if any(row.paired_scene_id is None for row in selected):
            raise ValueError("PLE-V54 validation changed row lacks paired scene")
        nlls = answer_nlls(
            bundle,
            tuple((bundle.prefixes[str(row.paired_scene_id)], row) for row in selected),
        )
        for row, value in zip(selected, nlls.tolist(), strict=True):
            wrong[(row.scene_id, row.question_id)] = float(value)
    margins = {
        key: wrong[key] - correct[key]
        for key in wrong
    }
    units: defaultdict[tuple[str, str], list[float]] = defaultdict(list)
    for row in changed:
        assert row.pair_id is not None and row.pair_question_key is not None
        units[(row.pair_id, row.pair_question_key)].append(
            margins[(row.scene_id, row.question_id)]
        )
    if len(units) != 26 or any(len(values) != 2 for values in units.values()):
        raise ValueError("PLE-V54 validation changed-unit inventory changed")
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
        "correct_nll_sha256": _canonical_hash(correct),
        "changed_margin_sha256": _canonical_hash(margins),
    }


@torch.inference_mode()
def evaluate_retention(
    bundle: ReaderBundle,
    corpus: Sequence[Mapping[str, str]],
    teachers: Sequence[torch.Tensor],
) -> dict[str, Any]:
    if len(corpus) != len(teachers):
        raise ValueError("PLE-V54 retention teacher count changed")
    ce_increases: list[float] = []
    kls: list[float] = []
    agreements: list[bool] = []
    for row, teacher in zip(corpus, teachers, strict=True):
        current = retention_logits(bundle, row["prompt"]).cpu()
        target = _retention_target_id(bundle, row["continuation"])
        baseline_ce = float(F.cross_entropy(teacher, torch.tensor([target])).item())
        current_ce = float(F.cross_entropy(current, torch.tensor([target])).item())
        teacher_probabilities = torch.softmax(teacher.float(), dim=-1)
        kl = F.kl_div(
            torch.log_softmax(current.float(), dim=-1),
            teacher_probabilities,
            reduction="batchmean",
        )
        ce_increases.append(current_ce - baseline_ce)
        kls.append(float(kl.item()))
        agreements.append(int(current.argmax()) == int(teacher.argmax()))
    return {
        "example_count": len(corpus),
        "mean_ce_increase_nats": sum(ce_increases) / len(ce_increases),
        "maximum_ce_increase_nats": max(ce_increases),
        "mean_kl_nats": sum(kls) / len(kls),
        "maximum_kl_nats": max(kls),
        "next_token_top1_agreement": sum(agreements) / len(agreements),
        "metrics_sha256": _canonical_hash(
            {"ce_increases": ce_increases, "kls": kls, "agreements": agreements}
        ),
    }


def _greedy_subset(rows: Sequence[ReaderRecord]) -> list[ReaderRecord]:
    by_scene: defaultdict[str, list[ReaderRecord]] = defaultdict(list)
    for row in rows:
        by_scene[row.scene_id].append(row)
    selected = [row for scene_id in VALIDATION_SCENES for row in by_scene[scene_id][:6]]
    if len(selected) != 96:
        raise ValueError("PLE-V54 greedy validation subset changed")
    return selected


def _baseline_prediction_index() -> dict[tuple[str, str], str]:
    result: dict[tuple[str, str], str] = {}
    for line in _resolve(BASELINE_PREDICTIONS).read_text(encoding="utf-8").splitlines():
        value = json.loads(line)
        key = (str(value["scene_id"]), str(value["question_id"]))
        result[key] = str(value["predicted_answer"])
    if len(result) != 384:
        raise ValueError("PLE-V54 baseline prediction inventory changed")
    return result


def _eos_ids(bundle: ReaderBundle) -> int | list[int] | None:
    values: list[int] = []
    for candidate in (
        getattr(bundle.language.tokenizer, "eos_token_id", None),
        getattr(getattr(bundle.language.model, "generation_config", None), "eos_token_id", None),
    ):
        if candidate is None:
            continue
        if isinstance(candidate, Sequence) and not isinstance(candidate, (str, bytes)):
            values.extend(int(value) for value in candidate)
        else:
            values.append(int(candidate))
    unique = sorted(set(values))
    return None if not unique else unique[0] if len(unique) == 1 else unique


@torch.inference_mode()
def evaluate_greedy(bundle: ReaderBundle, rows: Sequence[ReaderRecord]) -> dict[str, Any]:
    selected = _greedy_subset(rows)
    baseline = _baseline_prediction_index()
    candidate_correct = 0
    baseline_correct = 0
    prediction_hashes: list[dict[str, str]] = []
    bundle.installation.eval()
    bundle.language.decoder_module.eval()
    model_dtype = next(bundle.language.model.parameters()).dtype
    for row in selected:
        prompt = prompt_token_ids(
            bundle.language.tokenizer,
            str(bundle.runtime.config["language"]["system_prompt"]),
            row.question,
            bundle.language.device,
        )
        prefix = bundle.prefixes[row.scene_id].to(bundle.language.device, dtype=model_dtype)
        generated = bundle.language.generate_from_scene_prefix(
            prefix,
            prompt,
            max_new_tokens=int(bundle.runtime.config["language"]["max_answer_tokens"]),
            eos_token_ids=_eos_ids(bundle),
            scene_prefix_after_bos=scene_prefix_after_bos_setting(bundle.runtime.config),
            scene_boundary_mode=scene_boundary_mode_setting(bundle.runtime.config),
            fallback=generate_from_embeddings,
        )
        decoded = bundle.language.tokenizer.decode(
            generated[0].detach().cpu().tolist(), skip_special_tokens=True
        ).strip() or "unknown"
        key = (row.scene_id, row.question_id)
        candidate_correct += canonical_type_specific_match(
            row.answer_type, decoded, row.answer
        )
        baseline_correct += canonical_type_specific_match(
            row.answer_type, baseline[key], row.answer
        )
        prediction_hashes.append(
            {
                "scene_id": row.scene_id,
                "question_id": row.question_id,
                "normalized_prediction_sha256": hashlib.sha256(
                    normalize_answer(decoded).encode()
                ).hexdigest(),
                "prefix_sha256": prefix_sha256(prefix),
            }
        )
    return {
        "row_count": len(selected),
        "baseline_exact_correct": baseline_correct,
        "baseline_exact_accuracy": baseline_correct / len(selected),
        "candidate_exact_correct": candidate_correct,
        "candidate_exact_accuracy": candidate_correct / len(selected),
        "exact_accuracy_delta": (candidate_correct - baseline_correct) / len(selected),
        "prediction_hashes_sha256": _canonical_hash(prediction_hashes),
        "question_dependent_scene_retrieval": False,
    }


def _publish_checkpoint(bundle: ReaderBundle, report_summary: Mapping[str, Any]) -> dict[str, Any]:
    destination = _resolve(OUTPUT_CHECKPOINT)
    if destination.exists() or destination.is_symlink():
        raise FileExistsError("PLE-V54 checkpoint target already exists")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{destination.name}.", dir=destination.parent))
    try:
        weights = temporary / "adapter.safetensors"
        save_file(
            {
                key: tensor.detach().float().cpu().contiguous()
                for key, tensor in bundle.installation.state_module.state_dict().items()
            },
            weights,
        )
        metadata = {
            "schema_version": 1,
            "artifact": ARTIFACT,
            "base_checkpoint_sha256": _EXPECTED_BASE_FINGERPRINT,
            "base_runtime_config_effective_sha256": _EXPECTED_RUNTIME_EFFECTIVE_SHA256,
            "model_id": "google/gemma-4-E2B-it",
            "model_revision": "3e22461f65e89153144f8adb70e3b8c2cc9845a7",
            "fixed_prefix_tokens": 258,
            "scene_latents": 256,
            "scene_hidden_dimension": 1536,
            "prefix_computed_before_question": True,
            "question_dependent_scene_retrieval": False,
            "environmental_text_inputs": [],
            "oracle_runtime_access": False,
            "target_module": "model.language_model.per_layer_model_projection",
            "rank": 4,
            "alpha": 8.0,
            "dropout": 0.0,
            "trainable_parameter_count": 41_984,
            "adapter_state_sha256": bundle.installation.state_sha256(),
            "adapter_file_sha256": sha256_file(weights),
            "selection_summary_sha256": _canonical_hash(report_summary),
            "preregistration_sha256": sha256_file(PREREGISTRATION),
        }
        metadata_path = temporary / "runtime_metadata.json"
        metadata_path.write_text(
            json.dumps(metadata, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        os.rename(temporary, destination)
        return {
            "path": str(destination.relative_to(PROJECT_ROOT)),
            "adapter_file_sha256": metadata["adapter_file_sha256"],
            "runtime_metadata_sha256": sha256_file(destination / "runtime_metadata.json"),
            "adapter_state_sha256": metadata["adapter_state_sha256"],
        }
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def train_and_gate() -> dict[str, Any]:
    if _resolve(RESULT_REPORT).exists() or _resolve(OUTPUT_CHECKPOINT).exists():
        raise FileExistsError("PLE-V54 terminal result or checkpoint already exists")
    smoke = _read_json(SMOKE_REPORT)
    if smoke.get("passed") is not True or smoke.get("preregistration_sha256") != sha256_file(
        PREREGISTRATION
    ):
        raise ValueError("PLE-V54 passed, matching gradient smoke is required")
    preflight = structural_preflight()
    if preflight["passed"] is not True:
        raise RuntimeError("PLE-V54 structural preflight failed")
    started = time.perf_counter()
    torch.manual_seed(_SEED)
    random.seed(_SEED)
    bundle = _load_bundle(gradient_checkpointing=True)
    train_rows = load_training_records()
    validation_rows = load_validation_records()
    corpus = load_retention_corpus()
    teachers = retention_baseline(bundle, corpus)
    baseline_teacher = evaluate_teacher_forcing(bundle, validation_rows)
    baseline_retention = evaluate_retention(bundle, corpus, teachers)
    schedule = build_schedule(train_rows)
    optimizer = torch.optim.AdamW(
        bundle.installation.parameters(), lr=0.0003, weight_decay=0.0
    )
    training_trace: list[dict[str, Any]] = []
    maximum_gradient_norm = 0.0
    bundle.installation.train()
    for update, rows in enumerate(schedule, start=1):
        optimizer.zero_grad(set_to_none=True)
        losses: list[float] = []
        margins: list[float] = []
        for row in rows[:2]:
            loss, diagnostics = changed_side_loss(bundle, row)
            (loss / _GRADIENT_ACCUMULATION).backward()
            losses.append(float(loss.detach().cpu()))
            margins.append(diagnostics["margin"])
        for row in rows[2:]:
            loss = broad_loss(bundle, row)
            (loss / _GRADIENT_ACCUMULATION).backward()
            losses.append(float(loss.detach().cpu()))
        retention_index = (update - 1) % len(corpus)
        retention_loss = retention_kl_loss(
            bundle, corpus[retention_index], teachers[retention_index]
        )
        (0.2 * retention_loss).backward()
        gradient_norm = float(
            torch.nn.utils.clip_grad_norm_(bundle.installation.parameters(), 1.0).detach().cpu()
        )
        if not math.isfinite(gradient_norm) or gradient_norm <= 0.0:
            raise RuntimeError("PLE-V54 gradient norm is invalid")
        maximum_gradient_norm = max(maximum_gradient_norm, gradient_norm)
        optimizer.step()
        bundle.installation.validate_state()
        training_trace.append(
            {
                "update": update,
                "mean_qa_loss": sum(losses) / len(losses),
                "mean_preupdate_changed_margin": sum(margins) / len(margins),
                "retention_kl": float(retention_loss.detach().cpu()),
                "preclip_gradient_l2": gradient_norm,
                "adapter_state_sha256": bundle.installation.state_sha256(),
            }
        )
        print(
            json.dumps(
                {
                    "phase": "ple_v54_train",
                    "update": update,
                    "updates": len(schedule),
                    "qa_loss": training_trace[-1]["mean_qa_loss"],
                    "margin": training_trace[-1]["mean_preupdate_changed_margin"],
                    "retention_kl": training_trace[-1]["retention_kl"],
                    "gradient_l2": gradient_norm,
                },
                sort_keys=True,
            ),
            flush=True,
        )
    bundle.installation.eval()
    candidate_teacher = evaluate_teacher_forcing(bundle, validation_rows)
    candidate_retention = evaluate_retention(bundle, corpus, teachers)
    teacher_checks = {
        "validation_answer_nll_improvement": (
            baseline_teacher["answer_nll_mean"] - candidate_teacher["answer_nll_mean"]
            >= 0.03
        ),
        "changed_wrong_prefix_positive_margin_rate": (
            candidate_teacher["changed_positive_margin_rate"] >= 0.65
        ),
        "changed_wrong_prefix_positive_margin_rate_delta": (
            candidate_teacher["changed_positive_margin_rate"]
            - baseline_teacher["changed_positive_margin_rate"]
            >= 0.10
        ),
        "changed_pair_complete_unit_delta": (
            candidate_teacher["changed_complete_units"]
            - baseline_teacher["changed_complete_units"]
            >= 3
        ),
        "retention_mean_ce_increase": (
            candidate_retention["mean_ce_increase_nats"] <= 0.03
        ),
        "retention_mean_kl": candidate_retention["mean_kl_nats"] <= 0.02,
        "retention_next_token_top1_agreement": (
            candidate_retention["next_token_top1_agreement"] >= 0.98
        ),
    }
    greedy = None
    if all(teacher_checks.values()):
        greedy = evaluate_greedy(bundle, validation_rows)
    greedy_check = greedy is not None and greedy["exact_accuracy_delta"] >= 0.02
    checks = {**teacher_checks, "greedy_exact_accuracy_delta": bool(greedy_check)}
    passed = all(checks.values())
    selection_summary = {
        "baseline_teacher": baseline_teacher,
        "candidate_teacher": candidate_teacher,
        "baseline_retention": baseline_retention,
        "candidate_retention": candidate_retention,
        "greedy": greedy,
        "checks": checks,
        "passed": passed,
    }
    checkpoint = _publish_checkpoint(bundle, selection_summary) if passed else None
    report = {
        "schema_version": 1,
        "artifact": f"{ARTIFACT}_terminal_result",
        "status": "passed_checkpoint_published" if passed else "failed_no_checkpoint",
        "passed": passed,
        "promotion_eligible": passed,
        "checkpoint_published": checkpoint is not None,
        "checkpoint": checkpoint,
        "preregistration_sha256": sha256_file(PREREGISTRATION),
        "smoke_report_sha256": sha256_file(SMOKE_REPORT),
        "training": {
            "updates": len(schedule),
            "gradient_accumulation": _GRADIENT_ACCUMULATION,
            "trainable_parameter_count": bundle.installation.parameter_count,
            "maximum_preclip_gradient_l2": maximum_gradient_norm,
            "initial_trace": training_trace[:3],
            "final_trace": training_trace[-3:],
            "trace_sha256": _canonical_hash(training_trace),
            "final_adapter_state_sha256": bundle.installation.state_sha256(),
        },
        "selection": selection_summary,
        "fixed_prefix": {
            "shape": [1, 258, 1536],
            "computed_before_question": True,
            "same_prefix_for_unchanged_scene": True,
            "question_dependent_retrieval": False,
            "all_scene_latents_present": True,
        },
        "runtime_leakage": {
            "environmental_text_inputs": [],
            "oracle_runtime_access": False,
            "training_qa_not_in_runtime_checkpoint": True,
            "validation_answers_not_in_runtime_checkpoint": True,
        },
        "elapsed_seconds": time.perf_counter() - started,
        "memory": memory_metrics(),
    }
    _atomic_create_json(RESULT_REPORT, report)
    if not passed and _resolve(OUTPUT_CHECKPOINT).exists():
        raise RuntimeError("PLE-V54 failed run unexpectedly published a checkpoint")
    return report


def authenticate_result() -> dict[str, Any]:
    prereg = authenticate_preregistration(PREREGISTRATION)
    report = _read_json(RESULT_REPORT)
    passed = report.get("passed") is True
    checkpoint_exists = _resolve(OUTPUT_CHECKPOINT).is_dir()
    if report.get("artifact") != f"{ARTIFACT}_terminal_result":
        raise ValueError("PLE-V54 terminal artifact changed")
    if passed != checkpoint_exists or report.get("checkpoint_published") != passed:
        raise ValueError("PLE-V54 checkpoint publication does not match terminal result")
    return {
        "artifact": ARTIFACT,
        "status": report["status"],
        "passed": passed,
        "checkpoint_exists": checkpoint_exists,
        "result_report_sha256": sha256_file(RESULT_REPORT),
        "preregistration": prereg,
        "failed_checks": [
            name for name, value in report["selection"]["checks"].items() if not value
        ],
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "mode", choices=("preflight", "smoke", "train", "authenticate")
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    mode = _parser().parse_args(argv).mode
    result = {
        "preflight": structural_preflight,
        "smoke": gradient_smoke,
        "train": train_and_gate,
        "authenticate": authenticate_result,
    }[mode]()
    print(json.dumps(result, sort_keys=True, allow_nan=False), flush=True)
    return 0 if result.get("passed", True) else 1


__all__ = [
    "ReaderRecord",
    "answer_nlls",
    "authenticate_result",
    "build_schedule",
    "changed_side_loss",
    "gradient_smoke",
    "load_prefixes",
    "load_retention_corpus",
    "load_training_records",
    "load_validation_records",
    "memory_metrics",
    "structural_preflight",
    "train_and_gate",
]


if __name__ == "__main__":
    raise SystemExit(main())
