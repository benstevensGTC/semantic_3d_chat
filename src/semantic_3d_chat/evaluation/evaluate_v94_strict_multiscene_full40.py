"""Held-out V94 evaluation with a hard question/label process boundary.

The command is deliberately split into three phases:

``compile-memory``
    Opens only sanitized numeric maps and frozen numeric model artifacts.  It
    builds all six complete ``[1, 738, 1536]`` memories before any question is
    opened and writes a content-addressed numeric-only cache.
``predict``
    Loads and binds every cached memory first, then opens the frozen three-field
    question manifest.  It compares the exact V85 parent and fixed V94 bridge,
    plus paired-wrong, zero-payload, and shuffled-atlas controls.  Reference
    answers are blocked at the file-open boundary.
``score``
    Is the sole phase allowed to open the pinned answer-bearing validation
    JSONL.  It computes structured metrics and, unless disabled, answer-token
    NLL for the causal controls without serializing answers into predictions.

No phase performs question-dependent retrieval or environmental text
conversion.  The complete fixed memory is supplied through Gemma's native
continuous image-prefix path for every arm.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import math
import os
import shutil
import tempfile
import time
from collections import Counter, defaultdict
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Final

import torch
from safetensors.torch import load_file, save_file

from semantic_3d_chat.chat.file_audit import FileAccessAudit
from semantic_3d_chat.chat.runtime import StaticChatRuntime
from semantic_3d_chat.chat.runtime_config import load_runtime_config
from semantic_3d_chat.config import PROJECT_ROOT, project_path
from semantic_3d_chat.evaluation.baseline_io import atomic_write_jsonl, read_jsonl
from semantic_3d_chat.evaluation.question_manifest import QuestionManifest
from semantic_3d_chat.evaluation.v56_fresh_development_score import (
    EXPECTED_CHANGED_SIDE_COUNT,
    EXPECTED_CHANGED_UNIT_COUNT,
    EXPECTED_TYPE_COUNTS,
    _changed_metrics,
    canonical_answer_key,
    canonical_type_specific_match,
)
from semantic_3d_chat.evaluation.v75_fixed_atlas_behavior import _load_probe_bank
from semantic_3d_chat.evaluation.v75_official_validation_contract import (
    DEFAULT_QUESTIONS_MANIFEST,
    DEFAULT_REFERENCES,
    EXPECTED_QUESTION_COUNT,
    EXPECTED_REFERENCE_SHA256,
    EXPECTED_SCENE_IDS,
    authenticate_v75_control_checkpoint,
    sha256_file,
    validate_official_question_manifest,
)
from semantic_3d_chat.language.local_lm import load_local_language_model
from semantic_3d_chat.language.lora import LoRABankCollection, install_lora_banks
from semantic_3d_chat.language.prefix_injection import prefix_sha256
from semantic_3d_chat.language.v81_structured_dense_atlas_sidecar import (
    ATLAS_MEMORY_TOKENS,
    HIDDEN_SIZE,
    split_v75_v2_prefix_v81,
)
from semantic_3d_chat.scene_encoder.fixed_prefix_atlas_v75 import (
    compile_fixed_scene_atlas_v75_v2,
)
from semantic_3d_chat.scene_encoder.map_io import load_map_tensors
from semantic_3d_chat.training.train_v84_strict_bridge import (
    _generate_v84,
    _measure_nll_v84,
)

ARTIFACT: Final[str] = "gemma4_v94_strict_multiscene_full40_validation_v1"
CACHE_ARTIFACT: Final[str] = "v94_question_independent_evaluation_memory_cache_v1"
PREDICTION_ARTIFACT: Final[str] = "v94_question_only_same216_predictions_v1"
SCORE_ARTIFACT: Final[str] = "v94_label_isolated_same216_score_v1"
DEFAULT_CONFIG: Final[Path] = Path(
    "configs/experiments/gemma4_v94_strict_multiscene_full40.yaml"
)
DEFAULT_CONTROLLER: Final[Path] = Path(
    "data_gemma4/runtime/checkpoints/gemma4_v75_nll_control_release_v1"
)
DEFAULT_PROBE_BANK: Final[Path] = Path(
    "reports/gemma4/artifacts/v75_fixed_atlas_historical_internal_v1/probe_bank"
)
EXPECTED_CONTROLLER_WEIGHTS_SHA256: Final[str] = (
    "bb112f42ca5df71b88b4cd7721b9107f9be9b0dc01b612a4ace6212548da669c"
)
EXPECTED_CONTROLLER_METADATA_SHA256: Final[str] = (
    "a45a192d27336329580612524d43f71f08e3f472e5fe833747ffc1395e2aa2be"
)
EXPECTED_PROBE_WEIGHTS_SHA256: Final[str] = (
    "fb32c687dd787f108fab03e9745eefb2273891c2be990d0acf50ca111eb637e8"
)
EXPECTED_PROBE_METADATA_SHA256: Final[str] = (
    "3e736940f4c83b55e96aa5e36f6774fd007454508722f5b25ddc44f298c2518d"
)
MEMORY_SHAPE: Final[tuple[int, int, int]] = (1, 738, HIDDEN_SIZE)
PAIR_SCENE: Final[dict[str, str]] = {
    "scene_000057": "scene_000058",
    "scene_000058": "scene_000057",
    "scene_000059": "scene_000060",
    "scene_000060": "scene_000059",
    "scene_000061": "scene_000062",
    "scene_000062": "scene_000061",
}
ARMS: Final[tuple[str, ...]] = (
    "v94",
    "v85_parent",
    "paired_wrong",
    "zero_payload",
    "shuffled_atlas",
)
_HEX64: Final[frozenset[str]] = frozenset("0123456789abcdef")
_CACHE_MANIFEST_FIELDS: Final[frozenset[str]] = frozenset(
    {
        "artifact",
        "schema_version",
        "scene_ids",
        "scene_count",
        "shape_each",
        "dtype",
        "compiled_before_questions",
        "question_inputs_used",
        "question_dependent_retrieval",
        "all_memory_slots_retained",
        "environmental_text_inputs",
        "source_runtime_config_sha256",
        "source_v85_adapter_sha256",
        "source_v85_metadata_sha256",
        "source_controller_weights_sha256",
        "source_controller_metadata_sha256",
        "source_probe_weights_sha256",
        "source_probe_metadata_sha256",
        "scenes",
    }
)
_CACHE_SCENE_FIELDS: Final[frozenset[str]] = frozenset(
    {"filename", "file_sha256", "file_size_bytes", "memory_sha256"}
)
_PREDICTION_FIELDS: Final[frozenset[str]] = frozenset(
    {
        "artifact",
        "scene_id",
        "question_id",
        "paired_scene_id",
        "v94_prediction",
        "v85_parent_prediction",
        "paired_wrong_prediction",
        "zero_payload_prediction",
        "shuffled_atlas_prediction",
        "memory_sha256",
        "paired_memory_sha256",
        "zero_memory_sha256",
        "shuffled_memory_sha256",
        "prefix_hash_unchanged",
        "elapsed_seconds",
        "provenance_sha256",
    }
)


def _resolve(path: str | Path) -> Path:
    value = Path(path).expanduser()
    return (value if value.is_absolute() else PROJECT_ROOT / value).resolve()


def _canonical_sha256(value: object) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _is_sha256(value: object) -> bool:
    return isinstance(value, str) and len(value) == 64 and set(value) <= _HEX64


def _strict_mapping(value: object, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"V94 {field} must be a mapping")
    return value


def _load_config(path: str | Path = DEFAULT_CONFIG) -> dict[str, Any]:
    """Use the sealed V94 validator when available, with a strict draft fallback."""

    try:
        from semantic_3d_chat.evaluation.v94_strict_multiscene_preflight import (
            load_config_v94,
        )
    except ImportError:
        import yaml

        source = _resolve(path)
        raw = yaml.safe_load(source.read_text(encoding="utf-8"))
        if not isinstance(raw, Mapping) or set(raw) != {"v94"}:
            raise ValueError("V94 config must contain exactly one v94 mapping")
        config = dict(_strict_mapping(raw["v94"], "config"))
    else:
        config = load_config_v94(path)
    if (
        config.get("schema_version") != 94
        or config.get("artifact")
        != "gemma4_v94_strict_multiscene_full40_direct_memory_lora_v1"
    ):
        raise ValueError("V94 experiment identity changed")
    evaluation = _strict_mapping(config.get("evaluation"), "evaluation contract")
    if (
        tuple(evaluation.get("scene_ids", ())) != EXPECTED_SCENE_IDS
        or evaluation.get("scene_count") != 6
        or evaluation.get("pair_count") != 3
        or evaluation.get("row_count") != EXPECTED_QUESTION_COUNT
        or evaluation.get("changed_unit_count") != EXPECTED_CHANGED_UNIT_COUNT
        or evaluation.get("changed_side_count") != EXPECTED_CHANGED_SIDE_COUNT
        or evaluation.get("labels_opened_by_memory_compiler") is not False
        or evaluation.get("labels_opened_by_question_only_predictor") is not False
        or evaluation.get("labels_opened_only_by_separate_scorer") is not True
    ):
        raise ValueError("V94 held-out evaluation contract changed")
    return config


def _forbidden_runtime_roots() -> list[Path]:
    roots = [PROJECT_ROOT / "data" / "oracle"]
    roots.extend(PROJECT_ROOT.glob("data*/oracle"))
    roots.extend(PROJECT_ROOT.glob("data*/qa"))
    return list(dict.fromkeys(path.resolve() for path in roots))


def _runtime_audit() -> FileAccessAudit:
    return FileAccessAudit(
        forbidden_roots=_forbidden_runtime_roots(),
        forbidden_component_names={"oracle"},
        block_forbidden=True,
    )


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(
            json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _tensor_dtype_name(value: torch.Tensor) -> str:
    return str(value.dtype).removeprefix("torch.")


def zero_environment_payload_v94(memory: torch.Tensor) -> torch.Tensor:
    """Zero exactly 736 continuous payload slots while preserving BOI and EOI."""

    if tuple(memory.shape) != MEMORY_SHAPE or not torch.isfinite(memory).all():
        raise ValueError("V94 zero control requires finite [1,738,1536] memory")
    result = memory.detach().clone()
    result[:, 1:-1].zero_()
    if (
        not torch.equal(result[:, :1], memory[:, :1])
        or not torch.equal(result[:, -1:], memory[:, -1:])
        or torch.count_nonzero(result[:, 1:-1]).item() != 0
    ):
        raise RuntimeError("V94 zero control did not preserve only native boundaries")
    return result


def shuffle_atlas_values_v94(memory: torch.Tensor) -> torch.Tensor:
    """Roll complete four-value groups across all 96 fixed probe keys."""

    banks = split_v75_v2_prefix_v81(memory)
    values = banks.atlas_values.roll(shifts=1, dims=1)
    atlas = torch.cat((banks.probe_keys.unsqueeze(2), values), dim=2).reshape(
        memory.shape[0], ATLAS_MEMORY_TOKENS, HIDDEN_SIZE
    )
    result = torch.cat((banks.boi, atlas, banks.base_latents, banks.eoi), dim=1)
    if (
        tuple(result.shape) != MEMORY_SHAPE
        or not torch.equal(result[:, :1], memory[:, :1])
        or not torch.equal(result[:, -1:], memory[:, -1:])
        or not torch.equal(
            split_v75_v2_prefix_v81(result).base_latents, banks.base_latents
        )
    ):
        raise RuntimeError("V94 shuffled-atlas control changed fixed geometry/boundaries")
    return result.detach()


class EvaluationRuntimePrefixFactoryV94:
    """Reuse one exact V85 stack while encoding all six untouched numeric maps."""

    def __init__(
        self,
        config: dict[str, Any],
        checkpoint: str | Path,
        *,
        audit: FileAccessAudit,
    ) -> None:
        self.config = config
        self.audit = audit
        self.bootstrap = StaticChatRuntime.load(
            config,
            EXPECTED_SCENE_IDS[0],
            checkpoint=_resolve(checkpoint),
            audit=audit,
            local_files_only=True,
        )

    def _map_data(self, scene_id: str) -> Any:
        if scene_id not in EXPECTED_SCENE_IDS:
            raise ValueError(f"V94 evaluation scene is outside the sealed split: {scene_id}")
        source = _resolve(project_path(self.config, "maps", scene_id, "voxel_map.npz"))
        if not source.is_file() or source.is_symlink():
            raise FileNotFoundError(f"V94 sanitized numeric map unavailable: {source}")
        self.audit.record(source)
        data = load_map_tensors(
            source,
            self.config["scene"]["room_size_m"],
            device="cpu",
            input_voxel_size_m=self.config["scene_encoder"].get(
                "input_voxel_size_m"
            ),
        )
        if data.feature_dim != int(self.bootstrap.checkpoint_metadata["semantic_dim"]):
            raise ValueError(f"V94 semantic map dimension changed: {scene_id}")
        return data.to(self.bootstrap.language.device)

    def load(self, scene_id: str) -> StaticChatRuntime:
        if scene_id == self.bootstrap.scene_id:
            self.bootstrap.assert_prefix_unchanged()
            return self.bootstrap
        return StaticChatRuntime(
            config=self.config,
            scene_id=scene_id,
            checkpoint_path=self.bootstrap.checkpoint_path,
            checkpoint_metadata=self.bootstrap.checkpoint_metadata,
            language=self.bootstrap.language,
            map_data=self._map_data(scene_id),
            scene_model=self.bootstrap.scene_model,
            dense_aligner=self.bootstrap.dense_aligner,
            dense_sidecar_adapter=self.bootstrap.dense_sidecar_adapter,
            block_cross_residual=self.bootstrap.block_cross_residual,
            global_scene_residual=self.bootstrap.global_scene_residual,
            signed_x_scene_residual=self.bootstrap.signed_x_scene_residual,
            composer=self.bootstrap.composer,
            grounding=self.bootstrap.grounding,
            warnings=self.bootstrap.warnings,
            generation_function=self.bootstrap._generation_function,
        )


def _source_path(config: Mapping[str, Any], field: str, fallback: Path) -> Path:
    sources = _strict_mapping(config.get("sources"), "sources")
    raw = sources.get(field, fallback)
    if not isinstance(raw, (str, Path)):
        raise TypeError(f"V94 source {field} must be a path")
    return _resolve(raw)


def _authenticate_numeric_compiler_sources(config: Mapping[str, Any]) -> dict[str, Path]:
    runtime = _source_path(config, "runtime_config", Path("configs/runtime/gemma4_v85_strict_multiscene.yaml"))
    checkpoint = _source_path(
        config, "frozen_v85_checkpoint", Path("reports/gemma4/artifacts/v85_strict_runtime_candidate")
    )
    controller = _source_path(
        config, "evaluation_memory_controller", DEFAULT_CONTROLLER
    )
    probes = _source_path(config, "evaluation_probe_bank", DEFAULT_PROBE_BANK)
    expected = {
        runtime: _strict_mapping(config["sources"], "sources").get("runtime_config_sha256"),
        checkpoint / "adapter.safetensors": _strict_mapping(config["sources"], "sources").get(
            "frozen_v85_adapter_sha256"
        ),
        checkpoint / "runtime_metadata.json": _strict_mapping(
            config["sources"], "sources"
        ).get("frozen_v85_metadata_sha256"),
        controller / "control.safetensors": EXPECTED_CONTROLLER_WEIGHTS_SHA256,
        controller / "runtime_metadata.json": EXPECTED_CONTROLLER_METADATA_SHA256,
        probes / "probes.safetensors": EXPECTED_PROBE_WEIGHTS_SHA256,
        probes / "runtime_metadata.json": EXPECTED_PROBE_METADATA_SHA256,
    }
    for path, digest in expected.items():
        if not path.is_file() or path.is_symlink() or sha256_file(path) != digest:
            raise ValueError(f"V94 numeric compiler source changed: {path}")
    return {
        "runtime": runtime,
        "checkpoint": checkpoint,
        "controller": controller,
        "probes": probes,
    }


def _cache_manifest(
    root: Path,
    *,
    expected_source_hashes: Mapping[str, str],
) -> dict[str, Any]:
    source = root / "manifest.json"
    if root.is_symlink() or not source.is_file() or source.is_symlink():
        raise FileNotFoundError(f"V94 evaluation memory cache is unavailable: {root}")
    raw = json.loads(source.read_text(encoding="utf-8"))
    manifest = dict(_strict_mapping(raw, "memory-cache manifest"))
    if set(manifest) != _CACHE_MANIFEST_FIELDS:
        raise ValueError("V94 memory-cache manifest fields changed")
    exact = {
        "artifact": CACHE_ARTIFACT,
        "schema_version": 1,
        "scene_ids": list(EXPECTED_SCENE_IDS),
        "scene_count": 6,
        "shape_each": list(MEMORY_SHAPE),
        "dtype": "bfloat16",
        "compiled_before_questions": True,
        "question_inputs_used": False,
        "question_dependent_retrieval": False,
        "all_memory_slots_retained": True,
        "environmental_text_inputs": [],
        **expected_source_hashes,
    }
    if any(manifest.get(field) != value for field, value in exact.items()):
        raise ValueError("V94 memory-cache provenance changed")
    scenes = _strict_mapping(manifest.get("scenes"), "memory-cache scenes")
    if set(scenes) != set(EXPECTED_SCENE_IDS):
        raise ValueError("V94 memory-cache scene inventory changed")
    return manifest


def load_evaluation_memory_cache_v94(
    cache_path: str | Path,
    *,
    expected_source_hashes: Mapping[str, str],
    audit: FileAccessAudit | None = None,
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    """Load exactly six numeric memories and authenticate every tensor byte."""

    root = _resolve(cache_path)
    manifest = _cache_manifest(root, expected_source_hashes=expected_source_hashes)
    expected_files = {"manifest.json", *(f"{scene}.safetensors" for scene in EXPECTED_SCENE_IDS)}
    if {item.name for item in root.iterdir()} != expected_files:
        raise ValueError("V94 memory cache has unexpected files")
    memories: dict[str, torch.Tensor] = {}
    for scene_id in EXPECTED_SCENE_IDS:
        entry = dict(_strict_mapping(manifest["scenes"][scene_id], "cache scene"))
        if set(entry) != _CACHE_SCENE_FIELDS or entry.get("filename") != f"{scene_id}.safetensors":
            raise ValueError("V94 cache filenames/fields changed")
        source = root / str(entry["filename"])
        if audit is not None:
            audit.record(source)
        if (
            not source.is_file()
            or source.is_symlink()
            or source.stat().st_size != entry.get("file_size_bytes")
            or sha256_file(source) != entry.get("file_sha256")
        ):
            raise ValueError(f"V94 cached memory bytes changed: {scene_id}")
        state = load_file(str(source), device="cpu")
        if set(state) != {"scene_memory"}:
            raise ValueError("V94 cache safetensors must contain only scene_memory")
        memory = state["scene_memory"].detach().contiguous()
        if (
            tuple(memory.shape) != MEMORY_SHAPE
            or memory.dtype != torch.bfloat16
            or not torch.isfinite(memory).all()
            or prefix_sha256(memory) != entry.get("memory_sha256")
        ):
            raise ValueError(f"V94 cached memory tensor changed: {scene_id}")
        memories[scene_id] = memory
    return memories, manifest


def compile_evaluation_memory_cache_v94(
    config_path: str | Path = DEFAULT_CONFIG,
) -> dict[str, Any]:
    """Compile all six complete numeric memories before any question file opens."""

    config = _load_config(config_path)
    sources = _authenticate_numeric_compiler_sources(config)
    outputs = _strict_mapping(config.get("outputs"), "outputs")
    root = _resolve(str(outputs["evaluation_memory_cache"]))
    source_hashes = {
        "source_runtime_config_sha256": sha256_file(sources["runtime"]),
        "source_v85_adapter_sha256": sha256_file(
            sources["checkpoint"] / "adapter.safetensors"
        ),
        "source_v85_metadata_sha256": sha256_file(
            sources["checkpoint"] / "runtime_metadata.json"
        ),
        "source_controller_weights_sha256": EXPECTED_CONTROLLER_WEIGHTS_SHA256,
        "source_controller_metadata_sha256": EXPECTED_CONTROLLER_METADATA_SHA256,
        "source_probe_weights_sha256": EXPECTED_PROBE_WEIGHTS_SHA256,
        "source_probe_metadata_sha256": EXPECTED_PROBE_METADATA_SHA256,
    }
    if root.exists():
        memories, manifest = load_evaluation_memory_cache_v94(
            root, expected_source_hashes=source_hashes
        )
        return {
            "created": False,
            "cache_path": str(root),
            "memory_hashes": {
                scene: prefix_sha256(memory) for scene, memory in memories.items()
            },
            "manifest_sha256": _canonical_sha256(manifest),
        }

    root.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{root.name}.", dir=root.parent))
    audit = _runtime_audit()
    started = time.monotonic()
    try:
        with audit:
            runtime_config = load_runtime_config(sources["runtime"])
            factory = EvaluationRuntimePrefixFactoryV94(
                runtime_config, sources["checkpoint"], audit=audit
            )
            from semantic_3d_chat.chat.question_control_runtime import _load_control_head

            controller_identity = authenticate_v75_control_checkpoint(
                sources["controller"]
            )
            if (
                controller_identity.weights_sha256
                != EXPECTED_CONTROLLER_WEIGHTS_SHA256
                or controller_identity.runtime_metadata_sha256
                != EXPECTED_CONTROLLER_METADATA_SHA256
            ):
                raise ValueError("V94 compiler did not authenticate the exact V75 controller")
            controller, _metadata = _load_control_head(
                sources["controller"],
                hidden_size=HIDDEN_SIZE,
                device=torch.device("cpu"),
                audit=audit,
            )
            probes, _probe_metadata = _load_probe_bank(sources["probes"], audit)
            entries: dict[str, Any] = {}
            for scene_id in EXPECTED_SCENE_IDS:
                runtime = factory.load(scene_id)
                runtime.assert_prefix_unchanged()
                compiled = compile_fixed_scene_atlas_v75_v2(
                    runtime.scene_prefix.detach().cpu(), controller, probes
                )
                memory = compiled.scene_prefix.to(torch.bfloat16).contiguous()
                if tuple(memory.shape) != MEMORY_SHAPE or not torch.isfinite(memory).all():
                    raise RuntimeError(f"V94 compiler produced invalid memory: {scene_id}")
                destination = temporary / f"{scene_id}.safetensors"
                save_file(
                    {"scene_memory": memory},
                    str(destination),
                    metadata={
                        "artifact": CACHE_ARTIFACT,
                        "question_inputs_used": "false",
                        "environmental_text_serialized": "false",
                    },
                )
                entries[scene_id] = {
                    "filename": destination.name,
                    "file_sha256": sha256_file(destination),
                    "file_size_bytes": destination.stat().st_size,
                    "memory_sha256": prefix_sha256(memory),
                }
                del runtime, compiled, memory
                if torch.backends.mps.is_available():
                    torch.mps.empty_cache()
            manifest = {
                "artifact": CACHE_ARTIFACT,
                "schema_version": 1,
                "scene_ids": list(EXPECTED_SCENE_IDS),
                "scene_count": 6,
                "shape_each": list(MEMORY_SHAPE),
                "dtype": "bfloat16",
                "compiled_before_questions": True,
                "question_inputs_used": False,
                "question_dependent_retrieval": False,
                "all_memory_slots_retained": True,
                "environmental_text_inputs": [],
                **source_hashes,
                "scenes": entries,
            }
            _atomic_json(temporary / "manifest.json", manifest)
            load_evaluation_memory_cache_v94(
                temporary, expected_source_hashes=source_hashes
            )
        audit.assert_clean()
        os.rename(temporary, root)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return {
        "created": True,
        "cache_path": str(root),
        "memory_hashes": {
            scene: str(entries[scene]["memory_sha256"]) for scene in EXPECTED_SCENE_IDS
        },
        "manifest_sha256": _canonical_sha256(manifest),
        "protected_read_count": len(audit.forbidden_accesses()),
        "elapsed_seconds": time.monotonic() - started,
    }


def _question_manifest_path(config: Mapping[str, Any]) -> Path:
    outputs = _strict_mapping(config.get("outputs"), "outputs")
    configured = outputs.get("evaluation_question_manifest")
    if isinstance(configured, str) and _resolve(configured).is_file():
        return _resolve(configured)
    return _resolve(DEFAULT_QUESTIONS_MANIFEST)


def _reference_path(config: Mapping[str, Any]) -> Path:
    sources = _strict_mapping(config.get("sources"), "sources")
    value = sources.get("evaluation_qa_reserved_for_label_scorer", DEFAULT_REFERENCES)
    if not isinstance(value, (str, Path)):
        raise TypeError("V94 scorer reference path must be a path")
    return _resolve(value)


def _load_bound_memories_before_questions(
    config: Mapping[str, Any], audit: FileAccessAudit
) -> tuple[dict[str, torch.Tensor], dict[str, str], dict[str, Any]]:
    sources = _authenticate_numeric_compiler_sources(config)
    source_hashes = {
        "source_runtime_config_sha256": sha256_file(sources["runtime"]),
        "source_v85_adapter_sha256": sha256_file(
            sources["checkpoint"] / "adapter.safetensors"
        ),
        "source_v85_metadata_sha256": sha256_file(
            sources["checkpoint"] / "runtime_metadata.json"
        ),
        "source_controller_weights_sha256": EXPECTED_CONTROLLER_WEIGHTS_SHA256,
        "source_controller_metadata_sha256": EXPECTED_CONTROLLER_METADATA_SHA256,
        "source_probe_weights_sha256": EXPECTED_PROBE_WEIGHTS_SHA256,
        "source_probe_metadata_sha256": EXPECTED_PROBE_METADATA_SHA256,
    }
    cache = _resolve(str(_strict_mapping(config["outputs"], "outputs")["evaluation_memory_cache"]))
    memories, manifest = load_evaluation_memory_cache_v94(
        cache, expected_source_hashes=source_hashes, audit=audit
    )
    hashes = {scene: prefix_sha256(memory) for scene, memory in memories.items()}
    return memories, hashes, manifest


@dataclass(frozen=True)
class PredictorStackV94:
    language: Any
    collection: LoRABankCollection
    candidate: Mapping[str, Any]
    system_prompt: str
    max_new_tokens: int


def _load_predictor_stack_v94(config: Mapping[str, Any]) -> PredictorStackV94:
    from semantic_3d_chat.training.train_v94_strict_multiscene_full40 import (
        combined_lora_settings_v94,
        load_fixed_final_bridge_v94,
        load_frozen_v85_stack_v94,
    )

    runtime = load_runtime_config(str(_strict_mapping(config["sources"], "sources")["runtime_config"]))
    language_config = _strict_mapping(runtime["language"], "language config")
    language = load_local_language_model(
        str(language_config["model_id"]),
        str(language_config["revision"]),
        str(language_config["dtype"]),
        freeze=True,
        local_files_only=True,
        backend="gemma4",
        decoder_gradient_checkpointing=False,
    )
    collection = install_lora_banks(
        language.model, combined_lora_settings_v94(runtime, config)
    )
    if not isinstance(collection, LoRABankCollection):
        raise TypeError("V94 predictor LoRA installation failed")
    load_frozen_v85_stack_v94(
        collection, str(_strict_mapping(config["sources"], "sources")["frozen_v85_checkpoint"])
    )
    candidate = load_fixed_final_bridge_v94(
        collection, str(_strict_mapping(config["outputs"], "outputs")["fixed_final_candidate"])
    )
    collection.eval()
    language.model.eval()
    language.decoder_module.eval()
    return PredictorStackV94(
        language=language,
        collection=collection,
        candidate=candidate,
        system_prompt=str(language_config["system_prompt"]),
        max_new_tokens=int(language_config["max_answer_tokens"]),
    )


@contextlib.contextmanager
def v85_parent_bank_only_v94(collection: LoRABankCollection) -> Iterator[None]:
    """Disable only the fresh V94 delta, restoring it bit-for-bit afterward."""

    from semantic_3d_chat.training.train_v94_strict_multiscene_full40 import (
        FRESH_BANK_NAME,
    )

    installation = collection.bank(FRESH_BANK_NAME).installation
    if len(installation.adapters) != 1:
        raise ValueError("V94 parent comparator expects exactly one fresh adapter")
    adapter = installation.adapters[0]
    saved = adapter.lora_b.detach().clone()
    before = installation.state_sha256()
    try:
        with torch.no_grad():
            adapter.lora_b.zero_()
        yield
    finally:
        with torch.no_grad():
            adapter.lora_b.copy_(saved)
        if installation.state_sha256() != before:
            raise RuntimeError("V94 fresh bank did not restore after V85 comparator")


def _question_row(question: str, answer: str = "unknown") -> Any:
    return SimpleNamespace(question=question, answer=answer)


def _generate_arm(stack: PredictorStackV94, memory: torch.Tensor, question: str) -> str:
    return _generate_v84(
        stack.language,
        stack.system_prompt,
        memory,
        _question_row(question),
        max_new_tokens=stack.max_new_tokens,
    )


def _prediction_provenance(
    config_path: str | Path,
    config: Mapping[str, Any],
    manifest: QuestionManifest,
    memory_manifest: Mapping[str, Any],
    candidate: Mapping[str, Any],
) -> dict[str, Any]:
    value = {
        "artifact": PREDICTION_ARTIFACT,
        "schema_version": 1,
        "config_sha256": sha256_file(_resolve(config_path)),
        "question_manifest_sha256": manifest.manifest_sha256,
        "questions_sha256": manifest.questions_sha256,
        "memory_manifest_sha256": _canonical_sha256(memory_manifest),
        "candidate_weights_sha256": candidate.get("weights_sha256"),
        "candidate_state_sha256": candidate.get("state_sha256"),
        "scene_ids": list(EXPECTED_SCENE_IDS),
        "row_count": EXPECTED_QUESTION_COUNT,
        "arms": list(ARMS),
        "labels_opened": False,
        "questions_opened_after_all_memories_bound": True,
        "question_dependent_retrieval": False,
        "environmental_text_inputs": [],
    }
    value["provenance_sha256"] = _canonical_sha256(value)
    return value


def predict_question_only_v94(config_path: str | Path = DEFAULT_CONFIG) -> dict[str, Any]:
    """Run all same-row comparator/control arms without opening reference labels."""

    audit = _runtime_audit()
    started = time.monotonic()
    with audit:
        config = _load_config(config_path)
        output = _resolve(
            str(_strict_mapping(config["outputs"], "outputs")["evaluation_predictions"])
        )
        provenance_path = output.with_name(f"{output.name}.provenance.json")
        access_path = output.with_name(f"{output.name}.access.json")
        # This call must remain before validate_official_question_manifest.
        memories, hashes_before, memory_manifest = _load_bound_memories_before_questions(
            config, audit
        )
        controls = {
            scene: {
                "zero": zero_environment_payload_v94(memory),
                "shuffled": shuffle_atlas_values_v94(memory),
            }
            for scene, memory in memories.items()
        }
        questions = validate_official_question_manifest(_question_manifest_path(config))
        stack = _load_predictor_stack_v94(config)
        provenance = _prediction_provenance(
            config_path, config, questions, memory_manifest, stack.candidate
        )
        if provenance_path.exists():
            stored = json.loads(provenance_path.read_text(encoding="utf-8"))
            if stored != provenance:
                raise RuntimeError("V94 prediction resume provenance changed")
            existing = read_jsonl(output)
        else:
            _atomic_json(provenance_path, provenance)
            existing = []
        indexed = {
            (str(row["scene_id"]), str(row["question_id"])): row for row in existing
        }
        if len(indexed) != len(existing):
            raise ValueError("V94 prediction resume contains duplicate keys")
        records = list(existing)
        for ordinal, record in enumerate(questions.questions, 1):
            key = (record.scene_id, record.question_id)
            if key in indexed:
                continue
            memory = memories[record.scene_id]
            paired_scene = PAIR_SCENE[record.scene_id]
            before = prefix_sha256(memory)
            prediction = _generate_arm(stack, memory, record.question)
            with v85_parent_bank_only_v94(stack.collection):
                parent_prediction = _generate_arm(stack, memory, record.question)
            paired_prediction = _generate_arm(
                stack, memories[paired_scene], record.question
            )
            zero_prediction = _generate_arm(
                stack, controls[record.scene_id]["zero"], record.question
            )
            shuffled_prediction = _generate_arm(
                stack, controls[record.scene_id]["shuffled"], record.question
            )
            after = prefix_sha256(memory)
            row = {
                "artifact": PREDICTION_ARTIFACT,
                "scene_id": record.scene_id,
                "question_id": record.question_id,
                "paired_scene_id": paired_scene,
                "v94_prediction": prediction,
                "v85_parent_prediction": parent_prediction,
                "paired_wrong_prediction": paired_prediction,
                "zero_payload_prediction": zero_prediction,
                "shuffled_atlas_prediction": shuffled_prediction,
                "memory_sha256": before,
                "paired_memory_sha256": hashes_before[paired_scene],
                "zero_memory_sha256": prefix_sha256(
                    controls[record.scene_id]["zero"]
                ),
                "shuffled_memory_sha256": prefix_sha256(
                    controls[record.scene_id]["shuffled"]
                ),
                "prefix_hash_unchanged": before == after == hashes_before[record.scene_id],
                "elapsed_seconds": time.monotonic() - started,
                "provenance_sha256": provenance["provenance_sha256"],
            }
            records.append(row)
            atomic_write_jsonl(output, records)
            if ordinal == 1 or ordinal % 12 == 0 or ordinal == EXPECTED_QUESTION_COUNT:
                print(
                    json.dumps(
                        {
                            "event": "v94_question_only_prediction",
                            "ordinal": ordinal,
                            "total": EXPECTED_QUESTION_COUNT,
                            "scene_id": record.scene_id,
                            "question_id": record.question_id,
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()
        hashes_after = {scene: prefix_sha256(memory) for scene, memory in memories.items()}
    audit.assert_clean()
    if hashes_after != hashes_before or len(records) != EXPECTED_QUESTION_COUNT:
        raise RuntimeError("V94 prediction did not preserve all six fixed memories/rows")
    audit.save(access_path)
    return {
        "artifact": PREDICTION_ARTIFACT,
        "prediction_path": str(output),
        "prediction_sha256": sha256_file(output),
        "provenance_sha256": provenance["provenance_sha256"],
        "row_count": len(records),
        "scene_count": len(memories),
        "prefix_hash_invariant": hashes_before == hashes_after,
        "protected_read_count": len(audit.forbidden_accesses()),
        "labels_opened": False,
    }


def _validate_predictions(
    path: Path,
    manifest: QuestionManifest,
) -> tuple[list[dict[str, Any]], Mapping[str, Any]]:
    provenance_path = path.with_name(f"{path.name}.provenance.json")
    provenance = dict(
        _strict_mapping(
            json.loads(provenance_path.read_text(encoding="utf-8")),
            "prediction provenance",
        )
    )
    supplied_sha = provenance.pop("provenance_sha256", None)
    if supplied_sha != _canonical_sha256(provenance):
        raise ValueError("V94 prediction provenance digest changed")
    provenance["provenance_sha256"] = supplied_sha
    records = read_jsonl(path)
    expected_keys = {(row.scene_id, row.question_id) for row in manifest.questions}
    observed: set[tuple[str, str]] = set()
    by_scene: defaultdict[str, set[str]] = defaultdict(set)
    for record in records:
        if set(record) != _PREDICTION_FIELDS:
            raise ValueError("V94 question-only prediction fields changed")
        key = (str(record["scene_id"]), str(record["question_id"]))
        if key in observed:
            raise ValueError(f"Duplicate V94 prediction key: {key}")
        observed.add(key)
        if (
            record.get("artifact") != PREDICTION_ARTIFACT
            or record.get("paired_scene_id") != PAIR_SCENE.get(key[0])
            or record.get("provenance_sha256") != supplied_sha
            or record.get("prefix_hash_unchanged") is not True
            or not all(
                _is_sha256(record.get(field))
                for field in (
                    "memory_sha256",
                    "paired_memory_sha256",
                    "zero_memory_sha256",
                    "shuffled_memory_sha256",
                )
            )
        ):
            raise ValueError(f"V94 prediction contract changed: {key}")
        by_scene[key[0]].add(str(record["memory_sha256"]))
    if (
        len(records) != EXPECTED_QUESTION_COUNT
        or observed != expected_keys
        or set(by_scene) != set(EXPECTED_SCENE_IDS)
        or any(len(values) != 1 for values in by_scene.values())
    ):
        raise ValueError("V94 predictions lack exact coverage/prefix invariance")
    return records, provenance


def _load_references_for_scorer(
    config: Mapping[str, Any], manifest: QuestionManifest
) -> list[dict[str, Any]]:
    source = _reference_path(config)
    if source.is_symlink() or not source.is_file() or sha256_file(source) != EXPECTED_REFERENCE_SHA256:
        raise ValueError("V94 scorer labels differ from pinned validation bytes")
    records = read_jsonl(source)
    question_by_key = {
        (row.scene_id, row.question_id): row.question for row in manifest.questions
    }
    if len(records) != EXPECTED_QUESTION_COUNT:
        raise ValueError("V94 scorer requires exactly 216 reference rows")
    type_counts: Counter[str] = Counter()
    seen: set[tuple[str, str]] = set()
    for row in records:
        key = (str(row.get("scene_id")), str(row.get("question_id")))
        if (
            key in seen
            or row.get("question") != question_by_key.get(key)
            or not isinstance(row.get("answer"), str)
            or not isinstance(row.get("answer_type"), str)
        ):
            raise ValueError(f"V94 reference projection changed: {key}")
        seen.add(key)
        type_counts[str(row["answer_type"])] += 1
    if seen != set(question_by_key) or dict(sorted(type_counts.items())) != EXPECTED_TYPE_COUNTS:
        raise ValueError("V94 reference coverage/type inventory changed")
    return records


def _accuracy_for_arm(
    references: Sequence[Mapping[str, Any]],
    predictions: Mapping[tuple[str, str], Mapping[str, Any]],
    field: str,
) -> dict[str, Any]:
    by_type: defaultdict[str, list[bool]] = defaultdict(list)
    scored: list[bool] = []
    for reference in references:
        key = (str(reference["scene_id"]), str(reference["question_id"]))
        correct = canonical_type_specific_match(
            str(reference["answer_type"]),
            predictions[key][field],
            reference["answer"],
        )
        scored.append(correct)
        by_type[str(reference["answer_type"])].append(correct)
    return {
        "correct": sum(scored),
        "total": len(scored),
        "accuracy": sum(scored) / len(scored),
        "by_answer_type": {
            answer_type: {
                "correct": sum(values),
                "total": len(values),
                "accuracy": sum(values) / len(values),
            }
            for answer_type, values in sorted(by_type.items())
        },
    }


def _prediction_change_count(
    references: Sequence[Mapping[str, Any]],
    predictions: Mapping[tuple[str, str], Mapping[str, Any]],
    left_field: str,
    right_field: str,
) -> int:
    return sum(
        canonical_answer_key(str(reference["answer_type"]), predictions[
            (str(reference["scene_id"]), str(reference["question_id"]))
        ][left_field])
        != canonical_answer_key(str(reference["answer_type"]), predictions[
            (str(reference["scene_id"]), str(reference["question_id"]))
        ][right_field])
        for reference in references
    )


def score_records_v94(
    references: Sequence[Mapping[str, Any]],
    prediction_records: Sequence[Mapping[str, Any]],
    *,
    gates: Mapping[str, Any],
    nll_metrics: Mapping[str, float] | None,
    protected_read_count: int,
) -> dict[str, Any]:
    predictions = {
        (str(row["scene_id"]), str(row["question_id"])): row
        for row in prediction_records
    }
    v94 = _accuracy_for_arm(references, predictions, "v94_prediction")
    v85 = _accuracy_for_arm(references, predictions, "v85_parent_prediction")
    zero = _accuracy_for_arm(references, predictions, "zero_payload_prediction")
    shuffled = _accuracy_for_arm(references, predictions, "shuffled_atlas_prediction")
    changed = _changed_metrics(
        references,
        {
            key: {"predicted_answer": row["v94_prediction"]}
            for key, row in predictions.items()
        },
    )
    zero_changes = _prediction_change_count(
        references, predictions, "v94_prediction", "zero_payload_prediction"
    )
    shuffled_changes = _prediction_change_count(
        references, predictions, "v94_prediction", "shuffled_atlas_prediction"
    )
    by_type = v94["by_answer_type"]
    nll = dict(nll_metrics or {})
    gate_results = {
        "canonical_accuracy_at_least_0_65": v94["accuracy"]
        >= float(gates["canonical_accuracy_minimum"]),
        "canonical_accuracy_at_least_v85_plus_0_05": v94["accuracy"]
        >= v85["accuracy"]
        + float(gates["canonical_accuracy_margin_over_exact_v85_same_216_comparator"]),
        **{
            f"{answer_type}_correct_minimum": by_type[answer_type]["correct"]
            >= int(gates[f"{answer_type}_correct_minimum"])
            for answer_type in EXPECTED_TYPE_COUNTS
        },
        "changed_side_correct_minimum": changed["canonical_correct_sides"]
        >= int(gates["changed_side_correct_minimum"]),
        "complete_changed_units_minimum": changed["canonical_complete_units"]
        >= int(gates["complete_changed_units_minimum"]),
        "prediction_changing_units_minimum": changed[
            "canonical_prediction_changed_units"
        ]
        >= int(gates["canonical_prediction_changing_units_minimum"]),
        "zero_payload_prediction_change_minimum": zero_changes
        >= int(gates["zero_payload_prediction_change_minimum"]),
        "mean_changed_wrong_minus_correct_nll_minimum": nll.get(
            "mean_changed_wrong_minus_correct_nll", -math.inf
        )
        >= float(gates["mean_changed_side_wrong_minus_correct_nll_minimum"]),
        "zero_payload_mean_nll_gap_minimum": nll.get(
            "zero_payload_mean_nll_gap", -math.inf
        )
        >= float(gates["zero_payload_mean_nll_gap_minimum"]),
        "correct_nll_below_zero_payload": nll.get(
            "zero_payload_mean_nll_gap", -math.inf
        )
        > 0.0,
        "correct_nll_below_shuffled_scene": nll.get(
            "shuffled_atlas_mean_nll_gap", -math.inf
        )
        > 0.0,
        "prefix_hash_invariance": all(
            row.get("prefix_hash_unchanged") is True for row in prediction_records
        ),
        "protected_read_count_zero": protected_read_count
        <= int(gates["protected_read_count_maximum"]),
    }
    return {
        "v94": v94,
        "exact_v85_same_216_comparator": v85,
        "zero_payload": zero,
        "shuffled_atlas": shuffled,
        "absolute_accuracy_delta_v94_minus_v85": v94["accuracy"] - v85["accuracy"],
        "counterfactual": changed,
        "zero_payload_prediction_change_count": zero_changes,
        "shuffled_atlas_prediction_change_count": shuffled_changes,
        "nll_controls": nll if nll_metrics is not None else {"measured": False},
        "runtime_candidate_gates": gate_results,
        "runtime_candidate_gate_passed": all(gate_results.values()),
        "automatic_runtime_promotion": False,
    }


@torch.inference_mode()
def _measure_nll_controls_v94(
    config: Mapping[str, Any],
    references: Sequence[Mapping[str, Any]],
    memories: Mapping[str, torch.Tensor],
) -> dict[str, float]:
    stack = _load_predictor_stack_v94(config)
    totals: defaultdict[str, list[float]] = defaultdict(list)
    for reference in references:
        scene_id = str(reference["scene_id"])
        paired = PAIR_SCENE[scene_id]
        row = _question_row(str(reference["question"]), str(reference["answer"]))
        correct, _ = _measure_nll_v84(
            stack.language, stack.system_prompt, memories[scene_id], row
        )
        wrong, _ = _measure_nll_v84(
            stack.language, stack.system_prompt, memories[paired], row
        )
        zero, _ = _measure_nll_v84(
            stack.language,
            stack.system_prompt,
            zero_environment_payload_v94(memories[scene_id]),
            row,
        )
        shuffled, _ = _measure_nll_v84(
            stack.language,
            stack.system_prompt,
            shuffle_atlas_values_v94(memories[scene_id]),
            row,
        )
        correct_nll = float(correct["mean_nll"])
        totals["correct"].append(correct_nll)
        totals["wrong_gap"].append(float(wrong["mean_nll"]) - correct_nll)
        totals["zero_gap"].append(float(zero["mean_nll"]) - correct_nll)
        totals["shuffle_gap"].append(float(shuffled["mean_nll"]) - correct_nll)
        if reference.get("counterfactual_expected_change") is True:
            totals["changed_wrong_gap"].append(float(wrong["mean_nll"]) - correct_nll)
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
    return {
        "correct_scene_mean_nll": sum(totals["correct"]) / len(totals["correct"]),
        "mean_wrong_minus_correct_nll": sum(totals["wrong_gap"])
        / len(totals["wrong_gap"]),
        "mean_changed_wrong_minus_correct_nll": sum(totals["changed_wrong_gap"])
        / len(totals["changed_wrong_gap"]),
        "zero_payload_mean_nll_gap": sum(totals["zero_gap"])
        / len(totals["zero_gap"]),
        "shuffled_atlas_mean_nll_gap": sum(totals["shuffle_gap"])
        / len(totals["shuffle_gap"]),
    }


def score_label_isolated_v94(
    config_path: str | Path = DEFAULT_CONFIG,
    *,
    measure_nll: bool = True,
) -> dict[str, Any]:
    """Open pinned labels only here and emit aggregate, answer-free metrics."""

    config = _load_config(config_path)
    manifest = validate_official_question_manifest(_question_manifest_path(config))
    references = _load_references_for_scorer(config, manifest)
    output = _resolve(str(_strict_mapping(config["outputs"], "outputs")["evaluation_predictions"]))
    records, provenance = _validate_predictions(output, manifest)
    audit = FileAccessAudit(
        forbidden_roots=[path for path in _forbidden_runtime_roots() if path.name == "oracle"],
        forbidden_component_names={"oracle"},
        block_forbidden=True,
    )
    nll_metrics: Mapping[str, float] | None = None
    if measure_nll:
        with audit:
            memories, _hashes, _memory_manifest = _load_bound_memories_before_questions(
                config, audit
            )
            nll_metrics = _measure_nll_controls_v94(config, references, memories)
        audit.assert_clean()
    score = score_records_v94(
        references,
        records,
        gates=_strict_mapping(config["gates"], "gates"),
        nll_metrics=nll_metrics,
        protected_read_count=len(audit.forbidden_accesses()),
    )
    report = {
        "artifact": SCORE_ARTIFACT,
        "schema_version": 94,
        "status": (
            "passed_awaiting_separate_leakage_packaging"
            if score["runtime_candidate_gate_passed"]
            else "measured_gate_not_passed"
        ),
        "row_count": len(records),
        "scene_count": 6,
        "question_manifest_sha256": manifest.manifest_sha256,
        "reference_sha256": EXPECTED_REFERENCE_SHA256,
        "predictions_sha256": sha256_file(output),
        "prediction_provenance_sha256": provenance["provenance_sha256"],
        "labels_opened_only_by_this_scorer": True,
        "answers_or_questions_serialized": False,
        "metrics": score,
        "runtime_promotion_authorized": False,
    }
    destination = _resolve(str(_strict_mapping(config["outputs"], "outputs")["evaluation_score"]))
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"V94 create-once score exists: {destination}")
    _atomic_json(destination, report)
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("compile-memory")
    subparsers.add_parser("predict")
    scorer = subparsers.add_parser("score")
    scorer.add_argument("--skip-nll", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "compile-memory":
        result = compile_evaluation_memory_cache_v94(args.config)
    elif args.command == "predict":
        result = predict_question_only_v94(args.config)
    else:
        result = score_label_isolated_v94(
            args.config, measure_nll=not bool(args.skip_nll)
        )
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
