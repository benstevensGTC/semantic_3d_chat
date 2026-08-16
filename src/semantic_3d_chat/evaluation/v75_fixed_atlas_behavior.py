"""Bounded historical behavior test for the V75 fixed-prefix atlas.

The predictor compiles one complete numeric atlas for each of sixteen opaque
held scenes *before* it opens the user-question manifest.  That immutable
prefix is then reused for the atlas arm of every question.  The exact V75
question-conditioned controller and V54 base are comparator arms only.

Answer-bearing references live in a physically separate scorer directory and
are blocked by the predictor's file-access audit.  The scorer never loads a
model, controller, scene prefix, map, or oracle.  This remains an internal
diagnostic; structural correctness does not imply behavioral success.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import time
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Final

import torch
import yaml
from safetensors import safe_open
from safetensors.torch import load_file

from semantic_3d_chat.chat.file_audit import FileAccessAudit
from semantic_3d_chat.chat.question_control_runtime import _load_control_head
from semantic_3d_chat.chat.runtime import StaticChatRuntime
from semantic_3d_chat.chat.runtime_config import load_runtime_config
from semantic_3d_chat.config import PROJECT_ROOT
from semantic_3d_chat.language.local_lm import prompt_token_ids, question_token_ids
from semantic_3d_chat.language.prefix_injection import (
    prefix_sha256,
    scene_boundary_mode_setting,
    scene_prefix_after_bos_setting,
)
from semantic_3d_chat.scene_encoder.fixed_prefix_atlas import tensor_sha256
from semantic_3d_chat.scene_encoder.fixed_prefix_atlas_v75 import (
    compile_fixed_scene_atlas_v75_v2,
)
from semantic_3d_chat.scene_encoder.question_control_v75 import (
    DenseFullSceneContinuousControlV75,
)

SCENE_IDS: Final[tuple[str, ...]] = (
    "scene_000039",
    "scene_000040",
    "scene_000041",
    "scene_000042",
    "scene_000043",
    "scene_000044",
    "scene_000045",
    "scene_000046",
    "scene_000047",
    "scene_000048",
    "scene_000049",
    "scene_000050",
    "scene_000051",
    "scene_000052",
    "scene_000055",
    "scene_000056",
)
PROBE_COUNT: Final[int] = 96
HIDDEN_SIZE: Final[int] = 1536
BASE_PREFIX_TOKENS: Final[int] = 258
ATLAS_PREFIX_TOKENS: Final[int] = 738
ATLAS_MEMORY_TOKENS: Final[int] = 480
ROW_COUNT: Final[int] = 16
PREFIX_MANIFEST_SHA256: Final[str] = (
    "5a288a7fef65a957ba7b20132c63380cfadc7edbc37b32c1885037f939b9db61"
)
PREFIX_BASE_CHECKPOINT_SHA256: Final[str] = (
    "3e128b40c1b73bb32750285679cda6b1bea364e67465e986a94a81dfc95e81e8"
)
SOURCE_V75_CANDIDATE_SHA256: Final[str] = (
    "d01275538489b3493a8e1ff080109d1db46832be6ca2a26f6d89d161c597188a"
)
V75_RUNTIME_WEIGHTS_SHA256: Final[str] = (
    "bb112f42ca5df71b88b4cd7721b9107f9be9b0dc01b612a4ace6212548da669c"
)
GEMMA_MODEL_FILE_SHA256: Final[str] = (
    "2db5482b20d746879bb3ef79b5203e9075a2e2b98f54ec7c2f281c1477ddc550"
)
GEMMA_REVISION: Final[str] = "3e22461f65e89153144f8adb70e3b8c2cc9845a7"
_ROW_ID = re.compile(r"row_[0-9a-f]{24}")
_SHA256 = re.compile(r"[0-9a-f]{64}")
_CONFIG_KEYS = frozenset(
    {
        "schema_version",
        "status",
        "runtime_config",
        "base_checkpoint",
        "source_controller",
        "source_prefix_cache",
        "probe_bank",
        "predictor_questions",
        "scorer_forbidden_root",
        "scene_ids",
        "probe_count",
        "base_prefix_tokens",
        "atlas_prefix_tokens",
        "atlas_memory_tokens",
        "row_count",
        "layout",
        "output_predictions",
        "scope",
    }
)
_SCOPE = {
    "historical_training_pool_only": True,
    "pair_disjoint_smoke": True,
    "scene_disjoint_smoke": True,
    "question_disjoint_smoke": False,
    "all_prefixes_compiled_before_questions": True,
    "question_dependent_scene_processing": False,
    "question_dependent_retrieval": False,
    "official_validation_loaded": False,
    "official_test_loaded": False,
    "deferred_final_loaded": False,
    "oracle_loaded": False,
    "runtime_promotion_authorized": False,
}
_PROBE_SAFE_METADATA = {
    "artifact": "v75_fixed_atlas_numeric_probe_bank_v1",
    "schema_version": "1",
    "tensor_name": "probe_embeddings",
    "questions_or_answers_serialized": "false",
    "answer_codebook_serialized": "false",
    "environmental_text_serialized": "false",
    "runtime_promotion_authorized": "false",
}


def _resolve(path: str | Path) -> Path:
    value = Path(path).expanduser()
    rooted = value if value.is_absolute() else PROJECT_ROOT / value
    return Path(os.path.abspath(rooted))


def _sha256_file(path: Path, audit: FileAccessAudit | None = None) -> str:
    if audit is not None:
        audit.record(path)
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _strict_json(path: Path, audit: FileAccessAudit | None = None) -> dict[str, Any]:
    if audit is not None:
        audit.record(path)

    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"V75 atlas JSON contains duplicate field: {key}")
            result[key] = value
        return result

    value = json.loads(
        path.read_text(encoding="utf-8"), object_pairs_hook=reject_duplicates
    )
    if not isinstance(value, dict):
        raise TypeError("V75 atlas JSON must be an object")
    return value


def _guard_regular(path: Path, purpose: str) -> Path:
    current = Path(path.anchor)
    for component in path.parts[1:]:
        current /= component
        if current.is_symlink():
            raise ValueError(f"V75 atlas {purpose} path contains a symlink: {current}")
    if not path.is_file():
        raise FileNotFoundError(f"V75 atlas {purpose} is unavailable: {path}")
    return path


def load_behavior_config(
    path: str | Path, audit: FileAccessAudit | None = None
) -> dict[str, Any]:
    source = _guard_regular(_resolve(path), "behavior config")
    if audit is not None:
        audit.record(source)
    payload = yaml.safe_load(source.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping) or set(payload) != {"v75_fixed_atlas_behavior"}:
        raise ValueError("V75 atlas behavior config must contain exactly one mapping")
    config = payload["v75_fixed_atlas_behavior"]
    if not isinstance(config, Mapping) or set(config) != _CONFIG_KEYS:
        raise ValueError("V75 atlas behavior config fields changed")
    expected = {
        "schema_version": 1,
        "status": "historical_internal_behavior_diagnostic_not_promoted",
        "scene_ids": list(SCENE_IDS),
        "probe_count": PROBE_COUNT,
        "base_prefix_tokens": BASE_PREFIX_TOKENS,
        "atlas_prefix_tokens": ATLAS_PREFIX_TOKENS,
        "atlas_memory_tokens": ATLAS_MEMORY_TOKENS,
        "row_count": ROW_COUNT,
        "layout": "v2_atlas_then_all_256_base_latents",
        "scope": _SCOPE,
    }
    for field, value in expected.items():
        if config.get(field) != value:
            raise ValueError(f"V75 atlas behavior {field} changed")
    for field in (
        "runtime_config",
        "base_checkpoint",
        "source_controller",
        "source_prefix_cache",
        "probe_bank",
        "predictor_questions",
        "scorer_forbidden_root",
        "output_predictions",
    ):
        value = config.get(field)
        if not isinstance(value, str) or not value:
            raise TypeError(f"V75 atlas behavior {field} must be a nonempty path")
    if _resolve(config["scorer_forbidden_root"]).is_relative_to(
        _resolve(config["predictor_questions"])
    ):
        raise ValueError("V75 atlas predictor questions cannot live below scorer data")
    return dict(config)


def _load_probe_bank(
    root: Path, audit: FileAccessAudit
) -> tuple[torch.Tensor, dict[str, Any]]:
    if root.is_symlink() or not root.is_dir():
        raise FileNotFoundError(f"V75 atlas numeric probe bank is unavailable: {root}")
    inventory = {path.name for path in root.iterdir()}
    if inventory != {"probes.safetensors", "runtime_metadata.json"}:
        raise ValueError("V75 atlas numeric probe-bank inventory changed")
    probes_path = _guard_regular(root / "probes.safetensors", "probe tensor")
    metadata_path = _guard_regular(root / "runtime_metadata.json", "probe metadata")
    metadata = _strict_json(metadata_path, audit)
    exact = {
        "schema_version": 1,
        "artifact": "v75_fixed_atlas_numeric_probe_bank_v1",
        "status": "historical_internal_diagnostic_not_promoted",
        "probe_count": PROBE_COUNT,
        "hidden_size": HIDDEN_SIZE,
        "dtype": "torch.float32",
        "source_scope": "v73_historical_optimization_fold_only",
        "source_train_pair_count": 12,
        "source_train_scene_count": 24,
        "source_train_row_count": 576,
        "source_unique_question_count": PROBE_COUNT,
        "source_qa_sha256": (
            "01721bf904b1ab0b65ce8acac6e366287040873cda1356da6c70c4981abe7619"
        ),
        "source_v73_config_sha256": (
            "d208f28380e3f1810a688be8ea8a263831b6a741f7f90e667795637f39d841f1"
        ),
        "model_revision": GEMMA_REVISION,
        "model_file_sha256": GEMMA_MODEL_FILE_SHA256,
        "embedding_tensor_name": "model.language_model.embed_tokens.weight",
        "pooling": "mean_of_complete_question_token_embedding_sequence",
        "probe_order": "ascending_sha256_of_question_text_not_serialized",
        "questions_or_answers_serialized": False,
        "answer_codebook_serialized": False,
        "environmental_text_serialized": False,
        "official_validation_loaded": False,
        "official_test_loaded": False,
        "deferred_final_loaded": False,
        "oracle_loaded": False,
        "runtime_promotion_authorized": False,
    }
    if not set(exact).issubset(metadata) or any(
        metadata.get(field) != value for field, value in exact.items()
    ):
        raise ValueError("V75 atlas numeric probe metadata contract changed")
    if set(metadata) != set(exact) | {
        "probe_file_sha256",
        "probe_tensor_sha256",
        "source_question_hash_inventory_sha256",
    }:
        raise ValueError("V75 atlas numeric probe metadata fields changed")
    for field in (
        "probe_file_sha256",
        "probe_tensor_sha256",
        "source_question_hash_inventory_sha256",
    ):
        if not isinstance(metadata.get(field), str) or _SHA256.fullmatch(metadata[field]) is None:
            raise ValueError(f"V75 atlas probe {field} is not a SHA-256 digest")
    if _sha256_file(probes_path, audit) != metadata["probe_file_sha256"]:
        raise ValueError("V75 atlas numeric probe file changed")
    audit.record(probes_path)
    with safe_open(str(probes_path), framework="pt", device="cpu") as handle:
        if set(handle.keys()) != {"probe_embeddings"} or handle.metadata() != _PROBE_SAFE_METADATA:
            raise ValueError("V75 atlas numeric probe safetensors contract changed")
    state = load_file(str(probes_path), device="cpu")
    probes = state["probe_embeddings"].detach().float().contiguous()
    if (
        tuple(probes.shape) != (PROBE_COUNT, HIDDEN_SIZE)
        or not bool(torch.isfinite(probes).all())
        or bool((probes.norm(dim=-1) <= 1e-8).any())
        or tensor_sha256(probes) != metadata["probe_tensor_sha256"]
    ):
        raise ValueError("V75 atlas numeric probe tensor changed")
    return probes, metadata


def _load_base_prefixes(
    root: Path, scene_ids: Sequence[str], audit: FileAccessAudit
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    if root.is_symlink() or not root.is_dir():
        raise FileNotFoundError(f"V75 atlas prefix cache is unavailable: {root}")
    manifest_path = _guard_regular(root / "manifest.json", "prefix manifest")
    if _sha256_file(manifest_path, audit) != PREFIX_MANIFEST_SHA256:
        raise ValueError("V75 atlas prefix manifest changed")
    manifest = _strict_json(manifest_path, audit)
    if (
        manifest.get("artifact") != "question_independent_scene_prefix_cache_v1"
        or manifest.get("scene_count") != 40
        or manifest.get("base_checkpoint_sha256") != PREFIX_BASE_CHECKPOINT_SHA256
        or manifest.get("question_inputs_used") is not False
        or manifest.get("question_dependent_scene_retrieval") is not False
        or manifest.get("complete_scene_prefixes") is not True
        or manifest.get("environmental_text_inputs") != []
    ):
        raise ValueError("V75 atlas prefix manifest contract changed")
    entries = manifest.get("scenes")
    if not isinstance(entries, Mapping) or not set(scene_ids) <= set(entries):
        raise ValueError("V75 atlas prefix manifest lacks a required opaque scene")
    result: dict[str, torch.Tensor] = {}
    for scene_id in scene_ids:
        entry = entries[scene_id]
        if (
            not isinstance(entry, Mapping)
            or set(entry)
            != {
                "dtype",
                "file_sha256",
                "file_size_bytes",
                "filename",
                "prefix_sha256",
                "shape",
            }
            or entry.get("filename") != f"{scene_id}.safetensors"
            or entry.get("shape") != [1, BASE_PREFIX_TOKENS, HIDDEN_SIZE]
            or entry.get("dtype") != "bfloat16"
        ):
            raise ValueError("V75 atlas prefix filename is not opaque or exact")
        path = _guard_regular(root / str(entry["filename"]), "numeric base prefix")
        if (
            path.stat().st_size != entry.get("file_size_bytes")
            or _sha256_file(path, audit) != entry.get("file_sha256")
        ):
            raise ValueError(f"V75 atlas cached prefix changed: {scene_id}")
        audit.record(path)
        state = load_file(str(path), device="cpu")
        if set(state) != {"scene_prefix"}:
            raise ValueError("V75 atlas prefix file contains an unexpected tensor")
        prefix = state["scene_prefix"].detach().contiguous()
        if (
            tuple(prefix.shape) != (1, BASE_PREFIX_TOKENS, HIDDEN_SIZE)
            or prefix.dtype != torch.bfloat16
            or not bool(torch.isfinite(prefix).all())
            or prefix_sha256(prefix) != entry.get("prefix_sha256")
        ):
            raise ValueError("V75 atlas cached base prefix shape changed")
        result[scene_id] = prefix
    return result, manifest


def _load_predictor_questions(
    root: Path, audit: FileAccessAudit
) -> tuple[tuple[dict[str, str], ...], dict[str, Any]]:
    if root.is_symlink() or not root.is_dir():
        raise FileNotFoundError(f"V75 atlas predictor question root is unavailable: {root}")
    if {path.name for path in root.iterdir()} != {"questions.jsonl", "metadata.json"}:
        raise ValueError("V75 atlas predictor question inventory changed")
    metadata = _strict_json(
        _guard_regular(root / "metadata.json", "predictor question metadata"), audit
    )
    exact = {
        "schema_version": 1,
        "artifact": "v75_fixed_atlas_historical_smoke_predictor_questions_v1",
        "status": "historical_internal_pair_scene_disjoint_smoke",
        "row_count": ROW_COUNT,
        "scene_count": len(SCENE_IDS),
        "scene_ids": list(SCENE_IDS),
        "questions_are_user_text_only": True,
        "answers_or_labels_serialized": False,
        "oracle_fields_serialized": False,
        "pair_or_change_metadata_serialized": False,
        "official_validation_loaded": False,
        "official_test_loaded": False,
        "deferred_final_loaded": False,
    }
    if set(metadata) != set(exact) | {"questions_file_sha256"} or any(
        metadata.get(field) != value for field, value in exact.items()
    ):
        raise ValueError("V75 atlas predictor question metadata changed")
    questions_path = _guard_regular(root / "questions.jsonl", "predictor questions")
    if _sha256_file(questions_path, audit) != metadata.get("questions_file_sha256"):
        raise ValueError("V75 atlas predictor question file changed")
    rows: list[dict[str, str]] = []
    for line_number, line in enumerate(questions_path.read_text(encoding="utf-8").splitlines(), 1):
        value = json.loads(line)
        if not isinstance(value, dict) or set(value) != {"row_id", "scene_id", "question"}:
            raise ValueError(f"V75 atlas predictor row {line_number} fields changed")
        if (
            not isinstance(value["row_id"], str)
            or _ROW_ID.fullmatch(value["row_id"]) is None
            or value["scene_id"] not in SCENE_IDS
            or not isinstance(value["question"], str)
            or not value["question"]
        ):
            raise ValueError(f"V75 atlas predictor row {line_number} is invalid")
        rows.append(value)
    if (
        len(rows) != ROW_COUNT
        or len({row["row_id"] for row in rows}) != ROW_COUNT
        or tuple(sorted(row["scene_id"] for row in rows)) != SCENE_IDS
    ):
        raise ValueError("V75 atlas predictor row inventory changed")
    return tuple(rows), metadata


def _runtime_audit(scorer_root: Path) -> FileAccessAudit:
    forbidden: list[Path] = [
        scorer_root,
        PROJECT_ROOT / "data_gemma4" / "training",
        PROJECT_ROOT / "data" / "oracle",
        PROJECT_ROOT / "data" / "qa",
        PROJECT_ROOT / "configs" / "benchmarks" / "oracle",
        PROJECT_ROOT / "reports" / "gemma4" / "metrics" / "v75_official_validation_score.json",
    ]
    forbidden.extend(PROJECT_ROOT.glob("data*/oracle"))
    forbidden.extend(PROJECT_ROOT.glob("data*/qa"))
    return FileAccessAudit(
        forbidden,
        forbidden_component_names={"oracle", "validation", "validate", "test", "deferred"},
        block_forbidden=True,
    )


def _disable_decoder_checkpointing(language: Any) -> None:
    decoder = language.decoder_module
    disable = getattr(decoder, "gradient_checkpointing_disable", None)
    if not callable(disable):
        raise TypeError("V75 atlas decoder cannot disable gradient checkpointing")
    disable()
    decoder.eval()
    language.model.eval()
    for config in (getattr(language.model, "config", None), getattr(decoder, "config", None)):
        if config is not None and hasattr(config, "use_cache"):
            config.use_cache = True


def _question_embeddings(runtime: StaticChatRuntime, question: str) -> torch.Tensor:
    ids = question_token_ids(runtime.language.tokenizer, question, runtime.language.device)
    with torch.inference_mode():
        value = runtime.language.model.get_input_embeddings()(ids).detach().float()
    if value.ndim != 3 or value.shape[0] != 1 or value.shape[-1] != HIDDEN_SIZE:
        raise RuntimeError("V75 atlas live question embedding shape changed")
    return value


@torch.inference_mode()
def _generate(
    runtime: StaticChatRuntime,
    scene_prefix: torch.Tensor,
    question: str,
    control_tokens: torch.Tensor | None,
) -> str:
    language = runtime.language
    backend = language.prefix_backend
    if backend is None or language.backend_name != "gemma4":
        raise RuntimeError("V75 atlas predictor requires the Gemma 4 prefix backend")
    prompt_ids = prompt_token_ids(
        language.tokenizer,
        str(runtime.config["language"]["system_prompt"]),
        question,
        language.device,
    )
    prepared = backend.prepare(
        scene_prefix,
        prompt_ids,
        scene_prefix_after_bos=scene_prefix_after_bos_setting(runtime.config),
        scene_boundary_mode=scene_boundary_mode_setting(runtime.config),
        control_tokens=control_tokens,
    )
    generated = backend.generate(
        prepared,
        max_new_tokens=int(runtime.config["language"]["max_answer_tokens"]),
        eos_token_ids=runtime._eos_token_ids(),
    )
    return language.tokenizer.decode(
        generated[0].detach().cpu().tolist(), skip_special_tokens=True
    ).strip() or "unknown"


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() or path.is_symlink():
        raise FileExistsError(path)
    temporary = path.with_name(f".{path.name}.partial-{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.link(temporary, path)
    temporary.unlink()


def predict(config_path: str | Path) -> dict[str, Any]:
    """Compile every scene first, then run fixed-atlas/V75/V54 comparator arms."""

    preliminary = load_behavior_config(config_path)
    scorer_root = _resolve(preliminary["scorer_forbidden_root"])
    output = _resolve(preliminary["output_predictions"])
    if output.exists() or output.is_symlink():
        raise FileExistsError(output)
    audit = _runtime_audit(scorer_root)
    started = time.perf_counter()
    with audit:
        config = load_behavior_config(config_path, audit)
        runtime_config_path = _guard_regular(
            _resolve(config["runtime_config"]), "sanitized runtime config"
        )
        audit.record(runtime_config_path)
        runtime_config = load_runtime_config(runtime_config_path)
        probes, probe_metadata = _load_probe_bank(_resolve(config["probe_bank"]), audit)
        base_prefixes, prefix_manifest = _load_base_prefixes(
            _resolve(config["source_prefix_cache"]), SCENE_IDS, audit
        )
        controller, controller_metadata = _load_control_head(
            _resolve(config["source_controller"]),
            hidden_size=HIDDEN_SIZE,
            device=torch.device("cpu"),
            audit=audit,
        )
        if (
            type(controller) is not DenseFullSceneContinuousControlV75
            or controller_metadata.get("source_v75_candidate_sha256")
            != SOURCE_V75_CANDIDATE_SHA256
            or controller_metadata.get("weights_sha256") != V75_RUNTIME_WEIGHTS_SHA256
        ):
            raise ValueError("V75 atlas predictor did not load the exact sealed V75 controller")

        atlas_prefixes: dict[str, torch.Tensor] = {}
        atlas_audits: dict[str, dict[str, Any]] = {}
        for scene_id in SCENE_IDS:
            compiled = compile_fixed_scene_atlas_v75_v2(
                base_prefixes[scene_id], controller, probes
            )
            if (
                tuple(compiled.scene_prefix.shape)
                != (1, ATLAS_PREFIX_TOKENS, HIDDEN_SIZE)
                or compiled.audit.atlas_memory_token_count != ATLAS_MEMORY_TOKENS
                or compiled.audit.environment_latent_count != 256
                or not compiled.audit.every_probe_processed
                or not compiled.audit.complete_atlas_included
                or not compiled.audit.base_environment_tokens_preserved_exactly
                or not compiled.audit.atlas_key_value_tokens_preserved_exactly
                or compiled.audit.user_question_inputs_used_for_compilation
                or compiled.audit.question_dependent_scene_processing
                or compiled.audit.question_dependent_retrieval
            ):
                raise RuntimeError("V75 atlas compiled prefix failed its strict contract")
            atlas_prefixes[scene_id] = compiled.scene_prefix.detach().cpu().contiguous()
            atlas_audits[scene_id] = compiled.audit.as_dict()
        all_prefixes_compiled_before_question_manifest = True
        atlas_hashes_before = {
            scene_id: prefix_sha256(prefix) for scene_id, prefix in atlas_prefixes.items()
        }

        runtime = StaticChatRuntime.load(
            runtime_config,
            "scene_000011",
            checkpoint=_resolve(config["base_checkpoint"]),
            audit=audit,
            local_files_only=True,
        )
        _disable_decoder_checkpointing(runtime.language)
        backend_revision = getattr(runtime.language.prefix_backend, "model_revision", None)
        if runtime.language.backend_name != "gemma4" or backend_revision != GEMMA_REVISION:
            raise ValueError("V75 atlas predictor loaded an unexpected language model")
        device = torch.device(runtime.language.device)
        model_dtype = next(runtime.language.model.parameters()).dtype
        controller.to(device=device, dtype=torch.float32).eval()

        # This is intentionally after every scene atlas has been materialized.
        question_rows, question_metadata = _load_predictor_questions(
            _resolve(config["predictor_questions"]), audit
        )
        records: list[dict[str, Any]] = []
        for ordinal, row in enumerate(question_rows, 1):
            scene_id = row["scene_id"]
            question = row["question"]
            base = base_prefixes[scene_id].to(device=device, dtype=model_dtype)
            atlas = atlas_prefixes[scene_id].to(device=device, dtype=model_dtype)
            with torch.inference_mode():
                direct_control = controller(
                    base.float(), _question_embeddings(runtime, question)
                ).control_tokens
            control_rms = float(direct_control.float().square().mean().sqrt().cpu())
            record_started = time.perf_counter()
            atlas_prediction = _generate(runtime, atlas, question, None)
            direct_prediction = _generate(runtime, base, question, direct_control.to(base))
            v54_prediction = _generate(runtime, base, question, None)
            records.append(
                {
                    "row_id": row["row_id"],
                    "scene_id": scene_id,
                    "atlas_prefix_sha256": atlas_hashes_before[scene_id],
                    "base_prefix_sha256": prefix_sha256(base_prefixes[scene_id]),
                    "atlas_prediction": atlas_prediction,
                    "direct_v75_prediction": direct_prediction,
                    "v54_prediction": v54_prediction,
                    "direct_v75_control_rms": control_rms,
                    "elapsed_seconds": time.perf_counter() - record_started,
                }
            )
            print(
                json.dumps(
                    {
                        "event": "v75_fixed_atlas_behavior_row",
                        "ordinal": ordinal,
                        "total": ROW_COUNT,
                        "row_id": row["row_id"],
                        "scene_id": scene_id,
                        "atlas_prediction": atlas_prediction,
                        "direct_v75_prediction": direct_prediction,
                        "v54_prediction": v54_prediction,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )

        atlas_hashes_after = {
            scene_id: prefix_sha256(prefix) for scene_id, prefix in atlas_prefixes.items()
        }
        prefix_invariant = atlas_hashes_before == atlas_hashes_after and all(
            record["atlas_prefix_sha256"] == atlas_hashes_before[record["scene_id"]]
            for record in records
        )
        if not prefix_invariant:
            raise RuntimeError("V75 atlas prefix changed after user questions")
        if any(
            not math.isfinite(float(record["direct_v75_control_rms"]))
            for record in records
        ):
            raise RuntimeError("V75 atlas direct comparator produced a nonfinite control")
    audit.assert_clean()
    loaded_files = audit.unique_paths
    scorer_reads = [
        path for path in loaded_files if Path(path).is_relative_to(scorer_root)
    ]
    if scorer_reads:
        raise RuntimeError("V75 atlas predictor opened answer-bearing scorer data")

    payload: dict[str, Any] = {
        "artifact": "v75_fixed_atlas_historical_internal_predictions_v1",
        "status": "behavior_measured_not_promoted",
        "execution_valid": True,
        "row_count": len(records),
        "scene_count": len(SCENE_IDS),
        "arms": ["fixed_v75_atlas", "direct_exact_v75", "frozen_v54"],
        "probe_bank": {
            "probe_count": probe_metadata["probe_count"],
            "probe_file_sha256": probe_metadata["probe_file_sha256"],
            "probe_tensor_sha256": probe_metadata["probe_tensor_sha256"],
            "questions_or_answers_serialized": False,
            "answer_codebook_serialized": False,
        },
        "controller": {
            "architecture": controller_metadata["architecture"],
            "weights_sha256": controller_metadata["weights_sha256"],
            "source_v75_candidate_sha256": controller_metadata[
                "source_v75_candidate_sha256"
            ],
        },
        "base": {
            "runtime_checkpoint": config["base_checkpoint"],
            "prefix_cache_base_checkpoint_sha256": prefix_manifest[
                "base_checkpoint_sha256"
            ],
            "gemma_revision": backend_revision,
            "gemma_model_file_sha256": GEMMA_MODEL_FILE_SHA256,
        },
        "scene_prefix": {
            "layout": config["layout"],
            "base_tokens": BASE_PREFIX_TOKENS,
            "fixed_atlas_tokens": ATLAS_PREFIX_TOKENS,
            "atlas_memory_tokens": ATLAS_MEMORY_TOKENS,
            "base_environment_latents": 256,
            "all_scenes_compiled_before_question_manifest_opened": (
                all_prefixes_compiled_before_question_manifest
            ),
            "same_compiled_prefix_reused_for_every_question": True,
            "prefix_hashes_before": atlas_hashes_before,
            "prefix_hashes_after": atlas_hashes_after,
            "prefix_hashes_invariant": prefix_invariant,
            "question_inputs_used_for_compilation": False,
            "question_dependent_scene_processing": False,
            "question_dependent_retrieval": False,
            "semantic_or_spatial_top_k_selection": False,
            "all_256_base_latents_preserved": all(
                value["base_environment_tokens_preserved_exactly"]
                for value in atlas_audits.values()
            ),
            "every_probe_processed_for_every_scene": all(
                value["every_probe_processed"] for value in atlas_audits.values()
            ),
        },
        "question_manifest": {
            "questions_file_sha256": question_metadata["questions_file_sha256"],
            "user_question_text_only": True,
            "answers_or_labels_serialized": False,
        },
        "leakage": {
            "loaded_file_count": len(loaded_files),
            "loaded_files": loaded_files,
            "forbidden_access_count": len(audit.forbidden_accesses()),
            "forbidden_accesses": audit.forbidden_accesses(),
            "scorer_reference_files_loaded": False,
            "oracle_loaded": False,
            "training_artifacts_loaded": False,
            "official_validation_loaded": False,
            "official_test_loaded": False,
            "deferred_final_loaded": False,
        },
        "behavioral_accuracy_scored_in_predictor": False,
        "structural_compiler_implies_behavioral_success": False,
        "runtime_promotion_authorized": False,
        "elapsed_seconds": time.perf_counter() - started,
        "records": records,
    }
    _atomic_write_json(output, payload)
    return payload


def _load_reference_artifact(root: Path) -> tuple[dict[str, dict[str, str]], dict[str, Any]]:
    if root.is_symlink() or not root.is_dir():
        raise FileNotFoundError(f"V75 atlas scorer root is unavailable: {root}")
    if {path.name for path in root.iterdir()} != {"references.jsonl", "metadata.json"}:
        raise ValueError("V75 atlas scorer inventory changed")
    metadata = _strict_json(_guard_regular(root / "metadata.json", "scorer metadata"))
    exact = {
        "schema_version": 1,
        "artifact": "v75_fixed_atlas_historical_smoke_scorer_references_v1",
        "status": "evaluation_only_never_loaded_by_predictor",
        "row_count": ROW_COUNT,
        "unit_count": 8,
        "change_family_count": 8,
        "model_or_runtime_loaded_by_scorer": False,
        "physically_separate_from_predictor_questions": True,
        "official_validation_loaded": False,
        "official_test_loaded": False,
        "deferred_final_loaded": False,
        "oracle_loaded": False,
    }
    if set(metadata) != set(exact) | {"references_file_sha256"} or any(
        metadata.get(field) != value for field, value in exact.items()
    ):
        raise ValueError("V75 atlas scorer metadata changed")
    path = _guard_regular(root / "references.jsonl", "scorer references")
    if _sha256_file(path) != metadata.get("references_file_sha256"):
        raise ValueError("V75 atlas scorer reference file changed")
    rows: dict[str, dict[str, str]] = {}
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        value = json.loads(line)
        required = {"row_id", "answer", "answer_type", "change_type", "unit_id"}
        if (
            not isinstance(value, dict)
            or set(value) != required
            or any(not isinstance(value[field], str) or not value[field] for field in required)
            or _ROW_ID.fullmatch(value["row_id"]) is None
            or value["row_id"] in rows
        ):
            raise ValueError(f"V75 atlas scorer row {line_number} is invalid")
        rows[value["row_id"]] = value
    if len(rows) != ROW_COUNT:
        raise ValueError("V75 atlas scorer row count changed")
    return rows, metadata


def _aggregate_scored(
    joined: Sequence[Mapping[str, Any]], field: str
) -> dict[str, Any]:
    correct = sum(bool(row[field]) for row in joined)
    families: defaultdict[str, list[bool]] = defaultdict(list)
    for row in joined:
        families[str(row["change_type"])].append(bool(row[field]))
    return {
        "correct": correct,
        "total": len(joined),
        "accuracy": correct / len(joined),
        "by_change_type": {
            family: {
                "correct": sum(values),
                "total": len(values),
                "accuracy": sum(values) / len(values),
            }
            for family, values in sorted(families.items())
        },
    }


def _prediction_change_units(
    joined: Sequence[Mapping[str, Any]], prediction_field: str
) -> int:
    from semantic_3d_chat.evaluation.metrics import normalize_answer

    grouped: defaultdict[str, list[str]] = defaultdict(list)
    for row in joined:
        grouped[str(row["unit_id"])].append(normalize_answer(str(row[prediction_field])))
    if any(len(values) != 2 for values in grouped.values()):
        raise ValueError("V75 atlas scorer unit is not a two-sided pair")
    return sum(values[0] != values[1] for values in grouped.values())


def score(
    predictions_path: str | Path,
    references_root: str | Path,
    output_path: str | Path,
) -> dict[str, Any]:
    """Score completed predictions without loading any model or scene data."""

    from semantic_3d_chat.evaluation.v55_development_score import (
        canonical_type_specific_match,
    )

    prediction_path = _guard_regular(_resolve(predictions_path), "prediction artifact")
    predictions = _strict_json(prediction_path)
    if (
        predictions.get("artifact")
        != "v75_fixed_atlas_historical_internal_predictions_v1"
        or predictions.get("execution_valid") is not True
        or predictions.get("row_count") != ROW_COUNT
        or predictions.get("runtime_promotion_authorized") is not False
        or predictions.get("behavioral_accuracy_scored_in_predictor") is not False
    ):
        raise ValueError("V75 atlas prediction artifact contract changed")
    prefix = predictions.get("scene_prefix")
    leakage = predictions.get("leakage")
    if (
        not isinstance(prefix, Mapping)
        or prefix.get("prefix_hashes_invariant") is not True
        or prefix.get("all_scenes_compiled_before_question_manifest_opened") is not True
        or prefix.get("same_compiled_prefix_reused_for_every_question") is not True
        or prefix.get("question_inputs_used_for_compilation") is not False
        or prefix.get("question_dependent_scene_processing") is not False
        or prefix.get("question_dependent_retrieval") is not False
        or not isinstance(leakage, Mapping)
        or leakage.get("forbidden_access_count") != 0
        or leakage.get("scorer_reference_files_loaded") is not False
    ):
        raise ValueError("V75 atlas prediction structural/leakage evidence failed")
    records = predictions.get("records")
    if not isinstance(records, list) or len(records) != ROW_COUNT:
        raise ValueError("V75 atlas prediction record count changed")
    references, reference_metadata = _load_reference_artifact(_resolve(references_root))
    if {row.get("row_id") for row in records} != set(references):
        raise ValueError("V75 atlas prediction/reference row IDs differ")

    joined: list[dict[str, Any]] = []
    for record in records:
        required = {
            "row_id",
            "scene_id",
            "atlas_prefix_sha256",
            "base_prefix_sha256",
            "atlas_prediction",
            "direct_v75_prediction",
            "v54_prediction",
            "direct_v75_control_rms",
            "elapsed_seconds",
        }
        if not isinstance(record, Mapping) or set(record) != required:
            raise ValueError("V75 atlas prediction record fields changed")
        reference = references[str(record["row_id"])]
        scored = {**record, **reference}
        for arm, field in (
            ("atlas", "atlas_prediction"),
            ("direct_v75", "direct_v75_prediction"),
            ("v54", "v54_prediction"),
        ):
            scored[f"{arm}_correct"] = canonical_type_specific_match(
                reference["answer_type"], str(record[field]), reference["answer"]
            )
        joined.append(scored)

    atlas = _aggregate_scored(joined, "atlas_correct")
    direct = _aggregate_scored(joined, "direct_v75_correct")
    v54 = _aggregate_scored(joined, "v54_correct")
    result = {
        "artifact": "v75_fixed_atlas_historical_internal_score_v1",
        "status": "behavior_measured_not_promoted",
        "execution_valid": True,
        "scope": {
            "historical_training_pool_only": True,
            "optimization_pair_count": 12,
            "held_pair_count": 8,
            "held_scene_count": 16,
            "held_row_count": 16,
            "pair_disjoint": True,
            "scene_disjoint": True,
            "question_disjoint": False,
            "official_validation_loaded": False,
            "official_test_loaded": False,
            "deferred_final_loaded": False,
            "oracle_loaded": False,
        },
        "prediction_artifact_sha256": _sha256_file(prediction_path),
        "reference_artifact_sha256": reference_metadata["references_file_sha256"],
        "fixed_v75_atlas": atlas,
        "direct_exact_v75": direct,
        "frozen_v54": v54,
        "fixed_atlas_accuracy_gain_over_v54": atlas["accuracy"] - v54["accuracy"],
        "fixed_atlas_accuracy_gap_to_direct_v75": atlas["accuracy"] - direct["accuracy"],
        "prediction_change_units": {
            "fixed_v75_atlas": _prediction_change_units(joined, "atlas_prediction"),
            "direct_exact_v75": _prediction_change_units(
                joined, "direct_v75_prediction"
            ),
            "frozen_v54": _prediction_change_units(joined, "v54_prediction"),
            "total": 8,
        },
        "change_family_counts": dict(Counter(row["change_type"] for row in joined)),
        "prefix_invariance_passed": True,
        "predictor_reference_isolation_passed": True,
        "behavioral_accuracy_measured": True,
        "structural_compiler_implies_behavioral_success": False,
        "runtime_promotion_authorized": False,
        "protected_evaluation_authorized": False,
    }
    _atomic_write_json(_resolve(output_path), result)
    return result


def predict_main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run V75 fixed-atlas historical predictor")
    parser.add_argument(
        "--config",
        default="configs/experiments/gemma4_v75_fixed_prefix_atlas_behavior.yaml",
    )
    args = parser.parse_args(argv)
    payload = predict(args.config)
    print(
        json.dumps({key: value for key, value in payload.items() if key != "records"}, indent=2, sort_keys=True),
        flush=True,
    )
    return 0


def score_main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Score V75 fixed-atlas historical predictions")
    parser.add_argument(
        "--predictions",
        default="reports/gemma4/predictions/v75_fixed_atlas_historical_internal.json",
    )
    parser.add_argument(
        "--references",
        default=(
            "reports/gemma4/artifacts/v75_fixed_atlas_historical_internal_v1/scorer"
        ),
    )
    parser.add_argument(
        "--output",
        default="reports/gemma4/metrics/v75_fixed_atlas_historical_internal_score.json",
    )
    args = parser.parse_args(argv)
    payload = score(args.predictions, args.references, args.output)
    print(json.dumps(payload, indent=2, sort_keys=True), flush=True)
    return 0


__all__ = [
    "ATLAS_PREFIX_TOKENS",
    "BASE_PREFIX_TOKENS",
    "PROBE_COUNT",
    "ROW_COUNT",
    "SCENE_IDS",
    "load_behavior_config",
    "predict",
    "predict_main",
    "score",
    "score_main",
]
