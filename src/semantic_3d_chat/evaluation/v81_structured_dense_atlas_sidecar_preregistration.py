"""Seal and CPU-preflight the V81 immutable-atlas dense reader.

This module authenticates historical inputs, the exact V75-V2 layout, split
inventories, candidate boundaries, and a shape-faithful CPU surrogate.  It
does not instantiate Gemma, fit any parameter, use MPS, open a scorer during
prediction, or publish a checkpoint.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import resource
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Final

import torch
import torch.nn.functional as F
import yaml
from safetensors import safe_open
from safetensors.torch import load_file

from semantic_3d_chat.config import PROJECT_ROOT
from semantic_3d_chat.evaluation.v80_atlas_attention_reader_preregistration import (
    EXPECTED_BROAD_HELD_SHA256,
    EXPECTED_HELD_SMOKE_SHA256,
    select_broad_held_v80,
    select_held_smoke_v80,
)
from semantic_3d_chat.language.prefix_injection import prefix_sha256
from semantic_3d_chat.language.v81_structured_dense_atlas_sidecar import (
    ATLAS_MEMORY_TOKENS,
    ATLAS_UNIFORM_FLOOR_MASS,
    BASE_ENVIRONMENT_LATENTS,
    BASE_PREFIX_TOKENS,
    BASE_UNIFORM_FLOOR_MASS,
    CANDIDATE_TENSOR_NAMES,
    FIXED_PREFIX_TOKENS,
    HIDDEN_SIZE,
    INPUT_EMBEDDING_TENSOR_NAME,
    MINIMUM_ATLAS_WEIGHT,
    MINIMUM_BASE_WEIGHT,
    MODEL_BLOB_SHA256_IDENTITY,
    PROBE_COUNT,
    RAW_ATLAS_LOGIT_SCALE,
    TRAINABLE_PARAMETER_COUNT,
    VALUES_PER_PROBE,
    StructuredDenseAtlasSidecarV81,
    assert_fixed_prefix_identity_v81,
    assert_prefix_binding_v81,
    audit_v75_v2_prefix_v81,
    bind_fixed_prefix_before_question_v81,
    deterministic_atlas_read_v81,
    frozen_lm_head_logits_v81,
    latest_user_question_query_v81,
    reconstruct_base_v54_prefix_v81,
    sanitized_candidate_metadata_v81,
    split_v75_v2_prefix_v81,
)
from semantic_3d_chat.scene_encoder.fixed_prefix_atlas import tensor_sha256
from semantic_3d_chat.training.train_question_control_v73 import (
    EXPECTED_HELD_CHANGED_SIDES,
    EXPECTED_HELD_CHANGED_UNITS,
    EXPECTED_HELD_ROWS,
    EXPECTED_HELD_SCENES,
    EXPECTED_ROWS,
    EXPECTED_TRAIN_ROWS,
    EXPECTED_TRAIN_SCENES,
    changed_units_v73,
    load_config_v73,
    load_training_rows_v73,
    split_rows_v73,
)

CONFIG: Final[str] = "configs/experiments/gemma4_v81_strict_fixed_prefix_reader.yaml"
EXPECTED_CONFIG_SHA256: Final[str] = (
    "e908cb50840d6b868abfbdf9b31a795a1efb0f0476b1b9463dd3cf74918960bb"
)
READER_SOURCE: Final[str] = "src/semantic_3d_chat/language/v81_structured_dense_atlas_sidecar.py"
PREREG_SOURCE: Final[str] = (
    "src/semantic_3d_chat/evaluation/v81_structured_dense_atlas_sidecar_preregistration.py"
)
PREFLIGHT_SOURCE: Final[str] = "scripts/preflight_v81_strict_fixed_prefix_reader.py"
LAUNCH_SOURCE: Final[str] = "scripts/run_v81_strict_fixed_prefix_reader.sh"
TEST_SOURCE: Final[str] = "tests/test_v81_strict_fixed_prefix_reader.py"
EXPECTED_V75_STATE_SHAPES: Final[dict[str, tuple[int, ...]]] = {
    "coefficient_hidden.weight": (768, 512),
    "coefficient_output.weight": (448, 768),
    "key.weight": (128, 1536),
    "output_basis": (112, 1536),
    "query.weight": (512, 1536),
    "value.weight": (128, 1536),
}
EXPECTED_TOP_LEVEL_FIELDS: Final[frozenset[str]] = frozenset(
    {
        "schema_version",
        "artifact",
        "status",
        "seed",
        "inputs",
        "split",
        "fixed_memory_contract",
        "stage_a_deterministic_reader",
        "latest_user_query_contract",
        "stage_b_optional_postdecoder_fusion",
        "training_only_teacher_contract",
        "internal_held_gates",
        "predictor_scorer_isolation",
        "conditional_gemma_smoke",
        "memory_safety",
        "candidate_artifact_policy",
        "scope",
        "outputs",
    }
)


def _resolve(path: str | Path) -> Path:
    value = Path(path).expanduser()
    return (value if value.is_absolute() else PROJECT_ROOT / value).resolve()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with _resolve(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value: object) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def atomic_create_json(path: str | Path, value: Mapping[str, Any]) -> tuple[Path, str]:
    destination = _resolve(path)
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"V81 create-once output exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = (
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n"
    ).encode("utf-8")
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
    return destination, hashlib.sha256(payload).hexdigest()


def load_v81_config(path: str | Path = CONFIG) -> dict[str, Any]:
    source = _resolve(path)
    if source != _resolve(CONFIG):
        raise ValueError("V81 refuses a noncanonical experiment config path")
    if source.is_symlink() or not source.is_file():
        raise FileNotFoundError(f"V81 config is unavailable: {source}")
    if sha256_file(source) != EXPECTED_CONFIG_SHA256:
        raise ValueError("V81 preregistered config bytes changed")
    payload = yaml.safe_load(source.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping) or set(payload) != {"v81"}:
        raise ValueError("V81 config must contain exactly one v81 mapping")
    config = payload["v81"]
    if not isinstance(config, Mapping) or set(config) != EXPECTED_TOP_LEVEL_FIELDS:
        raise ValueError("V81 top-level config fields changed")
    exact_identity = {
        "schema_version": 1,
        "artifact": "gemma4_v81_strict_fixed_prefix_reader",
        "status": ("preregistered_model_free_preflight_only_no_fit_or_gemma_load_authorized"),
        "seed": 810081,
    }
    if any(config.get(key) != value for key, value in exact_identity.items()):
        raise ValueError("V81 experiment identity changed")
    _validate_config_contract(config)
    return dict(config)


def _validate_config_contract(config: Mapping[str, Any]) -> None:
    memory = config["fixed_memory_contract"]
    if (
        memory.get("fixed_prefix_tokens") != FIXED_PREFIX_TOKENS
        or memory.get("atlas_group_count") != PROBE_COUNT
        or memory.get("atlas_memory_tokens") != ATLAS_MEMORY_TOKENS
        or memory.get("base_environment_latents") != BASE_ENVIRONMENT_LATENTS
        or memory.get("hidden_size") != HIDDEN_SIZE
        or memory.get("same_738_tokens_reused_byte_identically_for_every_question") is not True
        or memory.get("only_atlas_values_and_base_latents_are_payload") is not True
        or any(
            memory.get(field) is not False
            for field in (
                "boi_eoi_are_payload",
                "probe_keys_are_payload",
                "question_queries_are_payload",
                "question_dependent_environment_selection",
                "question_dependent_retrieval",
                "semantic_or_spatial_top_k_selection",
            )
        )
    ):
        raise ValueError("V81 fixed-memory contract changed")
    stage_a = config["stage_a_deterministic_reader"]
    selection = stage_a.get("train_only_scale_selection", {})
    thresholds = stage_a.get("development_acceptance_thresholds", {})
    if (
        float(stage_a.get("atlas_logit_scale", 0.0)) != RAW_ATLAS_LOGIT_SCALE
        or float(stage_a.get("atlas_uniform_floor_mass", 0.0)) != ATLAS_UNIFORM_FLOOR_MASS
        or stage_a.get("learned_query_or_key_enabled") is not False
        or stage_a.get("direct_four_control_tokens_exposed_before_any_fusion") is not True
        or selection.get("scale_candidate_grid")
        != [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 100.0, 120.0, 160.0]
        or selection.get("selection_metric") != "mean_direct_v75_control_cosine"
        or selection.get("selection_rule") != "argmax_then_smallest_scale_tie_break"
        or float(selection.get("selected_scale", 0.0)) != RAW_ATLAS_LOGIT_SCALE
        or selection.get("per_candidate_scores_recorded_here") is not False
        or thresholds.get("retroactively_preregistered") is not False
        or float(thresholds.get("mean_control_cosine_minimum", 0.0)) != 0.99
        or float(thresholds.get("normalized_mse_maximum", 1.0)) != 0.04
        or thresholds.get("row_count") != 384
    ):
        raise ValueError("V81 deterministic Stage A contract changed")
    query = config["latest_user_query_contract"]
    if query != {
        "source": "latest_user_question_only",
        "tokenization": "existing_question_token_ids",
        "add_special_tokens": False,
        "include_system_prompt": False,
        "include_conversation_history": False,
        "include_assistant_answer": False,
        "pooling": "mean_of_complete_nonempty_token_embedding_sequence",
        "input_embedding_source": "frozen_gemma_input_embedding_rows",
        "input_embedding_tensor_name": INPUT_EMBEDDING_TENSOR_NAME,
        "model_blob_identity": MODEL_BLOB_SHA256_IDENTITY,
        "output_dtype": "float32",
        "detached": True,
        "maximum_question_tokens_source": ("runtime_config_language_max_question_tokens"),
        "prequestion_hashes_bound_before_tokenization": [
            "fixed_738_prefix",
            "atlas_memory",
            "base_258_prefix",
        ],
        "bound_hashes_reasserted_before_each_dense_read": True,
    }:
        raise ValueError("V81 latest-user query contract changed")
    stage_b = config["stage_b_optional_postdecoder_fusion"]
    if (
        stage_b.get("authorized_now") is not False
        or stage_b.get("trainable_parameter_count") != TRAINABLE_PARAMETER_COUNT
        or set(stage_b.get("candidate_tensor_names", ())) != CANDIDATE_TENSOR_NAMES
        or float(stage_b.get("base_uniform_floor_mass", 0.0)) != BASE_UNIFORM_FLOOR_MASS
        or stage_b.get("bias") is not False
        or stage_b.get("gemma_backward") is not False
    ):
        raise ValueError("V81 quarantined Stage B contract changed")
    safety = config["memory_safety"]
    if any(
        safety.get(field) is not True
        for field in (
            "gemma_load_forbidden_in_preflight",
            "fit_forbidden_in_preflight",
            "mps_use_forbidden_in_preflight",
        )
    ):
        raise ValueError("V81 preflight safety contract changed")
    if config["training_only_teacher_contract"].get("authorized_now") is not False:
        raise ValueError("V81 fit became authorized")
    isolation = config["predictor_scorer_isolation"]
    if not all(bool(value) for value in isolation.values()):
        raise ValueError("V81 predictor/scorer isolation weakened")
    publication = config["candidate_artifact_policy"]
    forbidden_payload_fields = (
        "probe_bank_serialized",
        "atlas_values_serialized",
        "base_latents_serialized",
        "environmental_prefix_cache_serialized",
        "questions_serialized",
        "answers_serialized",
        "prototypes_serialized",
        "class_ids_serialized",
        "teacher_cache_serialized",
        "environmental_text_serialized",
        "checkpoint_publication_authorized",
        "runtime_publication_authorized",
    )
    if (
        publication.get("numeric_weights_only") is not True
        or publication.get("sanitized_metadata_only") is not True
        or any(publication.get(field) is not False for field in forbidden_payload_fields)
    ):
        raise ValueError("V81 candidate boundary changed")
    scope = config["scope"]
    if scope != {
        "historical_v73_training_pool_only": True,
        "official_validation_loaded": False,
        "official_test_loaded": False,
        "deferred_final_loaded": False,
        "oracle_loaded": False,
        "cloud_inference": False,
    }:
        raise ValueError("V81 protected scope changed")


def _validate_inputs(config: Mapping[str, Any]) -> dict[str, Any]:
    inputs = config["inputs"]
    pinned = {
        inputs["v80_terminal"]: inputs["v80_terminal_sha256"],
        inputs["v73_config"]: inputs["v73_config_sha256"],
        inputs["historical_training_qa"]: inputs["historical_training_qa_sha256"],
        inputs["v75_direct_candidate"]: inputs["v75_direct_candidate_sha256"],
        inputs["v75_numeric_probe_bank"]: inputs["v75_numeric_probe_bank_sha256"],
        inputs["v75_numeric_probe_metadata"]: inputs["v75_numeric_probe_metadata_sha256"],
        inputs["runtime_config"]: inputs["runtime_config_sha256"],
        Path(inputs["v54_checkpoint"]) / "adapter.safetensors": inputs["v54_adapter_sha256"],
        Path(inputs["v54_checkpoint"]) / "runtime_metadata.json": inputs[
            "v54_runtime_metadata_sha256"
        ],
        inputs["v73_numeric_baseline"]: inputs["v73_numeric_baseline_sha256"],
        inputs["v75_held_smoke_baseline"]: inputs["v75_held_smoke_baseline_sha256"],
        inputs["v75_fixed_atlas_score"]: inputs["v75_fixed_atlas_score_sha256"],
        inputs["v78_dense_reader_evidence"]: inputs["v78_dense_reader_evidence_sha256"],
    }
    observed: dict[str, str] = {}
    for path, expected in pinned.items():
        current = sha256_file(path)
        if current != expected:
            raise ValueError(f"V81 pinned input changed: {path}")
        observed[str(path)] = current

    candidate_path = _resolve(inputs["v75_direct_candidate"])
    if candidate_path.stat().st_size != inputs["v75_direct_candidate_size_bytes"]:
        raise ValueError("V81 direct V75 candidate size changed")
    with safe_open(str(candidate_path), framework="pt", device="cpu") as archive:
        candidate_names = tuple(archive.keys())
        candidate_shapes = {
            name: tuple(archive.get_slice(name).get_shape()) for name in candidate_names
        }
        candidate_metadata = dict(archive.metadata() or {})
    if candidate_shapes != EXPECTED_V75_STATE_SHAPES:
        raise ValueError("V81 direct V75 candidate tensor layout changed")
    if (
        candidate_metadata.get("controller_architecture") != "v75"
        or candidate_metadata.get("runtime_publication_artifact") != "false"
        or candidate_metadata.get("answer_codebook_serialized") != "false"
        or candidate_metadata.get("held_optimization_rows") != "0"
    ):
        raise ValueError("V81 direct V75 candidate provenance changed")

    probe_path = _resolve(inputs["v75_numeric_probe_bank"])
    with safe_open(str(probe_path), framework="pt", device="cpu") as archive:
        if set(archive.keys()) != {"probe_embeddings"}:
            raise ValueError("V81 numeric probe tensor inventory changed")
        probe_shape = tuple(archive.get_slice("probe_embeddings").get_shape())
    probes = load_file(str(probe_path), device="cpu")["probe_embeddings"].float()
    if (
        probe_shape != (PROBE_COUNT, HIDDEN_SIZE)
        or tensor_sha256(probes) != inputs["v75_numeric_probe_tensor_sha256"]
        or bool(torch.any(probes.norm(dim=-1) <= 1e-8))
    ):
        raise ValueError("V81 numeric probe bank changed")

    prefix_manifest_path = _resolve(inputs["v54_prefix_cache"]) / "manifest.json"
    if sha256_file(prefix_manifest_path) != inputs["v54_prefix_manifest_sha256"]:
        raise ValueError("V81 prefix manifest changed")
    prefix_manifest = json.loads(prefix_manifest_path.read_text(encoding="utf-8"))
    if (
        prefix_manifest.get("artifact") != "question_independent_scene_prefix_cache_v1"
        or prefix_manifest.get("scene_count") != 40
        or prefix_manifest.get("question_inputs_used") is not False
        or prefix_manifest.get("question_dependent_scene_retrieval") is not False
        or prefix_manifest.get("complete_scene_prefixes") is not True
        or prefix_manifest.get("environmental_text_inputs") != []
    ):
        raise ValueError("V81 base prefix cache contract changed")
    prefix_files: list[list[Any]] = []
    for scene_id, record in sorted(prefix_manifest["scenes"].items()):
        path = _resolve(inputs["v54_prefix_cache"]) / record["filename"]
        if path.is_symlink() or sha256_file(path) != record["file_sha256"]:
            raise ValueError(f"V81 cached prefix changed: {scene_id}")
        with safe_open(str(path), framework="pt", device="cpu") as archive:
            if set(archive.keys()) != {"scene_prefix"}:
                raise ValueError(f"V81 cached prefix tensor changed: {scene_id}")
            shape = tuple(archive.get_slice("scene_prefix").get_shape())
            dtype = archive.get_slice("scene_prefix").get_dtype()
        if shape != (1, BASE_PREFIX_TOKENS, HIDDEN_SIZE) or dtype != "BF16":
            raise ValueError(f"V81 cached prefix shape/dtype changed: {scene_id}")
        prefix_files.append([scene_id, record["file_sha256"], list(shape), dtype])

    return {
        "authenticated_file_sha256": observed,
        "direct_v75_candidate_tensor_shapes": {
            key: list(value) for key, value in candidate_shapes.items()
        },
        "numeric_probe_shape": list(probe_shape),
        "numeric_probe_tensor_sha256": tensor_sha256(probes),
        "base_prefix_manifest_sha256": sha256_file(prefix_manifest_path),
        "base_prefix_scene_count": len(prefix_files),
        "base_prefix_inventory_sha256": canonical_sha256(prefix_files),
        "all_base_prefix_files_authenticated_without_tensor_materialization": True,
    }


def _validate_model_identity(config: Mapping[str, Any]) -> dict[str, Any]:
    inputs = config["inputs"]
    snapshot = (
        Path.home()
        / ".cache/huggingface/hub/models--google--gemma-4-E2B-it/snapshots"
        / inputs["model_revision"]
    )
    model_link = snapshot / "model.safetensors"
    config_path = snapshot / "config.json"
    if not model_link.exists() or not config_path.is_file():
        raise FileNotFoundError("V81 pinned local Gemma snapshot is incomplete")
    resolved_blob = model_link.resolve(strict=True)
    if resolved_blob.name != inputs["model_file_sha256_identity"]:
        raise ValueError("V81 local Gemma blob identity changed")
    model_config = json.loads(config_path.read_text(encoding="utf-8"))
    text = model_config.get("text_config")
    if (
        not isinstance(text, Mapping)
        or text.get("hidden_size") != HIDDEN_SIZE
        or text.get("num_hidden_layers") != 35
        or text.get("num_kv_shared_layers") != 20
        or text.get("sliding_window") != 512
    ):
        raise ValueError("V81 pinned Gemma topology changed")
    return {
        "model_id": inputs["model_id"],
        "revision": inputs["model_revision"],
        "model_blob_sha256_identity": resolved_blob.name,
        "hidden_size": text["hidden_size"],
        "num_hidden_layers": text["num_hidden_layers"],
        "model_file_opened": False,
        "model_instantiated": False,
    }


def _historical_inventory(config: Mapping[str, Any]) -> dict[str, Any]:
    inputs = config["inputs"]
    v73 = load_config_v73(inputs["v73_config"])
    if _resolve(v73["training_qa"]) != _resolve(inputs["historical_training_qa"]):
        raise ValueError("V81 and V73 historical QA paths differ")
    rows = load_training_rows_v73(inputs["historical_training_qa"])
    train, held = split_rows_v73(rows)
    train_scenes = {row.scene_id for row in train}
    held_scenes = {row.scene_id for row in held}
    train_pairs = {row.pair_id for row in train}
    held_pairs = {row.pair_id for row in held}
    train_classes = {row.answer_class for row in train}
    supported = tuple(row for row in held if row.answer_class in train_classes)
    held_units = changed_units_v73(held)
    # Pairwise change metrics require both answer classes to be represented by
    # the train-only bank.  A unit with one unsupported side excludes both.
    supported_changed_sides = 2 * sum(
        unit.left.answer_class in train_classes and unit.right.answer_class in train_classes
        for unit in held_units
    )
    smoke = select_held_smoke_v80(held)
    broad = select_broad_held_v80(held)
    train_questions = {row.question for row in train}
    held_questions = {row.question for row in held}
    unseen_rows = tuple(row for row in held if row.question not in train_questions)
    unseen_changed = sum(row.expected_change for row in unseen_rows)
    smoke_inventory = [[unit.change_type, unit.pair_id, unit.question_key] for unit in smoke]
    broad_inventory = [[row.answer_type, row.scene_id, row.question_id] for row in broad]
    result = {
        "all_rows": len(rows),
        "train_rows": len(train),
        "train_scenes": len(train_scenes),
        "train_pairs": len(train_pairs),
        "train_unique_question_texts": len(train_questions),
        "train_answer_classes": len(train_classes),
        "held_rows": len(held),
        "held_scenes": len(held_scenes),
        "held_pairs": len(held_pairs),
        "held_unique_question_texts": len(held_questions),
        "held_answer_classes": len({row.answer_class for row in held}),
        "supported_held_rows": len(supported),
        "unsupported_held_rows": len(held) - len(supported),
        "held_changed_units": len(held_units),
        "held_changed_sides": 2 * len(held_units),
        "supported_held_changed_sides": supported_changed_sides,
        "shared_question_texts": len(train_questions & held_questions),
        "unseen_held_question_texts": len(held_questions - train_questions),
        "unseen_held_question_rows": len(unseen_rows),
        "unseen_held_changed_rows": unseen_changed,
        "held_smoke_units": len(smoke),
        "held_smoke_sides": 2 * len(smoke),
        "held_smoke_sha256": canonical_sha256(smoke_inventory),
        "broad_held_rows": len(broad),
        "broad_held_sha256": canonical_sha256(broad_inventory),
        "pair_disjoint": train_pairs.isdisjoint(held_pairs),
        "scene_disjoint": train_scenes.isdisjoint(held_scenes),
        "question_disjoint": train_questions.isdisjoint(held_questions),
    }
    expected = {
        "all_rows": EXPECTED_ROWS,
        "train_rows": EXPECTED_TRAIN_ROWS,
        "train_scenes": EXPECTED_TRAIN_SCENES,
        "train_pairs": 12,
        "train_unique_question_texts": 96,
        "train_answer_classes": 28,
        "held_rows": EXPECTED_HELD_ROWS,
        "held_scenes": EXPECTED_HELD_SCENES,
        "held_pairs": 8,
        "held_unique_question_texts": 80,
        "held_answer_classes": 27,
        "supported_held_rows": 383,
        "unsupported_held_rows": 1,
        "held_changed_units": EXPECTED_HELD_CHANGED_UNITS,
        "held_changed_sides": EXPECTED_HELD_CHANGED_SIDES,
        "supported_held_changed_sides": 50,
        "shared_question_texts": 62,
        "unseen_held_question_texts": 18,
        "unseen_held_question_rows": 44,
        "unseen_held_changed_rows": 8,
        "held_smoke_units": 8,
        "held_smoke_sides": 16,
        "held_smoke_sha256": EXPECTED_HELD_SMOKE_SHA256,
        "broad_held_rows": 16,
        "broad_held_sha256": EXPECTED_BROAD_HELD_SHA256,
        "pair_disjoint": True,
        "scene_disjoint": True,
        "question_disjoint": False,
    }
    if result != expected:
        raise RuntimeError(f"V81 historical inventory changed: {result}")
    return result


def _maximum_rss_bytes() -> int:
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value if platform.system() == "Darwin" else value * 1024


def run_synthetic_cpu_preflight() -> dict[str, Any]:
    """Exercise exact role, floor, zero, and gradient-isolation contracts."""

    torch.manual_seed(810081)
    prefix = torch.randn(1, FIXED_PREFIX_TOKENS, HIDDEN_SIZE) * 0.02
    question = torch.randn(1, HIDDEN_SIZE)
    decoder = torch.randn(1, HIDDEN_SIZE) * 0.02
    prefix_before = prefix.detach().clone()
    audit = audit_v75_v2_prefix_v81(prefix)
    binding = bind_fixed_prefix_before_question_v81(prefix)

    class _SyntheticTokenizer:
        def __init__(self) -> None:
            self.calls: list[dict[str, Any]] = []

        def __call__(self, text: str, **kwargs: Any) -> dict[str, torch.Tensor]:
            self.calls.append({"text": text, **kwargs})
            return {"input_ids": torch.tensor([[3, 5, 7]], dtype=torch.long)}

    tokenizer = _SyntheticTokenizer()
    embedding = torch.nn.Embedding(16, HIDDEN_SIZE)
    embedding.requires_grad_(False)
    latest_query = latest_user_question_query_v81(
        tokenizer=tokenizer,
        embedding_layer=embedding,
        latest_user_question="latest user only",
        device=torch.device("cpu"),
        maximum_question_tokens=16,
        model_blob_sha256_identity=MODEL_BLOB_SHA256_IDENTITY,
        embedding_tensor_name=INPUT_EMBEDDING_TENSOR_NAME,
    )
    stage_a = deterministic_atlas_read_v81(prefix, question, binding=binding)
    banks = split_v75_v2_prefix_v81(prefix)
    manual_logits = (
        torch.einsum(
            "bd,bpd->bp",
            F.normalize(question.float(), dim=-1),
            F.normalize(banks.probe_keys.float(), dim=-1),
        )
        * RAW_ATLAS_LOGIT_SCALE
    )
    manual_weights = ATLAS_UNIFORM_FLOOR_MASS / PROBE_COUNT + (
        1.0 - ATLAS_UNIFORM_FLOOR_MASS
    ) * torch.softmax(manual_logits, dim=-1)
    manual_controls = torch.einsum("bp,bpvh->bvh", manual_weights, banks.atlas_values.float())
    bound_hash = prefix_sha256(prefix)
    assert_fixed_prefix_identity_v81(prefix, expected_sha256=bound_hash)
    _ = deterministic_atlas_read_v81(
        prefix,
        torch.randn_like(question),
        binding=binding,
    )
    assert_fixed_prefix_identity_v81(prefix, expected_sha256=bound_hash)
    assert_prefix_binding_v81(prefix, binding=binding)

    zero_prefix = torch.zeros_like(prefix)
    zero_binding = bind_fixed_prefix_before_question_v81(zero_prefix)
    zero_stage_a = deterministic_atlas_read_v81(
        zero_prefix,
        torch.zeros_like(question),
        binding=zero_binding,
    )

    sidecar = StructuredDenseAtlasSidecarV81(allow_stage_b=True).cpu()
    disabled = sidecar(prefix, question, decoder, binding=binding)
    enabled_initial = sidecar(
        prefix,
        question,
        decoder,
        binding=binding,
        enable_stage_b=True,
    )

    with torch.no_grad():
        sidecar.residual_output.weight.normal_(mean=0.0, std=0.01)
    zero_payload_prefix = prefix.detach().clone()
    zero_banks = split_v75_v2_prefix_v81(zero_payload_prefix)
    zero_banks.atlas_values.zero_()
    zero_banks.base_latents.zero_()
    zero_payload = sidecar(
        zero_payload_prefix,
        question,
        decoder,
        binding=bind_fixed_prefix_before_question_v81(zero_payload_prefix),
        enable_stage_b=True,
    )

    grad_prefix = prefix.detach().clone().requires_grad_(True)
    grad_question = question.detach().clone().requires_grad_(True)
    grad_decoder = decoder.detach().clone().requires_grad_(True)
    gradient_output = sidecar(
        grad_prefix,
        grad_question,
        grad_decoder,
        binding=bind_fixed_prefix_before_question_v81(grad_prefix.detach()),
        enable_stage_b=True,
    )
    frozen_lm_head = torch.nn.Linear(HIDDEN_SIZE, 31, bias=False)
    frozen_lm_head.requires_grad_(False)
    logits = frozen_lm_head_logits_v81(
        gradient_output.fused_hidden,
        frozen_lm_head=frozen_lm_head,
    )
    logits.square().mean().backward()
    parameter_gradients = {
        name: (None if parameter.grad is None else float(parameter.grad.detach().float().norm()))
        for name, parameter in sidecar.named_parameters()
    }

    metadata = sanitized_candidate_metadata_v81(weights_sha256="0" * 64)
    state = sidecar.candidate_state_dict()
    checks = {
        "device_cpu": prefix.device.type == "cpu",
        "fixed_prefix_shape_exact": tuple(prefix.shape) == (1, FIXED_PREFIX_TOKENS, HIDDEN_SIZE),
        "fixed_prefix_lossless_role_parse": audit.exact_reconstruction,
        "boundary_tokens_audited_not_payload": (
            audit.boundary_tokens_retained and not audit.boi_eoi_are_payload
        ),
        "probe_and_query_address_only": (
            not audit.probe_keys_are_payload and not audit.question_queries_are_payload
        ),
        "only_scene_banks_payload": audit.only_scene_values_and_base_latents_are_payload,
        "base258_reconstruction_exact": torch.equal(
            reconstruct_base_v54_prefix_v81(prefix),
            torch.cat((prefix[:, :1], prefix[:, 481:737], prefix[:, -1:]), dim=1),
        ),
        "stage_a_direct_four_control_shape": tuple(stage_a.reconstructed_controls.shape)
        == (1, VALUES_PER_PROBE, HIDDEN_SIZE),
        "stage_a_fixed_scale160_manual_equivalence": torch.equal(
            stage_a.atlas_weights, manual_weights
        )
        and torch.equal(stage_a.reconstructed_controls, manual_controls),
        "stage_a_all_96_groups_positive_floor": float(stage_a.atlas_weights.min())
        >= MINIMUM_ATLAS_WEIGHT - 1e-9,
        "stage_a_weights_sum_one": torch.allclose(
            stage_a.atlas_weights.sum(dim=-1), torch.ones(1), atol=1e-6, rtol=0.0
        ),
        "all_zero_738_exact_zero_stage_a": int(
            torch.count_nonzero(zero_stage_a.reconstructed_controls)
        )
        == 0
        and torch.equal(
            zero_stage_a.atlas_weights,
            torch.full_like(zero_stage_a.atlas_weights, 1.0 / PROBE_COUNT),
        ),
        "latest_user_query_exact_contract": (
            tokenizer.calls
            == [
                {
                    "text": "latest user only",
                    "add_special_tokens": False,
                    "return_tensors": "pt",
                }
            ]
            and latest_query.token_count == 3
            and latest_query.add_special_tokens is False
            and latest_query.included_system_prompt is False
            and latest_query.included_history is False
            and latest_query.included_answer is False
            and latest_query.detached is True
            and latest_query.query.dtype == torch.float32
            and torch.equal(
                latest_query.query,
                embedding(latest_query.token_ids).float().mean(dim=1),
            )
        ),
        "same_memory_byte_identical_across_questions": torch.equal(prefix, prefix_before)
        and prefix_sha256(prefix) == bound_hash,
        "stage_b_disabled_exact_noop": torch.equal(disabled.fused_hidden, decoder.float())
        and int(torch.count_nonzero(disabled.residual)) == 0,
        "stage_b_zero_initialized_exact_noop": torch.equal(
            enabled_initial.fused_hidden, decoder.float()
        )
        and int(torch.count_nonzero(enabled_initial.residual)) == 0,
        "stage_b_all_96_groups_positive_floor": float(
            enabled_initial.learned_atlas_weights.detach().min()
        )
        >= MINIMUM_ATLAS_WEIGHT - 1e-9,
        "stage_b_all_256_base_latents_positive_floor": float(
            enabled_initial.base_weights.detach().min()
        )
        >= MINIMUM_BASE_WEIGHT - 1e-9,
        "zero_environmental_payload_exact_zero_residual": int(
            torch.count_nonzero(zero_payload.residual)
        )
        == 0,
        "frozen_inputs_detached_no_backward": (
            grad_prefix.grad is None and grad_question.grad is None and grad_decoder.grad is None
        ),
        "sidecar_parameter_gradient_exists": any(
            value is not None and value > 0.0 for value in parameter_gradients.values()
        ),
        "residual_applied_before_frozen_lm_head": tuple(logits.shape) == (1, 31)
        and all(parameter.grad is None for parameter in frozen_lm_head.parameters()),
        "bias_free": all(
            module.bias is None
            for module in sidecar.modules()
            if isinstance(module, torch.nn.Linear)
        ),
        "candidate_tensor_inventory_exact": set(state) == CANDIDATE_TENSOR_NAMES,
        "candidate_parameter_count_exact": sum(value.numel() for value in state.values())
        == TRAINABLE_PARAMETER_COUNT,
        "metadata_sanitized": all(
            metadata[field] is False
            for field in (
                "probe_bank_serialized",
                "atlas_values_serialized",
                "base_latents_serialized",
                "environmental_prefix_cache_serialized",
                "questions_serialized",
                "answers_serialized",
                "prototypes_serialized",
                "class_ids_serialized",
                "teacher_cache_serialized",
                "prediction_cache_serialized",
                "environmental_text_serialized",
                "runtime_publication_authorized",
            )
        ),
        "gemma_not_loaded": True,
        "fit_not_executed": True,
        "mps_not_used": True,
    }
    if not all(value is True for value in checks.values()):
        failed = sorted(key for key, value in checks.items() if value is not True)
        raise RuntimeError(f"V81 synthetic CPU preflight failed: {failed}")
    return {
        "artifact": "gemma4_v81_structured_dense_atlas_sidecar_synthetic_cpu_v1",
        "passed": True,
        "checks": checks,
        "fixed_prefix_audit": audit.as_dict(),
        "stage_a": {
            "logit_scale": RAW_ATLAS_LOGIT_SCALE,
            "uniform_floor_mass": ATLAS_UNIFORM_FLOOR_MASS,
            "minimum_weight": float(stage_a.atlas_weights.min()),
            "maximum_weight": float(stage_a.atlas_weights.max()),
            "control_shape": list(stage_a.reconstructed_controls.shape),
        },
        "stage_b": {
            "authorized": False,
            "synthetic_contract_exercised_only": True,
            "atlas_minimum_weight": float(enabled_initial.learned_atlas_weights.detach().min()),
            "base_minimum_weight": float(enabled_initial.base_weights.detach().min()),
            "parameter_count": TRAINABLE_PARAMETER_COUNT,
            "candidate_numeric_bytes_fp32": sum(
                value.numel() * value.element_size() for value in state.values()
            ),
            "parameter_gradient_l2": parameter_gradients,
        },
        "maximum_process_rss_bytes": _maximum_rss_bytes(),
        "full_gemma_loaded": False,
        "optimizer_constructed": False,
        "fit_executed": False,
        "mps_used": False,
    }


def build_preregistration(config: Mapping[str, Any]) -> dict[str, Any]:
    observation = config["stage_a_deterministic_reader"][
        "post_selection_full_held_development_evidence"
    ]
    thresholds = config["stage_a_deterministic_reader"]["development_acceptance_thresholds"]
    return {
        "schema_version": 1,
        "artifact": "gemma4_v81_strict_fixed_prefix_reader_preregistration_v1",
        "status": "sealed_development_contract_no_fit_no_gemma_load",
        "config_path": CONFIG,
        "config_sha256": EXPECTED_CONFIG_SHA256,
        "implementation_sha256": {
            READER_SOURCE: sha256_file(READER_SOURCE),
            PREREG_SOURCE: sha256_file(PREREG_SOURCE),
            PREFLIGHT_SOURCE: sha256_file(PREFLIGHT_SOURCE),
            LAUNCH_SOURCE: sha256_file(LAUNCH_SOURCE),
            TEST_SOURCE: sha256_file(TEST_SOURCE),
        },
        "fixed_memory": dict(config["fixed_memory_contract"]),
        "stage_a": {
            "contract": dict(config["stage_a_deterministic_reader"]),
            "scale_selected_on_train_only_before_full_held_measurement": True,
            "full_held_observation_is_post_selection_development_evidence": True,
            "development_thresholds_are_not_retroactively_preregistered": True,
            "observation_passes_development_thresholds": (
                observation["mean_control_cosine"] >= thresholds["mean_control_cosine_minimum"]
                and observation["normalized_mse"] <= thresholds["normalized_mse_maximum"]
                and observation["seen_question_mean_cosine"]
                >= thresholds["seen_question_mean_cosine_minimum"]
                and observation["unseen_question_mean_cosine"]
                >= thresholds["unseen_question_mean_cosine_minimum"]
            ),
        },
        "stage_b": dict(config["stage_b_optional_postdecoder_fusion"]),
        "training_only_teacher_contract": dict(config["training_only_teacher_contract"]),
        "internal_held_gates": dict(config["internal_held_gates"]),
        "predictor_scorer_isolation": dict(config["predictor_scorer_isolation"]),
        "conditional_gemma_smoke": dict(config["conditional_gemma_smoke"]),
        "memory_safety": dict(config["memory_safety"]),
        "candidate_artifact_policy": dict(config["candidate_artifact_policy"]),
        "scope": dict(config["scope"]),
        "model_loaded": False,
        "fit_executed": False,
        "mps_used": False,
        "checkpoint_published": False,
        "runtime_promotion_authorized": False,
    }


def run_cpu_preflight(config_path: str | Path = CONFIG) -> dict[str, Any]:
    config = load_v81_config(config_path)
    inputs = _validate_inputs(config)
    historical = _historical_inventory(config)
    model_identity = _validate_model_identity(config)
    synthetic = run_synthetic_cpu_preflight()
    observation = config["stage_a_deterministic_reader"][
        "post_selection_full_held_development_evidence"
    ]
    thresholds = config["stage_a_deterministic_reader"]["development_acceptance_thresholds"]
    future_outputs = {
        key: _resolve(value).exists()
        for key, value in config["outputs"].items()
        if key.startswith(("future_", "prohibited_"))
    }
    checks = {
        "config_sealed": True,
        "pinned_inputs_authenticated": True,
        "all_40_base_prefix_files_authenticated": inputs["base_prefix_scene_count"] == 40,
        "historical_pair_scene_disjoint_split_exact": historical["pair_disjoint"]
        and historical["scene_disjoint"],
        "question_overlap_explicit_and_subgroups_locked": (
            not historical["question_disjoint"]
            and historical["shared_question_texts"] == 62
            and historical["unseen_held_question_texts"] == 18
        ),
        "stage_a_postselection_observation_passes_development_thresholds": (
            observation["mean_control_cosine"] >= thresholds["mean_control_cosine_minimum"]
            and observation["normalized_mse"] <= thresholds["normalized_mse_maximum"]
            and observation["seen_question_mean_cosine"]
            >= thresholds["seen_question_mean_cosine_minimum"]
            and observation["unseen_question_mean_cosine"]
            >= thresholds["unseen_question_mean_cosine_minimum"]
        ),
        "synthetic_cpu_contract_passed": synthetic["passed"] is True,
        "future_candidate_prediction_score_and_runtime_outputs_absent": not any(
            future_outputs.values()
        ),
        "full_gemma_not_loaded": model_identity["model_instantiated"] is False,
        "fit_not_executed": True,
        "mps_not_used": True,
        "runtime_publication_forbidden": config["candidate_artifact_policy"][
            "runtime_publication_authorized"
        ]
        is False,
    }
    if not all(checks.values()):
        failed = sorted(key for key, value in checks.items() if not value)
        raise RuntimeError(f"V81 CPU preflight failed: {failed}")
    return {
        "schema_version": 1,
        "artifact": "gemma4_v81_strict_fixed_prefix_reader_cpu_preflight_v1",
        "status": "cpu_preflight_pass_no_fit_or_gemma_smoke_authorized",
        "passed": True,
        "config_sha256": EXPECTED_CONFIG_SHA256,
        "checks": checks,
        "inputs": inputs,
        "historical_inventory": historical,
        "model_identity_metadata_only": model_identity,
        "synthetic_cpu": synthetic,
        "future_output_existence": future_outputs,
        "real_model": {
            "loaded": False,
            "weights_file_opened": False,
            "gradients_enabled": False,
            "optimizer_constructed": False,
            "fit_executed": False,
        },
        "official_validation_loaded": False,
        "official_test_loaded": False,
        "deferred_final_loaded": False,
        "oracle_loaded": False,
        "checkpoint_published": False,
        "runtime_promotion_authorized": False,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=CONFIG)
    parser.add_argument("--write-preregistration", action="store_true")
    parser.add_argument("--write-cpu-preflight", action="store_true")
    args = parser.parse_args(argv)
    config = load_v81_config(args.config)
    preregistration = build_preregistration(config)
    result: dict[str, Any] = {"preregistration": preregistration}
    if args.write_preregistration:
        path, digest = atomic_create_json(config["outputs"]["preregistration"], preregistration)
        result["preregistration_output"] = {"path": str(path), "sha256": digest}
    preflight = run_cpu_preflight(args.config)
    result["cpu_preflight"] = preflight
    if args.write_cpu_preflight:
        path, digest = atomic_create_json(config["outputs"]["cpu_preflight"], preflight)
        result["cpu_preflight_output"] = {"path": str(path), "sha256": digest}
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "CONFIG",
    "EXPECTED_CONFIG_SHA256",
    "atomic_create_json",
    "build_preregistration",
    "canonical_sha256",
    "load_v81_config",
    "main",
    "run_cpu_preflight",
    "run_synthetic_cpu_preflight",
    "sha256_file",
]
