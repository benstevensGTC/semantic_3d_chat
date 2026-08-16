"""Independent post-training strong causal controls for the V94 candidate.

The evaluator is intentionally outside V94's sealed preregistered evaluator.
It is diagnostic-only and cannot authorize promotion.  It has three process
boundaries:

``compile-controls``
    Authenticates the completed V94 training candidate and the attested six
    numeric source memories, then rebuilds complete 738-token memories from
    targeted numeric-map interventions *before opening any question file*.
``predict``
    Binds every original and controlled memory before opening the sanitized
    three-field question manifest.  Reference labels and oracle roots are
    blocked by a file-access audit.  The same fixed memory for an arm/scene is
    reused for all questions.
``score``
    Authenticates the completed prediction/access bundle first, then opens the
    pinned validation references in this separate model-free process and emits
    aggregate metrics only.

The two strongest targeted interventions happen before scene tokenization:

* ``semantic_payload_shuffle`` permutes all 3,072-D semantic voxel rows while
  leaving XYZ, RGB, normals, confidence, and observation counts byte-equal;
* ``position_spatial_shuffle`` permutes only voxel XYZ while retaining every
  semantic and non-position payload row byte-equal.

The direct-memory core also includes a broad destruction control that permutes
all 736 continuous environmental tokens while preserving their row multiset
and Gemma's native BOI/EOI boundaries. A fixed, label-blind 36-question profile
can run those direct-memory controls without compiling the costlier map arms.

RGB is an independently identifiable, nonzero input to the point-token
projection, so a zero-RGB control is included. Although the architecture also
accepts normals, all six sealed evaluation maps contain exactly zero normal
values; removing them would be an identical-input no-op and is explicitly
unsupported for this artifact. View direction exists in the raw fused map but
is not exposed by ``MapTensorData`` or consumed by the current point-token
projection, so viewpoint removal is likewise explicitly unsupported.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import sys
import tempfile
import time
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file

from semantic_3d_chat.chat.file_audit import FileAccessAudit
from semantic_3d_chat.chat.runtime import StaticChatRuntime
from semantic_3d_chat.chat.runtime_config import load_runtime_config
from semantic_3d_chat.config import PROJECT_ROOT, project_path
from semantic_3d_chat.evaluation import v94_strict_multiscene_evidence as v94_evidence
from semantic_3d_chat.evaluation.ablations import deterministic_permutation, file_sha256
from semantic_3d_chat.evaluation.baseline_io import atomic_write_jsonl, read_jsonl
from semantic_3d_chat.evaluation.control_predict import apply_map_control
from semantic_3d_chat.evaluation.question_manifest import (
    QuestionManifest,
    load_question_manifest,
    questions_sha256,
)
from semantic_3d_chat.evaluation.v56_fresh_development_score import (
    EXPECTED_TYPE_COUNTS,
    canonical_answer_key,
    canonical_type_specific_match,
)
from semantic_3d_chat.evaluation.v75_fixed_atlas_behavior import _load_probe_bank
from semantic_3d_chat.evaluation.v75_official_validation_contract import (
    EXPECTED_REFERENCE_SHA256,
    authenticate_v75_control_checkpoint,
)
from semantic_3d_chat.language.lora import tensor_state_sha256
from semantic_3d_chat.language.prefix_injection import prefix_sha256
from semantic_3d_chat.scene_encoder.fixed_prefix_atlas_v75 import (
    compile_fixed_scene_atlas_v75_v2,
)
from semantic_3d_chat.scene_encoder.map_io import MapTensorData, load_map_tensors

SCHEMA_VERSION: Final[int] = 1
ARTIFACT: Final[str] = "gemma4_v94_strong_causal_ablations_v1"
CACHE_ARTIFACT: Final[str] = "v94_strong_causal_memory_cache_v1"
PREDICTION_ARTIFACT: Final[str] = "v94_strong_causal_question_only_predictions_v1"
SCORE_ARTIFACT: Final[str] = "v94_strong_causal_label_isolated_score_v1"
SEED: Final[int] = 940_195
SCENE_IDS: Final[tuple[str, ...]] = tuple(
    f"scene_{index:06d}" for index in range(57, 63)
)
PAIR_SCENE: Final[dict[str, str]] = {
    "scene_000057": "scene_000058",
    "scene_000058": "scene_000057",
    "scene_000059": "scene_000060",
    "scene_000060": "scene_000059",
    "scene_000061": "scene_000062",
    "scene_000062": "scene_000061",
}
MEMORY_SHAPE: Final[tuple[int, int, int]] = (1, 738, 1536)
QUESTION_COUNT: Final[int] = 216

PRIMARY: Final[str] = "primary"
ZERO_FULL_SCENE: Final[str] = "zero_full_scene"
WRONG_SCENE_SWAP: Final[str] = "wrong_scene_swap"
FULL_INTERIOR_TOKEN_PERMUTATION: Final[str] = "full_interior_token_permutation"
SEMANTIC_PAYLOAD_SHUFFLE: Final[str] = "semantic_payload_shuffle"
POSITION_SPATIAL_SHUFFLE: Final[str] = "position_spatial_shuffle"
REMOVE_RGB: Final[str] = "remove_rgb"
REMOVE_NORMALS: Final[str] = "remove_normals"
CORE_CONDITIONS: Final[tuple[str, ...]] = (
    PRIMARY,
    ZERO_FULL_SCENE,
    WRONG_SCENE_SWAP,
    FULL_INTERIOR_TOKEN_PERMUTATION,
)
# Cache manifests are serialized with sorted JSON keys. Keep this tuple in the
# same canonical lexical order so authentication never relies on pre-JSON dict
# insertion order.
COMPILED_CONDITIONS: Final[tuple[str, ...]] = (
    POSITION_SPATIAL_SHUFFLE,
    REMOVE_RGB,
    SEMANTIC_PAYLOAD_SHUFFLE,
)
CONDITIONS: Final[tuple[str, ...]] = CORE_CONDITIONS + COMPILED_CONDITIONS
_MAP_CONTROL: Final[dict[str, str]] = {
    SEMANTIC_PAYLOAD_SHUFFLE: "semantic_shuffle",
    POSITION_SPATIAL_SHUFFLE: "position_shuffle",
    REMOVE_RGB: "remove_rgb",
}

_SEALED_NORMAL_VOXEL_COUNTS: Final[dict[str, int]] = {
    "scene_000057": 8608,
    "scene_000058": 8607,
    "scene_000059": 8606,
    "scene_000060": 8603,
    "scene_000061": 8688,
    "scene_000062": 8677,
}
_PREDICTION_FIELD: Final[dict[str, str]] = {
    condition: f"{condition}_prediction" for condition in CONDITIONS
}
_HASH_FIELD: Final[dict[str, str]] = {
    condition: f"{condition}_memory_sha256" for condition in CONDITIONS
}

CONFIG: Final[Path] = (
    PROJECT_ROOT / "configs/experiments/gemma4_v94_strict_multiscene_full40.yaml"
)
CONTROL_CACHE: Final[Path] = (
    PROJECT_ROOT / "reports/gemma4/artifacts/v94_strong_causal/evaluation_cache"
)
COMPILE_ACCESS: Final[Path] = (
    PROJECT_ROOT / "reports/gemma4/metrics/v94_strong_causal_compile_access.json"
)
COMPILE_RECEIPT: Final[Path] = (
    PROJECT_ROOT / "reports/gemma4/metrics/v94_strong_causal_compile_receipt.json"
)

_SHA256: Final[re.Pattern[str]] = re.compile(r"[0-9a-f]{64}")
_CACHE_TENSOR_METADATA: Final[dict[str, str]] = {
    "artifact": CACHE_ARTIFACT,
    "environmental_text_serialized": "false",
    "question_inputs_used": "false",
}


@dataclass(frozen=True)
class EvaluationProfile:
    name: str
    question_count: int
    questions_per_scene: int
    conditions: tuple[str, ...]
    output_stem: str


PROFILES: Final[dict[str, EvaluationProfile]] = {
    "representative-core": EvaluationProfile(
        name="representative-core",
        question_count=36,
        questions_per_scene=6,
        conditions=CORE_CONDITIONS,
        output_stem="v94_strong_causal_representative_core",
    ),
    "full": EvaluationProfile(
        name="full",
        question_count=QUESTION_COUNT,
        questions_per_scene=36,
        conditions=CONDITIONS,
        output_stem="v94_strong_causal",
    ),
}


@dataclass(frozen=True)
class EvaluationPaths:
    predictions: Path
    provenance: Path
    access: Path
    completion: Path
    score: Path


def evaluation_profile(name: str) -> EvaluationProfile:
    try:
        return PROFILES[name]
    except KeyError as exc:
        raise ValueError(f"Unknown V94 causal evaluation profile: {name}") from exc


def evaluation_paths(profile: EvaluationProfile) -> EvaluationPaths:
    predictions = (
        PROJECT_ROOT
        / "reports/gemma4/predictions"
        / f"{profile.output_stem}_question_only.jsonl"
    )
    return EvaluationPaths(
        predictions=predictions,
        provenance=predictions.with_name(f"{predictions.name}.provenance.json"),
        access=predictions.with_name(f"{predictions.name}.access.json"),
        completion=predictions.with_name(f"{predictions.name}.completion.json"),
        score=(
            PROJECT_ROOT
            / "reports/gemma4/metrics"
            / f"{profile.output_stem}_ablations.json"
        ),
    )


def _prediction_fields(profile: EvaluationProfile) -> frozenset[str]:
    return frozenset(
        {
            "artifact",
            "evaluation_profile",
            "scene_id",
            "question_id",
            "wrong_scene_id",
            *(_PREDICTION_FIELD[condition] for condition in profile.conditions),
            *(_HASH_FIELD[condition] for condition in profile.conditions),
            "all_memory_hashes_unchanged",
            "elapsed_seconds",
            "provenance_sha256",
        }
    )


def _canonical_sha256(value: object) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _require_sha256(value: object, label: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"V94 causal evaluator requires SHA-256 for {label}")
    return value


def _read_json(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise FileNotFoundError(path)

    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"Duplicate JSON field in {path}: {key}")
            result[key] = value
        return result

    value = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=reject_duplicates)
    if not isinstance(value, dict):
        raise TypeError(f"Expected one JSON object: {path}")
    return value


def _write_json_create_once(path: Path, payload: Mapping[str, Any]) -> None:
    if path.exists() or path.is_symlink():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(dict(payload), handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def _tensor_sha256(value: torch.Tensor) -> str:
    return tensor_state_sha256({"value": value.detach().cpu().contiguous()})


def _map_hashes(data: MapTensorData) -> dict[str, str]:
    return {
        name: _tensor_sha256(getattr(data, name))
        for name in (
            "semantic",
            "xyz",
            "rgb",
            "normal",
            "confidence",
            "observation_count",
        )
    }


def _normal_availability_record(
    data: MapTensorData, scene_id: str
) -> dict[str, Any]:
    """Fail closed unless this sealed artifact's normal channel is a true no-op."""

    expected_count = _SEALED_NORMAL_VOXEL_COUNTS.get(scene_id)
    normal = data.normal.detach().cpu().contiguous()
    nonzero_values = int(torch.count_nonzero(normal).item())
    nonzero_rows = int(torch.count_nonzero(torch.linalg.vector_norm(normal, dim=1)).item())
    if (
        expected_count is None
        or data.voxel_count != expected_count
        or tuple(normal.shape) != (expected_count, 3)
        or normal.dtype != torch.float32
        or not bool(torch.isfinite(normal).all())
        or nonzero_values != 0
        or nonzero_rows != 0
    ):
        raise RuntimeError(
            "V94 normal-channel availability changed; refusing to claim the "
            "sealed remove-normals no-op contract"
        )
    return {
        "scene_id": scene_id,
        "status": "unsupported_noop_all_zero_in_sealed_evaluation_map",
        "shape": [expected_count, 3],
        "dtype": "torch.float32",
        "nonzero_value_count": 0,
        "nonzero_row_count": 0,
        "rms": 0.0,
        "tensor_sha256": _tensor_sha256(normal),
        "remove_normals_memory_would_be_identical": True,
    }


def _expected_normal_availability() -> dict[str, dict[str, Any]]:
    return {
        scene_id: {
            "scene_id": scene_id,
            "status": "unsupported_noop_all_zero_in_sealed_evaluation_map",
            "shape": [voxel_count, 3],
            "dtype": "torch.float32",
            "nonzero_value_count": 0,
            "nonzero_row_count": 0,
            "rms": 0.0,
            "tensor_sha256": _tensor_sha256(
                torch.zeros((voxel_count, 3), dtype=torch.float32)
            ),
            "remove_normals_memory_would_be_identical": True,
        }
        for scene_id, voxel_count in _SEALED_NORMAL_VOXEL_COUNTS.items()
    }


def channel_identifiability_contract() -> dict[str, Any]:
    """Declare only channels that the current tokenizer consumes separately."""

    return {
        "semantic_features": {
            "independently_identifiable_before_scene_tokenization": True,
            "included_control": SEMANTIC_PAYLOAD_SHUFFLE,
        },
        "xyz_position": {
            "independently_identifiable_before_scene_tokenization": True,
            "included_control": POSITION_SPATIAL_SHUFFLE,
        },
        "rgb": {
            "independently_identifiable_before_scene_tokenization": True,
            "included_control": REMOVE_RGB,
        },
        "normal": {
            "independently_identifiable_before_scene_tokenization": True,
            "available_in_sealed_evaluation_artifact": False,
            "included_control": None,
            "excluded_control": REMOVE_NORMALS,
            "observed_nonzero_value_count_across_six_scenes": 0,
            "reason": (
                "all six sealed coarsened normal tensors are exactly zero; "
                "remove_normals would produce identical input and is not evidence"
            ),
        },
        "viewpoint": {
            "independently_identifiable_before_scene_tokenization": False,
            "included_control": None,
            "reason": (
                "view_direction is present in the fused NPZ but load_map_tensors "
                "does not expose it and PointTokenProjection does not consume it; "
                "zeroing it would produce an identical direct memory"
            ),
        },
    }


def apply_strong_map_control(
    data: MapTensorData,
    condition: str,
    *,
    seed: int,
    scene_id: str,
) -> tuple[MapTensorData, dict[str, Any]]:
    """Apply one targeted map intervention and prove unchanged channels."""

    if condition not in COMPILED_CONDITIONS:
        raise ValueError(f"V94 causal condition is not map-compiled: {condition}")
    before = _map_hashes(data)
    controlled, generic = apply_map_control(
        data,
        _MAP_CONTROL[condition],
        seed=seed,
        scene_id=scene_id,
    )
    after = _map_hashes(controlled)
    changed = sorted(name for name in before if before[name] != after[name])
    expected = {
        SEMANTIC_PAYLOAD_SHUFFLE: ["semantic"],
        POSITION_SPATIAL_SHUFFLE: ["xyz"],
        REMOVE_RGB: ["rgb"],
    }[condition]
    if changed != expected:
        raise RuntimeError(
            f"V94 causal {condition} changed unexpected channels: {changed}"
        )
    if condition == REMOVE_RGB and torch.count_nonzero(controlled.rgb).item() != 0:
        raise RuntimeError("V94 remove-RGB control did not zero RGB")
    receipt = {
        "condition": condition,
        "source_scope": "complete_coarsened_numeric_voxel_map",
        "source_voxel_count": int(data.source_voxel_count),
        "processed_voxel_count": int(data.voxel_count),
        "semantic_dimension": int(data.feature_dim),
        "changed_channels": changed,
        "unchanged_channel_sha256": {
            name: before[name] for name in before if name not in changed
        },
        "before_channel_sha256": before,
        "after_channel_sha256": after,
        "base_seed": int(seed),
        "derived_seed": generic.get("derived_seed"),
        "permutation_sha256": generic.get("permutation_sha256"),
        "all_voxels_processed": True,
        "question_inputs_used": False,
        "question_dependent_selection": False,
        "semantic_multiset_preserved": condition == SEMANTIC_PAYLOAD_SHUFFLE,
        "semantic_rows_retained_exactly": condition == POSITION_SPATIAL_SHUFFLE,
        "geometry_channels_retained_exactly": condition
        == SEMANTIC_PAYLOAD_SHUFFLE,
    }
    if condition in {SEMANTIC_PAYLOAD_SHUFFLE, POSITION_SPATIAL_SHUFFLE}:
        _require_sha256(receipt["permutation_sha256"], f"{condition} permutation")
    return controlled, receipt


def zero_full_scene_memory(memory: torch.Tensor) -> torch.Tensor:
    """Zero every environmental slot while preserving native BOI and EOI."""

    if (
        tuple(memory.shape) != MEMORY_SHAPE
        or not memory.is_floating_point()
        or not bool(torch.isfinite(memory).all())
    ):
        raise ValueError("V94 causal zero control requires finite [1,738,1536]")
    result = memory.detach().clone()
    result[:, 1:-1].zero_()
    if (
        not torch.equal(result[:, :1], memory[:, :1])
        or not torch.equal(result[:, -1:], memory[:, -1:])
        or torch.count_nonzero(result[:, 1:-1]).item() != 0
    ):
        raise RuntimeError("V94 causal zero control changed native boundaries")
    return result


def _derived_control_seed(scene_id: str, condition: str) -> int:
    payload = f"{SEED}:{scene_id}:{condition}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little")


def permute_full_scene_memory(
    memory: torch.Tensor, *, scene_id: str
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Permute all 736 environmental tokens while preserving native boundaries."""

    if (
        scene_id not in SCENE_IDS
        or tuple(memory.shape) != MEMORY_SHAPE
        or not memory.is_floating_point()
        or not bool(torch.isfinite(memory).all())
    ):
        raise ValueError("V94 causal permutation requires a sealed finite scene memory")
    derived_seed = _derived_control_seed(scene_id, FULL_INTERIOR_TOKEN_PERMUTATION)
    permutation = deterministic_permutation(MEMORY_SHAPE[1] - 2, derived_seed)
    indices = torch.from_numpy(permutation).to(memory.device)
    result = torch.cat(
        (
            memory[:, :1],
            memory[:, 1:-1].index_select(1, indices),
            memory[:, -1:],
        ),
        dim=1,
    ).contiguous()
    if (
        tuple(result.shape) != MEMORY_SHAPE
        or not torch.equal(result[:, :1], memory[:, :1])
        or not torch.equal(result[:, -1:], memory[:, -1:])
        or torch.equal(result[:, 1:-1], memory[:, 1:-1])
        or not torch.equal(
            result[:, 1:-1], memory[:, 1:-1].index_select(1, indices)
        )
    ):
        raise RuntimeError("V94 full-interior permutation contract failed")
    receipt = {
        "condition": FULL_INTERIOR_TOKEN_PERMUTATION,
        "scene_id": scene_id,
        "scope": "all_736_continuous_environment_tokens",
        "base_seed": SEED,
        "derived_seed": derived_seed,
        "algorithm": "numpy.PCG64.permutation",
        "permutation_sha256": hashlib.sha256(
            permutation.astype("<i8", copy=False).tobytes()
        ).hexdigest(),
        "interior_token_count": MEMORY_SHAPE[1] - 2,
        "native_boi_retained_exactly": True,
        "native_eoi_retained_exactly": True,
        "interior_token_rows_retained_exactly": True,
        "question_inputs_used": False,
        "question_dependent_selection": False,
    }
    return result, receipt


def select_profile_questions(
    manifest: QuestionManifest, profile: EvaluationProfile
) -> QuestionManifest:
    """Choose a label-blind, scene-balanced deterministic question subset."""

    if manifest.question_count != QUESTION_COUNT or manifest.scene_count != 6:
        raise ValueError("V94 causal profile selector requires the sealed 216 questions")
    if profile.name == "full":
        return manifest
    grouped = manifest.by_scene()
    selected_keys: set[tuple[str, str]] = set()
    for scene_id in SCENE_IDS:
        rows = grouped.get(scene_id, [])
        if len(rows) != 36:
            raise ValueError(f"V94 causal question count changed for {scene_id}")
        ranked = sorted(
            rows,
            key=lambda row: hashlib.sha256(
                f"{SEED}:{profile.name}:{scene_id}:{row.question_id}".encode()
            ).hexdigest(),
        )
        selected_keys.update(
            (row.scene_id, row.question_id)
            for row in ranked[: profile.questions_per_scene]
        )
    selected = tuple(
        row
        for row in manifest.questions
        if (row.scene_id, row.question_id) in selected_keys
    )
    if len(selected) != profile.question_count:
        raise RuntimeError("V94 causal representative selection coverage changed")
    return QuestionManifest(
        questions=selected,
        questions_sha256=questions_sha256(selected),
        source_qa_sha256=manifest.source_qa_sha256,
        manifest_path=manifest.manifest_path,
        manifest_sha256=manifest.manifest_sha256,
    )


def _source_chain_without_questions() -> tuple[dict[str, Any], dict[str, Any]]:
    """Authenticate training/candidate/cache without opening questions or labels."""

    root = PROJECT_ROOT.resolve()
    config, config_path = v94_evidence._load_sealed_config(root, CONFIG)
    preseal = v94_evidence._precompile_seal_identity(root, config, config_path)
    outputs = config["outputs"]
    prereg_path = (root / str(outputs["preregistration"])).resolve()
    cpu_path = (root / str(outputs["cpu_preflight"])).resolve()
    prereg = _read_json(prereg_path)
    seal = {
        "config_sha256": preseal["config_sha256"],
        "preregistration_sha256": file_sha256(prereg_path),
        "cpu_preflight_sha256": file_sha256(cpu_path),
        "authenticated_sources": prereg["authenticated_sources"],
    }
    candidate = v94_evidence._authenticate_training_and_candidate(root, config, seal)
    correct = v94_evidence._authenticate_memory_cache(root, config)
    compilation = v94_evidence._authenticate_compilation_attestation(
        root,
        config,
        config_sha256=seal["config_sha256"],
        cache=correct,
    )
    maps = v94_evidence._current_map_inventory(root, config)
    question_path = (root / str(config["sources"]["sanitized_evaluation_questions"])).resolve()
    label_path = (
        root / str(config["sources"]["evaluation_qa_reserved_for_label_scorer"])
    ).resolve()
    bindings = {
        "artifact": "v94_strong_causal_source_chain_v1",
        **seal,
        "config_path": config_path.relative_to(root).as_posix(),
        "training_report_sha256": candidate["training_report_sha256"],
        "candidate_weights_sha256": candidate["candidate_weights_sha256"],
        "candidate_metadata_sha256": candidate["candidate_metadata_sha256"],
        "candidate_state_sha256": candidate["candidate_state_sha256"],
        "candidate_weights_path": candidate["candidate_weights_path"]
        .relative_to(root)
        .as_posix(),
        "candidate_metadata_path": candidate["candidate_metadata_path"]
        .relative_to(root)
        .as_posix(),
        "correct_cache_manifest_sha256": correct["manifest_file_sha256"],
        "correct_cache_manifest_path": correct["manifest_path"]
        .relative_to(root)
        .as_posix(),
        "correct_memory_sha256": correct["memory_hashes"],
        "correct_memory_file_sha256": {
            scene: file_sha256(correct["memory_paths"][scene]) for scene in SCENE_IDS
        },
        "cache_precompile_attestation_sha256": compilation[
            "pre_attestation_sha256"
        ],
        "cache_postcompile_attestation_sha256": compilation[
            "post_attestation_sha256"
        ],
        "source_maps": maps,
        "sanitized_question_manifest_path": question_path.relative_to(root).as_posix(),
        "sanitized_question_manifest_declared_sha256": config["sources"][
            "sanitized_evaluation_questions_sha256"
        ],
        "reference_label_path": label_path.relative_to(root).as_posix(),
        "reference_label_declared_sha256": config["sources"]["evaluation_qa_sha256"],
        "questions_opened": False,
        "labels_opened": False,
        "oracle_opened": False,
    }
    bindings["source_chain_sha256"] = _canonical_sha256(bindings)
    return config, bindings


def _compile_forbidden_roots(config: Mapping[str, Any]) -> list[Path]:
    roots = [PROJECT_ROOT / "data/oracle"]
    roots.extend(PROJECT_ROOT.glob("data*/oracle"))
    roots.extend(PROJECT_ROOT.glob("data*/qa"))
    roots.extend(
        (
            PROJECT_ROOT / str(config["sources"]["sanitized_evaluation_questions"]),
            PROJECT_ROOT
            / str(config["sources"]["evaluation_qa_reserved_for_label_scorer"]),
        )
    )
    return list(dict.fromkeys(path.resolve() for path in roots))


def _control_runtime(
    factory: Any, scene_id: str, map_data: MapTensorData
) -> StaticChatRuntime:
    bootstrap = factory.bootstrap
    return StaticChatRuntime(
        config=factory.config,
        scene_id=scene_id,
        checkpoint_path=bootstrap.checkpoint_path,
        checkpoint_metadata=bootstrap.checkpoint_metadata,
        language=bootstrap.language,
        map_data=map_data,
        scene_model=bootstrap.scene_model,
        dense_aligner=bootstrap.dense_aligner,
        dense_sidecar_adapter=bootstrap.dense_sidecar_adapter,
        block_cross_residual=bootstrap.block_cross_residual,
        global_scene_residual=bootstrap.global_scene_residual,
        signed_x_scene_residual=bootstrap.signed_x_scene_residual,
        composer=bootstrap.composer,
        grounding=bootstrap.grounding,
        warnings=bootstrap.warnings,
        generation_function=bootstrap._generation_function,
    )


def _save_control_memory(
    path: Path, memory: torch.Tensor, *, condition: str, scene_id: str
) -> dict[str, Any]:
    if tuple(memory.shape) != MEMORY_SHAPE or memory.dtype != torch.bfloat16:
        raise ValueError("V94 causal cache requires BF16 [1,738,1536] memories")
    save_file(
        {"scene_memory": memory.detach().cpu().contiguous()},
        str(path),
        metadata={
            **_CACHE_TENSOR_METADATA,
            "condition": condition,
            "scene_id": scene_id,
        },
    )
    return {
        "filename": path.name,
        "file_sha256": file_sha256(path),
        "file_size_bytes": path.stat().st_size,
        "memory_sha256": prefix_sha256(memory),
    }


def compile_control_memory_cache() -> dict[str, Any]:
    """Compile all targeted controls before opening any question artifact."""

    if any(
        path.exists() or path.is_symlink()
        for path in (CONTROL_CACHE, COMPILE_ACCESS, COMPILE_RECEIPT)
    ):
        raise FileExistsError("V94 causal compile outputs are create-once")
    preconfig, _config_path = v94_evidence._load_sealed_config(
        PROJECT_ROOT.resolve(), CONFIG
    )
    audit = FileAccessAudit(
        forbidden_roots=_compile_forbidden_roots(preconfig),
        forbidden_component_names={"oracle"},
        block_forbidden=True,
    )
    CONTROL_CACHE.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{CONTROL_CACHE.name}.", dir=CONTROL_CACHE.parent)
    )
    started = time.monotonic()
    entries: dict[str, dict[str, Any]] = {
        condition: {} for condition in COMPILED_CONDITIONS
    }
    normal_availability: dict[str, dict[str, Any]] = {}
    try:
        with audit:
            config, bindings = _source_chain_without_questions()
            from semantic_3d_chat.chat.question_control_runtime import _load_control_head
            from semantic_3d_chat.evaluation.evaluate_v94_strict_multiscene_full40 import (
                EvaluationRuntimePrefixFactoryV94,
            )

            runtime_config_path = PROJECT_ROOT / str(config["sources"]["runtime_config"])
            runtime_config = load_runtime_config(runtime_config_path)
            parent_checkpoint = PROJECT_ROOT / str(
                config["sources"]["frozen_v85_checkpoint"]
            )
            factory = EvaluationRuntimePrefixFactoryV94(
                runtime_config, parent_checkpoint, audit=audit
            )
            controller_path = PROJECT_ROOT / str(
                config["sources"]["evaluation_memory_controller"]
            )
            controller_identity = authenticate_v75_control_checkpoint(controller_path)
            if (
                controller_identity.weights_sha256
                != config["sources"]["evaluation_memory_controller_weights_sha256"]
                or controller_identity.runtime_metadata_sha256
                != config["sources"]["evaluation_memory_controller_metadata_sha256"]
            ):
                raise ValueError("V94 causal compiler controller identity changed")
            controller, _controller_metadata = _load_control_head(
                controller_path,
                hidden_size=1536,
                device=torch.device("cpu"),
                audit=audit,
            )
            probe_path = PROJECT_ROOT / str(config["sources"]["evaluation_probe_bank"])
            probes, _probe_metadata = _load_probe_bank(probe_path, audit)
            for scene_id in SCENE_IDS:
                map_path = project_path(
                    runtime_config, "maps", scene_id, "voxel_map.npz"
                ).resolve()
                audit.record(map_path)
                data = load_map_tensors(
                    map_path,
                    runtime_config["scene"]["room_size_m"],
                    device="cpu",
                    input_voxel_size_m=runtime_config["scene_encoder"].get(
                        "input_voxel_size_m"
                    ),
                )
                if data.feature_dim != 3072:
                    raise ValueError("V94 causal map semantic dimension changed")
                normal_availability[scene_id] = _normal_availability_record(
                    data, scene_id
                )
                for condition in COMPILED_CONDITIONS:
                    controlled, transform = apply_strong_map_control(
                        data,
                        condition,
                        seed=SEED,
                        scene_id=scene_id,
                    )
                    runtime = _control_runtime(
                        factory,
                        scene_id,
                        controlled.to(factory.bootstrap.language.device),
                    )
                    runtime.assert_prefix_unchanged()
                    compiled = compile_fixed_scene_atlas_v75_v2(
                        runtime.scene_prefix.detach().cpu(), controller, probes
                    )
                    memory = compiled.scene_prefix.to(torch.bfloat16).contiguous()
                    filename = f"{scene_id}__{condition}.safetensors"
                    tensor = _save_control_memory(
                        temporary / filename,
                        memory,
                        condition=condition,
                        scene_id=scene_id,
                    )
                    entries[condition][scene_id] = {
                        **tensor,
                        "transform": transform,
                    }
                    del controlled, runtime, compiled, memory
                    if torch.backends.mps.is_available():
                        torch.mps.empty_cache()
            manifest = {
                "artifact": CACHE_ARTIFACT,
                "schema_version": SCHEMA_VERSION,
                "status": "terminal_posthoc_diagnostic_non_promotable",
                "terminal_diagnostic_only": True,
                "seed": SEED,
                "scene_ids": list(SCENE_IDS),
                "scene_count": 6,
                "compiled_conditions": list(COMPILED_CONDITIONS),
                "shape_each": list(MEMORY_SHAPE),
                "dtype": "bfloat16",
                "compiled_before_questions": True,
                "question_inputs_used": False,
                "question_dependent_retrieval": False,
                "all_memory_slots_retained": True,
                "environmental_text_inputs": [],
                "source_chain_sha256": bindings["source_chain_sha256"],
                "candidate_weights_sha256": bindings["candidate_weights_sha256"],
                "candidate_state_sha256": bindings["candidate_state_sha256"],
                "correct_cache_manifest_sha256": bindings[
                    "correct_cache_manifest_sha256"
                ],
                "correct_memory_sha256": bindings["correct_memory_sha256"],
                "source_maps": bindings["source_maps"],
                "channel_identifiability": channel_identifiability_contract(),
                "unsupported_channel_availability": {
                    "normal": normal_availability,
                },
                "controls": entries,
                "cannot_alter_v94_gates": True,
                "runtime_promotion_authorized": False,
            }
            _write_json_create_once(temporary / "manifest.json", manifest)
            os.rename(temporary, CONTROL_CACHE)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    audit.assert_clean()
    if COMPILE_ACCESS.exists() or COMPILE_ACCESS.is_symlink():
        raise FileExistsError(COMPILE_ACCESS)
    audit.save(COMPILE_ACCESS)
    loaded = set(audit.unique_paths)
    question_path = str(
        (PROJECT_ROOT / bindings["sanitized_question_manifest_path"]).resolve()
    )
    label_path = str((PROJECT_ROOT / bindings["reference_label_path"]).resolve())
    map_paths = {str((PROJECT_ROOT / row["path"]).resolve()) for row in bindings["source_maps"].values()}
    mandatory_source_paths = {
        str((PROJECT_ROOT / bindings["config_path"]).resolve()),
        str((PROJECT_ROOT / bindings["candidate_weights_path"]).resolve()),
        str((PROJECT_ROOT / bindings["candidate_metadata_path"]).resolve()),
        str((PROJECT_ROOT / bindings["correct_cache_manifest_path"]).resolve()),
        *map_paths,
    }
    if (
        question_path in loaded
        or label_path in loaded
        or not mandatory_source_paths <= loaded
    ):
        raise RuntimeError("V94 causal compilation access boundary is incomplete")
    cache = authenticate_control_cache(bindings=bindings, require_compile_receipt=False)
    receipt = {
        "artifact": "v94_strong_causal_compile_receipt_v1",
        "schema_version": SCHEMA_VERSION,
        "terminal_diagnostic_only": True,
        "source_chain_sha256": bindings["source_chain_sha256"],
        "control_cache_manifest_sha256": cache["manifest_file_sha256"],
        "control_memory_sha256": cache["memory_hashes"],
        "unsupported_channel_availability_sha256": _canonical_sha256(
            cache["manifest"]["unsupported_channel_availability"]
        ),
        "compile_access_sha256": file_sha256(COMPILE_ACCESS),
        "compile_loaded_file_inventory_sha256": _canonical_sha256(audit.unique_paths),
        "questions_opened": False,
        "labels_opened": False,
        "oracle_opened": False,
        "protected_read_count": 0,
        "runtime_promotion_authorized": False,
        "elapsed_seconds": time.monotonic() - started,
    }
    _write_json_create_once(COMPILE_RECEIPT, receipt)
    return {**receipt, "cache_path": str(CONTROL_CACHE)}


def authenticate_control_cache(
    *,
    bindings: Mapping[str, Any] | None = None,
    audit: FileAccessAudit | None = None,
    require_compile_receipt: bool = True,
) -> dict[str, Any]:
    if bindings is None:
        _config, current = _source_chain_without_questions()
        bindings = current
    manifest_path = CONTROL_CACHE / "manifest.json"
    manifest = _read_json(manifest_path)
    expected_files = {
        "manifest.json",
        *(
            f"{scene}__{condition}.safetensors"
            for condition in COMPILED_CONDITIONS
            for scene in SCENE_IDS
        ),
    }
    if (
        CONTROL_CACHE.is_symlink()
        or {path.name for path in CONTROL_CACHE.iterdir()} != expected_files
        or manifest.get("artifact") != CACHE_ARTIFACT
        or manifest.get("schema_version") != SCHEMA_VERSION
        or manifest.get("status") != "terminal_posthoc_diagnostic_non_promotable"
        or manifest.get("terminal_diagnostic_only") is not True
        or manifest.get("seed") != SEED
        or manifest.get("scene_ids") != list(SCENE_IDS)
        or manifest.get("compiled_conditions") != list(COMPILED_CONDITIONS)
        or manifest.get("shape_each") != list(MEMORY_SHAPE)
        or manifest.get("compiled_before_questions") is not True
        or manifest.get("question_inputs_used") is not False
        or manifest.get("question_dependent_retrieval") is not False
        or manifest.get("source_chain_sha256") != bindings["source_chain_sha256"]
        or manifest.get("candidate_weights_sha256")
        != bindings["candidate_weights_sha256"]
        or manifest.get("candidate_state_sha256") != bindings["candidate_state_sha256"]
        or manifest.get("correct_cache_manifest_sha256")
        != bindings["correct_cache_manifest_sha256"]
        or manifest.get("correct_memory_sha256") != bindings["correct_memory_sha256"]
        or manifest.get("source_maps") != bindings["source_maps"]
        or manifest.get("channel_identifiability")
        != channel_identifiability_contract()
        or manifest.get("unsupported_channel_availability")
        != {"normal": _expected_normal_availability()}
        or manifest.get("cannot_alter_v94_gates") is not True
        or manifest.get("runtime_promotion_authorized") is not False
    ):
        raise ValueError("V94 causal control-cache provenance changed")
    controls = manifest.get("controls")
    if not isinstance(controls, Mapping) or tuple(controls) != COMPILED_CONDITIONS:
        raise ValueError("V94 causal cache control order changed")
    memories: dict[str, dict[str, torch.Tensor]] = {
        condition: {} for condition in COMPILED_CONDITIONS
    }
    hashes: dict[str, dict[str, str]] = {
        condition: {} for condition in COMPILED_CONDITIONS
    }
    paths: dict[str, dict[str, Path]] = {
        condition: {} for condition in COMPILED_CONDITIONS
    }
    for condition in COMPILED_CONDITIONS:
        rows = controls[condition]
        if not isinstance(rows, Mapping) or tuple(sorted(rows)) != SCENE_IDS:
            raise ValueError(f"V94 causal cache scene inventory changed: {condition}")
        for scene_id in SCENE_IDS:
            row = rows[scene_id]
            if not isinstance(row, Mapping):
                raise TypeError("V94 causal cache entry is malformed")
            path = CONTROL_CACHE / f"{scene_id}__{condition}.safetensors"
            if audit is not None:
                audit.record(path)
            if (
                row.get("filename") != path.name
                or path.is_symlink()
                or not path.is_file()
                or path.stat().st_size != row.get("file_size_bytes")
                or file_sha256(path) != row.get("file_sha256")
            ):
                raise ValueError(f"V94 causal cache bytes changed: {scene_id}/{condition}")
            with safe_open(str(path), framework="pt", device="cpu") as handle:
                expected_metadata = {
                    **_CACHE_TENSOR_METADATA,
                    "condition": condition,
                    "scene_id": scene_id,
                }
                if set(handle.keys()) != {"scene_memory"} or handle.metadata() != expected_metadata:
                    raise ValueError("V94 causal cache tensor metadata changed")
            memory = load_file(str(path), device="cpu")["scene_memory"].contiguous()
            if (
                tuple(memory.shape) != MEMORY_SHAPE
                or memory.dtype != torch.bfloat16
                or not bool(torch.isfinite(memory).all())
                or prefix_sha256(memory) != row.get("memory_sha256")
            ):
                raise ValueError("V94 causal cache memory tensor changed")
            transform = row.get("transform")
            if (
                not isinstance(transform, Mapping)
                or transform.get("condition") != condition
                or transform.get("all_voxels_processed") is not True
                or transform.get("question_inputs_used") is not False
                or transform.get("question_dependent_selection") is not False
            ):
                raise ValueError("V94 causal cache transform receipt changed")
            memories[condition][scene_id] = memory
            hashes[condition][scene_id] = str(row["memory_sha256"])
            paths[condition][scene_id] = path
    result = {
        "manifest": manifest,
        "manifest_path": manifest_path,
        "manifest_file_sha256": file_sha256(manifest_path),
        "manifest_canonical_sha256": _canonical_sha256(manifest),
        "memories": memories,
        "memory_hashes": hashes,
        "memory_paths": paths,
    }
    if require_compile_receipt:
        receipt = _read_json(COMPILE_RECEIPT)
        access = _read_json(COMPILE_ACCESS)
        if (
            receipt.get("artifact") != "v94_strong_causal_compile_receipt_v1"
            or receipt.get("terminal_diagnostic_only") is not True
            or receipt.get("source_chain_sha256") != bindings["source_chain_sha256"]
            or receipt.get("control_cache_manifest_sha256")
            != result["manifest_file_sha256"]
            or receipt.get("control_memory_sha256") != hashes
            or receipt.get("unsupported_channel_availability_sha256")
            != _canonical_sha256(manifest["unsupported_channel_availability"])
            or receipt.get("compile_access_sha256") != file_sha256(COMPILE_ACCESS)
            or receipt.get("questions_opened") is not False
            or receipt.get("labels_opened") is not False
            or receipt.get("oracle_opened") is not False
            or receipt.get("protected_read_count") != 0
            or receipt.get("runtime_promotion_authorized") is not False
            or access.get("passed") is not True
            or access.get("forbidden_accesses") != []
        ):
            raise ValueError("V94 causal compilation receipt/access changed")
        result["compile_receipt_sha256"] = file_sha256(COMPILE_RECEIPT)
        result["compile_access_sha256"] = file_sha256(COMPILE_ACCESS)
    return result


def _load_correct_memories(
    source: Mapping[str, Any], *, audit: FileAccessAudit | None = None
) -> tuple[dict[str, torch.Tensor], dict[str, Path]]:
    cache_root = PROJECT_ROOT / "reports/gemma4/artifacts/v94_strict_multiscene_full40/evaluation_cache"
    memories: dict[str, torch.Tensor] = {}
    paths: dict[str, Path] = {}
    for scene_id in SCENE_IDS:
        path = cache_root / f"{scene_id}.safetensors"
        if audit is not None:
            audit.record(path)
        if file_sha256(path) != source["correct_memory_file_sha256"][scene_id]:
            raise ValueError(f"V94 causal correct memory bytes changed: {scene_id}")
        state = load_file(str(path), device="cpu")
        if set(state) != {"scene_memory"}:
            raise ValueError("V94 causal correct cache tensor inventory changed")
        memory = state["scene_memory"].contiguous()
        if prefix_sha256(memory) != source["correct_memory_sha256"][scene_id]:
            raise ValueError("V94 causal correct memory semantic hash changed")
        memories[scene_id] = memory
        paths[scene_id] = path
    return memories, paths


def _bind_profile_memories(
    profile: EvaluationProfile,
    source: Mapping[str, Any],
    *,
    audit: FileAccessAudit | None = None,
) -> tuple[
    dict[str, dict[str, torch.Tensor]],
    dict[str, dict[str, str]],
    dict[str, Any] | None,
    dict[str, Path],
    dict[str, dict[str, Any]],
]:
    """Load and construct every profile memory before questions are opened."""

    cache: dict[str, Any] | None = None
    if any(condition in COMPILED_CONDITIONS for condition in profile.conditions):
        cache = authenticate_control_cache(bindings=source, audit=audit)
    correct, correct_paths = _load_correct_memories(source, audit=audit)
    zero = {
        scene_id: zero_full_scene_memory(memory)
        for scene_id, memory in correct.items()
    }
    permuted: dict[str, torch.Tensor] = {}
    permutation_receipts: dict[str, dict[str, Any]] = {}
    for scene_id, memory in correct.items():
        permuted[scene_id], permutation_receipts[scene_id] = (
            permute_full_scene_memory(memory, scene_id=scene_id)
        )
    bound: dict[str, dict[str, torch.Tensor]] = {
        PRIMARY: correct,
        ZERO_FULL_SCENE: zero,
        WRONG_SCENE_SWAP: {
            scene_id: correct[PAIR_SCENE[scene_id]] for scene_id in SCENE_IDS
        },
        FULL_INTERIOR_TOKEN_PERMUTATION: permuted,
    }
    if cache is not None:
        bound.update(
            {
                condition: cache["memories"][condition]
                for condition in COMPILED_CONDITIONS
                if condition in profile.conditions
            }
        )
    if tuple(bound) != profile.conditions:
        raise RuntimeError("V94 causal profile memory inventory changed")
    hashes = {
        condition: {
            scene_id: prefix_sha256(bound[condition][scene_id])
            for scene_id in SCENE_IDS
        }
        for condition in profile.conditions
    }
    return bound, hashes, cache, correct_paths, permutation_receipts


def _questions(config: Mapping[str, Any]) -> QuestionManifest:
    path = PROJECT_ROOT / str(config["sources"]["sanitized_evaluation_questions"])
    manifest = load_question_manifest(path)
    if (
        manifest.manifest_sha256
        != config["sources"]["sanitized_evaluation_questions_sha256"]
        or manifest.source_qa_sha256 != config["sources"]["evaluation_qa_sha256"]
        or manifest.question_count != QUESTION_COUNT
        or manifest.scene_count != 6
        or {row.scene_id for row in manifest.questions} != set(SCENE_IDS)
    ):
        raise ValueError("V94 causal sanitized question manifest changed")
    return manifest


def _profile_questions(
    config: Mapping[str, Any], profile: EvaluationProfile
) -> QuestionManifest:
    return select_profile_questions(_questions(config), profile)


def _prediction_forbidden_roots(config: Mapping[str, Any]) -> list[Path]:
    roots = [PROJECT_ROOT / "data/oracle"]
    roots.extend(PROJECT_ROOT.glob("data*/oracle"))
    roots.extend(PROJECT_ROOT.glob("data*/qa"))
    roots.extend(
        (
            PROJECT_ROOT / str(config["sources"]["evaluation_qa_reserved_for_label_scorer"]),
            PROJECT_ROOT / "reports/gemma4/scorer_only",
        )
    )
    return list(dict.fromkeys(path.resolve() for path in roots))


def _prediction_provenance(
    source: Mapping[str, Any],
    cache: Mapping[str, Any] | None,
    questions: QuestionManifest,
    profile: EvaluationProfile,
    memory_hashes: Mapping[str, Mapping[str, str]],
    permutation_receipts: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    value = {
        "artifact": PREDICTION_ARTIFACT,
        "schema_version": SCHEMA_VERSION,
        "terminal_diagnostic_only": True,
        "evaluation_profile": profile.name,
        "question_selection": {
            "algorithm": (
                "complete_sealed_manifest"
                if profile.name == "full"
                else "six_per_scene_lowest_sha256_of_seed_profile_scene_question_id"
            ),
            "base_seed": SEED,
            "questions_per_scene": profile.questions_per_scene,
            "selected_questions_sha256": questions.questions_sha256,
        },
        "source_chain_sha256": source["source_chain_sha256"],
        "candidate_weights_sha256": source["candidate_weights_sha256"],
        "candidate_metadata_sha256": source["candidate_metadata_sha256"],
        "candidate_state_sha256": source["candidate_state_sha256"],
        "correct_cache_manifest_sha256": source["correct_cache_manifest_sha256"],
        "control_cache_manifest_sha256": (
            None if cache is None else cache["manifest_file_sha256"]
        ),
        "compile_receipt_sha256": (
            None if cache is None else cache["compile_receipt_sha256"]
        ),
        "compile_access_sha256": (
            None if cache is None else cache["compile_access_sha256"]
        ),
        "question_manifest_sha256": questions.manifest_sha256,
        "questions_sha256": questions.questions_sha256,
        "bound_memory_sha256": {
            condition: dict(memory_hashes[condition])
            for condition in profile.conditions
        },
        "full_interior_permutation_receipts": dict(permutation_receipts),
        "scene_ids": list(SCENE_IDS),
        "row_count": profile.question_count,
        "conditions": list(profile.conditions),
        "all_memories_bound_before_questions": True,
        "labels_opened": False,
        "question_dependent_retrieval": False,
        "environmental_text_inputs": [],
        "access_completion_required": True,
        "runtime_promotion_authorized": False,
    }
    value["provenance_sha256"] = _canonical_sha256(value)
    return value


def _validate_resume_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    questions: QuestionManifest,
    profile: EvaluationProfile,
    provenance_sha256: str,
) -> None:
    expected = {(row.scene_id, row.question_id) for row in questions.questions}
    seen: set[tuple[str, str]] = set()
    for row in rows:
        key = (str(row.get("scene_id")), str(row.get("question_id")))
        if (
            set(row) != _prediction_fields(profile)
            or key not in expected
            or key in seen
            or row.get("artifact") != PREDICTION_ARTIFACT
            or row.get("evaluation_profile") != profile.name
            or row.get("wrong_scene_id") != PAIR_SCENE.get(key[0])
            or row.get("provenance_sha256") != provenance_sha256
            or row.get("all_memory_hashes_unchanged") is not True
            or not all(
                isinstance(row.get(_PREDICTION_FIELD[condition]), str)
                for condition in profile.conditions
            )
            or not all(
                _SHA256.fullmatch(str(row.get(_HASH_FIELD[condition])))
                for condition in profile.conditions
            )
        ):
            raise ValueError(f"V94 causal prediction row changed: {key}")
        seen.add(key)


def predict_question_only(profile_name: str = "full") -> dict[str, Any]:
    """Generate a profile after binding every numeric memory and before labels."""

    profile = evaluation_profile(profile_name)
    paths = evaluation_paths(profile)
    if paths.completion.is_file():
        authenticated = authenticate_prediction_bundle(profile_name)
        return {
            "artifact": PREDICTION_ARTIFACT,
            "evaluation_profile": profile.name,
            "row_count": authenticated["row_count"],
            "prediction_sha256": authenticated["prediction_sha256"],
            "completed": True,
            "resumed": True,
            "terminal_diagnostic_only": True,
            "runtime_promotion_authorized": False,
        }
    if paths.access.exists() or paths.completion.exists():
        raise FileExistsError("V94 causal prediction has incomplete terminal evidence")
    started = time.monotonic()
    preconfig, _config_path = v94_evidence._load_sealed_config(
        PROJECT_ROOT.resolve(), CONFIG
    )
    audit = FileAccessAudit(
        forbidden_roots=_prediction_forbidden_roots(preconfig),
        forbidden_component_names={"oracle"},
        block_forbidden=True,
    )
    with audit:
        config, source = _source_chain_without_questions()
        # All original, wrong-scene, zero, permuted, and optional compiled
        # memories are fully bound before the first sanitized question read.
        bound, memory_hashes_before, cache, correct_paths, permutation_receipts = (
            _bind_profile_memories(profile, source, audit=audit)
        )
        questions = _profile_questions(config, profile)
        from semantic_3d_chat.evaluation.evaluate_v94_strict_multiscene_full40 import (
            _generate_arm,
            _load_predictor_stack_v94,
        )

        stack = _load_predictor_stack_v94(config)
        provenance = _prediction_provenance(
            source,
            cache,
            questions,
            profile,
            memory_hashes_before,
            permutation_receipts,
        )
        if paths.provenance.exists():
            if _read_json(paths.provenance) != provenance:
                raise ValueError("V94 causal prediction resume provenance changed")
        else:
            _write_json_create_once(paths.provenance, provenance)
        existing = read_jsonl(paths.predictions) if paths.predictions.is_file() else []
        _validate_resume_rows(
            existing,
            questions=questions,
            profile=profile,
            provenance_sha256=provenance["provenance_sha256"],
        )
        indexed = {
            (str(row["scene_id"]), str(row["question_id"])) for row in existing
        }
        rows = list(existing)
        for ordinal, question in enumerate(questions.questions, 1):
            key = (question.scene_id, question.question_id)
            if key in indexed:
                continue
            scene = question.scene_id
            predictions = {
                condition: _generate_arm(
                    stack, bound[condition][scene], question.question
                )
                for condition in profile.conditions
            }
            current_hashes = {
                condition: prefix_sha256(bound[condition][scene])
                for condition in profile.conditions
            }
            unchanged = all(
                current_hashes[condition] == memory_hashes_before[condition][scene]
                for condition in profile.conditions
            )
            row = {
                "artifact": PREDICTION_ARTIFACT,
                "evaluation_profile": profile.name,
                "scene_id": scene,
                "question_id": question.question_id,
                "wrong_scene_id": PAIR_SCENE[scene],
                **{
                    _PREDICTION_FIELD[condition]: predictions[condition]
                    for condition in profile.conditions
                },
                **{
                    _HASH_FIELD[condition]: current_hashes[condition]
                    for condition in profile.conditions
                },
                "all_memory_hashes_unchanged": unchanged,
                "elapsed_seconds": time.monotonic() - started,
                "provenance_sha256": provenance["provenance_sha256"],
            }
            rows.append(row)
            atomic_write_jsonl(paths.predictions, rows)
            if ordinal == 1 or ordinal % 6 == 0 or ordinal == profile.question_count:
                print(
                    json.dumps(
                        {
                            "event": "v94_strong_causal_prediction",
                            "evaluation_profile": profile.name,
                            "ordinal": ordinal,
                            "total": profile.question_count,
                            "scene_id": scene,
                            "question_id": question.question_id,
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()
        memory_hashes_after = {
            condition: {
                scene_id: prefix_sha256(bound[condition][scene_id])
                for scene_id in SCENE_IDS
            }
            for condition in profile.conditions
        }
    audit.assert_clean()
    if (
        len(rows) != profile.question_count
        or memory_hashes_after != memory_hashes_before
    ):
        raise RuntimeError("V94 causal prediction coverage or memory invariance failed")
    if paths.access.exists() or paths.access.is_symlink():
        raise FileExistsError(paths.access)
    audit.save(paths.access)
    access = _read_json(paths.access)
    loaded = set(access["loaded_files"])
    mandatory = {
        str(path.resolve()) for path in correct_paths.values()
    } | {
        str((PROJECT_ROOT / source["config_path"]).resolve()),
        str((PROJECT_ROOT / source["candidate_weights_path"]).resolve()),
        str((PROJECT_ROOT / source["candidate_metadata_path"]).resolve()),
        str((PROJECT_ROOT / source["correct_cache_manifest_path"]).resolve()),
        str((PROJECT_ROOT / source["sanitized_question_manifest_path"]).resolve()),
        *(
            str((PROJECT_ROOT / row["path"]).resolve())
            for row in source["source_maps"].values()
        ),
    }
    if cache is not None:
        mandatory.update(
            str(path.resolve())
            for condition in COMPILED_CONDITIONS
            if condition in profile.conditions
            for path in cache["memory_paths"][condition].values()
        )
        mandatory.update(
            {
                str(cache["manifest_path"].resolve()),
                str(COMPILE_RECEIPT.resolve()),
                str(COMPILE_ACCESS.resolve()),
            }
        )
    if (
        access.get("passed") is not True
        or access.get("forbidden_accesses") != []
        or not mandatory <= loaded
    ):
        raise RuntimeError("V94 causal prediction access evidence is incomplete")
    completion = {
        "artifact": "v94_strong_causal_prediction_completion_v1",
        "schema_version": SCHEMA_VERSION,
        "terminal_diagnostic_only": True,
        "evaluation_profile": profile.name,
        "source_chain_sha256": source["source_chain_sha256"],
        "candidate_weights_sha256": source["candidate_weights_sha256"],
        "candidate_state_sha256": source["candidate_state_sha256"],
        "correct_cache_manifest_sha256": source["correct_cache_manifest_sha256"],
        "control_cache_manifest_sha256": (
            None if cache is None else cache["manifest_file_sha256"]
        ),
        "question_manifest_sha256": questions.manifest_sha256,
        "questions_sha256": questions.questions_sha256,
        "prediction_provenance_sha256": provenance["provenance_sha256"],
        "prediction_provenance_file_sha256": file_sha256(paths.provenance),
        "prediction_sha256": file_sha256(paths.predictions),
        "prediction_access_sha256": file_sha256(paths.access),
        "loaded_file_inventory_sha256": _canonical_sha256(access["loaded_files"]),
        "row_count": len(rows),
        "scene_count": 6,
        "conditions": list(profile.conditions),
        "all_memories_bound_before_questions": True,
        "all_memory_hashes_invariant": True,
        "labels_opened": False,
        "oracle_opened": False,
        "protected_read_count": 0,
        "runtime_promotion_authorized": False,
        "elapsed_seconds": time.monotonic() - started,
    }
    _write_json_create_once(paths.completion, completion)
    return {**completion, "completed": True, "resumed": bool(existing)}


def authenticate_prediction_bundle(profile_name: str = "full") -> dict[str, Any]:
    """Authenticate predictions and access evidence without opening labels."""

    profile = evaluation_profile(profile_name)
    paths = evaluation_paths(profile)
    config, source = _source_chain_without_questions()
    bound, memory_hashes, cache, correct_paths, permutation_receipts = (
        _bind_profile_memories(profile, source)
    )
    questions = _profile_questions(config, profile)
    provenance = _read_json(paths.provenance)
    expected_provenance = _prediction_provenance(
        source,
        cache,
        questions,
        profile,
        memory_hashes,
        permutation_receipts,
    )
    if provenance != expected_provenance:
        raise ValueError("V94 causal prediction provenance changed")
    rows = read_jsonl(paths.predictions)
    _validate_resume_rows(
        rows,
        questions=questions,
        profile=profile,
        provenance_sha256=provenance["provenance_sha256"],
    )
    expected_keys = {(row.scene_id, row.question_id) for row in questions.questions}
    observed = {(str(row["scene_id"]), str(row["question_id"])) for row in rows}
    if len(rows) != profile.question_count or observed != expected_keys:
        raise ValueError("V94 causal predictions lack exact profile-row coverage")
    for row in rows:
        scene = str(row["scene_id"])
        if any(
            row[_HASH_FIELD[condition]] != memory_hashes[condition][scene]
            or prefix_sha256(bound[condition][scene]) != memory_hashes[condition][scene]
            for condition in profile.conditions
        ):
            raise ValueError("V94 causal prediction memory binding changed")
    access = _read_json(paths.access)
    completion = _read_json(paths.completion)
    mandatory = {
        str(path.resolve()) for path in correct_paths.values()
    } | {
        str((PROJECT_ROOT / source["config_path"]).resolve()),
        str((PROJECT_ROOT / source["candidate_weights_path"]).resolve()),
        str((PROJECT_ROOT / source["candidate_metadata_path"]).resolve()),
        str((PROJECT_ROOT / source["correct_cache_manifest_path"]).resolve()),
        str((PROJECT_ROOT / source["sanitized_question_manifest_path"]).resolve()),
        *(
            str((PROJECT_ROOT / row["path"]).resolve())
            for row in source["source_maps"].values()
        ),
    }
    if cache is not None:
        mandatory.update(
            str(path.resolve())
            for condition in COMPILED_CONDITIONS
            if condition in profile.conditions
            for path in cache["memory_paths"][condition].values()
        )
        mandatory.update(
            {
                str(cache["manifest_path"].resolve()),
                str(COMPILE_RECEIPT.resolve()),
                str(COMPILE_ACCESS.resolve()),
            }
        )
    if (
        access.get("passed") is not True
        or access.get("forbidden_accesses") != []
        or not mandatory <= set(access.get("loaded_files", []))
        or completion.get("artifact")
        != "v94_strong_causal_prediction_completion_v1"
        or completion.get("terminal_diagnostic_only") is not True
        or completion.get("evaluation_profile") != profile.name
        or completion.get("source_chain_sha256") != source["source_chain_sha256"]
        or completion.get("candidate_weights_sha256")
        != source["candidate_weights_sha256"]
        or completion.get("candidate_state_sha256") != source["candidate_state_sha256"]
        or completion.get("correct_cache_manifest_sha256")
        != source["correct_cache_manifest_sha256"]
        or completion.get("control_cache_manifest_sha256")
        != (None if cache is None else cache["manifest_file_sha256"])
        or completion.get("question_manifest_sha256") != questions.manifest_sha256
        or completion.get("questions_sha256") != questions.questions_sha256
        or completion.get("prediction_provenance_sha256")
        != provenance["provenance_sha256"]
        or completion.get("prediction_provenance_file_sha256")
        != file_sha256(paths.provenance)
        or completion.get("prediction_sha256") != file_sha256(paths.predictions)
        or completion.get("prediction_access_sha256") != file_sha256(paths.access)
        or completion.get("loaded_file_inventory_sha256")
        != _canonical_sha256(access["loaded_files"])
        or completion.get("row_count") != profile.question_count
        or completion.get("conditions") != list(profile.conditions)
        or completion.get("all_memories_bound_before_questions") is not True
        or completion.get("all_memory_hashes_invariant") is not True
        or completion.get("labels_opened") is not False
        or completion.get("oracle_opened") is not False
        or completion.get("protected_read_count") != 0
        or completion.get("runtime_promotion_authorized") is not False
    ):
        raise ValueError("V94 causal prediction completion/access binding changed")
    return {
        "config": config,
        "source": source,
        "cache": cache,
        "questions": questions,
        "provenance": provenance,
        "completion": completion,
        "rows": rows,
        "row_count": len(rows),
        "profile": profile,
        "paths": paths,
        "prediction_sha256": file_sha256(paths.predictions),
        "access_sha256": file_sha256(paths.access),
        "completion_sha256": file_sha256(paths.completion),
    }


def _accuracy(
    references: Sequence[Mapping[str, Any]],
    predictions: Mapping[tuple[str, str], Mapping[str, Any]],
    field: str,
) -> dict[str, Any]:
    values: list[bool] = []
    by_type: defaultdict[str, list[bool]] = defaultdict(list)
    for reference in references:
        key = (str(reference["scene_id"]), str(reference["question_id"]))
        correct = canonical_type_specific_match(
            str(reference["answer_type"]),
            predictions[key][field],
            reference["answer"],
        )
        values.append(correct)
        by_type[str(reference["answer_type"])].append(correct)
    return {
        "correct": sum(values),
        "total": len(values),
        "accuracy": sum(values) / len(values),
        "by_answer_type": {
            answer_type: {
                "correct": sum(items),
                "total": len(items),
                "accuracy": sum(items) / len(items),
            }
            for answer_type, items in sorted(by_type.items())
        },
    }


def score_records(
    references: Sequence[Mapping[str, Any]],
    prediction_rows: Sequence[Mapping[str, Any]],
    profile: EvaluationProfile | None = None,
) -> dict[str, Any]:
    profile = profile or evaluation_profile("full")
    if not references or len(references) != len(prediction_rows):
        raise ValueError("V94 causal score requires equal nonempty reference/prediction rows")
    predictions = {
        (str(row["scene_id"]), str(row["question_id"])): row
        for row in prediction_rows
    }
    reference_keys = {
        (str(row["scene_id"]), str(row["question_id"])) for row in references
    }
    if set(predictions) != reference_keys:
        raise ValueError("V94 causal scoring keys differ")
    arms = {
        condition: _accuracy(references, predictions, _PREDICTION_FIELD[condition])
        for condition in profile.conditions
    }
    primary = arms[PRIMARY]["accuracy"]
    comparisons: dict[str, Any] = {}
    for condition in profile.conditions[1:]:
        changes = sum(
            canonical_answer_key(
                str(reference["answer_type"]),
                predictions[(str(reference["scene_id"]), str(reference["question_id"]))][
                    _PREDICTION_FIELD[PRIMARY]
                ],
            )
            != canonical_answer_key(
                str(reference["answer_type"]),
                predictions[(str(reference["scene_id"]), str(reference["question_id"]))][
                    _PREDICTION_FIELD[condition]
                ],
            )
            for reference in references
        )
        comparisons[condition] = {
            "accuracy_drop_from_primary": primary - arms[condition]["accuracy"],
            "prediction_change_count": changes,
            "prediction_change_rate": changes / len(references),
        }
    return {
        "arms": arms,
        "comparisons": comparisons,
        "diagnostic_scope": "posthoc_non_preregistered_not_a_promotion_gate",
        "evaluation_profile": profile.name,
        "runtime_promotion_authorized": False,
    }


def _load_references(
    config: Mapping[str, Any], questions: QuestionManifest
) -> list[dict[str, Any]]:
    path = PROJECT_ROOT / str(config["sources"]["evaluation_qa_reserved_for_label_scorer"])
    if (
        path.is_symlink()
        or not path.is_file()
        or file_sha256(path) != EXPECTED_REFERENCE_SHA256
        or EXPECTED_REFERENCE_SHA256 != config["sources"]["evaluation_qa_sha256"]
    ):
        raise ValueError("V94 causal scorer reference bytes changed")
    rows = read_jsonl(path)
    full_questions = _questions(config)
    question_by_key = {
        (row.scene_id, row.question_id): row.question
        for row in full_questions.questions
    }
    seen: set[tuple[str, str]] = set()
    types: Counter[str] = Counter()
    for row in rows:
        key = (str(row.get("scene_id")), str(row.get("question_id")))
        if (
            key in seen
            or row.get("question") != question_by_key.get(key)
            or not isinstance(row.get("answer"), str)
            or not isinstance(row.get("answer_type"), str)
        ):
            raise ValueError(f"V94 causal reference projection changed: {key}")
        seen.add(key)
        types[str(row["answer_type"])] += 1
    if (
        len(rows) != QUESTION_COUNT
        or seen != set(question_by_key)
        or dict(sorted(types.items())) != EXPECTED_TYPE_COUNTS
    ):
        raise ValueError("V94 causal reference coverage/type inventory changed")
    selected_keys = {
        (row.scene_id, row.question_id) for row in questions.questions
    }
    selected = [
        row
        for row in rows
        if (str(row["scene_id"]), str(row["question_id"])) in selected_keys
    ]
    if len(selected) != questions.question_count:
        raise ValueError("V94 causal selected reference coverage changed")
    return selected


def score_label_isolated(profile_name: str = "full") -> dict[str, Any]:
    """Authenticate inference first, then score labels without loading Gemma."""

    profile = evaluation_profile(profile_name)
    paths = evaluation_paths(profile)
    if paths.score.exists() or paths.score.is_symlink():
        raise FileExistsError("V94 causal score is create-once")
    bundle = authenticate_prediction_bundle(profile_name)
    # Labels first become readable after the complete prediction/access bundle
    # above has been authenticated.
    references = _load_references(bundle["config"], bundle["questions"])
    metrics = score_records(references, bundle["rows"], profile)
    report = {
        "artifact": SCORE_ARTIFACT,
        "schema_version": SCHEMA_VERSION,
        "status": "terminal_measured_posthoc_diagnostic_non_promotable",
        "terminal_diagnostic_only": True,
        "evaluation_profile": profile.name,
        "source_chain_sha256": bundle["source"]["source_chain_sha256"],
        "candidate_weights_sha256": bundle["source"]["candidate_weights_sha256"],
        "candidate_state_sha256": bundle["source"]["candidate_state_sha256"],
        "correct_cache_manifest_sha256": bundle["source"][
            "correct_cache_manifest_sha256"
        ],
        "control_cache_manifest_sha256": (
            None
            if bundle["cache"] is None
            else bundle["cache"]["manifest_file_sha256"]
        ),
        "question_manifest_sha256": bundle["questions"].manifest_sha256,
        "prediction_sha256": bundle["prediction_sha256"],
        "prediction_access_sha256": bundle["access_sha256"],
        "prediction_completion_sha256": bundle["completion_sha256"],
        "reference_sha256": EXPECTED_REFERENCE_SHA256,
        "row_count": profile.question_count,
        "scene_count": 6,
        "conditions": list(profile.conditions),
        "prediction_bundle_authenticated_before_labels_opened": True,
        "labels_opened_only_by_separate_scorer": True,
        "scorer_loaded_model": False,
        "answers_questions_or_oracle_serialized": False,
        "channel_identifiability": channel_identifiability_contract(),
        "metrics": metrics,
        "limitations": [
            (
                "These are posthoc interventions, not preregistered V94 promotion "
                "gates; they cannot authorize runtime release."
            ),
            (
                "The full-interior permutation preserves all 736 token rows and "
                "native BOI/EOI but intentionally scrambles both atlas and base-latent "
                "ordering; it is a broad causal destruction control, not a channel-isolated one."
            ),
            (
                "Semantic and position controls isolate explicit voxel input channels, "
                "but the recompiled 738-token memories nonlinearly mix those channels."
            ),
            (
                "RGB removal is a valid pre-tokenizer channel control, but color "
                "information may also be redundantly encoded in visual semantics."
            ),
            (
                "Normal removal is excluded: every normal value in all six sealed "
                "evaluation maps is already zero, so the proposed arm is an input-identical no-op."
            ),
            (
                "View direction is not consumed by the current point tokenizer, so a "
                "viewpoint-removal arm is unsupported rather than reported as evidence."
            ),
            "No answer-token NLL is measured by this model-free scorer.",
            (
                "The representative-core profile is a label-blind 36-question "
                "screen of four direct-memory arms; targeted voxel-channel "
                "controls are measured only by the full profile."
            ),
        ],
        "cannot_alter_v94_gates": True,
        "cannot_authorize_packaging": True,
        "cannot_authorize_promotion": True,
        "automatic_runtime_promotion": False,
        "runtime_promotion_authorized": False,
    }
    _write_json_create_once(paths.score, report)
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=("compile-controls", "authenticate-cache", "predict", "authenticate", "score"),
    )
    parser.add_argument(
        "--profile",
        choices=tuple(PROFILES),
        default="full",
        help="A separately namespaced terminal diagnostic profile.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "compile-controls":
            result = compile_control_memory_cache()
        elif args.command == "authenticate-cache":
            result = authenticate_control_cache()
        elif args.command == "predict":
            result = predict_question_only(args.profile)
        elif args.command == "authenticate":
            result = authenticate_prediction_bundle(args.profile)
        else:
            result = score_label_isolated(args.profile)
    except (OSError, RuntimeError, TypeError, ValueError) as error:
        print(f"V94 strong causal {args.command} refused: {error}", file=sys.stderr)
        return 2
    printable = {
        key: value
        for key, value in result.items()
        if key
        not in {
            "cache",
            "config",
            "manifest",
            "memories",
            "memory_paths",
            "paths",
            "provenance",
            "questions",
            "rows",
            "source",
        }
    }
    print(json.dumps(printable, indent=2, sort_keys=True, allow_nan=False, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "COMPILED_CONDITIONS",
    "CONDITIONS",
    "CORE_CONDITIONS",
    "FULL_INTERIOR_TOKEN_PERMUTATION",
    "POSITION_SPATIAL_SHUFFLE",
    "PRIMARY",
    "REMOVE_RGB",
    "SEMANTIC_PAYLOAD_SHUFFLE",
    "WRONG_SCENE_SWAP",
    "ZERO_FULL_SCENE",
    "EvaluationProfile",
    "apply_strong_map_control",
    "authenticate_control_cache",
    "authenticate_prediction_bundle",
    "channel_identifiability_contract",
    "compile_control_memory_cache",
    "evaluation_paths",
    "evaluation_profile",
    "main",
    "permute_full_scene_memory",
    "predict_question_only",
    "score_label_isolated",
    "score_records",
    "select_profile_questions",
    "zero_full_scene_memory",
]
