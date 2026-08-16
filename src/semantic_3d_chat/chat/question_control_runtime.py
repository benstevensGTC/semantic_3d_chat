"""Static chat with a learned, vocabulary-free continuous question-control head."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Final

import torch
from safetensors.torch import load_file

from semantic_3d_chat.chat.file_audit import FileAccessAudit
from semantic_3d_chat.chat.grounding_sidecar_v78_runtime import (
    V78GroundingSidecarRuntime,
)
from semantic_3d_chat.chat.runtime import ChatAnswer, StaticChatRuntime
from semantic_3d_chat.chat.runtime_config import (
    effective_runtime_config_sha256,
    load_runtime_config,
    validate_runtime_config,
)
from semantic_3d_chat.config import PROJECT_ROOT, artifact_root
from semantic_3d_chat.evaluation.prediction_artifacts import (
    checkpoint_fingerprint,
)
from semantic_3d_chat.language.local_lm import prompt_token_ids, question_token_ids
from semantic_3d_chat.language.prefix_injection import (
    prefix_sha256,
    scene_boundary_mode_setting,
    scene_prefix_after_bos_setting,
)
from semantic_3d_chat.scene_encoder.question_control import FullSceneQuestionControl
from semantic_3d_chat.scene_encoder.question_control_v2 import (
    BoundedFullSceneQuestionControlV2,
)
from semantic_3d_chat.scene_encoder.question_control_v3 import (
    TeacherBasisFullSceneQuestionControlV3,
)
from semantic_3d_chat.scene_encoder.question_control_v4 import (
    SceneConditionedGateTeacherBasisControlV4,
)
from semantic_3d_chat.scene_encoder.question_control_v5 import (
    NormalizedFactorizedSceneQuestionControlV5,
)
from semantic_3d_chat.scene_encoder.question_control_v6 import (
    MagnitudeGatedTeacherBasisFullSceneQuestionControlV6,
)
from semantic_3d_chat.scene_encoder.question_control_v7 import (
    AlwaysOnTeacherBasisFullSceneQuestionControlV7,
)
from semantic_3d_chat.scene_encoder.question_control_v74 import (
    DenseFullSceneContinuousControlV74,
)
from semantic_3d_chat.scene_encoder.question_control_v75 import (
    DenseFullSceneContinuousControlV75,
)
from semantic_3d_chat.training.question_control_v4_checkpoint import (
    inherited_value_state_sha256,
)
from semantic_3d_chat.training.question_control_v5_checkpoint import (
    inherited_v60_state_sha256,
)
from semantic_3d_chat.training.question_control_v6_checkpoint import (
    v6_value_state_sha256,
)
from semantic_3d_chat.training.question_control_v7_checkpoint import (
    v7_value_state_sha256,
)
from semantic_3d_chat.training.question_control_v74_checkpoint import (
    V74_RUNTIME_ARCHITECTURE,
    V74_RUNTIME_METADATA_FIELDS,
    v74_state_sha256,
)
from semantic_3d_chat.training.question_control_v75_checkpoint import (
    V75_RUNTIME_ARCHITECTURE,
    V75_RUNTIME_METADATA_FIELDS,
    V75_RUNTIME_STATE_FIELDS,
    v75_state_sha256,
)

_SHA256 = re.compile(r"[0-9a-f]{64}")
_SAFE_ANSWER_PUNCTUATION: Final[frozenset[str]] = frozenset(".,;:'\"!?()-/")
_FORBIDDEN_CHECKPOINT_COMPONENTS = frozenset({"oracle", "qa"})
_SIGNATURE_CONTROL_TYPES = (
    BoundedFullSceneQuestionControlV2,
    TeacherBasisFullSceneQuestionControlV3,
    SceneConditionedGateTeacherBasisControlV4,
    NormalizedFactorizedSceneQuestionControlV5,
    MagnitudeGatedTeacherBasisFullSceneQuestionControlV6,
    AlwaysOnTeacherBasisFullSceneQuestionControlV7,
)


def _resolve(path: str | Path) -> Path:
    value = Path(path).expanduser()
    rooted = value if value.is_absolute() else PROJECT_ROOT / value
    return Path(os.path.abspath(rooted))


def question_control_training_artifact_root(config: Mapping[str, Any]) -> Path:
    """Return the exact derived-data training root beside runtime checkpoints.

    This is deliberately a path-root policy, not a forbidden component name:
    production inference imports implementation modules from
    ``semantic_3d_chat.training`` but must never read numeric teacher artifacts
    from the derived ``data_gemma4/training`` tree.
    """

    checkpoints = artifact_root(dict(config), "checkpoints").resolve()
    return checkpoints.parent / "training"


def block_question_control_training_artifacts(
    audit: FileAccessAudit | None,
    config: Mapping[str, Any],
) -> Path:
    """Add the exact training-artifact root to an active runtime audit."""

    root = question_control_training_artifact_root(config)
    if audit is not None and root not in audit.forbidden_roots:
        audit.forbidden_roots.append(root)
    return root


def _reject_symlink_components(path: Path) -> None:
    current = Path(path.anchor)
    for component in path.parts[1:]:
        current /= component
        if current.is_symlink():
            raise ValueError(
                f"Question-control checkpoint paths must not contain symlinks: {current}"
            )


def _checkpoint_files(checkpoint: str | Path) -> tuple[Path, Path, Path]:
    root = _resolve(checkpoint)
    _reject_symlink_components(root)
    if _FORBIDDEN_CHECKPOINT_COMPONENTS.intersection(
        component.casefold() for component in root.parts
    ):
        raise ValueError(
            "Question-control checkpoint must be physically separate from QA/oracle data"
        )
    if not root.is_dir():
        raise FileNotFoundError(f"Question-control checkpoint is unavailable: {root}")
    expected = {"control.safetensors", "runtime_metadata.json"}
    inventory = {item.name for item in root.iterdir()}
    if inventory != expected:
        raise ValueError(
            "Question-control checkpoint inventory must contain only sanitized runtime "
            f"files; expected={sorted(expected)} observed={sorted(inventory)}"
        )
    weights = root / "control.safetensors"
    metadata_path = root / "runtime_metadata.json"
    for item in (weights, metadata_path):
        if item.is_symlink() or not item.is_file():
            raise ValueError(
                f"Question-control checkpoint entries must be regular, non-symlink files: {item}"
            )
    return root, weights, metadata_path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sanitize_generated_answer(value: str) -> str:
    """Fail closed on decoder fragments that are not natural-language answers.

    This check is deliberately vocabulary-free: it neither maps outputs to an
    answer codebook nor adds scene information. It only prevents malformed
    symbolic fragments from reaching the interactive chat surface.
    """

    if not isinstance(value, str):
        raise TypeError("Generated answer must be text")
    normalized = " ".join(value.strip().split())
    if not normalized or not any(character.isalnum() for character in normalized):
        return "unknown"
    if any(
        not character.isascii()
        or not (
            character.isalnum()
            or character.isspace()
            or character in _SAFE_ANSWER_PUNCTUATION
        )
        for character in normalized
    ):
        return "unknown"
    return normalized


def _strict_json_object(path: Path) -> dict[str, Any]:
    def reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"Question-control runtime metadata has a duplicate field: {key}")
            result[key] = value
        return result

    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=reject_duplicate_keys,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("Question-control runtime metadata is invalid JSON") from error
    if not isinstance(value, dict):
        raise TypeError("Question-control runtime metadata must be a JSON object")
    return value


def _positive_integer(metadata: Mapping[str, Any], field: str) -> int:
    value = metadata.get(field)
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"Question-control {field} must be a positive integer")
    return value


def _finite_positive_number(metadata: Mapping[str, Any], field: str) -> float:
    value = metadata.get(field)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"Question-control {field} must be a finite positive number")
    converted = float(value)
    if not math.isfinite(converted) or converted <= 0.0:
        raise ValueError(f"Question-control {field} must be a finite positive number")
    return converted


def _load_control_head(
    checkpoint: str | Path,
    *,
    hidden_size: int,
    device: torch.device,
    audit: FileAccessAudit | None = None,
) -> tuple[
    FullSceneQuestionControl
    | BoundedFullSceneQuestionControlV2
    | TeacherBasisFullSceneQuestionControlV3
    | SceneConditionedGateTeacherBasisControlV4
    | NormalizedFactorizedSceneQuestionControlV5
    | MagnitudeGatedTeacherBasisFullSceneQuestionControlV6
    | AlwaysOnTeacherBasisFullSceneQuestionControlV7
    | DenseFullSceneContinuousControlV74
    | DenseFullSceneContinuousControlV75,
    dict[str, Any],
]:
    _, weights, metadata_path = _checkpoint_files(checkpoint)
    if audit is not None:
        audit.record(metadata_path)
    metadata = _strict_json_object(metadata_path)
    if metadata.get("architecture") == V75_RUNTIME_ARCHITECTURE:
        if set(metadata) != V75_RUNTIME_METADATA_FIELDS:
            raise ValueError("V75 question-control runtime metadata fields changed")
        uniform_floor_mass = _finite_positive_number(
            metadata, "uniform_floor_mass"
        )
        maximum_control_rms = _finite_positive_number(
            metadata, "maximum_control_rms"
        )
        environment_latents = _positive_integer(metadata, "environment_latents")
        query_count = _positive_integer(metadata, "query_count")
        model_dimension = _positive_integer(metadata, "model_dimension")
        decoder_hidden = _positive_integer(
            metadata, "coefficient_decoder_hidden_dimension"
        )
        output_basis_rank = _positive_integer(metadata, "output_basis_rank")
        if (
            type(metadata.get("schema_version")) is not int
            or metadata.get("schema_version") != 75
            or _positive_integer(metadata, "hidden_size") != hidden_size
            or _positive_integer(metadata, "control_tokens") != query_count
            or environment_latents > 4096
            or query_count > 32
            or model_dimension > 1024
            or decoder_hidden > 4096
            or output_basis_rank > hidden_size
            or uniform_floor_mass >= 1.0
            or maximum_control_rms > 1.0
            or metadata.get("saved_runtime_training_gate_required") is not True
            or metadata.get("saved_runtime_training_gate_passed") is not True
            or metadata.get("complete_scene_prefix_required") is not True
            or metadata.get("full_unchanged_prefix_retained_separately") is not True
            or metadata.get("prequestion_scene_key_value_cache") is not True
            or metadata.get("all_environment_latents_attended") is not True
            or metadata.get("positive_attention_floor") is not True
            or metadata.get("bilinear_question_scene_value_interaction") is not True
            or metadata.get("bias_free_nonlinear_coefficient_decoder") is not True
            or metadata.get("zero_preserving_coefficient_activation") is not True
            or metadata.get("question_only_output_path_exists") is not False
            or metadata.get("question_dependent_scene_retrieval") is not False
            or metadata.get("latent_selection_or_top_k_used") is not False
            or metadata.get("environmental_text_inputs") != []
            or metadata.get("training_answers_runtime_loaded") is not False
            or metadata.get("answer_text_runtime_loaded") is not False
            or metadata.get("answer_class_codebook_runtime_loaded") is not False
            or metadata.get("teacher_cache_runtime_loaded") is not False
            or metadata.get("oracle_runtime_loaded") is not False
            or metadata.get("question_or_answer_text_serialized") is not False
        ):
            raise ValueError("V75 question-control runtime contract mismatch")
        for field in (
            "weights_sha256",
            "base_checkpoint_sha256",
            "base_runtime_config_sha256",
            "source_v75_candidate_sha256",
            "source_v75_training_fit_state_sha256",
            "saved_runtime_training_gate_attestation_sha256",
        ):
            value = metadata.get(field)
            if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
                raise ValueError(f"V75 question-control {field} digest is invalid")
        if _sha256(weights) != metadata["weights_sha256"]:
            raise ValueError("V75 question-control weights changed")
        if audit is not None:
            audit.record(weights)
        state = load_file(str(weights), device="cpu")
        if set(state) != V75_RUNTIME_STATE_FIELDS:
            raise ValueError("V75 question-control state fields changed")
        if any(
            not value.is_floating_point() or not torch.isfinite(value).all()
            for value in state.values()
        ):
            raise ValueError(
                "V75 question-control checkpoint contains nonfinite or nonfloat tensors"
            )
        basis = state["output_basis"]
        key = state["key.weight"]
        value = state["value.weight"]
        query = state["query.weight"]
        coefficient_hidden = state["coefficient_hidden.weight"]
        coefficient_output = state["coefficient_output.weight"]
        if (
            basis.ndim != 2
            or tuple(basis.shape) != (output_basis_rank, hidden_size)
            or key.ndim != 2
            or tuple(key.shape) != (model_dimension, hidden_size)
            or tuple(value.shape) != tuple(key.shape)
            or tuple(query.shape)
            != (query_count * model_dimension, hidden_size)
            or tuple(coefficient_hidden.shape)
            != (decoder_hidden, query_count * model_dimension)
            or tuple(coefficient_output.shape)
            != (query_count * output_basis_rank, decoder_hidden)
        ):
            raise ValueError("V75 question-control tensor shapes changed")
        module = DenseFullSceneContinuousControlV75(
            hidden_size,
            basis,
            environment_latents=environment_latents,
            query_count=query_count,
            model_dimension=model_dimension,
            coefficient_decoder_hidden_dimension=decoder_hidden,
            uniform_floor_mass=uniform_floor_mass,
            maximum_control_rms=maximum_control_rms,
        )
        module.load_state_dict(state, strict=True)
        if v75_state_sha256(module) != metadata[
            "source_v75_training_fit_state_sha256"
        ]:
            raise ValueError("V75 numeric training-fit state changed")
        module = module.to(device=device, dtype=torch.float32).eval()
        return module, metadata
    if metadata.get("architecture") == V74_RUNTIME_ARCHITECTURE:
        if set(metadata) != V74_RUNTIME_METADATA_FIELDS:
            raise ValueError("V74 question-control runtime metadata fields changed")
        uniform_floor_mass = _finite_positive_number(
            metadata, "uniform_floor_mass"
        )
        maximum_control_rms = _finite_positive_number(
            metadata, "maximum_control_rms"
        )
        query_count = _positive_integer(metadata, "query_count")
        if (
            type(metadata.get("schema_version")) is not int
            or metadata.get("schema_version") != 74
            or _positive_integer(metadata, "hidden_size") != hidden_size
            or _positive_integer(metadata, "control_tokens") != query_count
            or uniform_floor_mass >= 1.0
            or maximum_control_rms > 1.0
            or metadata.get("saved_runtime_training_gate_required") is not True
            or metadata.get("saved_runtime_training_gate_passed") is not True
            or metadata.get("complete_scene_prefix_required") is not True
            or metadata.get("full_unchanged_prefix_retained_separately") is not True
            or metadata.get("prequestion_scene_key_value_cache") is not True
            or metadata.get("all_environment_latents_attended") is not True
            or metadata.get("positive_attention_floor") is not True
            or metadata.get("bilinear_question_scene_value_interaction") is not True
            or metadata.get("question_only_output_path_exists") is not False
            or metadata.get("question_dependent_scene_retrieval") is not False
            or metadata.get("latent_selection_or_top_k_used") is not False
            or metadata.get("environmental_text_inputs") != []
            or metadata.get("training_answers_runtime_loaded") is not False
            or metadata.get("answer_text_runtime_loaded") is not False
            or metadata.get("answer_class_codebook_runtime_loaded") is not False
            or metadata.get("teacher_cache_runtime_loaded") is not False
            or metadata.get("oracle_runtime_loaded") is not False
            or metadata.get("question_or_answer_text_serialized") is not False
        ):
            raise ValueError("V74 question-control runtime contract mismatch")
        for field in (
            "weights_sha256",
            "base_checkpoint_sha256",
            "base_runtime_config_sha256",
            "source_v74_training_fit_state_sha256",
            "saved_runtime_training_gate_attestation_sha256",
        ):
            value = metadata.get(field)
            if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
                raise ValueError(f"V74 question-control {field} digest is invalid")
        if _sha256(weights) != metadata["weights_sha256"]:
            raise ValueError("V74 question-control weights changed")
        if audit is not None:
            audit.record(weights)
        state = load_file(str(weights), device="cpu")
        basis = state.get("output_basis")
        if basis is None:
            raise ValueError("V74 checkpoint lacks its numeric output basis")
        module = DenseFullSceneContinuousControlV74(
            hidden_size,
            basis,
            environment_latents=_positive_integer(
                metadata, "environment_latents"
            ),
            query_count=query_count,
            model_dimension=_positive_integer(metadata, "model_dimension"),
            uniform_floor_mass=uniform_floor_mass,
            maximum_control_rms=maximum_control_rms,
        )
        if module.output_basis_rank != _positive_integer(
            metadata, "output_basis_rank"
        ):
            raise ValueError("V74 output-basis rank changed")
        missing, unexpected = module.load_state_dict(state, strict=False)
        if missing or unexpected:
            raise ValueError(
                "V74 question-control state mismatch: "
                f"missing={missing} unexpected={unexpected}"
            )
        if v74_state_sha256(module) != metadata[
            "source_v74_training_fit_state_sha256"
        ]:
            raise ValueError("V74 numeric training-fit state changed")
        module = module.to(device=device, dtype=torch.float32).eval()
        if any(
            not torch.isfinite(value).all()
            for value in module.state_dict().values()
        ):
            raise ValueError("V74 question-control checkpoint contains nonfinite tensors")
        return module, metadata
    if metadata.get("architecture") == "always_on_teacher_basis_full_scene_control_v7":
        required_v7 = {
            "schema_version",
            "architecture",
            "hidden_size",
            "control_tokens",
            "expected_environment_latents",
            "moment_count",
            "interaction_dim",
            "trunk_dim",
            "output_basis_rank",
            "maximum_control_rms",
            "initial_control_rms",
            "weights_sha256",
            "base_checkpoint_sha256",
            "base_runtime_config_sha256",
            "source_v66_training_fit_state_sha256",
            "always_on_continuous_control",
            "legacy_route_parameters_ignored",
            "question_dependent_scene_retrieval",
            "complete_scene_prefix_required",
            "environmental_text_inputs",
            "training_answers_runtime_loaded",
            "answer_class_codebook_runtime_loaded",
            "fixed_global_scene_moments",
            "boundary_tokens_excluded_from_scene_signature",
            "softmax_scene_attention_used",
            "control_values_scene_question_bilinear",
            "gate_scene_question_conditioned",
            "route_is_environmental_retrieval",
            "saved_runtime_training_gate_required",
            "saved_runtime_training_gate_passed",
            "saved_runtime_training_gate_attestation_sha256",
        }
        if set(metadata) != required_v7:
            raise ValueError("V7 question-control runtime metadata fields changed")
        maximum_rms = _finite_positive_number(metadata, "maximum_control_rms")
        if (
            type(metadata.get("schema_version")) is not int
            or metadata.get("schema_version") != 7
            or _positive_integer(metadata, "hidden_size") != hidden_size
            or metadata.get("always_on_continuous_control") is not True
            or metadata.get("legacy_route_parameters_ignored") is not True
            or metadata.get("question_dependent_scene_retrieval") is not False
            or metadata.get("complete_scene_prefix_required") is not True
            or metadata.get("environmental_text_inputs") != []
            or metadata.get("training_answers_runtime_loaded") is not False
            or metadata.get("answer_class_codebook_runtime_loaded") is not False
            or metadata.get("fixed_global_scene_moments") is not True
            or metadata.get("boundary_tokens_excluded_from_scene_signature") is not True
            or metadata.get("softmax_scene_attention_used") is not False
            or metadata.get("control_values_scene_question_bilinear") is not True
            or metadata.get("gate_scene_question_conditioned") is not False
            or metadata.get("route_is_environmental_retrieval") is not False
            or metadata.get("saved_runtime_training_gate_required") is not True
            or metadata.get("saved_runtime_training_gate_passed") is not True
        ):
            raise ValueError("V7 question-control runtime contract mismatch")
        for field in (
            "weights_sha256",
            "base_checkpoint_sha256",
            "base_runtime_config_sha256",
            "source_v66_training_fit_state_sha256",
            "saved_runtime_training_gate_attestation_sha256",
        ):
            value = metadata.get(field)
            if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
                raise ValueError(f"V7 question-control {field} digest is invalid")
        if _sha256(weights) != metadata["weights_sha256"]:
            raise ValueError("V7 question-control weights changed")
        if audit is not None:
            audit.record(weights)
        state = load_file(str(weights), device="cpu")
        basis = state.get("output_basis")
        if basis is None:
            raise ValueError("V7 checkpoint lacks its numeric output basis")
        module = AlwaysOnTeacherBasisFullSceneQuestionControlV7(
            hidden_size,
            basis,
            control_tokens=_positive_integer(metadata, "control_tokens"),
            expected_environment_latents=_positive_integer(
                metadata, "expected_environment_latents"
            ),
            moment_count=_positive_integer(metadata, "moment_count"),
            interaction_dim=_positive_integer(metadata, "interaction_dim"),
            trunk_dim=_positive_integer(metadata, "trunk_dim"),
            maximum_control_rms=maximum_rms,
            initial_control_rms=_finite_positive_number(metadata, "initial_control_rms"),
        )
        if module.output_basis_rank != _positive_integer(metadata, "output_basis_rank"):
            raise ValueError("V7 output-basis rank changed")
        missing, unexpected = module.load_state_dict(state, strict=False)
        if missing or unexpected:
            raise ValueError(
                f"V7 question-control state mismatch: missing={missing} unexpected={unexpected}"
            )
        if v7_value_state_sha256(module) != metadata["source_v66_training_fit_state_sha256"]:
            raise ValueError("V7 numeric V66 training-fit state changed")
        module = module.to(device=device, dtype=torch.float32).eval()
        if any(not torch.isfinite(value).all() for value in module.state_dict().values()):
            raise ValueError("V7 question-control checkpoint contains nonfinite tensors")
        return module, metadata
    if metadata.get("architecture") == "magnitude_gated_teacher_basis_full_scene_control_v6":
        required_v6 = {
            "schema_version",
            "architecture",
            "hidden_size",
            "control_tokens",
            "expected_environment_latents",
            "moment_count",
            "interaction_dim",
            "trunk_dim",
            "output_basis_rank",
            "maximum_control_rms",
            "initial_control_rms",
            "activation_rms_threshold",
            "activation_rms_aggregation",
            "weights_sha256",
            "base_checkpoint_sha256",
            "base_runtime_config_sha256",
            "source_v65_training_fit_state_sha256",
            "source_v65_value_state_sha256",
            "magnitude_gated_continuous_control",
            "exact_no_control_below_threshold",
            "question_dependent_scene_retrieval",
            "complete_scene_prefix_required",
            "environmental_text_inputs",
            "fixed_global_scene_moments",
            "boundary_tokens_excluded_from_scene_signature",
            "softmax_scene_attention_used",
            "control_values_scene_question_bilinear",
            "gate_scene_question_conditioned",
            "route_is_environmental_retrieval",
            "saved_runtime_training_gate_required",
            "saved_runtime_training_gate_passed",
            "saved_runtime_training_gate_attestation_sha256",
        }
        if set(metadata) != required_v6:
            raise ValueError("V6 question-control runtime metadata fields changed")
        activation_threshold = _finite_positive_number(metadata, "activation_rms_threshold")
        maximum_rms = _finite_positive_number(metadata, "maximum_control_rms")
        if (
            type(metadata.get("schema_version")) is not int
            or metadata.get("schema_version") != 6
            or _positive_integer(metadata, "hidden_size") != hidden_size
            or activation_threshold >= maximum_rms
            or metadata.get("activation_rms_aggregation") != "maximum_over_control_tokens"
            or metadata.get("magnitude_gated_continuous_control") is not True
            or metadata.get("exact_no_control_below_threshold") is not True
            or metadata.get("question_dependent_scene_retrieval") is not False
            or metadata.get("complete_scene_prefix_required") is not True
            or metadata.get("environmental_text_inputs") != []
            or metadata.get("fixed_global_scene_moments") is not True
            or metadata.get("boundary_tokens_excluded_from_scene_signature") is not True
            or metadata.get("softmax_scene_attention_used") is not False
            or metadata.get("control_values_scene_question_bilinear") is not True
            or metadata.get("gate_scene_question_conditioned") is not True
            or metadata.get("route_is_environmental_retrieval") is not False
            or metadata.get("saved_runtime_training_gate_required") is not True
            or metadata.get("saved_runtime_training_gate_passed") is not True
        ):
            raise ValueError("V6 question-control runtime contract mismatch")
        for field in (
            "weights_sha256",
            "base_checkpoint_sha256",
            "base_runtime_config_sha256",
            "source_v65_training_fit_state_sha256",
            "source_v65_value_state_sha256",
            "saved_runtime_training_gate_attestation_sha256",
        ):
            value = metadata.get(field)
            if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
                raise ValueError(f"V6 question-control {field} digest is invalid")
        if _sha256(weights) != metadata["weights_sha256"]:
            raise ValueError("V6 question-control weights changed")
        if audit is not None:
            audit.record(weights)
        state = load_file(str(weights), device="cpu")
        basis = state.get("output_basis")
        if basis is None:
            raise ValueError("V6 checkpoint lacks its numeric output basis")
        module = MagnitudeGatedTeacherBasisFullSceneQuestionControlV6(
            hidden_size,
            basis,
            control_tokens=_positive_integer(metadata, "control_tokens"),
            expected_environment_latents=_positive_integer(
                metadata, "expected_environment_latents"
            ),
            moment_count=_positive_integer(metadata, "moment_count"),
            interaction_dim=_positive_integer(metadata, "interaction_dim"),
            trunk_dim=_positive_integer(metadata, "trunk_dim"),
            maximum_control_rms=maximum_rms,
            initial_control_rms=_finite_positive_number(metadata, "initial_control_rms"),
            activation_rms_threshold=activation_threshold,
        )
        if module.output_basis_rank != _positive_integer(metadata, "output_basis_rank"):
            raise ValueError("V6 output-basis rank changed")
        missing, unexpected = module.load_state_dict(state, strict=False)
        if missing or unexpected:
            raise ValueError(
                f"V6 question-control state mismatch: missing={missing} unexpected={unexpected}"
            )
        if (
            metadata["source_v65_training_fit_state_sha256"]
            != metadata["source_v65_value_state_sha256"]
        ):
            raise ValueError("V6 saved value state differs from its V65 training fit")
        if v6_value_state_sha256(module) != metadata["source_v65_training_fit_state_sha256"]:
            raise ValueError("V6 numeric V65 value state changed")
        module = module.to(device=device, dtype=torch.float32).eval()
        if any(not torch.isfinite(value).all() for value in module.state_dict().values()):
            raise ValueError("V6 question-control checkpoint contains nonfinite tensors")
        return module, metadata
    if metadata.get("architecture") == "normalized_factorized_scene_question_route_v5":
        required_v5 = {
            "schema_version",
            "architecture",
            "hidden_size",
            "control_tokens",
            "expected_environment_latents",
            "moment_count",
            "interaction_dim",
            "trunk_dim",
            "output_basis_rank",
            "route_factor_rank",
            "maximum_control_rms",
            "initial_control_rms",
            "gate_threshold",
            "weights_sha256",
            "base_checkpoint_sha256",
            "base_runtime_config_sha256",
            "source_v60_checkpoint_sha256",
            "inherited_value_state_sha256",
            "only_factorized_gate_trainable",
            "question_dependent_scene_retrieval",
            "complete_scene_prefix_required",
            "environmental_text_inputs",
            "exact_no_control_route",
            "fixed_global_scene_moments",
            "boundary_tokens_excluded_from_scene_signature",
            "softmax_scene_attention_used",
            "control_values_scene_question_bilinear",
            "gate_scene_question_conditioned",
            "separate_question_scene_route_projections",
            "all_scene_moments_consumed_by_route",
            "normalized_route_factors",
            "low_rank_bilinear_route",
            "route_uses_inherited_value_trunk",
            "route_is_environmental_retrieval",
        }
        if set(metadata) != required_v5:
            raise ValueError("V5 question-control runtime metadata fields changed")
        if (
            type(metadata.get("schema_version")) is not int
            or metadata.get("schema_version") != 5
            or _positive_integer(metadata, "hidden_size") != hidden_size
            or metadata.get("only_factorized_gate_trainable") is not True
            or metadata.get("question_dependent_scene_retrieval") is not False
            or metadata.get("complete_scene_prefix_required") is not True
            or metadata.get("environmental_text_inputs") != []
            or metadata.get("exact_no_control_route") is not True
            or metadata.get("fixed_global_scene_moments") is not True
            or metadata.get("boundary_tokens_excluded_from_scene_signature") is not True
            or metadata.get("softmax_scene_attention_used") is not False
            or metadata.get("control_values_scene_question_bilinear") is not True
            or metadata.get("gate_scene_question_conditioned") is not True
            or metadata.get("separate_question_scene_route_projections") is not True
            or metadata.get("all_scene_moments_consumed_by_route") is not True
            or metadata.get("normalized_route_factors") is not True
            or metadata.get("low_rank_bilinear_route") is not True
            or metadata.get("route_uses_inherited_value_trunk") is not False
            or metadata.get("route_is_environmental_retrieval") is not False
        ):
            raise ValueError("V5 question-control runtime contract mismatch")
        for field in (
            "base_checkpoint_sha256",
            "base_runtime_config_sha256",
            "source_v60_checkpoint_sha256",
            "inherited_value_state_sha256",
        ):
            value = metadata.get(field)
            if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
                raise ValueError(f"V5 question-control {field} digest is invalid")
        expected_weights = metadata.get("weights_sha256")
        if not isinstance(expected_weights, str) or _SHA256.fullmatch(expected_weights) is None:
            raise ValueError("V5 question-control weights digest is invalid")
        if _sha256(weights) != expected_weights:
            raise ValueError("V5 question-control weights changed")
        if audit is not None:
            audit.record(weights)
        state = load_file(str(weights), device="cpu")
        basis = state.get("output_basis")
        if basis is None:
            raise ValueError("V5 checkpoint lacks its numeric output basis")
        module = NormalizedFactorizedSceneQuestionControlV5(
            hidden_size,
            basis,
            control_tokens=_positive_integer(metadata, "control_tokens"),
            expected_environment_latents=_positive_integer(
                metadata, "expected_environment_latents"
            ),
            moment_count=_positive_integer(metadata, "moment_count"),
            interaction_dim=_positive_integer(metadata, "interaction_dim"),
            trunk_dim=_positive_integer(metadata, "trunk_dim"),
            maximum_control_rms=_finite_positive_number(metadata, "maximum_control_rms"),
            initial_control_rms=_finite_positive_number(metadata, "initial_control_rms"),
            gate_threshold=_finite_positive_number(metadata, "gate_threshold"),
            route_factor_rank=_positive_integer(metadata, "route_factor_rank"),
        )
        if module.output_basis_rank != _positive_integer(metadata, "output_basis_rank"):
            raise ValueError("V5 output-basis rank changed")
        missing, unexpected = module.load_state_dict(state, strict=False)
        if missing or unexpected:
            raise ValueError(
                f"V5 question-control state mismatch: missing={missing} unexpected={unexpected}"
            )
        module.freeze_inherited_v60_state()
        if inherited_v60_state_sha256(module) != metadata["inherited_value_state_sha256"]:
            raise ValueError("V5 inherited V60 value state digest changed")
        module = module.to(device=device, dtype=torch.float32).eval()
        if any(not torch.isfinite(value).all() for value in module.state_dict().values()):
            raise ValueError("V5 question-control checkpoint contains nonfinite tensors")
        return module, metadata
    if metadata.get("architecture") == "scene_conditioned_gate_teacher_basis_control_v4":
        required_v4 = {
            "schema_version",
            "architecture",
            "hidden_size",
            "control_tokens",
            "expected_environment_latents",
            "moment_count",
            "interaction_dim",
            "trunk_dim",
            "output_basis_rank",
            "gate_hidden_dim",
            "maximum_control_rms",
            "initial_control_rms",
            "gate_threshold",
            "weights_sha256",
            "base_checkpoint_sha256",
            "base_runtime_config_sha256",
            "source_v60_checkpoint_sha256",
            "inherited_value_state_sha256",
            "only_gate_trainable",
            "question_dependent_scene_retrieval",
            "complete_scene_prefix_required",
            "environmental_text_inputs",
            "exact_no_control_route",
            "fixed_global_scene_moments",
            "boundary_tokens_excluded_from_scene_signature",
            "softmax_scene_attention_used",
            "control_values_scene_question_bilinear",
            "gate_scene_question_conditioned",
            "route_is_environmental_retrieval",
        }
        if set(metadata) != required_v4:
            raise ValueError("V4 question-control runtime metadata fields changed")
        if (
            type(metadata.get("schema_version")) is not int
            or metadata.get("schema_version") != 4
            or _positive_integer(metadata, "hidden_size") != hidden_size
            or metadata.get("only_gate_trainable") is not True
            or metadata.get("question_dependent_scene_retrieval") is not False
            or metadata.get("complete_scene_prefix_required") is not True
            or metadata.get("environmental_text_inputs") != []
            or metadata.get("exact_no_control_route") is not True
            or metadata.get("fixed_global_scene_moments") is not True
            or metadata.get("boundary_tokens_excluded_from_scene_signature") is not True
            or metadata.get("softmax_scene_attention_used") is not False
            or metadata.get("control_values_scene_question_bilinear") is not True
            or metadata.get("gate_scene_question_conditioned") is not True
            or metadata.get("route_is_environmental_retrieval") is not False
        ):
            raise ValueError("V4 question-control runtime contract mismatch")
        for field in (
            "base_checkpoint_sha256",
            "base_runtime_config_sha256",
            "source_v60_checkpoint_sha256",
            "inherited_value_state_sha256",
        ):
            value = metadata.get(field)
            if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
                raise ValueError(f"V4 question-control {field} digest is invalid")
        expected_weights = metadata.get("weights_sha256")
        if not isinstance(expected_weights, str) or _SHA256.fullmatch(expected_weights) is None:
            raise ValueError("V4 question-control weights digest is invalid")
        if _sha256(weights) != expected_weights:
            raise ValueError("V4 question-control weights changed")
        if audit is not None:
            audit.record(weights)
        state = load_file(str(weights), device="cpu")
        basis = state.get("output_basis")
        if basis is None:
            raise ValueError("V4 checkpoint lacks its numeric output basis")
        module = SceneConditionedGateTeacherBasisControlV4(
            hidden_size,
            basis,
            control_tokens=_positive_integer(metadata, "control_tokens"),
            expected_environment_latents=_positive_integer(
                metadata, "expected_environment_latents"
            ),
            moment_count=_positive_integer(metadata, "moment_count"),
            interaction_dim=_positive_integer(metadata, "interaction_dim"),
            trunk_dim=_positive_integer(metadata, "trunk_dim"),
            maximum_control_rms=_finite_positive_number(metadata, "maximum_control_rms"),
            initial_control_rms=_finite_positive_number(metadata, "initial_control_rms"),
            gate_threshold=_finite_positive_number(metadata, "gate_threshold"),
            gate_hidden_dim=_positive_integer(metadata, "gate_hidden_dim"),
        )
        if module.output_basis_rank != _positive_integer(metadata, "output_basis_rank"):
            raise ValueError("V4 output-basis rank changed")
        missing, unexpected = module.load_state_dict(state, strict=False)
        if missing or unexpected:
            raise ValueError(
                f"V4 question-control state mismatch: missing={missing} unexpected={unexpected}"
            )
        module.freeze_inherited_v60_state()
        if inherited_value_state_sha256(module) != metadata["inherited_value_state_sha256"]:
            raise ValueError("V4 inherited V60 value state digest changed")
        module = module.to(device=device, dtype=torch.float32).eval()
        if any(not torch.isfinite(value).all() for value in module.state_dict().values()):
            raise ValueError("V4 question-control checkpoint contains nonfinite tensors")
        return module, metadata
    if metadata.get("architecture") == "teacher_basis_full_scene_question_control_v3":
        required_v3 = {
            "schema_version",
            "architecture",
            "hidden_size",
            "control_tokens",
            "expected_environment_latents",
            "moment_count",
            "interaction_dim",
            "trunk_dim",
            "output_basis_rank",
            "maximum_control_rms",
            "initial_control_rms",
            "gate_threshold",
            "weights_sha256",
            "base_checkpoint_sha256",
            "base_runtime_config_sha256",
            "question_dependent_scene_retrieval",
            "complete_scene_prefix_required",
            "environmental_text_inputs",
            "exact_no_control_route",
            "fixed_global_scene_moments",
            "boundary_tokens_excluded_from_scene_signature",
            "softmax_scene_attention_used",
            "control_values_scene_question_bilinear",
            "route_is_environmental_retrieval",
        }
        if set(metadata) != required_v3:
            raise ValueError("V3 question-control runtime metadata fields changed")
        if (
            type(metadata.get("schema_version")) is not int
            or metadata.get("schema_version") != 3
            or _positive_integer(metadata, "hidden_size") != hidden_size
            or metadata.get("question_dependent_scene_retrieval") is not False
            or metadata.get("complete_scene_prefix_required") is not True
            or metadata.get("environmental_text_inputs") != []
            or metadata.get("exact_no_control_route") is not True
            or metadata.get("fixed_global_scene_moments") is not True
            or metadata.get("boundary_tokens_excluded_from_scene_signature") is not True
            or metadata.get("softmax_scene_attention_used") is not False
            or metadata.get("control_values_scene_question_bilinear") is not True
            or metadata.get("route_is_environmental_retrieval") is not False
        ):
            raise ValueError("V3 question-control runtime contract mismatch")
        for field in ("base_checkpoint_sha256", "base_runtime_config_sha256"):
            value = metadata.get(field)
            if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
                raise ValueError(f"V3 question-control {field} digest is invalid")
        expected_weights = metadata.get("weights_sha256")
        if not isinstance(expected_weights, str) or _SHA256.fullmatch(expected_weights) is None:
            raise ValueError("V3 question-control weights digest is invalid")
        if _sha256(weights) != expected_weights:
            raise ValueError("V3 question-control weights changed")
        if audit is not None:
            audit.record(weights)
        state = load_file(str(weights), device="cpu")
        basis = state.get("output_basis")
        if basis is None:
            raise ValueError("V3 checkpoint lacks its numeric output basis")
        module = TeacherBasisFullSceneQuestionControlV3(
            hidden_size,
            basis,
            control_tokens=_positive_integer(metadata, "control_tokens"),
            expected_environment_latents=_positive_integer(
                metadata, "expected_environment_latents"
            ),
            moment_count=_positive_integer(metadata, "moment_count"),
            interaction_dim=_positive_integer(metadata, "interaction_dim"),
            trunk_dim=_positive_integer(metadata, "trunk_dim"),
            maximum_control_rms=_finite_positive_number(metadata, "maximum_control_rms"),
            initial_control_rms=_finite_positive_number(metadata, "initial_control_rms"),
            gate_threshold=_finite_positive_number(metadata, "gate_threshold"),
        )
        if module.output_basis_rank != _positive_integer(metadata, "output_basis_rank"):
            raise ValueError("V3 output-basis rank changed")
        missing, unexpected = module.load_state_dict(state, strict=False)
        if missing or unexpected:
            raise ValueError(
                f"V3 question-control state mismatch: missing={missing} unexpected={unexpected}"
            )
        module = module.to(device=device, dtype=torch.float32).eval()
        if any(not torch.isfinite(value).all() for value in module.state_dict().values()):
            raise ValueError("V3 question-control checkpoint contains nonfinite tensors")
        return module, metadata
    if metadata.get("architecture") == "bounded_global_scene_question_control_v2":
        required_v2 = {
            "schema_version",
            "architecture",
            "hidden_size",
            "control_tokens",
            "expected_environment_latents",
            "moment_count",
            "interaction_dim",
            "output_rank",
            "maximum_control_rms",
            "initial_control_rms",
            "gate_threshold",
            "weights_sha256",
            "base_checkpoint_sha256",
            "base_runtime_config_sha256",
            "source_control_checkpoint_sha256",
            "question_dependent_scene_retrieval",
            "complete_scene_prefix_required",
            "environmental_text_inputs",
            "exact_no_control_route",
            "fixed_global_scene_moments",
            "boundary_tokens_excluded_from_scene_signature",
            "softmax_scene_attention_used",
        }
        if set(metadata) != required_v2:
            raise ValueError("V2 question-control runtime metadata fields changed")
        if (
            type(metadata.get("schema_version")) is not int
            or metadata.get("schema_version") != 2
            or _positive_integer(metadata, "hidden_size") != hidden_size
            or metadata.get("question_dependent_scene_retrieval") is not False
            or metadata.get("complete_scene_prefix_required") is not True
            or metadata.get("environmental_text_inputs") != []
            or metadata.get("exact_no_control_route") is not True
            or metadata.get("fixed_global_scene_moments") is not True
            or metadata.get("boundary_tokens_excluded_from_scene_signature") is not True
            or metadata.get("softmax_scene_attention_used") is not False
        ):
            raise ValueError("V2 question-control runtime contract mismatch")
        for field in (
            "base_checkpoint_sha256",
            "base_runtime_config_sha256",
            "source_control_checkpoint_sha256",
        ):
            value = metadata.get(field)
            if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
                raise ValueError(f"V2 question-control {field} digest is invalid")
        expected_weights = metadata.get("weights_sha256")
        if not isinstance(expected_weights, str) or _SHA256.fullmatch(expected_weights) is None:
            raise ValueError("V2 question-control weights digest is invalid")
        if _sha256(weights) != expected_weights:
            raise ValueError("V2 question-control weights changed")
        module = BoundedFullSceneQuestionControlV2(
            hidden_size,
            control_tokens=_positive_integer(metadata, "control_tokens"),
            expected_environment_latents=_positive_integer(
                metadata, "expected_environment_latents"
            ),
            moment_count=_positive_integer(metadata, "moment_count"),
            interaction_dim=_positive_integer(metadata, "interaction_dim"),
            output_rank=_positive_integer(metadata, "output_rank"),
            maximum_control_rms=_finite_positive_number(metadata, "maximum_control_rms"),
            initial_control_rms=_finite_positive_number(metadata, "initial_control_rms"),
            gate_threshold=_finite_positive_number(metadata, "gate_threshold"),
        )
        if module.gate_threshold >= 1.0:
            raise ValueError("V2 question-control gate_threshold must be below one")
        if audit is not None:
            audit.record(weights)
        state = load_file(str(weights), device="cpu")
        missing, unexpected = module.load_state_dict(state, strict=False)
        if missing or unexpected:
            raise ValueError(
                f"V2 question-control state mismatch: missing={missing} unexpected={unexpected}"
            )
        module = module.to(device=device, dtype=torch.float32).eval()
        if any(not torch.isfinite(value).all() for value in module.state_dict().values()):
            raise ValueError("V2 question-control checkpoint contains nonfinite tensors")
        return module, metadata
    required = {
        "schema_version",
        "architecture",
        "hidden_size",
        "attention_dim",
        "control_tokens",
        "uniform_floor",
        "output_scale",
        "weights_sha256",
        "base_checkpoint_sha256",
        "base_runtime_config_sha256",
        "question_dependent_scene_retrieval",
        "complete_scene_prefix_required",
        "environmental_text_inputs",
    }
    if set(metadata) != required:
        raise ValueError("Question-control runtime metadata fields changed")
    if type(metadata.get("schema_version")) is not int or metadata.get("schema_version") != 1:
        raise ValueError("Question-control runtime contract mismatch")
    metadata_hidden_size = _positive_integer(metadata, "hidden_size")
    if (
        metadata.get("architecture") != "full_scene_question_control_v1"
        or metadata_hidden_size != hidden_size
        or metadata.get("question_dependent_scene_retrieval") is not False
        or metadata.get("complete_scene_prefix_required") is not True
        or metadata.get("environmental_text_inputs") != []
    ):
        raise ValueError("Question-control runtime contract mismatch")
    attention_dim = _positive_integer(metadata, "attention_dim")
    control_tokens = _positive_integer(metadata, "control_tokens")
    uniform_floor = _finite_positive_number(metadata, "uniform_floor")
    if uniform_floor > 1.0:
        raise ValueError("Question-control uniform_floor must not exceed one")
    output_scale = _finite_positive_number(metadata, "output_scale")
    for field in ("base_checkpoint_sha256", "base_runtime_config_sha256"):
        value = metadata.get(field)
        if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
            raise ValueError(f"Question-control {field} digest is invalid")
    expected_weights = metadata.get("weights_sha256")
    if not isinstance(expected_weights, str) or _SHA256.fullmatch(expected_weights) is None:
        raise ValueError("Question-control weights digest is invalid")
    if _sha256(weights) != expected_weights:
        raise ValueError("Question-control weights changed")
    module = FullSceneQuestionControl(
        hidden_size,
        attention_dim=attention_dim,
        control_tokens=control_tokens,
        uniform_floor=uniform_floor,
        output_scale=output_scale,
    )
    if audit is not None:
        # Safetensors may use native reads that do not emit CPython open events.
        audit.record(weights)
    state = load_file(str(weights), device="cpu")
    missing, unexpected = module.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise ValueError(
            f"Question-control state mismatch: missing={missing} unexpected={unexpected}"
        )
    module = module.to(device=device, dtype=torch.float32).eval()
    if any(not torch.isfinite(value).all() for value in module.state_dict().values()):
        raise ValueError("Question-control checkpoint contains nonfinite tensors")
    return module, metadata


class QuestionControlledChatRuntime:
    """Preserve one static scene prefix and derive only continuous query tokens."""

    def __init__(
        self,
        base: StaticChatRuntime,
        control: (
            FullSceneQuestionControl
            | BoundedFullSceneQuestionControlV2
            | TeacherBasisFullSceneQuestionControlV3
            | SceneConditionedGateTeacherBasisControlV4
            | NormalizedFactorizedSceneQuestionControlV5
            | MagnitudeGatedTeacherBasisFullSceneQuestionControlV6
            | AlwaysOnTeacherBasisFullSceneQuestionControlV7
            | DenseFullSceneContinuousControlV74
            | DenseFullSceneContinuousControlV75
        ),
        control_metadata: dict[str, Any],
        grounding_sidecar: V78GroundingSidecarRuntime | None = None,
    ) -> None:
        self.base = base
        self.control = control
        self.control_metadata = dict(control_metadata)
        self.grounding_sidecar = grounding_sidecar
        self._grounding_scene_prefix = (
            None
            if grounding_sidecar is None
            else base.scene_prefix.detach().clone()
        )
        self.scene_prefix_hash = base.scene_prefix_hash
        self._questions_answered = 0
        self.last_control_audit: dict[str, Any] | None = None
        self.last_control_tokens_sha256: str | None = None
        self.last_environment_conditioned_input_sha256: str = self.scene_prefix_hash
        self.last_grounding_audit: dict[str, Any] | None = None
        self._scene_control_signature: torch.Tensor | None = None
        self._scene_control_signature_hash: str | None = None
        self._scene_control_key: torch.Tensor | None = None
        self._scene_control_value: torch.Tensor | None = None
        if isinstance(control, DenseFullSceneContinuousControlV74):
            # Dense-reader scene-only key/value projections are computed exactly once
            # before user text is accepted.  The immutable full prefix remains
            # separately supplied to Gemma on every answer.
            with torch.inference_mode():
                key, value = control.encode_scene(
                    base.scene_prefix.detach().float()
                )
            expected = (
                1,
                control.environment_latents,
                control.model_dimension,
            )
            if (
                not isinstance(key, torch.Tensor)
                or not isinstance(value, torch.Tensor)
                or tuple(key.shape) != expected
                or tuple(value.shape) != expected
                or not torch.isfinite(key).all()
                or not torch.isfinite(value).all()
            ):
                raise RuntimeError("Dense reader produced an invalid pre-question K/V cache")
            self._scene_control_key = key.detach().clone()
            self._scene_control_value = value.detach().clone()
            self._scene_control_signature_hash = prefix_sha256(
                torch.cat(
                    (self._scene_control_key, self._scene_control_value), dim=-1
                )
            )
        elif isinstance(control, _SIGNATURE_CONTROL_TYPES):
            # This is the sole full-scene control encoding.  It runs while the
            # runtime is being constructed, before user text is accepted, and
            # is detached so inference cannot retain a training graph.
            with torch.inference_mode():
                signature = control.encode_scene(base.scene_prefix.detach().float())
            if (
                not isinstance(signature, torch.Tensor)
                or signature.ndim != 3
                or signature.shape[0] != 1
                or not torch.isfinite(signature).all()
            ):
                raise RuntimeError("Question control produced an invalid scene signature")
            self._scene_control_signature = signature.detach().clone()
            self._scene_control_signature_hash = prefix_sha256(self._scene_control_signature)

    @classmethod
    def load(
        cls,
        config: dict[str, Any],
        scene_id: str,
        *,
        base_checkpoint: str | Path,
        control_checkpoint: str | Path,
        grounding_checkpoint: str | Path | None = None,
        audit: FileAccessAudit | None = None,
    ) -> QuestionControlledChatRuntime:
        if config.get("_runtime_safe_config") is not True:
            raise ValueError(
                "Question-controlled chat requires a standalone validated runtime config"
            )
        config_path = config.get("_config_path")
        if not isinstance(config_path, str) or not config_path:
            raise ValueError("Validated question-control config is missing its source path")
        supplied_config = validate_runtime_config(config)
        runtime_config = load_runtime_config(
            config_path,
            record_file=(None if audit is None else audit.record),
        )
        if effective_runtime_config_sha256(supplied_config) != (
            effective_runtime_config_sha256(runtime_config)
        ):
            raise ValueError(
                "In-memory question-control config differs from its standalone runtime file"
            )
        training_artifact_root = block_question_control_training_artifacts(
            audit, runtime_config
        )
        candidates: list[tuple[str | Path, str]] = [
            (base_checkpoint, "base checkpoint"),
            (control_checkpoint, "control checkpoint"),
        ]
        if grounding_checkpoint is not None:
            candidates.append((grounding_checkpoint, "grounding checkpoint"))
        for candidate, purpose in candidates:
            try:
                _resolve(candidate).relative_to(training_artifact_root)
            except ValueError:
                continue
            raise ValueError(
                f"Question-controlled {purpose} must be physically separate "
                "from the derived training-artifact root"
            )
        base = StaticChatRuntime.load(
            runtime_config,
            scene_id,
            checkpoint=base_checkpoint,
            audit=audit,
            local_files_only=True,
        )
        control, metadata = _load_control_head(
            control_checkpoint,
            hidden_size=base.language.hidden_size,
            device=base.language.device,
            audit=audit,
        )
        base_checkpoint_sha256, _ = checkpoint_fingerprint(base_checkpoint)
        if metadata.get("base_checkpoint_sha256") != base_checkpoint_sha256:
            raise ValueError(
                "Question-control head was trained against a different base checkpoint"
            )
        runtime_sha256 = effective_runtime_config_sha256(runtime_config)
        if metadata.get("base_runtime_config_sha256") != runtime_sha256:
            raise ValueError(
                "Question-control head was trained against a different runtime configuration"
            )
        grounding_sidecar = None
        if grounding_checkpoint is not None:
            grounding_sidecar = V78GroundingSidecarRuntime.load(
                grounding_checkpoint,
                scene_prefix=base.scene_prefix,
                room_min=base.map_data.room_min,
                room_max=base.map_data.room_max,
                base_checkpoint_sha256=base_checkpoint_sha256,
                base_runtime_config_sha256=runtime_sha256,
                model_id=str(runtime_config["language"]["model_id"]),
                model_revision=str(runtime_config["language"]["revision"]),
                device=base.language.device,
                audit=audit,
            )
        return cls(base, control, metadata, grounding_sidecar=grounding_sidecar)

    def _control_tokens(self, question: str) -> tuple[torch.Tensor | None, dict[str, Any]]:
        ids = question_token_ids(
            self.base.language.tokenizer,
            question,
            self.base.language.device,
        )
        maximum = int(self.base.config["language"]["max_question_tokens"])
        if ids.shape[1] > maximum:
            raise ValueError(f"Question exceeds the configured {maximum}-token control limit")
        embedding_layer = self.base.language.model.get_input_embeddings()
        with torch.inference_mode():
            question_embeddings = embedding_layer(ids).float()
            if isinstance(self.control, DenseFullSceneContinuousControlV74):
                if (
                    self._scene_control_key is None
                    or self._scene_control_value is None
                ):
                    raise RuntimeError(
                        "Dense-reader scene K/V was not cached before questions"
                    )
                output = self.control.forward_encoded(
                    self._scene_control_key,
                    self._scene_control_value,
                    question_embeddings,
                )
                dense_audit = self.control.audit()
                control = output.control_tokens
                audit = {
                    "architecture": self.control_metadata["architecture"],
                    "scene_token_count": dense_audit.environment_latents + 2,
                    "environment_latent_count": dense_audit.environment_latents,
                    "control_token_count": dense_audit.query_count,
                    "every_scene_token_influenced_output": (
                        dense_audit.all_latents_receive_positive_weight
                    ),
                    "minimum_attention_weight": (
                        dense_audit.minimum_attention_weight
                    ),
                    "positive_attention_floor": True,
                    "softmax_scene_attention_used": True,
                    "bilinear_question_scene_value_interaction": (
                        dense_audit.bilinear_question_scene_value_interaction
                    ),
                    "question_only_output_path_exists": (
                        dense_audit.question_only_output_path_exists
                    ),
                    "question_dependent_scene_retrieval": (
                        dense_audit.question_dependent_retrieval
                    ),
                    "latent_selection_or_top_k_used": False,
                    "immutable_full_prefix_retained_separately": (
                        dense_audit.immutable_full_prefix_retained_separately
                    ),
                    "prequestion_scene_key_value_cache": True,
                    "zero_scene_produces_exact_zero_controls": (
                        dense_audit.zero_scene_produces_exact_zero_controls
                    ),
                    "control_used": True,
                    "maximum_control_rms": float(
                        output.control_rms.detach().max().cpu()
                    ),
                    "saved_runtime_training_gate_required": True,
                    "training_answers_runtime_loaded": False,
                    "answer_text_runtime_loaded": False,
                    "answer_class_codebook_runtime_loaded": False,
                }
                if isinstance(self.control, DenseFullSceneContinuousControlV75):
                    audit.update(
                        {
                            "coefficient_decoder_hidden_dimension": (
                                self.control.coefficient_decoder_hidden_dimension
                            ),
                            "bias_free_nonlinear_coefficient_decoder": True,
                            "zero_preserving_coefficient_activation": True,
                        }
                    )
                self.last_control_audit = audit
                self.last_control_tokens_sha256 = prefix_sha256(control.detach())
                self.last_environment_conditioned_input_sha256 = prefix_sha256(
                    torch.cat(
                        (
                            self.base.scene_prefix.detach(),
                            control.detach().to(self.base.scene_prefix),
                        ),
                        dim=1,
                    )
                )
                return control, audit
            if isinstance(
                self.control,
                _SIGNATURE_CONTROL_TYPES,
            ):
                if self._scene_control_signature is None:
                    raise RuntimeError("Global scene signature was not cached before questions")
                output = self.control.forward_from_signature(
                    self._scene_control_signature, question_embeddings
                )
                bounded_audit = self.control.audit()
                control = output.control_tokens if bounded_audit.control_used else None
                audit = {
                    "architecture": self.control_metadata["architecture"],
                    "scene_token_count": bounded_audit.scene_token_count,
                    "environment_latent_count": bounded_audit.environment_latent_count,
                    "control_token_count": bounded_audit.control_token_count,
                    "scene_moment_count": bounded_audit.scene_moment_count,
                    "every_scene_token_influenced_output": (
                        bounded_audit.every_environment_latent_influenced_signature
                    ),
                    "question_dependent_scene_retrieval": (
                        bounded_audit.question_dependent_scene_retrieval
                    ),
                    "softmax_scene_attention_used": (bounded_audit.softmax_scene_attention_used),
                    "control_values_scene_question_bilinear": getattr(
                        bounded_audit,
                        "control_values_scene_question_bilinear",
                        True,
                    ),
                    "gate_scene_question_conditioned": getattr(
                        bounded_audit,
                        "gate_scene_question_conditioned",
                        False,
                    ),
                    "inherited_v60_state_frozen": getattr(
                        bounded_audit,
                        "inherited_v60_state_frozen",
                        False,
                    ),
                    "separate_question_scene_route_projections": getattr(
                        bounded_audit,
                        "separate_question_scene_route_projections",
                        False,
                    ),
                    "normalized_route_factors": getattr(
                        bounded_audit,
                        "normalized_route_factors",
                        False,
                    ),
                    "all_scene_moments_consumed_by_route": getattr(
                        bounded_audit,
                        "all_scene_moments_consumed_by_route",
                        False,
                    ),
                    "low_rank_bilinear_route": getattr(
                        bounded_audit,
                        "low_rank_bilinear_route",
                        False,
                    ),
                    "route_uses_inherited_value_trunk": getattr(
                        bounded_audit,
                        "route_uses_inherited_value_trunk",
                        True,
                    ),
                    "route_factor_rank": getattr(
                        bounded_audit,
                        "route_factor_rank",
                        None,
                    ),
                    "gate_probability": bounded_audit.gate_probability,
                    "control_used": bounded_audit.control_used,
                    "maximum_control_rms": bounded_audit.maximum_control_rms,
                    "exact_no_control_route": not bounded_audit.control_used,
                    "activation_rms": getattr(
                        bounded_audit,
                        "activation_rms",
                        None,
                    ),
                    "activation_rms_threshold": getattr(
                        bounded_audit,
                        "activation_rms_threshold",
                        None,
                    ),
                    "exact_no_control_below_threshold": getattr(
                        bounded_audit,
                        "exact_no_control_below_threshold",
                        False,
                    ),
                    "always_on_continuous_control": getattr(
                        bounded_audit,
                        "always_on_continuous_control",
                        False,
                    ),
                    "legacy_route_parameters_ignored": getattr(
                        bounded_audit,
                        "legacy_route_parameters_ignored",
                        False,
                    ),
                    "saved_runtime_training_gate_required": (
                        self.control_metadata.get(
                            "saved_runtime_training_gate_required",
                            False,
                        )
                    ),
                }
                self.last_control_audit = audit
                self.last_control_tokens_sha256 = (
                    None if control is None else prefix_sha256(control.detach())
                )
                self.last_environment_conditioned_input_sha256 = (
                    self.scene_prefix_hash
                    if control is None
                    else prefix_sha256(
                        torch.cat(
                            (
                                self.base.scene_prefix.detach(),
                                control.detach().to(self.base.scene_prefix),
                            ),
                            dim=1,
                        )
                    )
                )
                return control, audit
            control = self.control(self.base.scene_prefix.float(), question_embeddings)
        audit = self.control.audit()
        result = {
            "architecture": "full_scene_question_control_v1",
            "scene_token_count": audit.scene_token_count,
            "control_token_count": audit.control_token_count,
            "minimum_attention_weight": audit.minimum_attention_weight,
            "maximum_attention_weight": audit.maximum_attention_weight,
            "every_scene_token_influenced_output": (audit.every_scene_token_influenced_output),
        }
        self.last_control_audit = result
        self.last_control_tokens_sha256 = prefix_sha256(control.detach())
        self.last_environment_conditioned_input_sha256 = prefix_sha256(
            torch.cat(
                (
                    self.base.scene_prefix.detach(),
                    control.detach().to(self.base.scene_prefix),
                ),
                dim=1,
            )
        )
        return control, result

    @torch.inference_mode()
    def answer(self, question: str) -> ChatAnswer:
        question = question.strip()
        if not question:
            raise ValueError("Question must not be empty")
        self.assert_prefix_unchanged()
        started = time.perf_counter()
        prompt_ids = prompt_token_ids(
            self.base.language.tokenizer,
            str(self.base.config["language"]["system_prompt"]),
            question,
            self.base.language.device,
        )
        control, audit = self._control_tokens(question)
        if audit["every_scene_token_influenced_output"] is not True:
            raise RuntimeError("Question control omitted part of the scene prefix")
        backend = self.base.language.prefix_backend
        if backend is None:
            raise RuntimeError("Question-controlled Gemma runtime requires a prefix backend")
        prepared = backend.prepare(
            self.base.scene_prefix,
            prompt_ids,
            scene_prefix_after_bos=scene_prefix_after_bos_setting(self.base.config),
            scene_boundary_mode=scene_boundary_mode_setting(self.base.config),
            control_tokens=(None if control is None else control.to(self.base.scene_prefix)),
        )
        generated = backend.generate(
            prepared,
            max_new_tokens=int(self.base.config["language"]["max_answer_tokens"]),
            eos_token_ids=self.base._eos_token_ids(),
        )
        decoded = self.base.language.tokenizer.decode(
            generated[0].detach().cpu().tolist(), skip_special_tokens=True
        ).strip()
        answer = sanitize_generated_answer(decoded)
        grounding_ids = question_token_ids(
            self.base.language.tokenizer,
            question,
            self.base.language.device,
        )
        grounding_embeddings = self.base.language.model.get_input_embeddings()(grounding_ids)
        if self.grounding_sidecar is None:
            grounding_xyz, confidence, support_distance = self.base._predict_grounding(
                grounding_embeddings
            )
            self.last_grounding_audit = None
        else:
            if self._grounding_scene_prefix is None:
                raise RuntimeError("V78 grounding lost its bound environmental prefix")
            sidecar_output = self.grounding_sidecar.predict(
                grounding_embeddings,
                scene_prefix=self._grounding_scene_prefix,
                map_xyz=self.base.map_data.xyz,
                map_confidence=self.base.map_data.confidence,
            )
            grounding_xyz = sidecar_output.xyz_m
            confidence = sidecar_output.confidence
            support_distance = sidecar_output.support_distance_m
            self.last_grounding_audit = sidecar_output.audit
        self.assert_prefix_unchanged()
        self._questions_answered += 1
        return ChatAnswer(
            question=question,
            answer=answer,
            grounding_xyz_m=grounding_xyz,
            grounding_confidence=confidence,
            grounding_support_distance_m=support_distance,
            prefix_hash=self.scene_prefix_hash,
            generated_tokens=int(generated.shape[-1]),
            elapsed_seconds=time.perf_counter() - started,
        )

    def assert_prefix_unchanged(self) -> None:
        self.base.assert_prefix_unchanged()
        if self.base.scene_prefix_hash != self.scene_prefix_hash:
            raise RuntimeError("Question-controlled scene-prefix identity changed")
        if self.grounding_sidecar is not None:
            grounding_prefix = self._grounding_scene_prefix
            if grounding_prefix is None:
                raise RuntimeError("V78 grounding lost its bound environmental prefix")
            self.grounding_sidecar.assert_prefix_unchanged(grounding_prefix)
            active_prefix = self.base.scene_prefix
            if active_prefix.shape == grounding_prefix.shape:
                if prefix_sha256(active_prefix) != prefix_sha256(grounding_prefix):
                    raise RuntimeError(
                        "V78 grounding environmental prefix differs from generation prefix"
                    )
            else:
                # The embodied runtime may insert separately checkpointed numeric
                # robot-state tokens immediately before the unchanged scene-end
                # boundary.  The environmental prefix used by V78 must remain an
                # exact ordered subsequence; no scene latent may be replaced.
                environment_tokens = grounding_prefix.shape[1]
                if (
                    active_prefix.ndim != 3
                    or active_prefix.shape[0] != grounding_prefix.shape[0]
                    or active_prefix.shape[2] != grounding_prefix.shape[2]
                    or active_prefix.shape[1] <= environment_tokens
                    or active_prefix.shape[1] > environment_tokens + 16
                    or not torch.equal(
                        active_prefix[:, : environment_tokens - 1],
                        grounding_prefix[:, :-1].to(active_prefix),
                    )
                    or not torch.equal(
                        active_prefix[:, -1:],
                        grounding_prefix[:, -1:].to(active_prefix),
                    )
                    or not torch.isfinite(active_prefix.float()).all()
                ):
                    raise RuntimeError(
                        "V78 grounding environmental prefix is not preserved in the "
                        "numeric embodied prefix"
                    )
        if (
            self._scene_control_key is not None
            or self._scene_control_value is not None
        ):
            if (
                self._scene_control_key is None
                or self._scene_control_value is None
            ):
                raise RuntimeError("V74 cached scene K/V inventory changed")
            observed = prefix_sha256(
                torch.cat(
                    (self._scene_control_key, self._scene_control_value), dim=-1
                )
            )
            if observed != self._scene_control_signature_hash:
                raise RuntimeError(
                    "Question-controlled cached scene K/V changed unexpectedly"
                )
        if self._scene_control_signature is not None:
            observed = prefix_sha256(self._scene_control_signature)
            if observed != self._scene_control_signature_hash:
                raise RuntimeError(
                    "Question-controlled cached scene signature changed unexpectedly"
                )

    @property
    def questions_answered(self) -> int:
        return self._questions_answered

    def current_prefix_hash(self) -> str:
        self.assert_prefix_unchanged()
        return self.scene_prefix_hash

    @property
    def scene_control_signature_hash(self) -> str | None:
        return self._scene_control_signature_hash

    def grounding_sidecar_startup_audit(self) -> dict[str, Any] | None:
        if self.grounding_sidecar is None:
            return None
        return self.grounding_sidecar.startup_audit()

    def startup_summary(self) -> dict[str, Any]:
        summary = dict(self.base.startup_summary())
        summary.update(
            {
                "runtime_kind": "continuous_scene_question_control",
                "scene_prefix_hash": self.scene_prefix_hash,
                "scene_control_signature_sha256": (self._scene_control_signature_hash),
                "scene_prefix_computed_before_question": True,
                "questions_answered": self._questions_answered,
                "control_architecture": self.control_metadata.get("architecture"),
                "control_schema_version": self.control_metadata.get("schema_version"),
                "environmental_text_inputs": [],
                "question_dependent_scene_retrieval": False,
                "question_conditioned_scene_readout_tokens": True,
                "prequestion_scene_key_value_cache": isinstance(
                    self.control, DenseFullSceneContinuousControlV74
                ),
                "answer_class_codebook_runtime_loaded": False,
                "answer_text_runtime_loaded": False,
                "strict_fixed_environment_embedding_input": False,
                "optional_v78_grounding": self.grounding_sidecar_startup_audit(),
            }
        )
        return summary


__all__ = [
    "QuestionControlledChatRuntime",
    "block_question_control_training_artifacts",
    "question_control_training_artifact_root",
]
