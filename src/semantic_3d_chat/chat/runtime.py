"""Oracle-isolated static chat runtime for a continuous 3D scene prefix."""

from __future__ import annotations

import json
import math
import re
import time
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch

from semantic_3d_chat.chat.file_audit import FileAccessAudit
from semantic_3d_chat.config import PROJECT_ROOT, config_hash, project_path
from semantic_3d_chat.language.generation import generate_from_embeddings
from semantic_3d_chat.language.local_lm import (
    LocalLanguageModel,
    load_local_language_model,
    prompt_token_ids,
    question_token_ids,
)
from semantic_3d_chat.language.lora import (
    install_lora_adapters,
    lora_checkpoint_contract,
    lora_checkpoint_contract_mismatch,
    lora_optimizer_settings,
    lora_settings,
    validate_lora_checkpoint_state,
)
from semantic_3d_chat.language.prefix_injection import (
    ContinuousPrefixComposer,
    native_gemma4_image_contract_setting,
    prefix_sha256,
    scene_boundary_contract_mismatch,
    scene_boundary_mode_setting,
    scene_prefix_after_bos_contract_mismatch,
    scene_prefix_after_bos_setting,
)
from semantic_3d_chat.scene_encoder.map_io import MapTensorData, load_map_tensors
from semantic_3d_chat.scene_encoder.projector import SceneTokenizer, SceneTokenizerOutput
from semantic_3d_chat.training.checkpointing import load_adapter_checkpoint
from semantic_3d_chat.training.losses import QuestionGroundingHead, denormalize_xyz

GenerationFunction = Callable[
    [torch.nn.Module, torch.Tensor, torch.Tensor, int, int | Sequence[int] | None],
    torch.Tensor,
]

_OPAQUE_SCENE_ID = re.compile(r"scene_[0-9]{6}")
_FORBIDDEN_DATA_DIRECTORIES = frozenset({"oracle", "qa", "rendered", "features"})
_SUPPORTED_CHECKPOINT_SCHEMAS = frozenset({1, 2, 3})
_SCENE_TOKENIZER_CONTRACT_DEFAULTS: dict[str, int | float | None] = {
    "language_aligned_tail_dim": 0,
    "native_aligned_coverage_scale": 0.0,
    "learned_scene_token_scale": 1.0,
    "learned_scene_token_rms_target": None,
}


@dataclass(frozen=True)
class ChatAnswer:
    question: str
    answer: str
    grounding_xyz_m: tuple[float, float, float]
    grounding_confidence: float
    grounding_support_distance_m: float
    prefix_hash: str
    generated_tokens: int
    elapsed_seconds: float

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["grounding_xyz_m"] = list(self.grounding_xyz_m)
        return payload


def _guard_runtime_input(path: str | Path, purpose: str) -> Path:
    resolved = Path(path).expanduser().resolve()
    lowered_parts = {part.casefold() for part in resolved.parts}
    forbidden = sorted(lowered_parts & _FORBIDDEN_DATA_DIRECTORIES)
    if forbidden:
        raise ValueError(f"Refusing to load {purpose} from forbidden runtime path: {resolved}")
    return resolved


def _read_checkpoint_metadata(checkpoint: Path, audit: FileAccessAudit | None) -> dict[str, Any]:
    metadata_path = _guard_runtime_input(checkpoint / "metadata.json", "checkpoint metadata")
    if audit is not None:
        audit.record(metadata_path)
    with metadata_path.open("r", encoding="utf-8") as handle:
        metadata = json.load(handle)
    if not isinstance(metadata, dict):
        raise TypeError("Checkpoint metadata must be a JSON object")
    return metadata


def _equal_number(left: Any, right: Any, tolerance: float = 1e-9) -> bool:
    if left is None or right is None:
        return left is right
    try:
        return math.isclose(float(left), float(right), rel_tol=0.0, abs_tol=tolerance)
    except (TypeError, ValueError):
        return False


def _scene_tokenizer_contract(config: dict[str, Any]) -> dict[str, int | float | None]:
    settings = config["scene_encoder"]
    rms_target = settings.get("learned_scene_token_rms_target")
    return {
        "language_aligned_tail_dim": int(settings.get("language_aligned_tail_dim", 0)),
        "native_aligned_coverage_scale": float(settings.get("native_aligned_coverage_scale", 0.0)),
        "learned_scene_token_scale": float(settings.get("learned_scene_token_scale", 1.0)),
        "learned_scene_token_rms_target": (None if rms_target is None else float(rms_target)),
    }


def validate_checkpoint_contract(
    metadata: dict[str, Any],
    config: dict[str, Any],
    *,
    semantic_dim: int,
    language_hidden_dim: int,
    lora_parameter_count: int = 0,
) -> list[str]:
    """Enforce adapter-shape compatibility while surfacing unrelated config drift."""

    required = {
        "schema_version",
        "semantic_dim",
        "language_hidden_dim",
        "language_model_id",
        "language_revision",
        "scene_latents",
        "scene_model_dim",
        "input_voxel_size_m",
        "config_hash",
    }
    scene_tokenizer_contract = _scene_tokenizer_contract(config)
    uses_aligned_bypass = any(
        not _equal_number(value, _SCENE_TOKENIZER_CONTRACT_DEFAULTS[key])
        for key, value in scene_tokenizer_contract.items()
    )
    metadata_has_scene_tokenizer_contract = any(key in metadata for key in scene_tokenizer_contract)
    if uses_aligned_bypass or metadata_has_scene_tokenizer_contract:
        required.update(scene_tokenizer_contract)
    missing = sorted(required - metadata.keys())
    if missing:
        raise ValueError(f"Checkpoint metadata is missing required fields: {missing}")
    schema_version = metadata.get("schema_version")
    if schema_version not in _SUPPORTED_CHECKPOINT_SCHEMAS:
        raise ValueError(
            "Unsupported checkpoint metadata schema: "
            f"{schema_version}; supported={sorted(_SUPPORTED_CHECKPOINT_SCHEMAS)}"
        )
    expected = {
        "semantic_dim": int(semantic_dim),
        "language_hidden_dim": int(language_hidden_dim),
        "language_model_id": str(config["language"]["model_id"]),
        "language_revision": str(config["language"]["revision"]),
        "scene_latents": int(config["scene_encoder"]["global_latents"]),
        "scene_model_dim": int(config["scene_encoder"]["model_dim"]),
    }
    architecture_version = config["scene_encoder"].get("architecture_version")
    if architecture_version is not None:
        expected["scene_encoder_architecture_version"] = str(architecture_version)
    mismatches = {
        key: {"checkpoint": metadata.get(key), "runtime": value}
        for key, value in expected.items()
        if metadata.get(key) != value
    }
    configured_voxel_size = config["scene_encoder"].get("input_voxel_size_m")
    if not _equal_number(metadata.get("input_voxel_size_m"), configured_voxel_size):
        mismatches["input_voxel_size_m"] = {
            "checkpoint": metadata.get("input_voxel_size_m"),
            "runtime": configured_voxel_size,
        }
    for key, value in scene_tokenizer_contract.items():
        if key in metadata and not _equal_number(metadata[key], value):
            mismatches[key] = {"checkpoint": metadata[key], "runtime": value}
    prefix_layout_mismatch = scene_prefix_after_bos_contract_mismatch(
        metadata,
        scene_prefix_after_bos_setting(config),
    )
    if prefix_layout_mismatch is not None:
        mismatches["scene_prefix_after_bos"] = prefix_layout_mismatch
    boundary_mismatch = scene_boundary_contract_mismatch(
        metadata,
        scene_boundary_mode_setting(config),
        native_gemma4_image_contract_setting(config),
    )
    if boundary_mismatch is not None:
        mismatches["scene_boundary_mode"] = boundary_mismatch
    configured_lora = lora_settings(config)
    configured_lora_optimizer = lora_optimizer_settings(config, configured_lora)
    configured_lora_contract = lora_checkpoint_contract(
        configured_lora,
        configured_lora_optimizer,
        lora_parameter_count,
    )
    lora_mismatch = lora_checkpoint_contract_mismatch(metadata, configured_lora_contract)
    if lora_mismatch is not None:
        mismatches["lora"] = lora_mismatch
    if mismatches:
        raise ValueError(f"Checkpoint is incompatible with runtime architecture: {mismatches}")
    warnings: list[str] = []
    runtime_hash = config_hash(config)
    if metadata["config_hash"] != runtime_hash:
        warnings.append(
            "Full config hash differs from training, but all inference architecture fields match: "
            f"checkpoint={metadata['config_hash']} runtime={runtime_hash}"
        )
    return warnings


def construct_scene_tokenizer(
    config: dict[str, Any], semantic_dim: int, language_hidden_dim: int
) -> SceneTokenizer:
    settings = config["scene_encoder"]
    return SceneTokenizer(
        semantic_dim=semantic_dim,
        model_dim=int(settings["model_dim"]),
        language_hidden_dim=language_hidden_dim,
        block_size_m=float(settings["block_size_m"]),
        tokens_per_block=int(settings["tokens_per_block"]),
        global_latents=int(settings["global_latents"]),
        heads=int(settings["heads"]),
        global_layers=int(settings["global_layers"]),
        fourier_bands=int(settings["fourier_bands"]),
        coverage_temperature=float(settings.get("coverage_temperature", 0.35)),
        coverage_scale=float(settings.get("coverage_scale", 1.0)),
        query_identity_scale=float(settings.get("query_identity_scale", 0.5)),
        projection_skip_scale=float(settings.get("projection_skip_scale", 1.0)),
        semantic_skip_scale=float(settings.get("semantic_skip_scale", 1.0)),
        geometry_skip_scale=float(settings.get("geometry_skip_scale", 0.5)),
        block_content_residual_scale=float(settings.get("block_content_residual_scale", 1.0)),
        language_aligned_tail_dim=int(settings.get("language_aligned_tail_dim", 0)),
        native_aligned_coverage_scale=float(settings.get("native_aligned_coverage_scale", 0.0)),
        learned_scene_token_scale=float(settings.get("learned_scene_token_scale", 1.0)),
        learned_scene_token_rms_target=(
            None
            if settings.get("learned_scene_token_rms_target") is None
            else float(settings["learned_scene_token_rms_target"])
        ),
        architecture_version=str(settings["architecture_version"]),
    )


class StaticChatRuntime:
    """Question-independent scene memory with deterministic local generation."""

    def __init__(
        self,
        *,
        config: dict[str, Any],
        scene_id: str,
        checkpoint_path: Path,
        checkpoint_metadata: dict[str, Any],
        language: LocalLanguageModel,
        map_data: MapTensorData,
        scene_model: SceneTokenizer,
        composer: ContinuousPrefixComposer,
        grounding: QuestionGroundingHead,
        warnings: list[str] | None = None,
        generation_function: GenerationFunction = generate_from_embeddings,
    ) -> None:
        if not _OPAQUE_SCENE_ID.fullmatch(scene_id):
            raise ValueError("scene_id must be opaque and match scene_ followed by six digits")
        self.config = config
        self.scene_id = scene_id
        self.checkpoint_path = checkpoint_path
        self.checkpoint_metadata = dict(checkpoint_metadata)
        self.language = language
        self.map_data = map_data
        self.scene_model = scene_model.eval()
        self.composer = composer.eval()
        configured_layout = scene_prefix_after_bos_setting(config)
        if self.composer.scene_prefix_after_bos != configured_layout:
            raise ValueError(
                "Composer scene-prefix layout does not match runtime config: "
                f"{self.composer.scene_prefix_after_bos} != {configured_layout}"
            )
        configured_boundary_mode = scene_boundary_mode_setting(config)
        if self.composer.scene_boundary_mode != configured_boundary_mode:
            raise ValueError(
                "Composer scene-boundary mode does not match runtime config: "
                f"{self.composer.scene_boundary_mode} != {configured_boundary_mode}"
            )
        native_embeddings = self.language.scene_boundary_embeddings(configured_boundary_mode)
        if native_embeddings is not None:
            self.composer.validate_native_boundary_embeddings(native_embeddings)
        self.grounding = grounding.eval()
        self.warnings = list(warnings or [])
        self._generation_function = generation_function
        self._questions_answered = 0

        started = time.perf_counter()
        with torch.inference_mode():
            self.scene_output = self._encode_complete_scene()
            model_dtype = next(self.language.model.parameters()).dtype
            lm_scene_tokens = self.scene_output.scene_tokens.to(dtype=model_dtype)
            self.scene_prefix = self.composer.scene_prefix(lm_scene_tokens).detach()
        if self.scene_prefix.shape[1] != int(config["scene_encoder"]["global_latents"]) + 2:
            raise RuntimeError("Scene prefix length does not match configured global latent count")
        if not torch.isfinite(self.scene_prefix).all():
            raise RuntimeError("Scene prefix contains NaN or infinity")
        self.scene_prefix_hash = prefix_sha256(self.scene_prefix)
        self.prefix_build_seconds = time.perf_counter() - started

    @classmethod
    def load(
        cls,
        config: dict[str, Any],
        scene_id: str,
        checkpoint: str | Path = "data/checkpoints/best",
        *,
        audit: FileAccessAudit | None = None,
        local_files_only: bool = True,
        generation_function: GenerationFunction = generate_from_embeddings,
    ) -> StaticChatRuntime:
        if not _OPAQUE_SCENE_ID.fullmatch(scene_id):
            raise ValueError("scene_id must be opaque and match scene_ followed by six digits")
        checkpoint_path = Path(checkpoint)
        if not checkpoint_path.is_absolute():
            checkpoint_path = PROJECT_ROOT / checkpoint_path
        checkpoint_path = _guard_runtime_input(checkpoint_path, "adapter checkpoint")
        metadata = _read_checkpoint_metadata(checkpoint_path, audit)
        adapter_path = _guard_runtime_input(
            checkpoint_path / "adapter.safetensors", "adapter parameters"
        )
        if audit is not None:
            audit.record(adapter_path)

        map_path = _guard_runtime_input(
            project_path(config, "maps", scene_id, "voxel_map.npz"), "numeric voxel map"
        )
        if audit is not None:
            audit.record(map_path)
        map_data = load_map_tensors(
            map_path,
            config["scene"]["room_size_m"],
            device="cpu",
            input_voxel_size_m=config["scene_encoder"].get("input_voxel_size_m"),
        )
        if int(metadata.get("semantic_dim", -1)) != map_data.feature_dim:
            raise ValueError(
                "Checkpoint semantic dimension does not match the numeric voxel map: "
                f"{metadata.get('semantic_dim')} != {map_data.feature_dim}"
            )

        language = load_local_language_model(
            str(config["language"]["model_id"]),
            str(config["language"]["revision"]),
            str(config["language"]["dtype"]),
            freeze=True,
            local_files_only=local_files_only,
            backend=str(config["language"].get("backend", "auto")),
        )
        configured_boundary_mode = scene_boundary_mode_setting(config)
        configured_native_contract = native_gemma4_image_contract_setting(config)
        loaded_native_contract = language.scene_boundary_contract(configured_boundary_mode)
        if loaded_native_contract != configured_native_contract:
            raise ValueError(
                "Loaded language model does not satisfy configured scene-boundary contract: "
                f"loaded={loaded_native_contract} configured={configured_native_contract}"
            )
        configured_lora = lora_settings(config)
        lora_installation = install_lora_adapters(language.model, configured_lora)
        checkpoint_backend = metadata.get("language_backend")
        if checkpoint_backend is not None and checkpoint_backend != language.backend_name:
            raise ValueError(
                "Checkpoint language backend does not match the loaded runtime backend: "
                f"{checkpoint_backend} != {language.backend_name}"
            )
        warnings = validate_checkpoint_contract(
            metadata,
            config,
            semantic_dim=map_data.feature_dim,
            language_hidden_dim=language.hidden_size,
            lora_parameter_count=(
                0 if lora_installation is None else lora_installation.parameter_count
            ),
        )
        scene_model = construct_scene_tokenizer(config, map_data.feature_dim, language.hidden_size)
        composer = ContinuousPrefixComposer(
            language.hidden_size,
            scene_prefix_after_bos=scene_prefix_after_bos_setting(config),
            bos_token_id=language.bos_token_id,
            scene_boundary_mode=configured_boundary_mode,
            native_boundary_embeddings=language.scene_boundary_embeddings(configured_boundary_mode),
        )
        grounding = QuestionGroundingHead(
            int(config["scene_encoder"]["model_dim"]),
            language.hidden_size,
            int(config["scene_encoder"]["global_latents"]),
            int(config["scene_encoder"]["model_dim"]),
        )
        checkpoint_modules = {
            "scene_model": scene_model,
            "composer": composer,
            "grounding": grounding,
        }
        if lora_installation is not None:
            checkpoint_modules["lora"] = lora_installation.state_module
        loaded_metadata = load_adapter_checkpoint(
            checkpoint_path,
            checkpoint_modules,
            device="cpu",
        )
        if loaded_metadata != metadata:
            raise RuntimeError("Checkpoint metadata changed while the runtime was loading")
        if lora_installation is not None:
            validate_lora_checkpoint_state(metadata, lora_installation)
            language.model.requires_grad_(False)
        device = language.device
        scene_model = scene_model.to(device)
        composer = composer.to(device)
        grounding = grounding.to(device)
        map_data = map_data.to(device)
        return cls(
            config=config,
            scene_id=scene_id,
            checkpoint_path=checkpoint_path,
            checkpoint_metadata=metadata,
            language=language,
            map_data=map_data,
            scene_model=scene_model,
            composer=composer,
            grounding=grounding,
            warnings=warnings,
            generation_function=generation_function,
        )

    def _encode_complete_scene(self) -> SceneTokenizerOutput:
        data = self.map_data
        output = self.scene_model(
            data.semantic,
            data.xyz,
            data.rgb,
            data.normal,
            data.confidence,
            data.observation_count,
            data.room_min,
            data.room_max,
        )
        processed = int(output.audit["processed_voxels"].detach().cpu().item())
        if processed != data.voxel_count:
            raise RuntimeError(
                f"Incomplete full-scene encoding: processed {processed}/{data.voxel_count} voxels"
            )
        return output

    @property
    def questions_answered(self) -> int:
        return self._questions_answered

    def current_prefix_hash(self) -> str:
        return prefix_sha256(self.scene_prefix)

    def assert_prefix_unchanged(self) -> None:
        current = self.current_prefix_hash()
        if current != self.scene_prefix_hash:
            raise RuntimeError(
                "Question-independent scene prefix changed unexpectedly: "
                f"{self.scene_prefix_hash} != {current}"
            )

    def startup_summary(self) -> dict[str, Any]:
        summary = {
            "phase": "scene_ready",
            "scene_id": self.scene_id,
            "prefix_hash": self.scene_prefix_hash,
            "prefix_shape": list(self.scene_prefix.shape),
            "scene_latents": int(self.scene_output.scene_tokens.shape[1]),
            "language_hidden_dim": int(self.scene_prefix.shape[-1]),
            "language_backend": self.language.backend_name,
            "scene_prefix_after_bos": scene_prefix_after_bos_setting(self.config),
            "scene_boundary_mode": scene_boundary_mode_setting(self.config),
            "gemma4_native_image_contract": native_gemma4_image_contract_setting(self.config),
            "source_voxels": self.map_data.source_voxel_count,
            "processed_voxels": self.map_data.voxel_count,
            "occupied_blocks": int(self.scene_output.audit["voxel_counts"].numel()),
            "device": str(self.language.device),
            "prefix_build_seconds": self.prefix_build_seconds,
            "checkpoint": str(self.checkpoint_path),
            "warnings": self.warnings,
        }
        if "lora" in self.checkpoint_metadata:
            summary["lora"] = self.checkpoint_metadata["lora"]
        return summary

    def _question_token_count(self, question: str) -> int:
        encoded = self.language.tokenizer(
            question,
            add_special_tokens=False,
            return_tensors="pt",
        )
        input_ids = encoded["input_ids"] if isinstance(encoded, dict) else encoded.input_ids
        return int(input_ids.shape[-1])

    def _eos_token_ids(self) -> int | list[int] | None:
        values: list[int] = []
        tokenizer_eos = getattr(self.language.tokenizer, "eos_token_id", None)
        model_eos = getattr(
            getattr(self.language.model, "generation_config", None), "eos_token_id", None
        )
        for candidate in (tokenizer_eos, model_eos):
            if candidate is None:
                continue
            if isinstance(candidate, Sequence) and not isinstance(candidate, (str, bytes)):
                values.extend(int(value) for value in candidate)
            else:
                values.append(int(candidate))
        unique = sorted(set(values))
        if not unique:
            return None
        return unique[0] if len(unique) == 1 else unique

    @torch.inference_mode()
    def _predict_grounding(
        self, prompt_embeddings: torch.Tensor
    ) -> tuple[tuple[float, float, float], float, float]:
        normalized = self.grounding(
            self.scene_output.native_latents.float(), prompt_embeddings.float().mean(dim=1)
        )
        xyz = denormalize_xyz(normalized, self.map_data.room_min, self.map_data.room_max)[0]
        distances = torch.linalg.vector_norm(self.map_data.xyz - xyz, dim=-1)
        nearest_index = int(torch.argmin(distances).item())
        support_distance = float(distances[nearest_index].detach().cpu())
        map_confidence = float(
            self.map_data.confidence[nearest_index].detach().float().cpu().clamp(0.0, 1.0)
        )
        distance_scale = max(float(self.config["scene_encoder"]["block_size_m"]), 1e-6)
        confidence = max(
            0.0,
            min(1.0, map_confidence * math.exp(-support_distance / distance_scale)),
        )
        coordinates = tuple(float(value) for value in xyz.detach().float().cpu().tolist())
        return coordinates, confidence, support_distance

    def answer(self, question: str) -> ChatAnswer:
        question = question.strip()
        if not question:
            raise ValueError("Question must not be empty")
        max_question_tokens = int(self.config["language"]["max_question_tokens"])
        question_tokens = self._question_token_count(question)
        if question_tokens > max_question_tokens:
            raise ValueError(
                f"Question has {question_tokens} tokens; maximum is {max_question_tokens}"
            )
        self.assert_prefix_unchanged()
        started = time.perf_counter()
        prompt_ids = prompt_token_ids(
            self.language.tokenizer,
            str(self.config["language"]["system_prompt"]),
            question,
            self.language.device,
        )
        embedding_layer = self.language.model.get_input_embeddings()
        with torch.inference_mode():
            grounding_question_ids = question_token_ids(
                self.language.tokenizer, question, self.language.device
            )
            grounding_question_embeddings = embedding_layer(grounding_question_ids)
            generated = self.language.generate_from_scene_prefix(
                self.scene_prefix,
                prompt_ids,
                max_new_tokens=int(self.config["language"]["max_answer_tokens"]),
                eos_token_ids=self._eos_token_ids(),
                scene_prefix_after_bos=scene_prefix_after_bos_setting(self.config),
                scene_boundary_mode=scene_boundary_mode_setting(self.config),
                fallback=self._generation_function,
            )
            grounding_xyz, grounding_confidence, support_distance = self._predict_grounding(
                grounding_question_embeddings
            )
        decoded = self.language.tokenizer.decode(
            generated[0].detach().cpu().tolist(), skip_special_tokens=True
        ).strip()
        self.assert_prefix_unchanged()
        self._questions_answered += 1
        return ChatAnswer(
            question=question,
            answer=decoded or "unknown",
            grounding_xyz_m=grounding_xyz,
            grounding_confidence=grounding_confidence,
            grounding_support_distance_m=support_distance,
            prefix_hash=self.scene_prefix_hash,
            generated_tokens=int(generated.shape[-1]),
            elapsed_seconds=time.perf_counter() - started,
        )


__all__ = [
    "ChatAnswer",
    "StaticChatRuntime",
    "construct_scene_tokenizer",
    "validate_checkpoint_contract",
]
