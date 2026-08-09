"""Full-image dense patch extraction and sanitized-manifest CLI.

Each configured backend sends one complete 224x224 image through its vision
tower exactly once. CLIP retains middle/late patch states plus its aligned
projection. Gemma 4 retains pre-pool middle/late grids plus the native
post-pool vision-to-language projection broadcast to exact owning cells. No
crops or global-only image vectors are used by the primary feature artifact.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import re
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from semantic_3d_chat.config import PROJECT_ROOT, load_config, project_path
from semantic_3d_chat.device import safe_dtype, select_device
from semantic_3d_chat.rendering_io import FrameRecord, iter_frames
from semantic_3d_chat.vision.model_registry import DenseVisionModelSpec, get_model_spec
from semantic_3d_chat.vision.patch_features import DensePatchFeatures, extract_clip_streams

LOGGER = logging.getLogger(__name__)


class DenseImageEncoder(Protocol):
    """Architecture-neutral complete-image encoder contract."""

    def encode_image(self, image: Image.Image | np.ndarray) -> DensePatchFeatures: ...


def _stable_hash(value: object, length: int = 16) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:length]


def _image_sha256(image: Image.Image) -> str:
    rgb = image.convert("RGB")
    digest = hashlib.sha256()
    digest.update(np.asarray(rgb, dtype=np.uint8).tobytes(order="C"))
    digest.update(f"{rgb.width}x{rgb.height}:RGB".encode("ascii"))
    return digest.hexdigest()


def _resolved_model_revision(model_id: str, requested_revision: str) -> str:
    """Use the downloader's pinned commit when it matches this selection."""

    revisions_path = PROJECT_ROOT / "reports" / "metrics" / "model_revisions.json"
    if not revisions_path.is_file():
        return requested_revision
    try:
        revisions = json.loads(revisions_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return requested_revision
    for selection in revisions.values():
        if not isinstance(selection, dict):
            continue
        if (
            selection.get("model_id") == model_id
            and selection.get("requested_revision", "main") == requested_revision
            and isinstance(selection.get("resolved_revision"), str)
        ):
            return str(selection["resolved_revision"])
    return requested_revision


def _opaque_frame_key(frame_id: str) -> str:
    """Validate the renderer's opaque frame ID before using it as a filename."""

    if not isinstance(frame_id, str) or not re.fullmatch(
        r"(?:f|frame)_[0-9a-f]{6,64}", frame_id
    ):
        raise ValueError(
            "frame_id must be opaque and match f_<hex> or frame_<hex>"
        )
    return frame_id


def _opaque_scene_key(scene_id: str) -> str:
    if not isinstance(scene_id, str) or not re.fullmatch(r"scene_[0-9a-f]{6,64}", scene_id):
        raise ValueError("scene_id must be opaque and match scene_<hex>")
    return scene_id


def vision_cache_signature(
    spec: DenseVisionModelSpec,
    revision: str,
    middle_layer: int,
    late_layer: int,
    storage_dtype: torch.dtype,
    aligned_method: str = "tokenwise_projection",
) -> str:
    return _stable_hash(
        {
            "format": 1,
            "model_id": spec.model_id,
            "revision": revision,
            "image_size": spec.image_size,
            "patch_size": spec.patch_size,
            "middle_layer": middle_layer,
            "late_layer": late_layer,
            "native_dim": spec.native_dim * 2,
            "aligned_dim": spec.aligned_dim,
            "storage_dtype": str(storage_dtype),
            "aligned_method": aligned_method,
        }
    )


class DenseCLIPEncoder:
    """Extract localized CLIP features from complete, uncropped render frames."""

    def __init__(
        self,
        model: Any,
        processor: Any,
        spec: DenseVisionModelSpec,
        *,
        device: torch.device,
        compute_dtype: torch.dtype,
        storage_dtype: torch.dtype = torch.float16,
        middle_layer: int | None = None,
        late_layer: int | None = None,
        aligned_method: str = "tokenwise_projection",
    ) -> None:
        if spec.architecture != "clip":
            raise ValueError(f"DenseCLIPEncoder cannot load architecture {spec.architecture!r}")
        self.model = model.eval()
        self.processor = processor
        self.spec = spec
        self.device = device
        self.compute_dtype = compute_dtype
        self.storage_dtype = storage_dtype
        self.middle_layer = (
            spec.default_middle_layer if middle_layer is None else int(middle_layer)
        )
        self.late_layer = spec.default_late_layer if late_layer is None else int(late_layer)
        self.aligned_method = aligned_method
        if aligned_method not in {"tokenwise_projection", "maskclip_value"}:
            raise ValueError(f"Unsupported aligned_method: {aligned_method}")
        spec.validate_layers(self.middle_layer, self.late_layer)
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)

    @classmethod
    def from_pretrained(
        cls,
        model_id: str,
        *,
        revision: str = "main",
        device: torch.device | None = None,
        requested_dtype: str = "float16",
        storage_dtype: torch.dtype = torch.float16,
        middle_layer: int | None = None,
        late_layer: int | None = None,
        aligned_method: str = "tokenwise_projection",
        local_files_only: bool = False,
    ) -> DenseCLIPEncoder:
        """Load public CLIP weights for local inference."""

        from transformers import CLIPModel, CLIPProcessor

        spec = get_model_spec(model_id)
        revision = _resolved_model_revision(model_id, revision)
        selected_device = device or select_device()
        compute_dtype = safe_dtype(selected_device, requested_dtype)
        processor = CLIPProcessor.from_pretrained(
            model_id,
            revision=revision,
            local_files_only=local_files_only,
            use_fast=False,
        )
        model = CLIPModel.from_pretrained(
            model_id,
            revision=revision,
            local_files_only=local_files_only,
            dtype=compute_dtype,
        ).to(selected_device)
        return cls(
            model,
            processor,
            spec,
            device=selected_device,
            compute_dtype=compute_dtype,
            storage_dtype=storage_dtype,
            middle_layer=middle_layer,
            late_layer=late_layer,
            aligned_method=aligned_method,
        )

    def _prepare_image(self, image: Image.Image | np.ndarray) -> Image.Image:
        if isinstance(image, np.ndarray):
            if image.ndim != 3 or image.shape[2] != 3:
                raise ValueError(f"Expected RGB array [H, W, 3], got {image.shape}")
            if image.dtype != np.uint8:
                raise TypeError("RGB arrays must use uint8 values")
            complete_image = Image.fromarray(image)
        elif isinstance(image, Image.Image):
            complete_image = image.convert("RGB")
        else:
            raise TypeError(f"Unsupported image type: {type(image)!r}")

        expected = (self.spec.image_size, self.spec.image_size)
        if complete_image.size != expected:
            raise ValueError(
                f"Dense spatial mapping requires a complete {expected[0]}x{expected[1]} render; "
                f"got {complete_image.width}x{complete_image.height}. Re-render rather than crop."
            )
        return complete_image

    def encode_image(self, image: Image.Image | np.ndarray) -> DensePatchFeatures:
        """Encode one complete image with exactly one vision-transformer call."""

        complete_image = self._prepare_image(image)
        processed = self.processor(images=complete_image, return_tensors="pt")
        if "pixel_values" not in processed:
            raise ValueError("CLIP processor did not return pixel_values")
        pixel_values = processed["pixel_values"]
        expected_shape = (1, 3, self.spec.image_size, self.spec.image_size)
        if tuple(pixel_values.shape) != expected_shape:
            raise ValueError(
                f"Processor must produce one complete image shaped {expected_shape}, "
                f"got {tuple(pixel_values.shape)}"
            )
        pixel_values = pixel_values.to(device=self.device, dtype=self.compute_dtype)

        with torch.inference_mode():
            # This is the only vision-model call in the extraction path.  Middle,
            # late, and aligned streams all come from its returned hidden states.
            vision_outputs = self.model.vision_model(
                pixel_values=pixel_values,
                output_hidden_states=True,
            )
            return extract_clip_streams(
                vision_outputs,
                self.model,
                self.spec,
                self.middle_layer,
                self.late_layer,
                self.storage_dtype,
                self.aligned_method,
            )

    def encode_text_queries(
        self,
        queries: Sequence[str],
        *,
        normalize: bool = True,
    ) -> torch.Tensor:
        """Encode zero-shot text queries into CLIP's 512D comparison space."""

        if not queries or any(not isinstance(query, str) or not query.strip() for query in queries):
            raise ValueError("queries must contain at least one non-empty string")
        processed = self.processor(
            text=list(queries),
            padding=True,
            truncation=True,
            max_length=77,
            return_tensors="pt",
        )
        text_inputs = {
            key: value.to(self.device)
            for key, value in processed.items()
            if key in {"input_ids", "attention_mask", "position_ids"}
        }
        if "input_ids" not in text_inputs:
            raise ValueError("CLIP processor did not return input_ids")
        with torch.inference_mode():
            outputs = self.model.text_model(**text_inputs)
            embeddings = self.model.text_projection(outputs.pooler_output).float()
            if normalize:
                if torch.linalg.vector_norm(embeddings, dim=-1).eq(0).any():
                    raise ValueError("Text encoder returned a zero-norm embedding")
                embeddings = F.normalize(embeddings, dim=-1)
        if not torch.isfinite(embeddings).all():
            raise ValueError("Text encoder returned NaN or infinite values")
        return embeddings.detach().cpu()


@dataclass(frozen=True)
class FrameFeatureCache:
    root: Path
    signature: str

    def path_for(self, frame_id: str) -> Path:
        return self.root / f"{_opaque_frame_key(frame_id)}.npz"

    def load(
        self,
        frame_id: str,
        source_sha256: str,
    ) -> DensePatchFeatures | None:
        path = self.path_for(frame_id)
        if not path.is_file():
            return None
        features, metadata = DensePatchFeatures.load(path)
        expected = {
            "cache_signature": self.signature,
            "frame_key": _opaque_frame_key(frame_id),
            "source_rgb_sha256": source_sha256,
        }
        if any(metadata.get(key) != value for key, value in expected.items()):
            return None
        return features

    def save(
        self,
        frame_id: str,
        source_sha256: str,
        features: DensePatchFeatures,
        *,
        model_id: str,
        revision: str,
        aligned_method: str = "tokenwise_projection",
    ) -> Path:
        return features.save(
            self.path_for(frame_id),
            metadata={
                "format_version": "1",
                "cache_signature": self.signature,
                "frame_key": _opaque_frame_key(frame_id),
                "source_rgb_sha256": source_sha256,
                "model_id": model_id,
                "revision": revision,
                "native_layout": "middle_then_late",
                "aligned_layout": aligned_method,
            },
        )


def _reject_oracle_path(path: Path, purpose: str) -> None:
    if any("oracle" in part.lower() for part in path.resolve().parts):
        raise ValueError(f"Refusing to load {purpose} from oracle directory: {path}")


def _frame_image(frame: FrameRecord) -> Image.Image:
    _reject_oracle_path(frame.rgb_path, "RGB frame")
    with Image.open(frame.rgb_path) as source:
        return source.convert("RGB")


def _storage_dtype(name: str) -> torch.dtype:
    if name == "float16":
        return torch.float16
    if name == "float32":
        return torch.float32
    raise ValueError(f"Unsupported semantic feature dtype {name!r}; use float16 or float32")


def load_dense_image_encoder(
    spec: DenseVisionModelSpec,
    *,
    revision: str,
    device: torch.device | None,
    requested_dtype: str,
    storage_dtype: torch.dtype,
    middle_layer: int,
    late_layer: int,
    aligned_method: str,
    local_files_only: bool,
) -> DenseImageEncoder:
    """Load the selected backend without importing Gemma 4 in the CLIP environment."""

    common = {
        "revision": revision,
        "device": device,
        "requested_dtype": requested_dtype,
        "storage_dtype": storage_dtype,
        "middle_layer": middle_layer,
        "late_layer": late_layer,
        "aligned_method": aligned_method,
        "local_files_only": local_files_only,
    }
    if spec.architecture == "clip":
        return DenseCLIPEncoder.from_pretrained(spec.model_id, **common)
    if spec.architecture == "gemma4":
        from semantic_3d_chat.vision.gemma4_encoder import DenseGemma4Encoder

        return DenseGemma4Encoder.from_pretrained(spec.model_id, **common)
    raise ValueError(f"Unsupported dense vision architecture: {spec.architecture}")


def extract_manifest_features(
    config: dict[str, Any],
    scene_id: str,
    *,
    manifest_path: str | Path | None = None,
    output_root: str | Path | None = None,
    force: bool = False,
    local_files_only: bool = False,
    device: torch.device | None = None,
    encoder: DenseImageEncoder | None = None,
) -> dict[str, Any]:
    """Extract or reuse one cached dense artifact for every manifest frame."""

    scene_id = _opaque_scene_key(scene_id)
    vision = config["vision"]
    spec = get_model_spec(str(vision["model_id"]))
    if int(vision.get("batch_size", 1)) != 1:
        raise ValueError("The initial dense path requires batch_size=1: one call per complete image")
    feature_mode = str(vision.get("feature_mode", "middle_late_aligned"))
    if feature_mode not in {"middle_late_aligned", "middle_late_projected"}:
        raise ValueError(
            "Dense extraction supports middle_late_aligned or middle_late_projected"
        )
    expected_mode = "middle_late_projected" if spec.architecture == "gemma4" else "middle_late_aligned"
    if feature_mode != expected_mode:
        raise ValueError(
            f"{spec.architecture} requires feature_mode={expected_mode}; got {feature_mode}"
        )
    configured_size = int(vision.get("input_size", spec.image_size))
    if configured_size != spec.image_size:
        raise ValueError(
            f"Configured input_size={configured_size} disagrees with {spec.model_id} "
            f"native size {spec.image_size}"
        )
    requested_revision = str(vision.get("revision", "main"))
    revision = _resolved_model_revision(spec.model_id, requested_revision)
    middle_layer = int(vision.get("middle_layer", spec.default_middle_layer))
    late_layer = int(vision.get("late_layer", spec.default_late_layer))
    default_aligned_method = (
        "pooled_native_projector_broadcast"
        if spec.architecture == "gemma4"
        else "tokenwise_projection"
    )
    aligned_method = str(vision.get("aligned_method", default_aligned_method))
    spec.validate_layers(middle_layer, late_layer)
    # Compute and cache dtypes are deliberately independent. Gemma 4 is most
    # reliable on Apple Silicon in bfloat16, while NumPy's portable feature
    # artifacts remain float16 (or float32).
    storage_dtype = _storage_dtype(
        str(vision.get("storage_dtype", vision.get("dtype", "float16")))
    )
    signature = vision_cache_signature(
        spec, revision, middle_layer, late_layer, storage_dtype, aligned_method
    )

    source_manifest = (
        Path(manifest_path)
        if manifest_path is not None
        else project_path(config, "rendered", scene_id, "manifest.json")
    ).resolve()
    _reject_oracle_path(source_manifest, "rendering manifest")
    frames = list(iter_frames(source_manifest))
    if not frames:
        raise ValueError(f"Sanitized manifest contains no frames: {source_manifest}")
    frame_ids = [frame.frame_id for frame in frames]
    if len(frame_ids) != len(set(frame_ids)):
        raise ValueError("Sanitized manifest contains duplicate frame IDs")

    base_output = (
        Path(output_root)
        if output_root is not None
        else project_path(config, "features", scene_id)
    ).resolve()
    cache = FrameFeatureCache(base_output, signature)
    cache.root.mkdir(parents=True, exist_ok=True)

    active_encoder = encoder
    entries: list[dict[str, Any]] = []
    extracted = 0
    reused = 0
    for frame in frames:
        image = _frame_image(frame)
        source_sha256 = _image_sha256(image)
        features = None if force else cache.load(frame.frame_id, source_sha256)
        if features is None:
            if active_encoder is None:
                active_encoder = load_dense_image_encoder(
                    spec,
                    revision=revision,
                    device=device,
                    requested_dtype=str(vision.get("dtype", "float16")),
                    storage_dtype=storage_dtype,
                    middle_layer=middle_layer,
                    late_layer=late_layer,
                    aligned_method=aligned_method,
                    local_files_only=local_files_only,
                )
            features = active_encoder.encode_image(image)
            cache.save(
                frame.frame_id,
                source_sha256,
                features,
                model_id=spec.model_id,
                revision=revision,
                aligned_method=aligned_method,
            )
            extracted += 1
        else:
            reused += 1

        path = cache.path_for(frame.frame_id)
        entries.append(
            {
                "frame_id": frame.frame_id,
                "feature_path": path.name,
                "source_rgb_sha256": source_sha256,
                "grid_size": list(features.grid_size),
                "native_middle_late_dim": int(features.native_middle_late.shape[-1]),
                "clip_aligned_dim": int(features.clip_aligned.shape[-1]),
                "aligned_dim": int(features.aligned.shape[-1]),
                "spatial_feature_dim": int(features.spatial_features.shape[-1]),
                "component_offsets": list(features.component_offsets),
            }
        )
        LOGGER.info(
            "phase=vision scene=%s frame=%s grid=%sx%s native_dim=%s aligned_dim=%s",
            scene_id,
            _opaque_frame_key(frame.frame_id),
            *features.grid_size,
            features.native_middle_late.shape[-1],
            features.clip_aligned.shape[-1],
        )

    index = {
        "format_version": 1,
        "scene_id": scene_id,
        "cache_signature": signature,
        "model_id": spec.model_id,
        "architecture": spec.architecture,
        "requested_revision": requested_revision,
        "revision": revision,
        "input_size": spec.image_size,
        "patch_size": spec.patch_size,
        "spatial_grid_size": list(spec.grid_size),
        "pooling_kernel_size": spec.pooling_kernel_size,
        "middle_layer": middle_layer,
        "late_layer": late_layer,
        "aligned_method": aligned_method,
        "storage_dtype": str(storage_dtype).removeprefix("torch."),
        "frames": entries,
    }
    index_path = cache.root / "manifest.json"
    temporary = cache.root / ".manifest.json.tmp"
    try:
        temporary.write_text(json.dumps(index, indent=2) + "\n", encoding="utf-8")
        temporary.replace(index_path)
    finally:
        temporary.unlink(missing_ok=True)

    return {
        "scene_id": scene_id,
        "manifest": str(source_manifest),
        "feature_manifest": str(index_path),
        "cache_signature": signature,
        "frames": len(frames),
        "extracted": extracted,
        "reused": reused,
    }


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--scene", required=True)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Require model weights and processor files to already be cached",
    )
    parser.add_argument("--device", choices=("auto", "cpu", "mps"), default="auto")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    if args.device == "auto":
        device = select_device()
    else:
        device = torch.device(args.device)
        if device.type == "mps" and not torch.backends.mps.is_available():
            raise RuntimeError("MPS was requested but is unavailable")
    result = extract_manifest_features(
        load_config(args.config),
        args.scene,
        manifest_path=args.manifest,
        output_root=args.output_root,
        force=args.force,
        local_files_only=args.offline,
        device=device,
    )
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
