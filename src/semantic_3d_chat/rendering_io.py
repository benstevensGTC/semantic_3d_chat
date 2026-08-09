from __future__ import annotations

import json
import re
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

import numpy as np
from PIL import Image

OPAQUE_SCENE_ID = re.compile(r"scene_[0-9]{6}")
OPAQUE_FRAME_ID = re.compile(r"(?:f|frame)_[0-9]{6}")
OPAQUE_CAMERA_ID = re.compile(r"c_[0-9]{6}")
RUNTIME_MANIFEST_KEYS = frozenset(
    {
        "schema_version",
        "scene_id",
        "config_hash",
        "coordinate_system",
        "image_size",
        "horizontal_fov_degrees",
        "frames",
    }
)
RUNTIME_FRAME_KEYS = frozenset(
    {
        "frame_id",
        "camera_id",
        "frame_number",
        "timestamp",
        "rgb_path",
        "depth_path",
        "intrinsics",
        "camera_to_world",
    }
)
FORBIDDEN_RUNTIME_KEYS = frozenset(
    {
        "answer",
        "answers",
        "caption",
        "category",
        "description",
        "instance_id",
        "label",
        "labels",
        "object_id",
        "object_name",
        "objects",
        "oracle",
        "relationship",
        "relationships",
        "scene_graph",
        "segmentation",
        "semantic_label",
        "support_surface",
        "target_instance",
    }
)


@dataclass(frozen=True)
class FrameRecord:
    frame_id: str
    rgb_path: Path
    depth_path: Path
    intrinsics: np.ndarray
    camera_to_world: np.ndarray


def load_manifest(path: str | Path) -> dict:
    manifest_path = Path(path)
    with manifest_path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    serialized_keys = set()

    def visit(value: object) -> None:
        if isinstance(value, dict):
            serialized_keys.update(str(key).lower() for key in value)
            for child in value.values():
                visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)

    visit(manifest)
    overlap = FORBIDDEN_RUNTIME_KEYS & serialized_keys
    if overlap:
        raise ValueError(f"Sanitized manifest contains forbidden semantic keys: {sorted(overlap)}")
    unexpected_top_level = set(manifest) - RUNTIME_MANIFEST_KEYS
    if unexpected_top_level:
        raise ValueError(
            f"Runtime manifest contains unexpected keys: {sorted(unexpected_top_level)}"
        )
    scene_id = manifest.get("scene_id")
    if scene_id is not None and not OPAQUE_SCENE_ID.fullmatch(str(scene_id)):
        raise ValueError("Runtime manifest scene_id must be opaque")
    frames = manifest.get("frames")
    if not isinstance(frames, list):
        raise TypeError("Runtime manifest frames must be a list")
    for frame in frames:
        if not isinstance(frame, dict):
            raise TypeError("Runtime manifest frame records must be mappings")
        unexpected_frame_keys = set(frame) - RUNTIME_FRAME_KEYS
        if unexpected_frame_keys:
            raise ValueError(
                f"Runtime frame contains unexpected keys: {sorted(unexpected_frame_keys)}"
            )
        frame_id = str(frame.get("frame_id", ""))
        if not OPAQUE_FRAME_ID.fullmatch(frame_id):
            raise ValueError("Runtime manifest frame_id must be opaque")
        camera_id = frame.get("camera_id")
        if camera_id is not None and not OPAQUE_CAMERA_ID.fullmatch(str(camera_id)):
            raise ValueError("Runtime manifest camera_id must be opaque")
        for key, directory, suffix in (
            ("rgb_path", "rgb", ".png"),
            ("depth_path", "depth", ".npy"),
        ):
            relative_path = PurePosixPath(str(frame.get(key, "")))
            if (
                relative_path.is_absolute()
                or relative_path.parts != (directory, f"{frame_id}{suffix}")
            ):
                raise ValueError(f"Runtime manifest {key} must use its opaque frame ID")
    coordinate_system = manifest.get("coordinate_system")
    if coordinate_system is not None and coordinate_system != {
        "world": "x_right_y_forward_z_up",
        "camera": "x_right_y_down_z_forward",
        "units": "meters",
        "depth": "axial_camera_z",
    }:
        raise ValueError("Runtime manifest coordinate_system is not canonical")
    return manifest


def iter_frames(manifest_path: str | Path) -> Iterator[FrameRecord]:
    manifest_path = Path(manifest_path).resolve()
    manifest = load_manifest(manifest_path)
    for frame in manifest["frames"]:
        yield FrameRecord(
            frame_id=frame["frame_id"],
            rgb_path=(manifest_path.parent / frame["rgb_path"]).resolve(),
            depth_path=(manifest_path.parent / frame["depth_path"]).resolve(),
            intrinsics=np.asarray(frame["intrinsics"], dtype=np.float64),
            camera_to_world=np.asarray(frame["camera_to_world"], dtype=np.float64),
        )


def load_rgb_depth(frame: FrameRecord) -> tuple[np.ndarray, np.ndarray]:
    rgb = np.asarray(Image.open(frame.rgb_path).convert("RGB"), dtype=np.uint8)
    depth = np.load(frame.depth_path, allow_pickle=False).astype(np.float32, copy=False)
    if depth.shape != rgb.shape[:2]:
        raise ValueError(f"RGB/depth shape mismatch: {rgb.shape[:2]} vs {depth.shape}")
    if np.any(depth < 0) or np.any(np.isinf(depth)):
        raise ValueError("Depth contains negative or infinite values")
    return rgb, depth
