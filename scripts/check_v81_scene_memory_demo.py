"""Fast, model-free authentication for the prepared V81 room demo."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from semantic_3d_chat.chat.grounding_sidecar_v78_runtime import (
    authenticate_v78_grounding_checkpoint,
)
from semantic_3d_chat.chat.runtime_config import (
    effective_runtime_config_sha256,
    load_runtime_config,
)
from semantic_3d_chat.evaluation.prediction_artifacts import checkpoint_fingerprint
from semantic_3d_chat.scene_encoder.v81_scene_memory_artifact import (
    load_v81_scene_memory,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--scene", required=True)
    parser.add_argument("--base-checkpoint", required=True)
    parser.add_argument("--scene-memory", required=True)
    parser.add_argument(
        "--grounding-checkpoint",
        help="Optionally authenticate the V78 numeric grounding sidecar.",
    )
    return parser


def validate_v81_scene_memory_demo_inputs(
    *,
    config_path: str | Path,
    scene_id: str,
    base_checkpoint: str | Path,
    scene_memory: str | Path,
    grounding_checkpoint: str | Path | None = None,
) -> dict[str, Any]:
    """Authenticate V81 and an optional V78 sidecar without loading Gemma."""

    config = load_runtime_config(config_path)
    base_sha256, files = checkpoint_fingerprint(base_checkpoint)
    runtime_sha256 = effective_runtime_config_sha256(config)
    loaded = load_v81_scene_memory(
        scene_memory,
        expected_scene_id=scene_id,
        expected_base_checkpoint_sha256=base_sha256,
        expected_runtime_config_sha256=runtime_sha256,
        expected_model_device="cpu",
    )
    maps_root = Path(str(config["paths"]["maps_root"]))
    if not maps_root.is_absolute():
        maps_root = Path.cwd() / maps_root
    map_path = maps_root / scene_id / "voxel_map.npz"
    if not map_path.is_file() or map_path.is_symlink():
        raise FileNotFoundError(f"Prepared sanitized map is unavailable: {map_path}")
    grounding = None
    if grounding_checkpoint is not None:
        grounding = authenticate_v78_grounding_checkpoint(
            grounding_checkpoint,
            base_checkpoint_sha256=base_sha256,
            base_runtime_config_sha256=runtime_sha256,
            model_id=str(config["language"]["model_id"]),
            model_revision=str(config["language"]["revision"]),
        )
    return {
        "passed": True,
        "scene_id": scene_id,
        "base_checkpoint_sha256": base_sha256,
        "base_checkpoint_file_count": len(files),
        "runtime_config_sha256": runtime_sha256,
        "fixed_scene_memory_sha256": loaded.metadata["canonical_prefix_sha256"],
        "fixed_scene_memory_tensor_sha256": loaded.metadata["tensor_sha256"],
        "fixed_scene_memory_shape": loaded.metadata["shape"],
        "base_prefix_sha256": loaded.metadata["base_prefix_sha256"],
        "map_path": str(map_path.resolve()),
        "compiled_before_user_question": True,
        "questions_or_answers_serialized": False,
        "environmental_text_inputs": [],
        "optional_v78_grounding_supplied": grounding is not None,
        "optional_v78_grounding_checkpoint_authentication": grounding,
        "gemma_loaded": False,
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    report = validate_v81_scene_memory_demo_inputs(
        config_path=args.config,
        scene_id=args.scene,
        base_checkpoint=args.base_checkpoint,
        scene_memory=args.scene_memory,
        grounding_checkpoint=args.grounding_checkpoint,
    )
    print(
        json.dumps(
            report,
            sort_keys=True,
            allow_nan=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["main", "validate_v81_scene_memory_demo_inputs"]
