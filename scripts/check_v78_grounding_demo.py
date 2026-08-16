#!/usr/bin/env python3
"""Authenticate the optional V78 grounding sidecar without loading Gemma."""

from __future__ import annotations

import argparse
import json
import os
from collections.abc import Sequence
from pathlib import Path
from typing import Any

try:
    from scripts.check_v75_demo import (
        DEFAULT_BASE_CHECKPOINT,
        DEFAULT_CONFIG,
        DEFAULT_CONTROL_CHECKPOINT,
        validate_v75_demo_inputs,
    )
except ModuleNotFoundError:  # Direct ``python scripts/...`` execution.
    from check_v75_demo import (  # type: ignore[no-redef]
        DEFAULT_BASE_CHECKPOINT,
        DEFAULT_CONFIG,
        DEFAULT_CONTROL_CHECKPOINT,
        validate_v75_demo_inputs,
    )

from semantic_3d_chat.chat.grounding_sidecar_v78_runtime import (
    authenticate_v78_grounding_checkpoint,
)
from semantic_3d_chat.chat.runtime_config import (
    effective_runtime_config_sha256,
    load_runtime_config,
)
from semantic_3d_chat.config import PROJECT_ROOT

DEFAULT_GROUNDING_CHECKPOINT = Path(
    "data_gemma4/runtime/checkpoints/gemma4_v78_grounding_diagnostic_release_v1"
)


def _rooted(path: str | Path) -> Path:
    value = Path(path).expanduser()
    rooted = value if value.is_absolute() else PROJECT_ROOT / value
    return Path(os.path.abspath(rooted))


def validate_v78_grounding_demo_inputs(
    *,
    config_path: str | Path = DEFAULT_CONFIG,
    scene_id: str = "scene_000001",
    base_checkpoint: str | Path = DEFAULT_BASE_CHECKPOINT,
    control_checkpoint: str | Path = DEFAULT_CONTROL_CHECKPOINT,
    grounding_checkpoint: str | Path = DEFAULT_GROUNDING_CHECKPOINT,
) -> dict[str, Any]:
    """Authenticate V75 plus the explicitly optional V78 numeric sidecar."""

    v75 = validate_v75_demo_inputs(
        config_path=config_path,
        scene_id=scene_id,
        base_checkpoint=base_checkpoint,
        control_checkpoint=control_checkpoint,
    )
    config = load_runtime_config(config_path)
    runtime_sha256 = effective_runtime_config_sha256(config)
    grounding = authenticate_v78_grounding_checkpoint(
        grounding_checkpoint,
        base_checkpoint_sha256=str(v75["base_checkpoint_sha256"]),
        base_runtime_config_sha256=runtime_sha256,
        model_id=str(config["language"]["model_id"]),
        model_revision=str(config["language"]["revision"]),
    )
    return {
        "schema_version": 1,
        "artifact": "optional_v78_grounding_demo_preflight_v1",
        "passed": True,
        "scene_id": scene_id,
        "v75_answer_runtime": v75,
        "v78_numeric_grounding": grounding,
        "answer_generation_unchanged": True,
        "optional_grounding_only": True,
        "loads_gemma_model": False,
        "loads_scene_data": False,
        "loads_oracle_or_qa": False,
        "official_validation_evidence": False,
        "runtime_promotion_authorized": False,
        "environmental_text_inputs": [],
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--scene", default="scene_000001")
    parser.add_argument("--base-checkpoint", default=str(DEFAULT_BASE_CHECKPOINT))
    parser.add_argument("--control-checkpoint", default=str(DEFAULT_CONTROL_CHECKPOINT))
    parser.add_argument(
        "--grounding-checkpoint", default=str(DEFAULT_GROUNDING_CHECKPOINT)
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        report = validate_v78_grounding_demo_inputs(
            config_path=args.config,
            scene_id=args.scene,
            base_checkpoint=args.base_checkpoint,
            control_checkpoint=args.control_checkpoint,
            grounding_checkpoint=args.grounding_checkpoint,
        )
    except (OSError, RuntimeError, TypeError, ValueError) as error:
        print(f"V78 grounding demo preflight refused: {error}", file=os.sys.stderr)
        return 2
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "DEFAULT_GROUNDING_CHECKPOINT",
    "main",
    "validate_v78_grounding_demo_inputs",
]
