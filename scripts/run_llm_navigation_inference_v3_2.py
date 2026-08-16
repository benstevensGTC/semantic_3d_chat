#!/usr/bin/env python3
"""Run the V3 checkpoint with the V3.2 numeric runtime calibration."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import run_llm_navigation_inference as base

from semantic_3d_chat.config import PROJECT_ROOT
from semantic_3d_chat.robot.navigation_policy_v3_2 import (
    RUNTIME_INTERLOCK_VERSION,
    SemanticGroundedActionBackendV32,
)

_ORIGINAL_RUN_CONTRACT = base._run_contract
_THIS_SOURCE = PROJECT_ROOT / "scripts/run_llm_navigation_inference_v3_2.py"
_RUNTIME_SOURCE = (
    PROJECT_ROOT / "src/semantic_3d_chat/robot/navigation_policy_v3_2.py"
)


def _v3_2_run_contract(**kwargs: Any) -> dict[str, Any]:
    result = _ORIGINAL_RUN_CONTRACT(**kwargs)
    if kwargs.get("navigation_policy_version") != 3:
        raise ValueError("V3.2 inference requires the accepted V3 checkpoint contract")
    result.update(
        {
            "inference_source_sha256": base.file_sha256(_THIS_SOURCE),
            "navigation_policy_source_sha256": base.file_sha256(_RUNTIME_SOURCE),
            "navigation_runtime_interlock_version": RUNTIME_INTERLOCK_VERSION,
            "compound_scan_approach_numeric_planner": True,
            "planner_uses_current_continuous_map_geometry": True,
            "planner_environmental_text_inputs": [],
            "planner_oracle_inputs_at_runtime": False,
        }
    )
    return result


def install_v3_2_routing() -> type[SemanticGroundedActionBackendV32]:
    """Install the exact backend and run-contract hooks used by live main."""

    base._run_contract = _v3_2_run_contract
    base.SemanticGroundedActionBackendV3 = SemanticGroundedActionBackendV32
    return base.SemanticGroundedActionBackendV3


def main() -> int:
    if Path(__file__).resolve() != _THIS_SOURCE.resolve():
        raise RuntimeError("V3.2 inference source path is ambiguous")
    install_v3_2_routing()
    return base.main()


if __name__ == "__main__":
    raise SystemExit(main())
