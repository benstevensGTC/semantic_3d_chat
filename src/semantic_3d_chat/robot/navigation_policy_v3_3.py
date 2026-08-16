"""V3.3 sealed development successor for corrected live protocol routing.

The implementation is inherited from the CPU-diagnosed V3.2 numeric planner.
V3.3 exists as a distinct runtime identity so the failed, byte-sealed V3.2
attempt cannot be confused with this corrected successor.
"""

from __future__ import annotations

from typing import Final

from semantic_3d_chat.robot.navigation_policy_v3_2 import (
    SemanticGroundedActionBackendV32,
    is_compound_scan_approach_instruction,
    literal_navigation_instruction,
)

RUNTIME_INTERLOCK_VERSION: Final[str] = "v3.3"


class SemanticGroundedActionBackendV33(SemanticGroundedActionBackendV32):
    """Corrected envelope routing plus all-map numeric waypoint calibration."""

    runtime_interlock_version = RUNTIME_INTERLOCK_VERSION


__all__ = [
    "RUNTIME_INTERLOCK_VERSION",
    "SemanticGroundedActionBackendV33",
    "is_compound_scan_approach_instruction",
    "literal_navigation_instruction",
]
