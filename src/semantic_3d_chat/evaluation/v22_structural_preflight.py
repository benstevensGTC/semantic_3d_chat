"""Pinned V22 margin-rebalanced wrapper around the audited V21 preflight."""

from __future__ import annotations

import argparse

from semantic_3d_chat.evaluation.phase_aware_local_field_profile import (
    V22_LOCAL_FIELD_PROFILE,
)
from semantic_3d_chat.evaluation.v21_structural_preflight import (
    V21StructuralPreflightViolation,
    run_preflight,
    validate_v21_config_contract,
)

V22StructuralPreflightViolation = V21StructuralPreflightViolation


def validate_v22_config_contract(config):
    """Validate the exact, hash-pinned V22 controller contract."""

    return validate_v21_config_contract(config, profile=V22_LOCAL_FIELD_PROFILE)


def run_v22_preflight(config_path, report_path):
    return run_preflight(config_path, report_path, profile=V22_LOCAL_FIELD_PROFILE)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=V22_LOCAL_FIELD_PROFILE.config_path)
    parser.add_argument("--report", required=True)
    args = parser.parse_args()
    run_v22_preflight(args.config, args.report)


if __name__ == "__main__":  # pragma: no cover - local model command
    main()


__all__ = [
    "V22StructuralPreflightViolation",
    "run_v22_preflight",
    "validate_v22_config_contract",
]
