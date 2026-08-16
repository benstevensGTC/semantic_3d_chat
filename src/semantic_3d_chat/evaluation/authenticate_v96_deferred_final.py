"""Authenticate one create-once stage of the V96 deferred-final chain."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from typing import Any

from semantic_3d_chat.evaluation.nll_v96_deferred_final import (
    authenticate_nll_v96_final,
)
from semantic_3d_chat.evaluation.score_v96_deferred_final import (
    authenticate_structured_score_v96_final,
)
from semantic_3d_chat.evaluation.seal_v96_deferred_final import (
    authenticate_deferred_final_evidence_v96,
)
from semantic_3d_chat.evaluation.v96_deferred_final_common import (
    authenticate_prediction_bundle_v96_final,
)
from semantic_3d_chat.evaluation.v96_deferred_final_evaluation import (
    authenticate_materialized_inputs_v96_final,
    authenticate_preregistration_v96_final,
)


def authenticate_stage_v96_final(command: str) -> dict[str, Any]:
    if command == "preregistration":
        return authenticate_preregistration_v96_final()
    if command == "materialized-predictor":
        return authenticate_materialized_inputs_v96_final(label_process=False)
    if command == "materialized-label":
        return authenticate_materialized_inputs_v96_final(label_process=True)
    if command == "predictions-v96":
        return authenticate_prediction_bundle_v96_final("v96")
    if command == "predictions-v94":
        return authenticate_prediction_bundle_v96_final("v94")
    if command == "structured":
        return authenticate_structured_score_v96_final()
    if command == "nll":
        return authenticate_nll_v96_final()
    if command == "final":
        return authenticate_deferred_final_evidence_v96()
    raise ValueError(f"Unknown V96 deferred-final authentication stage: {command}")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=(
            "preregistration",
            "materialized-predictor",
            "materialized-label",
            "predictions-v96",
            "predictions-v94",
            "structured",
            "nll",
            "final",
        ),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    result = authenticate_stage_v96_final(args.command)
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["authenticate_stage_v96_final", "main"]
