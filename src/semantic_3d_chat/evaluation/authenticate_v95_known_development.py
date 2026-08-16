"""Independent model-free authenticators for V95 evaluation evidence.

These commands never open reference labels and never load Gemma.  They expose
the same fail-closed checks used by the final evidence sealer as an explicit
standalone process boundary.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from typing import Any

from semantic_3d_chat.evaluation.v95_known_development_common import (
    authenticate_nll_bundle_v95,
    authenticate_prediction_bundle_v95,
    authenticate_structured_score_v95,
)
from semantic_3d_chat.evaluation.v95_known_development_implementation import (
    hardened_evaluation_stage_v95,
)
from semantic_3d_chat.evaluation.v95_strict_causal_successor_preflight import CONFIG


@hardened_evaluation_stage_v95
def authenticate_v95(command: str, config: str) -> dict[str, Any]:
    if command == "prediction":
        bundle = authenticate_prediction_bundle_v95(config)
        return {
            "artifact": "gemma4_v95_prediction_bundle_authentication_v1",
            "authenticated": True,
            "row_count": len(bundle["rows"]),
            "candidate_fingerprint_sha256": bundle["fixed"].candidate["fingerprint_sha256"],
            "prediction_sha256": bundle["prediction_sha256"],
            "prediction_access_sha256": bundle["access_sha256"],
            "prediction_completion_sha256": bundle["completion_sha256"],
            "labels_opened": False,
            "model_loaded": False,
            "runtime_promotion_authorized": False,
        }
    if command == "structured":
        result = authenticate_structured_score_v95(config)
        return {
            "artifact": "gemma4_v95_structured_score_authentication_v1",
            "authenticated": True,
            "structured_score_sha256": result["sha256"],
            "labels_opened": False,
            "model_loaded": False,
            "runtime_promotion_authorized": False,
        }
    if command == "nll":
        result = authenticate_nll_bundle_v95(config)
        return {
            "artifact": "gemma4_v95_nll_bundle_authentication_v1",
            "authenticated": True,
            "nll_sha256": result["sha256"],
            "nll_access_sha256": result["access_sha256"],
            "nll_completion_sha256": result["completion_sha256"],
            "labels_opened": False,
            "model_loaded": False,
            "runtime_promotion_authorized": False,
        }
    raise ValueError(f"Unknown V95 authentication command: {command}")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prediction", "structured", "nll"))
    parser.add_argument("--config", default=str(CONFIG))
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    print(json.dumps(authenticate_v95(args.command, args.config), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["authenticate_v95", "main"]
