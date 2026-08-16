"""CPU-only authorizer for the unpromoted V96 explicit-candidate runtime.

This command is intentionally a separate process from chat.  It executes the
sealed V96 candidate/evidence authenticators, emits only hashes and sanitized
paths, and never loads Gemma.  The chat process refuses to load a model until
this command returns an authenticated PASS payload.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Final

from semantic_3d_chat.chat.runtime_config import (
    effective_runtime_config_sha256,
    load_runtime_config,
)
from semantic_3d_chat.chat.v96_explicit_candidate_runtime import (
    AUTHORIZATION_ARTIFACT,
    AUTHORIZATION_STATUS,
    V94_STATE_SHA256,
    V96CandidateAuthorization,
    validate_v96_pass_evidence,
)
from semantic_3d_chat.config import PROJECT_ROOT

DEFAULT_AUTHORIZATION_CONFIG: Final[str] = (
    "configs/experiments/gemma4_v96_atomic_pair_repair.yaml"
)


def _absolute(path: str | Path) -> Path:
    value = Path(path).expanduser()
    rooted = value if value.is_absolute() else PROJECT_ROOT / value
    return Path(os.path.abspath(rooted))


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with _absolute(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_authorization_payload(
    *,
    config_path: str | Path,
    config: Mapping[str, Any],
    v95_config: Mapping[str, Any],
    candidate: Mapping[str, Any],
    final_evidence: Mapping[str, Any],
    final_score_path: str | Path,
    evidence_path: str | Path,
) -> dict[str, Any]:
    """Assemble the strict hash-only handoff after official authentication."""

    pass_contract = validate_v96_pass_evidence(
        candidate=candidate,
        evidence=final_evidence,
    )
    sources = config["sources"]
    outputs = config["outputs"]
    v95_sources = v95_config["sources"]
    runtime_config_path = _absolute(sources["runtime_config"])
    runtime_config = load_runtime_config(runtime_config_path)
    if _sha256_file(runtime_config_path) != sources["runtime_config_sha256"]:
        raise ValueError("V96 authorized runtime-config bytes changed")

    payload = {
        "artifact": AUTHORIZATION_ARTIFACT,
        "schema_version": 96,
        "status": AUTHORIZATION_STATUS,
        "authorization_config_path": str(_absolute(config_path)),
        "authorization_config_sha256": _sha256_file(config_path),
        "runtime_config_path": str(runtime_config_path),
        "runtime_config_file_sha256": str(sources["runtime_config_sha256"]),
        "runtime_config_effective_sha256": effective_runtime_config_sha256(
            runtime_config
        ),
        "v85_checkpoint_path": str(_absolute(v95_sources["frozen_v85_checkpoint"])),
        "v85_adapter_sha256": str(v95_sources["frozen_v85_adapter_sha256"]),
        "v85_metadata_sha256": str(v95_sources["frozen_v85_metadata_sha256"]),
        "v94_bridge_path": str(_absolute(v95_sources["frozen_v94_fixed_final"])),
        "v94_weights_sha256": str(v95_sources["frozen_v94_bridge_sha256"]),
        "v94_metadata_sha256": str(
            v95_sources["frozen_v94_bridge_metadata_sha256"]
        ),
        "v94_state_sha256": V94_STATE_SHA256,
        "v95_bridge_path": str(_absolute(sources["frozen_v95_fixed_final"])),
        "v95_weights_sha256": str(sources["frozen_v95_bridge_sha256"]),
        "v95_metadata_sha256": str(sources["frozen_v95_bridge_metadata_sha256"]),
        "v95_state_sha256": str(pass_contract["frozen_v95_state_sha256"]),
        "v96_candidate_path": str(_absolute(outputs["fixed_final_candidate"])),
        "v96_weights_sha256": str(candidate["weights_sha256"]),
        "v96_metadata_file_sha256": str(candidate["metadata_file_sha256"]),
        "v96_metadata_canonical_sha256": str(
            candidate["metadata_canonical_sha256"]
        ),
        "v96_state_sha256": str(pass_contract["candidate_state_sha256"]),
        "candidate_fingerprint_sha256": str(
            pass_contract["candidate_fingerprint_sha256"]
        ),
        "config_sha256": str(candidate["config_sha256"]),
        "preregistration_sha256": str(candidate["preregistration_sha256"]),
        "cpu_preflight_sha256": str(candidate["cpu_preflight_sha256"]),
        "training_report_sha256": str(candidate["training_report_sha256"]),
        "final_score_path": str(_absolute(final_score_path)),
        "final_score_sha256": str(final_evidence["final_score_sha256"]),
        "evidence_path": str(_absolute(evidence_path)),
        "evidence_sha256": str(final_evidence["evidence_sha256"]),
        "implementation_seal_sha256": str(
            final_evidence["implementation_seal_sha256"]
        ),
        "implementation_source_inventory_sha256": str(
            final_evidence["implementation_source_inventory_sha256"]
        ),
        "v1_implementation_seal_sha256": str(
            pass_contract["v1_implementation_seal_sha256"]
        ),
        "v2_implementation_seal_sha256": str(
            pass_contract["v2_implementation_seal_sha256"]
        ),
        "candidate_attestation_file_sha256": str(
            pass_contract["candidate_attestation_file_sha256"]
        ),
        "candidate_attestation_identity_sha256": str(
            pass_contract["candidate_attestation_identity_sha256"]
        ),
        "candidate_attestation_immutable": True,
        "gate_results_sha256": str(pass_contract["gate_results_sha256"]),
        "gate_count": int(pass_contract["gate_count"]),
        "all_gate_results_passed": True,
        "candidate_authenticated": True,
        "pass_evidence_authenticated": True,
        "known_development_gate_passed": True,
        "scene_prefix_question_independent": True,
        "row_level_content_serialized": False,
        "environmental_text_inputs": [],
        "deferred_final_unlock_eligible": True,
        "automatic_runtime_promotion": False,
        "runtime_promotion_authorized": False,
        "explicit_candidate_flag_required": True,
    }
    return V96CandidateAuthorization.from_payload(payload).to_payload()


def authorize_v96_explicit_candidate(
    config_path: str | Path = DEFAULT_AUTHORIZATION_CONFIG,
) -> dict[str, Any]:
    """Run official full authentication without constructing a language model."""

    # Deferred imports make the process boundary explicit: none of these
    # evaluator/trainer surfaces are imported by the chat runtime module.
    from semantic_3d_chat.evaluation.seal_v96_known_development_v2 import (
        authenticate_final_evidence_v96,
    )
    from semantic_3d_chat.evaluation.v95_strict_causal_successor_preflight import (
        load_config_v95,
    )
    from semantic_3d_chat.evaluation.v96_atomic_pair_repair_preflight import (
        load_config_v96,
    )
    from semantic_3d_chat.evaluation.v96_known_development_common_v2 import (
        authenticate_fixed_final_candidate_v96,
        evaluation_paths_v96,
    )

    source = _absolute(config_path)
    config = load_config_v96(source, allow_draft=False)
    candidate = authenticate_fixed_final_candidate_v96(
        config,
        config_path=source,
    )
    final_evidence = authenticate_final_evidence_v96(source)
    paths = evaluation_paths_v96(config)
    v95_config = load_config_v95(config["sources"]["frozen_v95_config"], allow_draft=False)
    return build_authorization_payload(
        config_path=source,
        config=config,
        v95_config=v95_config,
        candidate=candidate,
        final_evidence=final_evidence,
        final_score_path=paths.final_score,
        evidence_path=paths.evidence,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=DEFAULT_AUTHORIZATION_CONFIG)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        payload = authorize_v96_explicit_candidate(args.config)
    except (OSError, RuntimeError, TypeError, ValueError) as error:
        print(f"V96 explicit-candidate authorization refused: {error}", file=sys.stderr)
        return 2
    print(json.dumps(payload, sort_keys=True, allow_nan=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "DEFAULT_AUTHORIZATION_CONFIG",
    "authorize_v96_explicit_candidate",
    "build_authorization_payload",
    "main",
]
