"""Two strictly label-blind predictors for V96 deferred-final evaluation."""

from __future__ import annotations

import argparse
import json
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch

from semantic_3d_chat.chat.file_audit import FileAccessAudit
from semantic_3d_chat.evaluation.evaluate_v94_strict_multiscene_full40 import (
    _generate_arm as generate_arm_v94,
)
from semantic_3d_chat.evaluation.evaluate_v94_strict_multiscene_full40 import (
    _load_config as load_config_v94,
)
from semantic_3d_chat.evaluation.evaluate_v94_strict_multiscene_full40 import (
    _load_predictor_stack_v94 as load_predictor_stack_v94,
)
from semantic_3d_chat.evaluation.predict_v96_known_development_v2 import (
    generate_arm_v96,
    load_predictor_stack_v96,
)
from semantic_3d_chat.evaluation.v85_strict_multiscene_preflight import (
    sha256_file_v85,
)
from semantic_3d_chat.evaluation.v95_deferred_final_materialization import (
    QUESTION_MANIFEST,
)
from semantic_3d_chat.evaluation.v96_atomic_pair_repair_preflight import (
    CONFIG,
    load_config_v96,
)
from semantic_3d_chat.evaluation.v96_deferred_final_common import (
    ARMS,
    PREDICTION_COMPLETION_ARTIFACT,
    PRIMARY,
    SCHEMA_VERSION,
    V94_COMPLETION_ARTIFACT,
    V94_PREDICTION_ARTIFACT,
    V96_PREDICTION_ARTIFACT,
    audit_report_v96_final,
    authenticate_fixed_inputs_before_questions_v96_final,
    authenticate_prediction_bundle_v96_final,
    load_questions_v96_final,
    prediction_forbidden_roots_v96_final,
    prediction_provenance_v96_final,
    validate_prediction_rows_v96_final,
    write_json_create_once_v96,
    write_jsonl_create_once_v96,
)
from semantic_3d_chat.evaluation.v96_deferred_final_evaluation import (
    PAIR_SCENE,
    QUESTION_COUNT,
    V94_CONFIG,
    hardened_deferred_evaluation_stage_v96,
    output_paths_v96_final,
)
from semantic_3d_chat.evaluation.v96_known_development_common_v2 import (
    assert_output_bundle_state_v96,
    authenticate_fixed_final_candidate_v96,
)
from semantic_3d_chat.language.prefix_injection import prefix_sha256


def _bundle_paths(model: str) -> tuple[Path, Path, Path, Path]:
    paths = output_paths_v96_final()
    prefix = "v96" if model == "v96" else "v94"
    return (
        paths[f"{prefix}_predictions"],
        paths[f"{prefix}_prediction_provenance"],
        paths[f"{prefix}_prediction_access"],
        paths[f"{prefix}_prediction_completion"],
    )


def _authenticate_v94_bytes(preregistration: Mapping[str, Any]) -> dict[str, Any]:
    contract = preregistration["v94_same_row_comparator"]
    config = Path(contract["experiment_config"])
    config = config if config.is_absolute() else CONFIG.resolve().parents[2] / config
    bridge = CONFIG.resolve().parents[2] / contract["fixed_final"]
    metadata = bridge / "runtime_metadata.json"
    weights = bridge / "bridge.safetensors"
    if (
        config.resolve() != V94_CONFIG.resolve()
        or config.is_symlink()
        or sha256_file_v85(config) != contract["experiment_config_sha256"]
        or bridge.is_symlink()
        or weights.is_symlink()
        or metadata.is_symlink()
        or sha256_file_v85(weights) != contract["bridge_sha256"]
        or sha256_file_v85(metadata) != contract["bridge_metadata_sha256"]
    ):
        raise ValueError("V94 same-row comparator bytes changed")
    return dict(contract)


@hardened_deferred_evaluation_stage_v96(label_process=False)
@torch.inference_mode()
def predict_deferred_final_v96(model: str) -> dict[str, Any]:
    """Run V96 four arms or the exact frozen V94 same-row comparator."""

    if model not in {"v96", "v94"}:
        raise ValueError("V96 deferred predictor model must be v96 or v94")
    outputs = _bundle_paths(model)
    states = [path.exists() or path.is_symlink() for path in outputs]
    if all(states):
        bundle = authenticate_prediction_bundle_v96_final(model)
        return {
            "artifact": (V96_PREDICTION_ARTIFACT if model == "v96" else V94_PREDICTION_ARTIFACT),
            "model": model,
            "row_count": len(bundle["rows"]),
            "prediction_sha256": bundle["prediction_sha256"],
            "reused_authenticated_create_once_bundle": True,
            "runtime_promotion_authorized": False,
        }
    assert_output_bundle_state_v96(outputs, complete=False)

    started = time.monotonic()
    audit = FileAccessAudit(
        prediction_forbidden_roots_v96_final(),
        forbidden_component_names={"oracle"},
        block_forbidden=True,
    )
    with audit:
        fixed = authenticate_fixed_inputs_before_questions_v96_final(audit=audit)
        if model == "v96":
            config = load_config_v96(CONFIG, allow_draft=False)
            stack = load_predictor_stack_v96(
                config,
                expected_candidate_state_sha256=str(fixed.candidate["state_sha256"]),
            )
        else:
            v94_contract = _authenticate_v94_bytes(
                fixed.materialized["preregistration"]
            )
            stack = load_predictor_stack_v94(load_config_v94(V94_CONFIG))
            if stack.candidate.get("state_sha256") != v94_contract["bridge_state_sha256"]:
                raise ValueError("Loaded V94 same-row comparator state changed")
        questions = load_questions_v96_final(fixed)
        provenance = prediction_provenance_v96_final(model=model, fixed=fixed, questions=questions)
        rows: list[dict[str, Any]] = []
        for ordinal, question in enumerate(questions.questions, 1):
            scene = question.scene_id
            if model == "v96":
                predictions = {
                    arm: generate_arm_v96(stack, fixed.memories[arm][scene], question.question)
                    for arm in ARMS
                }
                hashes = {arm: prefix_sha256(fixed.memories[arm][scene]) for arm in ARMS}
                row = {
                    "artifact": V96_PREDICTION_ARTIFACT,
                    "schema_version": SCHEMA_VERSION,
                    "scene_id": scene,
                    "question_id": question.question_id,
                    "paired_scene_id": PAIR_SCENE[scene],
                    **{f"{arm}_prediction": predictions[arm] for arm in ARMS},
                    **{f"{arm}_memory_sha256": hashes[arm] for arm in ARMS},
                    "all_memory_hashes_unchanged": all(
                        hashes[arm] == fixed.memory_hashes[arm][scene] for arm in ARMS
                    ),
                    "provenance_sha256": provenance["provenance_sha256"],
                }
            else:
                prediction = generate_arm_v94(
                    stack, fixed.memories[PRIMARY][scene], question.question
                )
                row = {
                    "artifact": V94_PREDICTION_ARTIFACT,
                    "schema_version": SCHEMA_VERSION,
                    "scene_id": scene,
                    "question_id": question.question_id,
                    "memory_sha256": prefix_sha256(fixed.memories[PRIMARY][scene]),
                    "prediction": prediction,
                    "provenance_sha256": provenance["provenance_sha256"],
                }
            rows.append(row)
            if ordinal == 1 or ordinal % 12 == 0 or ordinal == QUESTION_COUNT:
                print(
                    json.dumps(
                        {
                            "event": f"{model}_deferred_final_prediction",
                            "ordinal": ordinal,
                            "total": QUESTION_COUNT,
                            "scene_id": scene,
                            "question_id": question.question_id,
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()
        after_hashes = {
            arm: {scene: prefix_sha256(fixed.memories[arm][scene]) for scene in fixed.memories[arm]}
            for arm in ARMS
        }
        config = load_config_v96(CONFIG, allow_draft=False)
        candidate_after = authenticate_fixed_final_candidate_v96(
            config, config_path=CONFIG, audit=audit
        )
        if model == "v94":
            _authenticate_v94_bytes(fixed.materialized["preregistration"])
    audit.assert_clean()
    if candidate_after != fixed.candidate or after_hashes != fixed.memory_hashes:
        raise RuntimeError("V96 final predictor mutated candidate or scene memory")
    validate_prediction_rows_v96_final(
        rows,
        model=model,
        fixed=fixed,
        questions=questions,
        provenance_sha256=provenance["provenance_sha256"],
    )
    access = audit_report_v96_final(
        audit,
        question_path=QUESTION_MANIFEST,
        memory_paths=[path / "memory.safetensors" for path in fixed.memory_paths.values()],
    )
    if access["passed"] is not True:
        raise RuntimeError("V96 final predictor file isolation failed")
    predictions_path, provenance_path, access_path, completion_path = outputs
    write_jsonl_create_once_v96(predictions_path, rows)
    write_json_create_once_v96(provenance_path, provenance)
    write_json_create_once_v96(access_path, access)
    completion = {
        "artifact": (PREDICTION_COMPLETION_ARTIFACT if model == "v96" else V94_COMPLETION_ARTIFACT),
        "schema_version": SCHEMA_VERSION,
        "model": model,
        "candidate_fingerprint_before": fixed.candidate["fingerprint_sha256"],
        "candidate_fingerprint_after": candidate_after["fingerprint_sha256"],
        "candidate_attestation_file_sha256": fixed.candidate[
            "attestation_file_sha256"
        ],
        "candidate_attestation_identity_sha256": fixed.candidate[
            "attestation_identity_sha256"
        ],
        "v2_implementation_seal_sha256": fixed.candidate[
            "v2_implementation_seal_sha256"
        ],
        "all_memory_hashes_invariant": True,
        "all_memories_bound_before_questions": True,
        "prediction_sha256": sha256_file_v85(predictions_path),
        "provenance_file_sha256": sha256_file_v85(provenance_path),
        "access_sha256": sha256_file_v85(access_path),
        "row_count": QUESTION_COUNT,
        "elapsed_seconds": time.monotonic() - started,
        "labels_loaded": False,
        "oracle_loaded": False,
        "runtime_promotion_authorized": False,
    }
    write_json_create_once_v96(completion_path, completion)
    bundle = authenticate_prediction_bundle_v96_final(model)
    return {
        "artifact": rows[0]["artifact"],
        "model": model,
        "row_count": len(rows),
        "prediction_sha256": bundle["prediction_sha256"],
        "reused_authenticated_create_once_bundle": False,
        "runtime_promotion_authorized": False,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", choices=("v96", "v94"))
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    print(
        json.dumps(
            predict_deferred_final_v96(args.model),
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["main", "predict_deferred_final_v96"]
