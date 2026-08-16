"""Execute the mechanical V2 smoke amendment and otherwise exact V1 run."""

from __future__ import annotations

import argparse
import json
import math
import time
from collections.abc import Sequence
from typing import Any

import torch

from semantic_3d_chat.evaluation.fixed_prefix_ple_v54_v2_preregistration import (
    ARTIFACT,
    OUTPUT_CHECKPOINT,
    PREREGISTRATION,
    RESULT_REPORT,
    SMOKE_REPORT,
    authenticate_preregistration,
)
from semantic_3d_chat.training import train_fixed_prefix_ple_v54 as v1

RETENTION_SELF_KL_ABSOLUTE_TOLERANCE = 1e-05


def _activate_v2_paths() -> None:
    """Redirect V1's unchanged implementation to unique V2 artifacts in this process."""

    v1.ARTIFACT = ARTIFACT
    v1.PREREGISTRATION = PREREGISTRATION
    v1.SMOKE_REPORT = SMOKE_REPORT
    v1.RESULT_REPORT = RESULT_REPORT
    v1.OUTPUT_CHECKPOINT = OUTPUT_CHECKPOINT
    v1.authenticate_preregistration = authenticate_preregistration


def structural_preflight() -> dict[str, Any]:
    _activate_v2_paths()
    result = v1.structural_preflight()
    result["artifact"] = ARTIFACT
    result["v2_only_change"] = {
        "field": "gradient_smoke.retention_self_kl_absolute_tolerance",
        "value": RETENTION_SELF_KL_ABSOLUTE_TOLERANCE,
    }
    return result


def gradient_smoke() -> dict[str, Any]:
    """Repeat V1's exact microbatch with only the preregistered 1e-5 KL tolerance."""

    _activate_v2_paths()
    if v1._resolve(SMOKE_REPORT).exists():
        raise FileExistsError("PLE-V54 V2 smoke report already exists")
    preflight = structural_preflight()
    if preflight["passed"] is not True:
        raise RuntimeError("PLE-V54 V2 structural preflight failed")
    started = time.perf_counter()
    bundle = v1._load_bundle(gradient_checkpointing=True)
    train = v1.load_training_records()
    row = next(record for record in train if record.changed)
    corpus = v1.load_retention_corpus()
    teachers = v1.retention_baseline(bundle, corpus[:1])
    bundle.installation.train()
    bundle.language.model.zero_grad(set_to_none=True)
    loss, diagnostics = v1.changed_side_loss(bundle, row)
    retention = v1.retention_kl_loss(bundle, corpus[0], teachers[0])
    total = loss + 0.2 * retention
    total.backward()
    gradients = bundle.installation.gradient_norms()
    bundle.installation.validate_state()
    retention_value = float(retention.detach().cpu())
    passed = (
        bool(torch.isfinite(total).item())
        and float(gradients["total_l2"]) > 0.0
        and math.isfinite(diagnostics["margin"])
        and abs(retention_value) <= RETENTION_SELF_KL_ABSOLUTE_TOLERANCE
    )
    report = {
        "schema_version": 1,
        "artifact": f"{ARTIFACT}_gradient_smoke",
        "status": "passed" if passed else "failed",
        "passed": bool(passed),
        "device": str(bundle.language.device),
        "base_dtype": str(next(bundle.language.model.parameters()).dtype),
        "trainable_parameter_count": bundle.installation.parameter_count,
        "initial_adapter_state_sha256": bundle.installation.state_sha256(),
        "loss": float(total.detach().cpu()),
        "correct_answer_nll": diagnostics["correct_nll"],
        "wrong_answer_nll": diagnostics["wrong_nll"],
        "initial_wrong_prefix_margin": diagnostics["margin"],
        "initial_retention_kl": retention_value,
        "retention_self_kl_absolute_tolerance": (
            RETENTION_SELF_KL_ABSOLUTE_TOLERANCE
        ),
        "gradient_l2": gradients["total_l2"],
        "gradient_by_module": gradients["by_module"],
        "elapsed_seconds": time.perf_counter() - started,
        "memory": v1.memory_metrics(),
        "prefix_shape": [1, 258, 1536],
        "question_dependent_scene_retrieval": False,
        "environmental_text_inputs": [],
        "preregistration_sha256": v1.sha256_file(PREREGISTRATION),
        "v1_training_objective_or_gate_changed": False,
    }
    v1._atomic_create_json(SMOKE_REPORT, report)
    return report


def train_and_gate() -> dict[str, Any]:
    """Run the exact V1 training/gating body with unique V2 paths."""

    _activate_v2_paths()
    return v1.train_and_gate()


def authenticate_result() -> dict[str, Any]:
    _activate_v2_paths()
    return v1.authenticate_result()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("preflight", "smoke", "train", "authenticate"))
    mode = parser.parse_args(argv).mode
    result = {
        "preflight": structural_preflight,
        "smoke": gradient_smoke,
        "train": train_and_gate,
        "authenticate": authenticate_result,
    }[mode]()
    print(json.dumps(result, sort_keys=True, allow_nan=False), flush=True)
    return 0 if result.get("passed", True) else 1


if __name__ == "__main__":
    raise SystemExit(main())
