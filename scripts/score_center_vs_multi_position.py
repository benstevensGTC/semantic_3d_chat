#!/usr/bin/env python3
"""Score the existing center-scan versus 96-view multi-position reports.

This is intentionally model-free and report-only.  It opens exactly four
allowlisted measurement reports, authenticates their immutable byte digests,
checks that their scene/model/query contracts match, and writes a deterministic
comparison.  It never opens a voxel map, rendered frame, runtime checkpoint,
oracle, generated-QA file, or scorer reference.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Final

PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT: Final[Path] = Path(
    "reports/gemma4_multi_position/metrics/center_vs_multi_position.json"
)
DEFAULT_INPUTS: Final[dict[str, Path]] = {
    "center_map": Path("reports/gemma4/metrics/map_scene_000001.json"),
    "center_semantic": Path("reports/gemma4/metrics/gemma4_semantic_sanity_scene_000001.json"),
    "multi_position_map": Path("reports/gemma4_multi_position/metrics/map_scene_000001.json"),
    "multi_position_semantic": Path(
        "reports/gemma4_multi_position/metrics/gemma4_semantic_sanity_scene_000001.json"
    ),
}
EXPECTED_INPUT_SHA256: Final[dict[str, str]] = {
    "center_map": "7647b274be75805025837a51bd9c7b52e49ecb43423019bb88f58932df9f5e7d",
    "center_semantic": "ca18ea7c6c52b476f29a330d39437ee702adf9811d7b0313490682768655a1e6",
    "multi_position_map": ("6c1414229d3f0c4d02ebc09a23079b988d6441aba1bb6182568b4ed1b7910d53"),
    "multi_position_semantic": ("1a5194df1014ab9566bbb3dbb3e07c3cbd343bb6acf2688bba134c61bd72a19b"),
}


def _resolve(path: str | Path) -> Path:
    value = Path(path).expanduser()
    return (value if value.is_absolute() else PROJECT_ROOT / value).resolve()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_bytes(value: object) -> bytes:
    return (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _read_authenticated_object(role: str, path: Path) -> dict[str, Any]:
    if role not in EXPECTED_INPUT_SHA256:
        raise ValueError(f"Unknown multi-position input role: {role}")
    source = _resolve(path)
    if not source.is_file() or source.is_symlink():
        raise FileNotFoundError(f"Ablation input is not a regular file: {source}")
    observed = _sha256_file(source)
    if observed != EXPECTED_INPUT_SHA256[role]:
        raise ValueError(
            f"Ablation input digest differs for {role}: "
            f"expected={EXPECTED_INPUT_SHA256[role]} observed={observed}"
        )
    payload = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"Ablation input is not an object: {role}")
    return payload


def _finite_float(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _nonnegative_int(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")
    return value


def _measurement(
    name: str,
    map_report: Mapping[str, Any],
    semantic_report: Mapping[str, Any],
) -> dict[str, Any]:
    if (
        map_report.get("schema_version") != 1
        or map_report.get("phase") != "mapping_complete"
        or semantic_report.get("schema_version") != 1
        or semantic_report.get("phase") != "gemma4_semantic_sanity"
    ):
        raise ValueError(f"{name} report schema or phase differs")
    if map_report.get("scene_id") != "scene_000001" or semantic_report.get(
        "scene_id"
    ) != map_report.get("scene_id"):
        raise ValueError(f"{name} scene identity differs")
    frame_count = _nonnegative_int(map_report.get("frame_count"), name=f"{name} frame_count")
    occupied_voxels = _nonnegative_int(
        map_report.get("occupied_voxels"), name=f"{name} occupied_voxels"
    )
    total_observations = _nonnegative_int(
        map_report.get("total_observations"), name=f"{name} total_observations"
    )
    voxel_count = _nonnegative_int(
        semantic_report.get("voxel_count"), name=f"{name} semantic voxel_count"
    )
    if occupied_voxels < 1 or occupied_voxels != voxel_count:
        raise ValueError(f"{name} map and semantic voxel inventories differ")
    voxel_size = _finite_float(map_report.get("voxel_size_m"), name=f"{name} map voxel_size_m")
    semantic_voxel_size = _finite_float(
        semantic_report.get("voxel_size_m"),
        name=f"{name} semantic voxel_size_m",
    )
    if voxel_size <= 0.0 or voxel_size != semantic_voxel_size:
        raise ValueError(f"{name} voxel size differs")
    if (
        map_report.get("feature_dim") != 3072
        or map_report.get("semantic_dtype_on_disk") != "float16"
        or map_report.get("codec") != "identity-float16"
        or semantic_report.get("cpu_only") is not True
        or semantic_report.get("evaluation_only") is not True
        or semantic_report.get("local_files_only") is not True
    ):
        raise ValueError(f"{name} feature or evaluation contract differs")

    aggregate = semantic_report.get("aggregate")
    consistency = semantic_report.get("same_voxel_consistency")
    if not isinstance(aggregate, Mapping) or not isinstance(consistency, Mapping):
        raise TypeError(f"{name} semantic aggregate or consistency is unavailable")
    if consistency.get("available") is not True:
        raise ValueError(f"{name} consistency measurement is unavailable")
    if (
        _nonnegative_int(consistency.get("frames"), name=f"{name} consistency frames")
        != frame_count
    ):
        raise ValueError(f"{name} consistency frame count differs from map")
    same = consistency.get("same_voxel_similarity")
    different = consistency.get("different_voxel_similarity")
    if not isinstance(same, Mapping) or not isinstance(different, Mapping):
        raise TypeError(f"{name} consistency distributions are unavailable")
    multiview_voxels = _nonnegative_int(
        consistency.get("multiview_voxels"), name=f"{name} multiview_voxels"
    )
    unique_voxels = _nonnegative_int(consistency.get("unique_voxels"), name=f"{name} unique_voxels")
    if unique_voxels != occupied_voxels or multiview_voxels > unique_voxels:
        raise ValueError(f"{name} consistency voxel coverage differs")
    same_mean = _finite_float(same.get("mean"), name=f"{name} same mean")
    different_mean = _finite_float(different.get("mean"), name=f"{name} different mean")
    reported_gap = _finite_float(
        consistency.get("same_minus_different_mean"),
        name=f"{name} same-minus-different",
    )
    if abs(reported_gap - (same_mean - different_mean)) > 1e-12:
        raise ValueError(f"{name} consistency separation differs from means")
    query_count = _nonnegative_int(semantic_report.get("query_count"), name=f"{name} query_count")
    scorable = _nonnegative_int(aggregate.get("scorable_queries"), name=f"{name} scorable_queries")
    unscorable = _nonnegative_int(
        aggregate.get("unscorable_queries"), name=f"{name} unscorable_queries"
    )
    if scorable + unscorable != query_count or scorable < 1:
        raise ValueError(f"{name} semantic query inventory differs")

    return {
        "scan": {
            "frame_count": frame_count,
            "voxel_size_m": voxel_size,
            "feature_dim": 3072,
        },
        "coverage": {
            "occupied_voxels": occupied_voxels,
            "total_observations": total_observations,
            "observations_per_occupied_voxel": total_observations / occupied_voxels,
            "multiview_voxels": multiview_voxels,
            "multiview_voxel_fraction": multiview_voxels / unique_voxels,
            "frame_voxel_observations": _nonnegative_int(
                consistency.get("frame_voxel_observations"),
                name=f"{name} frame_voxel_observations",
            ),
        },
        "semantic_localization": {
            "query_count": query_count,
            "scorable_queries": scorable,
            "top1_localization_accuracy": _finite_float(
                aggregate.get("top1_localization_accuracy"),
                name=f"{name} top1 accuracy",
            ),
            "top_k_localization_accuracy": _finite_float(
                aggregate.get("top_k_localization_accuracy"),
                name=f"{name} top-k accuracy",
            ),
            "mean_precision_at_k": _finite_float(
                aggregate.get("mean_precision_at_k"), name=f"{name} P@k"
            ),
            "mean_random_precision_at_k": _finite_float(
                aggregate.get("mean_random_precision_at_k"),
                name=f"{name} random P@k",
            ),
            "ranking_k": _nonnegative_int(semantic_report.get("top_k"), name=f"{name} ranking k"),
        },
        "view_consistency": {
            "same_voxel_mean_cosine": same_mean,
            "same_voxel_count": _nonnegative_int(same.get("count"), name=f"{name} same count"),
            "same_voxel_pair_count": _nonnegative_int(
                consistency.get("same_voxel_pair_count"),
                name=f"{name} same pair count",
            ),
            "different_voxel_mean_cosine": different_mean,
            "different_voxel_count": _nonnegative_int(
                different.get("count"), name=f"{name} different count"
            ),
            "same_minus_different_mean_cosine": reported_gap,
        },
    }


def _delta(multi_position: float, center: float) -> float | int:
    return multi_position - center


def build_comparison(
    *,
    center_map_path: str | Path = DEFAULT_INPUTS["center_map"],
    center_semantic_path: str | Path = DEFAULT_INPUTS["center_semantic"],
    multi_position_map_path: str | Path = DEFAULT_INPUTS["multi_position_map"],
    multi_position_semantic_path: str | Path = DEFAULT_INPUTS["multi_position_semantic"],
) -> dict[str, Any]:
    """Authenticate the four reports and return a deterministic comparison."""

    paths = {
        "center_map": Path(center_map_path),
        "center_semantic": Path(center_semantic_path),
        "multi_position_map": Path(multi_position_map_path),
        "multi_position_semantic": Path(multi_position_semantic_path),
    }
    reports = {role: _read_authenticated_object(role, path) for role, path in paths.items()}
    center_map = reports["center_map"]
    center_semantic = reports["center_semantic"]
    multi_map = reports["multi_position_map"]
    multi_semantic = reports["multi_position_semantic"]
    center = _measurement("center", center_map, center_semantic)
    multi = _measurement("multi_position", multi_map, multi_semantic)

    shared_semantic_fields = (
        "scene_id",
        "voxel_size_m",
        "query_count",
        "top_k",
        "vision_model",
        "vision_revision",
        "language_embedding_model",
        "language_embedding_revision",
        "text_embedding_method",
        "feature_layout",
    )
    if any(
        center_semantic.get(field) != multi_semantic.get(field) for field in shared_semantic_fields
    ):
        raise ValueError("Center and multi-position semantic contracts differ")
    center_queries = center_semantic.get("queries")
    multi_queries = multi_semantic.get("queries")
    if not isinstance(center_queries, list) or not isinstance(multi_queries, list):
        raise TypeError("Semantic reports do not contain query inventories")
    center_query_ids = [
        row.get("query_id") if isinstance(row, Mapping) else None for row in center_queries
    ]
    multi_query_ids = [
        row.get("query_id") if isinstance(row, Mapping) else None for row in multi_queries
    ]
    if (
        center_query_ids != multi_query_ids
        or len(center_query_ids) != center["semantic_localization"]["query_count"]
        or len(set(center_query_ids)) != len(center_query_ids)
        or not all(isinstance(value, str) and value for value in center_query_ids)
    ):
        raise ValueError("Center and multi-position semantic query inventories differ")
    if center["scan"]["frame_count"] != 24 or multi["scan"]["frame_count"] != 96:
        raise ValueError("Expected the authenticated 24-view and 96-view scans")

    center_coverage = center["coverage"]
    multi_coverage = multi["coverage"]
    center_semantics = center["semantic_localization"]
    multi_semantics = multi["semantic_localization"]
    center_consistency = center["view_consistency"]
    multi_consistency = multi["view_consistency"]
    deltas = {
        "scan": {
            "frame_count": _delta(multi["scan"]["frame_count"], center["scan"]["frame_count"]),
            "frame_count_ratio": multi["scan"]["frame_count"] / center["scan"]["frame_count"],
        },
        "coverage": {
            "occupied_voxels": _delta(
                multi_coverage["occupied_voxels"],
                center_coverage["occupied_voxels"],
            ),
            "occupied_voxel_relative_change": (
                multi_coverage["occupied_voxels"] / center_coverage["occupied_voxels"] - 1.0
            ),
            "total_observations": _delta(
                multi_coverage["total_observations"],
                center_coverage["total_observations"],
            ),
            "observations_per_occupied_voxel": _delta(
                multi_coverage["observations_per_occupied_voxel"],
                center_coverage["observations_per_occupied_voxel"],
            ),
            "multiview_voxels": _delta(
                multi_coverage["multiview_voxels"],
                center_coverage["multiview_voxels"],
            ),
            "multiview_voxel_fraction": _delta(
                multi_coverage["multiview_voxel_fraction"],
                center_coverage["multiview_voxel_fraction"],
            ),
            "frame_voxel_observations": _delta(
                multi_coverage["frame_voxel_observations"],
                center_coverage["frame_voxel_observations"],
            ),
        },
        "semantic_localization": {
            "top1_localization_accuracy": _delta(
                multi_semantics["top1_localization_accuracy"],
                center_semantics["top1_localization_accuracy"],
            ),
            "top_k_localization_accuracy": _delta(
                multi_semantics["top_k_localization_accuracy"],
                center_semantics["top_k_localization_accuracy"],
            ),
            "mean_precision_at_k": _delta(
                multi_semantics["mean_precision_at_k"],
                center_semantics["mean_precision_at_k"],
            ),
            "mean_random_precision_at_k": _delta(
                multi_semantics["mean_random_precision_at_k"],
                center_semantics["mean_random_precision_at_k"],
            ),
        },
        "view_consistency": {
            "same_voxel_mean_cosine": _delta(
                multi_consistency["same_voxel_mean_cosine"],
                center_consistency["same_voxel_mean_cosine"],
            ),
            "different_voxel_mean_cosine": _delta(
                multi_consistency["different_voxel_mean_cosine"],
                center_consistency["different_voxel_mean_cosine"],
            ),
            "same_minus_different_mean_cosine": _delta(
                multi_consistency["same_minus_different_mean_cosine"],
                center_consistency["same_minus_different_mean_cosine"],
            ),
            "same_voxel_pair_count": _delta(
                multi_consistency["same_voxel_pair_count"],
                center_consistency["same_voxel_pair_count"],
            ),
        },
    }
    body: dict[str, Any] = {
        "schema": "semantic_3d_chat.center_vs_multi_position.v1",
        "artifact": "authenticated_report_only_scan_ablation",
        "scene_id": "scene_000001",
        "input_artifacts": {
            role: {
                "path": path.as_posix(),
                "sha256": EXPECTED_INPUT_SHA256[role],
            }
            for role, path in paths.items()
        },
        "contract": {
            "model_free": True,
            "report_only": True,
            "data_input_allowlist_exact": sorted(paths),
            "runtime_loaded": False,
            "voxel_maps_loaded": False,
            "rendered_frames_loaded": False,
            "model_checkpoint_loaded": False,
            "oracle_loaded": False,
            "qa_loaded": False,
            "protected_references_loaded": False,
            "same_scene": True,
            "same_voxel_size": True,
            "same_feature_layout": True,
            "same_vision_and_text_embedding_revision": True,
            "same_query_id_inventory": True,
            "question_count": len(center_query_ids),
            "ranking_k": center_semantics["ranking_k"],
        },
        "measurements": {
            "center_24_view": center,
            "multi_position_96_view": multi,
        },
        "delta_multi_position_minus_center": deltas,
        "directional_summary": {
            "occupied_voxel_coverage_increased": deltas["coverage"]["occupied_voxels"] > 0,
            "multiview_voxel_fraction_increased": deltas["coverage"]["multiview_voxel_fraction"]
            > 0.0,
            "top1_localization_improved": deltas["semantic_localization"][
                "top1_localization_accuracy"
            ]
            > 0.0,
            "top_k_localization_improved": deltas["semantic_localization"][
                "top_k_localization_accuracy"
            ]
            > 0.0,
            "precision_at_k_improved": deltas["semantic_localization"]["mean_precision_at_k"] > 0.0,
            "same_voxel_consistency_improved": deltas["view_consistency"]["same_voxel_mean_cosine"]
            > 0.0,
            "same_vs_different_separation_improved": deltas["view_consistency"][
                "same_minus_different_mean_cosine"
            ]
            > 0.0,
            "interpretation": (
                "The 96-view scan increased geometric and repeated-view coverage, "
                "but this single-scene report comparison did not improve top-1, "
                "P@k, same-voxel cosine, or same-vs-different separation; top-k "
                "was unchanged. It is an ablation measurement, not evidence that "
                "more views generally harm semantics."
            ),
        },
        "limitations": [
            "single deterministic development scene",
            "more camera positions and four-times as many views change together",
            "visible-surface coverage is proxied by occupied and multiview voxels",
            "no downstream language-model QA or navigation was run for this ablation",
        ],
        "scorer_source": "scripts/score_center_vs_multi_position.py",
        "scorer_source_sha256": _sha256_file(Path(__file__).resolve()),
    }
    return {**body, "artifact_sha256": _canonical_sha256(body)}


def _write_create_or_verify(path: Path, payload: Mapping[str, Any]) -> str:
    destination = _resolve(path)
    data = (
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    if destination.exists():
        if not destination.is_file() or destination.is_symlink():
            raise FileExistsError(f"Ablation output is not a regular file: {destination}")
        if destination.read_bytes() != data:
            raise FileExistsError(f"Ablation output exists with different bytes: {destination}")
        return "verified_existing"
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return "created"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--center-map", default=str(DEFAULT_INPUTS["center_map"]))
    parser.add_argument("--center-semantic", default=str(DEFAULT_INPUTS["center_semantic"]))
    parser.add_argument("--multi-position-map", default=str(DEFAULT_INPUTS["multi_position_map"]))
    parser.add_argument(
        "--multi-position-semantic",
        default=str(DEFAULT_INPUTS["multi_position_semantic"]),
    )
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    payload = build_comparison(
        center_map_path=args.center_map,
        center_semantic_path=args.center_semantic,
        multi_position_map_path=args.multi_position_map,
        multi_position_semantic_path=args.multi_position_semantic,
    )
    state = _write_create_or_verify(Path(args.output), payload)
    print(
        json.dumps(
            {
                "artifact_sha256": payload["artifact_sha256"],
                "output": str(_resolve(args.output)),
                "state": state,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "DEFAULT_INPUTS",
    "DEFAULT_OUTPUT",
    "EXPECTED_INPUT_SHA256",
    "build_comparison",
    "main",
]
