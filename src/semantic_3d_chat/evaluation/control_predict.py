"""Generate question-independent predictions for continuous-scene controls.

This module reads only a strict questions-only manifest. It never opens held-out
QA supervision and cannot accept reference answers, oracle fields, or targets.
Every scene/condition prefix is built once before its first question and then
reused unchanged.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Final, Protocol

import numpy as np
import torch

from semantic_3d_chat.chat.runtime import StaticChatRuntime
from semantic_3d_chat.config import (
    default_checkpoint_path,
    load_config,
    project_path,
    reports_root,
)
from semantic_3d_chat.evaluation.ablations import deterministic_permutation, file_sha256
from semantic_3d_chat.evaluation.prediction_artifacts import (
    AtomicPredictionJournal,
    PredictionProvenance,
    build_prediction_provenance,
)
from semantic_3d_chat.evaluation.question_manifest import (
    QuestionManifest,
    load_question_manifest,
    validate_question_manifest,
)
from semantic_3d_chat.language.prefix_injection import prefix_sha256
from semantic_3d_chat.scene_encoder.map_io import MapTensorData, load_map_tensors

CONTROL_CONDITIONS: Final[tuple[str, ...]] = (
    "primary",
    "empty_scene_prefix",
    "wrong_scene_prefix",
    "semantic_shuffle",
    "position_shuffle",
    "geometry_only",
    "semantics_without_xyz",
    "remove_rgb",
    "remove_normals",
)

MAP_CONTROL_CONDITIONS: Final[frozenset[str]] = frozenset(
    {
        "semantic_shuffle",
        "position_shuffle",
        "geometry_only",
        "semantics_without_xyz",
        "remove_rgb",
        "remove_normals",
    }
)


class AnswerRuntime(Protocol):
    """The small runtime surface used by the control driver and its tests."""

    scene_prefix: torch.Tensor
    scene_prefix_hash: str

    def answer(self, question: str) -> Any: ...

    def assert_prefix_unchanged(self) -> None: ...


@dataclass
class BuiltControlRuntime:
    runtime: AnswerRuntime
    prefix_source_scene_id: str
    metadata: dict[str, Any]
    release: Callable[[], None] | None = None

    def close(self) -> None:
        if self.release is not None:
            self.release()


RuntimeBuilder = Callable[[str, str, str], BuiltControlRuntime]


def _clone_map(data: MapTensorData) -> MapTensorData:
    return replace(
        data,
        semantic=data.semantic.clone(),
        xyz=data.xyz.clone(),
        rgb=data.rgb.clone(),
        normal=data.normal.clone(),
        confidence=data.confidence.clone(),
        observation_count=data.observation_count.clone(),
        room_min=data.room_min.clone(),
        room_max=data.room_max.clone(),
    )


def _control_seed(seed: int, scene_id: str, condition: str) -> int:
    if seed < 0:
        raise ValueError("Control seed must be non-negative")
    payload = f"{seed}:{scene_id}:{condition}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little")


def apply_map_control(
    data: MapTensorData,
    condition: str,
    *,
    seed: int,
    scene_id: str,
) -> tuple[MapTensorData, dict[str, Any]]:
    """Apply an in-memory control after deterministic map coarsening.

    Row count and feature dimensionality are invariant.  Consequently every
    coarsened input voxel still reaches the global scene encoder, while the raw
    high-resolution map on disk remains unchanged.
    """

    if condition not in MAP_CONTROL_CONDITIONS:
        raise ValueError(f"{condition!r} is not a map-level control")
    controlled = _clone_map(data)
    affected_fields: list[str]
    permutation_hash: str | None = None
    derived_seed = _control_seed(seed, scene_id, condition)

    if condition in {"semantic_shuffle", "position_shuffle"}:
        permutation = deterministic_permutation(controlled.voxel_count, derived_seed)
        permutation_hash = hashlib.sha256(
            np.asarray(permutation, dtype="<i8").tobytes()
        ).hexdigest()
        indices = torch.from_numpy(permutation).to(controlled.xyz.device)
    if condition == "semantic_shuffle":
        controlled.semantic = controlled.semantic.index_select(0, indices)
        affected_fields = ["semantic"]
    elif condition == "position_shuffle":
        controlled.xyz = controlled.xyz.index_select(0, indices)
        affected_fields = ["xyz"]
    elif condition == "geometry_only":
        controlled.semantic.zero_()
        affected_fields = ["semantic"]
    elif condition == "semantics_without_xyz":
        center = (controlled.room_min + controlled.room_max) * 0.5
        controlled.xyz.copy_(center.expand_as(controlled.xyz))
        affected_fields = ["xyz"]
    elif condition == "remove_rgb":
        controlled.rgb.zero_()
        affected_fields = ["rgb"]
    elif condition == "remove_normals":
        controlled.normal.zero_()
        affected_fields = ["normal"]
    else:  # pragma: no cover - exhaustive guard above makes this unreachable.
        raise AssertionError(condition)

    tensors = (
        controlled.semantic,
        controlled.xyz,
        controlled.rgb,
        controlled.normal,
        controlled.confidence,
        controlled.observation_count,
    )
    if any(not torch.isfinite(value).all() for value in tensors):
        raise RuntimeError(f"Control {condition} produced NaN or infinity")
    metadata = {
        "condition": condition,
        "scope": "coarsened_full_scene_map",
        "source_voxel_count": int(data.source_voxel_count),
        "processed_voxel_count": int(data.voxel_count),
        "feature_dim": int(data.feature_dim),
        "affected_fields": affected_fields,
        "base_seed": int(seed),
        "derived_seed": int(derived_seed),
        "permutation_sha256": permutation_hash,
        "question_dependent_selection": False,
    }
    return controlled, metadata


def deterministic_wrong_scene_sources(scene_ids: Sequence[str]) -> dict[str, str]:
    """Return a cyclic derangement over sorted opaque scene IDs."""

    ordered = sorted(set(scene_ids))
    if len(ordered) < 2:
        raise ValueError("wrong_scene_prefix requires at least two distinct scenes")
    return {
        scene_id: ordered[(index + 1) % len(ordered)]
        for index, scene_id in enumerate(ordered)
    }


def _zero_runtime_scene_memory(runtime: StaticChatRuntime) -> None:
    """Remove all continuous scene signal from language and grounding paths."""

    if runtime.scene_prefix.ndim != 3 or runtime.scene_prefix.shape[1] < 3:
        raise ValueError("Scene prefix must contain start, latent, and end tokens")
    # BOI/EOI (or learned start/end) are protocol delimiters, not environment
    # content.  Gemma's native prefix backend authenticates their exact frozen
    # embeddings, so an empty-scene control must preserve those delimiters and
    # zero only the environment-conditioned latent slots between them.
    empty_prefix = runtime.scene_prefix.detach().clone()
    empty_prefix[:, 1:-1].zero_()
    runtime.scene_prefix = empty_prefix
    runtime.scene_output = replace(
        runtime.scene_output,
        scene_tokens=torch.zeros_like(runtime.scene_output.scene_tokens),
        native_latents=torch.zeros_like(runtime.scene_output.native_latents),
        block_tokens=torch.zeros_like(runtime.scene_output.block_tokens),
    )
    runtime.scene_prefix_hash = prefix_sha256(runtime.scene_prefix)


class SharedControlRuntimeFactory:
    """Load model weights once and build one complete prefix per control scene."""

    def __init__(
        self,
        config: dict[str, Any],
        checkpoint: str | Path,
        bootstrap_scene_id: str,
        *,
        seed: int = 20260808,
    ) -> None:
        bootstrap = StaticChatRuntime.load(
            config,
            bootstrap_scene_id,
            checkpoint=checkpoint,
            local_files_only=True,
        )
        self.config = config
        self.checkpoint_path = bootstrap.checkpoint_path
        self.checkpoint_metadata = bootstrap.checkpoint_metadata
        self.language = bootstrap.language
        self.scene_model = bootstrap.scene_model
        # Controls must use the exact trained scene path that produced the
        # primary prefix.  Omitting any residual/sidecar here silently turns a
        # control into a different checkpoint rather than a changed input.
        self.dense_aligner = bootstrap.dense_aligner
        self.dense_sidecar_adapter = bootstrap.dense_sidecar_adapter
        self.block_cross_residual = bootstrap.block_cross_residual
        self.global_scene_residual = bootstrap.global_scene_residual
        self.signed_x_scene_residual = bootstrap.signed_x_scene_residual
        self.composer = bootstrap.composer
        self.grounding = bootstrap.grounding
        self.warnings = bootstrap.warnings
        self.seed = seed
        self._bootstrap_scene_id = bootstrap_scene_id
        self._bootstrap_runtime: StaticChatRuntime | None = bootstrap

    def _load_map(self, scene_id: str) -> MapTensorData:
        map_path = project_path(self.config, "maps", scene_id, "voxel_map.npz").resolve()
        if "oracle" in {part.casefold() for part in map_path.parts}:
            raise ValueError(f"Control runtime cannot load an oracle path: {map_path}")
        map_data = load_map_tensors(
            map_path,
            self.config["scene"]["room_size_m"],
            device="cpu",
            input_voxel_size_m=self.config["scene_encoder"].get("input_voxel_size_m"),
        )
        expected = int(self.checkpoint_metadata["semantic_dim"])
        if map_data.feature_dim != expected:
            raise ValueError(
                f"Scene {scene_id} feature dimension {map_data.feature_dim} != {expected}"
            )
        return map_data.to(self.language.device)

    def _build_from_map(self, scene_id: str, map_data: MapTensorData) -> StaticChatRuntime:
        return StaticChatRuntime(
            config=self.config,
            scene_id=scene_id,
            checkpoint_path=self.checkpoint_path,
            checkpoint_metadata=self.checkpoint_metadata,
            language=self.language,
            map_data=map_data,
            scene_model=self.scene_model,
            dense_aligner=self.dense_aligner,
            dense_sidecar_adapter=self.dense_sidecar_adapter,
            block_cross_residual=self.block_cross_residual,
            global_scene_residual=self.global_scene_residual,
            signed_x_scene_residual=self.signed_x_scene_residual,
            composer=self.composer,
            grounding=self.grounding,
            warnings=self.warnings,
        )

    def build(
        self, condition: str, target_scene_id: str, prefix_source_scene_id: str
    ) -> BuiltControlRuntime:
        if condition not in CONTROL_CONDITIONS:
            raise ValueError(f"Unknown control condition: {condition}")
        source_scene_id = (
            prefix_source_scene_id if condition == "wrong_scene_prefix" else target_scene_id
        )
        use_bootstrap = (
            condition == "primary"
            and source_scene_id == self._bootstrap_scene_id
            and self._bootstrap_runtime is not None
        )
        if use_bootstrap:
            runtime = self._bootstrap_runtime
            self._bootstrap_runtime = None
            assert runtime is not None
            metadata: dict[str, Any] = {
                "condition": condition,
                "scope": "coarsened_full_scene_map",
                "source_voxel_count": runtime.map_data.source_voxel_count,
                "processed_voxel_count": runtime.map_data.voxel_count,
                "feature_dim": runtime.map_data.feature_dim,
                "affected_fields": [],
                "question_dependent_selection": False,
            }
        else:
            map_data = self._load_map(source_scene_id)
            if condition in MAP_CONTROL_CONDITIONS:
                map_data, metadata = apply_map_control(
                    map_data,
                    condition,
                    seed=self.seed,
                    scene_id=target_scene_id,
                )
            else:
                metadata = {
                    "condition": condition,
                    "scope": (
                        "coarsened_full_scene_map"
                        if condition == "primary"
                        else "continuous_scene_prefix"
                    ),
                    "source_voxel_count": map_data.source_voxel_count,
                    "processed_voxel_count": map_data.voxel_count,
                    "feature_dim": map_data.feature_dim,
                    "affected_fields": (
                        ["scene_prefix_latents", "grounding_latents"]
                        if condition == "empty_scene_prefix"
                        else []
                    ),
                    "question_dependent_selection": False,
                }
            runtime = self._build_from_map(source_scene_id, map_data)
        if condition == "empty_scene_prefix":
            _zero_runtime_scene_memory(runtime)
        if condition == "wrong_scene_prefix":
            metadata["wrong_scene_pair"] = {
                "question_scene_id": target_scene_id,
                "prefix_source_scene_id": source_scene_id,
            }
        runtime.assert_prefix_unchanged()
        return BuiltControlRuntime(
            runtime=runtime,
            prefix_source_scene_id=source_scene_id,
            metadata=metadata,
        )

    def close(self) -> None:
        self._bootstrap_runtime = None
        gc.collect()
        if self.language.device.type == "mps":
            torch.mps.empty_cache()


def _prediction_record(
    condition: str,
    target_scene_id: str,
    question_id: str,
    built: BuiltControlRuntime,
    answer: Any,
) -> dict[str, Any]:
    return {
        "scene_id": target_scene_id,
        "question_id": question_id,
        "predicted_answer": answer.answer,
        "grounding_xyz": list(answer.grounding_xyz_m),
        "grounding_confidence": float(answer.grounding_confidence),
        "prefix_hash": answer.prefix_hash,
        "prefix_source_scene_id": built.prefix_source_scene_id,
        "condition": condition,
        "generated_tokens": int(answer.generated_tokens),
        "elapsed_seconds": float(answer.elapsed_seconds),
    }


def run_control_suite(
    question_manifest: QuestionManifest,
    *,
    runtime_builder: RuntimeBuilder,
    output_directory: str | Path,
    conditions: Sequence[str] = CONTROL_CONDITIONS,
    seed: int = 20260808,
    max_questions_per_scene: int | None = None,
    wrong_scene_sources: Mapping[str, str] | None = None,
    force: bool = False,
    prediction_provenance: Mapping[str, PredictionProvenance] | None = None,
) -> dict[str, Any]:
    """Run controls and emit one metrics-compatible JSONL file per condition."""

    if not conditions or len(set(conditions)) != len(conditions):
        raise ValueError("Control conditions must be a non-empty unique sequence")
    if seed < 0:
        raise ValueError("Control seed must be non-negative")
    unknown = sorted(set(conditions) - set(CONTROL_CONDITIONS))
    if unknown:
        raise ValueError(f"Unknown control conditions: {unknown}")
    if max_questions_per_scene is not None and max_questions_per_scene < 1:
        raise ValueError("max_questions_per_scene must be positive")
    question_manifest = validate_question_manifest(question_manifest)
    by_scene = question_manifest.by_scene()
    scene_ids = sorted(by_scene)
    if "wrong_scene_prefix" in conditions:
        wrong_sources = (
            dict(wrong_scene_sources)
            if wrong_scene_sources is not None
            else deterministic_wrong_scene_sources(scene_ids)
        )
        if set(wrong_sources) != set(scene_ids):
            raise ValueError("Wrong-scene mapping must define every question scene exactly once")
        if any(target == source for target, source in wrong_sources.items()):
            raise ValueError("Wrong-scene mapping cannot map a scene to itself")
    else:
        wrong_sources = {}

    output_root = Path(output_directory).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    manifest_path = output_root / "manifest.json"
    output_paths = {condition: output_root / f"{condition}.jsonl" for condition in conditions}
    if prediction_provenance is None:
        existing = [path for path in [manifest_path, *output_paths.values()] if path.exists()]
        if existing and not force:
            raise FileExistsError(f"Refusing to overwrite control results: {existing}")
    elif set(prediction_provenance) != set(conditions):
        raise ValueError("Prediction provenance must define every control condition exactly")

    started = time.perf_counter()
    condition_reports: dict[str, Any] = {}
    for condition in conditions:
        output_path = output_paths[condition]
        temporary = output_path.with_name(f".{output_path.name}.{os.getpid()}.tmp")
        journal = (
            AtomicPredictionJournal(
                output_path,
                prediction_provenance[condition],
                resume=not force,
            )
            if prediction_provenance is not None
            else None
        )
        resumed_count = len(journal.records) if journal is not None else 0
        count = resumed_count
        scene_reports: dict[str, Any] = {}
        condition_started = time.perf_counter()
        handle = None
        try:
            if journal is None:
                handle = temporary.open("w", encoding="utf-8")
            for target_scene_id in scene_ids:
                prefix_source = wrong_sources.get(target_scene_id, target_scene_id)
                built = runtime_builder(condition, target_scene_id, prefix_source)
                runtime = built.runtime
                prefix_hash = runtime.scene_prefix_hash
                runtime.assert_prefix_unchanged()
                selected = by_scene[target_scene_id]
                if max_questions_per_scene is not None:
                    selected = selected[:max_questions_per_scene]
                cached_hashes = (
                    {
                        str(item["prefix_hash"])
                        for item in journal.records
                        if item.get("scene_id") == target_scene_id
                        and item.get("prefix_hash") is not None
                    }
                    if journal is not None
                    else set()
                )
                if cached_hashes and cached_hashes != {prefix_hash}:
                    raise RuntimeError(
                        f"Cached prefix hash differs for {condition}/{target_scene_id}"
                    )
                scene_reports[target_scene_id] = {
                    "prefix_hash": prefix_hash,
                    "prefix_shape": list(runtime.scene_prefix.shape),
                    "prefix_source_scene_id": built.prefix_source_scene_id,
                    "question_count": len(selected),
                    "prefix_built_before_questions": True,
                    "metadata": built.metadata,
                }
                try:
                    for record in selected:
                        if journal is not None and journal.contains(
                            target_scene_id, record.question_id
                        ):
                            continue
                        answer = runtime.answer(record.question)
                        if answer.prefix_hash != prefix_hash:
                            raise RuntimeError(
                                f"Prefix changed during {condition}/{target_scene_id}"
                            )
                        prediction = _prediction_record(
                            condition,
                            target_scene_id,
                            record.question_id,
                            built,
                            answer,
                        )
                        if journal is not None:
                            journal.append(prediction)
                        else:
                            assert handle is not None
                            handle.write(
                                json.dumps(prediction, sort_keys=True, allow_nan=False) + "\n"
                            )
                            handle.flush()
                        count += 1
                    runtime.assert_prefix_unchanged()
                finally:
                    built.close()
                    del runtime, built
                    gc.collect()
                    if torch.backends.mps.is_available():
                        torch.mps.empty_cache()
            if handle is not None:
                handle.close()
                handle = None
                os.replace(temporary, output_path)
        finally:
            if handle is not None:
                handle.close()
            temporary.unlink(missing_ok=True)
        condition_reports[condition] = {
            "path": str(output_path),
            "sha256": file_sha256(output_path),
            "prediction_count": count,
            "resumed_prediction_count": resumed_count,
            "new_prediction_count": count - resumed_count,
            "prediction_provenance_sha256": (
                journal.provenance.sha256 if journal is not None else None
            ),
            "scene_count": len(scene_reports),
            "elapsed_seconds": time.perf_counter() - condition_started,
            "scenes": scene_reports,
        }

    manifest = {
        "schema_version": 1,
        "seed": int(seed),
        "conditions": condition_reports,
        "scene_count": len(scene_ids),
        "question_dependent_retrieval": False,
        "one_prefix_per_scene_condition": True,
        "questions": {
            "manifest_path": (
                str(question_manifest.manifest_path)
                if question_manifest.manifest_path is not None
                else None
            ),
            "manifest_sha256": question_manifest.manifest_sha256,
            "questions_sha256": question_manifest.questions_sha256,
            "source_qa_sha256": question_manifest.source_qa_sha256,
            "question_count": question_manifest.question_count,
        },
        "elapsed_seconds": time.perf_counter() - started,
    }
    temporary_manifest = manifest_path.with_name(f".{manifest_path.name}.{os.getpid()}.tmp")
    try:
        temporary_manifest.write_text(
            json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary_manifest, manifest_path)
    finally:
        temporary_manifest.unlink(missing_ok=True)
    return {**manifest, "manifest_path": str(manifest_path)}


def _parse_wrong_scene_pairs(values: Sequence[str]) -> dict[str, str] | None:
    if not values:
        return None
    result: dict[str, str] = {}
    for value in values:
        target, separator, source = value.partition("=")
        if not separator or not target or not source:
            raise ValueError(f"Expected TARGET=SOURCE, got {value!r}")
        if target in result:
            raise ValueError(f"Duplicate wrong-scene target: {target}")
        result[target] = source
    return result


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--split", default="test", choices=("train", "validation", "test"))
    parser.add_argument(
        "--questions-manifest",
        type=Path,
        help="Strict manifest produced by semantic_3d_chat.evaluation.prepare_questions",
    )
    parser.add_argument("--checkpoint")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--condition", action="append", choices=CONTROL_CONDITIONS)
    parser.add_argument("--seed", type=int, default=20260808)
    parser.add_argument("--max-questions-per-scene", type=int)
    parser.add_argument(
        "--wrong-scene-pair",
        action="append",
        default=[],
        metavar="TARGET=SOURCE",
        help="Override the default cyclic wrong-scene pairing; repeat for every scene.",
    )
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Discard compatible cached predictions instead of resuming them",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    config = load_config(args.config)
    questions_path = (
        args.questions_manifest
        or reports_root(config) / "questions" / f"{args.split}.json"
    )
    question_manifest = load_question_manifest(questions_path)
    scene_ids = sorted({record.scene_id for record in question_manifest.questions})
    checkpoint = Path(
        args.checkpoint or default_checkpoint_path(config)
    ).expanduser().resolve()
    selected_conditions = tuple(args.condition or CONTROL_CONDITIONS)
    wrong_scene_pairs = _parse_wrong_scene_pairs(args.wrong_scene_pair)
    run_details = {
        "seed": args.seed,
        "max_questions_per_scene": args.max_questions_per_scene,
        "wrong_scene_pairs": wrong_scene_pairs,
    }
    provenance = {
        condition: build_prediction_provenance(
            config,
            config_path=args.config,
            checkpoint_path=checkpoint,
            references_path=question_manifest.manifest_path,
            scene_ids=scene_ids,
            split=args.split,
            run_kind="continuous_scene_control",
            condition=json.dumps(
                {"condition": condition, **run_details},
                sort_keys=True,
                separators=(",", ":"),
            ),
        )
        for condition in selected_conditions
    }
    factory = SharedControlRuntimeFactory(
        config,
        checkpoint,
        scene_ids[0],
        seed=args.seed,
    )
    output = (
        args.output_dir
        or reports_root(config) / "predictions" / "controls" / args.split
    )
    try:
        manifest = run_control_suite(
            question_manifest,
            runtime_builder=factory.build,
            output_directory=output,
            conditions=selected_conditions,
            seed=args.seed,
            max_questions_per_scene=args.max_questions_per_scene,
            wrong_scene_sources=wrong_scene_pairs,
            force=args.force or args.no_resume,
            prediction_provenance=provenance,
        )
    finally:
        factory.close()
    print(json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
