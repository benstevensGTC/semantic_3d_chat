#!/usr/bin/env python3
"""Prepare a sealed numeric-only V82 train or historical-development cache."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch
import yaml
from safetensors import safe_open
from safetensors.torch import load_file

from semantic_3d_chat.chat.question_control_runtime import _load_control_head
from semantic_3d_chat.config import PROJECT_ROOT
from semantic_3d_chat.evaluation.v75_fixed_atlas_artifacts import (
    build_numeric_probe_tensor,
    ordered_probe_questions,
)
from semantic_3d_chat.scene_encoder.fixed_prefix_atlas import tensor_sha256
from semantic_3d_chat.scene_encoder.fixed_prefix_atlas_v75 import (
    compile_fixed_scene_atlas_v75_v2,
)
from semantic_3d_chat.scene_encoder.question_control_v75 import (
    DenseFullSceneContinuousControlV75,
)
from semantic_3d_chat.training.train_question_control_v73 import (
    RowV73,
    load_config_v73,
    load_embedding_assets_v73,
    load_prefixes_v73,
    load_training_rows_v73,
    split_rows_v73,
)
from semantic_3d_chat.training.v82_reader_artifacts import (
    save_v82_cache,
    sha256_file_v82,
)


def _resolve(path: str | Path) -> Path:
    value = Path(path).expanduser()
    return (value if value.is_absolute() else PROJECT_ROOT / value).resolve()


def _load_config(path: str | Path) -> dict[str, Any]:
    source = _resolve(path)
    if not source.is_file() or source.is_symlink():
        raise FileNotFoundError(source)
    payload = yaml.safe_load(source.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping) or set(payload) != {"v82"}:
        raise ValueError("V82 cache preparation config must contain exactly v82")
    config = payload["v82"]
    if not isinstance(config, Mapping) or config.get("schema_version") != 82:
        raise ValueError("V82 cache preparation config changed")
    return dict(config)


def _assert_source(path: Path, digest: str, label: str) -> Path:
    if not path.is_file() or path.is_symlink():
        raise FileNotFoundError(f"V82 {label} is unavailable: {path}")
    observed = sha256_file_v82(path)
    if observed != digest:
        raise ValueError(f"V82 {label} digest changed: {observed}")
    return path


def _load_probe_bank(root: Path, config: Mapping[str, Any]) -> tuple[torch.Tensor, dict[str, Any]]:
    tensor_path = _assert_source(
        root / "probes.safetensors",
        str(config["sources"]["probe_tensor_file_sha256"]),
        "probe tensor",
    )
    metadata_path = _assert_source(
        root / "runtime_metadata.json",
        str(config["sources"]["probe_metadata_sha256"]),
        "probe metadata",
    )
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    required = {
        "artifact": "v75_fixed_atlas_numeric_probe_bank_v1",
        "probe_count": 96,
        "hidden_size": 1536,
        "questions_or_answers_serialized": False,
        "environmental_text_serialized": False,
        "oracle_loaded": False,
        "official_validation_loaded": False,
        "official_test_loaded": False,
        "deferred_final_loaded": False,
    }
    if not isinstance(metadata, Mapping) or any(
        metadata.get(field) != value for field, value in required.items()
    ):
        raise ValueError("V82 source probe metadata contract changed")
    with safe_open(str(tensor_path), framework="pt", device="cpu") as handle:
        if set(handle.keys()) != {"probe_embeddings"}:
            raise ValueError("V82 source probe tensor inventory changed")
    probes = load_file(str(tensor_path), device="cpu")["probe_embeddings"].float()
    if (
        tuple(probes.shape) != (96, 1536)
        or not bool(torch.isfinite(probes).all())
        or bool(torch.any(probes.norm(dim=-1) <= 1e-8))
        or tensor_sha256(probes) != metadata.get("probe_tensor_sha256")
    ):
        raise ValueError("V82 source probe tensor changed")
    return probes.contiguous(), dict(metadata)


def _question_hash_inventory(questions: Sequence[str]) -> str:
    digests = [hashlib.sha256(value.encode("utf-8")).hexdigest() for value in questions]
    return hashlib.sha256(
        json.dumps(digests, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _targets_for_rows(
    rows: Sequence[RowV73],
    *,
    prefixes: Mapping[str, torch.Tensor],
    query_for_question: Mapping[str, torch.Tensor],
    controller: DenseFullSceneContinuousControlV75,
    paired: bool,
    batch_size: int = 48,
) -> torch.Tensor:
    result: list[torch.Tensor] = []
    controller.eval()
    with torch.inference_mode():
        for start in range(0, len(rows), batch_size):
            batch = rows[start : start + batch_size]
            scene_ids = [row.paired_scene_id if paired else row.scene_id for row in batch]
            scenes = torch.cat([prefixes[scene_id] for scene_id in scene_ids]).float()
            queries = torch.stack([query_for_question[row.question] for row in batch]).unsqueeze(1)
            output = controller(scenes, queries.float()).control_tokens
            result.append(output.detach().cpu().to(torch.bfloat16))
    return torch.cat(result)


def prepare_cache(config_path: str | Path, *, split: str, output: str | Path | None) -> dict[str, Any]:
    config = _load_config(config_path)
    sources = config["sources"]
    v73_path = _assert_source(
        _resolve(sources["v73_config"]), sources["v73_config_sha256"], "V73 config"
    )
    v73 = load_config_v73(v73_path)
    qa_path = _assert_source(
        _resolve(sources["historical_qa"]), sources["historical_qa_sha256"], "historical QA"
    )
    all_rows = load_training_rows_v73(qa_path)
    train_rows, held_rows = split_rows_v73(all_rows)
    if split == "train":
        rows = train_rows
        split_role = "historical_optimization_fold"
        expected = (576, 24)
        default_output = config["cache"]["training_output"]
    elif split == "historical-development":
        rows = held_rows
        split_role = "historical_pair_scene_disjoint_development_fold"
        expected = (384, 16)
        default_output = config["cache"]["development_output"]
    else:
        raise ValueError("V82 cache split must be train or historical-development")
    scene_ids = sorted({row.scene_id for row in rows})
    if (len(rows), len(scene_ids)) != expected:
        raise ValueError("V82 selected historical split inventory changed")
    paired_ids = {row.paired_scene_id for row in rows}
    if paired_ids != set(scene_ids):
        raise ValueError("V82 selected split paired-scene inventory changed")

    prefix_root = _resolve(sources["prefix_cache"])
    prefix_manifest = _assert_source(
        prefix_root / "manifest.json",
        sources["prefix_manifest_sha256"],
        "prefix manifest",
    )
    prefixes, _manifest = load_prefixes_v73(prefix_root, scene_ids)
    controller_root = _resolve(sources["v75_controller"])
    controller_weights = _assert_source(
        controller_root / "control.safetensors",
        sources["v75_controller_weights_sha256"],
        "V75 controller weights",
    )
    _assert_source(
        controller_root / "runtime_metadata.json",
        sources["v75_controller_metadata_sha256"],
        "V75 controller metadata",
    )
    controller, _controller_metadata = _load_control_head(
        controller_root, hidden_size=1536, device=torch.device("cpu")
    )
    if type(controller) is not DenseFullSceneContinuousControlV75:
        raise TypeError("V82 cache requires the exact V75 controller")
    probes, probe_metadata = _load_probe_bank(_resolve(sources["probe_bank"]), config)

    # Compile every complete scene memory before binding any row to a query.
    compiled_by_scene: dict[str, torch.Tensor] = {}
    with torch.inference_mode():
        for scene_id in scene_ids:
            compiled = compile_fixed_scene_atlas_v75_v2(
                prefixes[scene_id], controller, probes
            )
            if (
                tuple(compiled.scene_prefix.shape) != (1, 738, 1536)
                or not compiled.audit.every_probe_processed
                or not compiled.audit.complete_atlas_included
                or compiled.audit.question_dependent_retrieval
                or compiled.audit.semantic_or_spatial_top_k_selection
            ):
                raise RuntimeError("V82 source compiler produced an invalid memory")
            compiled_by_scene[scene_id] = compiled.scene_prefix.detach().cpu().contiguous()

    questions = tuple(sorted({row.question for row in rows}))
    if split == "train":
        ordered = ordered_probe_questions(train_rows)
        if set(questions) != set(ordered) or len(questions) != 96:
            raise ValueError("V82 training question inventory changed")
        if _question_hash_inventory(ordered) != probe_metadata.get(
            "source_question_hash_inventory_sha256"
        ):
            raise ValueError("V82 training question/probe opaque ordering changed")
        question_queries = probes.clone()
        query_index = {question: index for index, question in enumerate(ordered)}
        query_for_question = {
            question: question_queries[index]
            for question, index in query_index.items()
        }
    else:
        assets = load_embedding_assets_v73(v73["gemma_snapshot"], questions, {})
        if assets.model_file_sha256 != probe_metadata.get("model_file_sha256"):
            raise ValueError("V82 development query embedding model changed")
        question_queries = build_numeric_probe_tensor(
            questions,
            assets.questions,
        ) if len(questions) == 96 else torch.stack(
            [assets.questions[question].float().mean(dim=0) for question in questions]
        ).contiguous()
        query_index = {question: index for index, question in enumerate(questions)}
        query_for_question = {
            question: question_queries[index]
            for question, index in query_index.items()
        }

    scene_index = {scene_id: index for index, scene_id in enumerate(scene_ids)}
    scene_memories = torch.cat(
        [compiled_by_scene[scene_id].to(torch.bfloat16) for scene_id in scene_ids]
    )
    target_controls = _targets_for_rows(
        rows,
        prefixes=prefixes,
        query_for_question=query_for_question,
        controller=controller,
        paired=False,
    )
    paired_target_controls = _targets_for_rows(
        rows,
        prefixes=prefixes,
        query_for_question=query_for_question,
        controller=controller,
        paired=True,
    )
    tensors = {
        "scene_memories": scene_memories,
        "question_queries": question_queries.float().contiguous(),
        "row_scene_indices": torch.tensor(
            [scene_index[row.scene_id] for row in rows], dtype=torch.int64
        ),
        "row_paired_scene_indices": torch.tensor(
            [scene_index[row.paired_scene_id] for row in rows], dtype=torch.int64
        ),
        "row_query_indices": torch.tensor(
            [query_index[row.question] for row in rows], dtype=torch.int64
        ),
        "row_expected_change": torch.tensor(
            [row.expected_change for row in rows], dtype=torch.bool
        ),
        "target_controls": target_controls,
        "paired_target_controls": paired_target_controls,
    }
    destination = _resolve(output or default_output)
    metadata = save_v82_cache(
        destination,
        tensors,
        split_role=split_role,
        scene_ids=scene_ids,
        source_qa_sha256=sha256_file_v82(qa_path),
        source_v73_config_sha256=sha256_file_v82(v73_path),
        source_prefix_manifest_sha256=sha256_file_v82(prefix_manifest),
        source_controller_sha256=sha256_file_v82(controller_weights),
        source_probe_tensor_sha256=str(probe_metadata["probe_tensor_sha256"]),
    )
    return {
        "phase": "v82_numeric_cache_prepared",
        "output": str(destination),
        "split_role": split_role,
        "scene_count": metadata["scene_count"],
        "row_count": metadata["row_count"],
        "question_query_count": metadata["question_query_count"],
        "changed_row_count": metadata["changed_row_count"],
        "fixed_memory_shape_per_scene": metadata["fixed_memory_shape_per_scene"],
        "questions_or_answers_serialized": False,
        "environmental_text_serialized": False,
        "oracle_serialized": False,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="configs/experiments/gemma4_v82_strict_dense_learned_reader.yaml",
    )
    parser.add_argument(
        "--split", choices=("train", "historical-development"), required=True
    )
    parser.add_argument("--output")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    result = prepare_cache(args.config, split=args.split, output=args.output)
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
