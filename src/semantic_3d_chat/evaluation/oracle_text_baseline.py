"""Evaluation-only oracle-text upper bound using the local causal LM.

This module is intentionally below :mod:`semantic_3d_chat.evaluation`.  The
primary chat package never imports it.  Loading object labels here is permitted
only because this is the explicitly prohibited oracle-text upper-bound control.
"""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import torch

from semantic_3d_chat.config import PROJECT_ROOT, load_config
from semantic_3d_chat.evaluation.baseline_io import (
    atomic_write_jsonl,
    indexed_records,
    read_jsonl,
    sha256_file,
    text_fingerprint,
)
from semantic_3d_chat.language.local_lm import load_local_language_model

TextAnswerer = Callable[[str, str], str]


def _number(value: float) -> str:
    return f"{float(value):.3f}"


def oracle_scene_text(oracle: dict[str, Any]) -> str:
    """Serialize exact oracle geometry for the prohibited text upper bound."""
    instances = list(oracle.get("instances", []))
    by_id = {str(item["instance_id"]): item for item in instances}
    lines = ["Coordinates are meters in world axes X=right, Y=forward, Z=up."]
    object_counts = Counter(
        str(item["category"]) for item in instances if item["kind"] == "object"
    )
    lines.append(
        "Exact object category counts: "
        + "; ".join(f"{category}={count}" for category, count in sorted(object_counts.items()))
        + "."
    )

    def append_instances(title: str, selected: list[dict[str, Any]]) -> None:
        lines.append(title)
        for item in sorted(selected, key=lambda value: str(value["instance_id"])):
            center = item["expected_center_xyz_m"]
            dimensions = item["dimensions_m"]
            color = item.get("color", {}).get("name", "unknown")
            support_id = item.get("support_surface")
            support = by_id.get(str(support_id), {}).get("category") if support_id else None
            facts = (
                f"- category={item['category']}; color={color}; "
                f"center=({_number(center[0])},{_number(center[1])},{_number(center[2])}); "
                f"dimensions=({_number(dimensions[0])},{_number(dimensions[1])},"
                f"{_number(dimensions[2])})"
            )
            if support is not None:
                facts += f"; supported_by={support}"
            lines.append(facts + ".")

    append_instances("Exact object instances:", [item for item in instances if item["kind"] == "object"])
    append_instances("Exact room surfaces:", [item for item in instances if item["kind"] != "object"])

    predicate_names = {
        "left_of": "left",
        "right_of": "right",
        "in_front_of": "in front",
        "behind": "behind",
        "above": "above",
        "below": "below",
        "near": "near",
        "far": "far",
        "on": "on",
        "under": "under",
        "mounted_on": "mounted on",
    }
    relationships = []
    for relation in oracle.get("relationships", []):
        subject = by_id.get(str(relation.get("subject_instance_id")))
        object_ = by_id.get(str(relation.get("object_instance_id")))
        if subject is None or object_ is None:
            continue
        predicate = predicate_names.get(
            str(relation["predicate"]), str(relation["predicate"]).replace("_", " ")
        )
        relationships.append(
            f"- subject={subject['category']}; predicate={predicate}; object={object_['category']}."
        )
    if relationships:
        lines.append("Exact directed relationships (roles are significant):")
        lines.extend(sorted(set(relationships)))
    return "\n".join(lines)


class LocalOracleAnswerer:
    def __init__(self, config: dict[str, Any], *, local_files_only: bool = True) -> None:
        language = config["language"]
        baseline = config["evaluation"]["baselines"]["oracle_text"]
        self.system_prompt = str(baseline["system_prompt"])
        self.max_answer_tokens = int(baseline["max_answer_tokens"])
        self.local = load_local_language_model(
            str(language["model_id"]),
            revision=str(language["revision"]),
            requested_dtype=str(language.get("dtype", "float16")),
            freeze=True,
            local_files_only=local_files_only,
            backend=str(language.get("backend", "auto")),
        )

    def __call__(self, scene_text: str, question: str) -> str:
        messages = [
            {
                "role": "system",
                "content": f"{self.system_prompt}\n\n{scene_text}",
            },
            {"role": "user", "content": question},
        ]
        tokenizer = self.local.tokenizer
        encoded = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
        )
        # Transformers 4 commonly returns a tensor here. Transformers 5 may
        # return a BatchEncoding for the same call (including Gemma 4), so
        # normalize both representations before constructing the mask.
        input_ids = (
            encoded.input_ids
            if hasattr(encoded, "input_ids")
            else encoded["input_ids"]
            if isinstance(encoded, dict)
            else encoded
        ).to(self.local.device)
        if input_ids.ndim != 2 or input_ids.shape[1] == 0:
            raise ValueError("Oracle-text chat template returned no token IDs")
        attention_mask = torch.ones_like(input_ids)
        with torch.inference_mode():
            output = self.local.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                do_sample=False,
                max_new_tokens=self.max_answer_tokens,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        continuation = output[0, input_ids.shape[1] :].detach().cpu()
        return tokenizer.decode(continuation, skip_special_tokens=True).strip()


def run_oracle_text_baseline(
    config: dict[str, Any],
    references_path: str | Path,
    output_path: str | Path,
    *,
    answerer: TextAnswerer | None = None,
    limit: int | None = None,
    resume: bool = True,
    local_files_only: bool = True,
) -> dict[str, Any]:
    """Generate cached oracle-text predictions; QA answers are never given to the model."""
    references_source = Path(references_path).resolve()
    destination = Path(output_path).resolve()
    references = read_jsonl(references_source)
    if limit is not None:
        references = references[:limit]
    existing = indexed_records(destination) if resume else {}
    model_answerer = answerer
    model_revision = str(config["language"]["revision"])
    data_root = PROJECT_ROOT / str(config["paths"]["data_root"])
    oracle_root = data_root / "oracle"
    scene_cache: dict[str, str] = {}
    oracle_hashes: dict[str, str] = {}
    timings: list[float] = []
    generated: list[dict[str, Any]] = []
    for reference in references:
        scene_id = str(reference["scene_id"])
        question_id = str(reference["question_id"])
        question = str(reference["question"])
        question_sha256 = text_fingerprint(scene_id, question_id, question)
        if scene_id not in scene_cache:
            oracle_path = oracle_root / scene_id / "oracle.json"
            oracle = json.loads(oracle_path.read_text(encoding="utf-8"))
            scene_cache[scene_id] = oracle_scene_text(oracle)
            oracle_hashes[scene_id] = sha256_file(oracle_path)
        cached = existing.get((scene_id, question_id))
        if (
            cached is not None
            and cached.get("baseline") == "oracle_text_upper_bound"
            and cached.get("question_sha256") == question_sha256
            and cached.get("oracle_sha256") == oracle_hashes[scene_id]
            and cached.get("model_revision") == model_revision
        ):
            generated.append(cached)
            continue
        if model_answerer is None:
            model_answerer = LocalOracleAnswerer(config, local_files_only=local_files_only)
        started = time.perf_counter()
        answer = model_answerer(scene_cache[scene_id], question)
        timings.append(time.perf_counter() - started)
        generated.append(
            {
                "answer": answer,
                "baseline": "oracle_text_upper_bound",
                "model_revision": model_revision,
                "oracle_sha256": oracle_hashes[scene_id],
                "question_id": question_id,
                "question_sha256": question_sha256,
                "scene_id": scene_id,
            }
        )
        atomic_write_jsonl(destination, generated)
    # Also removes stale records when every requested prediction was cached.
    atomic_write_jsonl(destination, generated)
    return {
        "baseline": "oracle_text_upper_bound",
        "evaluation_only": True,
        "generated_count": len(generated),
        "new_prediction_count": len(timings),
        "mean_seconds_per_new_prediction": (
            sum(timings) / len(timings) if timings else 0.0
        ),
        "output_path": str(destination),
        "output_sha256": sha256_file(destination),
        "reference_path": str(references_source),
    }


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--references", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--allow-download", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    config = load_config(args.config)
    data_root = PROJECT_ROOT / str(config["paths"]["data_root"])
    reports_root = PROJECT_ROOT / str(config["paths"]["reports_root"])
    references = args.references or data_root / "qa" / "test.jsonl"
    output = args.output or reports_root / "predictions" / "oracle_text.jsonl"
    report = run_oracle_text_baseline(
        config,
        references,
        output,
        limit=args.limit,
        resume=not args.no_resume,
        local_files_only=not args.allow_download,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
