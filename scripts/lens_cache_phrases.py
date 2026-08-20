#!/usr/bin/env python3
"""Embed every grounding phrase once with Gemma and cache it.

The scaling sweep trains dozens of models over the same handful of phrases.
Loading a 2B decoder for each of them would dominate the runtime and change
nothing, since the embeddings are frozen either way.
"""

from __future__ import annotations

import argparse

import numpy as np

from semantic_3d_chat.config import PROJECT_ROOT
from semantic_3d_chat.spatial_lens.grounding_data import (
    available_rooms,
    embed_phrase_tokens,
    embed_phrases,
)
from semantic_3d_chat.spatial_lens.point_grounding_data import (
    disambiguation_examples,
    relational_examples,
    room_examples,
)

CACHE = PROJECT_ROOT / "data" / "spatial_lens" / "phrase_embeddings.npz"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()

    phrases: set[str] = set()
    for room in available_rooms():
        for example in room_examples(room):
            phrases.add(example.phrase)
        for example in relational_examples(room):
            phrases.add(example.phrase)
        for example in disambiguation_examples(room):
            phrases.add(example.phrase)
    ordered = sorted(phrases)
    print(f"{len(ordered)} distinct phrases across {len(available_rooms())} rooms")

    from semantic_3d_chat.language.local_lm import load_local_language_model

    language = load_local_language_model(
        "google/gemma-4-E2B-it",
        revision="3e22461f65e89153144f8adb70e3b8c2cc9845a7",
        requested_dtype="bfloat16",
        local_files_only=True,
    )
    vectors = embed_phrases(language, ordered)
    # Also keep the phrase one token at a time: a mean-pooled vector can name a
    # thing but cannot name a relation to a thing.
    words, mask = embed_phrase_tokens(language, ordered)
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        CACHE,
        phrases=np.asarray(ordered),
        vectors=vectors.astype(np.float32),
        words=words.astype(np.float16),
        word_mask=mask,
    )
    print(f"wrote {CACHE.relative_to(PROJECT_ROOT)}  pooled={vectors.shape} tokens={words.shape}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
