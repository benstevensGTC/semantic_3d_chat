"""Which local Gemma to run, and at which revision.

E2B is pinned to a snapshot because every cached hidden state in this repo was
computed against it and a silent upgrade would invalidate them. Anything else is
taken at main: nothing is cached for those, so there is nothing to invalidate,
and pinning a revision nobody has verified would only look rigorous.
"""

from __future__ import annotations

DEFAULT_MODEL = "google/gemma-4-E2B-it"
PINNED = {
    # The snapshot every cached hidden state in this repo was computed against.
    "google/gemma-4-E2B-it": "3e22461f65e89153144f8adb70e3b8c2cc9845a7",
}


def revision_for(model_id: str) -> str:
    return PINNED.get(model_id, "main")


def add_model_arguments(parser) -> None:
    """Give a script --model and --revision with the right defaults."""

    parser.add_argument("--model", default=DEFAULT_MODEL,
                        help="local Gemma to run; larger variants need the disk "
                             "and the RAM to hold them")
    parser.add_argument("--revision", default=None,
                        help="defaults to the pinned snapshot for E2B, main otherwise")


__all__ = ["DEFAULT_MODEL", "PINNED", "add_model_arguments", "revision_for"]
