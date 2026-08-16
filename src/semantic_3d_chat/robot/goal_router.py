"""Route user outcomes into semantic goals without exposing motor controls.

This grammar contains only generic task verbs.  Object/category phrases are
copied from the user's own text; the router has no inventory of the room and
cannot read scene files.  Low-level turns and metric movement are deliberately
not part of this user-facing surface—the learned policy and numeric planner own
those internal actions.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from typing import Final, Literal

SemanticGoalKind = Literal["lap", "approach", "face", "between"]

_SPACE: Final[re.Pattern[str]] = re.compile(r"\s+")
_TRAILING: Final[re.Pattern[str]] = re.compile(r"[.!?\s]+$")
_ARTICLE: Final[re.Pattern[str]] = re.compile(r"^(?:the|a|an)\s+", re.IGNORECASE)
_POLITE: Final[re.Pattern[str]] = re.compile(
    r"^(?:(?:please\s+)?(?:can|could|would|will)\s+you(?:\s+please)?\s+|please\s+)",
    re.IGNORECASE,
)

_LAP: Final[re.Pattern[str]] = re.compile(
    r"^(?:(?:do|make|take|complete|drive)\s+(?:a|one)\s+lap(?:\s+around\s+"
    r"(?:the\s+)?(?:room|space|environment))?|circle\s+(?:the\s+)?"
    r"(?:room|space|environment)|patrol\s+(?:around\s+)?(?:the\s+)?"
    r"(?:room|space|environment)|(?:explore|tour)\s+(?:the\s+)?"
    r"(?:room|space|environment)|make\s+(?:a\s+)?circuit\s+of\s+(?:the\s+)?"
    r"(?:room|space|environment))$",
    re.IGNORECASE,
)

_BETWEEN: Final[re.Pattern[str]] = re.compile(
    r"^(?:(?:go|move|navigate|drive|park|stand|stop)\s+)?between\s+"
    r"(?:the\s+)?(.+?)\s+and\s+(?:the\s+)?(.+?)$",
    re.IGNORECASE,
)

_APPROACH: Final[re.Pattern[str]] = re.compile(
    r"^(?:move\s+(?:close|closer)\s+to|walk\s+toward|move\s+toward|go\s+to|"
    r"navigate\s+to|drive\s+to|approach|park\s+(?:beside|near)|"
    r"go\s+(?:beside|near)|stand\s+(?:beside|near)|take\s+me\s+to)\s+"
    r"(?:the\s+)?(.+?)$",
    re.IGNORECASE,
)

_FACE: Final[re.Pattern[str]] = re.compile(
    r"^(?:face|turn\s+toward|turn\s+to\s+face|look\s+at|orient\s+toward)\s+"
    r"(?:the\s+)?(.+?)$",
    re.IGNORECASE,
)


def _target(value: str) -> str:
    result = _TRAILING.sub("", _SPACE.sub(" ", value.strip()))
    result = _ARTICLE.sub("", result).strip()
    if not result or len(result) > 256 or "\n" in result or "\r" in result:
        raise ValueError("Semantic goal target phrase is invalid")
    return result


@dataclass(frozen=True)
class SemanticGoalRequest:
    """One outcome-level request with only user-supplied target text."""

    kind: SemanticGoalKind
    targets: tuple[str, ...] = ()
    request_sha256: str = ""

    def __post_init__(self) -> None:
        expected = 0 if self.kind == "lap" else 2 if self.kind == "between" else 1
        if len(self.targets) != expected:
            raise ValueError("Semantic goal target count differs from goal kind")
        if not re.fullmatch(r"[0-9a-f]{64}", self.request_sha256):
            raise ValueError("Semantic goal request hash is invalid")


def parse_semantic_goal(text: str) -> SemanticGoalRequest | None:
    """Parse a high-level task or return ``None`` for ordinary scene dialogue."""

    if not isinstance(text, str) or not text.strip():
        raise ValueError("Semantic goal text must be non-empty")
    normalized = _SPACE.sub(" ", text.strip())
    normalized = _POLITE.sub("", normalized)
    normalized = _TRAILING.sub("", normalized).strip()
    digest = hashlib.sha256(text.strip().encode("utf-8")).hexdigest()
    if _LAP.fullmatch(normalized):
        return SemanticGoalRequest("lap", request_sha256=digest)
    if match := _BETWEEN.fullmatch(normalized):
        return SemanticGoalRequest(
            "between",
            (_target(match.group(1)), _target(match.group(2))),
            digest,
        )
    if match := _APPROACH.fullmatch(normalized):
        return SemanticGoalRequest("approach", (_target(match.group(1)),), digest)
    if match := _FACE.fullmatch(normalized):
        return SemanticGoalRequest("face", (_target(match.group(1)),), digest)
    return None


__all__ = ["SemanticGoalKind", "SemanticGoalRequest", "parse_semantic_goal"]
