from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from torch.utils.data import Dataset


@dataclass(frozen=True)
class QARecord:
    scene_id: str
    question_id: str
    question: str
    answer: str
    answer_type: str
    target_xyz: list[float] | None
    reference_xyz: list[float] | None = None
    counterfactual_pair_id: str | None = None
    counterfactual_question_key: str | None = None
    counterfactual_expected_change: bool | None = None
    counterfactual_role: str | None = None
    counterfactual_change_type: str | None = None


class SceneQADataset(Dataset[QARecord]):
    """Training-only reader; this module is never imported by chat inference."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.records = [
            QARecord(
                scene_id=item["scene_id"],
                question_id=item["question_id"],
                question=item["question"],
                answer=item["answer"],
                answer_type=item["answer_type"],
                target_xyz=item.get("target_xyz"),
                reference_xyz=item.get("reference_xyz"),
                counterfactual_pair_id=item.get("counterfactual_pair_id"),
                counterfactual_question_key=item.get("counterfactual_question_key"),
                counterfactual_expected_change=item.get("counterfactual_expected_change"),
                counterfactual_role=item.get("counterfactual_role"),
                counterfactual_change_type=item.get("counterfactual_change_type"),
            )
            for line in self.path.read_text(encoding="utf-8").splitlines()
            if (item := json.loads(line))
        ]

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> QARecord:
        return self.records[index]
