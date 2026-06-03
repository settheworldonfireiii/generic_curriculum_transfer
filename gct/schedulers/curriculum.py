from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from gct.schedulers.knapsack import KnapsackItem, allocate_knapsack
from gct.utils.jsonl import read_jsonl, write_jsonl


@dataclass(frozen=True)
class WorkItem:
    task_id: str
    prompt: str
    answer: str
    context_length: int
    priority: float
    metadata: dict[str, Any]

    @staticmethod
    def from_row(row: dict[str, Any]) -> "WorkItem":
        prompt = row["prompt"]
        answer = row["answer"]
        metadata = row.get("metadata", {})
        level = metadata.get("level", 1)
        try:
            level_priority = 1.0 / max(1.0, float(level))
        except (TypeError, ValueError):
            level_priority = 1.0
        return WorkItem(
            task_id=row["task_id"],
            prompt=prompt,
            answer=answer,
            context_length=len(prompt.split()) + len(answer.split()),
            priority=level_priority,
            metadata=metadata,
        )


class CurriculumPlanner:
    def __init__(self, token_capacity: int, target_utilization: float = 0.85) -> None:
        if not 0 < target_utilization <= 1:
            raise ValueError("target_utilization must be in (0, 1]")
        self.token_capacity = max(1, int(token_capacity * target_utilization))

    def plan(self, items: list[WorkItem]) -> list[WorkItem]:
        selected = allocate_knapsack(
            [
                KnapsackItem(
                    item_id=item.task_id,
                    value=1.0 / max(1, item.context_length),
                    cost=max(1, item.context_length),
                    priority=item.priority,
                )
                for item in items
            ],
            capacity=self.token_capacity,
        )
        selected_ids = {item.item_id for item in selected}
        return [item for item in items if item.task_id in selected_ids]


def write_curriculum_plan(tasks_path: Path, output_path: Path, token_capacity: int) -> Path:
    items = [WorkItem.from_row(row) for row in read_jsonl(tasks_path)]
    planned = CurriculumPlanner(token_capacity=token_capacity).plan(items)
    write_jsonl(
        output_path,
        [
            {
                "task_id": item.task_id,
                "prompt": item.prompt,
                "answer": item.answer,
                "context_length": item.context_length,
                "priority": item.priority,
                "metadata": item.metadata,
            }
            for item in planned
        ],
    )
    return output_path

