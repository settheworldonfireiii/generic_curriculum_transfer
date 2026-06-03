from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, Dataset

from gct.utils.jsonl import read_jsonl


class JsonlTaskDataset(Dataset[dict[str, Any]]):
    def __init__(self, path: Path) -> None:
        self.path = path
        self.rows = list(read_jsonl(path))

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.rows[index]


def collate_task_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "task_id": [row["task_id"] for row in rows],
        "prompt": [row["prompt"] for row in rows],
        "answer": [row["answer"] for row in rows],
        "metadata": [row.get("metadata", {}) for row in rows],
    }


def build_dataloader(
    path: Path,
    batch_size: int,
    num_workers: int,
    prefetch_factor: int = 2,
    shuffle: bool = False,
) -> DataLoader[dict[str, Any]]:
    dataset = JsonlTaskDataset(path)
    kwargs: dict[str, Any] = {
        "batch_size": batch_size,
        "shuffle": shuffle,
        "num_workers": num_workers,
        "collate_fn": collate_task_rows,
        "pin_memory": torch.cuda.is_available(),
    }
    if num_workers > 0:
        kwargs["prefetch_factor"] = prefetch_factor
        kwargs["persistent_workers"] = True
    return DataLoader(dataset, **kwargs)

