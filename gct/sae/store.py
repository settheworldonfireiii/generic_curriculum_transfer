from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

from gct.utils.jsonl import read_jsonl


def load_features_for_layers(rows_path: Path, layers: set[int]) -> dict[str, dict[str, float]]:
    by_task: dict[str, dict[str, float]] = {}
    for row in read_jsonl(rows_path):
        if int(row["layer"]) not in layers:
            continue
        features = by_task.setdefault(row["task_id"], {})
        for key, value in row["features"].items():
            features[key] = float(value)
    return by_task


def load_task_summary(path: Path) -> dict[str, dict[str, Any]]:
    if path.suffix == ".csv":
        with path.open() as f:
            rows = list(csv.DictReader(f))
    else:
        rows = list(read_jsonl(path))
    summary = {}
    for row in rows:
        task_id = row["task_id"]
        solved = _as_bool(row.get("solved", False))
        metadata = dict(row.get("metadata") or {})
        for key in ("level", "category"):
            if key in row and row[key] not in (None, ""):
                metadata[key] = row[key]
        summary[task_id] = {
            **row,
            "task_id": task_id,
            "solved": solved,
            "metadata": metadata,
        }
    return summary


def metadata_vector(row: dict[str, Any]) -> dict[str, float]:
    metadata = row.get("metadata") or {}
    vector: dict[str, float] = {}
    level = metadata.get("level")
    category = metadata.get("category")
    if level not in (None, ""):
        vector[f"level:{str(level).strip()}"] = 1.0
    if category not in (None, ""):
        vector[f"category:{str(category).strip().lower()}"] = 1.0
    return vector


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    return str(value).strip().lower() in {"true", "1", "yes", "y"}

