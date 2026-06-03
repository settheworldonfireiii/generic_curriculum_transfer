from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

from gct.utils.jsonl import read_jsonl


SUMMARY_FIELDS = [
    "task_id",
    "attempts",
    "successes",
    "solved",
    "success_rate",
    "first_success_sample",
    "level",
    "category",
]


def summarize_sweep(raw_path: Path, summary_path: Path, solved_ids_path: Path) -> dict[str, dict[str, Any]]:
    by_task: dict[str, dict[str, Any]] = {}
    for row in read_jsonl(raw_path):
        metadata = row.get("metadata") or {}
        current = by_task.setdefault(
            row["task_id"],
            {
                "task_id": row["task_id"],
                "attempts": 0,
                "successes": 0,
                "solved": False,
                "success_rate": 0.0,
                "first_success_sample": "",
                "level": metadata.get("level", ""),
                "category": metadata.get("category", ""),
                "metadata": metadata,
            },
        )
        current["attempts"] += 1
        if bool(row.get("exact")):
            current["successes"] += 1
            current["solved"] = True
            if current["first_success_sample"] == "":
                current["first_success_sample"] = int(row.get("sample_index", 0))

    rows = []
    for row in by_task.values():
        row["success_rate"] = row["successes"] / row["attempts"] if row["attempts"] else 0.0
        rows.append(row)
    rows.sort(key=lambda row: (-int(row["solved"]), -float(row["success_rate"]), row["task_id"]))

    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    solved = [row["task_id"] for row in rows if row["solved"]]
    solved_ids_path.parent.mkdir(parents=True, exist_ok=True)
    solved_ids_path.write_text("\n".join(solved) + ("\n" if solved else ""))
    return {row["task_id"]: row for row in rows}

