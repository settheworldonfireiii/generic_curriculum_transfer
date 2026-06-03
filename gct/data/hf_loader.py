from __future__ import annotations

from collections.abc import Iterable, Iterator
from typing import Any

from gct.config.schema import DatasetConfig
from gct.utils.hashing import stable_id


def normalize_row(row: dict[str, Any], index: int, config: DatasetConfig) -> dict[str, Any]:
    try:
        prompt = row[config.prompt_column]
        answer = row[config.answer_column]
    except KeyError as exc:
        raise KeyError(
            f"Dataset row is missing required column {exc!s}. "
            f"Pass --prompt-column/--answer-column for this dataset."
        ) from exc

    if config.id_column and config.id_column in row:
        task_id = str(row[config.id_column])
    else:
        task_id = stable_id(config.name, config.split, index, prompt, prefix="task")

    metadata: dict[str, Any] = {}
    if config.level_column and config.level_column in row:
        metadata["level"] = row[config.level_column]
    if config.category_column and config.category_column in row:
        metadata["category"] = row[config.category_column]

    return {
        "task_id": task_id,
        "prompt": str(prompt),
        "answer": str(answer),
        "source_dataset": config.name,
        "source_split": config.split,
        "metadata": metadata,
    }


def _passes_filter(row: dict[str, Any], expression: str | None) -> bool:
    if not expression:
        return True
    allowed_builtins = {"int": int, "float": float, "str": str, "len": len}
    try:
        return bool(eval(expression, {"__builtins__": allowed_builtins}, dict(row)))  # noqa: S307
    except Exception:
        return False


def load_hf_rows(config: DatasetConfig) -> Iterator[dict[str, Any]]:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError("Install datasets or run `pip install -e .` before loading HF datasets.") from exc

    dataset = load_dataset(
        config.name,
        config.config_name,
        split=config.split,
        streaming=config.streaming,
    )

    count = 0
    for index, row in enumerate(dataset):
        row_dict = dict(row)
        if not _passes_filter(row_dict, config.filter_expr):
            continue
        yield normalize_row(row_dict, index, config)
        count += 1
        if config.max_rows is not None and count >= config.max_rows:
            break


def rows_from_iterable(rows: Iterable[dict[str, Any]], config: DatasetConfig) -> Iterator[dict[str, Any]]:
    for index, row in enumerate(rows):
        if _passes_filter(row, config.filter_expr):
            yield normalize_row(row, index, config)

