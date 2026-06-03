from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

from gct.sae.metrics import MetricResult, arc_weighted_metric, metric_result
from gct.sae.store import metadata_vector


NEIGHBOR_FIELDS = [
    "target_task_id",
    "anchor_task_id",
    "neighbor_rank",
    "score_variant",
    "layer_regime",
    "combined_similarity",
    "cosine",
    "sae_cosine",
    "weighted_jaccard",
    "structure_cosine",
    "tanimoto",
    "target_level",
    "target_category",
    "anchor_level",
    "anchor_category",
]


def build_neighbor_rows(
    summary: dict[str, dict[str, Any]],
    features: dict[str, dict[str, float]],
    top_k: int,
    variant: str,
    layer_regime: str,
    allowed_target_ids: set[str] | None = None,
) -> list[dict[str, Any]]:
    solved_ids = [task_id for task_id, row in summary.items() if bool(row.get("solved"))]
    target_ids = [task_id for task_id, row in summary.items() if not bool(row.get("solved"))]
    if allowed_target_ids is not None:
        target_ids = [task_id for task_id in target_ids if task_id in allowed_target_ids]

    out_rows: list[dict[str, Any]] = []
    for target_id in target_ids:
        if target_id not in features:
            continue
        scored: list[tuple[float, str, MetricResult]] = []
        for anchor_id in solved_ids:
            if anchor_id not in features:
                continue
            result = _score_variant(
                variant,
                features[target_id],
                features[anchor_id],
                metadata_vector(summary[target_id]),
                metadata_vector(summary[anchor_id]),
            )
            scored.append((result.score, anchor_id, result))
        scored.sort(key=lambda item: (item[0], item[1]), reverse=True)
        for rank, (_score, anchor_id, result) in enumerate(scored[:top_k], start=1):
            target_meta = summary[target_id].get("metadata") or {}
            anchor_meta = summary[anchor_id].get("metadata") or {}
            out_rows.append(
                {
                    "target_task_id": target_id,
                    "anchor_task_id": anchor_id,
                    "neighbor_rank": rank,
                    "score_variant": variant,
                    "layer_regime": layer_regime,
                    "combined_similarity": result.score,
                    "cosine": result.sae_cosine,
                    "sae_cosine": result.sae_cosine,
                    "weighted_jaccard": result.weighted_jaccard,
                    "structure_cosine": result.structure_cosine,
                    "tanimoto": result.tanimoto,
                    "target_level": target_meta.get("level", ""),
                    "target_category": target_meta.get("category", ""),
                    "anchor_level": anchor_meta.get("level", ""),
                    "anchor_category": anchor_meta.get("category", ""),
                }
            )
    return out_rows


def write_neighbors(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=NEIGHBOR_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _score_variant(
    variant: str,
    target_features: dict[str, float],
    anchor_features: dict[str, float],
    target_structure: dict[str, float],
    anchor_structure: dict[str, float],
) -> MetricResult:
    if variant == "arc_weighted":
        return arc_weighted_metric(target_features, anchor_features, target_structure, anchor_structure)
    return metric_result(target_features, anchor_features, variant)

