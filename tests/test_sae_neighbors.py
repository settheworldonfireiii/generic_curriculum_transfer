from pathlib import Path

from gct.sae.neighbors import build_neighbor_rows
from gct.sae.store import load_features_for_layers
from gct.utils.jsonl import write_jsonl


def test_load_features_for_layers_merges_selected_layers(tmp_path: Path) -> None:
    rows = [
        {"task_id": "a", "layer": 10, "features": {"L10:1": 1.0}},
        {"task_id": "a", "layer": 16, "features": {"L16:2": 2.0}},
        {"task_id": "a", "layer": 28, "features": {"L28:3": 3.0}},
    ]
    path = tmp_path / "sae.jsonl"
    write_jsonl(path, rows)

    features = load_features_for_layers(path, {10, 16})

    assert features["a"] == {"L10:1": 1.0, "L16:2": 2.0}


def test_build_neighbor_rows_ranks_solved_anchors_and_respects_target_filter() -> None:
    features = {
        "target_keep": {"f1": 1.0},
        "target_drop": {"f2": 1.0},
        "anchor_good": {"f1": 1.0},
        "anchor_bad": {"f3": 1.0},
    }
    summary = {
        "target_keep": {"solved": False, "metadata": {"category": "x"}},
        "target_drop": {"solved": False, "metadata": {"category": "x"}},
        "anchor_good": {"solved": True, "metadata": {"category": "x"}},
        "anchor_bad": {"solved": True, "metadata": {"category": "y"}},
    }

    rows = build_neighbor_rows(
        summary=summary,
        features=features,
        top_k=2,
        variant="cos07",
        layer_regime="combo",
        allowed_target_ids={"target_keep"},
    )

    assert [row["target_task_id"] for row in rows] == ["target_keep", "target_keep"]
    assert rows[0]["anchor_task_id"] == "anchor_good"
    assert rows[0]["neighbor_rank"] == 1
    assert rows[0]["combined_similarity"] > rows[1]["combined_similarity"]

