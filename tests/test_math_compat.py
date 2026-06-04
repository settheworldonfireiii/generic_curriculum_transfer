import csv
from pathlib import Path
from types import SimpleNamespace

import pytest

from gct.cli import main
from gct.math_compat.commands import build_math_neighbors
from gct.math_compat.utils import (
    answers_match,
    build_problem_prompt,
    build_transfer_prompt,
    combined_metric,
    extract_final_answer,
    task_id,
    write_csv,
    write_jsonl,
)


def test_exact_math_cli_command_is_exposed() -> None:
    with pytest.raises(SystemExit) as exc:
        main(["math-sweep-sae", "--help"])

    assert exc.value.code == 0


def test_math_compat_prompts_and_grading_match_original_shape() -> None:
    assert task_id("test", 12) == "test_00012"
    assert "put the final answer in \\boxed{}" in build_problem_prompt("2+2?")
    transfer_prompt = build_transfer_prompt("anchor", "solution", "target")
    assert "Solved related problem:\nanchor" in transfer_prompt
    assert "Target solution:" in transfer_prompt

    assert extract_final_answer("Reasoning. Final answer is \\boxed{\\frac{1}{2}}") == "\\frac{1}{2}"
    assert answers_match("0.5", "1/2")
    assert answers_match("1,000,000", "1,\\!000,\\!000")


def test_math_compat_combined_metric_uses_original_weights() -> None:
    a = {"L10:1": 2.0, "L16:2": 1.0}
    b = {"L10:1": 1.0, "L24:3": 3.0}

    score, cosine, jaccard, tanimoto = combined_metric(a, b, "cos07")

    assert score == 0.70 * cosine + 0.15 * jaccard + 0.15 * tanimoto


def test_math_compat_builds_expected_neighbor_files(tmp_path: Path) -> None:
    out_dir = tmp_path / "results_math"
    out_dir.mkdir()
    write_csv(
        out_dir / "math_sweep_summary.csv",
        [
            {
                "task_id": "test_00000",
                "split": "test",
                "index": 0,
                "level": 1,
                "category": "Algebra",
                "attempts": 3,
                "successes": 1,
                "solved": True,
                "success_rate": 1 / 3,
                "first_success_sample": 0,
            },
            {
                "task_id": "test_00004",
                "split": "test",
                "index": 4,
                "level": 2,
                "category": "Algebra",
                "attempts": 3,
                "successes": 0,
                "solved": False,
                "success_rate": 0.0,
                "first_success_sample": "",
            },
        ],
        [
            "task_id",
            "split",
            "index",
            "level",
            "category",
            "attempts",
            "successes",
            "solved",
            "success_rate",
            "first_success_sample",
        ],
    )
    write_jsonl(
        out_dir / "math_sae_task_layer_rows.jsonl",
        [
            {"task_id": "test_00000", "layer": 10, "features": {"L10:1": 1.0}},
            {"task_id": "test_00004", "layer": 10, "features": {"L10:1": 0.5}},
            {"task_id": "test_00000", "layer": 28, "features": {"L28:4": 1.0}},
            {"task_id": "test_00004", "layer": 28, "features": {"L28:4": 0.5}},
        ],
    )

    build_math_neighbors(SimpleNamespace(out_dir=out_dir, top_k=1))

    path = out_dir / "neighbors_combo_cos07.csv"
    with path.open() as f:
        rows = list(csv.DictReader(f))

    assert rows[0]["target_task_id"] == "test_00004"
    assert rows[0]["anchor_task_id"] == "test_00000"
    assert rows[0]["score_variant"] == "cos07"
