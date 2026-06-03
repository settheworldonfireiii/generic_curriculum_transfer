from pathlib import Path

from gct.sweep.grading import answers_match, extract_final_answer
from gct.sweep.summary import summarize_sweep
from gct.utils.jsonl import write_jsonl


def test_extract_final_answer_prefers_boxed_answer() -> None:
    assert extract_final_answer("Reasoning here. Final is \\boxed{42}.") == "42"


def test_answers_match_numeric_and_string_forms() -> None:
    assert answers_match("0.5", "1/2")
    assert answers_match("\\frac{3}{4}", "3/4")
    assert answers_match("Algebra", " algebra ")


def test_summarize_sweep_writes_solved_split(tmp_path: Path) -> None:
    raw = tmp_path / "raw.jsonl"
    summary = tmp_path / "summary.csv"
    solved_ids = tmp_path / "solved_ids.txt"
    write_jsonl(
        raw,
        [
            {"task_id": "a", "sample_index": 0, "exact": False, "metadata": {"level": 1}},
            {"task_id": "a", "sample_index": 1, "exact": True, "metadata": {"level": 1}},
            {"task_id": "b", "sample_index": 0, "exact": False, "metadata": {"level": 2}},
        ],
    )

    rows = summarize_sweep(raw, summary, solved_ids)

    assert rows["a"]["solved"] is True
    assert rows["a"]["successes"] == 1
    assert rows["b"]["solved"] is False
    assert solved_ids.read_text().strip() == "a"

