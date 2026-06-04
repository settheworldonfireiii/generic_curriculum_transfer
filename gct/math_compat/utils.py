from __future__ import annotations

import csv
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


MATH_CONFIGS = [
    "algebra",
    "counting_and_probability",
    "geometry",
    "intermediate_algebra",
    "number_theory",
    "prealgebra",
    "precalculus",
]


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(json.dumps(row, separators=(",", ":")) + "\n")
        f.flush()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    with path.open() as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row, separators=(",", ":")) + "\n")


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def task_id(split: str, index: int) -> str:
    return f"{split}_{index:05d}"


def parse_level(value: Any) -> int | None:
    if value is None:
        return None
    match = re.search(r"\d+", str(value))
    return int(match.group()) if match else None


def build_problem_prompt(problem: str) -> str:
    return (
        "Solve the following competition math problem. Show concise reasoning, "
        "and put the final answer in \\boxed{}.\n\n"
        f"Problem:\n{problem}\n\nSolution:"
    )


def build_transfer_prompt(anchor_problem: str, anchor_solution: str, target_problem: str) -> str:
    return (
        "A solved related competition math problem is provided first. Use it as an analogy if useful, "
        "but solve the target problem from its own statement.\n\n"
        "Solved related problem:\n"
        f"{anchor_problem}\n\n"
        "Solved related solution:\n"
        f"{anchor_solution}\n\n"
        "Target problem:\n"
        f"{target_problem}\n\n"
        "Solve the target. Show concise reasoning, and put the final answer in \\boxed{}.\n\n"
        "Target solution:"
    )


def extract_boxed(text: str) -> str | None:
    marker = "\\boxed"
    idx = text.rfind(marker)
    if idx == -1:
        return None
    start = text.find("{", idx)
    if start == -1:
        return None
    depth = 0
    for pos in range(start, len(text)):
        if text[pos] == "{":
            depth += 1
        elif text[pos] == "}":
            depth -= 1
            if depth == 0:
                return text[start + 1 : pos].strip()
    return None


def extract_final_answer(text: str | None) -> str | None:
    if text is None:
        return None
    boxed = extract_boxed(text)
    if boxed:
        return boxed
    patterns = [
        r"(?:final answer|answer is|therefore|so the answer is)\s*[:=]?\s*([^\n.]+)",
        r"####\s*([^\n]+)",
    ]
    for pattern in patterns:
        matches = re.findall(pattern, text, flags=re.IGNORECASE)
        if matches:
            return matches[-1].strip()
    number_matches = re.findall(r"[-+]?\d+(?:\.\d+)?(?:/\d+)?", text)
    if number_matches:
        return number_matches[-1].strip()
    return None


def normalize_answer(answer: str | None) -> str:
    if answer is None:
        return ""
    text = str(answer).strip()
    text = text.replace("\\left", "").replace("\\right", "")
    text = text.replace("\\,", "").replace("\\!", "")
    text = text.replace("$", "")
    text = re.sub(r"\\text\{([^}]*)\}", r"\1", text)
    text = re.sub(r"\\mathrm\{([^}]*)\}", r"\1", text)
    text = text.replace(" ", "")
    text = text.rstrip(".")
    return text.lower()


def numeric_value(text: str) -> float | None:
    text = normalize_answer(text)
    frac = re.fullmatch(r"([-+]?\d+)/(\d+)", text)
    if frac:
        return float(frac.group(1)) / float(frac.group(2))
    try:
        return float(text)
    except ValueError:
        return None


def answers_match(predicted: str | None, expected: str | None) -> bool:
    p_norm = normalize_answer(predicted)
    e_norm = normalize_answer(expected)
    if not p_norm or not e_norm:
        return False
    if p_norm == e_norm:
        return True
    p_num = numeric_value(p_norm)
    e_num = numeric_value(e_norm)
    if p_num is not None and e_num is not None:
        return math.isclose(p_num, e_num, rel_tol=1e-9, abs_tol=1e-9)
    return False


def summarize_sweep(raw_path: Path, summary_path: Path, solved_ids_path: Path) -> dict[str, dict[str, Any]]:
    attempts_by_task: dict[str, dict[str, Any]] = {}
    for row in read_jsonl(raw_path):
        task = attempts_by_task.setdefault(
            row["task_id"],
            {
                "task_id": row["task_id"],
                "split": row["split"],
                "index": int(row["index"]),
                "level": row["level"],
                "category": row["category"],
                "attempts": 0,
                "successes": 0,
                "first_success_sample": "",
            },
        )
        task["attempts"] += 1
        if row["exact"]:
            task["successes"] += 1
            if task["first_success_sample"] == "":
                task["first_success_sample"] = int(row["sample_index"])
    rows = []
    for row in attempts_by_task.values():
        row["solved"] = row["successes"] > 0
        row["success_rate"] = row["successes"] / row["attempts"] if row["attempts"] else 0.0
        rows.append(row)
    rows.sort(key=lambda row: (-int(row["solved"]), -float(row["success_rate"]), row["task_id"]))
    write_csv(
        summary_path,
        rows,
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
    solved = [row["task_id"] for row in rows if row["solved"]]
    solved_ids_path.parent.mkdir(parents=True, exist_ok=True)
    solved_ids_path.write_text("\n".join(solved) + ("\n" if solved else ""))
    print(f"Wrote {summary_path}")
    print(f"Wrote {solved_ids_path}")
    print(f"Solved tasks: {len(solved)}/{len(rows)}")
    return {row["task_id"]: row for row in rows}


def aggregate_top_features(
    sae_rows_path: Path,
    solved: dict[str, bool],
    out_all: Path,
    out_solved_vs_unsolved: Path,
    top_n: int = 200,
) -> None:
    totals_by_layer: dict[int, Counter[str]] = defaultdict(Counter)
    solved_totals: dict[int, Counter[str]] = defaultdict(Counter)
    unsolved_totals: dict[int, Counter[str]] = defaultdict(Counter)
    solved_counts: Counter[int] = Counter()
    unsolved_counts: Counter[int] = Counter()

    for row in read_jsonl(sae_rows_path):
        layer = int(row["layer"])
        is_solved = bool(solved.get(row["task_id"], False))
        if is_solved:
            solved_counts[layer] += 1
        else:
            unsolved_counts[layer] += 1
        for feature_id, value in row["features"].items():
            totals_by_layer[layer][feature_id] += float(value)
            if is_solved:
                solved_totals[layer][feature_id] += float(value)
            else:
                unsolved_totals[layer][feature_id] += float(value)

    all_rows = []
    compare_rows = []
    for layer, counter in sorted(totals_by_layer.items()):
        for rank, (feature_id, value) in enumerate(counter.most_common(top_n), start=1):
            all_rows.append({"layer": layer, "rank": rank, "feature_id": feature_id, "total_activation": value})
        feature_ids = set(solved_totals[layer]) | set(unsolved_totals[layer])
        scored = []
        for feature_id in feature_ids:
            solved_mean = solved_totals[layer][feature_id] / max(solved_counts[layer], 1)
            unsolved_mean = unsolved_totals[layer][feature_id] / max(unsolved_counts[layer], 1)
            scored.append((solved_mean - unsolved_mean, feature_id, solved_mean, unsolved_mean))
        scored.sort(reverse=True)
        for rank, (diff, feature_id, solved_mean, unsolved_mean) in enumerate(scored[:top_n], start=1):
            compare_rows.append(
                {
                    "layer": layer,
                    "rank": rank,
                    "feature_id": feature_id,
                    "solved_mean_activation": solved_mean,
                    "unsolved_mean_activation": unsolved_mean,
                    "solved_minus_unsolved": diff,
                }
            )

    write_csv(out_all, all_rows, ["layer", "rank", "feature_id", "total_activation"])
    write_csv(
        out_solved_vs_unsolved,
        compare_rows,
        [
            "layer",
            "rank",
            "feature_id",
            "solved_mean_activation",
            "unsolved_mean_activation",
            "solved_minus_unsolved",
        ],
    )
    print(f"Wrote {out_all}")
    print(f"Wrote {out_solved_vs_unsolved}")


def sparse_dot(a: dict[str, float], b: dict[str, float]) -> float:
    if len(a) > len(b):
        a, b = b, a
    return sum(value * b.get(key, 0.0) for key, value in a.items())


def sparse_norm2(a: dict[str, float]) -> float:
    return sum(value * value for value in a.values())


def cosine(a: dict[str, float], b: dict[str, float], eps: float = 1e-12) -> float:
    return sparse_dot(a, b) / ((sparse_norm2(a) ** 0.5) * (sparse_norm2(b) ** 0.5) + eps)


def weighted_jaccard(a: dict[str, float], b: dict[str, float], eps: float = 1e-12) -> float:
    keys = set(a) | set(b)
    numerator = sum(min(a.get(key, 0.0), b.get(key, 0.0)) for key in keys)
    denominator = sum(max(a.get(key, 0.0), b.get(key, 0.0)) for key in keys)
    return numerator / (denominator + eps)


def tanimoto(a: dict[str, float], b: dict[str, float], eps: float = 1e-12) -> float:
    dot = sparse_dot(a, b)
    return dot / (sparse_norm2(a) + sparse_norm2(b) - dot + eps)


def combined_metric(a: dict[str, float], b: dict[str, float], variant: str) -> tuple[float, float, float, float]:
    c = cosine(a, b)
    j = weighted_jaccard(a, b)
    t = tanimoto(a, b)
    if variant == "cos07":
        score = 0.70 * c + 0.15 * j + 0.15 * t
    elif variant == "cos01":
        score = 0.10 * c + 0.45 * j + 0.45 * t
    else:
        raise ValueError(variant)
    return score, c, j, t
