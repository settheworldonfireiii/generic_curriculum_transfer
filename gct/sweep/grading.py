from __future__ import annotations

import math
import re


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
    if not text:
        return None
    boxed = extract_boxed(text)
    if boxed:
        return boxed
    patterns = [
        r"####\s*([^\n]+)",
        r"(?:final answer|answer is|therefore|so the answer is)\s*[:=]?\s*([^\n.]+)",
    ]
    for pattern in patterns:
        matches = re.findall(pattern, text, flags=re.IGNORECASE)
        if matches:
            return matches[-1].strip()
    number_matches = re.findall(r"[-+]?\d+(?:\.\d+)?(?:/\d+)?", text)
    if number_matches:
        return number_matches[-1].strip()
    return text.strip()


def normalize_answer(answer: str | None) -> str:
    if answer is None:
        return ""
    text = str(answer).strip()
    text = text.replace("\\left", "").replace("\\right", "")
    text = text.replace("\\,", "").replace("\\!", "")
    text = text.replace("$", "")
    text = re.sub(r"\\text\{([^}]*)\}", r"\1", text)
    text = re.sub(r"\\mathrm\{([^}]*)\}", r"\1", text)
    text = re.sub(r"\\frac\{([^{}]+)\}\{([^{}]+)\}", r"\1/\2", text)
    text = text.replace(" ", "")
    text = text.rstrip(".")
    return text.lower()


def numeric_value(text: str | None) -> float | None:
    normalized = normalize_answer(text)
    frac = re.fullmatch(r"([-+]?\d+(?:\.\d+)?)/([-+]?\d+(?:\.\d+)?)", normalized)
    if frac:
        denom = float(frac.group(2))
        if denom == 0:
            return None
        return float(frac.group(1)) / denom
    try:
        return float(normalized)
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

