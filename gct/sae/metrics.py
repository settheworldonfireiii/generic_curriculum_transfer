from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class MetricResult:
    score: float
    sae_cosine: float
    weighted_jaccard: float
    tanimoto: float
    structure_cosine: float = 0.0


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
        raise ValueError(f"Unknown SAE metric variant: {variant}")
    return score, c, j, t


def metric_result(a: dict[str, float], b: dict[str, float], variant: str) -> MetricResult:
    score, c, j, t = combined_metric(a, b, variant)
    return MetricResult(score=score, sae_cosine=c, weighted_jaccard=j, tanimoto=t)


def arc_weighted_metric(
    target_features: dict[str, float],
    anchor_features: dict[str, float],
    target_structure: dict[str, float],
    anchor_structure: dict[str, float],
) -> MetricResult:
    sae_cosine = cosine(target_features, anchor_features)
    jaccard = weighted_jaccard(target_features, anchor_features)
    structure = cosine(target_structure, anchor_structure)
    tanimoto_value = tanimoto(target_features, anchor_features)
    score = 0.70 * sae_cosine + 0.20 * jaccard + 0.10 * structure
    return MetricResult(
        score=score,
        sae_cosine=sae_cosine,
        weighted_jaccard=jaccard,
        tanimoto=tanimoto_value,
        structure_cosine=structure,
    )

