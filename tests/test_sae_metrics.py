from gct.sae.metrics import (
    arc_weighted_metric,
    combined_metric,
    cosine,
    tanimoto,
    weighted_jaccard,
)


def test_combined_metric_matches_previous_math_weights() -> None:
    a = {"f1": 2.0, "f2": 1.0}
    b = {"f1": 1.0, "f3": 3.0}

    score, c, j, t = combined_metric(a, b, "cos07")

    assert c == cosine(a, b)
    assert j == weighted_jaccard(a, b)
    assert t == tanimoto(a, b)
    assert score == 0.70 * c + 0.15 * j + 0.15 * t


def test_arc_weighted_metric_uses_sae_cosine_jaccard_and_structure_cosine() -> None:
    sae_a = {"f1": 1.0, "f2": 1.0}
    sae_b = {"f1": 1.0, "f3": 1.0}
    structure_a = {"category:algebra": 1.0}
    structure_b = {"category:algebra": 1.0}

    result = arc_weighted_metric(sae_a, sae_b, structure_a, structure_b)

    assert result.score == 0.70 * result.sae_cosine + 0.20 * result.weighted_jaccard + 0.10 * result.structure_cosine
    assert result.structure_cosine > 0.999

