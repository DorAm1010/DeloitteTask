import math

import pandas as pd
import pytest

from airport_agent.scoring import percentile_vs, resolve_weights, score_airports


def test_percentile_vs_basic():
    u = pd.Series([1, 2, 3, 4])
    assert percentile_vs(u, 0) == 0
    assert percentile_vs(u, 5) == 100
    assert percentile_vs(u, 2) == pytest.approx(37.5)  # 1 below + half of 1 tie, out of 4
    assert math.isnan(percentile_vs(u, float("nan")))


def test_weights_renormalise_and_validate():
    w = resolve_weights("terminal_expansion", {"growth": 2.0})
    assert sum(w.values()) == pytest.approx(1.0)
    assert w["growth"] > w["scale"]
    with pytest.raises(ValueError):
        resolve_weights("terminal_expansion", {"not_a_component": 1})


def test_scoring_is_deterministic_and_explainable():
    a = score_airports(["BOS", "PVD", "BDL"])
    b = score_airports(["BOS", "PVD", "BDL"])
    assert a == b
    ranks = [r["rank"] for r in a["results"]]
    assert ranks == sorted(ranks)
    for r in a["results"]:
        contributions = sum(c["contribution"] for c in r["components"] if c["contribution"] is not None)
        assert contributions == pytest.approx(r["score"], abs=0.5)  # rounding
        assert 0 <= r["score"] <= 100
        lo, hi = r["rank_range"]
        assert lo <= r["rank"] <= hi


def test_missing_components_lower_confidence():
    # HVN has no large-carrier on-time data, so the congestion component is missing.
    res = score_airports(["HVN"])["results"][0]
    assert res["confidence"] < 1
    assert any("on-time" in f.lower() for f in res["flags"])


def test_unknown_airport_reported():
    out = score_airports(["BOS", "ZZZ"])
    assert out["not_found"] == ["ZZZ"]
