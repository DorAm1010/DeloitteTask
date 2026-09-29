import json

from airport_agent.tools import run_tool


def chart(args):
    out, is_error = run_tool("show_chart", args)
    return json.loads(out), is_error


def test_score_breakdown_matches_scoring_model():
    spec, err = chart({"chart": "score_breakdown", "iatas": ["BOS", "PVD"]})
    assert not err
    c = spec["chart"]
    score = json.loads(run_tool("score_airports", {"iatas": ["BOS", "PVD"], "profile": "terminal_expansion"})[0])
    for i, r in enumerate(score["results"]):
        assert c["labels"][i].startswith(r["iata"])
        stacked = sum(s["values"][i] for s in c["series"])
        assert abs(stacked - r["score"]) < 0.5  # segments add up to the score


def test_monthly_trend_and_route_mix_shapes():
    trend, err = chart({"chart": "monthly_trend", "iatas": ["SFO", "LAX"], "metric": "load_factor"})
    assert not err and len(trend["chart"]["labels"]) == 24 and len(trend["chart"]["series"]) == 2
    assert all(v is None or 0 < v < 1 for v in trend["chart"]["series"][0]["values"])
    mix, err = chart({"chart": "route_mix", "iatas": ["ANC"]})
    passenger, all_flights = mix["chart"]["series"]
    assert all(a >= p for p, a in zip(passenger["values"], all_flights["values"]))


def test_chart_input_errors_go_back_to_the_model():
    _, err = chart({"chart": "route_mix", "iatas": ["ANC", "SFO"]})
    assert err
    _, err = chart({"chart": "pie", "iatas": ["ANC"]})
    assert err
