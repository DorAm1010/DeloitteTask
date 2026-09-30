import json

from airport_agent.tools import TOOLS, HANDLERS, run_tool


def test_every_tool_has_a_handler():
    assert {t["name"] for t in TOOLS} == set(HANDLERS)


def test_bad_input_becomes_tool_error_not_exception():
    out, is_error = run_tool("get_airport_profile", {"iata": "ZZZ"})
    assert is_error and "find_airports" in json.loads(out)["error"]
    out, is_error = run_tool("no_such_tool", {})
    assert is_error


def test_name_resolution():
    out, _ = run_tool("find_airports", {"query": "santa ana"})
    assert json.loads(out)["airports"][0]["iata"] == "SNA"


def test_route_mix_shares_are_bounded():
    out, is_error = run_tool("get_route_mix", {"iata": "ANC"})
    d = json.loads(out)
    assert not is_error
    for view in ("passenger_flights", "all_flights_incl_international_cargo"):
        assert 0 <= d[view]["long_haul_share"] <= 1
    # Counting international cargo can only add long-haul departures.
    assert (d["all_flights_incl_international_cargo"]["long_haul_departures"]
            >= d["passenger_flights"]["long_haul_departures"])


def test_unmet_demand_seat_gap_math():
    d = json.loads(run_tool("estimate_unmet_demand", {"iata": "SFO", "target_load_factor": 0.8})[0])
    expected = max(0, d["passengers_l12m"] / 0.8 - d["seats_l12m"])
    assert abs(d["estimated_seat_gap_annual"] - expected) <= 1
    low, high = d["seat_gap_range"]["annual_seats"]
    assert low <= d["estimated_seat_gap_annual"] <= high


def test_score_echoes_request_so_answers_can_cite_it():
    d = json.loads(run_tool("score_airports", {"region": "new england", "weights": {"scale": 0.4}})[0])
    assert d["request"] == {"region": "new england", "min_passengers": 100_000, "weights_requested": {"scale": 0.4}}
