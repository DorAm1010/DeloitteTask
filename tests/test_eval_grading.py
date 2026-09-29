import json

from tests.evals.grading import extract_answer_numbers, grade, grounding


def test_number_extraction_units_and_skips():
    nums = {n.text: n.value for n in extract_answer_numbers(
        "Load factor 82.6%, 1.0M seats, 3,000 mi, top 3 airports, data from 2025, 450k passengers")}
    assert nums["82.6%"] == 82.6
    assert nums["1.0M"] == 1_000_000
    assert nums["3,000"] == 3000
    assert nums["450k"] == 450_000
    assert "3" not in nums and "2025" not in nums  # small ints and years are not data


def test_grounding_accepts_rounding_and_percent_forms():
    tool_out = [json.dumps({"load_factor": 0.8255, "seat_gap": 1029860, "note": "above 3,000 statute miles"})]
    share, missing = grounding("LF is 82.6% (about 83%), gap ~1.0M seats, threshold 3,000 mi", tool_out)
    assert share == 1.0 and missing == []


def test_grounding_flags_invented_numbers():
    share, missing = grounding("Load factor is 91.2% and delays are 47 minutes", [json.dumps({"load_factor": 0.8255})])
    assert share == 0.0
    assert missing == ["91.2%", "47"]


def test_grade_checks_tools_args_mentions_and_caveats():
    case = {"id": "x", "expect_tools": ["get_route_mix(iata=ANC)", "find_airports|get_methodology"],
            "forbid_tools": ["score_airports"], "must_mention": ["cargo"]}
    calls = [{"name": "get_route_mix", "input": {"iata": "ANC", "long_haul_miles": 3000},
              "output": json.dumps({"share": 0.0518})},
             {"name": "get_methodology", "input": {}, "output": "{}"}]
    good = grade(case, "About 5.2% of passenger flights; cargo changes it. Assumptions & caveats: ...", calls)
    assert good.passed, good.checks
    bad = grade(case, "About 12.5% of flights.", calls[:1])
    assert not bad.passed
    assert not bad.checks["calls find_airports|get_methodology"]
    assert not bad.checks["mentions /cargo/"]
    assert not bad.checks["has caveats section"]
