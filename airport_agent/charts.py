"""Chart specs for the UI, built from the same deterministic functions as the answers.

The model only chooses the chart type and airports; every number comes from
our code, so a chart can never show a figure the model invented. A spec is
library-agnostic JSON (type, labels, series) that the web UI draws with Chart.js.
"""
from __future__ import annotations

from . import analysis, config
from .metrics import load_tables
from .scoring import score_airports

CHART_TYPES = {
    "score_breakdown": "Why airports rank where they do: each airport's score split into component contributions.",
    "monthly_trend": "A metric (load_factor, passengers or seats) month by month over the last 24 months.",
    "route_mix": "One airport's departures by distance band: passenger flights vs all flights incl. cargo.",
}
TREND_METRICS = ("load_factor", "passengers", "seats")
MAX_AIRPORTS = {"score_breakdown": 10, "monthly_trend": 4, "route_mix": 1}


def _score_breakdown(iatas: list[str], profile: str, weights: dict | None) -> dict:
    scored = score_airports(iatas, profile=profile, weight_overrides=weights)
    results = scored["results"]
    if not results:
        raise ValueError(f"No data for {scored['not_found']}. Use find_airports to check the codes.")
    components = results[0]["components"] if results else []
    return {
        "type": "bar", "stacked": True, "horizontal": True, "value_format": "number",
        "title": f"{profile.replace('_', ' ').title()} score by component (0-100)",
        "labels": [f"{r['iata']} ({r['score']})" for r in results],
        "series": [{"name": c["label"],
                    "values": [next(x["contribution"] for x in r["components"] if x["component"] == c["component"])
                               or 0 for r in results]}
                   for c in components],
        "source": scored["reference_universe"],
    }


def _monthly_trend(iatas: list[str], metric: str) -> dict:
    if metric not in TREND_METRICS:
        raise ValueError(f"metric must be one of {TREND_METRICS}")
    t100 = load_tables()["t100"]
    months = sorted(t100["month"].unique())[-24:]
    series = []
    for code in iatas:
        rows = t100[(t100["iata"] == code) & t100["month"].isin(months)].set_index("month")
        if rows.empty:
            raise ValueError(f"No monthly data for {code}")
        if metric == "load_factor":
            vals = (rows["passengers"] / rows["seats"].where(rows["seats"] > 0)).round(4)
        else:
            vals = rows[metric]
        series.append({"name": code, "values": [None if m not in vals.index or vals[m] != vals[m]
                                                else float(vals[m]) for m in months]})
    return {
        "type": "line", "value_format": "percent" if metric == "load_factor" else "number",
        "title": f"{metric.replace('_', ' ').capitalize()}, monthly ({months[0]} to {months[-1]})",
        "labels": months, "series": series, "source": "BTS T-100 airport totals (enplaned, all carriers)",
    }


def _route_mix(iata: str) -> dict:
    mix = analysis.route_mix(iata)
    if "error" in mix:
        raise ValueError(mix["error"])
    bands = [b for _, _, b in config.DISTANCE_BANDS]
    by_band = mix["departures_by_band"]
    return {
        "type": "bar", "value_format": "number",
        "title": f"{iata} departures by distance band (last 12 months)",
        "labels": bands,
        "series": [{"name": "Passenger flights", "values": [by_band["passenger_flights"][b] for b in bands]},
                   {"name": "All flights incl. international cargo",
                    "values": [by_band["all_flights_incl_international_cargo"][b] for b in bands]}],
        "source": "BTS on-time (domestic) + T-100 international",
    }


def build_chart(chart: str, iatas: list[str], profile: str = "terminal_expansion",
                metric: str = "load_factor", weights: dict | None = None) -> dict:
    if chart not in CHART_TYPES:
        raise ValueError(f"Unknown chart '{chart}'. Options: {list(CHART_TYPES)}")
    iatas = [c.upper() for c in iatas]
    if not iatas:
        raise ValueError("Provide at least one airport code.")
    if len(iatas) > MAX_AIRPORTS[chart]:
        raise ValueError(f"{chart} supports at most {MAX_AIRPORTS[chart]} airport(s); pick the most relevant.")
    if chart == "score_breakdown":
        spec = _score_breakdown(iatas, profile, weights)
    elif chart == "monthly_trend":
        spec = _monthly_trend(iatas, metric)
    else:
        spec = _route_mix(iatas[0])
    return {"chart": spec,
            "note": "The chart is now displayed to the user under your answer. Refer to it briefly; do not "
                    "repeat its numbers as a table."}
