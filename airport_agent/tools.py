"""Tool definitions the LLM can call, plus the dispatcher that runs them.

A "tool" is just: a name, a description (the model reads it to decide when to
call it), a JSON schema for the arguments, and a Python function. The model
never runs code itself - it asks us to run a tool, we run it and send back
the JSON result.
"""
from __future__ import annotations

import json
from typing import Any, Callable

from . import analysis, config
from .metrics import METRIC_DEFINITIONS, data_vintage
from .scoring import PROFILES, score_airports


def _score(iatas: list[str] | None = None, region: str | None = None, profile: str = "terminal_expansion",
           weights: dict[str, float] | None = None, min_passengers: int = 100_000) -> dict:
    if not iatas:
        if not region:
            raise ValueError("Provide either 'iatas' or 'region'.")
        found = analysis.find_airports(region=region, min_passengers=min_passengers, limit=50)
        iatas = [a["iata"] for a in found["airports"]]
    out = score_airports(iatas, profile=profile, weight_overrides=weights)
    out["data_vintage"] = data_vintage()
    return out


def _methodology() -> dict:
    return {
        "profiles": {
            name: {"description": p["description"],
                   "components": [{"key": c.key, "label": c.label, "metrics": list(c.metrics),
                                   "default_weight": c.weight, "rationale": c.rationale}
                                  for c in p["components"]]}
            for name, p in PROFILES.items()
        },
        "normalisation": ("Each metric is converted to a national percentile vs US airports with >= "
                          f"{config.SCORING_UNIVERSE_MIN_PAX:,} passengers; component = mean percentile; "
                          "composite = weighted mean; missing components are dropped and weights renormalised "
                          "(reported as 'confidence')."),
        "assumptions": {
            "long_haul_miles": config.LONG_HAUL_MILES,
            "target_load_factor": config.TARGET_LOAD_FACTOR,
            "passenger_route_min_pax_per_departure": config.PASSENGER_ROUTE_MIN_PAX_PER_DEP,
            "regions": config.REGIONS,
        },
        "metric_definitions": METRIC_DEFINITIONS,
        "not_modelled": ["construction cost", "airport finances / debt capacity", "lease & use agreements",
                         "land availability", "competition from nearby airports", "regulatory approvals"],
        "data_vintage": data_vintage(),
    }


IATA = {"type": "string", "description": "3-letter IATA airport code, e.g. 'SFO'."}

TOOLS: list[dict[str, Any]] = [
    {
        "name": "find_airports",
        "description": "Look up US airports by name/city text, state, or region (US Census divisions such as "
                       "'new england', 'pacific', 'mountain'). Use this to resolve names like 'Santa Ana' or 'LA' "
                       "to IATA codes, or to list candidates in a region. Results are sorted by passenger volume.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Free text matched against code, name and city."},
                "region": {"type": "string", "enum": sorted(config.REGIONS)},
                "state": {"type": "string", "description": "2-letter state code."},
                "min_passengers": {"type": "integer", "description": "Minimum last-12-month passengers."},
                "limit": {"type": "integer", "description": "Max results (default 15)."},
            },
        },
    },
    {
        "name": "get_airport_profile",
        "description": "All KPIs for one airport (traffic, load factor, growth, delays, taxi times, peak-hour load), "
                       "each with its national percentile and definition, plus the last 12 months of passengers "
                       "and seats and any curated analyst notes.",
        "input_schema": {"type": "object", "properties": {"iata": IATA}, "required": ["iata"]},
    },
    {
        "name": "score_airports",
        "description": "Deterministic 0-100 scoring and ranking of airports. profile='terminal_expansion' ranks "
                       "investment attractiveness for terminal/gate expansion; profile='congestion' measures how "
                       "congested airports are. Returns per-component percentiles, weights and point contributions "
                       "so you can explain exactly why an airport ranks where it does. Pass either explicit iatas "
                       "or a region. Optionally override component weights to run a sensitivity check.",
        "input_schema": {
            "type": "object",
            "properties": {
                "iatas": {"type": "array", "items": IATA},
                "region": {"type": "string", "enum": sorted(config.REGIONS)},
                "profile": {"type": "string", "enum": sorted(PROFILES)},
                "weights": {"type": "object", "additionalProperties": {"type": "number"},
                            "description": "Optional {component_key: weight}; renormalised to sum to 1."},
                "min_passengers": {"type": "integer",
                                   "description": "When using region: minimum passengers to include (default 100000)."},
            },
            "required": ["profile"],
        },
    },
    {
        "name": "get_route_mix",
        "description": "Route-distance mix for departures from one airport: share of long-haul flights, distance "
                       "bands, and top long-haul destinations, split into passenger flights vs all flights "
                       "including international cargo. Includes coverage caveats.",
        "input_schema": {
            "type": "object",
            "properties": {"iata": IATA,
                           "long_haul_miles": {"type": "integer",
                                               "description": f"Threshold in statute miles (default {config.LONG_HAUL_MILES})."}},
            "required": ["iata"],
        },
    },
    {
        "name": "estimate_unmet_demand",
        "description": "Estimate unmet (spilled) demand at an airport: seat gap vs a target load factor, growth of "
                       "passengers vs seats, recovery vs 2019, congestion signals with percentiles, rule-based "
                       "findings explaining WHY, live FAA events and curated context notes.",
        "input_schema": {
            "type": "object",
            "properties": {"iata": IATA,
                           "target_load_factor": {"type": "number",
                                                  "description": f"Default {config.TARGET_LOAD_FACTOR}."}},
            "required": ["iata"],
        },
    },
    {
        "name": "get_live_faa_status",
        "description": "Live FAA National Airspace System status: ground stops, ground delay programs, arrival/"
                       "departure delays and closures. Optionally filter to specific airports.",
        "input_schema": {"type": "object", "properties": {"iatas": {"type": "array", "items": IATA}}},
    },
    {
        "name": "get_methodology",
        "description": "Scoring methodology, weights, assumptions, metric definitions, what is NOT modelled, and "
                       "data vintage. Call this when the user asks how scores work or what assumptions are used.",
        "input_schema": {"type": "object", "properties": {}},
    },
]

HANDLERS: dict[str, Callable[..., dict]] = {
    "find_airports": analysis.find_airports,
    "get_airport_profile": analysis.airport_profile,
    "score_airports": _score,
    "get_route_mix": analysis.route_mix,
    "estimate_unmet_demand": analysis.unmet_demand,
    "get_live_faa_status": lambda iatas=None: analysis.live_faa_status(iatas),
    "get_methodology": _methodology,
}


def run_tool(name: str, args: dict) -> tuple[str, bool]:
    """Execute a tool call. Returns (json_text, is_error).

    Errors are returned to the model (not raised) so it can recover, e.g. by
    looking up the right airport code and trying again.
    """
    handler = HANDLERS.get(name)
    if handler is None:
        return json.dumps({"error": f"Unknown tool {name}"}), True
    try:
        return json.dumps(handler(**args), default=str), False
    except (ValueError, TypeError, KeyError) as exc:
        return json.dumps({"error": str(exc)}), True
