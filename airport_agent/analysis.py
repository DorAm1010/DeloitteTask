"""Deterministic analyses behind each agent tool.

Each function returns plain JSON-serialisable dicts that include the numbers,
the assumptions used and caveats about data coverage, so the LLM can explain
rather than guess.
"""
from __future__ import annotations

import json
import time
import xml.etree.ElementTree as ET
from functools import lru_cache

import numpy as np
import pandas as pd
import requests

from . import config
from .metrics import METRIC_DEFINITIONS, airport_features, data_vintage, load_tables
from .scoring import percentile_vs, reference_universe


def _clean(value):
    """Round floats and turn NaN into None so results serialise cleanly."""
    if isinstance(value, (float, np.floating)):
        return None if np.isnan(value) else round(float(value), 4)
    if isinstance(value, (np.integer,)):
        return int(value)
    return value


@lru_cache(maxsize=1)
def curated_notes() -> dict:
    path = config.REFERENCE_DIR / "airport_notes.json"
    return json.loads(path.read_text()) if path.exists() else {}


def _require(code: str) -> pd.Series:
    f = airport_features()
    code = code.upper()
    if code not in f.index:
        raise ValueError(f"No US traffic data for '{code}'. Use find_airports to look up the IATA code.")
    return f.loc[code]


# --------------------------------------------------------------------------- #
# Lookup
# --------------------------------------------------------------------------- #
def find_airports(query: str | None = None, region: str | None = None, state: str | None = None,
                  min_passengers: int = 0, limit: int = 15) -> dict:
    f = airport_features()
    mask = f["passengers_l12m"] >= min_passengers
    if region:
        states = config.REGIONS.get(region.lower())
        if states is None:
            raise ValueError(f"Unknown region '{region}'. Known: {sorted(config.REGIONS)}")
        mask &= f["state"].isin(states)
    if state:
        mask &= f["state"] == state.upper()
    if query:
        q = query.lower()
        text = (f.index.str.lower() + " " + f["name"].str.lower() + " " + f["city"].str.lower())
        mask &= text.str.contains(q, regex=False)
    hits = f[mask].head(limit)
    return {
        "count": int(mask.sum()),
        "airports": [{"iata": code, "name": r["name"], "city": r["city"], "state": r["state"],
                      "passengers_l12m": int(r["passengers_l12m"])} for code, r in hits.iterrows()],
        "note": "Sorted by passengers. Region definitions: " + (
            f"{region} = {config.REGIONS[region.lower()]}" if region else "US Census divisions."),
    }


# --------------------------------------------------------------------------- #
# Airport profile
# --------------------------------------------------------------------------- #
def airport_profile(iata: str) -> dict:
    code = iata.upper()
    row = _require(code)
    universe = reference_universe()
    metrics = {}
    for key, definition in METRIC_DEFINITIONS.items():
        value = row.get(key, np.nan)
        pct = percentile_vs(universe[key], value) if key in universe else float("nan")
        metrics[key] = {"value": _clean(value), "national_percentile": _clean(pct), "definition": definition}

    t100 = load_tables()["t100"]
    series = t100[t100["iata"] == code].sort_values("month").tail(12)
    monthly = [{"month": r["month"], "passengers": int(r["passengers"]), "seats": int(r["seats"]),
                "load_factor": _clean(r["passengers"] / r["seats"] if r["seats"] else np.nan)}
               for _, r in series.iterrows()]
    return {
        "iata": code, "name": row["name"], "city": row["city"], "state": row["state"],
        "metrics": metrics, "monthly_last_12": monthly,
        "curated_notes": curated_notes().get(code),
        "data_vintage": data_vintage(),
        "percentile_universe": f"{len(universe)} US airports with >= {config.SCORING_UNIVERSE_MIN_PAX:,} passengers",
    }


# --------------------------------------------------------------------------- #
# Route mix / long-haul share
# --------------------------------------------------------------------------- #
def _band(distance: float) -> str:
    for lo, hi, label in config.DISTANCE_BANDS:
        if lo <= distance < hi:
            return label
    return "unknown"


def route_mix(iata: str, long_haul_miles: int = config.LONG_HAUL_MILES) -> dict:
    code = iata.upper()
    row = _require(code)
    t = load_tables()
    dom = t["dom_routes"][t["dom_routes"]["origin"] == code] if len(t["dom_routes"]) else pd.DataFrame()
    intl = t["intl_routes"][t["intl_routes"]["origin"] == code].copy()
    intl = intl[intl["departures"] > 0]
    intl["pax_per_dep"] = intl["passengers"] / intl["departures"]
    intl["kind"] = np.where(intl["pax_per_dep"] >= config.PASSENGER_ROUTE_MIN_PAX_PER_DEP,
                            "passenger", "cargo-dominated")

    legs = []
    for _, r in dom.iterrows():
        legs.append(("domestic", "passenger", r["dest"], float(r["flights"]), float(r["distance_mi"])))
    for _, r in intl.iterrows():
        legs.append(("international", r["kind"], r["dest"], float(r["departures"]), float(r["distance_mi"])))
    legs = pd.DataFrame(legs, columns=["scope", "kind", "dest", "departures", "distance_mi"])
    if legs.empty:
        return {"iata": code, "error": "No route-level data for this airport."}
    legs = legs.dropna(subset=["distance_mi"])
    legs["band"] = legs["distance_mi"].apply(_band)
    legs["long_haul"] = legs["distance_mi"] >= long_haul_miles

    def share(df: pd.DataFrame) -> dict:
        total = df["departures"].sum()
        lh = df.loc[df["long_haul"], "departures"].sum()
        return {"departures": int(total), "long_haul_departures": int(lh),
                "long_haul_share": _clean(lh / total if total else np.nan)}

    pax = legs[legs["kind"] == "passenger"]
    bands = (pax.groupby("band")["departures"].sum() / pax["departures"].sum()).round(4).to_dict() if len(pax) else {}
    top_lh = (legs[legs["long_haul"]].sort_values("departures", ascending=False).head(10)
              .assign(departures=lambda d: d["departures"].astype(int))
              [["dest", "scope", "kind", "departures", "distance_mi"]].to_dict("records"))

    t100_deps = row["departures_l12m"]
    covered = legs["departures"].sum()
    return {
        "iata": code, "name": row["name"],
        "long_haul_threshold_miles": long_haul_miles,
        "passenger_flights": share(pax),
        "all_flights_incl_international_cargo": share(legs),
        "passenger_distance_bands": bands,
        "departures_by_band": {
            "passenger_flights": {b: int(pax.loc[pax["band"] == b, "departures"].sum())
                                  for _, _, b in config.DISTANCE_BANDS},
            "all_flights_incl_international_cargo": {b: int(legs.loc[legs["band"] == b, "departures"].sum())
                                                     for _, _, b in config.DISTANCE_BANDS},
        },
        "top_long_haul_routes": top_lh,
        "coverage": {
            "t100_departures_l12m_all_carriers": int(t100_deps),
            "departures_in_route_data": int(covered),
            "approx_coverage": _clean(min(covered / t100_deps, 1.0) if t100_deps else np.nan),
        },
        "assumptions": [
            f"Long haul = great-circle/route distance >= {long_haul_miles:,} statute miles (~6h+). No single industry "
            "standard exists (IATA uses 6h; some airlines use 3,000 mi).",
            "Domestic routes come from BTS on-time data: scheduled passenger flights of the large US carriers only "
            "(excludes cargo carriers like FedEx/UPS and small regionals).",
            "International routes come from T-100 international (all carriers, incl. cargo). The feed has no "
            f"service class, so a route averaging < {config.PASSENGER_ROUTE_MIN_PAX_PER_DEP} passengers per "
            "departure is classed as cargo-dominated.",
            "International counts are directionless in the source and halved to estimate departures.",
        ],
        "data_vintage": data_vintage(),
    }


# --------------------------------------------------------------------------- #
# Unmet demand
# --------------------------------------------------------------------------- #
def unmet_demand(iata: str, target_load_factor: float = config.TARGET_LOAD_FACTOR) -> dict:
    code = iata.upper()
    row = _require(code)
    universe = reference_universe()
    t100 = load_tables()["t100"]
    a = t100[t100["iata"] == code]
    y19 = a[a["month"].str.startswith("2019")][["passengers", "seats"]].sum()

    pax, seats = row["passengers_l12m"], row["seats_l12m"]
    lf = row["load_factor"]
    seats_needed = pax / target_load_factor
    seat_gap = max(0.0, seats_needed - seats)
    peak = a.sort_values("month").tail(12).assign(lf=lambda d: d["passengers"] / d["seats"].replace(0, np.nan))
    peak_row = peak.loc[peak["lf"].idxmax()] if len(peak) and peak["lf"].notna().any() else None

    def pct(metric):
        return _clean(percentile_vs(universe[metric], row.get(metric, np.nan)))

    signals = {
        "load_factor": {"value": _clean(lf), "national_percentile": pct("load_factor")},
        "pax_growth_yoy": {"value": _clean(row["pax_growth_yoy"])},
        "seat_growth_yoy": {"value": _clean(row["seat_growth_yoy"])},
        "pax_vs_2019": {"value": _clean(row["pax_vs_2019"])},
        "seats_vs_2019": {"value": _clean(seats / y19["seats"] - 1 if y19["seats"] else np.nan)},
        "nas_delay_min_per_arrival": {"value": _clean(row.get("nas_delay_min_per_arrival")),
                                      "national_percentile": pct("nas_delay_min_per_arrival")},
        "avg_taxi_out_min": {"value": _clean(row.get("avg_taxi_out_min")),
                             "national_percentile": pct("avg_taxi_out_min")},
        "dep_delay_rate": {"value": _clean(row.get("dep_delay_rate")),
                           "national_percentile": pct("dep_delay_rate")},
        "cancel_rate": {"value": _clean(row.get("cancel_rate")), "national_percentile": pct("cancel_rate")},
    }

    # Deterministic interpretation rules, so "why" is grounded in thresholds, not vibes.
    findings = []
    if lf >= target_load_factor:
        findings.append(f"Load factor {lf:.1%} is above the {target_load_factor:.0%} comfort level: flights are full "
                        "and some travellers are likely priced out or turned away (demand spill).")
    else:
        findings.append(f"Load factor {lf:.1%} is below {target_load_factor:.0%}: on average there are empty seats, "
                        "so unmet demand (if any) is concentrated in peak months/hours rather than year-round.")
    gap = row["pax_growth_yoy"] - row["seat_growth_yoy"]
    if not np.isnan(gap) and gap > 0.01:
        findings.append(f"Passengers grew {gap:+.1%} faster than seats over the last year: supply is not keeping up.")
    if not np.isnan(row["pax_vs_2019"]) and row["pax_vs_2019"] < -0.05:
        findings.append(f"Traffic is still {row['pax_vs_2019']:.1%} vs 2019, so the gap is not simply more demand "
                        "than ever - check capacity constraints or weaker demand.")
    nas_p = signals["nas_delay_min_per_arrival"]["national_percentile"]
    if nas_p is not None and nas_p >= 75:
        findings.append(f"ATC/volume (NAS) delay per arrival is in the top quarter nationally (percentile {nas_p:.1f}): "
                        "the binding constraint is likely airside (runways/airspace), not just terminal space.")
    if peak_row is not None and peak_row["lf"] >= 0.85:
        findings.append(f"Peak month {peak_row['month']} ran at {peak_row['lf']:.1%} load factor.")

    return {
        "iata": code, "name": row["name"],
        "method": ("Seat gap = seats needed to carry last-12-month passengers at the target load factor minus seats "
                   "actually flown. It is a floor on unmet demand: it ignores travellers who never tried to book "
                   "because fares were high or flights unavailable."),
        "target_load_factor": target_load_factor,
        "passengers_l12m": int(pax), "seats_l12m": int(seats),
        "estimated_seat_gap_annual": int(round(seat_gap)),
        "estimated_seat_gap_per_day": int(round(seat_gap / 365)),
        "seat_gap_pct_of_current_seats": _clean(seat_gap / seats if seats else np.nan),
        "peak_month": None if peak_row is None else {"month": peak_row["month"], "load_factor": _clean(peak_row["lf"])},
        "signals": signals,
        "findings": findings,
        "live_faa_status": live_faa_status([code]).get("events", []),
        "curated_notes": curated_notes().get(code),
        "data_vintage": data_vintage(),
    }


# --------------------------------------------------------------------------- #
# Live FAA status (the only live call; cached for 5 minutes)
# --------------------------------------------------------------------------- #
_faa_cache: dict = {"at": 0.0, "events": None}


def _fetch_faa_events() -> list[dict]:
    if _faa_cache["events"] is not None and time.time() - _faa_cache["at"] < 300:
        return _faa_cache["events"]
    xml = requests.get(config.FAA_NAS_STATUS_URL, timeout=15).text
    root = ET.fromstring(xml)
    events = []
    for dt in root.findall("Delay_type"):
        kind = dt.findtext("Name")
        for item in dt.iter():
            arpt = item.findtext("ARPT")
            if arpt and item.tag not in ("Delay_type",):
                detail = {child.tag: (child.text or "").strip() for child in item if child.tag != "ARPT"
                          and len(child) == 0}
                for child in item:  # nested arrival/departure delay blocks
                    if len(child):
                        detail[child.get("Type", child.tag)] = {c.tag: c.text for c in child}
                events.append({"airport": arpt, "type": kind, **detail})
    _faa_cache.update(at=time.time(), events=events)
    return events


def live_faa_status(codes: list[str] | None = None) -> dict:
    try:
        events = _fetch_faa_events()
    except Exception as exc:  # network issues must not break the conversation
        return {"error": f"FAA status feed unavailable: {exc}", "events": []}
    if codes:
        wanted = {c.upper() for c in codes}
        events = [e for e in events if e["airport"] in wanted]
    return {"source": config.FAA_NAS_STATUS_URL, "retrieved_utc": time.strftime("%Y-%m-%d %H:%M", time.gmtime()),
            "events": events,
            "note": "Live snapshot of FAA ground stops, ground delay programs, delays and closures. "
                    "An empty list means no active FAA event for the airport right now."}
