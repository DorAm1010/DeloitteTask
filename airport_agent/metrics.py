"""Turn the processed tables into one row of KPIs per airport.

Pure pandas, no LLM. Every metric here has a plain-English definition in
METRIC_DEFINITIONS so the agent (and the design doc) can explain it.
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np
import pandas as pd

from . import config

METRIC_DEFINITIONS = {
    "passengers_l12m": "Enplaned passengers (all carriers, domestic + international) in the last 12 months of T-100 data.",
    "seats_l12m": "Seats flown out of the airport in the same 12 months.",
    "departures_l12m": "Departures performed (passenger and cargo) in the same 12 months.",
    "load_factor": "Passengers / seats. High values mean flights are full; above ~80-85% demand starts to spill.",
    "pax_growth_yoy": "Last-12-month passengers vs the 12 months before (fraction, 0.05 = +5%).",
    "pax_added_l12m": "Absolute passengers added: last 12 months minus the 12 months before (negative = shrinking).",
    "seat_growth_yoy": "Same comparison for seats. Passengers growing faster than seats = tightening supply.",
    "pax_vs_2019": "Last-12-month passengers vs calendar 2019 (pre-COVID baseline).",
    "intl_pax_share": "Share of enplaned passengers on international departures.",
    "freight_lbs_l12m": "Freight enplaned (lbs), last 12 months.",
    "dep_delay_rate": "Share of flown departures leaving 15+ min late (BTS on-time, large US carriers only).",
    "arr_delay_rate": "Share of arrivals 15+ min late.",
    "avg_taxi_out_min": "Average gate-to-wheels-off time. Rises with runway/taxiway congestion.",
    "nas_delay_min_per_arrival": "Minutes of 'National Aviation System' delay per scheduled arrival - delay the FAA attributes to traffic volume, ATC and non-extreme weather. The best public proxy for airside congestion.",
    "cancel_rate": "Share of scheduled departures cancelled.",
    "daily_movements": "Average scheduled arrivals + departures per day (large US carriers only).",
    "peak_hour_movements": "Average scheduled movements in the busiest local hour.",
    "peak_to_average": "Peak hour / average hour over the 16 busiest hours. High = banked hub or constrained schedule.",
    "ontime_coverage": "Share of T-100 departures that the on-time data covers. Low values mean delay metrics describe only part of the traffic (e.g. cargo or regional carriers excluded).",
}


@lru_cache(maxsize=1)
def load_tables() -> dict[str, pd.DataFrame]:
    p = config.PROCESSED_DIR
    tables = {
        "airports": pd.read_csv(p / "airports_us.csv", keep_default_na=False),
        "world": pd.read_csv(p / "airports_world.csv", keep_default_na=False),
        "t100": pd.read_csv(p / "t100_airport_monthly.csv", keep_default_na=False),
        "intl_routes": pd.read_csv(p / "routes_international.csv", keep_default_na=False),
    }
    optional = {"ontime": "ontime_airport_monthly.csv", "dom_routes": "routes_domestic.csv",
                "hourly": "hourly_profile.csv"}
    for key, name in optional.items():
        path = p / name
        tables[key] = pd.read_csv(path, keep_default_na=False) if path.exists() else pd.DataFrame()
    return tables


def _months_back(latest: str, n: int) -> list[str]:
    end = pd.Period(latest, "M")
    return [str(end - i) for i in range(n)]


def data_vintage() -> dict[str, str]:
    """Which period each source covers - surfaced in every answer."""
    t = load_tables()
    vintage = {"t100_airport_totals": f"{_months_back(t['t100']['month'].max(), 12)[-1]} to {t['t100']['month'].max()}"}
    if len(t["intl_routes"]):
        r = t["intl_routes"].iloc[0]
        vintage["t100_international_routes"] = f"{r['period_start']} to {r['period_end']}"
    if len(t["ontime"]):
        vintage["bts_on_time"] = f"{t['ontime']['month'].min()} to {t['ontime']['month'].max()}"
    return vintage


def _t100_features(t100: pd.DataFrame) -> pd.DataFrame:
    latest = t100["month"].max()
    l12 = _months_back(latest, 12)
    p12 = _months_back(str(pd.Period(latest, "M") - 12), 12)
    cur = t100[t100["month"].isin(l12)].groupby("iata").sum(numeric_only=True)
    prev = t100[t100["month"].isin(p12)].groupby("iata").sum(numeric_only=True)
    y19 = t100[t100["month"].str.startswith("2019")].groupby("iata").sum(numeric_only=True)

    f = pd.DataFrame(index=cur.index)
    f["passengers_l12m"] = cur["passengers"]
    f["seats_l12m"] = cur["seats"]
    f["departures_l12m"] = cur["departures"]
    f["freight_lbs_l12m"] = cur["freight_lbs"]
    f["load_factor"] = cur["passengers"] / cur["seats"].replace(0, np.nan)
    f["pax_growth_yoy"] = cur["passengers"] / prev["passengers"].reindex(cur.index).replace(0, np.nan) - 1
    f["pax_added_l12m"] = cur["passengers"] - prev["passengers"].reindex(cur.index)
    f["seat_growth_yoy"] = cur["seats"] / prev["seats"].reindex(cur.index).replace(0, np.nan) - 1
    f["pax_vs_2019"] = cur["passengers"] / y19["passengers"].reindex(cur.index).replace(0, np.nan) - 1
    f["intl_pax_share"] = cur["intl_passengers"] / cur["passengers"].replace(0, np.nan)
    return f


def _ontime_features(ontime: pd.DataFrame, hourly: pd.DataFrame, t100_departures: pd.Series) -> pd.DataFrame:
    if ontime.empty:
        return pd.DataFrame()
    g = ontime.groupby("iata").sum(numeric_only=True)
    n_days = sum(pd.Period(m, "M").days_in_month for m in ontime["month"].unique())
    f = pd.DataFrame(index=g.index)
    flown = (g["sched_departures"] - g["cancelled"]).replace(0, np.nan)
    f["dep_delay_rate"] = g["dep_delayed_15"] / flown
    f["arr_delay_rate"] = g["arr_delayed_15"] / g["arr_reported"].replace(0, np.nan)
    f["avg_taxi_out_min"] = g["taxi_out_total"] / g["taxi_out_n"].replace(0, np.nan)
    f["nas_delay_min_per_arrival"] = g["nas_delay_min"] / g["sched_arrivals"].replace(0, np.nan)
    f["cancel_rate"] = g["cancelled"] / g["sched_departures"].replace(0, np.nan)
    f["daily_movements"] = (g["sched_departures"] + g["sched_arrivals"]) / n_days
    # Months in the on-time window may differ from the T-100 window; this is an
    # approximate coverage ratio (annualised flown departures vs T-100 departures).
    f["ontime_coverage"] = (flown * 365 / n_days) / t100_departures.reindex(g.index).replace(0, np.nan)
    if not hourly.empty:
        peak = hourly.groupby("iata")["avg_movements"].max()
        busy16 = hourly.sort_values("avg_movements", ascending=False).groupby("iata").head(16)
        f["peak_hour_movements"] = peak
        f["peak_to_average"] = peak / busy16.groupby("iata")["avg_movements"].mean()
    return f


@lru_cache(maxsize=1)
def airport_features() -> pd.DataFrame:
    """One row per US airport that has T-100 traffic, indexed by IATA code."""
    t = load_tables()
    f = _t100_features(t["t100"])
    f = f.join(_ontime_features(t["ontime"], t["hourly"], f["departures_l12m"]), how="left")
    ref = t["airports"].set_index("iata")[["name", "city", "state", "size_class"]]
    f = ref.join(f, how="inner")
    f["ontime_coverage"] = f["ontime_coverage"].clip(upper=1.0)
    return f.sort_values("passengers_l12m", ascending=False)
