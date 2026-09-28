"""Deterministic, explainable scoring.

The LLM never invents a score. It calls these functions and explains the output.

Method (same for every profile):
  1. Each component is one or more raw KPIs from metrics.airport_features().
  2. Each KPI is converted to a percentile (0-100) against a fixed national
     reference universe: US airports with >= SCORING_UNIVERSE_MIN_PAX enplaned
     passengers in the last 12 months. A national yardstick (instead of ranking
     only within the airports asked about) means "BOS scores 80" means the same
     thing whether you compare it with PVD or with ATL.
  3. Component score = mean of its KPI percentiles (flipped if lower is better).
  4. Composite score = weighted mean of component scores. If a component has no
     data (e.g. no on-time coverage), its weight is redistributed and the
     result's `confidence` drops to the share of weight that had data.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import config
from .metrics import airport_features


@dataclass(frozen=True)
class Component:
    key: str
    label: str
    metrics: tuple[str, ...]
    weight: float
    rationale: str
    higher_is_better: tuple[bool, ...] = field(default=())

    def direction(self, i: int) -> bool:
        return self.higher_is_better[i] if self.higher_is_better else True


PROFILES: dict[str, dict] = {
    "terminal_expansion": {
        "description": "Where would added terminal/gate capacity most likely be filled and pay back? "
                       "Rewards growing, full, large, capacity-stressed airports.",
        "components": [
            Component("growth", "Demand growth", ("pax_growth_yoy", "pax_added_l12m"), 0.25,
                      "Passenger growth fills new terminal capacity and grows per-passenger revenue (fees, "
                      "concessions, parking). Measured both as % growth (momentum) and passengers added "
                      "(volume), so a small airport growing fast off a tiny base does not dominate."),
            Component("utilization", "Seat utilization", ("load_factor",), 0.20,
                      "Full flights mean demand is already pressing on available capacity."),
            Component("supply_gap", "Demand outpacing supply", ("demand_supply_gap",), 0.10,
                      "Passengers growing faster than seats signals airlines are constrained, e.g. by gates."),
            Component("scale", "Scale", ("passengers_l12m",), 0.20,
                      "Larger passenger bases spread fixed renovation costs and reduce revenue risk."),
            Component("congestion", "Operational congestion",
                      ("avg_taxi_out_min", "nas_delay_min_per_arrival", "dep_delay_rate"), 0.15,
                      "Delays and long taxi times show infrastructure under stress. Caveat: runway/airspace "
                      "limits are not fixed by a terminal project alone."),
            Component("recovery", "Above pre-COVID peak", ("pax_vs_2019",), 0.10,
                      "Airports above 2019 volumes run facilities sized for less traffic."),
        ],
    },
    "congestion": {
        "description": "How congested is the airport today? Higher = more congested. "
                       "Combines delay, taxi time, ATC volume delay, schedule peaking and seat utilization.",
        "components": [
            Component("atc_volume_delay", "ATC / volume delay", ("nas_delay_min_per_arrival",), 0.30,
                      "NAS delay is the FAA-attributed delay from traffic volume and ATC - the cleanest congestion signal."),
            Component("taxi", "Taxi-out time", ("avg_taxi_out_min",), 0.20,
                      "Queues for the runway show up as longer taxi-out times."),
            Component("punctuality", "Departure delays", ("dep_delay_rate",), 0.20,
                      "Share of departures 15+ minutes late (all causes)."),
            Component("peaking", "Schedule peaking", ("peak_to_average",), 0.10,
                      "Sharp peaks concentrate demand into hours where capacity binds."),
            Component("utilization", "Seat utilization", ("load_factor",), 0.20,
                      "Full aircraft = passenger-side congestion (terminal, security, gates)."),
        ],
    },
}


def _feature_frame() -> pd.DataFrame:
    f = airport_features().copy()
    f["demand_supply_gap"] = f["pax_growth_yoy"] - f["seat_growth_yoy"]
    return f


def reference_universe(f: pd.DataFrame | None = None) -> pd.DataFrame:
    f = _feature_frame() if f is None else f
    return f[f["passengers_l12m"] >= config.SCORING_UNIVERSE_MIN_PAX]


def percentile_vs(universe: pd.Series, value: float) -> float:
    """Percentile of `value` within `universe` (0-100, ties count half)."""
    u = universe.dropna().to_numpy()
    if np.isnan(value) or len(u) == 0:
        return float("nan")
    return float(((u < value).mean() + 0.5 * (u == value).mean()) * 100)


def resolve_weights(profile: str, overrides: dict[str, float] | None) -> dict[str, float]:
    comps = PROFILES[profile]["components"]
    weights = {c.key: c.weight for c in comps}
    if overrides:
        unknown = set(overrides) - set(weights)
        if unknown:
            raise ValueError(f"Unknown component(s) {sorted(unknown)}; valid: {sorted(weights)}")
        weights.update({k: float(v) for k, v in overrides.items()})
    total = sum(weights.values())
    if total <= 0:
        raise ValueError("Weights must sum to a positive number")
    return {k: v / total for k, v in weights.items()}


def _flags(row: pd.Series) -> list[str]:
    flags = []
    if row["passengers_l12m"] < config.SCORING_UNIVERSE_MIN_PAX:
        flags.append(f"Below the {config.SCORING_UNIVERSE_MIN_PAX:,}-passenger reference universe; "
                     "percentiles compare it with larger airports.")
    cov = row.get("ontime_coverage")
    if pd.isna(cov):
        flags.append("No BTS on-time data (no large-carrier service): congestion metrics missing.")
    elif cov < 0.5:
        flags.append(f"On-time data covers only ~{cov:.0%} of departures (cargo/regional/foreign carriers "
                     "excluded), so delay metrics describe part of the traffic.")
    if pd.isna(row.get("pax_vs_2019")):
        flags.append("No 2019 baseline available.")
    return flags


def score_airports(codes: list[str], profile: str = "terminal_expansion",
                   weight_overrides: dict[str, float] | None = None) -> dict:
    if profile not in PROFILES:
        raise ValueError(f"Unknown profile '{profile}'. Options: {list(PROFILES)}")
    f = _feature_frame()
    codes = [c.upper() for c in codes]
    missing = [c for c in codes if c not in f.index]
    universe = reference_universe(f)
    weights = resolve_weights(profile, weight_overrides)
    comps = PROFILES[profile]["components"]

    results = []
    for code in [c for c in codes if c in f.index]:
        row = f.loc[code]
        comp_out, avail_weight, weighted_sum = [], 0.0, 0.0
        for comp in comps:
            pcts, raw = [], {}
            for i, metric in enumerate(comp.metrics):
                value = row.get(metric, np.nan)
                raw[metric] = None if pd.isna(value) else round(float(value), 4)
                p = percentile_vs(universe[metric], value)
                if not np.isnan(p):
                    pcts.append(p if comp.direction(i) else 100 - p)
            score = float(np.mean(pcts)) if pcts else float("nan")
            w = weights[comp.key]
            if pcts:
                avail_weight += w
                weighted_sum += w * score
            comp_out.append({"component": comp.key, "label": comp.label, "weight": round(w, 3),
                             "percentile": None if np.isnan(score) else round(score, 1),
                             "raw_values": raw, "why_it_matters": comp.rationale})
        composite = weighted_sum / avail_weight if avail_weight else float("nan")
        for c in comp_out:  # contribution in points to the 0-100 composite
            c["contribution"] = (None if c["percentile"] is None
                                 else round(c["weight"] / avail_weight * c["percentile"], 1))
        results.append({
            "iata": code, "name": row["name"], "city": row["city"], "state": row["state"],
            "score": None if np.isnan(composite) else round(composite, 1),
            "confidence": round(avail_weight, 2),
            "passengers_l12m": int(row["passengers_l12m"]),
            "components": comp_out, "flags": _flags(row),
        })

    results.sort(key=lambda r: -1 if r["score"] is None else r["score"], reverse=True)
    for i, r in enumerate(results, 1):
        r["rank"] = i
    _add_rank_stability(results, weights)
    return {
        "profile": profile,
        "profile_description": PROFILES[profile]["description"],
        "weights_used": {k: round(v, 3) for k, v in weights.items()},
        "reference_universe": f"{len(universe)} US airports with >= {config.SCORING_UNIVERSE_MIN_PAX:,} "
                              "enplaned passengers (last 12 months)",
        "scale": "0-100; 50 = median airport in the reference universe",
        "sensitivity_method": "Ranks recomputed under 500 random weightings centred on the weights used "
                              "(Dirichlet, seed fixed). rank_range and top3_share show how robust each rank is; "
                              "gaps of a few points are within modelling noise.",
        "results": results,
        "not_found": missing,
    }


def _add_rank_stability(results: list[dict], weights: dict[str, float], draws: int = 500, seed: int = 7) -> None:
    """How much does the ranking depend on our (subjective) weights?

    Re-scores with random weight vectors drawn around the chosen weights and
    records each airport's best/worst rank and how often it lands in the top 3.
    Deterministic thanks to the fixed seed.
    """
    if len(results) < 2:
        return
    keys = list(weights)
    pct = np.array([[next((c["percentile"] for c in r["components"] if c["component"] == k), None)
                     for k in keys] for r in results], dtype=float)
    rng = np.random.default_rng(seed)
    w = rng.dirichlet(np.array([weights[k] for k in keys]) * 20 + 0.05, size=draws)  # (draws, k)
    have = ~np.isnan(pct)                                              # (n, k)
    num = np.nan_to_num(pct)[None, :, :] * w[:, None, :]               # (draws, n, k)
    den = (have[None, :, :] * w[:, None, :]).sum(axis=2)
    scores = np.where(den > 0, num.sum(axis=2) / np.where(den > 0, den, 1), -1)
    ranks = (-scores).argsort(axis=1).argsort(axis=1) + 1              # (draws, n)
    for i, r in enumerate(results):
        r["rank_range"] = [int(ranks[:, i].min()), int(ranks[:, i].max())]
        r["top3_share"] = round(float((ranks[:, i] <= 3).mean()), 2)
