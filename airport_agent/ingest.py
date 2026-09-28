"""Download public aviation data and reduce it to small, analysis-ready tables.

Run once (takes a few minutes, mostly the BTS on-time files):

    python -m airport_agent.ingest            # everything, last 12 months of on-time data
    python -m airport_agent.ingest --only t100 # just one source

Why a batch ingest step instead of calling APIs live on every question?
  * BTS files are large (~30 MB per month) and the servers are slow/fragile.
  * Answers become reproducible: every number traces to a file with a date.
  * The agent stays fast and cheap; only the FAA delay feed is fetched live.
"""
from __future__ import annotations

import argparse
import io
import logging
import zipfile
from datetime import date

import pandas as pd
import requests

from . import config

log = logging.getLogger("ingest")


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _get(url: str, **kwargs) -> requests.Response:
    resp = requests.get(url, timeout=config.HTTP_TIMEOUT, **kwargs)
    resp.raise_for_status()
    return resp


def socrata_query(base: str, dataset: str, params: dict, page_size: int = 50_000) -> pd.DataFrame:
    """Page through a Socrata (SODA) endpoint. SoQL lets us filter/aggregate server-side."""
    frames, offset = [], 0
    while True:
        page = dict(params, **{"$limit": page_size, "$offset": offset})
        rows = _get(f"{base}/{dataset}.json", params=page).json()
        if not rows:
            break
        frames.append(pd.DataFrame(rows))
        if len(rows) < page_size:
            break
        offset += page_size
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def haversine_miles(lat1, lon1, lat2, lon2):
    """Great-circle distance in statute miles (vectorised with pandas/numpy inputs)."""
    import numpy as np
    r = 3958.8
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dphi = p2 - p1
    dlmb = np.radians(lon2) - np.radians(lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dlmb / 2) ** 2
    return 2 * r * np.arcsin(np.sqrt(a))


# --------------------------------------------------------------------------- #
# 1. Airport reference data (OurAirports)
# --------------------------------------------------------------------------- #
def ingest_airports() -> None:
    """Names, locations and states for every airport with an IATA code."""
    df = pd.read_csv(io.StringIO(_get(config.OURAIRPORTS_URL).text), keep_default_na=False)
    df = df[(df["iata_code"].str.len() == 3) & (df["type"] != "closed")]
    world = df[["iata_code", "name", "municipality", "iso_country", "latitude_deg", "longitude_deg"]]
    world = world.rename(columns={"iata_code": "iata", "municipality": "city", "iso_country": "country",
                                  "latitude_deg": "lat", "longitude_deg": "lon"})
    # Several rows can share an IATA code (e.g. a heliport); keep the largest airport type.
    rank = {"large_airport": 0, "medium_airport": 1, "small_airport": 2}
    df["_rank"] = df["type"].map(rank).fillna(9)
    df = df.sort_values("_rank").drop_duplicates("iata_code")
    world = world.loc[world["iata"].isin(df["iata_code"])].drop_duplicates("iata")
    world.to_csv(config.PROCESSED_DIR / "airports_world.csv", index=False)

    us = df[df["iso_country"] == "US"].copy()
    us["state"] = us["iso_region"].str.replace("US-", "", regex=False)
    us = us.rename(columns={"iata_code": "iata", "municipality": "city", "latitude_deg": "lat",
                            "longitude_deg": "lon", "type": "size_class"})
    cols = ["iata", "name", "city", "state", "lat", "lon", "size_class", "scheduled_service", "keywords"]
    us[cols].to_csv(config.PROCESSED_DIR / "airports_us.csv", index=False)
    log.info("airports: %d US, %d world", len(us), len(world))


# --------------------------------------------------------------------------- #
# 2. BTS T-100 airport summary (monthly passengers, seats, departures)
# --------------------------------------------------------------------------- #
T100_FIELDS = {
    "origin_airport_code": "iata",
    "reporting_month": "month",
    "total_departures": "departures",
    "total_passengers": "passengers",
    "total_seats": "seats",
    "total_freight_lbs": "freight_lbs",
    "total_distance_flight_sm": "avg_distance_mi",
    "domestic_departures": "dom_departures",
    "domestic_passengers": "dom_passengers",
    "domestic_seats": "dom_seats",
    "outbound_international": "intl_departures",
    "outbound_international_1": "intl_passengers",
    "outbound_international_seats": "intl_seats",
    "outbound_international_3": "intl_avg_distance_mi",
}


def ingest_t100(since: str = "2019-01-01") -> None:
    """Monthly totals per origin airport since 2019 (so we can compare with pre-COVID)."""
    df = socrata_query(config.BTS_SOCRATA, config.T100_AIRPORT_DATASET, {
        "$select": ",".join(T100_FIELDS),
        "$where": f"reporting_month >= '{since}'",
        "$order": "reporting_month, origin_airport_code",
    })
    df = df.rename(columns=T100_FIELDS)
    df["month"] = pd.to_datetime(df["month"]).dt.strftime("%Y-%m")
    num = [c for c in df.columns if c not in ("iata", "month")]
    df[num] = df[num].apply(pd.to_numeric, errors="coerce").fillna(0)
    # Drop airstrips with negligible traffic to keep the committed file small.
    busy = df.groupby("iata")["passengers"].sum()
    df = df[df["iata"].isin(busy[busy >= 10_000].index)]
    df.to_csv(config.PROCESSED_DIR / "t100_airport_monthly.csv", index=False)
    log.info("t100: %d rows, %s .. %s", len(df), df["month"].min(), df["month"].max())


# --------------------------------------------------------------------------- #
# 3. BTS T-100 international routes (US gateway <-> foreign airport)
# --------------------------------------------------------------------------- #
def ingest_international(months: int = 12) -> None:
    """Last N months of international departures and passengers per route.

    The feed is directionless (both directions summed), so we halve it to
    approximate departures *from* the US airport. Distances are computed from
    coordinates because the feed does not carry them.
    """
    latest = socrata_query(config.DOT_SOCRATA, config.T100_INTL_DATASET,
                           {"$select": "max(data_dte) as latest"})["latest"].iloc[0]
    end = pd.Timestamp(latest)
    start = (end - pd.DateOffset(months=months - 1)).strftime("%Y-%m-%d")
    df = socrata_query(config.DOT_SOCRATA, config.T100_INTL_DATASET, {
        "$select": "usg_apt, fg_apt, type, sum(total) as total",
        "$where": f"data_dte >= '{start}' AND type in ('Departures', 'Passengers', 'Seats')",
        "$group": "usg_apt, fg_apt, type",
    })
    df["total"] = pd.to_numeric(df["total"])
    wide = df.pivot_table(index=["usg_apt", "fg_apt"], columns="type", values="total",
                          aggfunc="sum", fill_value=0).reset_index()
    wide.columns = [str(c).lower() for c in wide.columns]
    wide = wide.rename(columns={"usg_apt": "origin", "fg_apt": "dest"})
    for col in ("departures", "passengers", "seats"):
        wide[col] = (wide.get(col, 0) / 2).round()   # directionless -> one-way estimate

    world = pd.read_csv(config.PROCESSED_DIR / "airports_world.csv", keep_default_na=False)
    coords = world.set_index("iata")[["lat", "lon"]]
    o = coords.reindex(wide["origin"]).to_numpy()
    d = coords.reindex(wide["dest"]).to_numpy()
    wide["distance_mi"] = haversine_miles(o[:, 0], o[:, 1], d[:, 0], d[:, 1]).round()
    wide["period_start"], wide["period_end"] = start[:7], end.strftime("%Y-%m")
    wide.to_csv(config.PROCESSED_DIR / "routes_international.csv", index=False)
    log.info("international: %d routes, %s .. %s", len(wide), start[:7], end.strftime("%Y-%m"))


# --------------------------------------------------------------------------- #
# 4. BTS on-time performance (every domestic flight of the large US carriers)
# --------------------------------------------------------------------------- #
ONTIME_COLS = ["Month", "Year", "Origin", "Dest", "CRSDepTime", "CRSArrTime", "DepDelay", "DepDel15",
               "ArrDel15", "TaxiOut", "TaxiIn", "Cancelled", "Diverted", "Distance",
               "NASDelay", "WeatherDelay", "CarrierDelay", "LateAircraftDelay"]


def _month_iter(end: date, months: int):
    y, m = end.year, end.month
    for _ in range(months):
        yield y, m
        m -= 1
        if m == 0:
            y, m = y - 1, 12


def _latest_ontime_month() -> date:
    """BTS publishes with a ~2 month lag; probe backwards for the newest file."""
    today = date.today()
    for y, m in _month_iter(today, 8):
        url = config.ONTIME_URL.format(year=y, month=m)
        if requests.head(url, timeout=config.HTTP_TIMEOUT).status_code == 200:
            return date(y, m, 1)
    raise RuntimeError("No recent BTS on-time file found")


def _load_ontime_month(y: int, m: int) -> pd.DataFrame:
    config.RAW_DIR.mkdir(parents=True, exist_ok=True)
    path = config.RAW_DIR / f"ontime_{y}_{m:02d}.zip"
    if not path.exists():
        log.info("downloading on-time %d-%02d ...", y, m)
        with requests.get(config.ONTIME_URL.format(year=y, month=m), stream=True,
                          timeout=config.HTTP_TIMEOUT) as r:
            r.raise_for_status()
            with open(path, "wb") as fh:
                for chunk in r.iter_content(1 << 20):
                    fh.write(chunk)
    with zipfile.ZipFile(path) as zf:  # the zip also contains a readme.html
        csv_name = next(n for n in zf.namelist() if n.endswith(".csv"))
        with zf.open(csv_name) as fh:
            return pd.read_csv(fh, usecols=ONTIME_COLS, low_memory=False)


def ingest_ontime(months: int = 12, keep_raw: bool = False) -> None:
    """Aggregate flight-level records into three small tables:

    ontime_airport_monthly.csv  delays, taxi times, cancellations per airport-month
    routes_domestic.csv         flights and distance per origin-destination pair
    hourly_profile.csv          average scheduled movements per local hour (peak-hour load)
    """
    end = _latest_ontime_month()
    month_list = list(_month_iter(end, months))
    first = f"{month_list[-1][0]}-{month_list[-1][1]:02d}"
    dep_parts, arr_parts, route_parts, hour_parts = [], [], [], []
    for y, m in month_list:
        df = _load_ontime_month(y, m)
        df["month"] = f"{y}-{m:02d}"
        flown = df[df["Cancelled"] == 0]

        dep_parts.append(df.groupby(["Origin", "month"]).agg(
            sched_departures=("Cancelled", "size"),
            cancelled=("Cancelled", "sum"),
            dep_delayed_15=("DepDel15", "sum"),
            dep_delay_min_total=("DepDelay", lambda s: s.clip(lower=0).sum()),
            taxi_out_total=("TaxiOut", "sum"),
            taxi_out_n=("TaxiOut", "count"),
        ).reset_index().rename(columns={"Origin": "iata"}))

        arr_parts.append(df.groupby(["Dest", "month"]).agg(
            sched_arrivals=("Cancelled", "size"),
            arr_delayed_15=("ArrDel15", "sum"),
            arr_reported=("ArrDel15", "count"),  # flown, non-diverted arrivals
            taxi_in_total=("TaxiIn", "sum"),
            taxi_in_n=("TaxiIn", "count"),
            nas_delay_min=("NASDelay", "sum"),
            weather_delay_min=("WeatherDelay", "sum"),
        ).reset_index().rename(columns={"Dest": "iata"}))

        route_parts.append(flown.groupby(["Origin", "Dest"]).agg(
            flights=("Cancelled", "size"), distance_mi=("Distance", "median")).reset_index())

        # Scheduled movements by local hour (departures at origin + arrivals at destination).
        dep_h = df.assign(hour=(df["CRSDepTime"] // 100) % 24).groupby(["Origin", "hour"]).size()
        arr_h = df.assign(hour=(df["CRSArrTime"] // 100) % 24).groupby(["Dest", "hour"]).size()
        hour_parts.append(pd.concat([dep_h.rename_axis(["iata", "hour"]),
                                     arr_h.rename_axis(["iata", "hour"])]).groupby(level=[0, 1]).sum()
                          .rename("movements").reset_index())
        if not keep_raw:
            (config.RAW_DIR / f"ontime_{y}_{m:02d}.zip").unlink(missing_ok=True)

    dep = pd.concat(dep_parts)
    arr = pd.concat(arr_parts)
    monthly = dep.merge(arr, on=["iata", "month"], how="outer").fillna(0)
    monthly.to_csv(config.PROCESSED_DIR / "ontime_airport_monthly.csv", index=False)

    routes = pd.concat(route_parts).groupby(["Origin", "Dest"]).agg(
        flights=("flights", "sum"), distance_mi=("distance_mi", "median")).reset_index()
    routes = routes.rename(columns={"Origin": "origin", "Dest": "dest"})
    routes["period_start"], routes["period_end"] = first, end.strftime("%Y-%m")
    routes.to_csv(config.PROCESSED_DIR / "routes_domestic.csv", index=False)

    # Average over every day in the window (not just days with flights in that hour).
    total_days = sum(pd.Period(f"{y}-{m:02d}").days_in_month for y, m in month_list)
    hours = pd.concat(hour_parts).groupby(["iata", "hour"])["movements"].sum().reset_index()
    hours["avg_movements"] = (hours["movements"] / total_days).round(2)
    hours[["iata", "hour", "avg_movements"]].to_csv(config.PROCESSED_DIR / "hourly_profile.csv", index=False)
    log.info("on-time: %s .. %s, %d airports, %d routes", first, end.strftime("%Y-%m"),
             monthly["iata"].nunique(), len(routes))


SOURCES = {
    "airports": ingest_airports,
    "t100": ingest_t100,
    "international": ingest_international,
    "ontime": ingest_ontime,
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--only", choices=SOURCES, help="run a single source")
    parser.add_argument("--months", type=int, default=12, help="months of on-time data (default 12)")
    parser.add_argument("--keep-raw", action="store_true", help="keep downloaded on-time zips")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    for name, fn in SOURCES.items():
        if args.only and name != args.only:
            continue
        if name == "ontime":
            fn(months=args.months, keep_raw=args.keep_raw)
        else:
            fn()


if __name__ == "__main__":
    main()
