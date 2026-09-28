"""Central configuration: paths, model settings and analyst assumptions.

Everything an analyst might want to tune (thresholds, regions, weights) lives
here so it is visible in one place and never hidden inside a prompt.
"""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"            # large downloads, git-ignored
PROCESSED_DIR = DATA_DIR / "processed"  # small aggregated tables, committed
REFERENCE_DIR = DATA_DIR / "reference"  # hand-curated context with sources

# --- LLM settings -----------------------------------------------------------
MODEL = os.getenv("ANTHROPIC_MODEL", "claude-opus-5")
EFFORT = os.getenv("AGENT_EFFORT", "high")  # low | medium | high | xhigh | max
MAX_AGENT_STEPS = int(os.getenv("AGENT_MAX_STEPS", "10"))
# Server-side refusal fallback (beta). If the model declines a request, the API
# re-runs it on Anthropic's recommended fallback model instead of failing.
USE_REFUSAL_FALLBACK = os.getenv("AGENT_REFUSAL_FALLBACK", "1") == "1"

# --- Public data sources ----------------------------------------------------
BTS_SOCRATA = "https://data.bts.gov/resource"
DOT_SOCRATA = "https://datahub.transportation.gov/resource"
T100_AIRPORT_DATASET = "r495-tyji"   # AFF - T100 Segment Summary By Origin Airport (monthly)
T100_INTL_DATASET = "udzf-9fvh"      # T-100 international, US gateway <-> foreign airport (monthly)
ONTIME_URL = ("https://transtats.bts.gov/PREZIP/"
              "On_Time_Reporting_Carrier_On_Time_Performance_1987_present_{year}_{month}.zip")
OURAIRPORTS_URL = "https://davidmegginson.github.io/ourairports-data/airports.csv"
FAA_NAS_STATUS_URL = "https://nasstatus.faa.gov/api/airport-status-information"
HTTP_TIMEOUT = 60

# --- Analyst assumptions (documented in DESIGN.md) ---------------------------
# Long haul: >= 3,000 statute miles (roughly 6+ hours block time). There is no
# single industry standard; this is exposed as a tool parameter.
LONG_HAUL_MILES = 3000
DISTANCE_BANDS = [(0, 500, "short (<500 mi)"),
                  (500, 1500, "medium (500-1,499 mi)"),
                  (1500, 3000, "long domestic (1,500-2,999 mi)"),
                  (3000, 100_000, "long haul (3,000+ mi)")]
# An international route averaging fewer passengers than this per departure is
# treated as cargo-dominated (the T-100 international feed has no service class).
PASSENGER_ROUTE_MIN_PAX_PER_DEP = 20
# Load factor considered "comfortably served". Above it, extra demand tends to
# spill (passengers who wanted a seat but could not get one at a sane fare).
TARGET_LOAD_FACTOR = 0.80
# Reference universe for percentile scoring: airports with at least this many
# enplaned passengers in the last 12 months (roughly FAA small hub and up).
SCORING_UNIVERSE_MIN_PAX = 500_000

REGIONS: dict[str, list[str]] = {
    "new england": ["CT", "ME", "MA", "NH", "RI", "VT"],
    "mid-atlantic": ["NJ", "NY", "PA"],
    "south atlantic": ["DE", "DC", "FL", "GA", "MD", "NC", "SC", "VA", "WV"],
    "east north central": ["IL", "IN", "MI", "OH", "WI"],
    "west north central": ["IA", "KS", "MN", "MO", "NE", "ND", "SD"],
    "east south central": ["AL", "KY", "MS", "TN"],
    "west south central": ["AR", "LA", "OK", "TX"],
    "mountain": ["AZ", "CO", "ID", "MT", "NV", "NM", "UT", "WY"],
    "pacific": ["AK", "CA", "HI", "OR", "WA"],
    "west coast": ["CA", "OR", "WA"],
}
