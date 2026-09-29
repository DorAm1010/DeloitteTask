"""Central configuration: paths, model settings and analyst assumptions.

Three sources, each for a different kind of setting:
  * .env                 secrets and model settings (API key, model, effort)
  * config/scoring.yaml  analyst assumptions and scoring weights (editable without code)
  * this file            paths, data-source URLs and pricing (engineering constants)
"""
from __future__ import annotations

import os
from pathlib import Path

import yaml
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
ENV_FILE = ROOT / ".env"
# Load the repo's own .env (not whichever one is found from the current directory).
# Real environment variables still win over the file.
load_dotenv(ENV_FILE)
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"            # large downloads, git-ignored
PROCESSED_DIR = DATA_DIR / "processed"  # small aggregated tables, committed
REFERENCE_DIR = DATA_DIR / "reference"  # hand-curated context with sources
PROMPTS_DIR = ROOT / "prompts"
SCORING_CONFIG_PATH = Path(os.getenv("SCORING_CONFIG", ROOT / "config" / "scoring.yaml"))
LOG_DIR = Path(os.getenv("AGENT_LOG_DIR", ROOT / "logs"))  # JSONL telemetry, git-ignored

# --- LLM settings -----------------------------------------------------------
MODEL = os.getenv("ANTHROPIC_MODEL", "claude-opus-5")
EFFORT = os.getenv("AGENT_EFFORT", "high")  # low | medium | high | xhigh | max
MAX_AGENT_STEPS = int(os.getenv("AGENT_MAX_STEPS", "10"))
# Server-side refusal fallback (beta). If the model declines a request, the API
# re-runs it on Anthropic's recommended fallback model instead of failing.
USE_REFUSAL_FALLBACK = os.getenv("AGENT_REFUSAL_FALLBACK", "1") == "1"
SYSTEM_PROMPT = (PROMPTS_DIR / "system.md").read_text().strip()

# USD per million tokens (input, output), used for the cost-per-answer estimate.
# Cache writes cost 1.25x input and cache reads 0.1x input.
MODEL_PRICING = {
    "claude-opus-5": (5.0, 25.0),
    "claude-opus-5-5": (4.0, 20.0),
    "claude-sonnet-5": (2.0, 10.0),
    "claude-haiku-4-5": (1.0, 5.0),
    "claude-fable-5-1": (10.0, 50.0),
}

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

# --- Analyst assumptions (config/scoring.yaml, documented in DESIGN.md) ------
SCORING_CONFIG: dict = yaml.safe_load(SCORING_CONFIG_PATH.read_text())
_assumptions = SCORING_CONFIG["assumptions"]
LONG_HAUL_MILES: int = _assumptions["long_haul_miles"]
DISTANCE_BANDS: list[tuple[float, float, str]] = [tuple(b) for b in _assumptions["distance_bands"]]
PASSENGER_ROUTE_MIN_PAX_PER_DEP: float = _assumptions["passenger_route_min_pax_per_dep"]
TARGET_LOAD_FACTOR: float = _assumptions["target_load_factor"]
SCORING_UNIVERSE_MIN_PAX: int = _assumptions["scoring_universe_min_pax"]
REGIONS: dict[str, list[str]] = SCORING_CONFIG["regions"]
