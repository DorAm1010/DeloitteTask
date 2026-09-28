# Airport Investment Intelligence Agent

A conversational AI agent that helps analysts find US airports where modernisation or expansion is most likely to pay
off, based on public flight and passenger data. It ranks airports with a **deterministic, explainable scoring model**,
and uses Claude only to understand questions, call the right tools, and explain the results.

> Design, scoring methodology, tradeoffs and where AI is used: **[DESIGN.md](DESIGN.md)**

Example questions it answers:
- *Which airports in New England are strong candidates for terminal expansion?*
- *Compare LA and Santa Ana airport congestion levels.*
- *What is the percentage of long haul flights out of Anchorage airport?*
- *What is the unmet flight demand in SFO airport and why?*
- Follow-ups like *"Why is Bangor ranked above Boston?"* or *"Re-rank with growth weighted double"*

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env                                   # then put your ANTHROPIC_API_KEY in .env

uvicorn airport_agent.server:app                       # open http://localhost:8000
# or, in the terminal:
python -m airport_agent.cli --trace
```

Processed data is committed under `data/processed/`, so the agent works immediately. To refresh from the public
sources (takes ~5 minutes; downloads ~400 MB of BTS on-time files, then deletes them):

```bash
python -m airport_agent.ingest                  # all sources
python -m airport_agent.ingest --only t100      # a single source
```

Tests (no API key or network needed; the agent loop is tested with a scripted fake model):

```bash
pytest
```

## Features
- **Public data:** BTS T-100 (passengers, seats, departures), BTS on-time performance (~7M flights: delays,
  taxi times, distances), T-100 international routes, OurAirports, and the **live FAA airport status feed**.
- **Deterministic scoring:** percentile-based composite scores with per-component contributions, confidence when
  data is missing, and a weight-sensitivity check (rank ranges).
- **Explainable answers:** every answer includes assumptions and caveats. The UI's "How I got this" panel shows
  every tool call with its inputs and outputs.
- **Conversational:** follow-up questions keep context; analysts can change weights or thresholds in plain English.
- **Voice (bonus):** speak questions and have answers read aloud, using the browser Web Speech API
  (Chrome/Edge/Safari).

## Project layout

```
airport_agent/
  config.py     model settings, data sources, analyst assumptions (thresholds, regions)
  ingest.py     download public data -> data/processed/*.csv
  metrics.py    per-airport KPIs + definitions
  scoring.py    deterministic scoring profiles, percentiles, sensitivity analysis
  analysis.py   route mix / long haul, unmet demand, airport profile, live FAA status
  tools.py      tool schemas the LLM sees + dispatcher
  agent.py      the Claude tool-use loop and system prompt
  server.py     FastAPI backend;  web/index.html  chat UI with voice
  cli.py        terminal chat
data/processed/ small aggregated tables (committed)
data/reference/ curated, sourced context notes (explanations only, never scores)
tests/          scoring, tools and agent-loop tests
```

## Configuration (`.env`)

| Variable | Default | Meaning |
|---|---|---|
| `ANTHROPIC_API_KEY` | (required) | Claude API key |
| `ANTHROPIC_MODEL` | `claude-opus-5` | Model used by the agent |
| `AGENT_EFFORT` | `high` | Reasoning effort (`low`/`medium` = faster, cheaper) |
| `AGENT_MAX_STEPS` | `10` | Maximum tool-call rounds per question |
| `AGENT_REFUSAL_FALLBACK` | `1` | Server-side fallback model if a request is declined |
