# Design: Airport Investment Intelligence Agent

## 1. Problem framing

The firm wants to find US airports where **renovation or expansion is most likely to pay off because flight and
passenger demand is outgrowing capacity**. "Profitable" can't be observed directly from public data, since we
don't have construction costs or airport finances. So the agent uses **proxies for how much latent demand a
capacity project would unlock**:

| Investment question | Observable proxy | Source |
|---|---|---|
| Is demand growing? | Passenger growth, as a % and as absolute passengers added | BTS T-100 |
| Is capacity already tight? | Load factor, passengers growing faster than seats | BTS T-100 |
| Is infrastructure stressed? | ATC/volume (NAS) delay, taxi-out time, delay rate, schedule peaking | BTS on-time |
| Is the revenue base big enough? | Passenger volume | BTS T-100 |
| Is the facility past its design point? | Traffic vs the 2019 peak | BTS T-100 |
| What's happening right now? | Ground stops and ground delay programs | FAA NAS status (live) |

The agent screens and explains. It doesn't predict returns, and it says so.

## 2. Architecture

```
            ┌──────────── offline (python -m airport_agent.ingest) ────────────┐
Public APIs │ BTS Socrata (T-100 by airport) · DOT Socrata (T-100 intl routes)  │
            │ BTS TranStats PREZIP (on-time, flight level) · OurAirports       │
            └───────────────┬───────────────────────────────────────────────────┘
                            ▼
                data/processed/*.csv   (small, versioned, dated)
                            ▼
   metrics.py ──► scoring.py ──► analysis.py        ◄── deterministic, unit-tested, no LLM
   (KPIs)        (percentile      (route mix, unmet demand,
                  scores,          profiles, live FAA feed)
                  sensitivity)
                            ▼
                tools.py  (8 tools: JSON schema + Python function)
                            ▼
                agent.py  (Claude tool-use loop, conversation memory)
                            ▼
          server.py (FastAPI) ──► web/index.html (chat + voice)     cli.py (terminal)
          (streams progress events, then the answer, as server-sent events)
```

**Why the split matters:** everything below `tools.py` is ordinary, testable Python that returns the same answer
every time. The LLM sits on top as the orchestrator and explainer. It can't change a number, only choose which
function to call and describe the result.

## 3. Data sources

| Source | Content | Coverage used | Access |
|---|---|---|---|
| BTS "T-100 Segment Summary by Origin Airport" (`data.bts.gov` r495-tyji) | Monthly departures, passengers, seats, freight, domestic/international split per airport | 2019-01 → 2026-04 | Socrata SODA API (SQL-like `$select/$where`), no key |
| DOT T-100 International (`datahub.transportation.gov` udzf-9fvh) | Departures, passengers and seats per US airport ↔ foreign airport | 2025 | Socrata, server-side `$group` aggregation |
| BTS Reporting Carrier On-Time Performance | Every domestic flight of the large US carriers: delays, taxi times, cancellations, distance | 2025-08 → 2026-07 (12 months, ~7M flights) | Static monthly zips (TranStats PREZIP) |
| OurAirports | Names, cities, states, coordinates (for distances) | current | CSV |
| FAA NAS Status | Live ground stops, ground delay programs, delays, closures | live, cached 5 min | XML API, no key |
| `data/reference/airport_notes.json` | Hand-curated context with source links (e.g. SFO's 2026 arrival-rate cut, SNA's legal passenger cap) | 2026-09-28 | Used only for explanations, never in scores |

The T-100 route-level segment table (with seats and distance per route for all carriers) would be the ideal
source. Its TranStats download form was **down for database maintenance** while this was built. That's one
concrete reason for the cached batch-ingest design.

## 4. Scoring methodology

### 4.1 Common method (`scoring.py`)
1. Each **component** is built from one or more KPIs.
2. Each KPI becomes a **percentile (0–100) against a fixed national reference universe**: the 141 US airports
   with at least 500k enplaned passengers in the last 12 months. The result reads as "better than X% of US
   airports of meaningful size".
3. Component score = mean of its KPI percentiles. Composite = **weighted mean** of the components.
4. **Missing data:** if a component has no data (for example, no large-carrier on-time coverage at HVN), it is
   dropped and the remaining weights are renormalised. `confidence` = share of the weight that had data.
5. **Sensitivity:** the ranking is recomputed under 500 random weightings (Dirichlet around the chosen weights,
   fixed seed). The result reports `rank_range` and `top3_share`, so small score gaps aren't over-read.
6. **Deterministic caveat flags**, e.g. an airport below the universe size, or low on-time coverage.

### 4.2 Profile: `terminal_expansion` (default investment screen)

| Component | KPIs | Weight | Why |
|---|---|---|---|
| Demand growth | passenger growth % YoY, passengers added | 25% | Growth fills new capacity. Using absolute passengers added as well stops tiny airports dominating on % alone |
| Seat utilization | load factor | 20% | Full flights mean demand is pressing on supply |
| Demand outpacing supply | passenger growth − seat growth | 10% | Airlines unable to add seats fast enough, e.g. gate-constrained |
| Scale | passengers (L12M) | 20% | Bigger base means lower revenue risk and fixed costs spread over more passengers |
| Operational congestion | taxi-out, NAS delay per arrival, departure delay rate | 15% | Infrastructure under stress (caveat: runway limits aren't fixed by a terminal) |
| Above pre-COVID peak | passengers vs 2019 | 10% | Facilities sized for less traffic |

### 4.3 Profile: `congestion`
NAS (ATC/volume) delay per arrival 30%, taxi-out 20%, departure-delay rate 20%, schedule peaking 10%, load
factor 20%. NAS delay gets the most weight because it is the FAA-attributed delay from volume and ATC, the
cleanest public congestion signal. Weather and airline delays are mostly excluded from it.

### 4.4 Unmet demand (`analysis.unmet_demand`)
- **Seat gap** = passengers ÷ target load factor (80%) − seats flown. This is a **floor**: it can't see
  travellers who never tried to book because fares were high or flights sold out. The 80% comfort level is a
  judgement call, so the gap is also reported across 78–82%, in seats and as a % of current seats.
- Rule-based **findings** explain the *why* against explicit thresholds: load factor above target, passengers
  outgrowing seats, still below 2019, NAS delay in the national top quartile (which points to an airside
  constraint), peak-month load factor ≥ 85%.
- Adds live FAA events and curated context.
- Example (SFO): 82.6% load factor (p90), NAS delay p99.6, taxi-out p97, peak month 89% load factor. The seat
  gap is about 1.0M seats/year (+3.2%), with a range of 0.2–1.9M (+0.7% to +5.8%). Curated context explains it: 750 ft runway spacing and the 2026 FAA
  arrival-rate cut from ~54 to ~36–42 per hour.

### 4.5 Long-haul share (`analysis.route_mix`)
- Long haul = **≥ 3,000 statute miles** (roughly 6h+). There's no single standard, so it's a tool parameter.
- Domestic legs come from on-time data (passenger flights of the large carriers). International legs come from
  T-100 international, where distance is computed from coordinates.
- The international feed has no service class, so a route averaging < 20 passengers per departure is labelled
  **cargo-dominated**. Both views are reported.
- Example (ANC): **~5% of passenger departures are long haul, but ~52% of all departures once international
  cargo is counted** (HKG, ICN, TPE, PVG freighters). The honest answer to "what % are long haul" depends on
  whether cargo counts, and the agent says so.

## 5. Where and how AI is used

| Done by the LLM (Claude) | Done by deterministic code |
|---|---|
| Understanding the question and resolving entities ("LA", "Santa Ana", "New England") via `find_airports` | All data retrieval and cleaning |
| Choosing which tools to call, in what order, possibly in parallel | Every metric, percentile, score, rank and estimate |
| Handling follow-ups using conversation history ("what if growth mattered more?" → re-score with new weights) | Sensitivity analysis and caveat flags |
| Turning tool JSON into a concise explanation with assumptions and caveats | Rule-based "findings" for unmet demand |
| Adding curated qualitative context, clearly labelled | Live FAA status parsing |

**Agent loop (`agent.py`):** a hand-written loop over the Claude Messages API. Send history + tool schemas. If
`stop_reason == "tool_use"`, run every requested tool and return all results in one `tool_result` message, then
repeat. Otherwise return the text. The full history is kept, which is what makes follow-up questions work.

**Guardrails**
- The system prompt requires every number to come from a tool result, rankings to come from `score_airports`,
  and an "Assumptions & caveats" section at the end of answers that present data or draw an inference (only
  when it is needed to read the answer correctly).
- Tool errors go back to the model as `is_error` results (e.g. an unknown airport code), so it can self-correct
  instead of crashing.
- A step limit (10) forces a final answer with `tool_choice: none`.
- The UI shows every tool call with its inputs and outputs ("How I got this"), so answers can be audited.
- Live progress: the loop emits an event before each model call and tool call, with a plain-English label per
  tool. The CLI shows a spinner and one line per step; the web UI streams them over server-sent events. The
  model's own reasoning isn't shown: it would be a model-written summary, while the tool steps are the real,
  checkable method.
- Server-side refusal fallback (beta), prompt caching of the stable prefix, and model/effort set through config.

**Model choice.** Claude, for reliable tool use and clear explanations over long JSON outputs. The model only
orchestrates and explains, so it is swappable: tools are plain Python with JSON schemas, and switching provider
means rewriting `_call_model` and the message format. I'd switch for on-premises hosting, data residency or cost.
A small classifier such as Laya (a BERT-based model that returns bounded choices with probabilities) could route
questions cheaply or flag low-confidence ones for clarification. That wasn't needed at this scale, and it would
need its own calibration eval.

**Charts.** A `show_chart` tool lets the model decide *whether* a chart helps and which of three types (score
breakdown, monthly trend, route mix). The chart's numbers are computed by `charts.py` from the same deterministic
functions, not typed by the model. The server passes the spec to the UI, which draws it with Chart.js. When to
chart is guided by the tool description and one rule in the system prompt: at most one chart, none for simple
answers, and no table repeating the chart. Evals check both directions (a chart when asked, none for a simple
fact).

**Memory and tokens (cost).** Memory is the conversation itself: each session keeps its message history
(questions, tool results, answers) in memory and resends it on every call, which is what makes follow-ups work;
nothing persists across sessions. History is therefore the main cost driver, so the stable prefix (tools, system
prompt, earlier turns) is prompt-cached and billed at about 10% on repeat calls, the data layer is cached in
process, and cost per answer is shown and logged. History isn't trimmed yet; the next step is dropping old tool
outputs or server-side compaction, plus a per-session budget.

**Configuration as data.** The system prompt lives in `prompts/system.md`, and every analyst judgement
(thresholds, regions, scoring components, weights and rationales) lives in `config/scoring.yaml`. Both can be
reviewed and diffed like code, and changed without touching Python. A client-specific investment thesis is just
another YAML file (`SCORING_CONFIG=...`).

**Observability.** Every tool call (name, input, duration, error) and every answer (tools used, latency, tokens,
estimated cost, grounding) is written as one JSON line to `logs/agent.log`. The UI shows latency, tokens and cost
under each answer.

**Live grounding check.** The eval set's hallucination check (`grounding.py`) also runs on every live answer, at no
cost: each number in the answer must match a number in the conversation's tool outputs within its displayed
rounding. Untraced numbers are listed under the answer and logged. Often they are the model's own arithmetic,
which the system prompt discourages. `python -m airport_agent.telemetry` turns the log into agent KPIs:
- share of fully grounded answers (the hallucination KPI)
- completion rate
- tool error rate
- mean cost, latency and tool calls per answer

These cover the operational KPIs (hallucination rate, autonomous completion, cost per analysis). Time-to-value is
measured outside the system, as analyst time per screen before and after.

**Evaluation.** Three layers:
1. Unit tests for the deterministic layer.
2. Agent-loop tests with a scripted fake model.
3. An eval set (`tests/evals/`) of 17 analyst questions run against the real model: the brief's four questions, chart use,
   follow-ups, variations, and out-of-scope and error cases.

Eval answers are graded automatically, without an LLM judge:
- **Grounding:** every number in the answer must match a number in that conversation's tool outputs, within its
  displayed rounding. This gives a measurable hallucination rate.
- **Tools:** the expected tools were called, with the expected arguments.
- **Content:** required facts are mentioned, and the caveats section is present.

The runner reports pass rate, mean grounding, cost and latency per case, and is the gate for any prompt, model
or effort change. The first two layers run in CI on every push. Evals cost money, so they run on demand.

## 6. Key tradeoffs

| Decision | Chosen | Alternative | Why |
|---|---|---|---|
| Data freshness | Batch ingest to small CSVs, plus a live FAA feed | Call APIs on every question | BTS files are ~30 MB/month and the servers are fragile (TranStats was down during the build). Batch gives reproducible, fast, cheap answers. Cost: data is 2–5 months behind, and each answer states the vintage. |
| Normalisation | Percentiles vs a fixed national universe | Min-max within the compared set; z-scores | Stable meaning across questions and robust to outliers (ATL, ORD). Cost: loses magnitude, so 81 vs 80 isn't meaningful, which the sensitivity output makes explicit. |
| Weights | Transparent expert weights, overridable per query, with a sensitivity check | Learned weights | No labelled "profitable renovation" outcomes to learn from. Explicit weights are auditable and easy to debate with a client. |
| Agent framework | Hand-written tool loop on the Anthropic SDK (~60 lines) | LangChain/LangGraph, SDK tool runner | Few moving parts, easy to explain and debug, no framework lock-in. The tool runner would be a drop-in replacement. |
| LLM role | Orchestrate and explain only | LLM does the analysis (e.g. text-to-SQL) | Numbers must be reproducible and defensible to an investment committee. |
| Delay data coverage | BTS on-time (large carriers, domestic) | FAA ASPM (full coverage) | ASPM needs an account. Coverage is measured per airport and flagged when low. |
| Long-haul definition | Distance ≥ 3,000 mi, exposed as a parameter | Block time > 6h | Distance is available for every route; block time isn't for international routes. |
| Unmet demand | Transparent seat-gap floor + rule findings | Econometric demand model or spill curves | Achievable in a day and fully explainable. Stated as a lower bound. |
| Progress while waiting | Stream agent steps (server-sent events) | Static "Analyzing…" message; show the model's thinking summary | Answers take 15–60 s. Showing the real tool steps explains the method as it happens; a thinking summary adds a second, unverifiable "why". |
| Voice | Browser Web Speech API | Server speech-to-text / text-to-speech | Zero cost, no audio leaves the browser, no extra keys. Works in Chrome, Edge and Safari. |
| Sessions | In-memory per browser session | Database | Fine for a demo. Production would persist sessions and add auth. |

## 7. Assumptions, uncertainty, scope
- **US airports only.** Passengers means *enplaned* at the origin (T-100), all carriers, passenger and cargo
  flights. Last 12 months = the latest 12 T-100 months available.
- Different sources cover **different periods** (T-100 to 2026-04, on-time to 2026-07, international to 2025-12).
  The agent reports this in every answer.
- On-time metrics describe **large US carriers only**. Per-airport coverage is measured, and a flag appears when
  it's below 50%.
- International counts are directionless in the source and halved to estimate departures.
- Not modelled: construction cost, airport finances and debt capacity, airline lease agreements, land,
  environmental and regulatory approval, competing airports.
- Curated notes are dated, hand-verified context. They explain results but never change a score.

**Deliberately out of scope**
- **Precision.** Numbers are indicative. The goal is a sound, explainable method, not audited figures, so there
  is no data-validation pipeline or architecture review beyond the tests.
- **Market signals** (federal grants, metro growth, events and conferences). Events are noisy and reflect an
  investor's thesis; grants and metro data would be context, never part of the score.
- **Passenger journey and revenue data** (FAA CATS financials, TSA throughput, ground access). These are the
  best next data sources, not needed to answer the four questions.
- **Reviews.** Small, self-selected samples add noise and bias to a deterministic score.
- **Decline diagnostics and turnaround screening.** Same data and method in reverse, but outside the brief.

## 8. Next steps
1. Add T-100 route-level segment data (all carriers, seats per route) when TranStats is back, for exact long-haul
   and cargo splits.
2. Add the FAA Terminal Area Forecast (TAF) for forward-looking demand, and FAA-reported runway capacity
   ("called rates") for true demand-vs-capacity ratios.
3. Add DB1B fares: high fares relative to distance are a strong signal of constrained supply.
4. Backtest: did airports with high scores in 2015–2018 later announce or deliver expansions?
5. **Phase 2 metrics with client data:** capital-to-capacity yield and grant-adjusted payback (need CAPEX
   estimates), dwell time and queue impact (need terminal operations data).
6. Grow the eval set from real usage: add a 👍/👎 button in the UI and turn every 👎 into a new eval case. Add
   an LLM judge (with human spot checks) for explanation quality, which the automatic checks can't measure.
