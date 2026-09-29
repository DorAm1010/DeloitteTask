"""Run the agent eval set against the real model (needs ANTHROPIC_API_KEY; costs a few dollars).

    python -m tests.evals.run                      # all cases
    python -m tests.evals.run --only brief_sfo_unmet_demand
    python -m tests.evals.run --fail-under 0.8     # non-zero exit for CI

Prints a pass/fail table and writes a JSON report to logs/. Use it after every
change to the system prompt, tool descriptions, model or effort level.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

import yaml

from airport_agent import config
from airport_agent.agent import AirportAgent
from tests.evals.grading import grade

CASES = Path(__file__).with_name("cases.yaml")


def run_case(case: dict) -> dict:
    agent = AirportAgent()
    calls, cost, latency, reply = [], 0.0, 0.0, None
    for turn in case["turns"]:
        reply = agent.ask(turn)
        calls += [{"name": c.name, "input": c.input, "output": c.output} for c in reply.tool_calls]
        cost += reply.cost_usd or 0.0
        latency += reply.latency_s
    g = grade(case, reply.text, calls)
    return {**asdict(g), "cost_usd": round(cost, 4), "latency_s": round(latency, 1),
            "tools_called": [c["name"] for c in calls], "answer": reply.text}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--only", nargs="*", help="case ids to run")
    parser.add_argument("--fail-under", type=float, default=None, help="exit 1 if pass rate is below this")
    args = parser.parse_args()

    cases = yaml.safe_load(CASES.read_text())
    if args.only:
        cases = [c for c in cases if c["id"] in args.only]
    print(f"Running {len(cases)} eval case(s) on {config.MODEL} (effort={config.EFFORT})\n")

    results = []
    for case in cases:
        r = run_case(case)
        results.append(r)
        mark = "PASS" if r["passed"] else "FAIL"
        gr = "-" if r["grounding"] is None else f"{r['grounding']:.0%}"
        print(f"{mark}  {r['case_id']:<28} grounding {gr:>5}  {r['latency_s']:>5}s  ${r['cost_usd']:.3f}")
        for name, ok in r["checks"].items():
            if not ok:
                print(f"      x {name}")
        if r["ungrounded_numbers"]:
            print(f"      ungrounded numbers: {r['ungrounded_numbers']}")

    grounded = [r["grounding"] for r in results if r["grounding"] is not None]
    summary = {
        "model": config.MODEL, "effort": config.EFFORT,
        "pass_rate": round(sum(r["passed"] for r in results) / len(results), 3),
        "mean_grounding": round(sum(grounded) / len(grounded), 3) if grounded else None,
        "total_cost_usd": round(sum(r["cost_usd"] for r in results), 3),
        "mean_latency_s": round(sum(r["latency_s"] for r in results) / len(results), 1),
    }
    print("\n" + json.dumps(summary, indent=1))

    config.LOG_DIR.mkdir(parents=True, exist_ok=True)
    out = config.LOG_DIR / f"eval_report_{time.strftime('%Y%m%d_%H%M%S')}.json"
    out.write_text(json.dumps({"summary": summary, "results": results}, indent=1))
    print(f"Report: {out}")
    if args.fail_under is not None and summary["pass_rate"] < args.fail_under:
        sys.exit(1)


if __name__ == "__main__":
    main()
