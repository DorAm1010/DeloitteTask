"""Observability: what did the agent do, how long did it take, what did it cost?

Every tool call and every answered question is written as one JSON line to
logs/agent.log (git-ignored). That file is what you would ship to a log
platform in production, and what the eval runner uses for cost and latency.
"""
from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass

from . import config

log = logging.getLogger("airport_agent")


@dataclass
class Usage:
    """Token usage summed over every model call that produced one answer."""
    input_tokens: int = 0
    output_tokens: int = 0
    cache_creation_input_tokens: int = 0
    cache_read_input_tokens: int = 0
    model_calls: int = 0

    def add(self, response) -> None:
        usage = getattr(response, "usage", None)
        self.model_calls += 1
        if usage is None:  # e.g. the fake client in tests
            return
        for name in ("input_tokens", "output_tokens", "cache_creation_input_tokens", "cache_read_input_tokens"):
            setattr(self, name, getattr(self, name) + (getattr(usage, name, 0) or 0))

    def cost_usd(self, model: str) -> float | None:
        """Estimated cost; None when the model's price isn't in config.MODEL_PRICING."""
        price = config.MODEL_PRICING.get(model)
        if price is None:
            return None
        per_in, per_out = price[0] / 1e6, price[1] / 1e6
        return round(self.input_tokens * per_in + self.output_tokens * per_out
                     + self.cache_creation_input_tokens * per_in * 1.25
                     + self.cache_read_input_tokens * per_in * 0.1, 5)

    def to_dict(self) -> dict:
        return asdict(self)


def log_event(kind: str, **fields) -> None:
    """Append one JSON line to logs/agent.log. Never lets logging break the app."""
    record = {"ts": time.strftime("%Y-%m-%dT%H:%M:%S"), "event": kind, **fields}
    log.debug("%s", record)
    try:
        config.LOG_DIR.mkdir(parents=True, exist_ok=True)
        with open(config.LOG_DIR / "agent.log", "a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, default=str) + "\n")
    except OSError as exc:
        log.warning("could not write telemetry: %s", exc)


def summarize(path=None) -> dict:
    """Agent KPIs from logs/agent.log: the numbers you'd put on a monitoring dashboard."""
    path = path or config.LOG_DIR / "agent.log"
    events = [json.loads(line) for line in open(path, encoding="utf-8")] if path.exists() else []
    answers = [e for e in events if e["event"] == "answer"]
    tools = [e for e in events if e["event"] == "tool_call"]
    if not answers:
        return {"answers": 0}
    graded = [a for a in answers if a.get("grounding") is not None]
    costs = [a["cost_usd"] for a in answers if a.get("cost_usd") is not None]
    return {
        "answers": len(answers),
        # Share of answers where every number traced to tool data (the hallucination KPI).
        "fully_grounded_answers": round(sum(a["grounding"] == 1 for a in graded) / len(graded), 3) if graded else None,
        "mean_grounding": round(sum(a["grounding"] for a in graded) / len(graded), 3) if graded else None,
        # Finished normally (not refused, truncated or stopped by the step limit).
        "completion_rate": round(sum(a["stop_reason"] == "end_turn" for a in answers) / len(answers), 3),
        "tool_error_rate": round(sum(t["is_error"] for t in tools) / len(tools), 3) if tools else None,
        "mean_cost_usd": round(sum(costs) / len(costs), 4) if costs else None,
        "mean_latency_s": round(sum(a["latency_s"] for a in answers) / len(answers), 2),
        "mean_tool_calls": round(sum(len(a["tools"]) for a in answers) / len(answers), 2),
    }


if __name__ == "__main__":  # python -m airport_agent.telemetry
    print(json.dumps(summarize(), indent=1))
