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
