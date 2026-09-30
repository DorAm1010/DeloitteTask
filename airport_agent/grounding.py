"""Grounding check: is every number in an answer traceable to tool outputs?

Deterministic and free (no model call). Used in two places:
  * live, on every answer (agent.py) - logged to telemetry and flagged in the UI
  * in the eval set (tests/evals/grading.py) - the hallucination-rate check

A number counts as grounded if it matches a tool-output number within its
displayed rounding (82.6% matches 0.8255; 1.0M matches 1,029,860). Years and
small integers (<= 10) are skipped as not data. Numbers the model derived
itself (e.g. doubling a figure) show up as ungrounded - intentionally, since
the system prompt says numbers must come from tools.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass

# A number with optional thousands separators, decimals, and a unit suffix.
NUMBER_RE = re.compile(
    r"(?<![\w.])(\d{1,3}(?:,\d{3})+|\d+)(\.\d+)?\s*(%|k\b|K\b|M\b|million\b|bn\b|B\b|billion\b)?")
SUFFIX = {"k": 1e3, "K": 1e3, "M": 1e6, "million": 1e6, "bn": 1e9, "B": 1e9, "billion": 1e9}


@dataclass
class AnswerNumber:
    text: str
    value: float
    tolerance: float


def extract_answer_numbers(text: str) -> list[AnswerNumber]:
    """Numbers stated in the answer, with a tolerance matching their displayed precision.

    Skipped as not data: years (1990-2035) and small integers (<= 10, e.g. "top 3").
    """
    out = []
    for m in NUMBER_RE.finditer(text):
        whole, frac, suffix = m.group(1), m.group(2) or "", m.group(3) or ""
        value = float(whole.replace(",", "") + frac)
        if not frac and not suffix and (value <= 10 or 1990 <= value <= 2035):
            continue
        scale = SUFFIX.get(suffix, 1.0)
        decimals = len(frac) - 1 if frac else 0
        tolerance = 0.5 * 10 ** (-decimals) * scale
        out.append(AnswerNumber(m.group(0).strip(), value * scale, tolerance))
    return out


def _walk(obj, found: set[float]) -> None:
    if isinstance(obj, bool):
        return
    if isinstance(obj, (int, float)):
        found.add(float(obj))
    elif isinstance(obj, dict):
        for k, v in obj.items():  # keys can carry numbers too, e.g. "medium (500-1,499 mi)"
            _walk(k, found)
            _walk(v, found)
    elif isinstance(obj, list):
        for v in obj:
            _walk(v, found)
    elif isinstance(obj, str):
        for n in extract_answer_numbers(obj):
            found.add(n.value)


def tool_numbers(outputs: list[str]) -> list[float]:
    """Every number in the tool outputs, plus the forms an answer may display it in."""
    raw: set[float] = set()
    for out in outputs:
        try:
            _walk(json.loads(out), raw)
        except ValueError:
            _walk(out, raw)
    candidates = set(raw)
    for v in raw:
        if abs(v) <= 1.5:
            candidates.add(v * 100)          # fractions shown as percentages
        candidates.add(abs(v))               # "+3%" vs -0.03 style sign handling
        if v <= 0:
            candidates.add(-v * 100)
    return sorted(candidates)


def grounding(answer: str, outputs: list[str]) -> tuple[float | None, list[str]]:
    """Share of answer numbers found in tool outputs (None if the answer has no numbers)."""
    numbers = extract_answer_numbers(answer)
    if not numbers:
        return None, []
    pool = tool_numbers(outputs)
    missing = []
    for n in numbers:
        slack = n.tolerance + 0.005 * abs(n.value) + 1e-9
        if not any(abs(n.value - t) <= slack for t in pool):
            missing.append(n.text)
    return 1 - len(missing) / len(numbers), missing
