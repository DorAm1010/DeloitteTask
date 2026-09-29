"""Automatic grading for agent evals - no LLM judge, fully deterministic.

Checks per case:
  * grounding       every number in the answer appears (within rounding) in that
                    case's tool outputs -> measures the hallucination rate
  * expected tools  the right tools were called, optionally with specific arguments
  * forbidden tools tools that must NOT be called (e.g. for out-of-scope questions)
  * must mention    regexes the answer must match (key facts, airport codes)
  * caveats         the answer has an assumptions/caveats section
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

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
        for v in obj.values():
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


def _arg_matches(actual, wanted: str) -> bool:
    wanted = wanted.strip().lower()
    if isinstance(actual, list):
        return wanted in [str(x).lower() for x in actual]
    return str(actual).lower() == wanted


def _tool_matches(spec: str, calls: list[dict]) -> bool:
    """spec: 'name', 'a|b' (any of), or 'name(key=value, ...)'."""
    for option in spec.split("|"):
        m = re.fullmatch(r"\s*(\w+)\s*(?:\((.*)\))?\s*", option)
        name, args = m.group(1), m.group(2)
        wanted = [kv.split("=", 1) for kv in args.split(",")] if args else []
        for call in calls:
            if call["name"] == name and all(
                    k.strip() in call["input"] and _arg_matches(call["input"][k.strip()], v) for k, v in wanted):
                return True
    return False


@dataclass
class Grade:
    case_id: str
    passed: bool
    checks: dict[str, bool] = field(default_factory=dict)
    grounding: float | None = None
    ungrounded_numbers: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


def grade(case: dict, answer: str, calls: list[dict]) -> Grade:
    g = Grade(case_id=case["id"], passed=True)
    names = [c["name"] for c in calls]

    for spec in case.get("expect_tools", []):
        ok = _tool_matches(spec, calls)
        g.checks[f"calls {spec}"] = ok
        if not ok:
            g.notes.append(f"expected tool {spec}, called {names}")
    for name in case.get("forbid_tools", []):
        ok = name not in names
        g.checks[f"does not call {name}"] = ok
    for pattern in case.get("must_mention", []):
        g.checks[f"mentions /{pattern}/"] = bool(re.search(pattern, answer, re.IGNORECASE))
    if case.get("require_caveats", True):
        g.checks["has caveats section"] = bool(re.search(r"assumption|caveat", answer, re.IGNORECASE))

    g.grounding, g.ungrounded_numbers = grounding(answer, [c["output"] for c in calls])
    min_grounding = case.get("min_grounding", 0.9)
    if g.grounding is not None and min_grounding is not None:
        g.checks[f"grounding >= {min_grounding:.0%}"] = g.grounding >= min_grounding

    g.passed = all(g.checks.values())
    return g
