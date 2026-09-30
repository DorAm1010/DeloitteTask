"""Automatic grading for agent evals - no LLM judge, fully deterministic.

Checks per case:
  * grounding       every number in the answer appears (within rounding) in that
                    case's tool outputs -> measures the hallucination rate
  * expected tools  the right tools were called, optionally with specific arguments
  * forbidden tools tools that must NOT be called (e.g. for out-of-scope questions)
  * must mention    regexes the answer must match (key facts, airport codes)
  * must not mention regexes the answer must NOT match (e.g. a buy/sell call)
  * caveats         the answer has an assumptions/caveats section
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from airport_agent.grounding import extract_answer_numbers, grounding, tool_numbers  # noqa: F401


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
    for pattern in case.get("must_not_mention", []):
        g.checks[f"does not mention /{pattern}/"] = not re.search(pattern, answer, re.IGNORECASE)
    if case.get("require_caveats", True):
        g.checks["has caveats section"] = bool(re.search(r"assumption|caveat", answer, re.IGNORECASE))

    g.grounding, g.ungrounded_numbers = grounding(answer, [c["output"] for c in calls])
    min_grounding = case.get("min_grounding", 0.9)
    if g.grounding is not None and min_grounding is not None:
        g.checks[f"grounding >= {min_grounding:.0%}"] = g.grounding >= min_grounding

    g.passed = all(g.checks.values())
    return g
