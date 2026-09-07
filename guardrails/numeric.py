"""Deterministic numeric auditing.

Rule: **the LLM never owns a number.** Every quantity in an answer must trace back
to (a) a value returned by a data MCP server, or (b) a value computed by the
``astro_compute`` MCP server. This module re-checks that in pure Python, so the
verification itself cannot hallucinate.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Sequence

from graph.grounding import TRIVIAL_NUMBERS, extract_numbers
from graph.schema import Fact
from graph.store import KnowledgeGraph

# Ordinals/years/percentages that are safe rhetorical numbers.
YEAR_RE = re.compile(r"\b(1[5-9]\d{2}|20\d{2})\b")

# Decimal SI rescalings only. Wider factors (AU, lunar distance, minutes, hours)
# were tried and rejected: with a 5% tolerance they make almost any number match
# almost any fact, which silently defeats the audit. Unit changes that are not
# powers of ten must go through the ``astro_compute.convert_distance`` tool, which
# puts the converted value in the knowledge graph as a fact in its own right.
SCALES = (1.0, 1e-3, 1e3, 1e-6, 1e6, 1e-9, 1e9, 1e-2, 1e2)


@dataclass
class NumberVerdict:
    value: float
    verified: bool
    matched_fact: str = ""
    scale_used: float = 1.0


@dataclass
class NumericAudit:
    ok: bool
    total: int
    verified: int
    unverified: list[float] = field(default_factory=list)
    verdicts: list[NumberVerdict] = field(default_factory=list)

    @property
    def verification_rate(self) -> float:
        return self.verified / self.total if self.total else 1.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "numbers_checked": self.total,
            "numbers_verified": self.verified,
            "verification_rate": round(self.verification_rate, 3),
            "unverified_numbers": self.unverified,
            "detail": [
                {"value": v.value, "verified": v.verified, "fact": v.matched_fact[:120]}
                for v in self.verdicts
            ],
        }

    def feedback(self) -> str:
        if self.ok:
            return ""
        return (
            "The following numbers are not backed by any tool result and must be removed "
            "or replaced with a value obtained from an MCP tool: "
            + ", ".join(f"{n:g}" for n in self.unverified[:10])
        )


def _candidate_values(kg: KnowledgeGraph, extra: Sequence[Fact] = ()) -> list[tuple[float, str]]:
    values = [(value, fact.to_sentence()) for fact, value in kg.numeric_facts()]
    for fact in extra:
        value = fact.numeric_value
        if value is not None:
            values.append((value, fact.to_sentence()))
    return values


def audit_numbers(
    text: str,
    kg: KnowledgeGraph,
    *,
    tolerance: float = 0.05,
    extra_facts: Sequence[Fact] = (),
    ignore_years: bool = True,
    min_rate: float = 1.0,
) -> NumericAudit:
    """Verify every non-trivial number in ``text`` against tool-provided values."""
    years = {float(y) for y in YEAR_RE.findall(text)} if ignore_years else set()
    numbers = [
        n for n in extract_numbers(text)
        if n not in TRIVIAL_NUMBERS and n not in years
    ]
    candidates = _candidate_values(kg, extra_facts)

    verdicts: list[NumberVerdict] = []
    for number in numbers:
        verdict = NumberVerdict(value=number, verified=False)
        for candidate, sentence in candidates:
            for scale in SCALES:
                scaled = candidate * scale
                if scaled == 0:
                    continue
                if abs(number - scaled) / abs(scaled) <= tolerance:
                    verdict = NumberVerdict(number, True, sentence, scale)
                    break
                # allow stated roundings such as "about 1.3 million"
                if abs(scaled) > 1 and abs(round(number, 2) - round(scaled, 2)) <= abs(scaled) * tolerance:
                    verdict = NumberVerdict(number, True, sentence, scale)
                    break
            if verdict.verified:
                break
        verdicts.append(verdict)

    verified = sum(1 for v in verdicts if v.verified)
    unverified = [v.value for v in verdicts if not v.verified]
    rate = verified / len(verdicts) if verdicts else 1.0
    return NumericAudit(
        ok=rate >= min_rate,
        total=len(verdicts),
        verified=verified,
        unverified=unverified,
        verdicts=verdicts,
    )
