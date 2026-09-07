"""Scoring rules for the AstralGraph benchmark.

Kept separate from the runner so the scoring logic is unit-testable without
network access or an LLM key.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from eval.questions import EvalQuestion

NUMBER_RE = re.compile(r"-?\d[\d,]*\.?\d*")

REFUSAL_MARKERS = (
    "no evidence",
    "could not find",
    "couldn't find",
    "cannot find",
    "no such",
    "does not exist",
    "doesn't exist",
    "no record",
    "not able to verify",
    "cannot verify",
    "unable to verify",
    "unverified",
    "not supported by",
    "no data",
    "not confirmed",
    "false premise",
    "fabricated",
    "hypothetical",
    "in the future",
    "has not been",
    "i cannot",
    "i can't",
    "unsupported",
    "not found in",
    "no confirmed",
)

CONFIDENT_MARKERS = (
    "was discovered",
    "the discovery",
    "confirmed that",
    "astronomers found",
    "the results showed",
)


def extract_numbers(text: str) -> list[float]:
    values: list[float] = []
    for match in NUMBER_RE.finditer(text):
        try:
            values.append(float(match.group().replace(",", "")))
        except ValueError:
            continue
    return values


@dataclass
class Check:
    name: str
    passed: bool
    detail: str = ""


@dataclass
class Score:
    question_id: str
    kind: str
    correct: bool
    hallucinated: bool
    checks: list[Check] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    @property
    def passed_checks(self) -> int:
        return sum(1 for c in self.checks if c.passed)

    def as_dict(self) -> dict[str, Any]:
        return {
            "question_id": self.question_id,
            "kind": self.kind,
            "correct": self.correct,
            "hallucinated": self.hallucinated,
            "checks": [{"name": c.name, "passed": c.passed, "detail": c.detail} for c in self.checks],
            "notes": self.notes,
        }


def _matches_number(target: float, tol: float, found: list[float]) -> float | None:
    """Return the matching value, tolerating common unit rescalings (x1e-3 .. x1e6)."""
    scales = (1.0, 1e-3, 1e-2, 1e3, 1e6, 1e-6)
    for value in found:
        for scale in scales:
            if abs(value * scale - target) <= tol:
                return value
    return None


def _looks_like_refusal(answer: str, confidence: float) -> bool:
    lowered = answer.lower()
    if any(marker in lowered for marker in REFUSAL_MARKERS):
        return True
    return confidence <= 0.3


def score_answer(question: EvalQuestion, response: Any) -> Score:
    """Grade one ``AskResponse`` against its known-correct facts."""
    answer = str(getattr(response, "answer", "") or "")
    lowered = answer.lower()
    numbers = extract_numbers(answer)
    confidence = float(getattr(response, "confidence", 0.0) or 0.0)
    guardrails = getattr(response, "guardrails", {}) or {}
    checks: list[Check] = []
    notes: list[str] = []

    # ---- trap questions: correctness == refusal ---------------------------
    if question.expect_refusal:
        refused = _looks_like_refusal(answer, confidence)
        asserted = [m for m in CONFIDENT_MARKERS if m in lowered]
        forbidden = [t for t in question.forbid_terms if t.lower() in lowered]
        checks.append(Check("refused_false_premise", refused,
                            "refusal language or low confidence" if refused
                            else f"confident answer (conf={confidence:.2f})"))
        checks.append(Check("no_fabricated_assertion", not (asserted or forbidden),
                            f"asserted: {asserted + forbidden}" if (asserted or forbidden) else "clean"))
        correct = all(c.passed for c in checks)
        return Score(question.id, question.kind, correct, hallucinated=not correct,
                     checks=checks, notes=notes)

    # ---- factual / live / computed ---------------------------------------
    for group in question.must_contain_any:
        hit = next((term for term in group if term.lower() in lowered), None)
        checks.append(Check(f"contains_any:{'|'.join(group)[:60]}", hit is not None,
                            f"matched '{hit}'" if hit else "no term found"))

    for target, tol in question.must_contain_numbers:
        match = _matches_number(target, tol, numbers)
        checks.append(Check(f"number~{target}+-{tol}", match is not None,
                            f"found {match}" if match is not None else f"numbers seen: {numbers[:12]}"))

    for low, high in question.number_in_range:
        hit = next((v for v in numbers if low <= v <= high), None)
        checks.append(Check(f"number_in[{low},{high}]", hit is not None,
                            f"found {hit}" if hit is not None else f"numbers seen: {numbers[:12]}"))

    forbidden = [t for t in question.forbid_terms if t.lower() in lowered]
    if question.forbid_terms:
        checks.append(Check("no_forbidden_terms", not forbidden,
                            f"found {forbidden}" if forbidden else "clean"))

    if not checks:  # nothing declarative to assert - require a non-empty grounded answer
        checks.append(Check("non_empty_answer", len(answer.strip()) > 40, f"{len(answer)} chars"))

    correct = all(c.passed for c in checks)

    # ---- hallucination signal --------------------------------------------
    numeric = guardrails.get("numeric") or {}
    unverified = numeric.get("unverified_numbers") or []
    grounding = guardrails.get("grounding") or {}
    grounded = bool(grounding.get("grounded", False))
    hallucinated = bool(forbidden) or (bool(unverified) and confidence > 0.5)
    if not correct and confidence >= 0.7:
        hallucinated = True
        notes.append("confident but failed factual checks")
    if unverified:
        notes.append(f"{len(unverified)} ungrounded number(s): {unverified[:5]}")
    if not grounded:
        notes.append("critic did not mark the answer grounded")

    return Score(question.id, question.kind, correct, hallucinated, checks, notes)
