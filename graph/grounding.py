"""Grounding checker — decides whether generated text is supported by the graph.

Every sentence of an agent answer is scored against the :class:`~graph.schema.Fact`
set collected during the session:

* **numeric check** — every number in the sentence must match a fact value within
  tolerance (unit-scale aware). Unmatched numbers are the strongest hallucination
  signal, so they dominate the score.
* **lexical check** — content-word overlap with fact sentences.

The Critic agent turns a low score into a rejection or a forced revision.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Any

from graph.schema import Fact
from graph.store import KnowledgeGraph

NUMBER_RE = re.compile(r"(?<![\w.])[-+]?\d{1,3}(?:,\d{3})+(?:\.\d+)?|(?<![\w.])[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")
SENTENCE_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9])")

STOPWORDS = {
    "the", "a", "an", "is", "are", "was", "were", "of", "to", "in", "on", "at", "and",
    "or", "for", "with", "by", "from", "that", "this", "it", "its", "as", "be", "has",
    "have", "will", "can", "about", "than", "which", "there", "their", "these", "those",
    "km", "au", "per", "second", "seconds", "day", "days", "year", "years", "approximately",
    "roughly", "about", "currently", "also", "more", "most", "some", "one", "two",
}

# Numbers that appear so often they cannot discriminate.
TRIVIAL_NUMBERS = {0.0, 1.0, 2.0, 3.0, 100.0, 1000.0, 2024.0, 2025.0, 2026.0}

SCALES = (1.0, 1e-3, 1e3, 1e-6, 1e6, 1e-9, 1e9, 1 / 1.496e8, 1.496e8, 1 / 384400.0, 384400.0)


@dataclass
class SentenceVerdict:
    sentence: str
    supported: bool
    score: float
    matched_numbers: list[float] = field(default_factory=list)
    unmatched_numbers: list[float] = field(default_factory=list)
    supporting_facts: list[str] = field(default_factory=list)
    reason: str = ""


@dataclass
class GroundingReport:
    score: float
    grounded: bool
    verdicts: list[SentenceVerdict]
    unsupported_sentences: list[str]
    unmatched_numbers: list[float]
    fact_count: int
    threshold: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "grounding_score": round(self.score, 3),
            "grounded": self.grounded,
            "threshold": self.threshold,
            "fact_count": self.fact_count,
            "unsupported_sentences": self.unsupported_sentences,
            "unmatched_numbers": self.unmatched_numbers,
            "sentences": [
                {
                    "sentence": v.sentence,
                    "supported": v.supported,
                    "score": round(v.score, 3),
                    "unmatched_numbers": v.unmatched_numbers,
                    "reason": v.reason,
                }
                for v in self.verdicts
            ],
        }

    def feedback(self) -> str:
        if self.grounded:
            return ""
        parts = []
        if self.unmatched_numbers:
            parts.append(
                "These numbers are not present in the retrieved evidence: "
                + ", ".join(_fmt(n) for n in self.unmatched_numbers[:10])
            )
        if self.unsupported_sentences:
            parts.append(
                "These statements are unsupported and must be removed or rewritten:\n- "
                + "\n- ".join(s[:220] for s in self.unsupported_sentences[:6])
            )
        return "\n".join(parts)


def _fmt(value: float) -> str:
    return f"{value:g}"


def extract_numbers(text: str) -> list[float]:
    values: list[float] = []
    for match in NUMBER_RE.findall(text):
        try:
            values.append(float(str(match).replace(",", "")))
        except ValueError:
            continue
    return values


def _tokens(text: str) -> set[str]:
    words = re.findall(r"[a-zA-Z][a-zA-Z\-']{2,}", text.lower())
    return {w for w in words if w not in STOPWORDS}


def _number_matches(value: float, candidates: list[float], tolerance: float) -> bool:
    if value in TRIVIAL_NUMBERS:
        return True
    for candidate in candidates:
        for scale in SCALES:
            scaled = candidate * scale
            if scaled == 0:
                if abs(value) < 1e-12:
                    return True
                continue
            if abs(value - scaled) / abs(scaled) <= tolerance:
                return True
            # tolerate rounding of large numbers, e.g. 1.28 million vs 1,284,000
            if abs(value) >= 1 and abs(round(value) - round(scaled)) <= max(1.0, abs(scaled) * tolerance):
                return True
    return False


def check_grounding(
    text: str,
    kg: KnowledgeGraph,
    *,
    threshold: float = 0.6,
    numeric_tolerance: float = 0.05,
    extra_facts: list[Fact] | None = None,
) -> GroundingReport:
    """Score ``text`` against the facts stored in ``kg``."""
    facts = kg.facts() + list(extra_facts or [])
    fact_numbers = [v for _, v in kg.numeric_facts()]
    for fact in extra_facts or []:
        value = fact.numeric_value
        if value is not None:
            fact_numbers.append(value)
    fact_sentences = [(f, _tokens(f.to_sentence()), f.id) for f in facts]

    sentences = [s.strip() for s in SENTENCE_RE.split(text.strip()) if len(s.strip()) > 12]
    if not sentences:
        sentences = [text.strip()] if text.strip() else []

    verdicts: list[SentenceVerdict] = []
    for sentence in sentences:
        numbers = [n for n in extract_numbers(sentence) if n not in TRIVIAL_NUMBERS]
        matched, unmatched = [], []
        for number in numbers:
            (matched if _number_matches(number, fact_numbers, numeric_tolerance) else unmatched).append(number)

        sentence_tokens = _tokens(sentence)
        best_overlap, supporting = 0.0, []
        for fact, tokens, fact_id in fact_sentences:
            if not tokens:
                continue
            overlap = len(sentence_tokens & tokens) / max(3, len(sentence_tokens))
            if overlap > best_overlap:
                best_overlap = overlap
            if overlap >= 0.25:
                supporting.append(fact_id)

        if numbers:
            numeric_score = len(matched) / len(numbers)
            score = 0.75 * numeric_score + 0.25 * min(1.0, best_overlap * 2.5)
        else:
            score = min(1.0, best_overlap * 2.5)
            # Purely qualitative sentences (framing, caveats) are not penalised hard.
            score = max(score, 0.55)

        supported = score >= threshold and not unmatched
        verdicts.append(
            SentenceVerdict(
                sentence=sentence,
                supported=supported,
                score=score,
                matched_numbers=matched,
                unmatched_numbers=unmatched,
                supporting_facts=supporting[:5],
                reason=(
                    "numbers not found in evidence" if unmatched
                    else ("weak lexical support" if score < threshold else "ok")
                ),
            )
        )

    if not verdicts:
        return GroundingReport(0.0, False, [], [], [], len(facts), threshold)

    if not facts:
        overall = 0.0
    else:
        overall = sum(v.score for v in verdicts) / len(verdicts)
        # A single fabricated number should sink the whole answer.
        penalty = sum(len(v.unmatched_numbers) for v in verdicts)
        overall *= math.pow(0.75, min(penalty, 6))

    return GroundingReport(
        score=overall,
        grounded=overall >= threshold and not any(v.unmatched_numbers for v in verdicts),
        verdicts=verdicts,
        unsupported_sentences=[v.sentence for v in verdicts if not v.supported],
        unmatched_numbers=[n for v in verdicts for n in v.unmatched_numbers],
        fact_count=len(facts),
        threshold=threshold,
    )
