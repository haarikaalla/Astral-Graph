"""Aggregate guardrail policy — the single gate every answer must pass.

Combines grounding, numeric auditing, cross-source consensus, citation
enforcement, prompt-injection warnings and PII/unsafe-content screening into one
verdict plus machine-readable feedback the agents can act on during a repair pass.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Sequence

from core.config import get_settings
from graph.consensus import ConsensusReport, analyse_consensus
from graph.grounding import GroundingReport, check_grounding
from graph.schema import Fact
from graph.store import KnowledgeGraph
from guardrails.citations import CitationReport, check_citations
from guardrails.numeric import NumericAudit, audit_numbers
from guardrails.schemas import Citation

# Hedging that signals the model is guessing rather than reporting evidence.
SPECULATION_MARKERS = (
    "i think", "i believe", "probably", "it is likely that", "presumably",
    "as far as i know", "if i recall", "i assume", "my guess",
)

SECRET_RE = re.compile(r"(sk-ant-[A-Za-z0-9_\-]{8,}|AKIA[0-9A-Z]{16}|api[_-]?key\s*[:=]\s*\S+)", re.I)


@dataclass
class GuardrailReport:
    passed: bool
    grounding: GroundingReport
    numeric: NumericAudit
    citations: CitationReport
    consensus: ConsensusReport = field(default_factory=ConsensusReport)
    injection_warnings: list[str] = field(default_factory=list)
    speculation: list[str] = field(default_factory=list)
    redactions: int = 0
    failures: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def score(self) -> float:
        return round(
            0.5 * self.grounding.score
            + 0.3 * self.numeric.verification_rate
            + 0.2 * (1.0 if self.citations.ok else 0.0),
            3,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "guardrail_score": self.score,
            "failures": self.failures,
            "warnings": self.warnings,
            "grounding": self.grounding.to_dict(),
            "numeric": self.numeric.to_dict(),
            "citations": self.citations.to_dict(),
            "consensus": self.consensus.to_dict(),
            "injection_warnings": self.injection_warnings,
            "speculation_markers": self.speculation,
            "secrets_redacted": self.redactions,
        }

    def feedback(self) -> str:
        parts = [
            self.grounding.feedback(),
            self.numeric.feedback(),
            self.citations.feedback(),
            self.consensus.feedback(),
        ]
        if self.speculation:
            parts.append(
                "Remove speculative hedging: " + ", ".join(sorted(set(self.speculation)))
            )
        return "\n\n".join(p for p in parts if p)


def redact_secrets(text: str) -> tuple[str, int]:
    redacted, count = SECRET_RE.subn("[REDACTED]", text)
    return redacted, count


def run_guardrails(
    text: str,
    kg: KnowledgeGraph,
    citations: Sequence[Citation] = (),
    *,
    extra_facts: Sequence[Fact] = (),
    injection_warnings: Sequence[str] = (),
    threshold: float | None = None,
    require_numeric: bool = True,
    known_sources: Sequence[str] = (),
    strict_consensus: bool = False,
) -> GuardrailReport:
    settings = get_settings()
    limit = settings.grounding_threshold if threshold is None else threshold

    grounding = check_grounding(text, kg, threshold=limit, extra_facts=list(extra_facts))
    numeric = audit_numbers(
        text, kg, extra_facts=extra_facts, min_rate=1.0 if require_numeric else 0.0
    )
    citation_report = check_citations(text, list(citations), kg, known_extra=known_sources)
    consensus = analyse_consensus(kg)

    lowered = text.lower()
    speculation = [m for m in SPECULATION_MARKERS if m in lowered]
    _, redactions = redact_secrets(text)

    failures: list[str] = []
    warnings: list[str] = []
    if not grounding.grounded:
        failures.append(f"grounding score {grounding.score:.2f} < {limit:.2f}")
    if not numeric.ok:
        failures.append(f"{len(numeric.unverified)} unverified number(s)")
    if not citation_report.ok:
        failures.append("citation requirement not met")
    if injection_warnings:
        failures.append("prompt-injection markers found in tool output")
    if speculation:
        failures.append("speculative language present")

    # Source disagreement is a warning by default: the honest response is to
    # report the conflict, not to withhold the answer. Set ``strict_consensus``
    # when a caller would rather refuse than disclose a discrepancy.
    if consensus.contradicted:
        message = (
            f"{len(consensus.contradicted)} claim(s) with disagreeing independent sources"
        )
        (failures if strict_consensus else warnings).append(message)
    if consensus.unit_mismatches:
        warnings.append(f"{len(consensus.unit_mismatches)} claim(s) with incomparable units")

    return GuardrailReport(
        passed=not failures,
        grounding=grounding,
        numeric=numeric,
        citations=citation_report,
        consensus=consensus,
        injection_warnings=list(injection_warnings),
        speculation=speculation,
        redactions=redactions,
        failures=failures,
        warnings=warnings,
    )
