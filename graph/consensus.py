"""Cross-source consensus and contradiction detection.

Grounding answers a narrow question: *does the graph contain this claim?* It says
nothing about the case where the graph contains the claim **twice, differently**.

This module closes that gap. Facts are clustered by ``(subject, predicate)``, and
each cluster containing two or more *independent* sources is adjudicated:

``corroborated``   independent sources agree within tolerance -> confidence boost
``contradicted``   independent sources disagree beyond tolerance -> surfaced as a caveat
``unit_mismatch``  same quantity reported in incompatible units -> not comparable
``single_source``  only one source spoke; nothing to cross-check

Corroboration writes ``SUPPORTS`` edges between the agreeing facts; disagreement
writes ``CONTRADICTS`` edges. Both directions are stored, so the graph itself
records that an adjudication happened rather than silently picking a winner.

Independence is judged by ``source_name`` (falling back to ``mcp_server``): the
same upstream answering twice is repetition, not corroboration.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Literal

from graph.schema import Edge, Fact, RelationType
from graph.store import KnowledgeGraph

#: Relative difference below which two numeric readings are "the same measurement".
DEFAULT_TOLERANCE = 0.05

#: Units that mean the same thing, so a cluster mixing them is still comparable.
_UNIT_ALIASES: dict[str, str] = {
    "": "",
    "km": "km",
    "kilometre": "km",
    "kilometres": "km",
    "kilometer": "km",
    "kilometers": "km",
    "ly": "ly",
    "light-year": "ly",
    "light-years": "ly",
    "lightyear": "ly",
    "lightyears": "ly",
    "pc": "pc",
    "parsec": "pc",
    "parsecs": "pc",
    "au": "au",
    "k": "K",
    "kelvin": "K",
    "deg": "deg",
    "degree": "deg",
    "degrees": "deg",
    "mt": "Mt",
    "megaton": "Mt",
    "megatons": "Mt",
    "count": "count",
}

ConsensusStatus = Literal["corroborated", "contradicted", "unit_mismatch", "single_source"]

_SEPARATORS = re.compile(r"[\s_\-]+")


def _normalise_key(text: str) -> str:
    return _SEPARATORS.sub(" ", str(text or "").strip().lower())


def _canonical_unit(unit: str | None) -> str:
    raw = str(unit or "").strip().lower()
    return _UNIT_ALIASES.get(raw, raw)


def _source_of(fact: Fact) -> str:
    return (fact.source_name or fact.mcp_server or "unknown").strip().lower()


def _relative_spread(values: list[float]) -> float:
    """Max pairwise relative difference, scaled by the larger magnitude."""
    if len(values) < 2:
        return 0.0
    low, high = min(values), max(values)
    scale = max(abs(low), abs(high))
    if scale == 0.0:
        return 0.0 if low == high else 1.0
    return abs(high - low) / scale


@dataclass
class FactCluster:
    """All readings of one ``(subject, predicate)`` pair, and how they compare."""

    subject: str
    predicate: str
    facts: list[Fact]
    status: ConsensusStatus
    sources: list[str] = field(default_factory=list)
    values: list[Any] = field(default_factory=list)
    units: list[str] = field(default_factory=list)
    spread: float = 0.0

    @property
    def independent_source_count(self) -> int:
        return len({_source_of(f) for f in self.facts})

    def describe(self) -> str:
        readings = ", ".join(
            f"{f.value}{(' ' + f.unit) if f.unit else ''} [{f.source_name or f.mcp_server}]"
            for f in self.facts
        )
        return f"{self.subject} — {self.predicate.replace('_', ' ')}: {readings}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "subject": self.subject,
            "predicate": self.predicate,
            "status": self.status,
            "sources": self.sources,
            "values": self.values,
            "units": self.units,
            "spread": round(self.spread, 4),
            "description": self.describe(),
        }


@dataclass
class ConsensusReport:
    """Adjudication across every multi-source cluster in the graph."""

    clusters: list[FactCluster] = field(default_factory=list)
    tolerance: float = DEFAULT_TOLERANCE

    @property
    def cross_checked(self) -> list[FactCluster]:
        """Clusters where more than one independent source spoke."""
        return [c for c in self.clusters if c.status != "single_source"]

    @property
    def corroborated(self) -> list[FactCluster]:
        return [c for c in self.clusters if c.status == "corroborated"]

    @property
    def contradicted(self) -> list[FactCluster]:
        return [c for c in self.clusters if c.status == "contradicted"]

    @property
    def unit_mismatches(self) -> list[FactCluster]:
        return [c for c in self.clusters if c.status == "unit_mismatch"]

    @property
    def agreement_rate(self) -> float:
        """Fraction of cross-checked clusters whose sources agreed."""
        checked = self.cross_checked
        if not checked:
            return 1.0
        return round(len(self.corroborated) / len(checked), 3)

    @property
    def ok(self) -> bool:
        return not self.contradicted

    def caveats(self, limit: int = 3) -> list[str]:
        """Human-readable warnings suitable for attaching to the final answer."""
        out = []
        for cluster in self.contradicted[:limit]:
            out.append(f"Sources disagree on {cluster.describe()}")
        for cluster in self.unit_mismatches[:limit]:
            out.append(f"Sources report incompatible units for {cluster.describe()}")
        return out

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "tolerance": self.tolerance,
            "clusters_total": len(self.clusters),
            "cross_checked": len(self.cross_checked),
            "corroborated": len(self.corroborated),
            "contradicted": len(self.contradicted),
            "unit_mismatches": len(self.unit_mismatches),
            "single_source": len(self.clusters) - len(self.cross_checked),
            "agreement_rate": self.agreement_rate,
            "caveats": self.caveats(),
            "conflicts": [c.to_dict() for c in self.contradicted[:10]],
            "corroborations": [c.to_dict() for c in self.corroborated[:10]],
        }

    def feedback(self) -> str:
        if not self.contradicted and not self.unit_mismatches:
            return ""
        lines = ["Independent sources do not agree. State the disagreement explicitly "
                 "rather than choosing one value silently:"]
        lines.extend(f"- {c.describe()}" for c in self.contradicted[:5])
        lines.extend(f"- (unit mismatch) {c.describe()}" for c in self.unit_mismatches[:3])
        return "\n".join(lines)


def _classify(facts: list[Fact], tolerance: float) -> tuple[ConsensusStatus, float]:
    """Decide whether a cluster's readings agree, conflict, or aren't comparable."""
    if len({_source_of(f) for f in facts}) < 2:
        return "single_source", 0.0

    units = {_canonical_unit(f.unit) for f in facts}
    numerics = [f.numeric_value for f in facts]

    if all(n is not None for n in numerics):
        # Mixed units on a numeric quantity are not comparable without conversion.
        if len(units - {""}) > 1:
            return "unit_mismatch", 0.0
        spread = _relative_spread([n for n in numerics if n is not None])
        return ("corroborated" if spread <= tolerance else "contradicted"), spread

    # Non-numeric: compare normalised text.
    rendered = {_normalise_key(f.value) for f in facts}
    if len(rendered) == 1:
        return "corroborated", 0.0
    # Containment counts as agreement ("Apophis" vs "99942 Apophis").
    ordered = sorted(rendered, key=len)
    if all(ordered[0] and ordered[0] in other for other in ordered[1:]):
        return "corroborated", 0.0
    return "contradicted", 1.0


def analyse_consensus(
    kg: KnowledgeGraph,
    *,
    tolerance: float = DEFAULT_TOLERANCE,
    link: bool = True,
) -> ConsensusReport:
    """Cluster the graph's facts by subject+predicate and adjudicate each cluster.

    When ``link`` is true the verdicts are written back as ``SUPPORTS`` /
    ``CONTRADICTS`` edges so the adjudication is itself part of the record.
    """
    buckets: dict[tuple[str, str], list[Fact]] = {}
    for fact in kg.facts():
        key = (_normalise_key(fact.subject), _normalise_key(fact.predicate))
        if not key[0] or not key[1]:
            continue
        buckets.setdefault(key, []).append(fact)

    clusters: list[FactCluster] = []
    for (subject, predicate), facts in buckets.items():
        status, spread = _classify(facts, tolerance)
        cluster = FactCluster(
            subject=facts[0].subject,
            predicate=facts[0].predicate,
            facts=facts,
            status=status,
            sources=sorted({f.source_name or f.mcp_server or "unknown" for f in facts}),
            values=[f.value for f in facts],
            units=sorted({f.unit or "" for f in facts}),
            spread=spread,
        )
        clusters.append(cluster)
        if link and status in ("corroborated", "contradicted"):
            _link_cluster(kg, cluster)

    report = ConsensusReport(clusters=clusters, tolerance=tolerance)
    return report


def _link_cluster(kg: KnowledgeGraph, cluster: FactCluster) -> None:
    """Write the adjudication into the graph as edges between the fact nodes."""
    relation = (
        RelationType.SUPPORTS if cluster.status == "corroborated" else RelationType.CONTRADICTS
    )
    properties = {
        "basis": "cross_source_consensus",
        "spread": round(cluster.spread, 4),
        "sources": ", ".join(cluster.sources),
    }
    for i, left in enumerate(cluster.facts):
        for right in cluster.facts[i + 1:]:
            if _source_of(left) == _source_of(right):
                continue  # same upstream twice is repetition, not corroboration
            kg.add_edge(Edge(source=left.id, target=right.id,
                             relation=relation, properties=dict(properties)))
            kg.add_edge(Edge(source=right.id, target=left.id,
                             relation=relation, properties=dict(properties)))
