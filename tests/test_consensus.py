"""Tests for cross-source consensus, contradiction detection and the evidence loop."""

from __future__ import annotations

import pytest

from agents.workflow import MAX_EVIDENCE_ROUNDS, _apply_evidence_requests, _critic_router
from core.mcp_client import ToolResult
from graph.builder import ingest_tool_result
from graph.consensus import analyse_consensus
from graph.schema import NodeType, RelationType
from graph.store import KnowledgeGraph
from guardrails.schemas import CriticVerdict, EvidenceRequest, Intent
from tests.conftest import make_fact


@pytest.fixture
def empty_kg() -> KnowledgeGraph:
    return KnowledgeGraph(session_id="pytest-consensus")


# --------------------------------------------------------------------------- #
# Cross-catalogue alignment
#
# Consensus only works if the two independent NEO catalogues describe the same
# body with the same subject and the same predicate names. That alignment is
# implicit -- it lives in the field names each server chooses -- so a rename on
# either side would silently reduce every claim to "single_source" and the whole
# verification layer would quietly stop doing anything. These tests use payload
# shapes captured from the live APIs so that such a rename fails loudly instead.
# --------------------------------------------------------------------------- #

# Trimmed from a real NeoWs `neo_lookup` response for Apophis.
NEOWS_PAYLOAD = {
    "name": "99942 Apophis (2004 MN4)",
    "absolute_magnitude_h": 19.09,
    "is_potentially_hazardous": True,
    "diameter_min_m": 404.16,
    "diameter_max_m": 903.73,
}

# Trimmed from a real JPL SBDB `sbdb_lookup` response for the same body.
SBDB_PAYLOAD = {
    "name": "99942 Apophis (2004 MN4)",
    "absolute_magnitude_h": 19.09,
    "is_potentially_hazardous": True,
    "designation": "99942",
    "diameter_m": 340.0,
}


def _ingest(kg: KnowledgeGraph, server: str, tool: str, payload: dict) -> None:
    ingest_tool_result(
        kg,
        ToolResult(server=server, tool=tool, ok=True, data=dict(payload)),
        agent="neo_agent",
    )


def test_neows_and_sbdb_agree_on_subject(empty_kg):
    """Both catalogues must land on one subject, not two near-identical ones."""
    _ingest(empty_kg, "nasa_neo", "neo_lookup", NEOWS_PAYLOAD)
    _ingest(empty_kg, "jpl_sbdb", "sbdb_lookup", SBDB_PAYLOAD)

    subjects = {fact.subject for fact in empty_kg.facts()}
    assert "99942 Apophis (2004 MN4)" in subjects


def test_independent_catalogues_produce_cross_checked_claims(empty_kg):
    """The end-to-end guard: real payloads must actually corroborate each other."""
    _ingest(empty_kg, "nasa_neo", "neo_lookup", NEOWS_PAYLOAD)
    _ingest(empty_kg, "jpl_sbdb", "sbdb_lookup", SBDB_PAYLOAD)

    report = analyse_consensus(empty_kg)

    assert len(report.cross_checked) > 0, (
        "no claim was cross-checked -- the two NEO servers have drifted apart on "
        "subject or predicate naming and consensus has silently stopped working"
    )
    corroborated = {c.predicate for c in report.clusters if c.status == "corroborated"}
    assert "absolute_magnitude_h" in corroborated
    assert "is_potentially_hazardous" in corroborated


def test_cross_check_needs_two_distinct_servers(empty_kg):
    """One catalogue alone must never look corroborated, however many fields it has."""
    _ingest(empty_kg, "nasa_neo", "neo_lookup", NEOWS_PAYLOAD)

    report = analyse_consensus(empty_kg)

    assert len(report.cross_checked) == 0


# --------------------------------------------------------------------------- #
# Consensus
# --------------------------------------------------------------------------- #


def test_agreeing_independent_sources_are_corroborated(empty_kg: KnowledgeGraph) -> None:
    empty_kg.add_fact(make_fact("Apophis", "diameter_m", 340.0, "m",
                                server="nasa_neo", source="NASA NeoWs"))
    empty_kg.add_fact(make_fact("Apophis", "diameter_m", 350.0, "m",
                                server="jpl_sbdb", source="NASA JPL SSD"))

    report = analyse_consensus(empty_kg)

    assert len(report.corroborated) == 1
    assert not report.contradicted
    assert report.agreement_rate == 1.0
    assert report.ok


def test_disagreeing_independent_sources_are_flagged(empty_kg: KnowledgeGraph) -> None:
    empty_kg.add_fact(make_fact("Apophis", "diameter_m", 340.0, "m",
                                server="nasa_neo", source="NASA NeoWs"))
    empty_kg.add_fact(make_fact("Apophis", "diameter_m", 900.0, "m",
                                server="jpl_sbdb", source="NASA JPL SSD"))

    report = analyse_consensus(empty_kg)

    assert len(report.contradicted) == 1
    assert not report.ok
    assert report.agreement_rate == 0.0
    assert "disagree" in " ".join(report.caveats()).lower()


def test_same_source_twice_is_not_corroboration(empty_kg: KnowledgeGraph) -> None:
    """Repetition from one upstream must not be mistaken for independent agreement."""
    empty_kg.add_fact(make_fact("Apophis", "diameter_m", 340.0, "m",
                                server="nasa_neo", source="NASA NeoWs"))
    empty_kg.add_fact(make_fact("Apophis", "diameter_m", 341.0, "m",
                                server="nasa_neo", source="NASA NeoWs"))

    report = analyse_consensus(empty_kg)

    assert not report.corroborated
    assert not report.contradicted
    assert len(report.cross_checked) == 0


def test_contradiction_writes_graph_edges(empty_kg: KnowledgeGraph) -> None:
    empty_kg.add_fact(make_fact("Bennu", "diameter_m", 490.0, "m",
                                server="nasa_neo", source="NASA NeoWs"))
    empty_kg.add_fact(make_fact("Bennu", "diameter_m", 1200.0, "m",
                                server="jpl_sbdb", source="NASA JPL SSD"))

    analyse_consensus(empty_kg, link=True)

    relations = [d.get("relation") for _, _, d in empty_kg.g.edges(data=True)]
    assert str(RelationType.CONTRADICTS) in relations


def test_corroboration_writes_supports_edges(empty_kg: KnowledgeGraph) -> None:
    empty_kg.add_fact(make_fact("Bennu", "diameter_m", 490.0, "m",
                                server="nasa_neo", source="NASA NeoWs"))
    empty_kg.add_fact(make_fact("Bennu", "diameter_m", 492.0, "m",
                                server="jpl_sbdb", source="NASA JPL SSD"))

    analyse_consensus(empty_kg, link=True)

    supports = [
        d for _, _, d in empty_kg.g.edges(data=True)
        if d.get("relation") == str(RelationType.SUPPORTS)
        and d.get("basis") == "cross_source_consensus"
    ]
    assert supports, "agreeing independent sources should be linked with SUPPORTS"


def test_incompatible_units_are_not_compared(empty_kg: KnowledgeGraph) -> None:
    """375 m and 0.375 km are the same size, but the module must not guess that."""
    empty_kg.add_fact(make_fact("Apophis", "diameter", 375.0, "m",
                                server="nasa_neo", source="NASA NeoWs"))
    empty_kg.add_fact(make_fact("Apophis", "diameter", 0.375, "km",
                                server="jpl_sbdb", source="NASA JPL SSD"))

    report = analyse_consensus(empty_kg)

    assert len(report.unit_mismatches) == 1
    assert not report.contradicted


def test_unit_aliases_are_treated_as_equal(empty_kg: KnowledgeGraph) -> None:
    empty_kg.add_fact(make_fact("Proxima Centauri b", "distance", 4.24, "ly",
                                server="exoplanet", source="NASA Exoplanet Archive"))
    empty_kg.add_fact(make_fact("Proxima Centauri b", "distance", 4.25, "light-years",
                                server="rag", source="arXiv astro-ph"))

    report = analyse_consensus(empty_kg)

    assert len(report.corroborated) == 1
    assert not report.unit_mismatches


def test_text_containment_counts_as_agreement(empty_kg: KnowledgeGraph) -> None:
    empty_kg.add_fact(make_fact("2004 MN4", "name", "apophis",
                                server="nasa_neo", source="NASA NeoWs"))
    empty_kg.add_fact(make_fact("2004 MN4", "name", "99942 apophis",
                                server="jpl_sbdb", source="NASA JPL SSD"))

    report = analyse_consensus(empty_kg)

    assert len(report.corroborated) == 1


def test_single_source_graph_reports_full_agreement(kg: KnowledgeGraph) -> None:
    """No cross-checks available must not be reported as disagreement."""
    report = analyse_consensus(kg)

    assert report.agreement_rate == 1.0
    assert report.ok
    assert report.to_dict()["cross_checked"] == 0


# --------------------------------------------------------------------------- #
# Evidence loop
# --------------------------------------------------------------------------- #


class _StubBudget:
    def __init__(self, allowed: bool = True) -> None:
        self.allowed = allowed

    def allows(self, _label: str) -> bool:
        return self.allowed


class _StubCtx:
    def __init__(self, allowed: bool = True) -> None:
        self.budget = _StubBudget(allowed)


def _state(**overrides):
    base = {
        "ctx": _StubCtx(),
        "evidence_requests": [{"domain": "literature", "question": "what is a PHA?",
                               "reason": "unsupported claim"}],
        "evidence_round": 1,
    }
    base.update(overrides)
    return base


def test_critic_router_requests_another_round() -> None:
    assert _critic_router(_state()) == ["literature_agent"]


def test_critic_router_stops_at_round_cap() -> None:
    state = _state(evidence_round=MAX_EVIDENCE_ROUNDS)
    assert _critic_router(state) == ["orchestrator"]


def test_critic_router_stops_when_budget_exhausted() -> None:
    state = _state(ctx=_StubCtx(allowed=False))
    assert _critic_router(state) == ["orchestrator"]


def test_critic_router_stops_without_requests() -> None:
    assert _critic_router(_state(evidence_requests=[])) == ["orchestrator"]


def test_critic_router_ignores_unknown_domain() -> None:
    state = _state(evidence_requests=[{"domain": "not_a_domain", "question": "x"}])
    assert _critic_router(state) == ["orchestrator"]


def test_evidence_requests_merge_into_intent() -> None:
    intent = Intent(normalized_question="what is a PHA?", domains=["literature"],
                    sub_questions=["original sub-question"])
    state = _state()

    updated = _apply_evidence_requests(intent, state, "literature_agent")

    assert "original sub-question" in updated.sub_questions
    assert "what is a PHA?" in updated.sub_questions


def test_evidence_requests_do_not_leak_across_agents() -> None:
    intent = Intent(normalized_question="asteroid question", domains=["neo"],
                    sub_questions=["only mine"])
    state = _state()

    updated = _apply_evidence_requests(intent, state, "neo_agent")

    assert updated.sub_questions == ["only mine"]


def test_evidence_request_is_a_validated_contract() -> None:
    verdict = CriticVerdict(
        verdict="revise",
        grounding_assessment="thin evidence",
        evidence_requests=[EvidenceRequest(domain="neo", question="Apophis 2029 distance",
                                           reason="no fact found")],
    )
    assert verdict.evidence_requests[0].domain == "neo"

    with pytest.raises(ValueError):
        EvidenceRequest(domain="astrology", question="what is my sign")  # type: ignore[arg-type]
