"""Shared fixtures for the AstralGraph test suite."""

from __future__ import annotations

import pytest

from graph.schema import Fact
from graph.store import KnowledgeGraph


def make_fact(subject: str, predicate: str, value, unit: str | None = None,
              server: str = "nasa_neo", source: str = "NASA NeoWs") -> Fact:
    return Fact(
        subject=subject,
        predicate=predicate,
        value=value,
        unit=unit,
        source_name=source,
        source_url="https://example.invalid/api",
        mcp_server=server,
        mcp_tool="test_tool",
        retrieved_at="2025-01-01T00:00:00+00:00",
        agent="test_agent",
    )


@pytest.fixture
def kg() -> KnowledgeGraph:
    graph = KnowledgeGraph(session_id="pytest")
    graph.add_fact(make_fact("Apophis", "diameter_max_m", 375.0, "m"))
    graph.add_fact(make_fact("Apophis", "close_approach_km", 31600.0, "km"))
    graph.add_fact(
        make_fact("computation:impact_energy", "energy_megatons_tnt", 75.1, "Mt TNT",
                  server="astro_compute", source="AstralGraph deterministic calculator")
    )
    return graph
