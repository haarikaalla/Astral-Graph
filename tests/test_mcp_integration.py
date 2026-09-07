"""MCP integration tests.

Marked ``integration`` because they spawn real MCP server subprocesses and call
live NASA / Open Notify endpoints. Run just the fast unit tests with::

    pytest -m "not integration"
"""

from __future__ import annotations

import pytest

from core.mcp_client import MCPBudgetExceeded, MCPHub, MCPPermissionError
from core.mcp_registry import AGENT_PERMISSIONS, build_registry

pytestmark = [pytest.mark.integration, pytest.mark.anyio]


@pytest.fixture
def anyio_backend():
    return "asyncio"


# --------------------------------------------------------------------------- #
# Registry (no subprocess needed)
# --------------------------------------------------------------------------- #
def test_every_custom_server_is_registered():
    registry = build_registry()
    for name in ("nasa_neo", "exoplanet", "iss", "eonet", "astro_compute", "rag"):
        assert name in registry
        assert registry[name].origin == "custom"


def test_official_servers_are_registered():
    registry = build_registry()
    for name in ("filesystem", "memory", "brave_search"):
        assert name in registry
        assert registry[name].origin == "official"


def test_permission_matrix_only_references_real_servers():
    registry = build_registry()
    for agent, servers in AGENT_PERMISSIONS.items():
        for server in servers:
            assert server in registry, f"{agent} -> unknown server {server}"


def test_no_agent_may_reach_every_server():
    """Least privilege: the permission matrix must actually restrict something."""
    registry = set(build_registry())
    for agent, servers in AGENT_PERMISSIONS.items():
        assert set(servers) != registry, f"{agent} has unrestricted MCP access"


# --------------------------------------------------------------------------- #
# Live stdio sessions
# --------------------------------------------------------------------------- #
async def test_compute_server_round_trip():
    async with MCPHub() as hub:
        result = await hub.call(
            "neo_agent", "astro_compute", "impact_energy",
            {"diameter_m": 100, "velocity_km_s": 20, "density_kg_m3": 3000},
        )
        assert result.ok, result.error
        assert result.data["data"]["energy_megatons_tnt"] == pytest.approx(75.1, rel=0.02)
        assert result.data["source"]["name"]


async def test_tool_discovery_reports_documented_tools():
    async with MCPHub() as hub:
        tools = await hub.list_tools(["astro_compute"])
        names = {t.name for t in tools}
        assert {"impact_energy", "habitable_zone", "convert_distance"} <= names
        assert all(t.description for t in tools)


async def test_permission_is_enforced_at_the_hub():
    async with MCPHub() as hub:
        with pytest.raises(MCPPermissionError):
            await hub.call("neo_agent", "exoplanet", "exoplanet_counts", {})


async def test_tool_call_budget_is_enforced():
    async with MCPHub() as hub:
        hub.settings.max_tool_calls_per_session = 1
        try:
            await hub.call("neo_agent", "astro_compute", "convert_distance",
                           {"value": 1, "from_unit": "au", "to_unit": "km"})
            with pytest.raises(MCPBudgetExceeded):
                await hub.call("neo_agent", "astro_compute", "convert_distance",
                               {"value": 2, "from_unit": "au", "to_unit": "km"})
        finally:
            hub.settings.max_tool_calls_per_session = 24


async def test_invalid_arguments_fail_without_crashing_the_session():
    async with MCPHub() as hub:
        bad = await hub.call("neo_agent", "astro_compute", "impact_energy",
                             {"diameter_m": -1, "velocity_km_s": 20})
        assert bad.ok is False or bad.data.get("ok") is False
        good = await hub.call("neo_agent", "astro_compute", "impact_energy",
                              {"diameter_m": 100, "velocity_km_s": 20})
        assert good.ok, "session must survive a bad call"


async def test_live_iss_position_is_physically_plausible():
    async with MCPHub() as hub:
        result = await hub.call("events_agent", "iss", "iss_now", {})
        if not result.ok:
            pytest.skip(f"ISS API unavailable: {result.error}")
        payload = result.data["data"]
        assert -52.0 <= payload["latitude_deg"] <= 52.0
        assert -180.0 <= payload["longitude_deg"] <= 180.0
