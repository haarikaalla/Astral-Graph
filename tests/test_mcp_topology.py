from core.mcp_registry import (
    AGENT_PERMISSIONS,
    INDEPENDENT_SOURCE_PAIRS,
    describe_mcp_topology,
)


def test_mcp_topology_has_no_unknown_permission_servers() -> None:
    topology = describe_mcp_topology()

    assert topology["unknown_permission_servers"] == []


def test_mcp_topology_exposes_agent_server_edges() -> None:
    topology = describe_mcp_topology()
    edges = {
        (edge["agent"], edge["server"])
        for edge in topology["connections"]
    }

    assert ("neo_agent", "nasa_neo") in edges
    assert ("neo_agent", "jpl_sbdb") in edges
    assert ("orchestrator", "filesystem") in edges
    assert topology["connection_count"] == sum(
        len(servers) for servers in AGENT_PERMISSIONS.values()
    )


def test_mcp_topology_exposes_consensus_pairs() -> None:
    topology = describe_mcp_topology()

    expected_pairs = [
        {"left": left, "right": right}
        for left, right in INDEPENDENT_SOURCE_PAIRS
    ]
    assert topology["independent_source_pairs"] == expected_pairs
