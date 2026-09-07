"""Developer smoke test: connect to every custom MCP server and call one tool.

    python scripts/smoke_mcp.py
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.mcp_client import MCPHub  # noqa: E402
from core.mcp_registry import describe_registry  # noqa: E402

PROBES: list[tuple[str, str, dict]] = [
    ("astro_compute", "impact_energy",
     {"diameter_m": 100, "velocity_km_s": 20, "density_kg_m3": 3000}),
    ("astro_compute", "convert_distance", {"value": 1, "from_unit": "pc", "to_unit": "ly"}),
    ("iss", "iss_now", {}),
    ("iss", "iss_crew", {}),
    ("eonet", "eonet_summary", {"days": 30}),
    ("nasa_neo", "neo_feed", {}),
    ("exoplanet", "exoplanet_counts", {}),
    ("rag", "literature_stats", {}),
]

AGENT_FOR = {
    "astro_compute": "neo_agent",
    "iss": "events_agent",
    "eonet": "events_agent",
    "nasa_neo": "neo_agent",
    "exoplanet": "exoplanet_agent",
    "rag": "literature_agent",
}


async def main() -> int:
    print("=== MCP registry ===")
    for spec in describe_registry():
        flag = "on " if spec["enabled"] else "OFF"
        print(f"  [{flag}] {spec['name']:<14} ({spec['origin']}) used by {spec['used_by']}")

    failures = 0
    async with MCPHub() as hub:
        hub.settings.max_tool_calls_per_session = 100
        print("\n=== tool discovery ===")
        for server in {p[0] for p in PROBES}:
            tools = await hub.list_tools([server])
            print(f"  {server:<14} {len(tools)} tools: {[t.name for t in tools]}")

        print("\n=== tool probes ===")
        for server, tool, args in PROBES:
            result = await hub.call(AGENT_FOR[server], server, tool, args)
            status = "OK  " if result.ok else "FAIL"
            preview = json.dumps(result.data, default=str)[:180] if result.ok else result.error[:180]
            print(f"  [{status}] {server}.{tool}: {preview}")
            failures += 0 if result.ok else 1

        if hub.failed_servers:
            print("\nunreachable servers:", hub.failed_servers)

    print(f"\n{len(PROBES) - failures}/{len(PROBES)} probes succeeded")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
