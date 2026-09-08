"""Registry of every MCP server AstralGraph can talk to, plus agent→server scoping.

This module is the single source of truth for the **MCP permission model**: an
agent may only open sessions to the servers listed for it in
:data:`AGENT_PERMISSIONS`. :mod:`guardrails.permissions` enforces it at call time.
"""

from __future__ import annotations

import os
import shutil
import sys
from dataclasses import dataclass, field
from typing import Literal

from core.config import get_settings

Origin = Literal["custom", "official"]


@dataclass(frozen=True)
class ServerSpec:
    """How to launch one MCP server over stdio."""

    name: str
    command: str
    args: list[str]
    description: str
    origin: Origin
    env: dict[str, str] = field(default_factory=dict)
    enabled: bool = True
    write_capable: bool = False


def _npx() -> str | None:
    """Resolve the npx executable (``npx.cmd`` on Windows)."""
    for candidate in ("npx.cmd", "npx") if os.name == "nt" else ("npx",):
        found = shutil.which(candidate)
        if found:
            return found
    return None


def build_registry() -> dict[str, ServerSpec]:
    settings = get_settings()
    py = sys.executable
    npx = _npx() if settings.enable_node_mcp_servers else None

    common_env = {
        "NASA_API_KEY": settings.nasa_api_key,
        "PYTHONUNBUFFERED": "1",
        "PYTHONIOENCODING": "utf-8",
    }

    specs: list[ServerSpec] = [
        # ---------------- custom servers (this repo) ----------------
        ServerSpec(
            name="nasa_neo",
            command=py,
            args=["-m", "mcp_servers.nasa_neo_server"],
            description="NASA NeoWs — near-Earth object feeds, lookups and catalogue stats.",
            origin="custom",
            env=common_env,
        ),
        ServerSpec(
            name="exoplanet",
            command=py,
            args=["-m", "mcp_servers.exoplanet_server"],
            description="NASA Exoplanet Archive TAP — confirmed planet search and statistics.",
            origin="custom",
            env=common_env,
        ),
        ServerSpec(
            name="iss",
            command=py,
            args=["-m", "mcp_servers.iss_server"],
            description="Open Notify — live ISS position, crew, ground distance.",
            origin="custom",
            env=common_env,
        ),
        ServerSpec(
            name="eonet",
            command=py,
            args=["-m", "mcp_servers.eonet_server"],
            description="NASA EONET v3 — natural events on Earth.",
            origin="custom",
            env=common_env,
        ),
        ServerSpec(
            name="astro_compute",
            command=py,
            args=["-m", "mcp_servers.astro_compute_server"],
            description="Deterministic astronomy calculators — all numbers come from here.",
            origin="custom",
            env=common_env,
        ),
        ServerSpec(
            name="rag",
            command=py,
            args=["-m", "mcp_servers.rag_server"],
            description="Local Chroma retrieval over arXiv astro-ph + HuggingFace astronomy data.",
            origin="custom",
            env={**common_env, "ASTRAL_CHROMA_DIR": str(settings.chroma_dir)},
        ),
        ServerSpec(
            name="jpl_sbdb",
            command=py,
            args=["-m", "mcp_servers.jpl_sbdb_server"],
            description=(
                "NASA JPL Solar System Dynamics — small-body lookup, close approaches and "
                "Sentry impact risk. Independent of NeoWs, so the two can be cross-checked."
            ),
            origin="custom",
            env=common_env,
        ),
        ServerSpec(
            name="space_weather",
            command=py,
            args=["-m", "mcp_servers.space_weather_server"],
            description="NASA DONKI + NOAA SWPC — solar flares, CMEs and geomagnetic storms.",
            origin="custom",
            env=common_env,
        ),
        ServerSpec(
            name="launch",
            command=py,
            args=["-m", "mcp_servers.launch_server"],
            description="Launch Library 2 — upcoming and historical orbital launches.",
            origin="custom",
            env=common_env,
        ),
        # ---------------- official pre-built servers ----------------
        ServerSpec(
            name="filesystem",
            command=npx or "npx",
            args=["-y", "@modelcontextprotocol/server-filesystem", str(settings.fs_root)],
            description="Official Filesystem server — persist reports, findings and logs.",
            origin="official",
            enabled=bool(npx),
            write_capable=True,
        ),
        ServerSpec(
            name="memory",
            command=npx or "npx",
            args=["-y", "@modelcontextprotocol/server-memory"],
            description="Official Memory server — persistent knowledge-graph memory across sessions.",
            origin="official",
            enabled=bool(npx),
            write_capable=True,
            env={"MEMORY_FILE_PATH": str(settings.fs_root / "mcp_memory.json")},
        ),
        ServerSpec(
            name="brave_search",
            command=npx or "npx",
            args=["-y", "@modelcontextprotocol/server-brave-search"],
            description="Official Brave Search server — literature and news lookup on the open web.",
            origin="official",
            enabled=bool(npx and settings.brave_api_key),
            env={"BRAVE_API_KEY": settings.brave_api_key},
        ),
    ]
    return {spec.name: spec for spec in specs}


# --------------------------------------------------------------------------- #
# Permission model: agent -> servers it is allowed to reach.
# --------------------------------------------------------------------------- #

AGENT_PERMISSIONS: dict[str, set[str]] = {
    "intent_parser": set(),                                  # pure reasoning, no tools
    "neo_agent": {"nasa_neo", "jpl_sbdb", "astro_compute"},
    "exoplanet_agent": {"exoplanet", "astro_compute"},
    "events_agent": {"iss", "eonet", "space_weather", "launch", "astro_compute"},
    "literature_agent": {"rag", "brave_search"},
    # The Critic may reach jpl_sbdb because it is a *different* upstream from the
    # one the NEO agent used: independent re-derivation, not re-reading the same
    # source. It still cannot touch nasa_neo.
    "critic_agent": {"astro_compute", "jpl_sbdb", "memory"},
    "orchestrator": {"filesystem", "memory"},
    "ingest": {"rag", "filesystem"},                          # offline pipelines
    "eval_harness": set(),
}

#: Servers whose data is treated as *evidence* (writable into the knowledge graph).
EVIDENCE_SERVERS = {
    "nasa_neo", "jpl_sbdb", "exoplanet", "iss", "eonet", "space_weather", "launch",
    "astro_compute", "rag", "brave_search",
}

#: Server pairs covering the same ground from different upstreams. The consensus
#: layer treats agreement between them as genuine corroboration.
INDEPENDENT_SOURCE_PAIRS = [("nasa_neo", "jpl_sbdb")]


def allowed_servers(agent: str) -> set[str]:
    return set(AGENT_PERMISSIONS.get(agent, set()))


def describe_registry() -> list[dict[str, object]]:
    return [
        {
            "name": spec.name,
            "origin": spec.origin,
            "enabled": spec.enabled,
            "write_capable": spec.write_capable,
            "description": spec.description,
            "command": f"{spec.command} {' '.join(spec.args)}",
            "used_by": sorted(a for a, s in AGENT_PERMISSIONS.items() if spec.name in s),
        }
        for spec in build_registry().values()
    ]
