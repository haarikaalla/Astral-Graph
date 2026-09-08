"""MCP server for **orbital launches** — Launch Library 2 (The Space Devs).

Tools
-----
``upcoming_launches``  Next scheduled orbital launches worldwide
``recent_launches``    Launches that already flew, with outcome
``launch_stats``       Aggregate counts by agency, rocket and success rate

Launch Library 2 is a free, keyless public API. It is rate-limited for anonymous
callers, so every response is cached and the tools ask for narrow windows.

Run standalone:  ``python -m mcp_servers.launch_server``
"""

from __future__ import annotations

from collections import Counter
from typing import Any

from mcp_servers._common import clamp, err, fetch_json, ok, utc_now_iso
from mcp_servers._compat import MCPServer

LL2_BASE = "https://ll.thespacedevs.com/2.2.0/launch"
SOURCE = "Launch Library 2 (The Space Devs)"

mcp = MCPServer("astral-launch")


def _text(value: Any, key: str = "name") -> str | None:
    """Launch Library returns either a nested object or a bare string per field.

    ``mode=list`` flattens ``mission``/``pad``/``location`` into plain strings, while
    the detailed endpoint nests them. Tolerate both so the tools keep working if the
    upstream shape changes.
    """
    if value is None:
        return None
    if isinstance(value, str):
        return value or None
    if isinstance(value, dict):
        inner = value.get(key)
        return inner if isinstance(inner, str) and inner else None
    return None


def _summarise(entry: dict[str, Any]) -> dict[str, Any]:
    name = entry.get("name") or ""
    rocket = entry.get("rocket")
    if isinstance(rocket, dict):
        config = rocket.get("configuration") or {}
        rocket_name = config.get("full_name") or config.get("name")
    else:
        # In list mode the name is "<rocket> | <mission>"; take the rocket half.
        rocket_name = name.split("|")[0].strip() or None

    provider = entry.get("launch_service_provider")
    status = entry.get("status") or {}
    return {
        "name": name or None,
        "net": entry.get("net"),
        "status": _text(status, "abbrev"),
        "status_detail": _text(status, "description"),
        "provider": _text(provider) or entry.get("lsp_name"),
        "rocket": rocket_name,
        "mission": _text(entry.get("mission")),
        "mission_type": entry.get("mission_type") or _text(entry.get("mission"), "type"),
        "orbit": _text(entry.get("orbit")),
        "pad": _text(entry.get("pad")),
        "location": _text(entry.get("location")),
        "window_start": entry.get("window_start"),
        "probability_percent": entry.get("probability"),
    }


async def _fetch(path: str, limit: int) -> tuple[list[dict[str, Any]], str, int]:
    rows = clamp(int(limit), 1, 50)
    url = f"{LL2_BASE}/{path}/"
    payload = await fetch_json(
        url,
        params={"limit": rows, "mode": "list"},
        cache_ttl=900.0,
    )
    return (payload.get("results") or []), url, rows


@mcp.tool()
async def upcoming_launches(limit: int = 10) -> dict:
    """The next scheduled orbital launches worldwide.

    Args:
        limit: How many launches to return (1..50).
    """
    try:
        results, url, rows = await _fetch("upcoming", limit)
    except Exception as exc:  # noqa: BLE001
        return err(str(exc), source=SOURCE, url=f"{LL2_BASE}/upcoming/",
                   hint="Launch Library rate-limits anonymous callers; retry shortly")

    launches = [_summarise(entry) for entry in results]
    return ok(
        {
            "requested": rows,
            "count": len(launches),
            "next_launch": launches[0] if launches else None,
            "providers": sorted({l["provider"] for l in launches if l["provider"]}),
            "launches": launches,
            "queried_at": utc_now_iso(),
        },
        source=SOURCE,
        url=url,
    )


@mcp.tool()
async def recent_launches(limit: int = 10) -> dict:
    """Orbital launches that have already flown, most recent first.

    Args:
        limit: How many launches to return (1..50).
    """
    try:
        results, url, rows = await _fetch("previous", limit)
    except Exception as exc:  # noqa: BLE001
        return err(str(exc), source=SOURCE, url=f"{LL2_BASE}/previous/")

    launches = [_summarise(entry) for entry in results]
    successes = sum(1 for l in launches if str(l.get("status") or "").lower() == "success")
    return ok(
        {
            "requested": rows,
            "count": len(launches),
            "success_count": successes,
            "success_rate_percent": (
                round(100.0 * successes / len(launches), 1) if launches else None
            ),
            "most_recent": launches[0] if launches else None,
            "launches": launches,
        },
        source=SOURCE,
        url=url,
    )


@mcp.tool()
async def launch_stats(sample: int = 40) -> dict:
    """Aggregate launch activity over the most recent flights.

    Args:
        sample: How many recent launches to aggregate over (10..50).
    """
    try:
        results, url, _ = await _fetch("previous", clamp(int(sample), 10, 50))
    except Exception as exc:  # noqa: BLE001
        return err(str(exc), source=SOURCE, url=f"{LL2_BASE}/previous/")

    launches = [_summarise(entry) for entry in results]
    if not launches:
        return err("no launches returned", source=SOURCE, url=url)

    providers = Counter(l["provider"] for l in launches if l["provider"])
    rockets = Counter(l["rocket"] for l in launches if l["rocket"])
    orbits = Counter(l["orbit"] for l in launches if l["orbit"])
    successes = sum(1 for l in launches if str(l.get("status") or "").lower() == "success")

    top_provider, top_provider_count = (providers.most_common(1) or [(None, 0)])[0]
    return ok(
        {
            "sample_size": len(launches),
            "success_count": successes,
            "success_rate_percent": round(100.0 * successes / len(launches), 1),
            "busiest_provider": top_provider,
            "busiest_provider_launches": top_provider_count,
            "by_provider": dict(providers.most_common(10)),
            "by_rocket": dict(rockets.most_common(10)),
            "by_orbit": dict(orbits.most_common(10)),
        },
        source=SOURCE,
        url=url,
    )


if __name__ == "__main__":
    mcp.run()
