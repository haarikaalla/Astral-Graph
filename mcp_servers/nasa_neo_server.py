"""MCP server wrapping the **NASA NeoWs (Near-Earth Object Web Service)** API.

Tools
-----
``neo_feed``            Close approaches in a date window (max 7 days per NASA rules)
``neo_lookup``         Full record for one object by SPK-ID / designation
``neo_browse``         Paginated catalogue browse
``neo_hazardous_today``Convenience: today's potentially-hazardous approaches
``neo_stats``          Catalogue-level counts

Run standalone:  ``python -m mcp_servers.nasa_neo_server``
"""

from __future__ import annotations

from datetime import date, timedelta
from typing import Any

from mcp_servers._common import clamp, err, fetch_json, nasa_key, ok, safe_float
from mcp_servers._compat import MCPServer

BASE = "https://api.nasa.gov/neo/rest/v1"
SOURCE = "NASA NeoWs"

mcp = MCPServer("astral-nasa-neo")


def _summarise(neo: dict[str, Any], approach: dict[str, Any] | None = None) -> dict[str, Any]:
    diameter = neo.get("estimated_diameter", {}).get("meters", {})
    record: dict[str, Any] = {
        "id": neo.get("id"),
        "neo_reference_id": neo.get("neo_reference_id"),
        "name": (neo.get("name") or "").strip(),
        "absolute_magnitude_h": safe_float(neo.get("absolute_magnitude_h")),
        "diameter_min_m": safe_float(diameter.get("estimated_diameter_min")),
        "diameter_max_m": safe_float(diameter.get("estimated_diameter_max")),
        "is_potentially_hazardous": bool(neo.get("is_potentially_hazardous_asteroid")),
        "is_sentry_object": bool(neo.get("is_sentry_object")),
        "nasa_jpl_url": neo.get("nasa_jpl_url"),
    }
    approach = approach or (neo.get("close_approach_data") or [{}])[0]
    if approach:
        record["close_approach"] = {
            "date": approach.get("close_approach_date_full") or approach.get("close_approach_date"),
            "epoch_ms": approach.get("epoch_date_close_approach"),
            "relative_velocity_km_s": safe_float(
                (approach.get("relative_velocity") or {}).get("kilometers_per_second")
            ),
            "miss_distance_km": safe_float((approach.get("miss_distance") or {}).get("kilometers")),
            "miss_distance_lunar": safe_float((approach.get("miss_distance") or {}).get("lunar")),
            "orbiting_body": approach.get("orbiting_body"),
        }
    return record


@mcp.tool()
async def neo_feed(start_date: str = "", end_date: str = "", hazardous_only: bool = False) -> dict:
    """List near-Earth objects making a close approach in a date window.

    Args:
        start_date: ISO date ``YYYY-MM-DD``. Defaults to today (UTC).
        end_date: ISO date. Defaults to ``start_date``. NASA allows a 7-day span max.
        hazardous_only: When true, return only potentially hazardous asteroids.
    """
    start = start_date or date.today().isoformat()
    end = end_date or start
    try:
        if date.fromisoformat(end) - date.fromisoformat(start) > timedelta(days=7):
            return err(
                "date range exceeds NASA's 7-day limit",
                source=SOURCE,
                hint="Split the query into 7-day windows.",
            )
    except ValueError as exc:
        return err(f"invalid date: {exc}", source=SOURCE, hint="Use YYYY-MM-DD.")

    url = f"{BASE}/feed"
    params = {"start_date": start, "end_date": end, "api_key": nasa_key()}
    try:
        payload = await fetch_json(url, params=params)
    except Exception as exc:  # noqa: BLE001
        return err(str(exc), source=SOURCE, url=url)

    by_day: dict[str, list[dict[str, Any]]] = {}
    for day, objects in sorted((payload.get("near_earth_objects") or {}).items()):
        rows = [_summarise(o) for o in objects]
        if hazardous_only:
            rows = [r for r in rows if r["is_potentially_hazardous"]]
        by_day[day] = sorted(
            rows, key=lambda r: (r.get("close_approach") or {}).get("miss_distance_km") or 9e18
        )

    flat = [r for rows in by_day.values() for r in rows]
    return ok(
        {
            "start_date": start,
            "end_date": end,
            "element_count": len(flat),
            "hazardous_count": sum(1 for r in flat if r["is_potentially_hazardous"]),
            "closest": flat and min(
                flat, key=lambda r: (r.get("close_approach") or {}).get("miss_distance_km") or 9e18
            ) or None,
            "by_date": by_day,
        },
        source=SOURCE,
        url=url,
    )


@mcp.tool()
async def neo_lookup(asteroid_id: str) -> dict:
    """Fetch the full record (orbit + all close approaches) for a single NEO.

    Args:
        asteroid_id: NeoWs id / SPK-ID, e.g. ``3542519`` or ``2099942`` (Apophis).
    """
    if not asteroid_id.strip():
        return err("asteroid_id is required", source=SOURCE)
    url = f"{BASE}/neo/{asteroid_id.strip()}"
    try:
        payload = await fetch_json(url, params={"api_key": nasa_key()})
    except Exception as exc:  # noqa: BLE001
        return err(str(exc), source=SOURCE, url=url)

    record = _summarise(payload)
    orbit = payload.get("orbital_data") or {}
    record["orbital_data"] = {
        "orbit_class": (orbit.get("orbit_class") or {}).get("orbit_class_type"),
        "orbit_class_description": (orbit.get("orbit_class") or {}).get("orbit_class_description"),
        "semi_major_axis_au": safe_float(orbit.get("semi_major_axis")),
        "eccentricity": safe_float(orbit.get("eccentricity")),
        "inclination_deg": safe_float(orbit.get("inclination")),
        "orbital_period_days": safe_float(orbit.get("orbital_period")),
        "perihelion_distance_au": safe_float(orbit.get("perihelion_distance")),
        "aphelion_distance_au": safe_float(orbit.get("aphelion_distance")),
        "minimum_orbit_intersection_au": safe_float(orbit.get("minimum_orbit_intersection")),
        "first_observation_date": orbit.get("first_observation_date"),
        "last_observation_date": orbit.get("last_observation_date"),
    }
    approaches = payload.get("close_approach_data") or []
    record["close_approach_count"] = len(approaches)
    record["next_approaches"] = [
        _summarise(payload, a)["close_approach"] for a in approaches[:5]
    ]
    return ok(record, source=SOURCE, url=url)


@mcp.tool()
async def neo_browse(page: int = 0, size: int = 20) -> dict:
    """Browse the NEO catalogue page by page.

    Args:
        page: Zero-based page index.
        size: Records per page (1-40).
    """
    url = f"{BASE}/neo/browse"
    params = {"page": max(0, page), "size": clamp(size, 1, 40), "api_key": nasa_key()}
    try:
        payload = await fetch_json(url, params=params)
    except Exception as exc:  # noqa: BLE001
        return err(str(exc), source=SOURCE, url=url)
    return ok(
        {
            "page": payload.get("page"),
            "objects": [_summarise(o) for o in payload.get("near_earth_objects", [])],
        },
        source=SOURCE,
        url=url,
    )


@mcp.tool()
async def neo_hazardous_today() -> dict:
    """Potentially hazardous asteroids approaching Earth today (UTC)."""
    return await neo_feed(hazardous_only=True)


@mcp.tool()
async def neo_stats() -> dict:
    """Catalogue-level statistics: total known NEOs and how many are hazardous."""
    url = f"{BASE}/stats"
    try:
        payload = await fetch_json(url, params={"api_key": nasa_key()})
    except Exception as exc:  # noqa: BLE001
        return err(str(exc), source=SOURCE, url=url)
    return ok(payload, source=SOURCE, url=url)


if __name__ == "__main__":
    mcp.run(transport="stdio")
