"""MCP server wrapping **NASA EONET v3** (Earth Observatory Natural Event Tracker).

Tools
-----
``eonet_events``        Filter natural events by status/category/days/bbox
``eonet_categories``    List the 13 EONET event categories
``eonet_event``         One event by id
``eonet_nearby_events`` Open events within a radius of a lat/lon
``eonet_summary``       Counts of open events per category (for the knowledge graph)

Run standalone:  ``python -m mcp_servers.eonet_server``
"""

from __future__ import annotations

import math
from typing import Any

from mcp_servers._common import clamp, err, fetch_json, ok
from mcp_servers._compat import MCPServer

BASE = "https://eonet.gsfc.nasa.gov/api/v3"
SOURCE = "NASA EONET v3"
EARTH_RADIUS_KM = 6371.0088

mcp = MCPServer("astral-eonet")


def _point(event: dict[str, Any]) -> tuple[float, float] | None:
    geometry = event.get("geometry") or []
    if not geometry:
        return None
    coords = geometry[-1].get("coordinates")
    if not isinstance(coords, list) or len(coords) < 2:
        return None
    try:
        if isinstance(coords[0], list):  # Polygon -> use first vertex
            lon, lat = coords[0][0][:2] if isinstance(coords[0][0], list) else coords[0][:2]
        else:
            lon, lat = coords[:2]
        return float(lat), float(lon)
    except (TypeError, ValueError, IndexError):
        return None


def _shape(event: dict[str, Any]) -> dict[str, Any]:
    geometry = event.get("geometry") or []
    latlon = _point(event)
    return {
        "id": event.get("id"),
        "title": event.get("title"),
        "description": event.get("description"),
        "closed": event.get("closed"),
        "status": "closed" if event.get("closed") else "open",
        "categories": [c.get("title") for c in event.get("categories", [])],
        "category_ids": [c.get("id") for c in event.get("categories", [])],
        "latest_date": geometry[-1].get("date") if geometry else None,
        "first_date": geometry[0].get("date") if geometry else None,
        "latitude": latlon[0] if latlon else None,
        "longitude": latlon[1] if latlon else None,
        "geometry_points": len(geometry),
        "sources": [s.get("url") for s in event.get("sources", [])],
        "link": event.get("link"),
    }


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    p1, p2 = math.radians(lat1), math.radians(lat2)
    a = (
        math.sin((p2 - p1) / 2) ** 2
        + math.cos(p1) * math.cos(p2) * math.sin(math.radians(lon2 - lon1) / 2) ** 2
    )
    return 2 * EARTH_RADIUS_KM * math.asin(min(1.0, math.sqrt(a)))


@mcp.tool()
async def eonet_events(
    status: str = "open",
    category: str = "",
    days: int = 30,
    limit: int = 30,
    bbox: str = "",
) -> dict:
    """List natural events tracked by NASA EONET.

    Args:
        status: ``open``, ``closed`` or ``all``.
        category: Category id, e.g. ``wildfires``, ``severeStorms``, ``volcanoes``,
            ``seaLakeIce``, ``earthquakes``, ``floods``, ``drought``, ``dustHaze``,
            ``landslides``, ``manmade``, ``snow``, ``tempExtremes``, ``waterColor``.
        days: Look-back window in days (1-365).
        limit: Max events returned (1-200).
        bbox: Optional ``minLon,maxLat,maxLon,minLat`` bounding box.
    """
    if status not in {"open", "closed", "all"}:
        return err("status must be open, closed or all", source=SOURCE)
    params: dict[str, Any] = {"days": clamp(days, 1, 365), "limit": clamp(limit, 1, 200)}
    if status != "all":
        params["status"] = status
    if category.strip():
        params["category"] = category.strip()
    if bbox.strip():
        parts = bbox.split(",")
        if len(parts) != 4:
            return err("bbox must be 'minLon,maxLat,maxLon,minLat'", source=SOURCE)
        params["bbox"] = bbox.strip()

    url = f"{BASE}/events"
    try:
        payload = await fetch_json(url, params=params)
    except Exception as exc:  # noqa: BLE001
        return err(str(exc), source=SOURCE, url=url)

    events = [_shape(e) for e in payload.get("events", [])]
    counts: dict[str, int] = {}
    for event in events:
        for name in event["categories"]:
            counts[name] = counts.get(name, 0) + 1
    return ok(
        {
            "filters": params,
            "count": len(events),
            "counts_by_category": dict(sorted(counts.items(), key=lambda kv: -kv[1])),
            "events": events,
        },
        source=SOURCE,
        url=url,
    )


@mcp.tool()
async def eonet_categories() -> dict:
    """List every EONET event category with its id, title and description."""
    url = f"{BASE}/categories"
    try:
        payload = await fetch_json(url, cache_ttl=3600.0)
    except Exception as exc:  # noqa: BLE001
        return err(str(exc), source=SOURCE, url=url)
    return ok(
        {
            "count": len(payload.get("categories", [])),
            "categories": [
                {"id": c.get("id"), "title": c.get("title"), "description": c.get("description")}
                for c in payload.get("categories", [])
            ],
        },
        source=SOURCE,
        url=url,
    )


@mcp.tool()
async def eonet_event(event_id: str) -> dict:
    """Fetch one EONET event by id, e.g. ``EONET_6543``.

    Args:
        event_id: The EONET event identifier.
    """
    if not event_id.strip():
        return err("event_id is required", source=SOURCE)
    url = f"{BASE}/events/{event_id.strip()}"
    try:
        payload = await fetch_json(url)
    except Exception as exc:  # noqa: BLE001
        return err(str(exc), source=SOURCE, url=url)
    return ok(_shape(payload), source=SOURCE, url=url)


@mcp.tool()
async def eonet_nearby_events(
    latitude: float, longitude: float, radius_km: float = 1000.0, days: int = 60, limit: int = 15
) -> dict:
    """Open natural events within ``radius_km`` of a point.

    Args:
        latitude: Degrees, -90..90.
        longitude: Degrees, -180..180.
        radius_km: Search radius in kilometres (10-20000).
        days: Look-back window in days.
        limit: Max events to return after filtering.
    """
    if not -90 <= latitude <= 90 or not -180 <= longitude <= 180:
        return err("latitude must be -90..90 and longitude -180..180", source=SOURCE)
    radius = max(10.0, min(20000.0, radius_km))
    base = await eonet_events(status="open", days=days, limit=200)
    if not base.get("ok"):
        return base

    scored = []
    for event in base["data"]["events"]:
        if event["latitude"] is None or event["longitude"] is None:
            continue
        distance = _haversine_km(latitude, longitude, event["latitude"], event["longitude"])
        if distance <= radius:
            scored.append({**event, "distance_km": round(distance, 1)})
    scored.sort(key=lambda e: e["distance_km"])
    return ok(
        {
            "origin": {"latitude": latitude, "longitude": longitude, "radius_km": radius},
            "count": len(scored),
            "events": scored[: clamp(limit, 1, 100)],
        },
        source=SOURCE,
        url=f"{BASE}/events",
    )


@mcp.tool()
async def eonet_summary(days: int = 30) -> dict:
    """Counts of currently-open events per category — compact enough for the KG.

    Args:
        days: Look-back window in days (1-365).
    """
    base = await eonet_events(status="open", days=days, limit=200)
    if not base.get("ok"):
        return base
    data = base["data"]
    return ok(
        {
            "days": days,
            "total_open_events": data["count"],
            "counts_by_category": data["counts_by_category"],
            "most_active_category": next(iter(data["counts_by_category"]), None),
            "sample_titles": [e["title"] for e in data["events"][:10]],
        },
        source=SOURCE,
        url=f"{BASE}/events",
    )


if __name__ == "__main__":
    mcp.run(transport="stdio")
