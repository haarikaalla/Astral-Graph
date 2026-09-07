"""MCP server wrapping the **Open Notify ISS** API (plus a resilient fallback).

Tools
-----
``iss_now``            Current ISS sub-satellite point (lat/lon) and timestamp
``iss_crew``           People currently in space, grouped by spacecraft
``iss_ground_distance``Great-circle distance from the ISS to a lat/lon on Earth
``iss_track``          Short forward track sampled from live telemetry

Open Notify is a small community service that goes down occasionally, so
``iss_now`` transparently falls back to ``wheretheiss.at`` and reports which
upstream actually answered.

Run standalone:  ``python -m mcp_servers.iss_server``
"""

from __future__ import annotations

import math
from typing import Any

from mcp_servers._common import clamp, err, fetch_json, ok, safe_float, utc_now_iso
from mcp_servers._compat import MCPServer

OPEN_NOTIFY_NOW = "http://api.open-notify.org/iss-now.json"
OPEN_NOTIFY_ASTROS = "http://api.open-notify.org/astros.json"
WTIA = "https://api.wheretheiss.at/v1/satellites/25544"
SOURCE = "Open Notify ISS API"
FALLBACK_SOURCE = "wheretheiss.at (fallback)"
EARTH_RADIUS_KM = 6371.0088

mcp = MCPServer("astral-iss")


async def _position() -> tuple[dict[str, Any], str, str]:
    """Return (position, source_name, url). Tries Open Notify, then wheretheiss.at."""
    try:
        payload = await fetch_json(OPEN_NOTIFY_NOW, cache_ttl=5.0)
        pos = payload.get("iss_position") or {}
        return (
            {
                "latitude_deg": safe_float(pos.get("latitude")),
                "longitude_deg": safe_float(pos.get("longitude")),
                "timestamp_unix": payload.get("timestamp"),
                "altitude_km": None,
                "velocity_km_h": None,
            },
            SOURCE,
            OPEN_NOTIFY_NOW,
        )
    except Exception:  # noqa: BLE001
        payload = await fetch_json(WTIA, cache_ttl=5.0)
        return (
            {
                "latitude_deg": safe_float(payload.get("latitude")),
                "longitude_deg": safe_float(payload.get("longitude")),
                "timestamp_unix": payload.get("timestamp"),
                "altitude_km": safe_float(payload.get("altitude")),
                "velocity_km_h": safe_float(payload.get("velocity")),
            },
            FALLBACK_SOURCE,
            WTIA,
        )


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = p2 - p1
    dlam = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlam / 2) ** 2
    return 2 * EARTH_RADIUS_KM * math.asin(min(1.0, math.sqrt(a)))


@mcp.tool()
async def iss_now() -> dict:
    """Current latitude/longitude of the International Space Station."""
    try:
        pos, source, url = await _position()
    except Exception as exc:  # noqa: BLE001
        return err(str(exc), source=SOURCE, url=OPEN_NOTIFY_NOW)
    pos["queried_at"] = utc_now_iso()
    pos["norad_id"] = 25544
    return ok(pos, source=source, url=url)


@mcp.tool()
async def iss_crew() -> dict:
    """Everyone currently in space, grouped by spacecraft, with the ISS crew isolated."""
    try:
        payload = await fetch_json(OPEN_NOTIFY_ASTROS, cache_ttl=600.0)
    except Exception as exc:  # noqa: BLE001
        return err(str(exc), source=SOURCE, url=OPEN_NOTIFY_ASTROS)

    people = payload.get("people") or []
    by_craft: dict[str, list[str]] = {}
    for person in people:
        by_craft.setdefault(person.get("craft", "Unknown"), []).append(person.get("name", "?"))
    return ok(
        {
            "total_people_in_space": int(payload.get("number", len(people))),
            "iss_crew_size": len(by_craft.get("ISS", [])),
            "by_craft": by_craft,
            "people": people,
        },
        source=SOURCE,
        url=OPEN_NOTIFY_ASTROS,
    )


@mcp.tool()
async def iss_ground_distance(latitude: float, longitude: float) -> dict:
    """Great-circle distance from the ISS ground track to a point on Earth.

    Args:
        latitude: Observer latitude in degrees (-90..90).
        longitude: Observer longitude in degrees (-180..180).
    """
    if not -90 <= latitude <= 90 or not -180 <= longitude <= 180:
        return err("latitude must be -90..90 and longitude -180..180", source=SOURCE)
    try:
        pos, source, url = await _position()
    except Exception as exc:  # noqa: BLE001
        return err(str(exc), source=SOURCE, url=OPEN_NOTIFY_NOW)

    lat, lon = pos["latitude_deg"], pos["longitude_deg"]
    if lat is None or lon is None:
        return err("upstream returned no position", source=source, url=url)

    ground_km = _haversine_km(latitude, longitude, lat, lon)
    altitude = pos.get("altitude_km") or 420.0
    slant_km = math.sqrt(ground_km**2 + altitude**2)
    # Visible when within the ISS horizon circle for its altitude.
    horizon_km = EARTH_RADIUS_KM * math.acos(
        EARTH_RADIUS_KM / (EARTH_RADIUS_KM + altitude)
    )
    return ok(
        {
            "observer": {"latitude_deg": latitude, "longitude_deg": longitude},
            "iss_position": pos,
            "ground_distance_km": round(ground_km, 2),
            "approx_slant_range_km": round(slant_km, 2),
            "horizon_radius_km": round(horizon_km, 2),
            "currently_above_horizon": ground_km <= horizon_km,
            "assumed_altitude_km": altitude,
        },
        source=source,
        url=url,
    )


@mcp.tool()
async def iss_track(samples: int = 3, spacing_seconds: int = 30) -> dict:
    """Sample the ISS position a few times to show its direction of travel.

    Args:
        samples: Number of samples (2-6).
        spacing_seconds: Delay between samples (5-60).
    """
    import asyncio

    n = clamp(samples, 2, 6)
    gap = clamp(spacing_seconds, 5, 60)
    points: list[dict[str, Any]] = []
    source, url = SOURCE, OPEN_NOTIFY_NOW
    for i in range(n):
        try:
            pos, source, url = await _position()
        except Exception as exc:  # noqa: BLE001
            return err(str(exc), source=SOURCE, url=OPEN_NOTIFY_NOW)
        points.append(pos)
        if i < n - 1:
            await asyncio.sleep(gap)

    total_km = sum(
        _haversine_km(
            points[i]["latitude_deg"], points[i]["longitude_deg"],
            points[i + 1]["latitude_deg"], points[i + 1]["longitude_deg"],
        )
        for i in range(len(points) - 1)
        if points[i]["latitude_deg"] is not None and points[i + 1]["latitude_deg"] is not None
    )
    seconds = gap * (len(points) - 1)
    return ok(
        {
            "points": points,
            "sampled_seconds": seconds,
            "ground_track_km": round(total_km, 2),
            "implied_ground_speed_km_s": round(total_km / seconds, 3) if seconds else None,
        },
        source=source,
        url=url,
    )


if __name__ == "__main__":
    mcp.run(transport="stdio")
