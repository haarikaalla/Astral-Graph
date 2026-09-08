"""MCP server for **space weather** — NASA DONKI plus NOAA SWPC.

Tools
-----
``solar_flares``           Recent solar flares with class and peak time (DONKI FLR)
``coronal_mass_ejections`` Recent CMEs with speed and direction (DONKI CME)
``geomagnetic_storms``     Recent geomagnetic storms with Kp index (DONKI GST)
``space_weather_now``      Current planetary K-index and storm level (NOAA SWPC)

Two upstreams on purpose. DONKI is NASA's curated event catalogue and needs the
NASA key; NOAA SWPC is a live telemetry feed that needs **no key at all**, so
``space_weather_now`` keeps working when DONKI is rate-limited.

Run standalone:  ``python -m mcp_servers.space_weather_server``
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

from mcp_servers._common import clamp, err, fetch_json, nasa_key, ok, safe_float, utc_now_iso
from mcp_servers._compat import MCPServer

DONKI_BASE = "https://api.nasa.gov/DONKI"
NOAA_KP = "https://services.swpc.noaa.gov/json/planetary_k_index_1m.json"
DONKI_SOURCE = "NASA DONKI (Space Weather Database)"
NOAA_SOURCE = "NOAA Space Weather Prediction Center"

#: NOAA G-scale thresholds, keyed by planetary K-index.
_G_SCALE = [
    (9.0, "G5 — extreme"),
    (8.0, "G4 — severe"),
    (7.0, "G3 — strong"),
    (6.0, "G2 — moderate"),
    (5.0, "G1 — minor"),
    (0.0, "G0 — quiet to unsettled"),
]

mcp = MCPServer("astral-space-weather")


def _window(days: int) -> tuple[str, str, int]:
    span = clamp(int(days), 1, 60)
    end = datetime.now(timezone.utc).date()
    start = end - timedelta(days=span)
    return start.isoformat(), end.isoformat(), span


async def _donki(endpoint: str, days: int) -> tuple[list[dict[str, Any]], str, int]:
    start, end, span = _window(days)
    url = f"{DONKI_BASE}/{endpoint}"
    payload = await fetch_json(
        url,
        params={"startDate": start, "endDate": end, "api_key": nasa_key()},
        cache_ttl=1800.0,
    )
    return (payload if isinstance(payload, list) else []), url, span


def _flare_energy_rank(class_type: str) -> int:
    """Order flare classes so 'strongest' is comparable: A < B < C < M < X."""
    order = {"A": 0, "B": 1, "C": 2, "M": 3, "X": 4}
    return order.get(str(class_type or "")[:1].upper(), -1)


@mcp.tool()
async def solar_flares(days: int = 7) -> dict:
    """Solar flares recorded in the last N days, strongest first.

    Args:
        days: Look-back window in days (1..60).
    """
    try:
        events, url, span = await _donki("FLR", days)
    except Exception as exc:  # noqa: BLE001
        return err(str(exc), source=DONKI_SOURCE, url=f"{DONKI_BASE}/FLR",
                   hint="a free NASA_API_KEY avoids DEMO_KEY rate limits")

    flares = [
        {
            "flare_id": e.get("flrID"),
            "class_type": e.get("classType"),
            "begin_time": e.get("beginTime"),
            "peak_time": e.get("peakTime"),
            "end_time": e.get("endTime"),
            "source_location": e.get("sourceLocation"),
            "active_region": e.get("activeRegionNum"),
        }
        for e in events
    ]
    flares.sort(key=lambda f: _flare_energy_rank(f["class_type"]), reverse=True)
    strongest = flares[0] if flares else None
    return ok(
        {
            "window_days": span,
            "count": len(flares),
            "strongest_flare": strongest,
            "strongest_class": (strongest or {}).get("class_type"),
            "x_class_count": sum(1 for f in flares if str(f["class_type"] or "").startswith("X")),
            "m_class_count": sum(1 for f in flares if str(f["class_type"] or "").startswith("M")),
            "flares": flares[:40],
        },
        source=DONKI_SOURCE,
        url=url,
    )


@mcp.tool()
async def coronal_mass_ejections(days: int = 7) -> dict:
    """Coronal mass ejections in the last N days, fastest first.

    Args:
        days: Look-back window in days (1..60).
    """
    try:
        events, url, span = await _donki("CME", days)
    except Exception as exc:  # noqa: BLE001
        return err(str(exc), source=DONKI_SOURCE, url=f"{DONKI_BASE}/CME")

    ejections = []
    for event in events:
        analyses = event.get("cmeAnalyses") or []
        best = analyses[0] if analyses else {}
        ejections.append(
            {
                "activity_id": event.get("activityID"),
                "start_time": event.get("startTime"),
                "speed_km_s": safe_float(best.get("speed")),
                "half_angle_deg": safe_float(best.get("halfAngle")),
                "latitude_deg": safe_float(best.get("latitude")),
                "longitude_deg": safe_float(best.get("longitude")),
                "is_earth_directed": bool(best.get("isMostAccurate")) and
                                     abs(safe_float(best.get("longitude")) or 180.0) < 45.0,
                "note": (event.get("note") or "")[:300],
            }
        )
    with_speed = [e for e in ejections if e["speed_km_s"] is not None]
    with_speed.sort(key=lambda e: e["speed_km_s"], reverse=True)
    return ok(
        {
            "window_days": span,
            "count": len(ejections),
            "fastest_cme": with_speed[0] if with_speed else None,
            "fastest_speed_km_s": with_speed[0]["speed_km_s"] if with_speed else None,
            "earth_directed_count": sum(1 for e in ejections if e["is_earth_directed"]),
            "events": ejections[:30],
        },
        source=DONKI_SOURCE,
        url=url,
    )


@mcp.tool()
async def geomagnetic_storms(days: int = 30) -> dict:
    """Geomagnetic storms in the last N days, with peak Kp index.

    Args:
        days: Look-back window in days (1..60).
    """
    try:
        events, url, span = await _donki("GST", days)
    except Exception as exc:  # noqa: BLE001
        return err(str(exc), source=DONKI_SOURCE, url=f"{DONKI_BASE}/GST")

    storms = []
    for event in events:
        kp_values = [safe_float(k.get("kpIndex")) for k in (event.get("allKpIndex") or [])]
        kp_values = [k for k in kp_values if k is not None]
        storms.append(
            {
                "storm_id": event.get("gstID"),
                "start_time": event.get("startTime"),
                "peak_kp_index": max(kp_values) if kp_values else None,
                "observations": len(kp_values),
            }
        )
    peaks = [s["peak_kp_index"] for s in storms if s["peak_kp_index"] is not None]
    return ok(
        {
            "window_days": span,
            "count": len(storms),
            "max_kp_index": max(peaks) if peaks else None,
            "severity": _g_scale(max(peaks)) if peaks else "no storms recorded",
            "storms": storms[:30],
        },
        source=DONKI_SOURCE,
        url=url,
    )


def _g_scale(kp: float) -> str:
    for threshold, label in _G_SCALE:
        if kp >= threshold:
            return label
    return "G0 — quiet"


@mcp.tool()
async def space_weather_now() -> dict:
    """Current planetary K-index and storm level from NOAA SWPC. Needs no API key."""
    try:
        payload = await fetch_json(NOAA_KP, cache_ttl=300.0)
    except Exception as exc:  # noqa: BLE001
        return err(str(exc), source=NOAA_SOURCE, url=NOAA_KP)

    readings = payload if isinstance(payload, list) else []
    if not readings:
        return err("NOAA returned no K-index readings", source=NOAA_SOURCE, url=NOAA_KP)

    latest = readings[-1]
    kp = safe_float(latest.get("kp_index"))
    recent = [safe_float(r.get("kp_index")) for r in readings[-1440:]]
    recent = [k for k in recent if k is not None]
    return ok(
        {
            "planetary_k_index": kp,
            "storm_level": _g_scale(kp) if kp is not None else "unknown",
            "geomagnetic_storm_in_progress": bool(kp is not None and kp >= 5.0),
            "observed_at": latest.get("time_tag"),
            "max_kp_last_24h": max(recent) if recent else None,
            "readings_considered": len(recent),
            "queried_at": utc_now_iso(),
        },
        source=NOAA_SOURCE,
        url=NOAA_KP,
    )


if __name__ == "__main__":
    mcp.run()
