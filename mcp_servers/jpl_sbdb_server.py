"""MCP server wrapping **JPL Solar System Dynamics** — an *independent* NEO source.

Tools
-----
``sbdb_lookup``       Orbit + physical parameters for one small body
``close_approaches``  Upcoming close approaches to Earth (CAD API)
``sentry_risk``       Impact-probability table from the Sentry monitoring system
``sbdb_compare``      Side-by-side of the same body's key numbers, for cross-checking

Why this exists
---------------
``nasa_neo`` already covers near-Earth objects via NeoWs. This server deliberately
duplicates part of that coverage using a **different upstream** (JPL SSD rather
than NeoWs) so the consensus layer in :mod:`graph.consensus` has two genuinely
independent readings to adjudicate. Agreement between them is real corroboration;
disagreement is surfaced rather than silently resolved.

It also needs no API key at all, so NEO questions keep working when the shared
NASA ``DEMO_KEY`` is rate-limited.

Run standalone:  ``python -m mcp_servers.jpl_sbdb_server``
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

from mcp_servers._common import clamp, err, fetch_json, ok, safe_float, utc_now_iso
from mcp_servers._compat import MCPServer

SBDB_API = "https://ssd-api.jpl.nasa.gov/sbdb.api"
CAD_API = "https://ssd-api.jpl.nasa.gov/cad.api"
SENTRY_API = "https://ssd-api.jpl.nasa.gov/sentry.api"
SOURCE = "NASA JPL Solar System Dynamics"

AU_KM = 149_597_870.7
LD_KM = 384_400.0

mcp = MCPServer("astral-jpl-sbdb")


def _physical_parameters(payload: dict[str, Any]) -> dict[str, Any]:
    """Flatten SBDB's ``phys_par`` list into a plain mapping."""
    out: dict[str, Any] = {}
    for entry in payload.get("phys_par") or []:
        name = str(entry.get("name", "")).strip()
        if name:
            out[name] = safe_float(entry.get("value"))
    return out


def _orbital_elements(payload: dict[str, Any]) -> dict[str, Any]:
    orbit = payload.get("orbit") or {}
    elements = {}
    for entry in orbit.get("elements") or []:
        name = str(entry.get("name", "")).strip()
        if name:
            elements[name] = safe_float(entry.get("value"))
    return {
        "orbit_class": ((orbit.get("orbit_class") or {}).get("name")),
        "eccentricity": elements.get("e"),
        "semi_major_axis_au": elements.get("a"),
        "perihelion_au": elements.get("q"),
        "aphelion_au": elements.get("ad"),
        "inclination_deg": elements.get("i"),
        "orbital_period_days": elements.get("per"),
        "minimum_orbit_intersection_au": safe_float(orbit.get("moid")),
        "observations_used": orbit.get("n_obs_used"),
        "data_arc_days": orbit.get("data_arc"),
    }


@mcp.tool()
async def sbdb_lookup(designation: str) -> dict:
    """Orbit and physical parameters for an asteroid or comet from JPL SBDB.

    Args:
        designation: Name, designation or SPK-ID, e.g. "Apophis", "99942", "2024 YR4".
    """
    target = (designation or "").strip()
    if not target:
        return err("designation is required", source=SOURCE, url=SBDB_API)
    try:
        payload = await fetch_json(
            SBDB_API,
            params={"sstr": target, "phys-par": "true", "full-prec": "false"},
            cache_ttl=3600.0,
        )
    except Exception as exc:  # noqa: BLE001
        return err(str(exc), source=SOURCE, url=SBDB_API,
                   hint="try a fuller designation, e.g. '99942 Apophis'")

    obj = payload.get("object") or {}
    if not obj:
        return err(f"no small body matched '{target}'", source=SOURCE, url=SBDB_API)

    physical = _physical_parameters(payload)
    diameter_km = physical.get("diameter")
    diameter_m = (diameter_km * 1000.0) if diameter_km is not None else None
    return ok(
        {
            # `name` and `is_potentially_hazardous` deliberately mirror the NeoWs
            # field names. Identical subjects and predicates across the two
            # independent catalogues are what allow the consensus layer to compare
            # them; without the alignment every claim would look single-source.
            "name": obj.get("fullname"),
            "is_potentially_hazardous": bool(obj.get("pha")),
            "designation": obj.get("des"),
            "full_name": obj.get("fullname"),
            "kind": obj.get("kind"),
            "neo": bool(obj.get("neo")),
            "potentially_hazardous": bool(obj.get("pha")),
            "absolute_magnitude_h": physical.get("H"),
            "diameter_km": diameter_km,
            "diameter_m": diameter_m,
            "albedo": physical.get("albedo"),
            "rotation_period_h": physical.get("rot_per"),
            "density_g_cm3": physical.get("density"),
            **_orbital_elements(payload),
            "queried_at": utc_now_iso(),
        },
        source=SOURCE,
        url=f"{SBDB_API}?sstr={target}",
    )


@mcp.tool()
async def close_approaches(
    days: int = 7,
    max_distance_lunar: float = 10.0,
    limit: int = 20,
) -> dict:
    """Asteroids passing near Earth in the next N days, from the JPL CAD API.

    Args:
        days: Look-ahead window in days (1..365).
        max_distance_lunar: Maximum miss distance in lunar distances (0.1..50).
        limit: Maximum rows to return (1..100).
    """
    window = clamp(int(days), 1, 365)
    rows_wanted = clamp(int(limit), 1, 100)
    max_au = max(0.1, min(50.0, float(max_distance_lunar))) * LD_KM / AU_KM

    start = datetime.now(timezone.utc).date()
    end = start + timedelta(days=window)
    try:
        payload = await fetch_json(
            CAD_API,
            params={
                "date-min": start.isoformat(),
                "date-max": end.isoformat(),
                "dist-max": f"{max_au:.6f}",
                "sort": "date",
                "limit": rows_wanted,
            },
            cache_ttl=900.0,
        )
    except Exception as exc:  # noqa: BLE001
        return err(str(exc), source=SOURCE, url=CAD_API)

    fields = payload.get("fields") or []
    index = {name: i for i, name in enumerate(fields)}
    approaches = []
    for row in payload.get("data") or []:
        def field(name: str) -> Any:
            position = index.get(name)
            return row[position] if position is not None and position < len(row) else None

        distance_au = safe_float(field("dist"))
        diameter_min, diameter_max = _diameter_from_h(safe_float(field("h")))
        approaches.append(
            {
                "designation": field("des"),
                "close_approach_time": field("cd"),
                "miss_distance_au": distance_au,
                "miss_distance_km": (distance_au * AU_KM) if distance_au is not None else None,
                "miss_distance_lunar": (
                    (distance_au * AU_KM / LD_KM) if distance_au is not None else None
                ),
                "relative_velocity_km_s": safe_float(field("v_rel")),
                "absolute_magnitude_h": safe_float(field("h")),
                "estimated_diameter_min_m": diameter_min,
                "estimated_diameter_max_m": diameter_max,
            }
        )

    closest = min(
        (a for a in approaches if a["miss_distance_km"] is not None),
        key=lambda a: a["miss_distance_km"],
        default=None,
    )
    return ok(
        {
            "window_days": window,
            "count": len(approaches),
            "max_distance_lunar": max_distance_lunar,
            "closest_approach": closest,
            "approaches": approaches,
        },
        source=SOURCE,
        url=CAD_API,
    )


def _diameter_from_h(h: float | None, albedo: float = 0.14) -> tuple[float | None, float | None]:
    """Standard H-to-diameter relation, bracketed by a plausible albedo range."""
    if h is None:
        return None, None
    def diameter(a: float) -> float:
        return (1329.0 / (a**0.5)) * (10 ** (-0.2 * h)) * 1000.0
    return round(diameter(0.25), 1), round(diameter(0.05), 1)


async def _resolve_designation(target: str) -> str | None:
    """Sentry rejects common names, so translate one into a real designation first."""
    if not target or target.replace(" ", "").isdigit():
        return target or None
    try:
        payload = await fetch_json(
            SBDB_API, params={"sstr": target}, cache_ttl=86400.0
        )
    except Exception:  # noqa: BLE001
        return None
    return ((payload.get("object") or {}).get("des")) or None


@mcp.tool()
async def sentry_risk(designation: str = "") -> dict:
    """Impact-risk assessment from JPL's Sentry system.

    With no designation, returns the current highest-risk objects being monitored.

    Args:
        designation: Optional object designation or name, e.g. "99942" or "Bennu".
    """
    target = (designation or "").strip()
    resolved = await _resolve_designation(target) if target else None
    if target and resolved is None:
        return err(f"could not resolve '{target}' to a small-body designation",
                   source=SOURCE, url=SBDB_API)

    params = {"des": resolved} if resolved else {"all": "1"}
    try:
        payload = await fetch_json(SENTRY_API, params=params, cache_ttl=3600.0)
    except Exception as exc:  # noqa: BLE001
        return err(str(exc), source=SOURCE, url=SENTRY_API,
                   hint="Sentry only tracks objects with a non-zero impact probability")

    summary = payload.get("summary") or {}
    entries = payload.get("data") or []

    if target:
        # An object that was *removed* from the risk list is a meaningful, positive
        # result rather than a lookup failure. Apophis is the canonical example: it
        # was removed in 2021 once a 100-year impact was ruled out.
        if payload.get("removed"):
            return ok(
                {
                    "designation": resolved or target,
                    "monitored_by_sentry": False,
                    "removed_from_risk_list": True,
                    "interpretation": (
                        f"{target} was removed from the Sentry risk list, meaning "
                        "further observations ruled out the potential impacts that "
                        "originally placed it there."
                    ),
                },
                source=SOURCE,
                url=SENTRY_API,
            )
        if not entries and not summary:
            return ok(
                {
                    "designation": target,
                    "monitored_by_sentry": False,
                    "interpretation": (
                        f"{target} is not on the Sentry risk list, meaning no potential "
                        "impacts were found over the analysed window."
                    ),
                },
                source=SOURCE,
                url=SENTRY_API,
            )
        cumulative = safe_float(summary.get("ps_cum"))
        return ok(
            {
                "designation": summary.get("des") or target,
                "monitored_by_sentry": True,
                "full_name": summary.get("fullname"),
                "impact_probability_cumulative": safe_float(summary.get("ip")),
                "palermo_scale_cumulative": cumulative,
                "torino_scale_max": safe_float(summary.get("ts_max")),
                "potential_impacts": len(entries),
                "year_range": summary.get("year_range"),
                "diameter_km": safe_float(summary.get("diameter")),
                "velocity_km_s": safe_float(summary.get("v_inf")),
                "queried_at": utc_now_iso(),
            },
            source=SOURCE,
            url=SENTRY_API,
        )

    ranked = []
    for entry in entries[:25]:
        ranked.append(
            {
                "designation": entry.get("des"),
                "full_name": entry.get("fullname"),
                "impact_probability_cumulative": safe_float(entry.get("ip")),
                "palermo_scale_cumulative": safe_float(entry.get("ps")),
                "torino_scale_max": safe_float(entry.get("ts")),
                "diameter_km": safe_float(entry.get("diameter")),
                "year_range": entry.get("range"),
            }
        )
    return ok(
        {
            "monitored_objects": len(entries),
            "showing": len(ranked),
            "objects": ranked,
            "note": (
                "Sentry lists objects with a non-zero computed impact probability. "
                "Nearly all resolve to zero as more observations arrive."
            ),
        },
        source=SOURCE,
        url=SENTRY_API,
    )


@mcp.tool()
async def sbdb_compare(designation: str) -> dict:
    """Key cross-checkable numbers for one body, shaped for source comparison.

    Returns the same quantities ``nasa_neo`` reports, so the consensus layer can
    hold two independent readings of each against one another.

    Args:
        designation: Object name or designation, e.g. "Apophis".
    """
    looked_up = await sbdb_lookup(designation)
    if not looked_up.get("ok"):
        return looked_up

    data = looked_up["data"]
    return ok(
        {
            "designation": data.get("designation"),
            "comparable_facts": {
                "diameter_m": data.get("diameter_m"),
                "absolute_magnitude_h": data.get("absolute_magnitude_h"),
                "potentially_hazardous": data.get("potentially_hazardous"),
                "orbital_period_days": data.get("orbital_period_days"),
                "minimum_orbit_intersection_au": data.get("minimum_orbit_intersection_au"),
            },
            "independent_of": "NASA NeoWs",
        },
        source=SOURCE,
        url=f"{SBDB_API}?sstr={designation}",
    )


if __name__ == "__main__":
    mcp.run()
