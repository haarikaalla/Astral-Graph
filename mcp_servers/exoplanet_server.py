"""MCP server wrapping the **NASA Exoplanet Archive** TAP service.

Security note
-------------
Agents never pass raw ADQL. Every tool builds the query from validated, typed
arguments, and the one advanced escape hatch (``exoplanet_adql``) is restricted to
single-statement read-only ``SELECT``s against a whitelist of tables.

Run standalone:  ``python -m mcp_servers.exoplanet_server``
"""

from __future__ import annotations

import re
from typing import Any

from mcp_servers._common import clamp, err, fetch_json, ok, safe_float
from mcp_servers._compat import MCPServer

TAP = "https://exoplanetarchive.ipac.caltech.edu/TAP/sync"
SOURCE = "NASA Exoplanet Archive (TAP)"
ALLOWED_TABLES = {"pscomppars", "ps", "k2pandc", "cumulative", "toi"}

CORE_COLUMNS = (
    "pl_name,hostname,sy_snum,sy_pnum,discoverymethod,disc_year,disc_facility,"
    "pl_orbper,pl_orbsmax,pl_rade,pl_bmasse,pl_dens,pl_eqt,pl_insol,"
    "st_teff,st_rad,st_mass,st_spectype,sy_dist"
)

NUMERIC_FILTERS = {
    "max_distance_pc": ("sy_dist", "<="),
    "min_distance_pc": ("sy_dist", ">="),
    "max_radius_earth": ("pl_rade", "<="),
    "min_radius_earth": ("pl_rade", ">="),
    "max_mass_earth": ("pl_bmasse", "<="),
    "min_mass_earth": ("pl_bmasse", ">="),
    "max_period_days": ("pl_orbper", "<="),
    "min_period_days": ("pl_orbper", ">="),
    "min_year": ("disc_year", ">="),
    "max_year": ("disc_year", "<="),
    "max_eq_temp_k": ("pl_eqt", "<="),
    "min_eq_temp_k": ("pl_eqt", ">="),
}

mcp = MCPServer("astral-exoplanet")


def _quote(value: str) -> str:
    return "'" + str(value).replace("'", "''") + "'"


async def _run(query: str) -> Any:
    return await fetch_json(TAP, params={"query": query, "format": "json"}, timeout=60.0)


def _clean(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    numeric = {
        "pl_orbper", "pl_orbsmax", "pl_rade", "pl_bmasse", "pl_dens", "pl_eqt",
        "pl_insol", "st_teff", "st_rad", "st_mass", "sy_dist",
    }
    out = []
    for row in rows:
        cleaned = {k: (safe_float(v) if k in numeric else v) for k, v in row.items()}
        out.append(cleaned)
    return out


@mcp.tool()
async def exoplanet_search(
    hostname: str = "",
    discovery_method: str = "",
    discovery_facility: str = "",
    max_distance_pc: float = 0.0,
    min_distance_pc: float = 0.0,
    max_radius_earth: float = 0.0,
    min_radius_earth: float = 0.0,
    max_mass_earth: float = 0.0,
    min_mass_earth: float = 0.0,
    max_period_days: float = 0.0,
    min_period_days: float = 0.0,
    min_year: int = 0,
    max_year: int = 0,
    max_eq_temp_k: float = 0.0,
    min_eq_temp_k: float = 0.0,
    order_by: str = "sy_dist",
    limit: int = 25,
) -> dict:
    """Search confirmed exoplanets (table ``pscomppars``, one row per planet).

    All numeric filters are ignored when left at ``0``. Example: nearest planets
    smaller than 2 Earth radii -> ``max_radius_earth=2, order_by='sy_dist'``.

    Args:
        hostname: Exact host-star name, e.g. ``TRAPPIST-1``.
        discovery_method: e.g. ``Transit``, ``Radial Velocity``, ``Microlensing``.
        discovery_facility: e.g. ``Kepler``, ``Transiting Exoplanet Survey Satellite (TESS)``.
        order_by: Column to sort ascending by. One of the core columns.
        limit: Max rows (1-200).
    """
    where: list[str] = []
    if hostname.strip():
        where.append(f"hostname = {_quote(hostname.strip())}")
    if discovery_method.strip():
        where.append(f"discoverymethod = {_quote(discovery_method.strip())}")
    if discovery_facility.strip():
        where.append(f"disc_facility LIKE {_quote('%' + discovery_facility.strip() + '%')}")

    local = locals()
    for arg, (column, op) in NUMERIC_FILTERS.items():
        value = local.get(arg) or 0
        if value:
            where.append(f"{column} {op} {value}")

    order = order_by if order_by in CORE_COLUMNS.split(",") else "sy_dist"
    clause = (" WHERE " + " AND ".join(where)) if where else ""
    n = clamp(limit, 1, 200)
    query = (
        f"SELECT TOP {n} {CORE_COLUMNS} FROM pscomppars{clause} "
        f"ORDER BY {order} ASC"
    )
    try:
        rows = await _run(query)
    except Exception as exc:  # noqa: BLE001
        return err(str(exc), source=SOURCE, url=TAP, hint=f"query={query}")
    return ok(
        {"count": len(rows), "query": query, "planets": _clean(rows)},
        source=SOURCE,
        url=TAP,
    )


@mcp.tool()
async def exoplanet_by_name(planet_name: str) -> dict:
    """Look up one confirmed exoplanet by its full name, e.g. ``Kepler-186 f``.

    Args:
        planet_name: Planet designation as used by the archive.
    """
    if not planet_name.strip():
        return err("planet_name is required", source=SOURCE)
    query = (
        f"SELECT TOP 5 {CORE_COLUMNS} FROM pscomppars "
        f"WHERE pl_name = {_quote(planet_name.strip())}"
    )
    try:
        rows = await _run(query)
    except Exception as exc:  # noqa: BLE001
        return err(str(exc), source=SOURCE, url=TAP)
    if not rows:
        return err(
            f"no confirmed planet named {planet_name!r}",
            source=SOURCE,
            url=TAP,
            hint="Names use a space before the letter, e.g. 'Kepler-186 f'.",
        )
    return ok(_clean(rows)[0], source=SOURCE, url=TAP)


@mcp.tool()
async def exoplanet_counts(group_by: str = "") -> dict:
    """Total confirmed exoplanet count, optionally grouped.

    Args:
        group_by: ``""`` (total), ``discoverymethod``, ``disc_year`` or ``disc_facility``.
    """
    allowed = {"", "discoverymethod", "disc_year", "disc_facility"}
    if group_by not in allowed:
        return err(f"group_by must be one of {sorted(allowed)}", source=SOURCE)
    if not group_by:
        query = "SELECT COUNT(*) AS n FROM pscomppars"
    else:
        query = (
            f"SELECT {group_by}, COUNT(*) AS n FROM pscomppars "
            f"GROUP BY {group_by} ORDER BY n DESC"
        )
    try:
        rows = await _run(query)
    except Exception as exc:  # noqa: BLE001
        return err(str(exc), source=SOURCE, url=TAP)
    if not group_by:
        return ok({"total_confirmed_planets": int(rows[0]["n"]), "query": query},
                  source=SOURCE, url=TAP)
    return ok({"group_by": group_by, "buckets": rows, "query": query}, source=SOURCE, url=TAP)


@mcp.tool()
async def exoplanet_habitable_candidates(limit: int = 20, max_distance_pc: float = 100.0) -> dict:
    """Rocky planets receiving roughly Earth-like insolation (0.3-1.7 S_Earth).

    This is a *catalogue filter*, not a habitability claim — downstream agents must
    verify the numbers with the deterministic calculators.

    Args:
        limit: Max rows (1-100).
        max_distance_pc: Distance cut in parsecs.
    """
    n = clamp(limit, 1, 100)
    query = (
        f"SELECT TOP {n} {CORE_COLUMNS} FROM pscomppars "
        f"WHERE pl_insol BETWEEN 0.3 AND 1.7 AND pl_rade <= 1.8 "
        f"AND sy_dist <= {max_distance_pc} ORDER BY sy_dist ASC"
    )
    try:
        rows = await _run(query)
    except Exception as exc:  # noqa: BLE001
        return err(str(exc), source=SOURCE, url=TAP)
    return ok(
        {
            "criteria": "0.3 <= pl_insol <= 1.7 and pl_rade <= 1.8",
            "count": len(rows),
            "planets": _clean(rows),
            "query": query,
        },
        source=SOURCE,
        url=TAP,
    )


@mcp.tool()
async def exoplanet_adql(query: str) -> dict:
    """Advanced: run a read-only ADQL ``SELECT`` against the archive.

    Rejected unless it is a single ``SELECT`` statement over a whitelisted table
    (``pscomppars``, ``ps``, ``k2pandc``, ``cumulative``, ``toi``) with no
    semicolons or DDL/DML keywords.

    Args:
        query: The ADQL statement.
    """
    q = " ".join(query.split())
    if not re.match(r"(?i)^select\s", q):
        return err("only SELECT statements are permitted", source=SOURCE)
    if ";" in q:
        return err("multiple statements are not permitted", source=SOURCE)
    if re.search(r"(?i)\b(insert|update|delete|drop|create|alter|grant|revoke|exec)\b", q):
        return err("write/DDL keywords are not permitted", source=SOURCE)
    tables = set(re.findall(r"(?i)\bfrom\s+([a-z0-9_]+)", q)) | set(
        re.findall(r"(?i)\bjoin\s+([a-z0-9_]+)", q)
    )
    unknown = {t.lower() for t in tables} - ALLOWED_TABLES
    if unknown:
        return err(f"table(s) not whitelisted: {sorted(unknown)}", source=SOURCE)
    if not re.search(r"(?i)\btop\s+\d+", q):
        q = re.sub(r"(?i)^select\s", "SELECT TOP 100 ", q, count=1)
    try:
        rows = await _run(q)
    except Exception as exc:  # noqa: BLE001
        return err(str(exc), source=SOURCE, url=TAP, hint=f"query={q}")
    return ok({"count": len(rows), "query": q, "rows": rows}, source=SOURCE, url=TAP)


if __name__ == "__main__":
    mcp.run(transport="stdio")
