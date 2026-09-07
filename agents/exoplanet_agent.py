"""Exoplanet Agent — confirmed planets and host stars.

MCP scope: ``exoplanet``, ``astro_compute``.
"""

from __future__ import annotations

import re
from typing import Any

from guardrails.schemas import Intent, ToolCall
from agents.base import BaseAgent

PLANET_RE = re.compile(r"\b([A-Z][A-Za-z0-9]*(?:-\d+)?)\s+([b-j])\b")
HOST_RE = re.compile(r"\b(TRAPPIST-1|Proxima\s+Centauri|Kepler-\d+|K2-\d+|TOI-\d+|"
                     r"HD\s?\d+|GJ\s?\d+|WASP-\d+|HAT-P-\d+|LHS\s?\d+|TRAPPIST-\d+)\b", re.I)

METHODS = {
    "transit": "Transit",
    "radial velocity": "Radial Velocity",
    "microlensing": "Microlensing",
    "imaging": "Imaging",
    "direct imaging": "Imaging",
    "astrometry": "Astrometry",
    "timing": "Transit Timing Variations",
}

_NUM = r"(\d+(?:\.\d+)?)"
DISTANCE_PC_RE = re.compile(
    r"(?:within|closer than|less than|under|inside)\s+" + _NUM + r"\s*(?:pc\b|parsecs?\b)", re.I
)
DISTANCE_LY_RE = re.compile(
    r"(?:within|closer than|less than|under|inside)\s+" + _NUM + r"\s*(?:ly\b|light[- ]years?\b)",
    re.I,
)
MAX_RADIUS_RE = re.compile(
    r"(?:smaller than|less than|under|below|up to)\s+" + _NUM + r"\s*(?:earth\s+)?radii", re.I
)
MIN_RADIUS_RE = re.compile(
    r"(?:larger than|bigger than|greater than|more than|above|at least)\s+" + _NUM
    + r"\s*(?:earth\s+)?radii", re.I
)

LY_PER_PC = 3.26156


def explicit_filters(question: str) -> dict[str, float]:
    """Turn stated numeric constraints into NASA Exoplanet Archive query filters."""
    filters: dict[str, float] = {}

    distance_pc = DISTANCE_PC_RE.search(question)
    distance_ly = DISTANCE_LY_RE.search(question)
    if distance_pc:
        filters["max_distance_pc"] = float(distance_pc.group(1))
    elif distance_ly:
        filters["max_distance_pc"] = round(float(distance_ly.group(1)) / LY_PER_PC, 3)

    max_radius = MAX_RADIUS_RE.search(question)
    if max_radius:
        filters["max_radius_earth"] = float(max_radius.group(1))
    min_radius = MIN_RADIUS_RE.search(question)
    if min_radius:
        filters["min_radius_earth"] = float(min_radius.group(1))
    return filters


class ExoplanetAgent(BaseAgent):
    name = "exoplanet_agent"
    role = "Exoplanet Agent (NASA Exoplanet Archive)"
    max_calls = 4

    def heuristic_plan(self, intent: Intent) -> list[ToolCall]:
        question = f"{self.ctx.question} {intent.normalized_question}"
        lowered = question.lower()
        calls: list[ToolCall] = []

        # Explicit numeric filters ("within 10 parsecs", "smaller than 2 Earth radii")
        # must become an actual catalogue query, not a generic search.
        filters = explicit_filters(question)
        if filters:
            calls.append(
                ToolCall(server="exoplanet", tool="exoplanet_search",
                         arguments={**filters, "limit": 25, "order_by": "sy_dist"},
                         rationale="numeric filters stated in the question")
            )

        for match in PLANET_RE.finditer(question):
            calls.append(
                ToolCall(server="exoplanet", tool="exoplanet_by_name",
                         arguments={"planet_name": f"{match.group(1)} {match.group(2)}"},
                         rationale="named planet")
            )

        hosts = {m.group(0).strip() for m in HOST_RE.finditer(question)}
        for host in list(hosts)[:2]:
            calls.append(
                ToolCall(server="exoplanet", tool="exoplanet_search",
                         arguments={"hostname": _normalise_host(host), "limit": 25,
                                    "order_by": "pl_orbper"},
                         rationale=f"all planets of {host}")
            )

        if any(w in lowered for w in ("how many", "total", "count", "number of confirmed")):
            group_by = ""
            if "method" in lowered or "discovered by" in lowered:
                group_by = "discoverymethod"
            elif "year" in lowered or "per year" in lowered:
                group_by = "disc_year"
            elif "telescope" in lowered or "facility" in lowered or "mission" in lowered:
                group_by = "disc_facility"
            calls.append(
                ToolCall(server="exoplanet", tool="exoplanet_counts",
                         arguments={"group_by": group_by}, rationale="catalogue counts")
            )

        if any(w in lowered for w in ("habitable", "life", "earth-like", "earthlike", "goldilocks")):
            calls.append(
                ToolCall(server="exoplanet", tool="exoplanet_habitable_candidates",
                         arguments={"limit": 15, "max_distance_pc": 100.0},
                         rationale="habitable-zone candidates")
            )

        if not calls:
            arguments: dict[str, Any] = {"limit": 20, "order_by": "sy_dist"}
            for phrase, method in METHODS.items():
                if phrase in lowered:
                    arguments["discovery_method"] = method
                    break
            if "nearest" in lowered or "closest" in lowered:
                arguments["max_distance_pc"] = 15.0
            if "smallest" in lowered or "rocky" in lowered or "earth-sized" in lowered:
                arguments["max_radius_earth"] = 1.8
            calls.append(
                ToolCall(server="exoplanet", tool="exoplanet_search", arguments=arguments,
                         rationale="general catalogue search")
            )
        return calls[: self.max_calls]

    def followups(self, intent: Intent) -> list[ToolCall]:
        lowered = f"{self.ctx.question} {intent.normalized_question}".lower()
        planet = self._first_planet()
        if planet is None:
            return []

        calls: list[ToolCall] = []
        if any(w in lowered for w in ("temperature", "hot", "cold", "habitable", "insolation")):
            teff, radius, axis = (
                planet.get("st_teff"), planet.get("st_rad"), planet.get("pl_orbsmax")
            )
            if teff and radius and axis:
                calls.append(
                    ToolCall(server="astro_compute", tool="equilibrium_temperature",
                             arguments={"star_teff_k": float(teff),
                                        "star_radius_solar": float(radius),
                                        "semi_major_axis_au": float(axis), "albedo": 0.3},
                             rationale=f"equilibrium temperature of {planet.get('pl_name')}")
                )
        if any(w in lowered for w in ("density", "gravity", "rocky", "composition", "escape")):
            mass, radius = planet.get("pl_bmasse"), planet.get("pl_rade")
            if mass and radius:
                calls.append(
                    ToolCall(server="astro_compute", tool="planet_bulk_properties",
                             arguments={"mass_earth": float(mass), "radius_earth": float(radius)},
                             rationale=f"bulk properties of {planet.get('pl_name')}")
                )
        if any(w in lowered for w in ("light-year", "light year", "how far", "distance")):
            distance = planet.get("sy_dist")
            if distance:
                calls.append(
                    ToolCall(server="astro_compute", tool="convert_distance",
                             arguments={"value": float(distance), "from_unit": "pc",
                                        "to_unit": "ly"},
                             rationale="parsec to light-year conversion")
                )
        if any(w in lowered for w in ("transit depth", "how deep", "detectab")):
            radius, star = planet.get("pl_rade"), planet.get("st_rad")
            if radius and star:
                calls.append(
                    ToolCall(server="astro_compute", tool="transit_depth",
                             arguments={"planet_radius_earth": float(radius),
                                        "star_radius_solar": float(star)},
                             rationale="transit depth")
                )
        return calls[:3]

    def _first_planet(self) -> dict[str, Any] | None:
        for result in self.results:
            if not result.ok or result.server != "exoplanet":
                continue
            data = (result.data or {}).get("data") if isinstance(result.data, dict) else None
            if isinstance(data, dict):
                if data.get("planets"):
                    return data["planets"][0]
                if data.get("pl_name"):
                    return data
        return None


def _normalise_host(host: str) -> str:
    cleaned = " ".join(host.split())
    if cleaned.lower().startswith("proxima"):
        return "Proxima Cen"
    return cleaned
