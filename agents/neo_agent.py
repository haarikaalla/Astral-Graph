"""NEO Agent — near-Earth objects. MCP scope: ``nasa_neo``, ``astro_compute``."""

from __future__ import annotations

import re
from datetime import date, timedelta
from typing import Any

from guardrails.schemas import Intent, ToolCall
from agents.base import BaseAgent

ID_RE = re.compile(r"\b\d{5,7}\b")
NAMED = {
    "apophis": "2099942",
    "bennu": "2101955",
    "didymos": "2065803",
    "eros": "2000433",
    "ryugu": "2162173",
    "itokawa": "2025143",
}

_NUM = r"(\d[\d,]*(?:\.\d+)?)"
DIAMETER_M_RE = re.compile(_NUM + r"\s*(?:m\b|met(?:re|er)s?\b)", re.I)
DIAMETER_KM_RE = re.compile(_NUM + r"\s*(?:km\b|kilomet(?:re|er)s?\b)\s*(?:wide|across|diameter)", re.I)
VELOCITY_RE = re.compile(_NUM + r"\s*km\s*(?:/|\s+per\s+)\s*s(?:ec(?:ond)?)?\b", re.I)
DENSITY_RE = re.compile(_NUM + r"\s*kg\s*(?:/|\s+per\s+)\s*m\s*\^?3\b", re.I)


def _as_float(text: str) -> float:
    return float(text.replace(",", ""))


def explicit_impact_params(question: str) -> dict[str, float] | None:
    """Pull user-supplied impactor parameters out of the question text.

    When the user states the numbers themselves ("a 100 metre asteroid at 20 km/s"),
    those values must drive the deterministic calculator — never a value borrowed
    from an unrelated catalogue object.
    """
    velocity = VELOCITY_RE.search(question)
    diameter_km = DIAMETER_KM_RE.search(question)
    diameter_m = DIAMETER_M_RE.search(question)
    if not velocity or not (diameter_km or diameter_m):
        return None

    params: dict[str, float] = {
        "diameter_m": _as_float(diameter_km.group(1)) * 1000.0 if diameter_km
        else _as_float(diameter_m.group(1)),
        "velocity_km_s": _as_float(velocity.group(1)),
    }
    density = DENSITY_RE.search(question)
    params["density_kg_m3"] = _as_float(density.group(1)) if density else 3000.0
    if not (0.1 <= params["diameter_m"] <= 1e6 and 0.1 <= params["velocity_km_s"] <= 300):
        return None
    return params


class NEOAgent(BaseAgent):
    name = "neo_agent"
    role = "NEO Agent (near-Earth objects, close approaches, impact risk)"
    max_calls = 5

    def heuristic_plan(self, intent: Intent) -> list[ToolCall]:
        question = f"{self.ctx.question} {intent.normalized_question}".lower()
        calls: list[ToolCall] = []

        # A self-contained hypothetical impactor is answered by the calculator alone.
        params = explicit_impact_params(self.ctx.question)
        if params is not None:
            return [
                ToolCall(server="astro_compute", tool="impact_energy", arguments=params,
                         rationale="impactor parameters stated in the question")
            ]

        named_targets: list[str] = []
        for entity in intent.entities:
            key = entity.lower()
            if key in NAMED:
                named_targets.append(entity)
                calls.append(
                    ToolCall(server="nasa_neo", tool="neo_lookup",
                             arguments={"asteroid_id": NAMED[key]},
                             rationale=f"named object {entity}")
                )
        for token in ID_RE.findall(self.ctx.question):
            calls.append(
                ToolCall(server="nasa_neo", tool="neo_lookup",
                         arguments={"asteroid_id": token}, rationale="explicit SPK-ID")
            )
        for name, spk in NAMED.items():
            if name in question and not any(c.arguments.get("asteroid_id") == spk for c in calls):
                named_targets.append(name)
                calls.append(
                    ToolCall(server="nasa_neo", tool="neo_lookup",
                             arguments={"asteroid_id": spk}, rationale=f"mentioned {name}")
                )

        # Cross-check the named body against JPL, a different upstream from NeoWs.
        # Two independent readings let the consensus layer corroborate or dispute.
        if named_targets:
            calls.append(
                ToolCall(server="jpl_sbdb", tool="sbdb_lookup",
                         arguments={"designation": named_targets[0]},
                         rationale="independent second source for cross-checking")
            )
            if any(w in question for w in ("hit", "impact", "risk", "danger", "collide",
                                           "threat", "hazard")):
                calls.append(
                    ToolCall(server="jpl_sbdb", tool="sentry_risk",
                             arguments={"designation": named_targets[0]},
                             rationale="JPL Sentry impact-probability assessment")
                )

        if any(word in question for word in ("how many known", "total", "catalogue", "catalog",
                                             "statistics", "how many neo")):
            calls.append(ToolCall(server="nasa_neo", tool="neo_stats",
                                  rationale="catalogue-level statistics"))

        if not calls or any(w in question for w in ("today", "this week", "upcoming", "next",
                                                    "closest", "approach", "hazardous")):
            start = date.today()
            days = 7 if any(w in question for w in ("week", "7 days", "upcoming", "next")) else 0
            calls.append(
                ToolCall(
                    server="nasa_neo",
                    tool="neo_feed",
                    arguments={
                        "start_date": start.isoformat(),
                        "end_date": (start + timedelta(days=days)).isoformat(),
                        "hazardous_only": "hazardous" in question,
                    },
                    rationale="close-approach window",
                )
            )
            # JPL's CAD API answers the same question without an API key, so the
            # feed still resolves when NeoWs is rate-limited.
            calls.append(
                ToolCall(server="jpl_sbdb", tool="close_approaches",
                         arguments={"days": max(days, 1), "max_distance_lunar": 10.0,
                                    "limit": 20},
                         rationale="keyless independent close-approach feed")
            )
        return calls[: self.max_calls]

    def followups(self, intent: Intent) -> list[ToolCall]:
        """Chain the deterministic calculator onto whatever the data server returned."""
        question = f"{self.ctx.question} {intent.normalized_question}".lower()
        calls: list[ToolCall] = []

        # Torino band needs an energy, which only exists after impact_energy ran.
        if "torino" in question:
            megatons = self._computed_megatons()
            if megatons is not None:
                calls.append(
                    ToolCall(server="astro_compute", tool="torino_scale_band",
                             arguments={"energy_megatons": round(megatons, 3),
                                        "impact_probability": 0.0},
                             rationale="Torino band for the computed impact energy")
                )

        wants_energy = any(
            w in question for w in ("energy", "megaton", "impact", "damage", "destructive",
                                    "how powerful", "torino")
        )
        if not wants_energy or explicit_impact_params(self.ctx.question) is not None:
            return calls

        record = self._largest_object()
        if record is None:
            return calls
        diameter = record.get("diameter_max_m")
        velocity = (record.get("close_approach") or {}).get("relative_velocity_km_s")
        if not diameter or not velocity:
            return calls
        calls.append(
            ToolCall(
                server="astro_compute",
                tool="impact_energy",
                arguments={
                    "diameter_m": round(float(diameter), 2),
                    "velocity_km_s": round(float(velocity), 3),
                    "density_kg_m3": 3000.0,
                },
                rationale=f"deterministic impact energy for {record.get('name')}",
            )
        )
        return calls

    def _computed_megatons(self) -> float | None:
        for result in self.results:
            if not (result.ok and result.tool == "impact_energy"):
                continue
            payload = (result.data or {}).get("data") if isinstance(result.data, dict) else None
            if isinstance(payload, dict) and payload.get("energy_megatons_tnt") is not None:
                return float(payload["energy_megatons_tnt"])
        return None

    def _largest_object(self) -> dict[str, Any] | None:
        """Pick the most relevant object from whatever the NEO server returned."""
        candidates: list[dict[str, Any]] = []
        for result in self.results:
            if not result.ok or result.server != "nasa_neo":
                continue
            data = (result.data or {}).get("data") if isinstance(result.data, dict) else None
            if not isinstance(data, dict):
                continue
            if "by_date" in data:
                for rows in data["by_date"].values():
                    candidates.extend(rows)
            elif "objects" in data:
                candidates.extend(data["objects"])
            elif "name" in data:
                candidates.append(data)
        if not candidates:
            return None
        return max(candidates, key=lambda r: r.get("diameter_max_m") or 0.0)
