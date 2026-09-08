"""ISS / Earth-Events Agent.

MCP scope: ``iss``, ``eonet``, ``astro_compute``.
"""

from __future__ import annotations

import re

from guardrails.schemas import Intent, ToolCall
from agents.base import BaseAgent

COORD_RE = re.compile(r"(-?\d{1,2}(?:\.\d+)?)\s*[,°]?\s*[NnSs]?\s*[,/]\s*(-?\d{1,3}(?:\.\d+)?)")

CITY_COORDS = {
    "london": (51.5074, -0.1278),
    "new york": (40.7128, -74.0060),
    "tokyo": (35.6762, 139.6503),
    "delhi": (28.6139, 77.2090),
    "new delhi": (28.6139, 77.2090),
    "mumbai": (19.0760, 72.8777),
    "bengaluru": (12.9716, 77.5946),
    "bangalore": (12.9716, 77.5946),
    "hyderabad": (17.3850, 78.4867),
    "chennai": (13.0827, 80.2707),
    "sydney": (-33.8688, 151.2093),
    "paris": (48.8566, 2.3522),
    "berlin": (52.5200, 13.4050),
    "san francisco": (37.7749, -122.4194),
    "los angeles": (34.0522, -118.2437),
    "cape town": (-33.9249, 18.4241),
    "sao paulo": (-23.5505, -46.6333),
    "moscow": (55.7558, 37.6173),
    "singapore": (1.3521, 103.8198),
    "dubai": (25.2048, 55.2708),
}

CATEGORY_WORDS = {
    "wildfire": "wildfires", "wildfires": "wildfires", "fire": "wildfires",
    "volcano": "volcanoes", "volcanoes": "volcanoes", "eruption": "volcanoes",
    "storm": "severeStorms", "storms": "severeStorms", "hurricane": "severeStorms",
    "cyclone": "severeStorms", "typhoon": "severeStorms",
    "flood": "floods", "floods": "floods", "flooding": "floods",
    "iceberg": "seaLakeIce", "sea ice": "seaLakeIce", "ice": "seaLakeIce",
    "drought": "drought", "landslide": "landslides", "dust": "dustHaze",
    "haze": "dustHaze", "snow": "snow", "earthquake": "earthquakes",
}


class EventsAgent(BaseAgent):
    name = "events_agent"
    role = "ISS, Earth events, space weather & launch agent"
    max_calls = 4

    def heuristic_plan(self, intent: Intent) -> list[ToolCall]:
        question = f"{self.ctx.question} {intent.normalized_question}".lower()
        calls: list[ToolCall] = []

        wants_iss = any(w in question for w in ("iss", "space station", "astronaut", "crew",
                                                "in space", "overhead", "orbiting"))
        wants_events = any(w in question for w in CATEGORY_WORDS) or any(
            w in question for w in ("eonet", "natural event", "disaster", "event")
        )
        wants_weather = any(w in question for w in (
            "solar flare", "flare", "cme", "coronal mass", "geomagnetic", "solar storm",
            "space weather", "aurora", "kp index", "sunspot", "solar activity",
        ))
        wants_launch = any(w in question for w in (
            "launch", "rocket", "liftoff", "lift-off", "falcon", "starship", "ariane",
            "soyuz", "next flight",
        ))

        if wants_weather:
            if any(w in question for w in ("now", "current", "today", "right now", "aurora")):
                calls.append(ToolCall(server="space_weather", tool="space_weather_now",
                                      rationale="live planetary K-index, no API key needed"))
            if any(w in question for w in ("flare", "solar activity", "sunspot")):
                calls.append(ToolCall(server="space_weather", tool="solar_flares",
                                      arguments={"days": 7}, rationale="recent solar flares"))
            if any(w in question for w in ("cme", "coronal mass")):
                calls.append(ToolCall(server="space_weather", tool="coronal_mass_ejections",
                                      arguments={"days": 7}, rationale="recent CMEs"))
            if any(w in question for w in ("geomagnetic", "storm", "aurora")):
                calls.append(ToolCall(server="space_weather", tool="geomagnetic_storms",
                                      arguments={"days": 30},
                                      rationale="recent geomagnetic storms"))

        if wants_launch:
            if any(w in question for w in ("next", "upcoming", "scheduled", "when is")):
                calls.append(ToolCall(server="launch", tool="upcoming_launches",
                                      arguments={"limit": 10}, rationale="next launches"))
            elif any(w in question for w in ("how many", "rate", "statistics", "busiest",
                                             "success")):
                calls.append(ToolCall(server="launch", tool="launch_stats",
                                      arguments={"sample": 40}, rationale="launch activity"))
            else:
                calls.append(ToolCall(server="launch", tool="recent_launches",
                                      arguments={"limit": 10}, rationale="recent launches"))

        if wants_iss:
            if any(w in question for w in ("who", "crew", "astronaut", "people in space",
                                           "how many people")):
                calls.append(ToolCall(server="iss", tool="iss_crew",
                                      rationale="people currently in space"))
            coords = self._coords(question)
            if coords:
                calls.append(
                    ToolCall(server="iss", tool="iss_ground_distance",
                             arguments={"latitude": coords[0], "longitude": coords[1]},
                             rationale="distance from ISS to the named location")
                )
            elif not calls or "where" in question or "position" in question or "now" in question:
                calls.append(ToolCall(server="iss", tool="iss_now",
                                      rationale="current ISS position"))

        if wants_events or not calls:
            category = next(
                (cat for word, cat in CATEGORY_WORDS.items() if word in question), ""
            )
            coords = self._coords(question)
            if coords and category == "":
                calls.append(
                    ToolCall(server="eonet", tool="eonet_nearby_events",
                             arguments={"latitude": coords[0], "longitude": coords[1],
                                        "radius_km": 1500.0, "days": 60, "limit": 15},
                             rationale="events near the named location")
                )
            elif category:
                calls.append(
                    ToolCall(server="eonet", tool="eonet_events",
                             arguments={"status": "open", "category": category, "days": 60,
                                        "limit": 30},
                             rationale=f"open {category} events")
                )
            else:
                calls.append(
                    ToolCall(server="eonet", tool="eonet_summary", arguments={"days": 30},
                             rationale="summary of currently open natural events")
                )
        return calls[: self.max_calls]

    def _coords(self, question: str) -> tuple[float, float] | None:
        for city, coords in CITY_COORDS.items():
            if city in question:
                return coords
        match = COORD_RE.search(question)
        if match:
            try:
                lat, lon = float(match.group(1)), float(match.group(2))
                if -90 <= lat <= 90 and -180 <= lon <= 180:
                    return lat, lon
            except ValueError:
                return None
        return None
