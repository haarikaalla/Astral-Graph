"""Intent Parser — the entry agent. Uses no tools (scope: none)."""

from __future__ import annotations

import re

from core.telemetry import get_logger, log_event, span
from guardrails.schemas import Intent
from agents import prompts
from agents.state import RunContext

log = get_logger("agents.intent")

KEYWORDS: dict[str, tuple[str, ...]] = {
    "neo": (
        "asteroid", "asteroids", "neo", "near-earth", "near earth", "comet", "meteor",
        "impact", "hazardous", "close approach", "apophis", "bennu", "didymos",
        "impactor", "collision", "torino", "pha",
    ),
    "exoplanet": (
        "exoplanet", "exoplanets", "planet", "planets", "trappist", "kepler-", "proxima",
        "host star", "habitable", "transit", "radial velocity", "tess", "super-earth",
        "hot jupiter", "k2-", "toi-", "gliese", "wasp-",
    ),
    "events": (
        "iss", "space station", "astronaut", "astronauts", "in space", "crew",
        "wildfire", "wildfires", "volcano", "volcanoes", "storm", "storms", "eonet",
        "hurricane", "flood", "floods", "earthquake", "drought", "natural event",
        "overhead", "orbiting earth", "people in space",
    ),
    "literature": (
        "what is", "explain", "how does", "why", "definition", "define", "paper",
        "papers", "research", "study", "studies", "arxiv", "method", "history",
        "compare", "difference between", "concept", "theory",
    ),
}

COMPUTE_KEYWORDS = (
    "energy", "megaton", "how far", "how fast", "how long", "convert", "calculate",
    "temperature", "period", "distance", "velocity", "mass", "density", "gravity",
    "escape velocity", "light-year", "light year", "compare size", "bigger", "brighter",
)

ENTITY_RE = re.compile(
    r"\b(?:[A-Z][A-Za-z]+-\d+[A-Za-z]?(?:\s+[a-h])?|TRAPPIST-1[a-h]?|\d{5,7}\b|"
    r"[A-Z]{2,}-\d+|Apophis|Bennu|Ryugu|Didymos|Dimorphos|Ceres|Vesta|Eros)\b"
)

RISK_WORDS = ("hit earth", "will it hit", "impact earth", "danger", "dangerous",
              "hazardous", "risk", "threat", "collide", "destroy")


class IntentParserAgent:
    name = "intent_parser"
    role = "Intent Parser"

    def __init__(self, ctx: RunContext) -> None:
        self.ctx = ctx

    def heuristic(self, question: str) -> Intent:
        lowered = question.lower()
        domains = [d for d, words in KEYWORDS.items() if any(w in lowered for w in words)]
        if "literature" in domains and len(domains) > 1:
            # keep literature only when it is genuinely conceptual
            if not any(w in lowered for w in ("explain", "what is", "why", "how does", "define")):
                domains.remove("literature")
        if not domains:
            domains = ["literature"]
        # Risk and safety questions need the reference corpus as well as live data:
        # public catalogues give orbits, not the "will it hit us" context, and the
        # NASA APIs can be rate-limited on the shared DEMO_KEY.
        if any(w in lowered for w in RISK_WORDS) and "literature" not in domains:
            domains.append("literature")
        entities = sorted(set(ENTITY_RE.findall(question)))
        return Intent(
            normalized_question=question.strip()[:600],
            domains=domains,  # type: ignore[arg-type]
            entities=entities[:12],
            needs_computation=any(w in lowered for w in COMPUTE_KEYWORDS),
            needs_literature="literature" in domains,
            confidence=0.5,
            reasoning="keyword heuristic (no LLM configured or LLM call failed)",
        )

    async def run(self, question: str) -> Intent:
        with span(self.ctx.trace, self.name, "agent", agent=self.name) as sp:
            if not self.ctx.llm_available:
                intent = self.heuristic(question)
            else:
                try:
                    intent = await self.ctx.llm.structured(
                        system=prompts.INTENT_PARSER,
                        user=f"User question: {question}",
                        schema=Intent,
                        trace=self.ctx.trace,
                        name="intent_parser",
                    )
                except Exception as exc:  # noqa: BLE001
                    log_event(log, "intent_llm_failed", error=str(exc)[:200])
                    intent = self.heuristic(question)
                    sp["_ok"] = False

            # Union with the heuristic so a confident model can never blind the router.
            heuristic = self.heuristic(question)
            merged = list(dict.fromkeys([*intent.domains, *heuristic.domains]))
            if "general" in merged and len(merged) > 1:
                merged.remove("general")
            intent.domains = merged[:5]  # type: ignore[assignment]
            intent.needs_computation = intent.needs_computation or heuristic.needs_computation
            intent.entities = list(dict.fromkeys([*intent.entities, *heuristic.entities]))[:12]

            sp["domains"] = intent.domains
            sp["entities"] = intent.entities
        return intent
