"""Turn MCP tool results into provenance-carrying knowledge-graph facts.

This is the bridge that makes grounding possible: nothing an agent says can be
verified unless the underlying tool output was first normalised into
:class:`~graph.schema.Fact` objects here.
"""

from __future__ import annotations

from typing import Any, TYPE_CHECKING

from graph.schema import Edge, Fact, NodeType, RelationType, make_id
from graph.store import KnowledgeGraph

if TYPE_CHECKING:  # pragma: no cover
    from core.mcp_client import ToolResult

SCALAR = (str, int, float, bool)

# Keys that never make useful standalone facts.
SKIP_KEYS = {
    "query", "formula", "filters", "note", "reference", "criteria", "link",
    "sources", "geometry_points", "neo_reference_id", "nasa_jpl_url",
}

UNITS = {
    "_km": "km", "_m": "m", "_au": "AU", "_pc": "pc", "_ly": "ly",
    "_k": "K", "_c": "°C", "_days": "days", "_years": "years", "_seconds": "s",
    "_km_s": "km/s", "_km_h": "km/h", "_deg": "°", "_kg": "kg", "_joules": "J",
    "_ppm": "ppm", "_percent": "%", "_m_s2": "m/s²", "_w_m2": "W/m²",
    "_earth": "M/R_Earth", "_solar": "Solar units", "_megatons_tnt": "Mt TNT",
}


def _unit_for(key: str) -> str | None:
    for suffix, unit in UNITS.items():
        if key.endswith(suffix):
            return unit
    return None


def _provenance(payload: Any, result: "ToolResult", agent: str) -> dict[str, Any]:
    source = payload.get("source", {}) if isinstance(payload, dict) else {}
    return {
        "source_name": source.get("name") or result.server,
        "source_url": source.get("url", ""),
        "mcp_server": result.server,
        "mcp_tool": result.tool,
        "retrieved_at": source.get("retrieved_at", ""),
        "agent": agent,
    }


def _emit(subject: str, mapping: dict[str, Any], prov: dict[str, Any],
          prefix: str = "") -> list[Fact]:
    facts: list[Fact] = []
    for key, value in mapping.items():
        if key in SKIP_KEYS or value is None:
            continue
        predicate = f"{prefix}{key}"
        if isinstance(value, SCALAR):
            facts.append(Fact(subject=subject, predicate=predicate, value=value,
                              unit=_unit_for(key), **prov))
        elif isinstance(value, dict) and len(value) <= 12:
            facts.extend(_emit(subject, value, prov, prefix=f"{predicate}."))
    return facts


# --------------------------------------------------------------------------- #
# Per-server extractors
# --------------------------------------------------------------------------- #


def _neo_facts(data: Any, prov: dict[str, Any], kg: KnowledgeGraph) -> list[Fact]:
    facts: list[Fact] = []

    def one(record: dict[str, Any]) -> None:
        name = record.get("name") or record.get("id")
        if not name:
            return
        kg.add_entity(str(name), NodeType.ASTEROID)
        facts.extend(_emit(str(name), record, prov))

    if isinstance(data, dict):
        if "by_date" in data:
            for day, rows in data["by_date"].items():
                for record in rows:
                    one(record)
            facts.append(Fact(subject=f"NEO feed {data.get('start_date')}..{data.get('end_date')}",
                              predicate="element_count", value=data.get("element_count"), **prov))
            facts.append(Fact(subject=f"NEO feed {data.get('start_date')}..{data.get('end_date')}",
                              predicate="hazardous_count", value=data.get("hazardous_count"),
                              **prov))
        elif "objects" in data:
            for record in data["objects"]:
                one(record)
        elif "name" in data:
            one(data)
        elif "near_earth_object_count" in data or "close_approach_count" in data:
            facts.extend(_emit("NEO catalogue", data, prov))
    return facts


def _exoplanet_facts(data: Any, prov: dict[str, Any], kg: KnowledgeGraph) -> list[Fact]:
    facts: list[Fact] = []

    def one(row: dict[str, Any]) -> None:
        name = row.get("pl_name")
        if not name:
            return
        planet_id = kg.add_entity(str(name), NodeType.PLANET)
        host = row.get("hostname")
        if host:
            star_id = kg.add_entity(str(host), NodeType.STAR)
            kg.add_edge(Edge(source=planet_id, target=star_id, relation=RelationType.ORBITS))
            kg.add_edge(Edge(source=star_id, target=planet_id, relation=RelationType.HOSTS))
        facts.extend(_emit(str(name), row, prov))

    if isinstance(data, dict):
        if "planets" in data:
            rows = data["planets"]
            for row in rows:
                one(row)
            names = [str(r.get("pl_name")) for r in rows if r.get("pl_name")]
            if names:
                # A roll-up fact so answers can name the matches instead of only
                # reciting per-planet numbers.
                facts.append(Fact(subject="Exoplanet Archive query",
                                  predicate="matching_planets",
                                  value=", ".join(names[:12]), **prov))
                facts.append(Fact(subject="Exoplanet Archive query",
                                  predicate="matching_planet_count",
                                  value=len(rows), **prov))
        elif "pl_name" in data:
            one(data)
        elif "total_confirmed_planets" in data:
            facts.append(Fact(subject="NASA Exoplanet Archive", predicate="total_confirmed_planets",
                              value=data["total_confirmed_planets"], **prov))
        elif "buckets" in data:
            for bucket in data["buckets"][:25]:
                key = next((v for k, v in bucket.items() if k != "n"), "?")
                facts.append(Fact(subject=f"Exoplanets by {data.get('group_by')}",
                                  predicate=str(key), value=bucket.get("n"), **prov))
    return facts


def _iss_facts(data: Any, prov: dict[str, Any], kg: KnowledgeGraph) -> list[Fact]:
    if not isinstance(data, dict):
        return []
    kg.add_entity("International Space Station", NodeType.SPACECRAFT)
    facts = _emit("International Space Station",
                  {k: v for k, v in data.items() if isinstance(v, SCALAR)}, prov)
    for craft, people in (data.get("by_craft") or {}).items():
        kg.add_entity(str(craft), NodeType.SPACECRAFT)
        facts.append(Fact(subject=str(craft), predicate="crew_size", value=len(people), **prov))
        facts.append(Fact(subject=str(craft), predicate="crew", value=", ".join(people), **prov))
    if "iss_position" in data:
        facts.extend(_emit("International Space Station", data["iss_position"], prov))
    if "ground_distance_km" in data:
        facts.extend(_emit("ISS ground distance", data, prov))
    return facts


def _eonet_facts(data: Any, prov: dict[str, Any], kg: KnowledgeGraph) -> list[Fact]:
    facts: list[Fact] = []
    if not isinstance(data, dict):
        return facts
    for event in (data.get("events") or [])[:40]:
        title = event.get("title")
        if not title:
            continue
        event_id = kg.add_entity(str(title), NodeType.EARTH_EVENT)
        facts.extend(_emit(str(title), {
            k: v for k, v in event.items() if k not in {"description"}
        }, prov))
        if event.get("latitude") is not None:
            loc = kg.add_entity(
                f"{event['latitude']:.2f},{event['longitude']:.2f}", NodeType.LOCATION
            )
            kg.add_edge(Edge(source=event_id, target=loc, relation=RelationType.LOCATED_AT))
    for key in ("total_open_events", "count", "most_active_category"):
        if key in data:
            facts.append(Fact(subject="EONET open events", predicate=key, value=data[key], **prov))
    for category, count in (data.get("counts_by_category") or {}).items():
        facts.append(Fact(subject="EONET open events", predicate=f"count.{category}",
                          value=count, **prov))
    return facts


def _compute_facts(data: Any, prov: dict[str, Any], kg: KnowledgeGraph, tool: str) -> list[Fact]:
    if not isinstance(data, dict):
        return []
    subject = f"computation:{tool}"
    kg.add_entity(subject, NodeType.COMPUTATION)
    return _emit(subject, data, prov)


def _rag_facts(data: Any, prov: dict[str, Any], kg: KnowledgeGraph) -> list[Fact]:
    facts: list[Fact] = []
    if not isinstance(data, dict):
        return facts
    for doc in (data.get("results") or [])[:20]:
        title = doc.get("title") or doc.get("id") or "document"
        kg.add_document(str(doc.get("id", title)), str(title),
                        url=doc.get("url", ""), score=doc.get("score"),
                        source=doc.get("source", ""))
        facts.append(
            Fact(subject=str(title), predicate="excerpt",
                 value=(doc.get("text") or "")[:400], confidence=float(doc.get("score") or 0.5),
                 **{**prov, "source_name": doc.get("source") or prov["source_name"],
                    "source_url": doc.get("url") or prov["source_url"]})
        )
    return facts


def _web_facts(result: "ToolResult", prov: dict[str, Any], kg: KnowledgeGraph) -> list[Fact]:
    """Brave Search returns free text; keep it as low-confidence citation material."""
    text = result.raw_text or str(result.data)
    facts = []
    for chunk in [c.strip() for c in text.split("\n\n") if c.strip()][:8]:
        facts.append(
            Fact(subject="web_search_result", predicate="snippet", value=chunk[:500],
                 confidence=0.4, **prov)
        )
    return facts


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #

ENTITY_TYPES = {
    "nasa_neo": NodeType.ASTEROID,
    "jpl_sbdb": NodeType.ASTEROID,
    "exoplanet": NodeType.PLANET,
    "iss": NodeType.SPACECRAFT,
    "eonet": NodeType.EARTH_EVENT,
    "space_weather": NodeType.EARTH_EVENT,
    "launch": NodeType.SPACECRAFT,
    "astro_compute": NodeType.COMPUTATION,
    "rag": NodeType.DOCUMENT,
    "brave_search": NodeType.DOCUMENT,
}


def ingest_tool_result(kg: KnowledgeGraph, result: "ToolResult", agent: str) -> list[Fact]:
    """Normalise one MCP tool result into facts and write them into the graph."""
    if not result.ok:
        return []

    payload = result.data
    prov = _provenance(payload, result, agent)
    data = payload.get("data") if isinstance(payload, dict) and "data" in payload else payload

    match result.server:
        case "nasa_neo" | "jpl_sbdb":
            # Both describe the same bodies. Sharing an extractor keeps subjects and
            # predicates aligned, which is what lets the consensus layer compare them.
            facts = _neo_facts(data, prov, kg)
        case "exoplanet":
            facts = _exoplanet_facts(data, prov, kg)
        case "iss":
            facts = _iss_facts(data, prov, kg)
        case "eonet":
            facts = _eonet_facts(data, prov, kg)
        case "astro_compute":
            facts = _compute_facts(data, prov, kg, result.tool)
        case "rag":
            facts = _rag_facts(data, prov, kg)
        case "brave_search":
            facts = _web_facts(result, prov, kg)
        case _:
            facts = _emit(f"{result.server}.{result.tool}", data, prov) if isinstance(data, dict) else []

    entity_type = ENTITY_TYPES.get(result.server, NodeType.FACT)
    kg.add_facts(facts, entity_type)
    return facts


def register_question(kg: KnowledgeGraph, question: str, session_id: str) -> str:
    return kg.add_entity(question[:200], NodeType.QUESTION, session_id=session_id)


def link_answer(kg: KnowledgeGraph, question_id: str, claim_ids: list[str]) -> None:
    for claim_id in claim_ids:
        kg.add_edge(Edge(source=claim_id, target=question_id, relation=RelationType.ANSWERS))


__all__ = ["ingest_tool_result", "register_question", "link_answer", "make_id"]
