"""Typed vocabulary for the AstralGraph knowledge graph."""

from __future__ import annotations

import hashlib
import time
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field


class NodeType(StrEnum):
    QUESTION = "Question"
    ASTEROID = "Asteroid"
    PLANET = "Planet"
    STAR = "Star"
    SPACECRAFT = "Spacecraft"
    EARTH_EVENT = "EarthEvent"
    LOCATION = "Location"
    DOCUMENT = "Document"
    SOURCE = "Source"
    FACT = "Fact"
    CLAIM = "Claim"
    AGENT = "Agent"
    COMPUTATION = "Computation"


class RelationType(StrEnum):
    HAS_FACT = "HAS_FACT"
    SOURCED_FROM = "SOURCED_FROM"
    PRODUCED_BY = "PRODUCED_BY"
    ABOUT = "ABOUT"
    SUPPORTS = "SUPPORTS"
    CONTRADICTS = "CONTRADICTS"
    ORBITS = "ORBITS"
    HOSTS = "HOSTS"
    LOCATED_AT = "LOCATED_AT"
    CITES = "CITES"
    ANSWERS = "ANSWERS"
    DERIVED_FROM = "DERIVED_FROM"


def make_id(*parts: Any) -> str:
    raw = "|".join(str(p) for p in parts)
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]


class Node(BaseModel):
    id: str
    type: NodeType
    label: str
    properties: dict[str, Any] = Field(default_factory=dict)
    created_at: float = Field(default_factory=time.time)


class Edge(BaseModel):
    source: str
    target: str
    relation: RelationType
    properties: dict[str, Any] = Field(default_factory=dict)


class Fact(BaseModel):
    """An atomic, provenance-carrying assertion. The unit of grounding."""

    subject: str
    predicate: str
    value: Any
    unit: str | None = None
    source_name: str = "unknown"
    source_url: str = ""
    mcp_server: str = ""
    mcp_tool: str = ""
    retrieved_at: str = ""
    confidence: float = 1.0
    agent: str = ""

    @property
    def id(self) -> str:
        return make_id(self.subject, self.predicate, self.value, self.mcp_server)

    @property
    def numeric_value(self) -> float | None:
        if isinstance(self.value, bool):
            return None
        if isinstance(self.value, (int, float)):
            return float(self.value)
        try:
            return float(str(self.value).replace(",", "").strip())
        except (TypeError, ValueError):
            return None

    def to_sentence(self) -> str:
        unit = f" {self.unit}" if self.unit else ""
        return f"{self.subject} — {self.predicate.replace('_', ' ')}: {self.value}{unit}"

    def citation(self) -> str:
        return f"[{self.source_name}]" + (f"({self.source_url})" if self.source_url else "")
