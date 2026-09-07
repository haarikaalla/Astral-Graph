"""Knowledge-graph storage.

Default backend is in-process **NetworkX** (zero setup, free). If ``NEO4J_URI`` is
set the same API writes to **Neo4j** as well, so the graph survives restarts and
can be browsed in Neo4j Browser.
"""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any, Iterable

import networkx as nx

from core.config import get_settings
from core.telemetry import get_logger, log_event
from graph.schema import Edge, Fact, Node, NodeType, RelationType, make_id

log = get_logger("graph")


class KnowledgeGraph:
    """Directed multigraph of entities, facts, sources and claims."""

    def __init__(self, session_id: str = "global") -> None:
        self.session_id = session_id
        self.g = nx.MultiDiGraph()
        self._lock = threading.RLock()
        self._neo4j = _Neo4jMirror()

    # ---- writes ----------------------------------------------------------
    def add_node(self, node: Node) -> str:
        with self._lock:
            if self.g.has_node(node.id):
                self.g.nodes[node.id]["properties"].update(node.properties)
            else:
                self.g.add_node(
                    node.id,
                    type=str(node.type),
                    label=node.label,
                    properties=dict(node.properties),
                    created_at=node.created_at,
                )
        self._neo4j.merge_node(node)
        return node.id

    def add_edge(self, edge: Edge) -> None:
        with self._lock:
            if not self.g.has_node(edge.source) or not self.g.has_node(edge.target):
                return
            self.g.add_edge(
                edge.source, edge.target, key=str(edge.relation),
                relation=str(edge.relation), **edge.properties,
            )
        self._neo4j.merge_edge(edge)

    def add_entity(self, name: str, node_type: NodeType, **properties: Any) -> str:
        return self.add_node(
            Node(id=make_id(node_type, name.lower()), type=node_type, label=name,
                 properties=properties)
        )

    def add_fact(self, fact: Fact, entity_type: NodeType = NodeType.FACT) -> str:
        """Store a fact plus its subject entity and source, fully linked."""
        subject_id = self.add_entity(fact.subject, entity_type)
        fact_id = self.add_node(
            Node(
                id=fact.id,
                type=NodeType.FACT,
                label=fact.to_sentence(),
                properties=fact.model_dump(),
            )
        )
        self.add_edge(Edge(source=subject_id, target=fact_id, relation=RelationType.HAS_FACT))
        self.add_edge(Edge(source=fact_id, target=subject_id, relation=RelationType.ABOUT))

        if fact.source_name:
            source_id = self.add_node(
                Node(
                    id=make_id("source", fact.source_name, fact.source_url),
                    type=NodeType.SOURCE,
                    label=fact.source_name,
                    properties={"url": fact.source_url, "mcp_server": fact.mcp_server},
                )
            )
            self.add_edge(
                Edge(
                    source=fact_id, target=source_id, relation=RelationType.SOURCED_FROM,
                    properties={"tool": fact.mcp_tool, "retrieved_at": fact.retrieved_at},
                )
            )
        if fact.agent:
            agent_id = self.add_entity(fact.agent, NodeType.AGENT)
            self.add_edge(Edge(source=fact_id, target=agent_id, relation=RelationType.PRODUCED_BY))
        return fact_id

    def add_facts(self, facts: Iterable[Fact], entity_type: NodeType = NodeType.FACT) -> list[str]:
        return [self.add_fact(f, entity_type) for f in facts]

    def add_claim(self, text: str, agent: str, supported_by: list[str], grounded: bool) -> str:
        claim_id = self.add_node(
            Node(
                id=make_id("claim", text),
                type=NodeType.CLAIM,
                label=text,
                properties={"agent": agent, "grounded": grounded},
            )
        )
        for fact_id in supported_by:
            self.add_edge(
                Edge(source=fact_id, target=claim_id, relation=RelationType.SUPPORTS)
            )
        return claim_id

    def add_document(self, doc_id: str, title: str, **properties: Any) -> str:
        return self.add_node(
            Node(id=make_id("doc", doc_id), type=NodeType.DOCUMENT, label=title,
                 properties={"doc_id": doc_id, **properties})
        )

    # ---- reads -----------------------------------------------------------
    def facts(self) -> list[Fact]:
        out: list[Fact] = []
        with self._lock:
            for _, attrs in self.g.nodes(data=True):
                if attrs.get("type") == str(NodeType.FACT):
                    try:
                        out.append(Fact.model_validate(attrs.get("properties", {})))
                    except Exception:  # noqa: BLE001
                        continue
        return out

    def facts_about(self, subject_substring: str) -> list[Fact]:
        needle = subject_substring.lower()
        return [f for f in self.facts() if needle in f.subject.lower()]

    def numeric_facts(self) -> list[tuple[Fact, float]]:
        pairs = []
        for fact in self.facts():
            value = fact.numeric_value
            if value is not None:
                pairs.append((fact, value))
        return pairs

    def entities(self, node_type: NodeType | None = None) -> list[dict[str, Any]]:
        with self._lock:
            return [
                {"id": nid, **attrs}
                for nid, attrs in self.g.nodes(data=True)
                if node_type is None or attrs.get("type") == str(node_type)
            ]

    def sources(self) -> list[dict[str, Any]]:
        return self.entities(NodeType.SOURCE)

    def stats(self) -> dict[str, Any]:
        with self._lock:
            by_type: dict[str, int] = {}
            for _, attrs in self.g.nodes(data=True):
                key = attrs.get("type", "?")
                by_type[key] = by_type.get(key, 0) + 1
            return {
                "nodes": self.g.number_of_nodes(),
                "edges": self.g.number_of_edges(),
                "nodes_by_type": by_type,
                "backend": "neo4j+networkx" if self._neo4j.active else "networkx",
            }

    def evidence_bundle(self, limit: int = 60) -> str:
        """Compact, citation-carrying text block handed to the Critic/Orchestrator."""
        lines = []
        for i, fact in enumerate(self.facts()[:limit], start=1):
            lines.append(f"F{i}. {fact.to_sentence()}  {fact.citation()}")
        return "\n".join(lines) or "(no grounded facts collected)"

    # ---- export ----------------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        with self._lock:
            return {
                "nodes": [{"id": n, **a} for n, a in self.g.nodes(data=True)],
                "edges": [
                    {"source": u, "target": v, **d} for u, v, d in self.g.edges(data=True)
                ],
                "stats": self.stats(),
            }

    def to_cytoscape(self) -> dict[str, Any]:
        data = self.to_dict()
        return {
            "nodes": [
                {"data": {"id": n["id"], "label": n.get("label", "")[:60], "type": n.get("type")}}
                for n in data["nodes"]
            ],
            "edges": [
                {"data": {"source": e["source"], "target": e["target"],
                          "label": e.get("relation", "")}}
                for e in data["edges"]
            ],
        }

    def save(self, path: str | Path) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(self.to_dict(), indent=2, default=str), encoding="utf-8")
        log_event(log, "graph_saved", path=str(target), **self.stats())
        return target


class _Neo4jMirror:
    """Optional write-through mirror to Neo4j; silently inert when not configured."""

    def __init__(self) -> None:
        settings = get_settings()
        self.active = False
        self._driver = None
        if not settings.neo4j_enabled:
            return
        try:
            from neo4j import GraphDatabase

            self._driver = GraphDatabase.driver(
                settings.neo4j_uri, auth=(settings.neo4j_user, settings.neo4j_password)
            )
            self._driver.verify_connectivity()
            self.active = True
            log_event(log, "neo4j_connected", uri=settings.neo4j_uri)
        except Exception as exc:  # noqa: BLE001
            log_event(log, "neo4j_unavailable", error=str(exc))
            self._driver = None

    def merge_node(self, node: Node) -> None:
        if not self.active or self._driver is None:
            return
        try:
            with self._driver.session() as session:
                session.run(
                    f"MERGE (n:{node.type} {{id: $id}}) SET n.label = $label, n.props = $props",
                    id=node.id, label=node.label,
                    props=json.dumps(node.properties, default=str),
                )
        except Exception as exc:  # noqa: BLE001
            log_event(log, "neo4j_write_failed", error=str(exc))
            self.active = False

    def merge_edge(self, edge: Edge) -> None:
        if not self.active or self._driver is None:
            return
        try:
            with self._driver.session() as session:
                session.run(
                    "MATCH (a {id: $s}), (b {id: $t}) "
                    f"MERGE (a)-[r:{edge.relation}]->(b) SET r.props = $props",
                    s=edge.source, t=edge.target,
                    props=json.dumps(edge.properties, default=str),
                )
        except Exception as exc:  # noqa: BLE001
            log_event(log, "neo4j_write_failed", error=str(exc))
            self.active = False


_GRAPHS: dict[str, KnowledgeGraph] = {}
_GRAPH_LOCK = threading.Lock()


def get_graph(session_id: str = "global") -> KnowledgeGraph:
    with _GRAPH_LOCK:
        if session_id not in _GRAPHS:
            _GRAPHS[session_id] = KnowledgeGraph(session_id)
        return _GRAPHS[session_id]
