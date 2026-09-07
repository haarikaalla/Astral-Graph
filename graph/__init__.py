"""Knowledge-graph layer: every agent output is grounded in facts stored here."""

from graph.schema import Edge, Fact, Node, NodeType, RelationType
from graph.store import KnowledgeGraph, get_graph
from graph.builder import ingest_tool_result
from graph.grounding import GroundingReport, check_grounding

__all__ = [
    "Edge",
    "Fact",
    "Node",
    "NodeType",
    "RelationType",
    "KnowledgeGraph",
    "get_graph",
    "ingest_tool_result",
    "GroundingReport",
    "check_grounding",
]
