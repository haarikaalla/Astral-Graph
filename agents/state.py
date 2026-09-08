"""Shared state and per-run context for the LangGraph workflow."""

from __future__ import annotations

import operator
from dataclasses import dataclass, field
from typing import Annotated, Any, TypedDict

from core.llm import ClaudeClient, get_llm
from core.mcp_client import MCPHub
from core.telemetry import Trace
from graph.store import KnowledgeGraph
from guardrails.budget import SessionBudget


@dataclass
class RunContext:
    """Everything the agents share for one question. Not serialised."""

    session_id: str
    question: str
    hub: MCPHub
    kg: KnowledgeGraph
    trace: Trace
    budget: SessionBudget
    llm: ClaudeClient = field(default_factory=get_llm)
    injection_warnings: list[str] = field(default_factory=list)
    permission_reports: list[dict[str, Any]] = field(default_factory=list)
    known_sources: list[str] = field(default_factory=list)

    @property
    def llm_available(self) -> bool:
        return self.llm.available


def _merge_dicts(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    return {**left, **right}


class AstralState(TypedDict, total=False):
    """LangGraph state. Parallel branches merge through the annotated reducers."""

    question: str
    session_id: str
    ctx: RunContext

    intent: dict[str, Any]
    route: list[str]

    findings: Annotated[list[dict[str, Any]], operator.add]
    agent_errors: Annotated[list[dict[str, Any]], operator.add]
    agents_run: Annotated[list[str], operator.add]
    guardrail_reports: Annotated[dict[str, Any], _merge_dicts]

    critic: dict[str, Any]
    evidence_requests: list[dict[str, Any]]
    evidence_round: int
    final: dict[str, Any]
    guardrails: dict[str, Any]
