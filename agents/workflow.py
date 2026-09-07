"""LangGraph orchestration.

    intent_parser
         │  (conditional fan-out on the parsed domains)
         ├── neo_agent ─────────┐
         ├── exoplanet_agent ───┤
         ├── events_agent ──────┤──▶ critic_agent ──▶ orchestrator ──▶ END
         └── literature_agent ──┘

Domain agents run in parallel within one LangGraph super-step; the Critic only
runs once every dispatched branch has finished. If ``langgraph`` is unavailable the
same topology is executed by a small asyncio fallback so the system still runs.
"""

from __future__ import annotations

import uuid
from typing import Any

from core.config import get_settings
from core.llm import get_llm
from core.mcp_client import MCPHub
from core.telemetry import Trace, get_logger, log_event, push_trace_to_langfuse, span
from graph.store import KnowledgeGraph
from guardrails.budget import SessionBudget
from guardrails.schemas import AskResponse, Intent
from agents.critic_agent import CriticAgent
from agents.events_agent import EventsAgent
from agents.exoplanet_agent import ExoplanetAgent
from agents.intent_parser import IntentParserAgent
from agents.literature_agent import LiteratureAgent
from agents.neo_agent import NEOAgent
from agents.orchestrator import OrchestratorAgent
from agents.state import AstralState, RunContext

log = get_logger("workflow")

DOMAIN_AGENTS = {
    "neo": ("neo_agent", NEOAgent),
    "exoplanet": ("exoplanet_agent", ExoplanetAgent),
    "events": ("events_agent", EventsAgent),
    "literature": ("literature_agent", LiteratureAgent),
    "general": ("literature_agent", LiteratureAgent),
}


# --------------------------------------------------------------------------- #
# Node implementations (shared by both executors)
# --------------------------------------------------------------------------- #


async def node_intent(state: AstralState) -> dict[str, Any]:
    ctx: RunContext = state["ctx"]
    intent = await IntentParserAgent(ctx).run(state["question"])
    route = list(dict.fromkeys(DOMAIN_AGENTS[d][0] for d in intent.domains if d in DOMAIN_AGENTS))
    if not route:
        route = ["literature_agent"]
    log_event(log, "routed", session_id=ctx.session_id, domains=intent.domains, route=route)
    return {"intent": intent.model_dump(), "route": route, "agents_run": ["intent_parser"]}


def _make_domain_node(agent_cls: type) -> Any:
    async def node(state: AstralState) -> dict[str, Any]:
        ctx: RunContext = state["ctx"]
        intent = Intent.model_validate(state["intent"])
        agent = agent_cls(ctx)
        try:
            return await agent.run(intent)
        except Exception as exc:  # noqa: BLE001
            log_event(log, "agent_crashed", agent=agent.name, error=str(exc)[:300])
            return {
                "agent_errors": [{"agent": agent.name, "error": f"{type(exc).__name__}: {exc}"}],
                "agents_run": [agent.name],
            }

    return node


async def node_critic(state: AstralState) -> dict[str, Any]:
    ctx: RunContext = state["ctx"]
    return await CriticAgent(ctx).run(list(state.get("findings", [])))


async def node_orchestrator(state: AstralState) -> dict[str, Any]:
    ctx: RunContext = state["ctx"]
    return await OrchestratorAgent(ctx).run(
        list(state.get("findings", [])), dict(state.get("critic", {}))
    )


def _router(state: AstralState) -> list[str]:
    return list(state.get("route") or ["literature_agent"])


# --------------------------------------------------------------------------- #
# Workflow
# --------------------------------------------------------------------------- #


class AstralWorkflow:
    """Compiled LangGraph app with an asyncio fallback."""

    def __init__(self) -> None:
        self.app = None
        self.backend = "asyncio-fallback"
        try:
            from langgraph.graph import END, START, StateGraph

            builder = StateGraph(AstralState)
            builder.add_node("intent_parser", node_intent)
            for _, (node_name, agent_cls) in DOMAIN_AGENTS.items():
                if node_name not in builder.nodes:
                    builder.add_node(node_name, _make_domain_node(agent_cls))
            builder.add_node("critic_agent", node_critic)
            builder.add_node("orchestrator", node_orchestrator)

            builder.add_edge(START, "intent_parser")
            builder.add_conditional_edges(
                "intent_parser",
                _router,
                {name: name for name, _ in
                 {v[0]: v[1] for v in DOMAIN_AGENTS.values()}.items()},
            )
            for node_name in {v[0] for v in DOMAIN_AGENTS.values()}:
                builder.add_edge(node_name, "critic_agent")
            builder.add_edge("critic_agent", "orchestrator")
            builder.add_edge("orchestrator", END)

            self.app = builder.compile()
            self.backend = "langgraph"
        except Exception as exc:  # noqa: BLE001
            log_event(log, "langgraph_unavailable", error=str(exc)[:200])

    async def invoke(self, state: AstralState) -> AstralState:
        if self.app is not None:
            return await self.app.ainvoke(state, {"recursion_limit": 25})  # type: ignore[return-value]
        return await self._fallback(state)

    async def _fallback(self, state: AstralState) -> AstralState:
        import asyncio

        merged: dict[str, Any] = dict(state)
        merged.update(await node_intent(state))  # type: ignore[arg-type]
        node_map = {v[0]: v[1] for v in DOMAIN_AGENTS.values()}
        tasks = [
            _make_domain_node(node_map[name])(merged)  # type: ignore[arg-type]
            for name in merged["route"]
            if name in node_map
        ]
        findings: list[dict[str, Any]] = []
        agents_run: list[str] = list(merged.get("agents_run", []))
        errors: list[dict[str, Any]] = []
        reports: dict[str, Any] = {}
        for result in await asyncio.gather(*tasks, return_exceptions=True):
            if isinstance(result, dict):
                findings.extend(result.get("findings", []))
                agents_run.extend(result.get("agents_run", []))
                errors.extend(result.get("agent_errors", []))
                reports.update(result.get("guardrail_reports", {}))
        merged.update({"findings": findings, "agents_run": agents_run,
                       "agent_errors": errors, "guardrail_reports": reports})
        merged.update(await node_critic(merged))  # type: ignore[arg-type]
        merged.update(await node_orchestrator(merged))  # type: ignore[arg-type]
        return merged  # type: ignore[return-value]


_WORKFLOW: AstralWorkflow | None = None


def get_workflow() -> AstralWorkflow:
    global _WORKFLOW
    if _WORKFLOW is None:
        _WORKFLOW = AstralWorkflow()
        log_event(log, "workflow_ready", backend=_WORKFLOW.backend)
    return _WORKFLOW


# --------------------------------------------------------------------------- #
# Public entry point
# --------------------------------------------------------------------------- #


async def answer_question(
    question: str,
    *,
    session_id: str | None = None,
    include_trace: bool = True,
    include_graph: bool = False,
    max_tool_calls: int | None = None,
) -> AskResponse:
    """Run the full AstralGraph pipeline for one question."""
    settings = get_settings()
    sid = session_id or uuid.uuid4().hex[:12]
    trace = Trace(session_id=sid, question=question)
    kg = KnowledgeGraph(session_id=sid)
    budget = SessionBudget(max_tool_calls=max_tool_calls or settings.max_tool_calls_per_session)

    async with MCPHub(trace=trace) as hub:
        if max_tool_calls:
            hub.settings.max_tool_calls_per_session = max_tool_calls
        ctx = RunContext(
            session_id=sid, question=question, hub=hub, kg=kg, trace=trace,
            budget=budget, llm=get_llm(),
        )
        state: AstralState = {"question": question, "session_id": sid, "ctx": ctx,
                              "findings": [], "agents_run": [], "agent_errors": [],
                              "guardrail_reports": {}}
        with span(trace, "pipeline", "workflow", question=question[:200]):
            result = await get_workflow().invoke(state)

        budget.record_spend(trace.usd_cost)
        final = result.get("final", {}) or {}
        guardrails = result.get("guardrails", {}) or {}
        grounding = (guardrails.get("grounding") or {})

        response = AskResponse(
            session_id=sid,
            question=question,
            answer=final.get("answer", "No answer could be produced."),
            key_points=final.get("key_points", []),
            citations=final.get("citations", []),
            confidence=float(final.get("confidence", 0.0)),
            caveats=final.get("caveats", []),
            grounded=bool(grounding.get("grounded", False)),
            grounding_score=float(grounding.get("grounding_score", 0.0)),
            critic_verdict=str((result.get("critic") or {}).get("verdict", "unknown")),
            agents_run=list(dict.fromkeys(result.get("agents_run", []))),
            guardrails=guardrails,
            knowledge_graph=(kg.to_cytoscape() if include_graph else kg.stats()),
            trace=(trace.summary() if include_trace else {}),
            latency_ms=trace.elapsed_ms,
            usd_cost=round(trace.usd_cost, 6),
        )

    push_trace_to_langfuse(trace, response.answer)
    log_event(
        log, "question_answered", session_id=sid, latency_ms=round(trace.elapsed_ms, 1),
        tool_calls=len(trace.tool_calls()), grounded=response.grounded,
        verdict=response.critic_verdict, usd=response.usd_cost,
    )
    return response
