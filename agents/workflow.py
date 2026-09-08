"""LangGraph orchestration.

    intent_parser
         │  (conditional fan-out on the parsed domains)
         ├── neo_agent ─────────┐
         ├── exoplanet_agent ───┤
         ├── events_agent ──────┤──▶ critic_agent ──▶ orchestrator ──▶ END
         └── literature_agent ──┘        │
                    ▲                     │
                    └── evidence loop ───┘  (critic may demand another round)

Domain agents run in parallel within one LangGraph super-step; the Critic only
runs once every dispatched branch has finished.

The Critic is not limited to accepting or rejecting. When evidence is thin it can
emit :class:`~guardrails.schemas.EvidenceRequest` objects, which route targeted
sub-questions back to the domain agents for another round. The loop is bounded by
:data:`MAX_EVIDENCE_ROUNDS` and by the session tool-call budget, so it always
terminates.

If ``langgraph`` is unavailable the same topology is executed by a small asyncio
fallback so the system still runs.
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

#: Hard ceiling on critic-triggered retrieval rounds. The loop also stops early
#: when the budget is exhausted or the critic stops asking.
MAX_EVIDENCE_ROUNDS = 2


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


def _make_domain_node(agent_cls: type, node_name: str = "") -> Any:
    async def node(state: AstralState) -> dict[str, Any]:
        ctx: RunContext = state["ctx"]
        intent = Intent.model_validate(state["intent"])
        intent = _apply_evidence_requests(intent, state, node_name)
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


def _apply_evidence_requests(intent: Intent, state: AstralState, node_name: str) -> Intent:
    """Fold any critic follow-ups for this agent into the intent it re-runs on."""
    requests = state.get("evidence_requests") or []
    if not requests or not node_name:
        return intent
    mine = [
        r for r in requests
        if DOMAIN_AGENTS.get(str(r.get("domain", "")), ("",))[0] == node_name
    ]
    if not mine:
        return intent
    follow_ups = [str(r.get("question", ""))[:400] for r in mine if r.get("question")]
    merged = list(dict.fromkeys([*intent.sub_questions, *follow_ups]))[:6]
    log_event(log, "evidence_round_requested", agent=node_name, follow_ups=len(follow_ups))
    return intent.model_copy(update={"sub_questions": merged})


async def node_critic(state: AstralState) -> dict[str, Any]:
    ctx: RunContext = state["ctx"]
    result = await CriticAgent(ctx).run(list(state.get("findings", [])))
    result["evidence_round"] = int(state.get("evidence_round", 0)) + 1
    return result


async def node_orchestrator(state: AstralState) -> dict[str, Any]:
    ctx: RunContext = state["ctx"]
    return await OrchestratorAgent(ctx).run(
        list(state.get("findings", [])), dict(state.get("critic", {}))
    )


def _router(state: AstralState) -> list[str]:
    return list(state.get("route") or ["literature_agent"])


def _critic_router(state: AstralState) -> list[str]:
    """After the Critic: either gather more evidence, or synthesise the answer.

    Another round is granted only when the Critic asked for one, the round cap is
    not yet reached, and the session still has tool-call budget. Any of those
    failing sends the run to the Orchestrator, which answers with what it has.
    """
    requests = state.get("evidence_requests") or []
    rounds = int(state.get("evidence_round", 0))
    ctx: RunContext = state["ctx"]

    if not requests or rounds >= MAX_EVIDENCE_ROUNDS:
        return ["orchestrator"]
    if not ctx.budget.allows("evidence_loop"):
        log_event(log, "evidence_loop_budget_exhausted", round=rounds)
        return ["orchestrator"]

    targets = list(dict.fromkeys(
        DOMAIN_AGENTS[str(r.get("domain", ""))][0]
        for r in requests
        if str(r.get("domain", "")) in DOMAIN_AGENTS
    ))
    if not targets:
        return ["orchestrator"]
    log_event(log, "evidence_loop_entered", round=rounds, targets=targets)
    return targets


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
                    builder.add_node(node_name, _make_domain_node(agent_cls, node_name))
            builder.add_node("critic_agent", node_critic)
            builder.add_node("orchestrator", node_orchestrator)

            domain_nodes = {v[0] for v in DOMAIN_AGENTS.values()}
            builder.add_edge(START, "intent_parser")
            builder.add_conditional_edges(
                "intent_parser",
                _router,
                {name: name for name in domain_nodes},
            )
            for node_name in domain_nodes:
                builder.add_edge(node_name, "critic_agent")
            # The critic may send the run back for another evidence round.
            builder.add_conditional_edges(
                "critic_agent",
                _critic_router,
                {**{name: name for name in domain_nodes}, "orchestrator": "orchestrator"},
            )
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

        async def gather(targets: list[str]) -> None:
            tasks = [
                _make_domain_node(node_map[name], name)(merged)  # type: ignore[arg-type]
                for name in targets
                if name in node_map
            ]
            for result in await asyncio.gather(*tasks, return_exceptions=True):
                if not isinstance(result, dict):
                    continue
                merged["findings"] = [*merged.get("findings", []), *result.get("findings", [])]
                merged["agents_run"] = [*merged.get("agents_run", []), *result.get("agents_run", [])]
                merged["agent_errors"] = [
                    *merged.get("agent_errors", []), *result.get("agent_errors", [])
                ]
                merged["guardrail_reports"] = {
                    **merged.get("guardrail_reports", {}), **result.get("guardrail_reports", {})
                }

        await gather(list(merged.get("route", [])))
        merged.update(await node_critic(merged))  # type: ignore[arg-type]

        # Same bounded evidence loop the LangGraph topology runs.
        while True:
            targets = _critic_router(merged)  # type: ignore[arg-type]
            if targets == ["orchestrator"]:
                break
            await gather(targets)
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
                              "guardrail_reports": {}, "evidence_requests": [],
                              "evidence_round": 0}
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
            evidence_rounds=int(result.get("evidence_round", 1) or 1),
            source_agreement=float(
                (guardrails.get("consensus") or {}).get("agreement_rate", 1.0)
            ),
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
