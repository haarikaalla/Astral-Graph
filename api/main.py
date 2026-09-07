"""FastAPI application.

Endpoints
---------
``POST /ask``            Run the full multi-agent pipeline (primary deliverable)
``POST /ask/stream``     Same, as a Server-Sent Events progress stream
``GET  /health``         Liveness + configuration summary
``GET  /mcp/servers``    The MCP registry and the agent→server permission matrix
``GET  /mcp/tools``      Live tool discovery across all reachable servers
``GET  /rag/stats``      Vector-index health
``POST /rag/ingest``     Trigger corpus ingestion
``GET  /graph/{sid}``    Cytoscape-shaped knowledge graph for one session
``GET  /eval/latest``    Most recent benchmark report
"""

from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path
from typing import Any, AsyncIterator

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from sse_starlette.sse import EventSourceResponse

from core.config import get_settings
from core.mcp_client import MCPHub
from core.mcp_registry import AGENT_PERMISSIONS, describe_registry
from core.telemetry import configure_logging, get_logger, log_event
from graph.store import get_graph
from guardrails.schemas import AskRequest, AskResponse
from agents.workflow import answer_question, get_workflow

log = get_logger("api")

app = FastAPI(
    title="AstralGraph",
    version="0.1.0",
    description=(
        "MCP-native multi-agent AI research assistant for space and astronomy. "
        "Every agent is an MCP client; every answer is grounded in a knowledge graph."
    ),
)
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)

_SESSIONS: dict[str, dict[str, Any]] = {}


@app.on_event("startup")
async def _startup() -> None:
    configure_logging()
    settings = get_settings()
    settings.ensure_dirs()
    log_event(
        log, "api_startup",
        llm_enabled=settings.llm_enabled,
        workflow_backend=get_workflow().backend,
        nasa_key="DEMO_KEY" if settings.nasa_api_key == "DEMO_KEY" else "configured",
    )


# --------------------------------------------------------------------------- #
# Core endpoint
# --------------------------------------------------------------------------- #


@app.post("/ask", response_model=AskResponse, summary="Ask an astronomy research question")
async def ask(request: AskRequest) -> AskResponse:
    try:
        response = await answer_question(
            request.question,
            session_id=request.session_id,
            include_trace=request.include_trace,
            include_graph=request.include_graph,
            max_tool_calls=request.max_tool_calls,
        )
    except Exception as exc:  # noqa: BLE001
        log_event(log, "ask_failed", error=str(exc)[:400])
        raise HTTPException(status_code=500, detail=f"pipeline error: {exc}") from exc
    _SESSIONS[response.session_id] = response.model_dump()
    return response


@app.post("/ask/stream", summary="Ask with a Server-Sent Events progress stream")
async def ask_stream(request: AskRequest) -> EventSourceResponse:
    async def event_source() -> AsyncIterator[dict[str, Any]]:
        started = time.perf_counter()
        task = asyncio.create_task(
            answer_question(
                request.question,
                session_id=request.session_id,
                include_trace=True,
                include_graph=request.include_graph,
                max_tool_calls=request.max_tool_calls,
            )
        )
        stage = 0
        stages = [
            "parsing intent", "dispatching domain agents", "calling MCP servers",
            "building knowledge graph", "critic grounding check", "synthesising answer",
        ]
        while not task.done():
            yield {
                "event": "progress",
                "data": json.dumps({
                    "stage": stages[min(stage, len(stages) - 1)],
                    "elapsed_ms": round((time.perf_counter() - started) * 1000, 1),
                }),
            }
            stage += 1
            await asyncio.sleep(1.0)
        try:
            result = await task
            yield {"event": "result", "data": result.model_dump_json()}
        except Exception as exc:  # noqa: BLE001
            yield {"event": "error", "data": json.dumps({"error": str(exc)})}

    return EventSourceResponse(event_source())


# --------------------------------------------------------------------------- #
# Introspection
# --------------------------------------------------------------------------- #


@app.get("/health", summary="Liveness and configuration summary")
async def health() -> dict[str, Any]:
    settings = get_settings()
    from rag.store import get_store

    try:
        rag_stats = get_store().stats()
    except Exception as exc:  # noqa: BLE001
        rag_stats = {"error": str(exc)[:200]}
    return {
        "status": "ok",
        "version": app.version,
        "llm_configured": settings.llm_enabled,
        "model": settings.model,
        "nasa_key": "DEMO_KEY (rate limited)" if settings.nasa_api_key == "DEMO_KEY" else "configured",
        "brave_search": bool(settings.brave_api_key),
        "neo4j": settings.neo4j_enabled,
        "langfuse": settings.langfuse_enabled,
        "workflow_backend": get_workflow().backend,
        "mcp_servers": [s["name"] for s in describe_registry() if s["enabled"]],
        "mcp_servers_disabled": [s["name"] for s in describe_registry() if not s["enabled"]],
        "rag": rag_stats,
        "budgets": {
            "max_tool_calls_per_session": settings.max_tool_calls_per_session,
            "max_retries_per_agent": settings.max_retries_per_agent,
            "grounding_threshold": settings.grounding_threshold,
        },
    }


@app.get("/mcp/servers", summary="MCP registry and agent permission matrix")
async def mcp_servers() -> dict[str, Any]:
    return {
        "servers": describe_registry(),
        "permissions": {agent: sorted(servers) for agent, servers in AGENT_PERMISSIONS.items()},
    }


@app.get("/mcp/tools", summary="Live tool discovery across reachable MCP servers")
async def mcp_tools(server: str | None = Query(default=None)) -> dict[str, Any]:
    names = [server] if server else [s["name"] for s in describe_registry() if s["enabled"]]
    async with MCPHub() as hub:
        tools = await hub.list_tools(names)
        return {
            "requested": names,
            "unreachable": hub.failed_servers,
            "tool_count": len(tools),
            "tools": [
                {"server": t.server, "name": t.name, "description": t.description,
                 "input_schema": t.schema}
                for t in tools
            ],
        }


# --------------------------------------------------------------------------- #
# RAG
# --------------------------------------------------------------------------- #


class IngestRequest(BaseModel):
    arxiv: int = Field(default=0, ge=0, le=2000)
    huggingface: int = Field(default=0, ge=0, le=5000)
    seed: bool = True
    reset: bool = False


@app.get("/rag/stats", summary="Vector index health")
async def rag_stats() -> dict[str, Any]:
    from rag.retriever import get_retriever

    return get_retriever().stats()


@app.post("/rag/ingest", summary="Ingest arXiv / HuggingFace / seed corpus")
async def rag_ingest(request: IngestRequest) -> dict[str, Any]:
    from rag.ingest import run as run_ingest

    return await asyncio.to_thread(
        run_ingest, request.arxiv, request.huggingface, request.seed, request.reset
    )


# --------------------------------------------------------------------------- #
# Knowledge graph + results
# --------------------------------------------------------------------------- #


@app.get("/graph/{session_id}", summary="Knowledge graph for a session")
async def session_graph(session_id: str) -> dict[str, Any]:
    kg = get_graph(session_id)
    if kg.g.number_of_nodes() == 0:
        path = get_settings().fs_root / "graphs" / f"{session_id}.json"
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
        raise HTTPException(status_code=404, detail="no graph for that session")
    return {"stats": kg.stats(), "cytoscape": kg.to_cytoscape()}


@app.get("/session/{session_id}", summary="Full stored response for a session")
async def session(session_id: str) -> dict[str, Any]:
    if session_id in _SESSIONS:
        return _SESSIONS[session_id]
    path = get_settings().fs_root / "reports" / f"{session_id}.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    raise HTTPException(status_code=404, detail="unknown session")


@app.get("/eval/latest", summary="Most recent benchmark report")
async def eval_latest() -> Any:
    results_dir = Path(__file__).resolve().parent.parent / "eval" / "results"
    reports = sorted(results_dir.glob("report_*.json"))
    if not reports:
        raise HTTPException(status_code=404, detail="no evaluation has been run yet")
    return JSONResponse(json.loads(reports[-1].read_text(encoding="utf-8")))


def run() -> None:  # pragma: no cover
    import uvicorn

    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=False)


if __name__ == "__main__":  # pragma: no cover
    run()
