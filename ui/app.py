"""AstralGraph Streamlit dashboard.

Run with::

    streamlit run ui/app.py

Talks to the FastAPI service if it is running, otherwise calls the pipeline
in-process. Shows the answer, its citations, the guardrail verdict, the knowledge
graph that grounds it, live MCP server status and the latest benchmark table.
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.config import REPO_ROOT, get_settings  # noqa: E402
from core.mcp_registry import describe_registry  # noqa: E402

st.set_page_config(page_title="AstralGraph", page_icon="*", layout="wide")

RESULTS_DIR = REPO_ROOT / "eval" / "results"


# --------------------------------------------------------------------------- #
# Backend calls
# --------------------------------------------------------------------------- #
def ask_via_api(question: str, base_url: str) -> dict | None:
    try:
        import httpx

        response = httpx.post(
            f"{base_url.rstrip('/')}/ask",
            json={"question": question, "include_graph": True, "include_trace": True},
            timeout=180.0,
        )
        response.raise_for_status()
        return response.json()
    except Exception:  # noqa: BLE001 - fall back to in-process
        return None


def ask_in_process(question: str) -> dict:
    from agents.workflow import answer_question

    response = asyncio.run(
        answer_question(question, include_trace=True, include_graph=True)
    )
    return response.model_dump()


# --------------------------------------------------------------------------- #
# Rendering
# --------------------------------------------------------------------------- #
def render_graph(graph: dict) -> None:
    elements = graph.get("elements") or graph.get("nodes")
    if not elements:
        st.caption("No graph elements returned for this session.")
        st.json(graph)
        return

    try:
        import networkx as nx
        import plotly.graph_objects as go
    except ImportError:
        st.json(graph)
        return

    nodes = graph.get("nodes", [])
    edges = graph.get("edges", [])
    if not nodes:
        st.json(graph)
        return

    g = nx.Graph()
    for node in nodes:
        g.add_node(node["id"], **node)
    for edge in edges:
        if edge.get("source") in g and edge.get("target") in g:
            g.add_edge(edge["source"], edge["target"])

    positions = nx.spring_layout(g, seed=7, k=0.6)
    edge_x, edge_y = [], []
    for source, target in g.edges():
        x0, y0 = positions[source]
        x1, y1 = positions[target]
        edge_x += [x0, x1, None]
        edge_y += [y0, y1, None]

    palette = {
        "Fact": "#4cc9f0", "Claim": "#f72585", "Source": "#ffd166",
        "Question": "#ffffff", "Agent": "#90be6d", "Answer": "#f8961e",
    }
    node_x = [positions[n][0] for n in g.nodes()]
    node_y = [positions[n][1] for n in g.nodes()]
    colours = [palette.get(g.nodes[n].get("type", ""), "#9d4edd") for n in g.nodes()]
    labels = [
        f"{g.nodes[n].get('type', '?')}: {str(g.nodes[n].get('label', n))[:90]}"
        for n in g.nodes()
    ]

    figure = go.Figure()
    figure.add_trace(go.Scatter(x=edge_x, y=edge_y, mode="lines",
                                line=dict(width=0.6, color="#555"), hoverinfo="none"))
    figure.add_trace(go.Scatter(x=node_x, y=node_y, mode="markers", text=labels,
                                hoverinfo="text",
                                marker=dict(size=11, color=colours,
                                            line=dict(width=0.5, color="#111"))))
    figure.update_layout(showlegend=False, height=520, margin=dict(l=0, r=0, t=10, b=0),
                         xaxis=dict(visible=False), yaxis=dict(visible=False))
    st.plotly_chart(figure, use_container_width=True)


def render_guardrails(guardrails: dict) -> None:
    if not guardrails:
        st.caption("No guardrail report.")
        return

    grounding = guardrails.get("grounding", {})
    numeric = guardrails.get("numeric", {})
    citations = guardrails.get("citations", {})

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Guardrail score", f"{guardrails.get('guardrail_score', 0):.2f}")
    col2.metric("Grounding", f"{grounding.get('grounding_score', 0):.2f}",
                "grounded" if grounding.get("grounded") else "not grounded")
    col3.metric("Numbers verified",
                f"{numeric.get('numbers_verified', 0)}/{numeric.get('numbers_checked', 0)}")
    col4.metric("Citations", "ok" if citations.get("ok") else "missing")

    if guardrails.get("failures"):
        st.warning("Guardrail failures: " + "; ".join(guardrails["failures"]))
    if numeric.get("unverified_numbers"):
        st.error("Numbers not backed by any tool result: "
                 + ", ".join(f"{n:g}" for n in numeric["unverified_numbers"][:10]))
    if guardrails.get("injection_warnings"):
        st.error("Prompt-injection markers found in tool output: "
                 + ", ".join(guardrails["injection_warnings"]))

    with st.expander("Full guardrail report"):
        st.json(guardrails)


def render_trace(trace: dict) -> None:
    spans = [s for s in trace.get("spans", []) if s.get("kind") == "mcp_tool"]
    if spans:
        st.dataframe(
            [
                {
                    "agent": s.get("agent", ""),
                    "server": s.get("server", ""),
                    "tool": s.get("tool", ""),
                    "ms": s.get("ms", 0),
                    "ok": s.get("ok", False),
                }
                for s in spans
            ],
            use_container_width=True, hide_index=True,
        )
    stats = trace.get("server_stats", {})
    if stats:
        st.caption("Per-server call success rate")
        st.dataframe(
            [
                {"server": name, **values,
                 "success_rate": f"{(values['ok'] / values['calls']):.0%}" if values["calls"] else "-"}
                for name, values in stats.items()
            ],
            use_container_width=True, hide_index=True,
        )


# --------------------------------------------------------------------------- #
# Layout
# --------------------------------------------------------------------------- #
settings = get_settings()

st.title("AstralGraph")
st.caption("Multi-agent space & astronomy research assistant — every fact arrives "
           "through an MCP server and is grounded in a knowledge graph.")

with st.sidebar:
    st.subheader("Configuration")
    st.write("**Claude synthesis:**", "enabled" if settings.llm_enabled else "deterministic fallback")
    st.write("**Model:**", settings.model)
    st.write("**NASA key:**", "custom" if settings.nasa_api_key != "DEMO_KEY" else "DEMO_KEY")
    api_url = st.text_input("FastAPI base URL", "http://127.0.0.1:8000")
    use_api = st.checkbox("Use the API if reachable", value=True)

    st.subheader("MCP servers")
    for spec in describe_registry():
        icon = "on" if spec["enabled"] else "off"
        st.write(f"`{spec['name']}` ({spec['origin']}) — **{icon}**")
        st.caption("used by: " + ", ".join(spec["used_by"]))

tab_ask, tab_bench, tab_about = st.tabs(["Ask", "Benchmark", "Architecture"])

with tab_ask:
    question = st.text_input(
        "Question",
        placeholder="How many people are in space right now, and where is the ISS?",
    )
    examples = [
        "Which near-Earth asteroids come closest to Earth today?",
        "How many confirmed exoplanets are in the NASA Exoplanet Archive?",
        "What is the impact energy in megatons of a 100 metre asteroid at 20 km/s?",
        "How far is the ISS from London right now?",
        "What exactly makes an asteroid a Potentially Hazardous Asteroid?",
    ]
    picked = st.selectbox("Or pick an example", ["-"] + examples)
    if picked != "-":
        question = picked

    if st.button("Ask", type="primary") and question.strip():
        with st.spinner("Running the agent pipeline over MCP..."):
            payload = ask_via_api(question, api_url) if use_api else None
            if payload is None:
                payload = ask_in_process(question)

        st.subheader("Answer")
        st.markdown(payload.get("answer", ""))

        if payload.get("key_points"):
            st.markdown("**Key points**")
            for point in payload["key_points"]:
                st.markdown(f"- {point}")

        if payload.get("caveats"):
            for caveat in payload["caveats"]:
                st.info(caveat)

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Confidence", f"{payload.get('confidence', 0):.2f}")
        col2.metric("Critic verdict", payload.get("critic_verdict", "?"))
        col3.metric("Latency", f"{payload.get('latency_ms', 0)/1000:.1f} s")
        col4.metric("Cost", f"${payload.get('usd_cost', 0):.5f}")

        st.caption("Agents run: " + ", ".join(payload.get("agents_run", [])))

        if payload.get("citations"):
            st.subheader("Citations")
            for citation in payload["citations"]:
                label = citation.get("label", "")
                url = citation.get("url", "")
                st.markdown(f"- {'[' + label + '](' + url + ')' if url else label}"
                            f" *({citation.get('kind', '')})*")

        st.subheader("Guardrails")
        render_guardrails(payload.get("guardrails", {}))

        st.subheader("Knowledge graph")
        render_graph(payload.get("knowledge_graph", {}))

        st.subheader("MCP trace")
        render_trace(payload.get("trace", {}))

with tab_bench:
    latest = RESULTS_DIR / "latest.md"
    latest_json = RESULTS_DIR / "latest.json"
    if latest.exists():
        st.markdown(latest.read_text(encoding="utf-8"))
        if latest_json.exists():
            with st.expander("Raw per-question records"):
                st.json(json.loads(latest_json.read_text(encoding="utf-8"))["records"])
    else:
        st.info("No benchmark results yet. Run `python -m eval.harness`.")

with tab_about:
    st.markdown(
        """
### How a question flows

1. **Intent Parser** classifies the question into domains — no tool access at all.
2. **Domain agents** run in parallel, each restricted to its own MCP servers:
   NEO Agent, Exoplanet Agent, Events Agent, Literature/RAG Agent.
3. Every tool result is ingested into a **knowledge graph** as facts with provenance.
4. The **Critic** re-checks the findings against the graph in pure Python first,
   then asks Claude for a second opinion. Deterministic result wins.
5. The **Orchestrator** synthesises the final answer, which is checked again for
   grounding, numeric support and citations, and repaired or downgraded if needed.

### Why MCP

No agent ever calls an API directly. Everything goes through an MCP server, so the
permission matrix, call budget, prompt-injection scan and provenance capture all
happen in one place.
"""
    )
    st.subheader("Agent to MCP server permissions")
    from core.mcp_registry import AGENT_PERMISSIONS

    st.dataframe(
        [{"agent": agent, "allowed servers": ", ".join(sorted(servers)) or "none (no tool access)"}
         for agent, servers in AGENT_PERMISSIONS.items()],
        use_container_width=True, hide_index=True,
    )
