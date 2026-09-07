"""MCP server exposing the local RAG index (Chroma + sentence-transformers).

Keeping retrieval behind MCP means the Literature agent obeys the same
permission-scoped, budgeted, traced call path as every other agent — no module
reaches into the vector store directly at query time.

Run standalone:  ``python -m mcp_servers.rag_server``
"""

from __future__ import annotations

from mcp_servers._common import clamp, err, ok
from mcp_servers._compat import MCPServer
from rag.ingest import run as run_ingest
from rag.retriever import get_retriever

SOURCE = "AstralGraph RAG (arXiv astro-ph + HuggingFace + curated corpus)"
URL = "local://chroma"

mcp = MCPServer("astral-rag")


@mcp.tool()
def literature_search(query: str, top_k: int = 5, source_filter: str = "") -> dict:
    """Semantic search over the indexed astronomy literature.

    Returns citation-carrying passages. Every literature claim an agent makes must
    quote one of these ``citation`` strings.

    Args:
        query: Natural-language search query.
        top_k: Number of passages to return (1-15).
        source_filter: Optional exact source name, e.g. ``arXiv astro-ph``.
    """
    if not query.strip():
        return err("query is required", source=SOURCE, url=URL)
    retriever = get_retriever()
    chunks = retriever.search(
        query.strip(), k=clamp(top_k, 1, 15), source_filter=source_filter.strip() or None
    )
    if not chunks:
        return err(
            "the literature index is empty or returned no matches",
            source=SOURCE,
            url=URL,
            hint="Run: python -m rag.ingest --seed --arxiv 200",
        )
    return ok(
        {
            "query": query,
            "count": len(chunks),
            "results": [c.to_dict() for c in chunks],
            "context_block": retriever.context_block(chunks),
        },
        source=SOURCE,
        url=URL,
    )


@mcp.tool()
def literature_compare(query_a: str, query_b: str, top_k: int = 3) -> dict:
    """Retrieve evidence for two topics side by side, for comparison questions.

    Args:
        query_a: First topic.
        query_b: Second topic.
        top_k: Passages per topic (1-8).
    """
    retriever = get_retriever()
    k = clamp(top_k, 1, 8)
    a = retriever.search(query_a, k=k)
    b = retriever.search(query_b, k=k)
    return ok(
        {
            "topic_a": {"query": query_a, "results": [c.to_dict() for c in a]},
            "topic_b": {"query": query_b, "results": [c.to_dict() for c in b]},
        },
        source=SOURCE,
        url=URL,
    )


@mcp.tool()
def literature_stats() -> dict:
    """Index health: backend in use, chunk counts and per-source breakdown."""
    return ok(get_retriever().stats(), source=SOURCE, url=URL)


@mcp.tool()
def literature_reindex(arxiv: int = 0, huggingface: int = 0, seed: bool = True,
                       reset: bool = False) -> dict:
    """Rebuild or extend the literature index (admin tool; slow, network-bound).

    Args:
        arxiv: Number of recent arXiv astro-ph abstracts to fetch.
        huggingface: Number of HuggingFace dataset rows to fetch.
        seed: Include the bundled curated corpus.
        reset: Wipe the collection first.
    """
    stats = run_ingest(arxiv=arxiv, hf=huggingface, seed=seed, reset=reset)
    return ok(stats, source=SOURCE, url=URL)


if __name__ == "__main__":
    mcp.run(transport="stdio")
