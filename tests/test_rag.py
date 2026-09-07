"""RAG retrieval tests.

These exercise whatever backend is installed: Chroma + sentence-transformers when
available, otherwise the built-in JSON/bag-of-words fallback. Either way the
retriever must return the curated passage that answers the query, with a citation.
"""

from __future__ import annotations

import pytest

from rag.retriever import get_retriever
from rag.store import get_store


@pytest.fixture(scope="module")
def retriever():
    store = get_store()
    if store.stats().get("chunks", 0) == 0:
        pytest.skip("RAG index is empty - run 'python -m rag.ingest --seed' first")
    return get_retriever()


def test_retrieves_the_pha_definition(retriever):
    hits = retriever.search("What makes an asteroid potentially hazardous?", k=5)
    assert hits
    blob = " ".join(h.text.lower() for h in hits)
    assert "0.05" in blob and "22" in blob


def test_retrieves_the_habitable_zone_limits(retriever):
    hits = retriever.search("conservative habitable zone limits for the Sun", k=5)
    blob = " ".join(h.text for h in hits)
    assert "Kopparapu" in blob or "0.99" in blob


def test_retrieves_the_parsec_definition(retriever):
    hits = retriever.search("how many light years is a parsec", k=5)
    assert any("3.26" in h.text for h in hits)


def test_every_hit_carries_a_citation(retriever):
    for hit in retriever.search("transit method planet radius", k=5):
        assert hit.citation
        assert hit.source


def test_context_block_is_numbered_and_bounded(retriever):
    block = retriever.context_block(retriever.search("exoplanet detection methods", k=3))
    assert "[1]" in block
    assert len(block) < 20000


def test_k_is_respected(retriever):
    assert len(retriever.search("asteroid", k=3)) <= 3


def test_unrelated_query_does_not_invent_results(retriever):
    """Retrieval may return nothing relevant, but must never fabricate text."""
    for hit in retriever.search("quarterly revenue of a software company", k=3):
        assert hit.text.strip()
        assert hit.source.strip()
