"""Retrieval with MMR diversification and mandatory citation metadata."""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Any

from core.config import get_settings
from rag.embeddings import cosine, get_embedder
from rag.store import SearchHit, VectorStore, get_store


@dataclass
class RetrievedChunk:
    id: str
    text: str
    score: float
    title: str
    source: str
    url: str
    authors: str = ""
    published: str = ""

    @property
    def citation(self) -> str:
        bits = [self.title]
        if self.published:
            bits.append(self.published)
        label = " — ".join(b for b in bits if b)
        return f"[{label} · {self.source}]" + (f"({self.url})" if self.url else "")

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "title": self.title,
            "source": self.source,
            "url": self.url,
            "authors": self.authors,
            "published": self.published,
            "score": round(self.score, 4),
            "text": self.text,
            "citation": self.citation,
        }


class Retriever:
    """Query the vector store and return citation-carrying chunks."""

    def __init__(self, store: VectorStore | None = None) -> None:
        self.store = store or get_store()
        self.settings = get_settings()

    def search(
        self,
        query: str,
        k: int | None = None,
        *,
        source_filter: str | None = None,
        diversify: bool = True,
        min_score: float = 0.0,
    ) -> list[RetrievedChunk]:
        top_k = k or self.settings.rag_top_k
        pool = self.store.search(
            query, k=max(top_k * 3, top_k),
            where={"source": source_filter} if source_filter else None,
        )
        pool = [h for h in pool if h.score >= min_score]
        pool = _prefer_reference_sources(pool)
        selected = _mmr(query, pool, top_k) if diversify else pool[:top_k]
        return [_to_chunk(h) for h in selected]

    def context_block(self, chunks: list[RetrievedChunk], max_chars: int = 6000) -> str:
        """Numbered evidence block for the Literature agent prompt."""
        lines, used = [], 0
        for i, chunk in enumerate(chunks, start=1):
            entry = f"[{i}] {chunk.title} ({chunk.source})\n{chunk.text}\nURL: {chunk.url or 'n/a'}"
            if used + len(entry) > max_chars:
                break
            lines.append(entry)
            used += len(entry)
        return "\n\n".join(lines) or "(no literature retrieved)"

    def stats(self) -> dict[str, Any]:
        return self.store.stats()


# Definitional passages (NASA/IAU/CNEOS reference text and the curated corpus)
# answer "what exactly is X" questions far better than a random paper abstract.
# The corpus is dominated by arXiv abstracts by volume, so without this nudge the
# reference passages get buried.
_REFERENCE_MARKERS = ("nasa", "iau", "cneos", "jpl", "esa", "noaa", "standard",
                      "reference", "anthropic", "kopparapu")
_REFERENCE_BOOST = 0.12


def _prefer_reference_sources(hits: list[SearchHit]) -> list[SearchHit]:
    for hit in hits:
        source = str((hit.metadata or {}).get("source", "")).lower()
        if any(marker in source for marker in _REFERENCE_MARKERS):
            hit.score += _REFERENCE_BOOST
    return sorted(hits, key=lambda h: h.score, reverse=True)


def _to_chunk(hit: SearchHit) -> RetrievedChunk:
    meta = hit.metadata or {}
    return RetrievedChunk(
        id=hit.id,
        text=hit.text,
        score=hit.score,
        title=str(meta.get("title", "untitled")),
        source=str(meta.get("source", "unknown")),
        url=str(meta.get("url", "")),
        authors=str(meta.get("authors", "")),
        published=str(meta.get("published", "")),
    )


def _mmr(query: str, hits: list[SearchHit], k: int, lam: float = 0.7) -> list[SearchHit]:
    """Maximal Marginal Relevance: relevance minus redundancy."""
    if len(hits) <= k:
        return hits
    embedder = get_embedder()
    vectors = embedder.encode([h.text for h in hits])
    selected: list[int] = []
    candidates = list(range(len(hits)))
    while candidates and len(selected) < k:
        best_index, best_value = candidates[0], -1e9
        for index in candidates:
            redundancy = max(
                (cosine(vectors[index], vectors[s]) for s in selected), default=0.0
            )
            value = lam * hits[index].score - (1 - lam) * redundancy
            if value > best_value:
                best_index, best_value = index, value
        selected.append(best_index)
        candidates.remove(best_index)
    return [hits[i] for i in selected]


_RETRIEVER: Retriever | None = None
_LOCK = threading.Lock()


def get_retriever() -> Retriever:
    global _RETRIEVER
    with _LOCK:
        if _RETRIEVER is None:
            _RETRIEVER = Retriever()
        return _RETRIEVER
