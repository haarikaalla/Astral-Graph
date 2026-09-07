"""Corpus ingestion for the RAG layer.

Sources
-------
1. **arXiv astro-ph** abstracts via the public Atom export API (no key required).
2. **HuggingFace astronomy dataset** — configurable; defaults to a small
   astronomy QA/abstract set and degrades gracefully when offline.
3. **Bundled seed corpus** — ~30 curated, citation-carrying astronomy reference
   passages so retrieval works with zero network access (used by the test suite).

CLI
---
``python -m rag.ingest --arxiv 400 --hf --seed``
``python -m rag.ingest --reset --seed``
"""

from __future__ import annotations

import argparse
import re
import time
from dataclasses import dataclass, field
from typing import Any, Iterable

import httpx

from core.telemetry import get_logger, log_event
from rag.seed_corpus import SEED_DOCUMENTS
from rag.store import get_store

log = get_logger("rag.ingest")

ARXIV_API = "https://export.arxiv.org/api/query"
ARXIV_CATEGORIES = ("astro-ph.EP", "astro-ph.IM", "astro-ph.SR", "astro-ph.GA", "astro-ph.HE")

CHUNK_CHARS = 1100
CHUNK_OVERLAP = 150


@dataclass
class Document:
    id: str
    title: str
    text: str
    source: str
    url: str = ""
    authors: str = ""
    published: str = ""
    categories: str = ""
    extra: dict[str, Any] = field(default_factory=dict)


# --------------------------------------------------------------------------- #
# Chunking
# --------------------------------------------------------------------------- #


def chunk_text(text: str, size: int = CHUNK_CHARS, overlap: int = CHUNK_OVERLAP) -> list[str]:
    clean = re.sub(r"\s+", " ", text or "").strip()
    if len(clean) <= size:
        return [clean] if clean else []
    chunks, start = [], 0
    while start < len(clean):
        end = min(len(clean), start + size)
        if end < len(clean):
            boundary = clean.rfind(". ", start + int(size * 0.5), end)
            if boundary != -1:
                end = boundary + 1
        chunks.append(clean[start:end].strip())
        if end >= len(clean):
            break
        start = max(end - overlap, start + 1)
    return [c for c in chunks if c]


def index_documents(documents: Iterable[Document]) -> int:
    store = get_store()
    texts: list[str] = []
    metadatas: list[dict[str, Any]] = []
    ids: list[str] = []
    for doc in documents:
        for i, chunk in enumerate(chunk_text(doc.text)):
            ids.append(f"{doc.id}::{i}")
            texts.append(f"{doc.title}\n\n{chunk}" if i == 0 else chunk)
            metadatas.append(
                {
                    "doc_id": doc.id,
                    "title": doc.title,
                    "source": doc.source,
                    "url": doc.url,
                    "authors": doc.authors,
                    "published": doc.published,
                    "categories": doc.categories,
                    "chunk_index": i,
                    **doc.extra,
                }
            )
    return store.add(texts, metadatas, ids)


# --------------------------------------------------------------------------- #
# arXiv
# --------------------------------------------------------------------------- #


def fetch_arxiv(max_results: int = 300, categories: Iterable[str] = ARXIV_CATEGORIES,
                page_size: int = 100) -> list[Document]:
    """Fetch recent astro-ph abstracts from the arXiv Atom API."""
    try:
        import feedparser
    except ImportError:  # pragma: no cover
        log_event(log, "arxiv_skipped", reason="feedparser not installed")
        return []

    query = " OR ".join(f"cat:{c}" for c in categories)
    documents: list[Document] = []
    fetched = 0
    while fetched < max_results:
        batch = min(page_size, max_results - fetched)
        params = {
            "search_query": query,
            "start": fetched,
            "max_results": batch,
            "sortBy": "submittedDate",
            "sortOrder": "descending",
        }
        try:
            response = httpx.get(ARXIV_API, params=params, timeout=45.0, follow_redirects=True)
            response.raise_for_status()
        except Exception as exc:  # noqa: BLE001
            log_event(log, "arxiv_fetch_failed", error=str(exc)[:200], fetched=fetched)
            break

        feed = feedparser.parse(response.text)
        if not feed.entries:
            break
        for entry in feed.entries:
            arxiv_id = str(entry.get("id", "")).rsplit("/", 1)[-1]
            documents.append(
                Document(
                    id=f"arxiv:{arxiv_id}",
                    title=re.sub(r"\s+", " ", entry.get("title", "")).strip(),
                    text=re.sub(r"\s+", " ", entry.get("summary", "")).strip(),
                    source="arXiv astro-ph",
                    url=entry.get("link", f"https://arxiv.org/abs/{arxiv_id}"),
                    authors=", ".join(a.get("name", "") for a in entry.get("authors", []))[:300],
                    published=str(entry.get("published", ""))[:10],
                    categories=", ".join(t.get("term", "") for t in entry.get("tags", [])),
                )
            )
        fetched += len(feed.entries)
        time.sleep(3.0)  # arXiv asks for >=3s between requests
    log_event(log, "arxiv_fetched", count=len(documents))
    return documents


# --------------------------------------------------------------------------- #
# HuggingFace
# --------------------------------------------------------------------------- #

HF_CANDIDATES = [
    # (repo_id, config, split, text_fields, title_field)
    ("UniverseTBD/arxiv-qa-astro-ph", None, "train", ("question", "answer"), "question"),
    ("charlieoneill/jsalt-astroph-dataset", None, "train", ("abstract", "conclusions"), "filename"),
    ("0xZee/dataset-CoT-Space-Physics-Astrophysics-76", None, "train",
     ("question", "answer"), "question"),
]


def fetch_huggingface(limit: int = 500, repo_id: str | None = None) -> list[Document]:
    """Load an astronomy dataset from the HuggingFace Hub (first one that resolves)."""
    try:
        from datasets import load_dataset
    except ImportError:  # pragma: no cover
        log_event(log, "hf_skipped", reason="datasets not installed")
        return []

    candidates = (
        [(repo_id, None, "train", ("text", "abstract", "answer", "content"), "title")]
        if repo_id
        else HF_CANDIDATES
    )
    for name, config, split, fields, title_field in candidates:
        try:
            dataset = load_dataset(name, config, split=f"{split}[:{limit}]")
        except Exception as exc:  # noqa: BLE001
            log_event(log, "hf_dataset_failed", dataset=name, error=str(exc)[:160])
            continue

        documents: list[Document] = []
        for i, row in enumerate(dataset):
            parts = [str(row[f]) for f in fields if f in row and row[f]]
            if not parts:
                parts = [str(v) for v in row.values() if isinstance(v, str) and len(v) > 40][:2]
            if not parts:
                continue
            title = str(row.get(title_field) or f"{name} #{i}")[:200]
            documents.append(
                Document(
                    id=f"hf:{name}:{i}",
                    title=title,
                    text="\n\n".join(parts),
                    source=f"HuggingFace: {name}",
                    url=f"https://huggingface.co/datasets/{name}",
                    extra={"dataset": name},
                )
            )
        log_event(log, "hf_fetched", dataset=name, count=len(documents))
        return documents

    log_event(log, "hf_all_failed", tried=[c[0] for c in candidates])
    return []


# --------------------------------------------------------------------------- #
# Seed corpus
# --------------------------------------------------------------------------- #


def seed_documents() -> list[Document]:
    return [
        Document(
            id=f"seed:{i}",
            title=doc["title"],
            text=doc["text"],
            source=doc["source"],
            url=doc.get("url", ""),
            extra={"seed": True},
        )
        for i, doc in enumerate(SEED_DOCUMENTS)
    ]


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def run(arxiv: int = 0, hf: int = 0, seed: bool = True, reset: bool = False,
        hf_repo: str | None = None) -> dict[str, Any]:
    store = get_store()
    if reset:
        store.reset()
    total = 0
    if seed:
        total += index_documents(seed_documents())
    if arxiv:
        total += index_documents(fetch_arxiv(arxiv))
    if hf:
        total += index_documents(fetch_huggingface(hf, hf_repo))
    stats = store.stats()
    stats["chunks_added_this_run"] = total
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest astronomy literature into the RAG store")
    parser.add_argument("--arxiv", type=int, default=0, help="number of arXiv astro-ph abstracts")
    parser.add_argument("--hf", type=int, default=0, help="number of HuggingFace rows")
    parser.add_argument("--hf-repo", type=str, default=None, help="explicit HF dataset repo id")
    parser.add_argument("--seed", action="store_true", help="include the bundled seed corpus")
    parser.add_argument("--reset", action="store_true", help="wipe the collection first")
    args = parser.parse_args()

    if not (args.arxiv or args.hf or args.seed):
        args.seed = True
        args.arxiv = 200

    stats = run(arxiv=args.arxiv, hf=args.hf, seed=args.seed, reset=args.reset,
                hf_repo=args.hf_repo)
    print("\nRAG index ready:")
    for key, value in stats.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
