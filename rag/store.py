"""Vector store abstraction.

Primary backend is **Chroma** (persistent, local, free). If ``chromadb`` cannot be
imported the store degrades to a JSON-backed brute-force cosine index, which is
plenty fast for the ~5-10k chunks this project ingests.
"""

from __future__ import annotations

import json
import threading
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from core.config import get_settings
from core.telemetry import get_logger, log_event
from rag.embeddings import Embedder, cosine, get_embedder

log = get_logger("rag.store")

COLLECTION = "astral_literature"


@dataclass
class SearchHit:
    id: str
    text: str
    score: float
    metadata: dict[str, Any]


class VectorStore:
    """Uniform add/search API over Chroma or the JSON fallback."""

    def __init__(self, collection: str = COLLECTION, embedder: Embedder | None = None) -> None:
        self.settings = get_settings()
        self.collection_name = collection
        self.embedder = embedder or get_embedder()
        self._lock = threading.RLock()
        self._chroma = None
        self._collection = None
        self._fallback: dict[str, dict[str, Any]] = {}
        self._fallback_path = Path(self.settings.chroma_dir) / f"{collection}_fallback.json"
        self.backend = "uninitialised"
        self._init_backend()

    # ---- backend selection ----------------------------------------------
    def _init_backend(self) -> None:
        try:
            import chromadb
            from chromadb.config import Settings as ChromaSettings

            self._chroma = chromadb.PersistentClient(
                path=str(self.settings.chroma_dir),
                settings=ChromaSettings(anonymized_telemetry=False, allow_reset=True),
            )
            self._collection = self._chroma.get_or_create_collection(
                name=self.collection_name, metadata={"hnsw:space": "cosine"}
            )
            stored_dim = self._stored_dimension()
            if stored_dim is not None and stored_dim != self.embedder.dimension:
                # The index was built by a different embedder. Querying it would
                # raise deep inside Chroma on every call, so degrade openly here
                # instead of failing opaquely later.
                raise RuntimeError(
                    f"index embedding dimension {stored_dim} does not match active "
                    f"embedder dimension {self.embedder.dimension}; "
                    f"re-run `python -m rag.ingest` to rebuild it"
                )
            self.backend = "chroma"
        except Exception as exc:  # noqa: BLE001
            self._chroma = None
            self._collection = None
            self.backend = "json-cosine"
            self._load_fallback()
            log_event(log, "chroma_unavailable", error=str(exc)[:200], backend=self.backend)
        log_event(log, "vector_store_ready", backend=self.backend, count=self.count())

    def _stored_dimension(self) -> int | None:
        """Dimension of vectors already persisted, or ``None`` if the index is empty."""
        try:
            if self._collection.count() == 0:
                return None
            vectors = self._collection.peek(limit=1).get("embeddings")
            if vectors is None or len(vectors) == 0:
                return None
            return len(vectors[0])
        except Exception:  # noqa: BLE001
            return None

    def _load_fallback(self) -> None:
        if self._fallback_path.exists():
            try:
                self._fallback = json.loads(self._fallback_path.read_text(encoding="utf-8"))
            except Exception:  # noqa: BLE001
                self._fallback = {}

    def _save_fallback(self) -> None:
        self._fallback_path.parent.mkdir(parents=True, exist_ok=True)
        self._fallback_path.write_text(json.dumps(self._fallback), encoding="utf-8")

    # ---- writes ----------------------------------------------------------
    def add(
        self,
        texts: Sequence[str],
        metadatas: Sequence[dict[str, Any]],
        ids: Sequence[str] | None = None,
        batch_size: int = 128,
    ) -> int:
        if not texts:
            return 0
        ids = list(ids or [uuid.uuid4().hex for _ in texts])
        added = 0
        with self._lock:
            for start in range(0, len(texts), batch_size):
                chunk_texts = list(texts[start : start + batch_size])
                chunk_meta = [_flatten(m) for m in metadatas[start : start + batch_size]]
                chunk_ids = ids[start : start + batch_size]
                vectors = self.embedder.encode(chunk_texts)
                if self.backend == "chroma" and self._collection is not None:
                    self._collection.upsert(
                        ids=chunk_ids, documents=chunk_texts,
                        metadatas=chunk_meta, embeddings=vectors,
                    )
                else:
                    for cid, text, meta, vector in zip(chunk_ids, chunk_texts, chunk_meta, vectors):
                        self._fallback[cid] = {"text": text, "metadata": meta, "vector": vector}
                added += len(chunk_texts)
            if self.backend != "chroma":
                self._save_fallback()
        log_event(log, "chunks_indexed", count=added, backend=self.backend)
        return added

    def reset(self) -> None:
        with self._lock:
            if self.backend == "chroma" and self._chroma is not None:
                try:
                    self._chroma.delete_collection(self.collection_name)
                except Exception:  # noqa: BLE001
                    pass
                self._collection = self._chroma.get_or_create_collection(
                    name=self.collection_name, metadata={"hnsw:space": "cosine"}
                )
            else:
                self._fallback = {}
                self._save_fallback()

    # ---- reads -----------------------------------------------------------
    def count(self) -> int:
        if self.backend == "chroma" and self._collection is not None:
            try:
                return int(self._collection.count())
            except Exception:  # noqa: BLE001
                return 0
        return len(self._fallback)

    def search(
        self, query: str, k: int = 5, where: dict[str, Any] | None = None
    ) -> list[SearchHit]:
        if self.count() == 0:
            return []
        vector = self.embedder.encode_one(query)
        if self.backend == "chroma" and self._collection is not None:
            result = self._collection.query(
                query_embeddings=[vector], n_results=min(k, self.count()),
                where=where or None, include=["documents", "metadatas", "distances"],
            )
            hits: list[SearchHit] = []
            docs = (result.get("documents") or [[]])[0]
            metas = (result.get("metadatas") or [[]])[0]
            dists = (result.get("distances") or [[]])[0]
            ids = (result.get("ids") or [[]])[0]
            for cid, doc, meta, dist in zip(ids, docs, metas, dists):
                hits.append(
                    SearchHit(id=cid, text=doc or "", score=max(0.0, 1.0 - float(dist)),
                              metadata=dict(meta or {}))
                )
            return hits

        scored = []
        for cid, record in self._fallback.items():
            if where and any(record["metadata"].get(key) != value for key, value in where.items()):
                continue
            scored.append(
                SearchHit(id=cid, text=record["text"],
                          score=cosine(vector, record["vector"]), metadata=record["metadata"])
            )
        scored.sort(key=lambda h: h.score, reverse=True)
        return scored[:k]

    def stats(self) -> dict[str, Any]:
        sources: dict[str, int] = {}
        if self.backend != "chroma":
            for record in self._fallback.values():
                key = record["metadata"].get("source", "?")
                sources[key] = sources.get(key, 0) + 1
        elif self._collection is not None:
            try:
                data = self._collection.get(include=["metadatas"], limit=10000)
                for meta in data.get("metadatas") or []:
                    key = (meta or {}).get("source", "?")
                    sources[key] = sources.get(key, 0) + 1
            except Exception:  # noqa: BLE001
                pass
        return {
            "backend": self.backend,
            "embedding_backend": self.embedder.backend,
            "embedding_model": self.embedder.model_name,
            "collection": self.collection_name,
            "chunks": self.count(),
            "chunks_by_source": sources,
            "path": str(self.settings.chroma_dir),
        }


def _flatten(metadata: dict[str, Any]) -> dict[str, Any]:
    """Chroma only accepts scalar metadata values."""
    flat: dict[str, Any] = {}
    for key, value in metadata.items():
        if value is None:
            continue
        if isinstance(value, (str, int, float, bool)):
            flat[key] = value
        elif isinstance(value, (list, tuple)):
            flat[key] = ", ".join(str(v) for v in value)[:800]
        else:
            flat[key] = str(value)[:800]
    return flat


_STORE: VectorStore | None = None
_STORE_LOCK = threading.Lock()


def get_store() -> VectorStore:
    global _STORE
    with _STORE_LOCK:
        if _STORE is None:
            _STORE = VectorStore()
        return _STORE
