"""Embedding backends.

Primary: ``sentence-transformers/all-MiniLM-L6-v2`` (free, local, 384-d).
Fallback: a deterministic hashed bag-of-words embedder so the pipeline still runs
on machines where torch is unavailable — the eval harness records which backend
was used so results stay honest.
"""

from __future__ import annotations

import hashlib
import math
import re
import threading
from typing import Sequence

from core.config import get_settings
from core.telemetry import get_logger, log_event

log = get_logger("rag.embeddings")

_TOKEN_RE = re.compile(r"[a-z0-9][a-z0-9\-]{1,}")
_FALLBACK_DIM = 512


class Embedder:
    """Wraps whichever embedding backend is available."""

    def __init__(self, model_name: str | None = None) -> None:
        settings = get_settings()
        self.model_name = model_name or settings.embed_model
        self._model = None
        self._lock = threading.Lock()
        self.backend = "pending"

    def _load(self) -> None:
        if self.backend != "pending":
            return
        with self._lock:
            if self.backend != "pending":
                return
            try:
                from sentence_transformers import SentenceTransformer
            except Exception as exc:  # noqa: BLE001
                self.backend = "hashed-bow"
                log_event(
                    log, "embedder_fallback", error=str(exc)[:200], backend=self.backend,
                    hint="pip install sentence-transformers for semantic embeddings",
                )
                return

            # A cached model must not be defeated by a flaky network. The first
            # attempt may reach out to the Hub for an update check; if that fails
            # (offline, proxy, SSL interception) retry strictly from local cache
            # before degrading, because the fallback embedder has a *different*
            # dimension and would invalidate an already-built index.
            online_error: str | None = None
            for local_only in (False, True):
                try:
                    self._model = SentenceTransformer(
                        self.model_name, local_files_only=local_only
                    )
                    self.backend = "sentence-transformers"
                    log_event(
                        log, "embedder_loaded", model=self.model_name,
                        backend=self.backend, from_cache=local_only,
                        recovered_from=online_error if local_only else None,
                    )
                    return
                except Exception as exc:  # noqa: BLE001
                    if not local_only:
                        online_error = str(exc)[:200]

            self.backend = "hashed-bow"
            log_event(
                log, "embedder_fallback", error=online_error, backend=self.backend,
                dimension=_FALLBACK_DIM,
                hint="model unavailable online and not in the local cache; "
                     "an index built with all-MiniLM-L6-v2 will not be readable",
            )

    @property
    def dimension(self) -> int:
        self._load()
        if self._model is not None:
            # Renamed in sentence-transformers 5.x; accept either spelling.
            getter = (getattr(self._model, "get_embedding_dimension", None)
                      or self._model.get_sentence_embedding_dimension)
            return int(getter())
        return _FALLBACK_DIM

    def encode(self, texts: Sequence[str]) -> list[list[float]]:
        self._load()
        if not texts:
            return []
        if self._model is not None:
            vectors = self._model.encode(
                list(texts), normalize_embeddings=True, show_progress_bar=False
            )
            return [list(map(float, v)) for v in vectors]
        return [_hashed_embedding(t) for t in texts]

    def encode_one(self, text: str) -> list[float]:
        return self.encode([text])[0]


def _hashed_embedding(text: str, dim: int = _FALLBACK_DIM) -> list[float]:
    """Deterministic hashed bag-of-words + bigrams, L2-normalised."""
    vector = [0.0] * dim
    tokens = _TOKEN_RE.findall(text.lower())
    grams = tokens + [f"{a}_{b}" for a, b in zip(tokens, tokens[1:])]
    for gram in grams:
        digest = hashlib.md5(gram.encode("utf-8")).digest()
        index = int.from_bytes(digest[:4], "little") % dim
        sign = 1.0 if digest[4] & 1 else -1.0
        vector[index] += sign
    norm = math.sqrt(sum(v * v for v in vector)) or 1.0
    return [v / norm for v in vector]


_EMBEDDER: Embedder | None = None
_EMBEDDER_LOCK = threading.Lock()


def get_embedder() -> Embedder:
    global _EMBEDDER
    with _EMBEDDER_LOCK:
        if _EMBEDDER is None:
            _EMBEDDER = Embedder()
        return _EMBEDDER


def cosine(a: Sequence[float], b: Sequence[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a)) or 1.0
    nb = math.sqrt(sum(y * y for y in b)) or 1.0
    return dot / (na * nb)
