"""RAG layer: sentence-transformers embeddings + Chroma vector store over
arXiv astro-ph abstracts and a HuggingFace astronomy dataset."""

from rag.embeddings import Embedder, get_embedder
from rag.store import VectorStore, get_store
from rag.retriever import Retriever, RetrievedChunk, get_retriever

__all__ = [
    "Embedder",
    "get_embedder",
    "VectorStore",
    "get_store",
    "Retriever",
    "RetrievedChunk",
    "get_retriever",
]
