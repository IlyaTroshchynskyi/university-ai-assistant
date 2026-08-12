"""Sparse (keyword/BM25) embeddings via FastEmbed, mirroring ``OpenAIEmbedder``. A cached
``SparseTextEmbedding`` model singleton (like the other factories) plus a ``SparseEmbedder``
service that takes the model as an injected dependency, so it can be shared and swapped/mocked
in tests.

Sparse vectors store only non-zero ``{index: weight}`` pairs and match on exact tokens (program
codes, numbers, names) that dense embeddings blur away. FastEmbed produces them locally (no API
cost) via the ONNX ``Qdrant/bm25`` model. The document side emits raw term weights; Qdrant
applies the IDF half of BM25 server-side (see the collection's ``Modifier.IDF`` in vector_store).

Docs:
- Sparse vectors:   https://qdrant.tech/documentation/concepts/vectors/#sparse-vectors
- FastEmbed + BM25: https://qdrant.tech/documentation/fastembed/fastembed-splade/
- BM25 / IDF idea:  https://qdrant.tech/articles/bm25-fusion/
"""

import asyncio
from functools import lru_cache

from fastembed import SparseEmbedding, SparseTextEmbedding
from qdrant_client import models

from app.settings import get_settings


class SparseEmbedder:
    """Wraps a FastEmbed ``SparseTextEmbedding``. FastEmbed is synchronous (CPU/ONNX), so each
    call is pushed to a worker thread to keep the event loop free."""

    def __init__(self, model: SparseTextEmbedding):
        self._model = model

    async def embed(self, texts: list[str]) -> list[models.SparseVector]:
        """Document-side vectors (one per text), used at ingest time."""
        return await asyncio.to_thread(self._embed_docs, texts)

    async def embed_query(self, text: str) -> models.SparseVector:
        """Query-side vector. BM25 weights a query differently from a document, so this uses
        FastEmbed's dedicated ``query_embed`` rather than ``embed``."""
        return await asyncio.to_thread(self._embed_query, text)

    def _embed_docs(self, texts: list[str]) -> list[models.SparseVector]:
        return [self._to_sparse_vector(e) for e in self._model.embed(texts)]

    def _embed_query(self, text: str) -> models.SparseVector:
        # A single query yields exactly one sparse embedding.
        return [self._to_sparse_vector(e) for e in self._model.query_embed(text)][0]

    @staticmethod
    def _to_sparse_vector(embedding: SparseEmbedding) -> models.SparseVector:
        """Convert a FastEmbed ``SparseEmbedding`` (numpy indices/values) into the Qdrant type."""
        return models.SparseVector(indices=embedding.indices.tolist(), values=embedding.values.tolist())


@lru_cache
def get_sparse_model() -> SparseTextEmbedding:
    """The FastEmbed model singleton. Built once (downloads/loads the ONNX weights on first
    use) and cached, like ``get_openai_client``."""
    return SparseTextEmbedding(model_name=get_settings().SPARSE_MODEL)


def get_sparse_embedder() -> SparseEmbedder:
    return SparseEmbedder(get_sparse_model())
