"""University knowledge base as a service: ingest PDFs (extract -> split -> embed -> upsert)
and search, all over Qdrant. Every collaborator — the vector store, the dense + sparse
embedders, the text splitter and the PDF loader — is injected, so each can be swapped or mocked
independently.

Search is hybrid: the query is embedded both densely (semantic) and sparsely (BM25 keyword) and
Qdrant fuses the two with RRF (see ``QdrantService.hybrid_search``)."""

import asyncio
import logging
import uuid

from qdrant_client import models
from qdrant_client.models import PointStruct

from app.ai_assistant.university_knowladge.embedder import get_embedder, OpenAIEmbedder
from app.ai_assistant.university_knowladge.pdf_loader import PdfLoader
from app.ai_assistant.university_knowladge.sparse_embedder import get_sparse_embedder, SparseEmbedder
from app.ai_assistant.university_knowladge.text_splitter import RecursiveTextSplitter, TextSplitter
from app.ai_assistant.university_knowladge.vector_store import (
    DENSE_VECTOR,
    get_qdrant_client,
    QdrantService,
    SPARSE_VECTOR,
)
from app.settings import get_settings, Settings

logger = logging.getLogger(__name__)

# Stable namespace so re-ingesting the same source updates chunks in place instead of duplicating.
_ID_NAMESPACE = uuid.UUID('a1b2c3d4-0000-0000-0000-000000000000')

# Chunks per embedding/upsert request. Keeps a large document within the embedding API's
# per-request limits (batch size + token budget) and bounds how many vectors are held at once.
_EMBED_BATCH_SIZE = 100


class KnowledgeService:
    def __init__(
        self,
        qdrant: QdrantService,
        embedder: OpenAIEmbedder,
        sparse_embedder: SparseEmbedder,
        splitter: TextSplitter,
        pdf_loader: PdfLoader,
        settings: Settings,
    ):
        self._qdrant = qdrant
        self._embedder = embedder
        self._sparse_embedder = sparse_embedder
        self._splitter = splitter
        self._pdf_loader = pdf_loader
        self._min_score = settings.SEARCH_MIN_SCORE

    async def ingest_pdf(self, data: bytes, source: str, doc_type: str) -> int:
        """Extract, chunk, embed and store a PDF. Returns the number of chunks written.
        ``source`` labels the origin (e.g. the filename) and makes chunk ids deterministic;
        ``doc_type`` (e.g. ``"program" | "policy" | "general"``) is an optional filter facet.

        Chunks are embedded and upserted in batches, so a large document stays within the
        embedding API's per-request limits and never holds every vector in memory at once. Each
        point carries **both** named vectors (dense + sparse)."""
        await self._qdrant.ensure_collection()

        text = self._pdf_loader.load(data)
        chunks = self._splitter.split(text)
        if not chunks:
            return 0

        for start in range(0, len(chunks), _EMBED_BATCH_SIZE):
            batch = chunks[start : start + _EMBED_BATCH_SIZE]
            # Independent (dense = OpenAI network call, sparse = CPU in a worker thread), so run
            # both concurrently in a task group and overlap the latencies instead of paying them
            # in sequence.
            async with asyncio.TaskGroup() as tg:
                dense_task = tg.create_task(self._embedder.embed(batch))
                sparse_task = tg.create_task(self._sparse_embedder.embed(batch))

            dense_vectors, sparse_vectors = dense_task.result(), sparse_task.result()
            points = [
                PointStruct(
                    id=str(uuid.uuid5(_ID_NAMESPACE, f'{source}:{start + j}')),
                    vector={DENSE_VECTOR: dense_vectors[j], SPARSE_VECTOR: sparse_vectors[j]},
                    payload={'text': chunk, 'source': source, 'chunk': start + j, 'doc_type': doc_type},
                )
                for j, chunk in enumerate(batch)
            ]
            await self._qdrant.upsert(points)

        logger.info('Ingested %d chunk(s) from %r', len(chunks), source)
        return len(chunks)

    async def search(
        self,
        query: str,
        limit: int = 4,
        source: str | None = None,
        doc_type: str | None = 'general',
    ) -> str:
        """Joined text of the chunks most relevant to ``query`` (empty if none clear the dense
        floor). The query is embedded both densely and sparsely and fused with RRF; an optional
        ``source`` / ``doc_type`` filter scopes the search to matching documents."""
        dense_vector = (await self._embedder.embed([query]))[0]
        sparse_vector = await self._sparse_embedder.embed_query(query)

        hits = await self._qdrant.hybrid_search(
            dense=dense_vector,
            sparse=sparse_vector,
            limit=limit,
            query_filter=self._build_filter(source, doc_type),
            score_threshold=self._min_score,
        )
        return '\n\n'.join(h.payload['text'] for h in hits if h.payload)

    @staticmethod
    def _build_filter(source: str | None, doc_type: str | None) -> models.Filter | None:
        """Build a Qdrant ``must`` filter from the optional scoping facets, or ``None`` if unscoped.
        Docs: https://qdrant.tech/documentation/concepts/filtering/"""
        conditions = []
        if source:
            conditions.append(models.FieldCondition(key='source', match=models.MatchValue(value=source)))
        if doc_type:
            conditions.append(models.FieldCondition(key='doc_type', match=models.MatchValue(value=doc_type)))
        return models.Filter(must=conditions) if conditions else None


def get_knowledge_service() -> KnowledgeService:
    """Default wiring of the knowledge service from the cached singletons. Construct
    ``KnowledgeService`` directly (with fakes) in tests."""
    settings = get_settings()
    return KnowledgeService(
        qdrant=QdrantService(get_qdrant_client(), settings),
        embedder=get_embedder(),
        sparse_embedder=get_sparse_embedder(),
        splitter=RecursiveTextSplitter(),
        pdf_loader=PdfLoader(),
        settings=settings,
    )
