"""University knowledge base as a service: ingest PDFs (extract -> split -> embed -> upsert)
and search, all over Qdrant. Every collaborator — the vector store, the dense + sparse
embedders, the text splitter and the PDF loader — is injected, so each can be swapped or mocked
independently.

Search is hybrid: the query is embedded both densely (semantic) and sparsely (BM25 keyword) and
Qdrant fuses the two with RRF (see ``QdrantService.hybrid_search``). The fused candidate pool is
then reranked with MMR (Maximal Marginal Relevance) so near-duplicate chunks don't crowd out the
result set (see ``_mmr``)."""

import asyncio
import logging
import uuid

import numpy as np
from qdrant_client import models
from qdrant_client.models import PointStruct, ScoredPoint

from app.ai_assistant.university_knowladge.embedder import get_embedder, OpenAIEmbedder
from app.ai_assistant.university_knowladge.pdf_loader import PdfLoader
from app.ai_assistant.university_knowladge.sparse_embedder import get_sparse_embedder, SparseEmbedder
from app.ai_assistant.university_knowladge.structured.loader import get_structured_loader, StructuredDocumentLoader
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

NO_RESULTS = 'No relevant information was found in the university knowledge base.'


def join_passages(hits: list[ScoredPoint]) -> str:
    return '\n\n'.join(hit.payload['text'] for hit in hits if hit.payload) or NO_RESULTS


class KnowledgeService:
    def __init__(
        self,
        qdrant: QdrantService,
        embedder: OpenAIEmbedder,
        sparse_embedder: SparseEmbedder,
        splitter: TextSplitter,
        pdf_loader: PdfLoader,
        structured_loader: StructuredDocumentLoader,
        settings: Settings,
    ):
        self._qdrant = qdrant
        self._embedder = embedder
        self._sparse_embedder = sparse_embedder
        self._splitter = splitter
        self._pdf_loader = pdf_loader
        self._structured_loader = structured_loader
        self._min_score = settings.SEARCH_MIN_SCORE
        self._mmr_lambda = settings.SEARCH_MMR_LAMBDA
        self._mmr_fetch_mult = settings.SEARCH_MMR_FETCH_MULT

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

        payloads = [
            {'text': chunk, 'source': source, 'chunk': index, 'doc_type': doc_type}
            for index, chunk in enumerate(chunks)
        ]
        await self._embed_and_upsert(chunks, payloads, source)

        logger.info('Ingested %d chunk(s) from %r', len(chunks), source)
        return len(chunks)

    async def ingest_pdf_structured(self, data: bytes, source: str, doc_type: str) -> int:
        """Same contract as :meth:`ingest_pdf`, but chunked by the structure-aware pipeline
        (``structured/``) instead of by flat character splitting: the PDF is parsed into typed
        elements, tables get an LLM summary prepended and images an LLM description, and tables
        and images are then emitted as atomic chunks while text is split within its section.

        The payload keeps every key :meth:`ingest_pdf` writes — ``text``, ``source``, ``chunk``,
        ``doc_type`` — so both chunkings are searchable through the same code path and the default
        ``doc_type`` filter in :meth:`search`. ``Chunk.index`` is stored as ``chunk`` for that
        reason. The three structure facets (``element_type``, ``page``, ``section``) are added on
        top; nothing filters on them yet, but they are what a citation needs."""
        await self._qdrant.ensure_collection()

        chunks = await self._structured_loader.load(data, source)
        if not chunks:
            return 0

        texts = [chunk.text for chunk in chunks]
        payloads = [
            {
                'text': chunk.text,
                'source': chunk.source,
                'chunk': chunk.index,
                'doc_type': doc_type,
                'element_type': chunk.element_type,
                'page': chunk.page,
                'section': chunk.section,
            }
            for chunk in chunks
        ]
        await self._embed_and_upsert(texts, payloads, source)

        logger.info('Ingested %d structured chunk(s) from %r', len(chunks), source)
        return len(chunks)

    async def _embed_and_upsert(self, texts: list[str], payloads: list[dict], source: str) -> None:
        for start in range(0, len(texts), _EMBED_BATCH_SIZE):
            batch = texts[start : start + _EMBED_BATCH_SIZE]
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
                    payload=payloads[start + j],
                )
                for j in range(len(batch))
            ]
            await self._qdrant.upsert(points)

    async def search(
        self,
        query: str,
        limit: int = 3,
        source: str | None = None,
        doc_type: str | None = 'general',
    ) -> list[ScoredPoint]:
        """The chunks most relevant to ``query`` (empty if none clear the dense floor). The query
        is embedded both densely and sparsely and fused with RRF; an optional ``source`` /
        ``doc_type`` filter scopes the search to matching documents.

        Three rather than four: across the evaluation goldens the fourth chunk never adds coverage
        and costs a quarter of the returned text, which is dead weight in the answer prompt."""
        dense_vector = (await self._embedder.embed([query]))[0]
        sparse_vector = await self._sparse_embedder.embed_query(query)

        hits = await self._qdrant.hybrid_search(
            dense=dense_vector,
            sparse=sparse_vector,
            limit=limit,
            query_filter=self._build_filter(source, doc_type),
            score_threshold=self._min_score,
        )
        self._log_hits(query, hits)
        return hits

    async def search_mmr(
        self,
        query: str,
        limit: int = 3,
        source: str | None = None,
        doc_type: str | None = 'general',
    ) -> list[ScoredPoint]:
        """Like :meth:`search`, but MMR-reranked. Hybrid search first fetches a larger candidate
        pool (``limit * SEARCH_MMR_FETCH_MULT``); MMR then reranks it down to ``limit`` so the
        results stay relevant *and* diverse instead of returning several paraphrases of the same
        passage.

        Chunks are stored with an overlap, so plain relevance ranking readily fills the result set
        with adjacent chunks that repeat each other's text; MMR spends those slots on genuinely new
        passages instead. Returns the hits themselves — same shape as :meth:`search`, so callers
        can swap one for the other."""
        dense_vector = (await self._embedder.embed([query]))[0]
        sparse_vector = await self._sparse_embedder.embed_query(query)

        candidates = await self._qdrant.hybrid_search(
            dense=dense_vector,
            sparse=sparse_vector,
            limit=limit * self._mmr_fetch_mult,
            query_filter=self._build_filter(source, doc_type),
            score_threshold=self._min_score,
            with_vectors=True,
        )
        hits = self._rerank_mmr(dense_vector, candidates, limit)
        self._log_hits(query, hits)
        return hits

    @staticmethod
    def _log_hits(query: str, hits: list[ScoredPoint]) -> None:
        if not hits:
            logger.info('Retriever hits (0) for %r', query)
            return

        lines = []
        for hit in hits:
            payload = hit.payload or {}
            kind = payload.get('element_type', '-')
            page = payload.get('page', '-')
            section = str(payload.get('section') or '-')[:34]
            source, chunk = payload.get('source', '-'), payload.get('chunk', '-')
            lines.append(f'  {hit.score:.3f}  {kind:<6} p{page:<4} {section:<34} {source}#{chunk}')

        logger.info('Retriever hits (%d) for %r:\n%s', len(hits), query, '\n'.join(lines))

    def _rerank_mmr(self, query_vector: list[float], candidates: list[ScoredPoint], limit: int) -> list[ScoredPoint]:
        """Reorder ``candidates`` by MMR and keep the top ``limit``. Candidates missing their dense
        vector (e.g. an older point stored without one) are dropped from the diversity step."""
        embedded = [c for c in candidates if c.vector is not None]
        if len(embedded) <= 1:
            return embedded[:limit]

        cand_vectors = [c.vector[DENSE_VECTOR] for c in embedded]
        order = self._mmr(query_vector, cand_vectors, k=limit, lambda_mult=self._mmr_lambda)
        return [embedded[i] for i in order]

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

    def _mmr(
        self,
        query_vector: list[float],
        candidate_vectors: list[list[float]],
        k: int,
        lambda_mult: float,
    ) -> list[int]:
        """Maximal Marginal Relevance: greedily pick indices into ``candidate_vectors`` that balance
        relevance to the query against novelty versus the already-picked chunks. Returns up to ``k``
        indices in selection order.

        Each step maximises ``lambda * sim(query, c) - (1 - lambda) * max sim(c, already_selected)``,
        where ``sim`` is cosine similarity. ``lambda_mult`` in ``[0, 1]`` slides from pure diversity
        (0) to pure relevance (1). Docs: Carbonell & Goldstein, 1998.
        """
        candidates = self._normalize(np.asarray(candidate_vectors, dtype=np.float64))
        query = self._normalize(np.asarray([query_vector], dtype=np.float64))[0]

        relevance = candidates @ query  # cosine sim of each candidate to the query
        # Pairwise cosine sim between candidates; column j reused as "sim to selected j" below.
        pairwise = candidates @ candidates.T

        k = min(k, len(candidate_vectors))
        selected: list[int] = [int(np.argmax(relevance))]
        while len(selected) < k:
            redundancy = pairwise[:, selected].max(axis=1)  # closeness to the nearest selected chunk
            scores = lambda_mult * relevance - (1.0 - lambda_mult) * redundancy
            scores[selected] = -np.inf  # never re-pick
            selected.append(int(np.argmax(scores)))
        return selected

    @staticmethod
    def _normalize(matrix: np.ndarray) -> np.ndarray:  # type: ignore[explicit-any]
        """Row-normalize so dot products read as cosine similarity; zero rows are left untouched."""
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        return matrix / np.where(norms == 0.0, 1.0, norms)


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
        structured_loader=get_structured_loader(),
        settings=settings,
    )
