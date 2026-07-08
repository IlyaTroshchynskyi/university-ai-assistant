"""Async Qdrant access: a cached client singleton plus a thin read/write service over one
collection. The client is a cached singleton (like ``build_llm`` / ``get_settings``); the
service takes it as an injected dependency, so it can be shared and swapped in tests.

The collection stores two **named** vectors per point — ``dense`` (semantic, OpenAI) and
``sparse`` (keyword/BM25, FastEmbed) — and search fuses them with Reciprocal Rank Fusion (RRF)
so exact tokens and meaning both contribute.

Docs:
- Named vectors:       https://qdrant.tech/documentation/concepts/vectors/#named-vectors
- Sparse vectors/IDF:  https://qdrant.tech/documentation/concepts/vectors/#sparse-vectors
- Hybrid queries/RRF:  https://qdrant.tech/documentation/concepts/hybrid-queries/
- Payload filtering:   https://qdrant.tech/documentation/concepts/filtering/
- Payload indexes:     https://qdrant.tech/documentation/concepts/indexing/#payload-index
"""

from functools import lru_cache

from qdrant_client import AsyncQdrantClient, models
from qdrant_client.models import PointStruct, ScoredPoint

from app.settings import get_settings, Settings

# Named-vector keys used across the collection schema, upserts and queries.
DENSE_VECTOR = 'dense'
SPARSE_VECTOR = 'sparse'


@lru_cache
def get_qdrant_client() -> AsyncQdrantClient:
    """The async Qdrant client, built once and cached. Connection is lazy — no network call
    happens until the first request is made."""
    return AsyncQdrantClient(url=get_settings().QDRANT_URL)


class QdrantService:
    """Reads from and writes to a single Qdrant collection. The client is injected (DI) rather
    than built inside, so callers control its lifecycle and tests can pass a fake."""

    def __init__(self, client: AsyncQdrantClient, settings: Settings):
        self._client = client
        self._collection = settings.QDRANT_COLLECTION

    async def ensure_collection(self) -> None:
        """Create the named dense+sparse collection and the payload indexes if they don't exist
        yet. Idempotent — safe to call before every ingest.

        The sparse vector uses ``Modifier.IDF`` so Qdrant applies the IDF half of BM25 scoring
        server-side (FastEmbed only supplies term frequencies).
        Docs: https://qdrant.tech/documentation/concepts/vectors/#sparse-vectors
        """
        if await self._client.collection_exists(self._collection):
            return

        await self._client.create_collection(
            self._collection,
            vectors_config={
                DENSE_VECTOR: models.VectorParams(size=1536, distance=models.Distance.COSINE),
            },
            sparse_vectors_config={
                SPARSE_VECTOR: models.SparseVectorParams(modifier=models.Modifier.IDF),
            },
        )

        # Docs: https://qdrant.tech/documentation/concepts/indexing/#payload-index
        await self._client.create_payload_index(self._collection, 'source', models.PayloadSchemaType.KEYWORD)
        await self._client.create_payload_index(self._collection, 'doc_type', models.PayloadSchemaType.KEYWORD)

    async def upsert(self, points: list[PointStruct]) -> None:
        await self._client.upsert(self._collection, points=points)

    async def search(
        self,
        query_vector: list[float],
        limit: int = 5,
        score_threshold: float | None = None,
    ) -> list[ScoredPoint]:
        """Dense-only search: the ``limit`` nearest points to ``query_vector`` (payloads
        included). ``score_threshold`` is enforced server-side, so weak matches never come back.
        Docs: https://qdrant.tech/documentation/concepts/search/"""
        response = await self._client.query_points(
            self._collection,
            query=query_vector,
            using=DENSE_VECTOR,
            limit=limit,
            score_threshold=score_threshold,
            with_payload=True,
        )
        return response.points

    async def hybrid_search(
        self,
        dense: list[float],
        sparse: models.SparseVector,
        limit: int = 5,
        query_filter: models.Filter | None = None,
        score_threshold: float | None = None,
    ) -> list[ScoredPoint]:
        """Fuse a dense and a sparse query with RRF and return the top ``limit`` points.

        Each branch is prefetched independently (``limit * 5`` candidates) and Qdrant fuses their
        ranks. ``score_threshold`` is a **cosine** floor, so it goes on the dense prefetch only —
        the fused RRF score (~``1/(60+rank)``) is not comparable to cosine and is left unfiltered.
        Docs: https://qdrant.tech/documentation/concepts/hybrid-queries/
        """
        response = await self._client.query_points(
            self._collection,
            prefetch=[
                models.Prefetch(query=sparse, using=SPARSE_VECTOR, limit=limit * 5),
                models.Prefetch(
                    query=dense,
                    using=DENSE_VECTOR,
                    limit=limit * 5,
                    score_threshold=score_threshold,
                ),
            ],
            query=models.FusionQuery(fusion=models.Fusion.RRF),
            query_filter=query_filter,
            limit=limit,
            with_payload=True,
        )
        return response.points
