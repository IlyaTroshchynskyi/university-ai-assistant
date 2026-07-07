"""Async Qdrant access: a cached client singleton plus a thin read/write service over one
collection. The client is a cached singleton (like ``build_llm`` / ``get_settings``); the
service takes it as an injected dependency, so it can be shared and swapped in tests."""

from functools import lru_cache

from qdrant_client import AsyncQdrantClient
from qdrant_client.models import PointStruct, ScoredPoint

from app.settings import get_settings, Settings


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
        self._collection = get_settings().QDRANT_COLLECTION

    async def upsert(self, points: list[PointStruct]) -> None:
        """Write (insert or update) points — each an id + vector + payload — into the collection."""
        await self._client.upsert(self._collection, points=points)

    async def search(self, query_vector: list[float], limit: int = 5) -> list[ScoredPoint]:
        """Read the ``limit`` nearest points to ``query_vector`` (payloads included)."""
        response = await self._client.query_points(
            self._collection,
            query=query_vector,
            limit=limit,
            with_payload=True,
        )
        return response.points
