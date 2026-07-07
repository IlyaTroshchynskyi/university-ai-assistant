"""University knowledge base as a service: ingest PDFs (extract -> split -> embed -> upsert)
and search, all over Qdrant. Every collaborator — the vector store, the embedder, the text
splitter and the PDF loader — is injected, so each can be swapped or mocked independently."""

import logging
import uuid

from qdrant_client.models import PointStruct

from app.ai_assistant.university_knowladge.embedder import get_embedder, OpenAIEmbedder
from app.ai_assistant.university_knowladge.pdf_loader import PdfLoader
from app.ai_assistant.university_knowladge.text_splitter import RecursiveTextSplitter, TextSplitter
from app.ai_assistant.university_knowladge.vector_store import get_qdrant_client, QdrantService
from app.settings import get_settings

logger = logging.getLogger(__name__)

# Below this cosine similarity a hit is treated as unrelated (so an off-topic question returns
# nothing rather than the nearest chunk by default).
_MIN_SCORE = 0.3

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
        splitter: TextSplitter,
        pdf_loader: PdfLoader,
    ):
        self._qdrant = qdrant
        self._embedder = embedder
        self._splitter = splitter
        self._pdf_loader = pdf_loader

    async def ingest_pdf(self, data: bytes, source: str) -> int:
        """Extract, chunk, embed and store a PDF. Returns the number of chunks written.
        ``source`` labels the origin (e.g. the filename) and makes chunk ids deterministic.

        Chunks are embedded and upserted in batches, so a large document stays within the
        embedding API's per-request limits and never holds every vector in memory at once."""
        text = self._pdf_loader.load(data)
        chunks = self._splitter.split(text)
        if not chunks:
            return 0

        for start in range(0, len(chunks), _EMBED_BATCH_SIZE):
            batch = chunks[start : start + _EMBED_BATCH_SIZE]
            vectors = await self._embedder.embed(batch)
            points = [
                PointStruct(
                    id=str(uuid.uuid5(_ID_NAMESPACE, f'{source}:{start + j}')),
                    vector=vectors[j],
                    payload={'text': chunk, 'source': source, 'chunk': start + j},
                )
                for j, chunk in enumerate(batch)
            ]
            await self._qdrant.upsert(points)

        logger.info('Ingested %d chunk(s) from %r', len(chunks), source)
        return len(chunks)

    async def search(self, query: str, limit: int = 4) -> str:
        """Joined text of the chunks most relevant to ``query`` (empty if none clear threshold)."""
        query_vector = (await self._embedder.embed([query]))[0]
        hits = await self._qdrant.search(query_vector, limit=limit)
        return '\n\n'.join(h.payload['text'] for h in hits if h.payload and h.score >= _MIN_SCORE)


def get_knowledge_service() -> KnowledgeService:
    """Default wiring of the knowledge service from the cached singletons. Construct
    ``KnowledgeService`` directly (with fakes) in tests."""
    settings = get_settings()
    return KnowledgeService(
        qdrant=QdrantService(get_qdrant_client(), settings),
        embedder=get_embedder(),
        splitter=RecursiveTextSplitter(),
        pdf_loader=PdfLoader(),
    )
