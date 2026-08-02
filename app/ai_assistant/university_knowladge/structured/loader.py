"""Façade that runs the structure-aware pipeline end to end: parse a PDF into
typed elements, enrich tables and images concurrently, then chunk. Returns
retrieval-ready :class:`Chunk`s. ``get_structured_loader`` wires the defaults from
the cached OpenAI client and settings, mirroring ``get_embedder``."""

import asyncio
import logging
import os

from app.ai_assistant.university_knowladge.embedder import get_openai_client
from app.ai_assistant.university_knowladge.structured.chunker import ElementChunker
from app.ai_assistant.university_knowladge.structured.enrichment import (
    Enricher,
    TableSummaryEnricher,
    VisionImageEnricher,
)
from app.ai_assistant.university_knowladge.structured.parser import Parser, StructuredPdfParser
from app.ai_assistant.university_knowladge.structured.schemas import Chunk, DocElement
from app.settings import get_settings

logger = logging.getLogger(__name__)


class StructuredDocumentLoader:
    def __init__(
        self,
        parser: Parser,
        table_enricher: Enricher,
        image_enricher: Enricher,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        enrichment_enabled: bool = True,
    ):
        self._parser = parser
        self._table_enricher = table_enricher
        self._image_enricher = image_enricher
        self._chunker = ElementChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)  # Todo make it as param
        self._enrichment_enabled = enrichment_enabled

    async def load(self, data: bytes, source: str) -> list[Chunk]:
        elements = self._parser.parse(data, source)
        try:
            if self._enrichment_enabled:
                await self._enrich(elements)
            return self._chunker.chunk(elements, source)
        finally:
            self._cleanup_images(elements)

    async def _enrich(self, elements: list[DocElement]) -> None:
        """Enrich every table and image concurrently — enrichers mutate the element
        in place, overlapping the independent LLM latencies."""
        async with asyncio.TaskGroup() as tg:
            for element in elements:
                if element.kind == 'table':
                    tg.create_task(self._table_enricher.enrich(element))
                elif element.kind == 'image':
                    tg.create_task(self._image_enricher.enrich(element))

    @staticmethod
    def _cleanup_images(elements: list[DocElement]) -> None:
        for element in elements:
            if element.kind == 'image':
                try:
                    os.unlink(element.raw)
                except OSError:
                    pass


def get_structured_loader(enrichment_enabled: bool | None = None) -> StructuredDocumentLoader:
    """Default wiring from the cached OpenAI client and settings. Pass
    ``enrichment_enabled`` to override the settings default (e.g. the CLI's
    ``--no-enrich``)."""
    settings = get_settings()
    client = get_openai_client()
    enabled = settings.ENRICHMENT_ENABLED if enrichment_enabled is None else enrichment_enabled
    return StructuredDocumentLoader(
        parser=StructuredPdfParser(),
        table_enricher=TableSummaryEnricher(client, settings.TABLE_SUMMARY_MODEL),
        image_enricher=VisionImageEnricher(client, settings.VISION_MODEL),
        chunk_size=settings.STRUCTURED_CHUNK_SIZE,
        chunk_overlap=settings.STRUCTURED_CHUNK_OVERLAP,
        enrichment_enabled=enabled,
    )
