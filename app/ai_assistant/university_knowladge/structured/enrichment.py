"""Element enrichers: turn a table into a summarised markdown chunk and an image
into a description plus a transcription of any text baked into it.

Both share an injected ``AsyncOpenAI`` client and degrade gracefully — a failed
LLM call (or an unreadable image) falls back to the raw markdown / a placeholder
and is logged, so one element never aborts a document's ingestion."""

from abc import ABC, abstractmethod
import base64
import logging
from pathlib import Path

from openai import AsyncOpenAI

from app.ai_assistant.university_knowladge.structured.prompts import TABLE_PROMPT, VISION_PROMPT
from app.ai_assistant.university_knowladge.structured.schemas import DocElement

logger = logging.getLogger(__name__)


class Enricher(ABC):
    """Fills an element's ``text`` with enriched content (e.g. a table summary or an
    image description). Behind this interface so the loader depends on the abstraction."""

    @abstractmethod
    async def enrich(self, element: DocElement) -> DocElement: ...


class TableSummaryEnricher(Enricher):
    """Prepends a one-sentence LLM summary to a table's markdown."""

    def __init__(self, client: AsyncOpenAI, model: str):
        self._client = client
        self._model = model

    async def enrich(self, element: DocElement) -> DocElement:
        summary = await self._summarise(element.raw)
        element.text = f'{summary}\n\n{element.raw}' if summary else element.raw
        return element

    async def _summarise(self, markdown: str) -> str | None:
        try:
            response = await self._client.chat.completions.create(
                model=self._model,
                messages=[
                    {'role': 'system', 'content': TABLE_PROMPT},
                    {'role': 'user', 'content': markdown},
                ],
            )
            return (response.choices[0].message.content or '').strip() or None
        except Exception:
            logger.warning('Table summary failed; falling back to raw markdown', exc_info=True)
            return None


class VisionImageEnricher(Enricher):
    """Describes an image and transcribes its embedded text via a vision model."""

    def __init__(self, client: AsyncOpenAI, model: str):
        self._client = client
        self._model = model

    async def enrich(self, element: DocElement) -> DocElement:
        element.text = await self._describe(element.raw, element.page)
        return element

    async def _describe(self, path: str, page: int) -> str:
        try:
            data = Path(path).read_bytes()
        except OSError:
            logger.warning('Image %r unreadable; using placeholder', path, exc_info=True)
            return f'[image on page {page}]'

        b64 = base64.b64encode(data).decode('ascii')
        try:
            response = await self._client.chat.completions.create(
                model=self._model,
                messages=[
                    {
                        'role': 'user',
                        'content': [
                            {'type': 'text', 'text': VISION_PROMPT},
                            {'type': 'image_url', 'image_url': {'url': f'data:image/png;base64,{b64}'}},
                        ],
                    }
                ],
            )
            return (response.choices[0].message.content or '').strip() or f'[image on page {page}]'
        except Exception:
            logger.warning('Vision description failed for %r; using placeholder', path, exc_info=True)
            return f'[image on page {page}]'
