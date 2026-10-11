import asyncio
import base64
from collections.abc import Sequence
from functools import lru_cache
from typing import cast, Protocol, TypeAlias

from langchain.chat_models import init_chat_model
from langchain_core.language_models import LanguageModelInput
from langchain_core.messages import HumanMessage
from langchain_core.runnables import Runnable

from app.ai_assistant_langchain.applicant_documents.prompts import build_extraction_prompt
from app.ai_assistant_langchain.applicant_documents.schemas import DocumentFields, DocumentImage
from app.settings import get_settings

VISION_TIMEOUT_SECONDS = 60

VisionLlm: TypeAlias = Runnable[LanguageModelInput, DocumentFields]


class Extractor(Protocol):
    async def extract(self, image: DocumentImage) -> DocumentFields: ...


class OpenAIVisionExtractor:
    def __init__(self, llm: VisionLlm) -> None:
        self._llm = llm

    async def extract(self, image: DocumentImage) -> DocumentFields:
        message = HumanMessage(
            content=[
                {'type': 'text', 'text': build_extraction_prompt()},
                {
                    'type': 'image',
                    'base64': base64.b64encode(image.content).decode(),
                    'mime_type': image.mime_type.value,
                },
            ]
        )
        return await self._llm.ainvoke([message])


async def extract_document_fields(documents: Sequence[DocumentImage], extractor: Extractor) -> list[DocumentFields]:
    async with asyncio.TaskGroup() as task_group:
        tasks = [task_group.create_task(extractor.extract(image)) for image in documents]
    return [task.result() for task in tasks]


@lru_cache
def get_extractor() -> Extractor:
    settings = get_settings()
    llm = init_chat_model(
        model=settings.VISION_MODEL,
        temperature=0,
        timeout=VISION_TIMEOUT_SECONDS,
        max_retries=2,
        api_key=settings.OPENAI_API_KEY,
        disable_streaming=True,
    ).with_structured_output(DocumentFields)
    return OpenAIVisionExtractor(cast(VisionLlm, llm))
