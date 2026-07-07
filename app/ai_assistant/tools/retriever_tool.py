import logging

from crewai.tools import BaseTool
from pydantic import BaseModel, Field

from app.ai_assistant.university_knowladge.knowledge_service import get_knowledge_service

logger = logging.getLogger(__name__)


class RetrieverToolInput(BaseModel):
    """Input schema for the RetrieverTool."""

    query: str = Field(
        description="The user's question or search query about the university.",
    )


class RetrieverTool(BaseTool):
    name: str = 'University Knowledge Retriever'
    description: str = (
        'General-purpose knowledge base about the university: programs, courses, admissions, '
        'tuition, scholarships, deadlines, policies and other free-form information. Pass a '
        'natural-language query and it returns the most relevant context passages. For a '
        'program, a single query with the program name returns all of its information at '
        'once — you do not need to query each topic (tuition, courses, …) separately.'
    )
    args_schema: type[BaseModel] = RetrieverToolInput

    async def _run(self, query: str) -> str:
        """Retrieve relevant passages for the query.

        CrewAI detects that this is a coroutine and awaits it, so the async Qdrant search runs
        directly here. The collection is populated by ingesting PDFs via ``POST /documents``.
        """
        logger.info('Retriever query: %r', query)
        passages = await get_knowledge_service().search(query)
        return passages or 'No relevant information was found in the university knowledge base.'
