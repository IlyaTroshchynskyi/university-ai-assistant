import logging

from langchain_core.tools import tool

from app.ai_assistant.university_knowladge.knowledge_service import get_knowledge_service
from app.ai_assistant_langchain.agent_schemas import RetrieverToolInput

logger = logging.getLogger(__name__)


@tool(args_schema=RetrieverToolInput)
async def retriever(query: str) -> str:
    """Retrieve relevant passages for the query.

    'General-purpose knowledge base about the university: programs, courses, admissions, '
    'tuition, scholarships, deadlines, policies and other free-form information. Pass a '
    'natural-language query and it returns the most relevant context passages. For a '
    'program, a single query with the program name returns all of its information at '
    'once — you do not need to query each topic (tuition, courses, …) separately.'
    """
    logger.info('Retriever query: %r', query)
    passages = await get_knowledge_service().search(query)
    return passages or 'No relevant information was found in the university knowledge base.'
