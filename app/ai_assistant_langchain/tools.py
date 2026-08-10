import logging

from langchain_core.tools import tool

from app.ai_assistant.university_knowladge.knowledge_service import get_knowledge_service, join_passages
from app.ai_assistant_langchain.agent_schemas import RetrieverToolInput

logger = logging.getLogger(__name__)


@tool(args_schema=RetrieverToolInput)
async def retriever(query: str) -> str:
    """Retrieve relevant passages for the query.

    'General-purpose knowledge base about the university: programs, courses, admissions, '
    'tuition, scholarships, deadlines, policies and other free-form information. Pass a '
    'natural-language query and it returns the most relevant context passages. Within a '
    'single program, one query with the program name returns all of that program at once '
    '— you do not need to query each topic (tuition, courses, …) separately. Facts from '
    'another section are a different matter: scholarship eligibility, fees, deadlines and '
    'policies are not pulled in by a query about a program, and need a query of their own.'
    """
    logger.info('Retriever query: %r', query)
    return join_passages(await get_knowledge_service().search(query))
