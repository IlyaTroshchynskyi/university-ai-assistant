import logging

from langchain_core.tools import tool

from app.ai_assistant.university_knowladge.knowledge_service import get_knowledge_service, join_passages
from app.ai_assistant_langchain.agent_schemas import FindPersonToolInput, FindPlaceToolInput, RetrieverToolInput
from app.api.v1.places.places_repository import open_places_repository
from app.api.v1.professors.professors_repository import open_professors_repository

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


@tool(args_schema=FindPersonToolInput)
async def find_person(name: str) -> list[dict] | str:
    """Look up a specific professor or staff member BY THEIR NAME.

    Returns their title, faculty, office/room, email and office hours. Use this when the user asks
    about a named person, e.g. "What is Professor Ivan's email?" or "When are Peter's office
    hours?". The knowledge base does not hold staff records — a name goes here, not to `retriever`.
    """
    async with open_professors_repository() as repo:
        professors = await repo.find_professors_by_name(name)

    logger.info('FindPerson %r -> %s', name, f'{len(professors)} match(es)' if professors else 'none')
    if not professors:
        return f'No professor found with the name {name!r}.'

    return [professor.model_dump() for professor in professors]


@tool(args_schema=FindPlaceToolInput)
async def find_place(name: str) -> dict | str:
    """Look up a specific campus facility BY ITS NAME.

    Library, cafeteria, gym, admissions office, dormitory and the like. Returns its building, floor
    and opening hours. Use this for questions such as "When does the library open?" or "Where is
    the cafeteria?". Opening hours and locations are not in the knowledge base — they belong here,
    not to `retriever`.
    """
    async with open_places_repository() as repo:
        place = await repo.find_place_by_name(name)

    logger.info('FindPlace %r -> %s', name, place.name if place else 'none')
    if place is None:
        return f'No campus place found with the name {name!r}.'

    return place.model_dump()
