import logging

from langchain_core.tools import tool

from app.ai_assistant.university_knowladge.knowledge_service import get_knowledge_service, join_passages
from app.ai_assistant_langchain.agent_schemas import (
    CompareProgramsToolInput,
    FindPersonToolInput,
    FindPlaceToolInput,
    RetrieverToolInput,
)
from app.ai_assistant_langchain.graphs.compare_programs.graph import get_compare_programs_graph
from app.api.v1.places.places_repository import open_places_repository
from app.api.v1.professors.professors_repository import open_professors_repository

logger = logging.getLogger(__name__)

# How many programmes one comparison covers. Not a schema constraint: ``compare_programs`` is
# ``return_direct``, so a list rejected by pydantic would reach the applicant as its error text.
MAX_PROGRAMS = 5


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


@tool(args_schema=CompareProgramsToolInput, return_direct=True)
async def compare_programs(programs: list[str]) -> str:
    """Compare named study programmes side by side — focus, courses, duration, admission
    requirements, tuition and career outcomes.

    Use this when the message weighs programmes against each other: "Economics or Business
    Analytics?", "what is the difference between Data Science and Cybersecurity?", "Economics vs
    Business Analytics vs Data Science", "which of them is cheaper?". It looks all of them up at
    once and returns the finished side-by-side answer, so it replaces the `retriever` calls you
    would otherwise make — do not call `retriever` for any of them, before this tool or after it.

    Pass every programme the applicant named, in the order they named them: two, three, up to five.
    A question about a single programme goes to `retriever`, however much detail it asks for, and so
    does a comparison of anything that is not a programme — scholarships, faculties, campus places.

    What this returns is the finished answer, and the applicant reads it exactly as it comes back:
    the turn ends here, and there is no later step in which you could rewrite it, shorten it or add
    to it. So call it once you have all the programme names — and not before.
    """
    # Case-insensitively unique, in the order given. Two names for one programme is not a comparison:
    # each duplicate would search the same thing again and take a second block in a prompt whose
    # every rule is about keeping the programmes apart.
    unique: dict[str, str] = {}
    for program in programs:
        unique.setdefault(program.strip().casefold(), program.strip())

    named = [program for program in unique.values() if program]
    logger.info('CompareProgrammes %s', ' vs '.join(repr(program) for program in named) or '(nothing named)')

    # Answered here rather than by a ``min_length`` on the schema: this tool is ``return_direct``, so
    # a rejected argument list would reach the applicant as pydantic's error text.
    if len(named) < 2:
        if named:
            return f'{named[0]} is one programme, not two. Which other one should I compare it with?'
        return 'Which programmes would you like me to compare?'

    # Every extra programme is another embedding, another search and another block in the merge
    # prompt, and a side-by-side answer stops being readable long before the cost stops growing.
    compared, dropped = named[:MAX_PROGRAMS], named[MAX_PROGRAMS:]

    graph = get_compare_programs_graph()
    state = await graph.ainvoke({'programs': compared})
    comparison: str = state['comparison']

    if not dropped:
        return comparison

    return (
        f'{comparison}\n\nI compared the first {MAX_PROGRAMS} — {", ".join(dropped)} '
        f'{"is" if len(dropped) == 1 else "are"} not in here. Ask again for '
        f'{"it" if len(dropped) == 1 else "them"} and I will cover '
        f'{"it" if len(dropped) == 1 else "them"}.'
    )


ASSISTANT_TOOLS = [retriever, find_person, find_place, compare_programs]


DIRECT_ANSWER_TOOLS = frozenset(item.name for item in ASSISTANT_TOOLS if item.return_direct)
