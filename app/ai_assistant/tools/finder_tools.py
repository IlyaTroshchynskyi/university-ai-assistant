import logging

from crewai.tools import BaseTool
from pydantic import BaseModel, Field

from app.core.dynamodb.university_repository import open_university_repository

logger = logging.getLogger(__name__)


class FindPersonToolInput(BaseModel):
    """Input schema for FindPersonTool."""

    name: str = Field(
        ...,
        description='The full name or first name of the professor or staff member to look up.',
    )


class FindPersonTool(BaseTool):
    name: str = 'Find Professor'
    description: str = (
        'Look up a specific professor or staff member BY THEIR NAME. Returns their title, '
        'faculty, office/room, email and office hours. Use this when the user asks about a '
        'named person, e.g. "What is Professor Ivan\'s email?" or "When are Peter\'s office hours?".'
    )
    args_schema: type[BaseModel] = FindPersonToolInput

    async def _run(self, name: str) -> list[dict] | str:
        """Look up professors by name."""
        async with open_university_repository() as repo:
            professors = await repo.find_professors_by_name(name)

        logger.info('FindPerson %r -> %s', name, f'{len(professors)} match(es)' if professors else 'none')
        if professors is None:
            return f'No professor found with the name {name!r}.'
        return [professor.model_dump() for professor in professors]


class FindPlaceToolInput(BaseModel):
    """Input schema for FindPlaceTool."""

    name: str = Field(
        description="The name of the campus place to look up, e.g. 'Main Library', 'Cafeteria', 'Gym'.",
    )


class FindPlaceTool(BaseTool):
    name: str = 'Find Campus Place'
    description: str = (
        'Look up a specific campus facility BY ITS NAME (library, cafeteria, gym, admissions '
        'office, dormitory, etc.). Returns its building, floor and opening hours. Use this for '
        'questions like "When does the library open?" or "Where is the cafeteria?".'
    )
    args_schema: type[BaseModel] = FindPlaceToolInput

    async def _run(self, name: str) -> dict | str:
        """Look up a campus place (building, floor, opening hours) by name."""
        async with open_university_repository() as repo:
            place = await repo.find_place_by_name(name)

        logger.info('FindPlace %r -> %s', name, place.name if place else 'none')
        if place is None:
            return f'No campus place found with the name {name!r}.'
        return place.model_dump()
