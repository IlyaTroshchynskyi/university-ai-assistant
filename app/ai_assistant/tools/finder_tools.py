from crewai.tools import BaseTool
from pydantic import BaseModel, Field


class FindPersonToolInput(BaseModel):
    """Input schema for FindPersonTool."""

    name: str = Field(
        ...,
        description='The exact full name (or first name) of the professor or staff member to look up.',
    )


class FindPersonTool(BaseTool):
    name: str = 'Find Professor'
    description: str = (
        'Look up a specific professor or staff member BY THEIR NAME. Returns their title, '
        'faculty, office/room, email and office hours. Use this when the user asks about a '
        'named person, e.g. "What is Professor Ivan\'s email?" or "When are Peter\'s office hours?".'
    )
    args_schema: type[BaseModel] = FindPersonToolInput

    async def _run(self, name: str) -> dict | str:
        """Look up a professor's faculty, office, email and office hours by name."""
        # TODO: replace this in-memory stub with a real lookup (DB / API).
        people = {
            'Ivan': {
                'id': 1,
                'full_name': 'Ivan Ivanov',
                'title': 'Professor',
                'faculty_id': 1,
                'faculties': [],
                'email': 'ivan@university.edu',
                'room_id': 'A-201',
                'office_hour': '10:00-15:00',
            },
            'Peter': {
                'id': 2,
                'full_name': 'Peter Petrov',
                'title': 'Associate Professor',
                'faculty_id': 1,
                'faculties': [],
                'email': 'peter@university.edu',
                'room_id': 'B-105',
                'office_hour': '10:00-15:00',
            },
        }
        return people.get(name, f'No professor found with the name {name!r}.')


class FindPlaceToolInput(BaseModel):
    """Input schema for FindPlaceTool."""

    name: str = Field(
        ...,
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
        # TODO: replace this in-memory stub with a real lookup (DB / API).
        places = [
            {
                'id': 1,
                'name': 'Cafeteria',
                'building': 'Student Center',
                'floor': '1',
                'opening_hours': 'Mon-Fri 07:30-20:00; Sat 09:00-16:00; Sun closed',
            },
            {
                'id': 2,
                'name': 'Main Library',
                'building': 'Main Library',
                'floor': '1-3',
                'opening_hours': 'Mon-Fri 08:00-22:00; Sat-Sun 10:00-18:00',
            },
            {
                'id': 3,
                'name': 'Gym (Fitness Center)',
                'building': 'Student Center',
                'floor': '2',
                'opening_hours': 'Daily 06:00-23:00',
            },
            {
                'id': 4,
                'name': 'Admissions Office',
                'building': 'Keynes Hall',
                'floor': '1',
                'opening_hours': 'Mon-Fri 09:00-17:00',
            },
            {
                'id': 5,
                'name': 'Dormitory (Riverside Residence)',
                'building': 'Riverside Residence',
                'floor': 'lobby',
                'opening_hours': 'Front desk 24/7',
            },
        ]
        for item in places:
            if item['name'].lower() == name.lower():
                return item
        return f'No campus place found with the name {name!r}.'
