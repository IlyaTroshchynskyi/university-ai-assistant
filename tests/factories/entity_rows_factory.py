from polyfactory.factories.pydantic_factory import ModelFactory
from pydantic import BaseModel

from app.core.dynamodb.indexes import normalize_name_key
from app.core.dynamodb.schemas import Place, Professor

PROFESSOR_NAME = 'Dr. Alan Whitfield'
PROFESSOR_TITLE = 'Professor'
PROFESSOR_OFFICE_HOURS = 'Mon 14:00–16:00'
PLACE_NAME = 'Main Library'
PLACE_OPENING_HOURS = 'Mon–Fri 08:00–22:00; Sat–Sun 10:00–18:00'


class ProfessorFactory(ModelFactory[Professor]):
    __model__ = Professor

    full_name = PROFESSOR_NAME
    title = PROFESSOR_TITLE
    office_hours = PROFESSOR_OFFICE_HOURS


class PlaceFactory(ModelFactory[Place]):
    __model__ = Place

    name = PLACE_NAME
    building = PLACE_NAME
    opening_hours = PLACE_OPENING_HOURS


class ProfessorRow(BaseModel):
    pk: str
    sk: str = '#META'
    gsi1pk: str
    gsi1sk: str
    gsi_name_pk: str = 'PROF'
    gsi_name_sk: str
    entity_type: str = 'professor'
    id: str
    full_name: str
    title: str
    faculty_id: str
    email: str
    room_id: str
    office_hours: str

    @classmethod
    def from_entity(cls, professor: Professor) -> 'ProfessorRow':
        return cls(
            pk=f'PROF#{professor.id}',
            gsi1pk=f'FACULTY#{professor.faculty_id}',
            gsi1sk=f'PROF#{professor.full_name}',
            gsi_name_sk=normalize_name_key(professor.full_name),
            **{field: str(value) for field, value in professor.model_dump().items()},
        )


class PlaceRow(BaseModel):
    pk: str
    sk: str = '#META'
    name_lower: str
    entity_type: str = 'place'
    id: str
    name: str
    building: str
    floor: str
    opening_hours: str

    @classmethod
    def from_entity(cls, place: Place) -> 'PlaceRow':
        return cls(
            pk=f'PLACE#{place.id}',
            name_lower=place.name.lower(),
            **{field: str(value) for field, value in place.model_dump().items()},
        )
