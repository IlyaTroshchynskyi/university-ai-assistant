from polyfactory.factories.pydantic_factory import ModelFactory
from pydantic import BaseModel

from app.core.dynamodb.schemas import Place

PLACE_NAME = 'Main Library'
PLACE_OPENING_HOURS = 'Mon–Fri 08:00–22:00; Sat–Sun 10:00–18:00'


class PlaceFactory(ModelFactory[Place]):
    __model__ = Place

    name = PLACE_NAME
    building = PLACE_NAME
    opening_hours = PLACE_OPENING_HOURS


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
