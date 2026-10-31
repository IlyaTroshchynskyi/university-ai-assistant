from polyfactory.factories.pydantic_factory import ModelFactory

from app.api.v1.rooms.schemas import RoomCreate


class RoomCreationFactory(ModelFactory[RoomCreate]):
    __model__ = RoomCreate
