from polyfactory.factories.pydantic_factory import ModelFactory

from app.api.v1.rooms.schemas import CreateRoom


class RoomCreationFactory(ModelFactory[CreateRoom]):
    __model__ = CreateRoom
    __check_model__ = False
