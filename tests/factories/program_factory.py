from polyfactory.factories.pydantic_factory import ModelFactory

from app.api.v1.programs.schemas import ProgramCreate


class ProgramCreationFactory(ModelFactory[ProgramCreate]):
    __model__ = ProgramCreate
    __check_model__ = False
