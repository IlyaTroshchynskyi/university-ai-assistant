from polyfactory.factories.pydantic_factory import ModelFactory

from app.api.v1.faculty.schemas import FacultyCreate


class FacultyCreationFactory(ModelFactory[FacultyCreate]):
    __model__ = FacultyCreate
    __check_model__ = False
