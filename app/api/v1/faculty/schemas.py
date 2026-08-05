from typing import Annotated, ClassVar

from pydantic import BaseModel, computed_field, StringConstraints

from app.core.dynamodb.base_items import TableItem


def normalize_faculty_name(name: str) -> str:
    return ' '.join(name.lower().split())


class FacultyCreate(BaseModel):
    name: Annotated[str, StringConstraints(strip_whitespace=True, min_length=2, max_length=100)]


class FacultyItem(TableItem, FacultyCreate):
    entity: ClassVar[str] = 'FACULTY'


class FacultyNameItem(BaseModel):
    """The row that reserves a faculty name, so no two faculties can hold the same one.

    DynamoDB enforces uniqueness on the primary key and nowhere else, and a faculty's own key is
    ``FACULTY#{uuid4}`` — freshly minted on every create, so ``attribute_not_exists(pk)`` on it can
    never fire. Keying a second row *by the name* gives the name a primary key of its own; written
    in the same transaction as the faculty (``FacultyRepository.create_faculty``), its condition is
    what a second 'Computer Science' collides with.

    It deliberately carries no ``entity_type`` and no GSI keys: nothing indexes it, and the faculty
    listing — a scan filtered on ``entity_type`` — steps over it rather than trying to read it as a
    faculty. ``faculty_id`` points back at the row holding the name, which is also what a lookup by
    name would need.
    """

    entity: ClassVar[str] = 'FACULTY_NAME'
    unique_sk: ClassVar[str] = '#UNIQUE'

    name: str
    faculty_id: str

    @classmethod
    def key(cls, name: str) -> dict[str, str]:
        """The primary key reserving ``name`` — what deleting the reservation needs, built from the
        name alone."""
        return {'pk': f'{cls.entity}#{normalize_faculty_name(name)}', 'sk': cls.unique_sk}

    @computed_field
    @property
    def pk(self) -> str:
        return f'{self.entity}#{normalize_faculty_name(self.name)}'

    @computed_field
    @property
    def sk(self) -> str:
        return self.unique_sk
