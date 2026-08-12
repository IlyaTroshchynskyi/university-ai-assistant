from typing import Annotated, ClassVar

from pydantic import BaseModel, computed_field, Field, StringConstraints

from app.api.v1.faculty.schemas import FacultyItem
from app.api.v1.programs.enums import Degree
from app.core.dynamodb.base_items import TableItem
from app.core.dynamodb.indexes import normalize_name


class ProgramCreate(BaseModel):
    name: Annotated[str, StringConstraints(strip_whitespace=True, min_length=2, max_length=120)]
    faculty_id: str
    degree: Degree
    duration_years: int = Field(ge=1, le=8)
    tuition_usd: int = Field(ge=0, le=1_000_000)


class Program(ProgramCreate):
    id: str


class ProgramItem(TableItem, ProgramCreate):
    entity: ClassVar[str] = 'PROGRAM'

    @computed_field  # type: ignore[prop-decorator]
    @property
    def gsi1pk(self) -> str:
        return f'{FacultyItem.entity}#{self.faculty_id}'

    @computed_field  # type: ignore[prop-decorator]
    @property
    def gsi1sk(self) -> str:
        return f'{self.entity}#{self.name}'


class ProgramNameItem(BaseModel):
    entity: ClassVar[str] = 'PROGRAM_NAME'
    unique_sk: ClassVar[str] = '#UNIQUE'

    faculty_id: str
    name: str
    # Points back at ``ProgramItem.id`` — the uuid of the row holding the name.
    program_id: str

    @classmethod
    def key(cls, faculty_id: str, name: str) -> dict[str, str]:
        return {'pk': f'{cls.entity}#{faculty_id}#{normalize_name(name)}', 'sk': cls.unique_sk}

    @computed_field  # type: ignore[prop-decorator]
    @property
    def pk(self) -> str:
        return self.key(self.faculty_id, self.name)['pk']

    @computed_field  # type: ignore[prop-decorator]
    @property
    def sk(self) -> str:
        return self.unique_sk
