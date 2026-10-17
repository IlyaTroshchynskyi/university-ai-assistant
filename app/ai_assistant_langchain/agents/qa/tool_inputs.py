from typing import Annotated

from pydantic import BaseModel, BeforeValidator, Field

from app.api.v1.rooms.enums import Weekday
from app.api.v1.scholarships.enums import Faculty

DEFAULT_COMMON_ISSUE_COUNT = 5
MAX_COMMON_ISSUE_COUNT = 20


class RetrieverToolInput(BaseModel):
    """Input schema for the RetrieverTool."""

    query: str = Field(
        description="The user's question or search query about the university.",
    )


class FindPersonToolInput(BaseModel):
    """Input schema for the FindPersonTool."""

    name: str = Field(
        description='The full name or first name of the professor or staff member to look up.',
    )


class FindPlaceToolInput(BaseModel):
    """Input schema for the FindPlaceTool."""

    name: str = Field(
        description="The name of the campus place to look up, e.g. 'Main Library', 'Cafeteria', 'Gym'.",
    )


def _blank_to_none(value: object) -> object:
    """A filter of nothing but whitespace is a filter that was not given."""
    return None if isinstance(value, str) and not value.strip() else value


OptionalFilter = Annotated[str | None, BeforeValidator(_blank_to_none)]


class GetScheduleToolInput(BaseModel):
    group: OptionalFilter = Field(
        default=None,
        description="The student group's code, exactly as the applicant gave it, e.g. 'CS-1'.",
    )
    professor: OptionalFilter = Field(
        default=None,
        description='The full name or first name of the professor whose classes are wanted.',
    )
    course: OptionalFilter = Field(
        default=None,
        description="The course's name, or a distinctive part of it, e.g. 'Data Structures'.",
    )
    weekday: Weekday | None = Field(
        default=None,
        description="The day of the week, as one of 'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'.",
    )


class CheckScholarshipToolInput(BaseModel):
    gpa: float = Field(
        ge=0,
        le=4,
        description="The applicant's GPA on the 4.0 scale, exactly as they stated it.",
    )
    family_income: float | None = Field(
        default=None,
        ge=0,
        description=('The applicant\'s family income per year in US dollars, as a plain number: 25000 for "$25k".'),
    )
    entrance_exam_score: float | None = Field(
        default=None,
        ge=0,
        le=100,
        description="The applicant's NIT entrance exam score, from 0 to 100.",
    )
    is_female: bool | None = Field(
        default=None,
        description='True when the applicant has said they are female, false when they have said they are not.',
    )
    faculty: Faculty | None = Field(
        default=None,
        description='The faculty the programme the applicant named belongs to.',
    )
    annual_tuition: float | None = Field(
        default=None,
        gt=0,
        description='The annual tuition of that programme in US dollars, as a plain number.',
    )


class ListCommonIssuesToolInput(BaseModel):
    limit: int = Field(
        default=DEFAULT_COMMON_ISSUE_COUNT,
        ge=1,
        le=MAX_COMMON_ISSUE_COUNT,
        description='How many issues to return, the most-voted first.',
    )


class CompareProgramsToolInput(BaseModel):
    """Input schema for the CompareProgramsTool."""

    programs: list[str] = Field(
        description=(
            'The programmes to compare, in the order the applicant named them, each named as they '
            'named it: ["Economics", "Business Analytics"]. The name on its own — no degree, no '
            'faculty, and not the word "programme" appended to it. Between two and five.'
        ),
    )
