# `import datetime`, as in `booking_tools.py`: the tool argument is itself called `date`, and a
# field of that name shadows the imported class, which stops pydantic at import time.
import datetime
from typing import Annotated, NotRequired, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.types import Interrupt
from pydantic import BaseModel, BeforeValidator, EmailStr, Field

from app.api.v1.rooms.enums import Weekday
from app.api.v1.scholarships.enums import Faculty


class CustomContext(BaseModel):
    user_id: str


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


class CompareProgramsToolInput(BaseModel):
    """Input schema for the CompareProgramsTool."""

    programs: list[str] = Field(
        description=(
            'The programmes to compare, in the order the applicant named them, each named as they '
            'named it: ["Economics", "Business Analytics"]. The name on its own — no degree, no '
            'faculty, and not the word "programme" appended to it. Between two and five.'
        ),
    )


class ListFreeSlotsToolInput(BaseModel):
    """Input schema for the ListFreeSlotsTool."""

    date: datetime.date = Field(
        description="The day to look at, 'YYYY-MM-DD'.",
    )


class ListMyBookingsToolInput(BaseModel):
    """Input schema for the ListMyBookingsTool."""

    applicant_email: EmailStr = Field(
        description='The email address the appointment was booked under.',
    )


class SlotKeyToolInput(BaseModel):
    """The three fields that identify one slot. All of them come from `list_free_slots`."""

    slot_id: int = Field(description='The slot id.')
    date: datetime.date = Field(description="The slot's date, 'YYYY-MM-DD'.")
    # A pattern, not `datetime.time`: the sort key is built as f'{start_time}#{slot_id}', and
    # `time.isoformat()` renders '09:00' as '09:00:00', so the key would stop matching for every
    # slot there is. The pattern checks the one thing that matters — the shape the key is made of —
    # and changes nothing. It also rejects '9:00', which today keys '9:00#101' and comes back as a
    # phantom "that slot is no longer available".
    start_time: str = Field(
        pattern=r'^([01]\d|2[0-3]):[0-5]\d$',
        description="The slot's start time, 'HH:MM'.",
    )


class BookAppointmentToolInput(SlotKeyToolInput):
    """Input schema for the BookAppointmentTool."""

    applicant_email: EmailStr = Field(
        description=(
            'The email address the applicant gave, exactly as they wrote it. It identifies the booking '
            'and is how they are reached about it later.'
        ),
    )
    topic: str = Field(
        description=(
            'The subject the applicant said they want to discuss, in their own words: what they want '
            'out of the meeting, not a name for the meeting itself.'
        ),
    )


class CancelAppointmentToolInput(SlotKeyToolInput):
    """Input schema for the CancelAppointmentTool."""

    applicant_email: EmailStr = Field(
        description='The email address the appointment is booked under.',
    )


class AgentResponse(TypedDict):
    messages: list[BaseMessage]

    __interrupt__: NotRequired[list[Interrupt]]
    """Present only when the graph paused. LangGraph puts it into the returned state itself, which
    is why it is spelled with the dunders: this is the key it actually uses, not one we chose."""
