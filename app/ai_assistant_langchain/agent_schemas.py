# `import datetime`, as in `booking_tools.py`: the tool argument is itself called `date`, and a
# field of that name shadows the imported class, which stops pydantic at import time.
import datetime
from typing import NotRequired, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.types import Interrupt
from pydantic import BaseModel, EmailStr, Field


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


class ListFreeSlotsToolInput(BaseModel):
    """Input schema for the ListFreeSlotsTool."""

    date: datetime.date = Field(
        description=(
            "The day to look at, 'YYYY-MM-DD'. Required, and it is the day the applicant asked "
            'for. When they named no day, use `find_earliest_open_slots` instead of guessing one '
            'here.'
        ),
    )


class ListMyBookingsToolInput(BaseModel):
    """Input schema for the ListMyBookingsTool."""

    applicant_email: EmailStr = Field(
        description='The email address the appointment was booked under. Ask for it; it is the key.',
    )


class SlotKeyToolInput(BaseModel):
    """The three fields that identify one slot. All of them come from `list_free_slots`."""

    slot_id: int = Field(description='The slot id, exactly as `list_free_slots` returned it.')
    date: datetime.date = Field(description="The slot's date, 'YYYY-MM-DD', exactly as `list_free_slots` returned it.")
    # A pattern, not `datetime.time`: the sort key is built as f'{start_time}#{slot_id}', and
    # `time.isoformat()` renders '09:00' as '09:00:00', so the key would stop matching for every
    # slot there is. The pattern checks the one thing that matters — the shape the key is made of —
    # and changes nothing. It also rejects '9:00', which today keys '9:00#101' and comes back as a
    # phantom "that slot is no longer available".
    start_time: str = Field(
        pattern=r'^([01]\d|2[0-3]):[0-5]\d$',
        description="The slot's start time, 'HH:MM', exactly as `list_free_slots` returned it.",
    )


class BookAppointmentToolInput(SlotKeyToolInput):
    """Input schema for the BookAppointmentTool."""

    applicant_email: EmailStr = Field(
        description=(
            'The email address the applicant gave you. Ask for it and use it exactly as they wrote '
            'it; never invent one or build it from their name. It identifies the booking and is how '
            'they are reached about it later.'
        ),
    )
    topic: str = Field(
        description=(
            'What the applicant told you, in their own words, that they want to discuss. The test '
            'is whether you can point at the message they said it in: if you cannot quote them, '
            'you do not have a topic, and being a required field is not permission to invent one. '
            'Nothing describing the meeting itself passes that test — "consultation", '
            '"appointment", "advising", "admissions consultation" and the like restate what they '
            'are booking, not what they want out of it. If they have not said, do not call this '
            'tool: ask them, and call it once they answer.'
        ),
    )


class CancelAppointmentToolInput(SlotKeyToolInput):
    """Input schema for the CancelAppointmentTool."""

    applicant_email: EmailStr = Field(
        description=(
            'The email address the appointment is booked under, exactly as `list_my_bookings` '
            'returned it. Only that address can cancel it, so look the appointment up rather than '
            'cancelling from a date and time the applicant merely named.'
        ),
    )


class AgentResponse(TypedDict):
    messages: list[BaseMessage]

    __interrupt__: NotRequired[list[Interrupt]]
    """Present only when the graph paused. LangGraph puts it into the returned state itself, which
    is why it is spelled with the dunders: this is the key it actually uses, not one we chose."""
