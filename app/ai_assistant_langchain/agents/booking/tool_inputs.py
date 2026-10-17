# `import datetime`, as in `tools.py`: the tool argument is itself called `date`, and a
# field of that name shadows the imported class, which stops pydantic at import time.
import datetime

from pydantic import BaseModel, EmailStr, Field


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
