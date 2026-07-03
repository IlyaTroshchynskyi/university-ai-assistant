from typing import Any

from crewai.tools import BaseTool
from pydantic import BaseModel, Field

# TODO: replace this in-memory list with a real store (DB / calendar API).
SLOTS = [
    {
        'id': 1,
        'date': '2026-10-06',
        'start_time': '09:00',
        'end_time': '09:30',
        'topic': None,
        'booked_by': None,
        'status': 'open',
    },
    {
        'id': 2,
        'date': '2026-10-06',
        'start_time': '10:00',
        'end_time': '10:30',
        'topic': None,
        'booked_by': None,
        'status': 'open',
    },
    {
        'id': 3,
        'date': '2026-10-06',
        'start_time': '11:00',
        'end_time': '11:30',
        'topic': None,
        'booked_by': None,
        'status': 'open',
    },
    {
        'id': 4,
        'date': '2026-10-06',
        'start_time': '13:00',
        'end_time': '13:30',
        'topic': None,
        'booked_by': None,
        'status': 'open',
    },
    {
        'id': 5,
        'date': '2026-10-06',
        'start_time': '14:00',
        'end_time': '14:30',
        'topic': None,
        'booked_by': None,
        'status': 'open',
    },
    {
        'id': 6,
        'date': '2026-10-08',
        'start_time': '09:00',
        'end_time': '09:30',
        'topic': None,
        'booked_by': None,
        'status': 'open',
    },
    {
        'id': 7,
        'date': '2026-10-08',
        'start_time': '10:00',
        'end_time': '10:30',
        'topic': None,
        'booked_by': None,
        'status': 'open',
    },
    {
        'id': 8,
        'date': '2026-10-08',
        'start_time': '11:00',
        'end_time': '11:30',
        'topic': None,
        'booked_by': None,
        'status': 'open',
    },
    {
        'id': 9,
        'date': '2026-10-08',
        'start_time': '13:00',
        'end_time': '13:30',
        'topic': None,
        'booked_by': None,
        'status': 'open',
    },
    {
        'id': 10,
        'date': '2026-10-08',
        'start_time': '14:00',
        'end_time': '14:30',
        'topic': None,
        'booked_by': None,
        'status': 'open',
    },
    {
        'id': 11,
        'date': '2026-10-13',
        'start_time': '09:00',
        'end_time': '09:30',
        'topic': None,
        'booked_by': None,
        'status': 'open',
    },
    {
        'id': 12,
        'date': '2026-10-13',
        'start_time': '10:00',
        'end_time': '10:30',
        'topic': None,
        'booked_by': None,
        'status': 'open',
    },
    {
        'id': 13,
        'date': '2026-10-13',
        'start_time': '11:00',
        'end_time': '11:30',
        'topic': None,
        'booked_by': None,
        'status': 'open',
    },
    {
        'id': 14,
        'date': '2026-10-13',
        'start_time': '13:00',
        'end_time': '13:30',
        'topic': None,
        'booked_by': None,
        'status': 'open',
    },
    {
        'id': 15,
        'date': '2026-10-13',
        'start_time': '14:00',
        'end_time': '14:30',
        'topic': None,
        'booked_by': None,
        'status': 'open',
    },
]


class ProposedSlot(BaseModel):
    """A slot the booking agent proposes. Note: proposing does NOT book it."""

    slot_id: int = Field(description='The id of the chosen open slot.')
    date: str = Field(description="The slot's date, 'YYYY-MM-DD'.")
    start_time: str = Field(description="The slot's start time, 'HH:MM'.")
    topic: str = Field(
        default='',
        description=(
            'What the applicant wants to discuss (e.g. "scholarships"), taken from their '
            'request — recorded on the booking. Empty if not stated. NOT a property of the slot.'
        ),
    )
    applicant_name: str = Field(default='', description="The applicant's name if they gave one, else empty.")
    message: str = Field(
        description=(
            'A friendly one- or two-sentence message to show the applicant that proposes '
            'this slot and asks them to confirm (yes/no) or suggest another time.'
        ),
    )


# --- Plain functions: the single source of truth for slot state changes. The tools
# below and the flow's deterministic booking gate both call these. ---


def list_open_slots(date: str | None = None) -> list[dict]:
    """Return open slots, optionally filtered to a single date."""
    open_slots = [s for s in SLOTS if s['status'] == 'open']
    if date:
        open_slots = [s for s in open_slots if s['date'] == date]
    return open_slots


def book_slot(slot_id: int, applicant_name: str, topic: str) -> bool:
    """Book an open slot for the applicant. Returns True on success, False if the slot
    is unknown or no longer open."""
    for slot in SLOTS:
        if slot['id'] == slot_id:
            if slot['status'] != 'open':
                return False
            slot['booked_by'] = applicant_name
            slot['topic'] = topic
            slot['status'] = 'booked'
            return True
    return False


def cancel_slot(slot_id: int) -> bool:
    """Cancel a booked slot and free it again. Returns True on success, False if the slot
    is unknown or was not booked."""
    for slot in SLOTS:
        if slot['id'] == slot_id:
            if slot['status'] != 'booked':
                return False
            slot['booked_by'] = None
            slot['status'] = 'open'
            return True
    return False


class ListFreeSlotsToolInput(BaseModel):
    """Input schema for ListFreeSlotsTool."""

    date: str | None = Field(
        default=None,
        description="Optional date to filter by, in 'YYYY-MM-DD' format. Omit to list all open slots.",
    )


class ListFreeSlotsTool(BaseTool):
    name: str = 'List Free Slots'
    description: str = (
        'List open admissions consultation slots. Optionally pass a date '
        "('YYYY-MM-DD') to only see that day. Returns each slot's id, date, time and "
        'topic. Use this before proposing a slot — never invent slots.'
    )
    args_schema: type[BaseModel] = ListFreeSlotsToolInput

    async def _run(self, date: str | None = None) -> Any:
        """Return open slots, optionally filtered to a single date."""
        open_slots = list_open_slots(date)
        if not open_slots:
            return f'No open consultation slots{f" on {date}" if date else ""}.'
        return open_slots


# NOTE: there is no Book/Cancel BaseTool. Writing a booking is deliberately kept out of
# the agent's hands (the human-in-the-loop guard): the flow calls book_slot() / cancel_slot()
# directly, in code, only after the human confirms.
