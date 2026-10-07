import datetime
import logging

from langchain_core.tools import tool
from pydantic import EmailStr

from app.ai_assistant_langchain.agent_schemas import (
    BookAppointmentToolInput,
    CancelAppointmentToolInput,
    ListFreeSlotsToolInput,
    ListMyBookingsToolInput,
)
from app.core.dynamodb.slots_repository import open_slots_repository

logger = logging.getLogger(__name__)


@tool(args_schema=ListFreeSlotsToolInput)
async def list_free_slots(date: datetime.date) -> list[dict] | str:
    """List the admissions consultation slots that are still open on one day.

    Returns each slot's id, date, start and end time, or a sentence saying the day has none. A slot
    carries no subject of its own: what the applicant wants to discuss is recorded when it is booked.
    """
    # The table keys days as strings; the schema parses them as dates. This is where the two meet.
    day = date.isoformat()
    async with open_slots_repository() as repo:
        slots = await repo.list_open_slots_on_date(day)

    logger.info('ListFreeSlots date=%r -> %d slot(s)', day, len(slots))
    if not slots:
        return f'No open consultation slots on {day}.'

    return [slot.model_dump() for slot in slots]


@tool
async def find_earliest_open_slots() -> list[dict] | str:
    """Find the soonest open consultation slots there are, across every day at once, earliest first.

    Returns each slot's id, date, start and end time, or a sentence saying no upcoming day has any.
    """
    # From today, never from the epoch: a slot left open on a day already past is not an
    # appointment anyone can take, and it would otherwise be the first thing offered.
    today = datetime.date.today().isoformat()
    async with open_slots_repository() as repo:
        slots = await repo.list_earliest_open_slots(today)

    logger.info('FindEarliestOpenSlots from=%r -> %d slot(s)', today, len(slots))
    if not slots:
        return 'There are no open consultation slots on any upcoming day at all.'

    return [slot.model_dump() for slot in slots]


@tool(args_schema=ListMyBookingsToolInput)
async def list_my_bookings(applicant_email: EmailStr) -> list[dict] | str:
    """List the consultation appointments already booked under an email address.

    Returns each booking's id, date, start time and topic, or a sentence saying nothing is booked
    under that address.
    """
    async with open_slots_repository() as repo:
        slots = await repo.list_student_bookings(applicant_email)

    logger.info('ListMyBookings email=%r -> %d booking(s)', applicant_email, len(slots))
    if not slots:
        return f'No appointments are booked under {applicant_email}.'

    return [slot.model_dump() for slot in slots]


@tool(args_schema=BookAppointmentToolInput)
async def book_appointment(
    slot_id: int, date: datetime.date, start_time: str, applicant_email: EmailStr, topic: str
) -> str:
    """Book one open consultation slot for an applicant.

    The booking is recorded against the applicant's email address, which is how they are reached
    about the appointment afterwards. The slot is only booked if it is still open: someone else may
    have taken it since it was listed, and then the result says so instead of booking. Otherwise
    the result names the appointment that was booked.
    """
    day = date.isoformat()
    async with open_slots_repository() as repo:
        slot = await repo.book_slot(day, start_time, slot_id, booked_by=applicant_email, topic=topic)

    if slot is None:
        # A lost conditional write, not a failure: the slot was taken (or never existed) between the
        # listing and now. The agent is told in words it can relay and recover from.
        logger.info('BookAppointment slot=%s %s %s -> not open', slot_id, day, start_time)
        return f'Slot {slot_id} on {day} at {start_time} is no longer available.'

    logger.info('BookAppointment slot=%s %s %s -> booked for %r', slot_id, day, start_time, applicant_email)
    return f'Booked slot {slot_id} on {day} at {start_time} for {applicant_email}, about {topic}.'


@tool(args_schema=CancelAppointmentToolInput)
async def cancel_appointment(slot_id: int, date: datetime.date, start_time: str, applicant_email: EmailStr) -> str:
    """Cancel a booked consultation and release the slot so someone else can take it.

    Only the address the appointment is booked under can cancel it: an appointment that is not
    booked, or is booked under a different address, is left untouched and the result says so.
    """
    day = date.isoformat()
    async with open_slots_repository() as repo:
        slot = await repo.cancel_slot(day, start_time, slot_id, booked_by=applicant_email)

    if slot is None:
        # Deliberately does not say which of the two it was: whether somebody *else* holds that time
        # is not this applicant's to learn.
        logger.info('CancelAppointment slot=%s %s %s -> no booking under %r', slot_id, day, start_time, applicant_email)
        return (
            f'There is no appointment under {applicant_email} in slot {slot_id} on {day} at {start_time}, '
            'so nothing was cancelled.'
        )

    logger.info('CancelAppointment slot=%s %s %s -> cancelled', slot_id, day, start_time)
    return f'Cancelled the appointment in slot {slot_id} on {day} at {start_time}. The slot is open again.'


BOOKING_TOOLS = [list_free_slots, find_earliest_open_slots, list_my_bookings, book_appointment, cancel_appointment]
