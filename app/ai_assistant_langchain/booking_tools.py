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

    The date is required: ask the applicant which day they want before calling this. Returns each
    slot's id, date, start and end time — the id, date and start time are what `book_appointment`
    needs. A slot carries no subject of its own; what the applicant wants to discuss is recorded
    when it is booked, so any open slot can be used for any topic. These are the only appointments
    that exist: call this before proposing anything, and never offer a time it did not return.
    """
    # The table keys days as strings; the schema parses them as dates. This is where the two meet.
    day = date.isoformat()
    async with open_slots_repository() as repo:
        slots = await repo.list_open_slots_on_date(day)

    logger.info('ListFreeSlots date=%r -> %d slot(s)', day, len(slots))
    if not slots:
        return (
            f'No open consultation slots on {day}. Tell the applicant that day is empty, then call '
            f'`find_earliest_open_slots` and offer them the soonest appointment there is.'
        )

    return [slot.model_dump() for slot in slots]


@tool
async def find_earliest_open_slots() -> list[dict] | str:
    """Find the soonest open consultation slots there are, across every day at once, earliest first.

    This is what answers a request that names no day — "as soon as possible", "the nearest date you
    have", "when are you free?", "sometime next week". Do not answer those by picking a day and
    checking it with `list_free_slots`: the next free appointment may be weeks out, and a couple of
    empty days in a row tells you nothing about when the real one is. Call this when a day the
    applicant did name turns out to be empty, too, so you can offer the soonest alternative instead
    of sending them away to guess another date. Returns each slot's id, date, start and end time —
    the id, date and start time are what `book_appointment` needs.
    """
    # From today, never from the epoch: a slot left open on a day already past is not an
    # appointment anyone can take, and it would otherwise be the first thing offered.
    today = datetime.date.today().isoformat()
    async with open_slots_repository() as repo:
        slots = await repo.list_earliest_open_slots(today)

    logger.info('FindEarliestOpenSlots from=%r -> %d slot(s)', today, len(slots))
    if not slots:
        return 'There are no open consultation slots on any upcoming day at all. Tell the applicant so plainly.'

    return [slot.model_dump() for slot in slots]


@tool(args_schema=ListMyBookingsToolInput)
async def list_my_bookings(applicant_email: EmailStr) -> list[dict] | str:
    """List the consultation appointments already booked under an email address.

    This is how you find an appointment the applicant wants to move or cancel: they know their email
    address, not the slot id, so ask for the address and look it up here rather than asking them for
    details they do not have. Returns each booking's id, date, start time and topic — the id, date
    and start time are what `cancel_appointment` needs.
    """
    async with open_slots_repository() as repo:
        slots = await repo.list_student_bookings(applicant_email)

    logger.info('ListMyBookings email=%r -> %d booking(s)', applicant_email, len(slots))
    if not slots:
        return f'No appointments are booked under {applicant_email}. Check the address with the applicant.'

    return [slot.model_dump() for slot in slots]


@tool(args_schema=BookAppointmentToolInput)
async def book_appointment(
    slot_id: int, date: datetime.date, start_time: str, applicant_email: EmailStr, topic: str
) -> str:
    """Book one open consultation slot for an applicant.

    Use the id, date and start time exactly as `list_free_slots` returned them. The booking is
    recorded against the applicant's email address, so ask for it before calling this — it is how
    they are reached about the appointment afterwards. The slot is only booked if it is still open —
    someone else may have taken it since you listed it, and then this says so instead of booking. In
    that case list the open slots again and propose one of those.
    """
    day = date.isoformat()
    async with open_slots_repository() as repo:
        slot = await repo.book_slot(day, start_time, slot_id, booked_by=applicant_email, topic=topic)

    if slot is None:
        # A lost conditional write, not a failure: the slot was taken (or never existed) between the
        # listing and now. The agent is told in words it can relay and recover from.
        logger.info('BookAppointment slot=%s %s %s -> not open', slot_id, day, start_time)
        return (
            f'Slot {slot_id} on {day} at {start_time} is no longer available. '
            'List the open slots again and propose one of those.'
        )

    logger.info('BookAppointment slot=%s %s %s -> booked for %r', slot_id, day, start_time, applicant_email)
    return f'Booked slot {slot_id} on {day} at {start_time} for {applicant_email}, about {topic}.'


@tool(args_schema=CancelAppointmentToolInput)
async def cancel_appointment(slot_id: int, date: datetime.date, start_time: str, applicant_email: EmailStr) -> str:
    """Cancel a booked consultation and release the slot so someone else can take it.

    Use the id, date, start time and email exactly as `list_my_bookings` returned them. Only the
    address the appointment is booked under can cancel it: an appointment that is not booked, or is
    booked under a different address, is left untouched and this says so.
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
            'so nothing was cancelled. Look the applicant’s bookings up again and work from those.'
        )

    logger.info('CancelAppointment slot=%s %s %s -> cancelled', slot_id, day, start_time)
    return f'Cancelled the appointment in slot {slot_id} on {day} at {start_time}. The slot is open again.'
