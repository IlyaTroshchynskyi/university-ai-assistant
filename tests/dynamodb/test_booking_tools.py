import pytest

from app.ai_assistant_langchain.booking_tools import (
    book_appointment,
    cancel_appointment,
    find_earliest_open_slots,
    list_free_slots,
    list_my_bookings,
)

DAY = '2026-10-13'
EMAIL = 'james.carter@example.com'
SLOT = {'slot_id': 101, 'date': DAY, 'start_time': '09:00'}


@pytest.mark.usefixtures('slots_table')
class TestBookingToolsOnAnEmptyCalendar:
    async def test_day_with_nothing_open_is_reported_as_a_fact(self) -> None:
        result = await list_free_slots.ainvoke({'date': DAY})

        assert result == 'No open consultation slots on 2026-10-13.'

    async def test_calendar_with_nothing_open_is_reported_as_a_fact(self) -> None:
        result = await find_earliest_open_slots.ainvoke({})

        assert result == 'There are no open consultation slots on any upcoming day at all.'

    async def test_address_with_no_bookings_is_reported_as_a_fact(self) -> None:
        result = await list_my_bookings.ainvoke({'applicant_email': EMAIL})

        assert result == 'No appointments are booked under james.carter@example.com.'

    async def test_slot_that_is_not_open_is_reported_as_a_fact(self) -> None:
        result = await book_appointment.ainvoke({**SLOT, 'applicant_email': EMAIL, 'topic': 'my scholarship'})

        assert result == 'Slot 101 on 2026-10-13 at 09:00 is no longer available.'

    async def test_cancellation_that_found_no_booking_is_reported_as_a_fact(self) -> None:
        result = await cancel_appointment.ainvoke({**SLOT, 'applicant_email': EMAIL})

        assert result == (
            'There is no appointment under james.carter@example.com in slot 101 on 2026-10-13 at 09:00, so '
            'nothing was cancelled.'
        )
