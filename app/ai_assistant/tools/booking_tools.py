from crewai.tools import BaseTool
from pydantic import BaseModel, Field

from app.core.dynamodb.schemas import SlotStatus
from app.core.dynamodb.slots_repository import open_slots_repository


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

    async def _run(self, date: str | None = None) -> list[dict] | str:
        """Return open slots from DynamoDB (``appointment_slots``): a single date (P9) if ``date`` is
        given, else all open slots across dates (P9b). Never invents slots — reads the live table."""
        async with open_slots_repository() as repo:
            slots = (
                await repo.list_open_slots_on_date(date) if date else await repo.list_slots_by_status(SlotStatus.OPEN)
            )
        if not slots:
            return f'No open consultation slots{f" on {date}" if date else ""}.'
        return [slot.model_dump() for slot in slots]


# NOTE: there is no Book/Cancel BaseTool. Writing a booking is deliberately kept out of the agent's
# hands (the human-in-the-loop guard): the flow calls SlotsRepository.book_slot() / cancel_slot()
# directly, in code (see main_flow.do_book / cancel_booking), only after the human confirms.
