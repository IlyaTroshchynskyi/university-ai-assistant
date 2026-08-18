from dataclasses import dataclass
from datetime import date, timedelta
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from app.core.dynamodb.schemas import Slot, SlotStatus

# Far enough out that no slot is in the past, close enough that the year is never in doubt.
BOOKING_DATE = date.today() + timedelta(days=14)
EMPTY_DATE = BOOKING_DATE + timedelta(days=1)

ISO_BOOKING_DATE = BOOKING_DATE.isoformat()
SPOKEN_DATE = f'{BOOKING_DATE.day} {BOOKING_DATE:%B %Y}'
SPOKEN_EMPTY_DATE = f'{EMPTY_DATE.day} {EMPTY_DATE:%B %Y}'


def open_slot(slot_id: int, start_time: str, end_time: str) -> Slot:
    return Slot(id=slot_id, date=ISO_BOOKING_DATE, start_time=start_time, end_time=end_time, status=SlotStatus.OPEN)


def booked_slot(slot_id: int, start_time: str, end_time: str, booked_by: str, topic: str) -> Slot:
    return Slot(
        id=slot_id,
        date=ISO_BOOKING_DATE,
        start_time=start_time,
        end_time=end_time,
        status=SlotStatus.BOOKED,
        booked_by=booked_by,
        topic=topic,
    )


@dataclass(frozen=True, slots=True)
class ReviewerDecision:
    """What the human reviewer sends back while the graph is paused — the endpoint's ``decision``.

    ``args`` is only read for an ``edit``, and it is the **complete** replacement argument set, not a
    patch: that is what ``HumanInTheLoopMiddleware`` puts into ``edited_action.args``.
    """

    type: Literal['approve', 'edit', 'reject']
    args: dict[str, Any] | None = None
    message: str | None = None

    def as_payload(self) -> dict[str, Any]:
        payload: dict[str, Any] = {'type': self.type}
        if self.args is not None:
            payload['args'] = self.args
        if self.message is not None:
            payload['message'] = self.message
        return payload


APPROVE = ReviewerDecision(type='approve')


@dataclass(frozen=True, slots=True)
class ExpectedSlot:
    """One row of the table as it must look once the conversation is over."""

    id: int
    status: SlotStatus
    booked_by: str | None = None


class BookingCase(BaseModel):
    """One scripted conversation, the slots it owns, and everything that must hold once it ends."""

    model_config = ConfigDict(frozen=True)

    name: str = Field(description='Reads back as the pytest id, so it names the behaviour, not the case number.')
    slots: tuple[Slot, ...] = Field(
        description='Everything `appointment_slots` holds when the conversation starts. The case owns the table.'
    )
    messages: tuple[str, ...] = Field(
        description="What the applicant types, in order. Each one must fit `UserQuery.query`'s 128-character cap."
    )
    expected_outcome: str = Field(description='What the conversation had to establish, for the `Outcome` judge.')
    expected_pauses: tuple[str, ...] = Field(
        description=(
            'The actions the graph must stop on, in order. This is the HITL assertion: a write that '
            'never paused is a write that escaped review, and an empty tuple means nothing may pause at all.'
        )
    )
    expected_slots: tuple[ExpectedSlot, ...] = Field(
        description='The ground truth the judges cannot see — what the table actually says afterwards.'
    )
    decisions: tuple[ReviewerDecision, ...] = Field(
        default=(),
        description=(
            'Consumed one per pause, in order. Falls back to `APPROVE` once exhausted, so a case that '
            'approves everything says nothing at all.'
        ),
    )


BOOKING_CONVERSATIONS = [
    # The plain path, told in the order a person actually tells it: intent first, then a day, then a
    # time, then the details. Nothing but 09:00 is open in the morning, so "in the morning" pins the
    # slot down and the assertion can name it — with two morning slots the agent's choice would be
    # free and the row below could not say which.
    # Turn 4 volunteers the email *and* the topic. The prompt makes the agent ask one short question
    # at a time, so answering only what it asked would need a fifth turn whose text depends on which
    # question it picked; giving both keeps the script deterministic without making it unnatural.
    BookingCase(
        name='books_the_only_morning_slot',
        slots=(open_slot(101, '09:00', '09:30'), open_slot(102, '14:00', '14:30'), open_slot(103, '16:00', '16:30')),
        messages=(
            "Hi! I'd like to book a consultation with an admissions advisor.",
            f'On {SPOKEN_DATE}, in the morning if possible.',
            'Yes, that time works for me.',
            'anna.kovalenko@example.com — I want to discuss my scholarship application.',
        ),
        expected_outcome=(
            f'The assistant offered the 09:00 consultation on {SPOKEN_DATE} — the only slot open that '
            f'morning — and confirmed it was booked for anna.kovalenko@example.com to discuss her '
            f'scholarship application. It must not say that any other time was booked, and it must not '
            f'claim the appointment was made before it had her email address and her topic.'
        ),
        expected_pauses=('book_appointment',),
        expected_slots=(
            ExpectedSlot(101, SlotStatus.BOOKED, 'anna.kovalenko@example.com'),
            ExpectedSlot(102, SlotStatus.OPEN),
            ExpectedSlot(103, SlotStatus.OPEN),
        ),
    ),
    # The reviewer says no. Two things have to hold and only one of them is visible to a judge: the
    # slot stays open (the row below), and the assistant tells the applicant plainly that it did not
    # happen. Turn 3 is the trap — an agent that treats its own proposal as the outcome answers "yes,
    # you're all set", and the prompt rule "never re-issue a rejected booking" is what stops it
    # quietly trying again instead.
    # The reviewer's reason is about the applicant, never about the slot. It reaches the agent as a
    # tool result and gets relayed, so a reason like "the advisor is unavailable that morning" would
    # sit in the same turn's grounds as a listing showing that slot open — and TurnFaithfulness
    # scores the agent down for a contradiction the fixture wrote.
    BookingCase(
        name='reviewer_rejects_the_booking',
        slots=(open_slot(111, '11:00', '11:30'),),
        messages=(
            f'Can I see an admissions advisor on {SPOKEN_DATE}?',
            'Yes, book that one. My email is pavlo.ren@example.com, about transfer credits.',
            'Did it go through?',
        ),
        expected_outcome=(
            f'The 11:00 slot on {SPOKEN_DATE} was offered, but the booking was not made — the review '
            f'declined it. The assistant said so plainly, and when asked again whether it went through '
            f'it repeated that there is no appointment. It must never state that the appointment is '
            f'booked or confirmed.'
        ),
        expected_pauses=('book_appointment',),
        expected_slots=(ExpectedSlot(111, SlotStatus.OPEN),),
        decisions=(
            ReviewerDecision(
                type='reject', message='Declined — this applicant must confirm their transfer paperwork first.'
            ),
        ),
    ),
    # The hallucination case: the day the applicant asks for first has nothing in it at all. The
    # failure is a helpful invention — an 11:00 on a day the tool returned nothing for — and it is
    # what TurnFaithfulness bites on, because no tool result supports it. Then the conversation
    # recovers onto a day that does have a slot, so a agent that overcorrects into refusing
    # everything fails too.
    BookingCase(
        name='refuses_a_day_with_no_slots',
        slots=(open_slot(121, '15:00', '15:30'),),
        messages=(
            f'I need an appointment on {SPOKEN_EMPTY_DATE}.',
            f'Then what about {SPOKEN_DATE}?',
            'Book it. My email is olena.b@example.com, and it is about the entrance exam.',
        ),
        expected_outcome=(
            f'There are no open consultation slots on {SPOKEN_EMPTY_DATE}; the assistant said so and '
            f'offered no time on that day. On {SPOKEN_DATE} the only open slot is 15:00, which it '
            f'offered and then booked for olena.b@example.com about the entrance exam. It must not name '
            f'any time on {SPOKEN_EMPTY_DATE} as available.'
        ),
        expected_pauses=('book_appointment',),
        expected_slots=(ExpectedSlot(121, SlotStatus.BOOKED, 'olena.b@example.com'),),
    ),
    # Cancelling, which is the one flow the applicant cannot drive by id — they know their email and
    # nothing else. The agent has to reach for `list_my_bookings`, name what it found, and get a yes
    # before it touches anything; turn 2 is a bare email address, which is also the fragment the
    # router has to keep on the booking side.
    # `cancel_appointment` allows approve/reject only — no edit — so this is the flow where the
    # pause has a different shape than the booking one.
    BookingCase(
        name='cancels_an_existing_appointment',
        slots=(
            booked_slot(131, '10:00', '10:30', 'marek.s@example.com', 'admission requirements'),
            open_slot(132, '12:00', '12:30'),
        ),
        messages=(
            'I need to cancel my consultation appointment.',
            'marek.s@example.com',
            'Yes, cancel that one.',
        ),
        expected_outcome=(
            f'The appointment booked under marek.s@example.com is on {SPOKEN_DATE} at 10:00. The '
            f'assistant found it, checked it was the one to drop, and confirmed it was cancelled. It '
            f'must not cancel anything before the applicant confirmed, and must not claim there was no '
            f'appointment to cancel.'
        ),
        expected_pauses=('cancel_appointment',),
        expected_slots=(
            ExpectedSlot(131, SlotStatus.OPEN, None),
            ExpectedSlot(132, SlotStatus.OPEN),
        ),
    ),
    # The reviewer approves a *different* time than the one the agent proposed. Nothing else in the
    # suite exercises it: the agent asked for 09:00, the tool result says 15:00, and repeating its
    # own proposal back is the failure — which is exactly what it did until the router stopped
    # dropping the second turn into qa.
    # The args below replace the model's entirely — an edit is a whole argument set, not a patch — so
    # the topic here is the one that lands in the row, whatever wording the model chose.
    # The prompt also asks the agent to *say* the time changed, and this outcome does not require it:
    # gpt-4o-mini confirms the booked time correctly but never contrasts it with the 09:00 it named a
    # turn earlier, on any phrasing of the rule. Asserting it here would keep the suite red over a
    # nicety instead of over the edit itself, so it stays a known gap. A stronger model may close it.
    BookingCase(
        name='reviewer_edits_the_time',
        slots=(open_slot(141, '09:00', '09:30'), open_slot(142, '15:00', '15:30')),
        messages=(
            f'Book me an advisor slot on {SPOKEN_DATE} at 09:00.',
            'iryna.lev@example.com, about the dorm application.',
        ),
        expected_outcome=(
            f'The appointment on {SPOKEN_DATE} was booked at 15:00, not at the 09:00 the assistant '
            f'proposed, and the assistant confirmed 15:00 as the time the applicant has. It must not '
            f'tell the applicant that 09:00 was booked or confirmed. Whether it also pointed out that '
            f'15:00 differs from the 09:00 it suggested earlier does not affect this judgement.'
        ),
        expected_pauses=('book_appointment',),
        expected_slots=(
            ExpectedSlot(141, SlotStatus.OPEN),
            ExpectedSlot(142, SlotStatus.BOOKED, 'iryna.lev@example.com'),
        ),
        decisions=(
            ReviewerDecision(
                type='edit',
                args={
                    'slot_id': 142,
                    'date': ISO_BOOKING_DATE,
                    'start_time': '15:00',
                    'applicant_email': 'iryna.lev@example.com',
                    'topic': 'dorm application',
                },
            ),
        ),
    ),
]
