#!/usr/bin/env python
import logging
from typing import Literal

from crewai.flow import Flow, human_feedback, listen, or_, PendingFeedbackContext, persist, router, start
from crewai.flow.async_feedback import HumanFeedbackPending

from app.ai_assistant.compare_flow import CompareProgramsFlow
from app.ai_assistant.crews.booking_crew.booking_crew import BookingCrew
from app.ai_assistant.crews.university_crew.university_crew import UniversityCrew
from app.ai_assistant.doc_verification.pipeline import verify
from app.ai_assistant.main_flow_service import MainFlowService
from app.ai_assistant.persistence import FLOW_PERSISTENCE
from app.ai_assistant.schemas import AssistantState
from app.ai_assistant.tools.booking_tools import ProposedSlot
from app.core.dynamodb.slots_repository import open_slots_repository
from app.settings import get_settings

logger = logging.getLogger(__name__)

# Slots booked by each session, so a later "cancel" request knows what to free. We keep the whole
# ProposedSlot (not just the id) because cancelling needs the slot's DynamoDB key (date + start).
SESSION_BOOKINGS: dict[str, list[ProposedSlot]] = {}  # Todo move to DynamoDb


class DeferProvider:
    """Non-blocking human-feedback provider: instead of reading the console, it pauses
    the flow so a later HTTP request (via ``resume_async``) can deliver the reply."""

    def request_feedback(self, context: PendingFeedbackContext, flow: Flow):
        raise HumanFeedbackPending(context=context)


@persist(persistence=FLOW_PERSISTENCE)  # in-memory store so a paused flow can be restored to resume
class UniversityAssistantFlow(Flow[AssistantState]):
    # The LLM-backed helper service, injected as a required dependency (shared/mockable).
    # A bare underscore annotation makes pydantic treat it as a private attr, not a field.
    _service: MainFlowService

    def __init__(self, service: MainFlowService, **kwargs):
        super().__init__(**kwargs)
        self._service = service

    @start()
    def receive_question(self):
        # question / session_id / documents arrive via kickoff(inputs=...) and are bound
        # straight into state by CrewAI before this runs, so there is nothing to unpack here —
        # this @start just anchors the flow entry that route_intent branches from.
        logger.info('Question: %s', self.state.question)

    @router(receive_question)
    async def route_intent(self) -> Literal['qa', 'booking', 'cancel', 'compare', 'verify']:
        if self.state.documents:
            logger.info('Routed intent: verify (%d document(s) attached)', len(self.state.documents))
            return 'verify'

        intent = await self._service.classify_intent(self.state.question)
        logger.info('Routed intent: %s', intent)
        return intent

    @listen('compare')
    async def compare_programs(self):
        # Delegate to the standalone fan-out/fan-in sub-flow, then adopt its answer.
        logger.info('Comparing programs for: %s', self.state.question)
        sub = CompareProgramsFlow()
        await sub.kickoff_async(inputs={'question': self.state.question})
        self.state.answer = sub.state.answer

    @listen('verify')
    async def verify_documents(self):
        # Single vision call: the LLM judges all documents at once and returns the message.
        logger.info('Verifying %d document(s)', len(self.state.documents))
        verdict = await verify(self.state.documents)
        self.state.answer = verdict.message

    @listen('qa')
    async def answer_question(self):
        logger.info('Answering: %s', self.state.question)
        result = (
            await UniversityCrew()
            .crew()
            .kickoff_async(inputs={'question': self.state.question, 'history': self.state.history})
        )
        self.state.answer = result.raw
        logger.info('Answer generated')

    @human_feedback(
        message='Please confirm the proposed consultation slot: reply yes, no, or suggest another time.',
        emit=['approve', 'reject', 'change'],
        llm=get_settings().MODEL_NAME,
        provider=DeferProvider(),
        default_outcome='reject',
    )
    @listen(or_('booking', 'change'))
    async def propose_booking(self):
        request = self._build_proposal_request()
        logger.info('Proposing a consultation slot for: %s', request)

        result = await BookingCrew().crew().kickoff_async(inputs={'question': request, 'history': self.state.history})
        proposal = await self._service.resolve_slot(result.pydantic, self.state.proposed)

        if proposal is None:
            self.state.answer = await self._service.phrase(
                'There are no open consultation slots at all right now.', request
            )
            return self.state.answer

        self.state.proposed = proposal
        # The proposal text is written by the booking agent itself (its `message` field).
        self.state.answer = proposal.message or await self._service.phrase(
            f'Propose an open consultation on {proposal.date} at {proposal.start_time} about '
            f'{proposal.topic}; ask them to confirm or suggest another time.',
            request,
        )
        logger.info('Proposed slot %s; awaiting confirmation', proposal.slot_id)
        return self.state.answer

    def _build_proposal_request(self) -> str:
        """The request handed to BookingCrew: the human's message, plus (on a 'change'
        re-entry) context about the slot already proposed so wishes like "next day" have
        an anchor."""
        feedback = self.last_human_feedback
        request = feedback.feedback if feedback and feedback.feedback else self.state.question
        if self.state.proposed:
            p = self.state.proposed
            request += (
                f' (Context: you already proposed slot id {p.slot_id} on {p.date} at '
                f'{p.start_time}. The applicant asked for something different — read their words '
                f'literally: another DAY → a slot on a different date; another time → a different '
                f'time. Pick an OPEN slot accordingly, and NOT slot id {p.slot_id}.)'
            )
        return request

    @listen('approve')
    async def do_book(self):
        # The human approved -> the write is done here, deterministically, by code.
        feedback = self.last_human_feedback
        user_message = feedback.feedback if feedback else self.state.question
        proposed = self.state.proposed
        who = proposed.applicant_name or self.state.session_id
        async with open_slots_repository() as repo:
            booked = await repo.book_slot(proposed.date, proposed.start_time, proposed.slot_id, who, proposed.topic)
        if booked:
            SESSION_BOOKINGS.setdefault(self.state.session_id, []).append(proposed)
            self.state.answer = await self._service.phrase(
                f'The consultation is now booked for {proposed.date} at '
                f'{proposed.start_time} about {proposed.topic}. Confirm it warmly.',
                user_message,
            )
        else:
            self.state.answer = await self._service.phrase(
                f'The slot on {proposed.date} at {proposed.start_time} is no longer available; '
                f'suggest asking for another time.',
                user_message,
            )
        logger.info('Booked slot %s: %s', proposed.slot_id, booked)

    @listen('reject')
    async def do_reject(self):
        feedback = self.last_human_feedback
        user_message = feedback.feedback if feedback else self.state.question
        self.state.answer = await self._service.phrase(
            'The applicant declined; nothing was booked and the request is cancelled.',
            user_message,
        )
        logger.info('Booking rejected for slot %s', self.state.proposed.slot_id if self.state.proposed else None)

    @listen('cancel')
    async def cancel_booking(self):
        # Cancel the slots this session booked. The write is plain code, like booking.
        bookings = SESSION_BOOKINGS.get(self.state.session_id, [])
        cancelled: list[ProposedSlot] = []
        remaining: list[ProposedSlot] = []
        async with open_slots_repository() as repo:
            for slot in bookings:
                if await repo.cancel_slot(slot.date, slot.start_time, slot.slot_id):
                    cancelled.append(slot)
                else:
                    remaining.append(slot)
        SESSION_BOOKINGS[self.state.session_id] = remaining
        if cancelled:
            self.state.answer = await self._service.phrase(
                f"Cancelled the applicant's consultation (slot id {cancelled[0].slot_id}); the time is free again.",
                self.state.question,
            )
        else:
            self.state.answer = await self._service.phrase(
                'The applicant has no active consultation booking to cancel.',
                self.state.question,
            )
        logger.info('Cancelled slots %s for session %s', [s.slot_id for s in cancelled], self.state.session_id)

    @listen(or_(answer_question, do_book, do_reject, cancel_booking, compare_programs, verify_documents))
    def show_answer(self):
        logger.info('Answer: %s', self.state.answer)
