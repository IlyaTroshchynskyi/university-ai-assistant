#!/usr/bin/env python
import asyncio
import base64
import logging
from typing import Literal

from crewai import LLM
from crewai.flow import Flow, human_feedback, listen, or_, PendingFeedbackContext, persist, router, start
from crewai.flow.async_feedback import HumanFeedbackPending
from dotenv import load_dotenv
from pydantic import BaseModel, Field

from app.ai_assistant.compare_flow import CompareProgramsFlow
from app.ai_assistant.conversation_store import CONVERSATION_STORE
from app.ai_assistant.crews.booking_crew.booking_crew import BookingCrew
from app.ai_assistant.crews.university_crew.university_crew import UniversityCrew
from app.ai_assistant.doc_verification.pipeline import verify
from app.ai_assistant.persistence import FLOW_PERSISTENCE
from app.ai_assistant.tools.booking_tools import book_slot, cancel_slot, list_open_slots, ProposedSlot

load_dotenv()  # load OPENAI_API_KEY etc. from .env

logger = logging.getLogger(__name__)

DEFAULT_SESSION = 'default'
MAX_HISTORY_MESSAGES = 10  # how many recent messages to feed back into the prompt

# A small, cheap LLM used to classify intent and phrase replies — more robust than
# keyword matching (handles paraphrases, other languages, indirect phrasing).
_CLASSIFIER_LLM = LLM(model='gpt-4o-mini', temperature=0)

# Conversation history lives in CONVERSATION_STORE (SQLite by default; in-memory optional).

# Sessions whose flow is paused waiting for a human's booking confirmation, mapped to
# the paused flow's id so the next message can resume it. TODO: swap for a real store.
SESSION_FLOWS: dict[str, str] = {}

# Slot ids booked by each session, so a later "cancel" request knows what to free.
SESSION_BOOKINGS: dict[str, list[int]] = {}


class DeferProvider:
    """Non-blocking human-feedback provider: instead of reading the console, it pauses
    the flow so a later HTTP request (via ``resume_async``) can deliver the reply."""

    def request_feedback(self, context: PendingFeedbackContext, flow: Flow):
        raise HumanFeedbackPending(context=context)


def _format_history(history: list[dict]) -> str:
    """Render recent conversation turns for the prompt, or '' if there are none."""
    if not history:
        return ''
    recent = history[-MAX_HISTORY_MESSAGES:]
    lines = [f'{m["role"]}: {m["content"]}' for m in recent]
    return 'Conversation so far:\n' + '\n'.join(lines) + '\n'


class IntentDecision(BaseModel):
    """Structured output for intent routing."""

    intent: Literal['booking', 'cancel', 'qa', 'compare'] = Field(
        description='The single best-matching intent for the user message.'
    )


async def _classify_intent(message: str) -> Literal['booking', 'cancel', 'qa', 'compare']:
    """Decide the user's intent via the LLM (structured output, not keywords)."""
    prompt = (
        'You classify a user message for a university assistant. Choose the single best '
        'intent:\n'
        '- "cancel" if the user wants to cancel or call off an existing consultation or '
        'appointment they already have.\n'
        '- "compare" if the user wants to compare two university programs against each '
        'other.\n'
        '- "booking" if the user wants to schedule, book, arrange, or reschedule a '
        'consultation.\n'
        '- "qa" for anything else (questions, greetings, small talk).\n\n'
        f'Message: "{message}"'
    )
    return (await _CLASSIFIER_LLM.acall(prompt, response_model=IntentDecision)).intent


class ConfirmationCheck(BaseModel):
    """Structured output: is the message a reply to the pending booking confirmation?"""

    kind: Literal['reply', 'new'] = Field(
        description=(
            '"reply" if the message answers the pending booking confirmation (accepting, '
            'declining, or asking for another time); "new" if it is an unrelated new request.'
        )
    )


async def _is_confirmation_reply(message: str, proposed: ProposedSlot | None) -> bool:
    """True if `message` answers a pending booking confirmation (accept / decline / ask
    for another time), rather than an unrelated new request. Structured output, not keywords."""
    context = ''
    if proposed:
        context = (
            f'The assistant just proposed a consultation slot on {proposed.date} at '
            f'{proposed.start_time} and asked the user to confirm.\n'
        )
    prompt = (
        'A university assistant is waiting for the user to confirm a proposed booking.\n'
        f'{context}'
        "Decide whether the user's next message is a reply to that confirmation "
        '(accepting, declining, or asking for a different time/date) or an unrelated new '
        'request (a different question or a brand-new topic).\n\n'
        f'Message: "{message}"'
    )
    return (await _CLASSIFIER_LLM.acall(prompt, response_model=ConfirmationCheck)).kind == 'reply'


async def _phrase(facts: str, user_message: str) -> str:
    """Let the LLM write a natural reply from a few facts (no hardcoded user strings).

    The reply is written in the same language the user used in ``user_message``.
    """
    prompt = (
        'You are a friendly university admissions assistant. Write a short, natural reply '
        '(one or two sentences) to the applicant based only on these facts — do not invent '
        'anything beyond them. Reply in the SAME language the applicant used in their '
        f'message below.\n\nApplicant message: "{user_message}"\nFacts: {facts}'
    )
    return (await _CLASSIFIER_LLM.acall(prompt)).strip()


def _resolve_slot(proposal: ProposedSlot | None, previous: ProposedSlot | None) -> ProposedSlot | None:
    """Return a valid, genuinely-open slot to propose: the agent's choice if it is open,
    otherwise the first open slot that isn't the one just proposed. None if none are open.

    This keeps the guard that the agent can never invent a slot.
    """
    open_slots = list_open_slots()
    if proposal is not None and proposal.slot_id in {s['id'] for s in open_slots}:
        return proposal

    prev_id = previous.slot_id if previous else None
    candidates = [s for s in open_slots if s['id'] != prev_id]
    if not candidates:
        return None
    s = candidates[0]
    return ProposedSlot(
        slot_id=s['id'],
        date=s['date'],
        start_time=s['start_time'],
        topic='',
        applicant_name='',
        message='',
    )


class AssistantState(BaseModel):
    question: str = ''
    answer: str = ''
    history: str = ''
    session_id: str = DEFAULT_SESSION
    # The slot currently proposed and awaiting the human's confirmation. Kept in state so
    # it survives the @human_feedback pause/resume (local variables do not).
    proposed: ProposedSlot | None = None
    # Base64-encoded documents uploaded with this turn, if any (strings so they survive the
    # JSON state persistence). Their presence routes to verification.
    documents: list[str] = []


@persist(persistence=FLOW_PERSISTENCE)  # in-memory store so a paused flow can be restored to resume
class UniversityAssistantFlow(Flow[AssistantState]):
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

        intent = await _classify_intent(self.state.question)
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
        llm='gpt-4o-mini',
        provider=DeferProvider(),
        default_outcome='reject',
    )
    @listen(or_('booking', 'change'))
    async def propose_booking(self):
        request = self._build_proposal_request()
        logger.info('Proposing a consultation slot for: %s', request)

        result = await BookingCrew().crew().kickoff_async(inputs={'question': request, 'history': self.state.history})
        proposal = _resolve_slot(result.pydantic, self.state.proposed)

        if proposal is None:
            self.state.answer = await _phrase('There are no open consultation slots at all right now.', request)
            return self.state.answer

        self.state.proposed = proposal
        # The proposal text is written by the booking agent itself (its `message` field).
        self.state.answer = proposal.message or await _phrase(
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
        booked = book_slot(proposed.slot_id, who, proposed.topic)
        if booked:
            SESSION_BOOKINGS.setdefault(self.state.session_id, []).append(proposed.slot_id)
            self.state.answer = await _phrase(
                f'The consultation is now booked for {proposed.date} at '
                f'{proposed.start_time} about {proposed.topic}. Confirm it warmly.',
                user_message,
            )
        else:
            self.state.answer = await _phrase(
                f'The slot on {proposed.date} at {proposed.start_time} is no longer available; '
                f'suggest asking for another time.',
                user_message,
            )
        logger.info('Booked slot %s: %s', proposed.slot_id, booked)

    @listen('reject')
    async def do_reject(self):
        feedback = self.last_human_feedback
        user_message = feedback.feedback if feedback else self.state.question
        self.state.answer = await _phrase(
            'The applicant declined; nothing was booked and the request is cancelled.',
            user_message,
        )
        logger.info('Booking rejected for slot %s', self.state.proposed.slot_id if self.state.proposed else None)

    @listen('cancel')
    async def cancel_booking(self):
        # Cancel the slots this session booked. The write is plain code, like booking.
        slot_ids = SESSION_BOOKINGS.get(self.state.session_id, [])
        cancelled = [sid for sid in slot_ids if cancel_slot(sid)]
        SESSION_BOOKINGS[self.state.session_id] = [sid for sid in slot_ids if sid not in cancelled]
        if cancelled:
            self.state.answer = await _phrase(
                f"Cancelled the applicant's consultation (slot id {cancelled[0]}); the time is free again.",
                self.state.question,
            )
        else:
            self.state.answer = await _phrase(
                'The applicant has no active consultation booking to cancel.',
                self.state.question,
            )
        logger.info('Cancelled slots %s for session %s', cancelled, self.state.session_id)

    @listen(or_(answer_question, do_book, do_reject, cancel_booking, compare_programs, verify_documents))
    def show_answer(self):
        logger.info('Answer: %s', self.state.answer)


def _abandon_pending(session_id: str, flow_id: str) -> None:
    """Drop a paused booking the user walked away from."""
    FLOW_PERSISTENCE.clear_pending_feedback(flow_id)
    SESSION_FLOWS.pop(session_id, None)


async def _resume_or_start(question: str, history: list[dict], session_id: str, documents: list[str]):
    """Resume the session's paused booking if this message answers the confirmation;
    otherwise start a fresh flow (abandoning any stale pause)."""
    paused_flow_id = SESSION_FLOWS.get(session_id)
    if paused_flow_id:
        flow = UniversityAssistantFlow.from_pending(paused_flow_id, persistence=FLOW_PERSISTENCE)
        if await _is_confirmation_reply(question, flow.state.proposed):
            # Refresh history so a re-proposal ('change') keeps the full context
            # (original topic, the slot already proposed), not the stale kickoff snapshot.
            flow.state.history = _format_history(history)
            return flow, await flow.resume_async(feedback=question)
        # User changed the subject — drop the pending booking and handle fresh.
        _abandon_pending(session_id, paused_flow_id)

    flow = UniversityAssistantFlow()
    result = await flow.kickoff_async(
        inputs={
            'question': question,
            'history': _format_history(history),
            'session_id': session_id,
            'documents': documents,
        }
    )
    return flow, result


async def answer_question(
    question: str, session_id: str = DEFAULT_SESSION, documents: list[bytes] | None = None
) -> str:
    """Answer a university question, remembering the session's prior messages. ``documents``
    are raw uploaded image bytes; if any are attached, the turn is routed to document
    verification.

    If the session's flow is paused awaiting a booking confirmation and the new message
    answers that confirmation, this resumes it with the message as the human's feedback;
    otherwise it starts a fresh flow.
    """
    history = CONVERSATION_STORE.history(session_id)

    # Encode to base64 so the images are JSON-serializable in the persisted flow state.
    documents_b64 = [base64.b64encode(doc).decode() for doc in documents or []]
    flow, result = await _resume_or_start(question, history, session_id, documents_b64)

    answer = flow.state.answer
    if isinstance(result, HumanFeedbackPending):
        # Still mid-booking: remember the paused flow so the next reply resumes it.
        SESSION_FLOWS[session_id] = flow.state.id
    else:
        SESSION_FLOWS.pop(session_id, None)

    CONVERSATION_STORE.add(session_id, 'user', question)
    CONVERSATION_STORE.add(session_id, 'assistant', answer)
    return answer


# --- Sync entrypoints (for the CLI scripts in pyproject.toml) ---


def kickoff():
    logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')
    flow = UniversityAssistantFlow()
    asyncio.run(flow.kickoff_async())


def plot():
    flow = UniversityAssistantFlow()
    flow.plot()


def run_with_trigger():
    """
    Run the flow with a trigger payload, e.g.:

        run_with_trigger '{"question": "How do I apply for a scholarship?"}'
    """
    import json
    import sys

    logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')

    if len(sys.argv) < 2:
        raise Exception('No trigger payload provided. Please provide JSON payload as argument.')

    try:
        trigger_payload = json.loads(sys.argv[1])
    except json.JSONDecodeError:
        raise Exception('Invalid JSON payload provided as argument')

    flow = UniversityAssistantFlow()

    try:
        return asyncio.run(flow.kickoff_async({'crewai_trigger_payload': trigger_payload}))
    except Exception as e:
        raise Exception(f'An error occurred while running the flow with trigger: {e}')


if __name__ == '__main__':
    kickoff()
