import base64

from crewai import LLM
from crewai.flow.async_feedback import HumanFeedbackPending

from app.ai_assistant.conversation_store import CONVERSATION_STORE
from app.ai_assistant.persistence import FLOW_PERSISTENCE
from app.ai_assistant.schemas import ConfirmationCheck, IntentDecision, IntentType
from app.ai_assistant.tools.booking_tools import list_open_slots, ProposedSlot

DEFAULT_SESSION = 'default'
MAX_HISTORY_MESSAGES = 10  # how many recent messages to feed back into the prompt

# Sessions whose flow is paused waiting for a human's booking confirmation, mapped to the
# paused flow's id so the next message can resume it. Module-level so it survives across the
# per-request service instances. TODO: swap for a real store.
SESSION_FLOWS: dict[str, str] = {}


class MainFlowService:
    def __init__(self, llm: LLM):
        self._llm = llm

    async def answer_question(
        self, question: str, session_id: str = DEFAULT_SESSION, documents: list[bytes] | None = None
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
        flow, result = await self._resume_or_start(question, history, session_id, documents_b64)

        answer = flow.state.answer
        if isinstance(result, HumanFeedbackPending):
            # Still mid-booking: remember the paused flow so the next reply resumes it.
            SESSION_FLOWS[session_id] = flow.state.id
        else:
            SESSION_FLOWS.pop(session_id, None)

        CONVERSATION_STORE.add(session_id, 'user', question)
        CONVERSATION_STORE.add(session_id, 'assistant', answer)
        return answer

    async def _resume_or_start(self, question: str, history: list[dict], session_id: str, documents: list[str]):
        """Resume the session's paused booking if this message answers the confirmation;
        otherwise start a fresh flow (abandoning any stale pause)."""
        # Imported lazily: main_flow imports this module, so a top-level import would cycle.
        from app.ai_assistant.main_flow import UniversityAssistantFlow

        paused_flow_id = SESSION_FLOWS.get(session_id)
        if paused_flow_id:
            flow = UniversityAssistantFlow.from_pending(paused_flow_id, persistence=FLOW_PERSISTENCE, service=self)
            if await self._is_confirmation_reply(question, flow.state.proposed):
                # Refresh history so a re-proposal ('change') keeps the full context
                # (original topic, the slot already proposed), not the stale kickoff snapshot.
                flow.state.history = self._format_history(history)
                return flow, await flow.resume_async(feedback=question)
            # User changed the subject — drop the pending booking and handle fresh.
            self._abandon_pending(session_id, paused_flow_id)

        flow = UniversityAssistantFlow(self)
        result = await flow.kickoff_async(
            inputs={
                'question': question,
                'history': self._format_history(history),
                'session_id': session_id,
                'documents': documents,
            }
        )
        return flow, result

    @staticmethod
    def _abandon_pending(session_id: str, flow_id: str) -> None:
        """Drop a paused booking the user walked away from."""
        FLOW_PERSISTENCE.clear_pending_feedback(flow_id)
        SESSION_FLOWS.pop(session_id, None)

    async def phrase(self, facts: str, user_message: str) -> str:
        """Let the LLM write a natural reply from a few facts (no hardcoded user strings).

        The reply is written in the same language the user used in ``user_message``.
        """
        prompt = (
            'You are a friendly university admissions assistant. Write a short, natural reply '
            '(one or two sentences) to the applicant based only on these facts — do not invent '
            'anything beyond them. Reply in the SAME language the applicant used in their '
            f'message below.\n\nApplicant message: "{user_message}"\nFacts: {facts}'
        )
        return (await self._llm.acall(prompt)).strip()

    async def classify_intent(self, message: str) -> IntentType:
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
        return (await self._llm.acall(prompt, response_model=IntentDecision)).intent

    @staticmethod
    def resolve_slot(proposal: ProposedSlot | None, previous: ProposedSlot | None) -> ProposedSlot | None:
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

    async def _is_confirmation_reply(self, message: str, proposed: ProposedSlot | None) -> bool:
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
        return (await self._llm.acall(prompt, response_model=ConfirmationCheck)).kind == 'reply'

    @staticmethod
    def _format_history(history: list[dict]) -> str:
        """Render recent conversation turns for the prompt, or '' if there are none."""
        if not history:
            return ''
        recent = history[-MAX_HISTORY_MESSAGES:]
        lines = [f'{m["role"]}: {m["content"]}' for m in recent]
        return 'Conversation so far:\n' + '\n'.join(lines) + '\n'
