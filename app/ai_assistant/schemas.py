from typing import Literal

from pydantic import BaseModel, Field

from app.ai_assistant.tools.booking_tools import ProposedSlot

IntentType = Literal['booking', 'cancel', 'qa', 'compare']


class IntentDecision(BaseModel):
    """Structured output for intent routing."""

    intent: IntentType = Field(description='The single best-matching intent for the user message.')


class ConfirmationCheck(BaseModel):
    """Structured output: is the message a reply to the pending booking confirmation?"""

    kind: Literal['reply', 'new'] = Field(
        description=(
            '"reply" if the message answers the pending booking confirmation (accepting, '
            'declining, or asking for another time); "new" if it is an unrelated new request.'
        )
    )


class AssistantState(BaseModel):
    question: str = ''
    answer: str = ''
    history: str = ''
    session_id: str = 'default'  # real value arrives via kickoff(inputs=...); default lets Flow build initial state
    # The slot currently proposed and awaiting the human's confirmation. Kept in state so
    # it survives the @human_feedback pause/resume (local variables do not).
    proposed: ProposedSlot | None = None
    # Base64-encoded documents uploaded with this turn, if any (strings so they survive the
    # JSON state persistence). Their presence routes to verification.
    documents: list[str] = []
