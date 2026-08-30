from dataclasses import dataclass, field
import logging
from typing import Any
import uuid

from deepeval.test_case import ConversationalTestCase, Turn
from httpx import AsyncClient
import pytest
from types_aiobotocore_dynamodb import DynamoDBClient

from app.ai_assistant_langchain.main_graph import build_main_graph
from tests.conftest import TestBaseClientClass
from tests.factories.factory_creators import reset_test_slot_rows
from tests.factories.factory_getters import get_test_slots_on_date
from tests.integration.booking_conversations import (
    APPROVE,
    BOOKING_CONVERSATIONS,
    BookingCase,
    ISO_BOOKING_DATE,
)
from tests.integration.metrics import assert_metrics, conversation_metrics
from tests.integration.thread_state import get_tool_responses

# Session loop for the same reason as the neighbouring suites: the tests drive the session-scoped
# ``not_auth_client``. ``eval_checkpointer`` but not ``knowledge_base_populated`` — booking reads
# DynamoDB and never touches Qdrant.
pytestmark = [
    pytest.mark.evaluation,
    pytest.mark.asyncio(loop_scope='session'),
    pytest.mark.usefixtures('eval_checkpointer', 'eval_conversations_table'),
]

logger = logging.getLogger(__name__)

ENDPOINT = '/langchain-assistant'
PENDING = 'pending_approval'

# One applicant message can pause more than once — a reschedule books and then cancels — but not
# indefinitely. Without this a model that re-issues a rejected booking turns the resume loop into a
# spend-money-forever loop.
MAX_PAUSES_PER_MESSAGE = 3


@dataclass(slots=True, frozen=True)
class ConversationRun:
    turns: list[Turn] = field(default_factory=list)
    """The dialogue as DeepEval scores it. The reviewer's approvals are deliberately **not** turns:
    the applicant never sees the review happen, the booking prompt says so, and a judge shown an
    ``[approved]`` line would start scoring the reviewer's work as if the assistant had said it."""

    pauses: list[dict[str, Any]] = field(default_factory=list)
    """Every ``pending`` payload the endpoint returned, in order."""


async def run_booking_conversation(client: AsyncClient, user_id: str, case: BookingCase) -> ConversationRun:
    """Post every message in order on one thread, answering each approval pause as the case scripts it."""
    run = ConversationRun()
    decisions = list(case.decisions)
    seen = 0

    for message in case.messages:
        response = await client.post(ENDPOINT, json={'query': message, 'user_id': user_id})
        assert response.status_code == 200, response.text
        reply = response.json()

        pauses_left = MAX_PAUSES_PER_MESSAGE
        while reply.get('status') == PENDING:
            assert pauses_left, f'still paused after {MAX_PAUSES_PER_MESSAGE} rounds of decisions on {message!r}'
            pauses_left -= 1

            pending = reply['pending']
            assert pending, f'the endpoint reported {PENDING} without a `pending` payload: {reply}'
            run.pauses.extend(pending)

            # One decision per action, in the order the pause listed them — a single model turn can
            # ask for two, and the middleware answers all of them or none.
            answers = [decisions.pop(0) if decisions else APPROVE for _ in pending]
            response = await client.post(
                ENDPOINT,
                json={'user_id': user_id, 'decisions': [answer.as_payload() for answer in answers]},
            )
            assert response.status_code == 200, response.text
            reply = response.json()

        grounds, seen = await get_tool_responses(build_main_graph(), user_id, seen)
        run.turns.append(Turn(role='user', content=message))
        run.turns.append(Turn(role='assistant', content=reply['message'], retrieval_context=grounds or None))

    return run


class TestBookingConversations(TestBaseClientClass):
    @pytest.mark.parametrize('case', BOOKING_CONVERSATIONS, ids=lambda case: case.name)
    async def test_booking_conversation(self, case: BookingCase, session_dynamo_client: DynamoDBClient) -> None:
        await reset_test_slot_rows(session_dynamo_client, case.slots)

        run = await run_booking_conversation(self.not_auth_client, str(uuid.uuid4()), case)

        logger.info(
            '%s transcript\n%s\n  pauses: %s',
            case.name,
            '\n'.join(f'  {turn.role:>9}: {turn.content}' for turn in run.turns),
            [pause['action'] for pause in run.pauses],
        )

        slots = await get_test_slots_on_date(ISO_BOOKING_DATE, session_dynamo_client)
        for expected in case.expected_slots:
            assert expected.id in slots, f'slot {expected.id} is gone from the table (found {sorted(slots)})'
            actual = slots[expected.id]
            assert (actual.status, actual.booked_by) == (expected.status, expected.booked_by), (
                f'slot {expected.id} is {actual.status} / {actual.booked_by!r}, '
                f'expected {expected.status} / {expected.booked_by!r}'
            )

        assert [pause['action'] for pause in run.pauses] == list(case.expected_pauses)

        test_case = ConversationalTestCase(
            name=case.name,
            expected_outcome=case.expected_outcome,
            turns=run.turns,
        )

        await assert_metrics(test_case, conversation_metrics())
