from typing import Generator
from unittest.mock import patch
from uuid import uuid4

from langchain_core.messages import AIMessage
import pytest
from types_aiobotocore_dynamodb import DynamoDBClient

from app.ai_assistant_langchain.history.repository import ConversationHistoryRepository
from app.ai_assistant_langchain.history.schemas import NewMessage, Role, utc_now
from app.ai_assistant_langchain.schemas import ChatSchemaOut, PendingApproval
from app.core.dynamodb.base_service import DynamoDBService
from app.settings import get_settings
from tests.api.factories import PATH_TO_KNOWLEDGE_SERVICE, StubKnowledgeService, tool_call_reply
from tests.api.test_chat_checkpointer import TestBaseCheckpointerClass
from tests.conftest import TestBaseAgentClass

ENDPOINT = '/langchain-assistant'
HISTORY_ENDPOINT = ENDPOINT + '/{user_id}/history'

# When each line of a two-turn conversation was said. Fixed rather than generated, so what a replay
# is asserted to return is an order somebody can read off the page — and in the past, so a turn
# posted during a test, stamped from the real clock, always sorts after a seeded one.
ASKED_AT, ANSWERED_AT = '2026-08-01T09:00:00.000000Z', '2026-08-01T09:00:05.000000Z'
ASKED_AGAIN_AT, ANSWERED_AGAIN_AT = '2026-08-01T09:01:00.000000Z', '2026-08-01T09:01:04.000000Z'

PAUSED = ChatSchemaOut(
    status='pending_approval',
    message='Please confirm book_appointment',
    pending=[
        PendingApproval(
            action='book_appointment',
            args={'slot_id': 101, 'date': '2026-10-06'},
            allowed_decisions=['approve', 'reject'],
        )
    ],
)


@pytest.fixture
def knowledge() -> Generator[StubKnowledgeService, None, None]:
    stub = StubKnowledgeService('Bachelor of Computer Science, 4 years, 12000 EUR per year.')
    with patch(PATH_TO_KNOWLEDGE_SERVICE, return_value=stub):
        yield stub


class TestChatHistoryEndpoint(TestBaseAgentClass):
    """``GET /langchain-assistant/{user_id}/history`` — a read, and only a read.

    The rows are seeded through the repository rather than posted, because a replay is a query
    against ``conversation_history_test``: no model is involved in one, so none is set up here. The
    checkpointer is (``TestBaseAgentClass``) — the endpoint also asks the thread whether it is
    paused, and that reads ``agent_checkpoints_test``.
    """

    @pytest.fixture(autouse=True)
    def _a_provide_repository(self, dynamo_client: DynamoDBClient, conversations_table: DynamoDBService) -> None:
        self.repository = ConversationHistoryRepository(dynamo_client, get_settings())
        self.user_id = str(uuid4())

    async def _seed_first_turn(self, user_id: str) -> None:
        await self.repository.append_turn(
            user_id,
            [
                NewMessage(Role.USER, 'How long is the CS bachelor?', ASKED_AT),
                NewMessage(Role.ASSISTANT, 'Four years.', ANSWERED_AT),
            ],
        )

    async def test_a_stored_turn_is_replayed(self) -> None:
        await self._seed_first_turn(self.user_id)

        response = await self.not_auth_client.get(HISTORY_ENDPOINT.format(user_id=self.user_id))

        assert response.status_code == 200
        assert response.json() == [
            {'role': 'user', 'content': 'How long is the CS bachelor?', 'pending': None},
            {'role': 'assistant', 'content': 'Four years.', 'pending': None},
        ]

    async def test_turns_replay_in_the_order_they_were_had(self) -> None:
        await self._seed_first_turn(self.user_id)
        await self.repository.append_turn(
            self.user_id,
            [
                NewMessage(Role.USER, 'How much does it cost?', ASKED_AGAIN_AT),
                NewMessage(Role.ASSISTANT, '12000 EUR per year.', ANSWERED_AGAIN_AT),
            ],
        )

        response = await self.not_auth_client.get(HISTORY_ENDPOINT.format(user_id=self.user_id))

        assert response.status_code == 200
        assert [line['content'] for line in response.json()] == [
            'How long is the CS bachelor?',
            'Four years.',
            'How much does it cost?',
            '12000 EUR per year.',
        ]

    async def test_a_turn_still_waiting_for_its_answer_replays_as_it_stands(self) -> None:
        """What a thread paused on an approval looks like from the table's side: the question is
        there, the answer is not yet. The ``pending`` row a live pause adds comes from the graph, and
        this thread has no checkpoint to be paused on."""
        await self.repository.append_turn(self.user_id, [NewMessage(Role.USER, 'Book me a slot on Tuesday', ASKED_AT)])

        response = await self.not_auth_client.get(HISTORY_ENDPOINT.format(user_id=self.user_id))

        assert response.status_code == 200
        assert response.json() == [{'role': 'user', 'content': 'Book me a slot on Tuesday', 'pending': None}]

    async def test_histories_do_not_leak_between_users(self) -> None:
        other_user = str(uuid4())
        await self._seed_first_turn(self.user_id)
        await self.repository.append_turn(other_user, [NewMessage(Role.USER, 'Where is the library?', utc_now())])

        response = await self.not_auth_client.get(HISTORY_ENDPOINT.format(user_id=other_user))

        assert response.status_code == 200
        assert [line['content'] for line in response.json()] == ['Where is the library?']

    async def test_a_conversation_that_has_not_started_replays_empty(self) -> None:
        """Not a 404: a client opening a fresh thread would have to treat its own first visit as an
        error."""
        response = await self.not_auth_client.get(HISTORY_ENDPOINT.format(user_id=str(uuid4())))

        assert response.status_code == 200
        assert response.json() == []


class TestChatTurnIsRecorded(TestBaseCheckpointerClass):
    """``POST /langchain-assistant`` — the writing half, so here a turn is actually run.

    The graph is the production one and its checkpointer is the production one; what is scripted is
    the model, because a real one would put an OpenAI call (and its bill, and its variability) into
    every ``pytest``. The rows it writes are read back from the real table.
    """

    @pytest.fixture(autouse=True)
    def _a_provide_repository(self, dynamo_client: DynamoDBClient, conversations_table: DynamoDBService) -> None:
        self.repository = ConversationHistoryRepository(dynamo_client, get_settings())

    async def test_a_turn_stores_the_question_and_the_answer(self) -> None:
        user_id = str(uuid4())
        self.stub_model.replies = [AIMessage(content='Four years.')]

        response = await self.not_auth_client.post(
            ENDPOINT, json={'query': 'How long is the CS bachelor?', 'user_id': user_id}
        )

        assert response.status_code == 200
        stored = await self.repository.list_message_by_user_id(user_id)
        assert [(message.role, message.content) for message in stored] == [
            (Role.USER, 'How long is the CS bachelor?'),
            (Role.ASSISTANT, 'Four years.'),
        ]

    async def test_the_stored_turn_is_what_was_said_not_the_agent_s_working_notes(
        self, knowledge: StubKnowledgeService
    ) -> None:
        """The turn calls a tool, so the graph's state holds four messages: the question, the call,
        its result and the answer. Two of them are the agent's notes and were never shown to
        anybody — the transcript is the other two."""
        user_id = str(uuid4())
        self.stub_model.replies = [
            tool_call_reply('computer science tuition'),
            AIMessage(content='Tuition is 12000 EUR per year.'),
        ]

        await self.not_auth_client.post(ENDPOINT, json={'query': 'How much is CS tuition?', 'user_id': user_id})

        state = await self.agent.aget_state({'configurable': {'thread_id': user_id}})
        assert len(state.values['messages']) == 4
        stored = await self.repository.list_message_by_user_id(user_id)
        assert [(message.role, message.content) for message in stored] == [
            (Role.USER, 'How much is CS tuition?'),
            (Role.ASSISTANT, 'Tuition is 12000 EUR per year.'),
        ]

    async def test_a_turn_is_appended_after_the_conversation_it_continues(self) -> None:
        """The earlier turn is seeded rather than posted: what is under test is where the new one
        lands, and a second live turn would only be a slower way of arranging that."""
        user_id = str(uuid4())
        await self.repository.append_turn(
            user_id,
            [
                NewMessage(Role.USER, 'How long is the CS bachelor?', ASKED_AT),
                NewMessage(Role.ASSISTANT, 'Four years.', ANSWERED_AT),
            ],
        )
        self.stub_model.replies = [AIMessage(content='12000 EUR per year.')]

        await self.not_auth_client.post(ENDPOINT, json={'query': 'How much does it cost?', 'user_id': user_id})

        stored = await self.repository.list_message_by_user_id(user_id)
        assert [(message.role, message.content) for message in stored] == [
            (Role.USER, 'How long is the CS bachelor?'),
            (Role.ASSISTANT, 'Four years.'),
            (Role.USER, 'How much does it cost?'),
            (Role.ASSISTANT, '12000 EUR per year.'),
        ]

    async def test_a_turn_is_stored_under_the_user_who_had_it(self) -> None:
        other_user, user_id = str(uuid4()), str(uuid4())
        await self.repository.append_turn(other_user, [NewMessage(Role.USER, 'Where is the library?', ASKED_AT)])
        self.stub_model.replies = [AIMessage(content='12000 EUR per year.')]

        await self.not_auth_client.post(ENDPOINT, json={'query': 'How much does it cost?', 'user_id': user_id})

        stored = await self.repository.list_message_by_user_id(user_id)
        assert [(message.role, message.content) for message in stored] == [
            (Role.USER, 'How much does it cost?'),
            (Role.ASSISTANT, '12000 EUR per year.'),
        ]
        assert [message.content for message in await self.repository.list_message_by_user_id(other_user)] == [
            'Where is the library?'
        ]
