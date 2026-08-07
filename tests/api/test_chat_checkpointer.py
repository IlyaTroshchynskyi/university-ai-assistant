from typing import Generator
from unittest.mock import AsyncMock, patch
from uuid import uuid4

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
import pytest

from app.ai_assistant_langchain.schemas import ChatSchemaOut
from tests.conftest import TestBaseClientAgentClass
from tests.factories.factory_creators import seed_agent_state

PATH_TO_RUN_AGENT = 'app.ai_assistant_langchain.agent_service.AgentService.run_agent'
PATH_TO_KNOWLEDGE_SERVICE = 'app.ai_assistant_langchain.tools.get_knowledge_service'
ENDPOINT = '/langchain-assistant'


class StubKnowledgeService:
    """Replaces the Qdrant-backed service so the retriever tool never leaves the process."""

    def __init__(self, passages: str) -> None:
        self.passages = passages
        self.queries: list[str] = []

    async def search(self, query: str) -> str:
        self.queries.append(query)
        return self.passages


@pytest.fixture
def knowledge() -> Generator[StubKnowledgeService, None, None]:
    stub = StubKnowledgeService('Bachelor of Computer Science, 4 years, 12000 EUR per year.')
    with patch(PATH_TO_KNOWLEDGE_SERVICE, return_value=stub):
        yield stub


def tool_call_reply(query: str, call_id: str = 'call_1') -> AIMessage:
    return AIMessage(content='', tool_calls=[{'id': call_id, 'name': 'retriever', 'args': {'query': query}}])


class TestChatWithUserCheckpointer(TestBaseClientAgentClass):
    """Tests for ``POST /langchain-assistant`` with ``run_agent`` mocked out."""

    async def test_chat_success(self) -> None:
        """A valid query gets the assistant's last message back."""
        reply = 'The Computer Science bachelor runs for 4 years.'
        with patch(
            PATH_TO_RUN_AGENT, new_callable=AsyncMock, return_value={'messages': [AIMessage(content=reply)]}
        ) as agent_mock:
            response = await self.not_auth_client.post(
                ENDPOINT, json={'query': 'How long is the CS bachelor?', 'user_id': str(uuid4())}
            )

        assert response.status_code == 200
        data = ChatSchemaOut.model_validate(response.json())
        assert data.message == reply
        agent_mock.assert_awaited_once()

    async def test_chat_passes_user_id_as_thread_id(self) -> None:
        """The user_id from the body is what keys the conversation thread."""
        user_id = str(uuid4())
        with patch(
            PATH_TO_RUN_AGENT, new_callable=AsyncMock, return_value={'messages': [AIMessage(content='Hi!')]}
        ) as agent_mock:
            await self.not_auth_client.post(ENDPOINT, json={'query': 'Hello', 'user_id': user_id})

        messages, kwargs = agent_mock.await_args.args[0], agent_mock.await_args.kwargs
        assert kwargs['user_id'] == user_id
        assert [message.content for message in messages] == ['Hello']


class TestChatMemoryCheckpointer(TestBaseClientAgentClass):
    """Tests for the conversation memory — ``run_agent`` is not mocked at all, so these read and
    write real state in the shared ``InMemorySaver``."""

    async def test_first_turn_sends_system_prompt_and_query_only(self) -> None:
        """A brand-new thread carries nothing but the system prompt and the incoming question."""
        self.stub_model.replies = [AIMessage(content='Hi!')]

        response = await self.not_auth_client.post(ENDPOINT, json={'query': 'Hello', 'user_id': str(uuid4())})

        assert response.status_code == 200
        sent = self.stub_model.calls[0]
        assert [type(message) for message in sent] == [SystemMessage, HumanMessage]
        assert sent[-1].content == 'Hello'

    async def test_seeded_history_is_replayed_to_the_model(self) -> None:
        """
        Seeds real messages directly into the shared InMemorySaver, then asserts the next request
        replays them — the endpoint itself only ever passes the new message.
        """
        user_id = str(uuid4())
        await seed_agent_state(
            self.agent,
            user_id,
            [HumanMessage(content='Question 1'), AIMessage(content='Answer 1')],
        )
        self.stub_model.replies = [AIMessage(content='Answer 2')]

        response = await self.not_auth_client.post(ENDPOINT, json={'query': 'Question 2', 'user_id': user_id})

        assert response.status_code == 200
        assert [message.content for message in self.stub_model.calls[0][1:]] == [
            'Question 1',
            'Answer 1',
            'Question 2',
        ]

    async def test_second_turn_replays_the_first(self) -> None:
        """Two live turns on one thread: the second call sees everything the first produced."""
        user_id = str(uuid4())
        self.stub_model.replies = [AIMessage(content='Answer 1'), AIMessage(content='Answer 2')]

        await self.not_auth_client.post(ENDPOINT, json={'query': 'Question 1', 'user_id': user_id})
        await self.not_auth_client.post(ENDPOINT, json={'query': 'Question 2', 'user_id': user_id})

        assert [message.content for message in self.stub_model.calls[1][1:]] == [
            'Question 1',
            'Answer 1',
            'Question 2',
        ]

    async def test_threads_are_isolated_per_user(self) -> None:
        """A different user_id is a different thread, so nothing leaks across users."""
        self.stub_model.replies = [AIMessage(content='Answer 1'), AIMessage(content='Answer 2')]

        await self.not_auth_client.post(ENDPOINT, json={'query': 'Question 1', 'user_id': str(uuid4())})
        await self.not_auth_client.post(ENDPOINT, json={'query': 'Question 2', 'user_id': str(uuid4())})

        second_call = self.stub_model.calls[1]
        assert [type(message) for message in second_call] == [SystemMessage, HumanMessage]
        assert 'Question 1' not in [message.content for message in second_call]

    async def test_state_accumulates_for_the_thread(self) -> None:
        """Both turns end up persisted under the thread, replies included."""
        user_id = str(uuid4())
        self.stub_model.replies = [AIMessage(content='Answer 1'), AIMessage(content='Answer 2')]

        await self.not_auth_client.post(ENDPOINT, json={'query': 'Question 1', 'user_id': user_id})
        await self.not_auth_client.post(ENDPOINT, json={'query': 'Question 2', 'user_id': user_id})

        state = await self.agent.aget_state({'configurable': {'thread_id': user_id}})
        assert [message.content for message in state.values['messages']] == [
            'Question 1',
            'Answer 1',
            'Question 2',
            'Answer 2',
        ]


class TestChatToolCallsCheckpointer(TestBaseClientAgentClass):
    """Tool calls are the part of the history that breaks most easily: OpenAI rejects an AIMessage
    whose tool_calls have no matching ToolMessage, so the pairing has to survive in state."""

    async def test_tool_call_and_result_are_stored_as_a_pair(self, knowledge: StubKnowledgeService) -> None:
        """The retriever call and its result land in state with matching ids."""
        user_id = str(uuid4())
        self.stub_model.replies = [
            tool_call_reply('computer science tuition'),
            AIMessage(content='Tuition is 12000 EUR per year.'),
        ]

        response = await self.not_auth_client.post(
            ENDPOINT, json={'query': 'How much is CS tuition?', 'user_id': user_id}
        )

        assert response.status_code == 200
        assert knowledge.queries == ['computer science tuition']

        state = await self.agent.aget_state({'configurable': {'thread_id': user_id}})
        messages = state.values['messages']
        assert [type(message) for message in messages] == [HumanMessage, AIMessage, ToolMessage, AIMessage]
        assert messages[1].tool_calls[0]['id'] == messages[2].tool_call_id

    async def test_tool_result_is_replayed_on_the_next_turn(self, knowledge: StubKnowledgeService) -> None:
        """The next turn resends the whole pair, which is what keeps the request valid."""
        user_id = str(uuid4())
        self.stub_model.replies = [
            tool_call_reply('computer science tuition'),
            AIMessage(content='Tuition is 12000 EUR per year.'),
            AIMessage(content='Yes, that covers lab fees.'),
        ]

        await self.not_auth_client.post(ENDPOINT, json={'query': 'How much is CS tuition?', 'user_id': user_id})
        await self.not_auth_client.post(ENDPOINT, json={'query': 'Does that include lab fees?', 'user_id': user_id})

        third_call = self.stub_model.calls[2]
        assert [type(message) for message in third_call] == [
            SystemMessage,
            HumanMessage,
            AIMessage,
            ToolMessage,
            AIMessage,
            HumanMessage,
        ]
        assert third_call[3].content == knowledge.passages
