import asyncio
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import AsyncGenerator, Generator
from unittest.mock import AsyncMock, patch
from uuid import uuid4

from fastapi import FastAPI
from httpx import AsyncClient
from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage, HumanMessage, ToolMessage
import pytest
from types_aiobotocore_dynamodb import DynamoDBClient

from app.ai_assistant_langchain import runs
from app.ai_assistant_langchain.agent_schemas import CustomContext
from app.ai_assistant_langchain.agent_service import AgentService
from app.ai_assistant_langchain.enums import GraphNode
from app.ai_assistant_langchain.history.repository import ConversationHistoryRepository
from app.ai_assistant_langchain.history.schemas import Role
from app.ai_assistant_langchain.history.service import ConversationHistoryService
from app.ai_assistant_langchain.schemas import ChatSchemaOut, UserQuery
from app.ai_assistant_langchain.service import ChatService, MODEL_NODE, TOOL_FAILURE_MESSAGE
from app.ai_assistant_langchain.stream_schemas import ErrorEvent, TokenEvent
from app.core.dynamodb.base_service import DynamoDBService
from app.core.exceptions import ConflictingStatusError
from app.lifespan import lifespan
from app.settings import get_settings
from tests.api.factories import (
    booking_tool_call_reply,
    compare_tool_call_reply,
    PATH_TO_KNOWLEDGE_SERVICE,
    StubKnowledgeService,
    tool_call_reply,
)
from tests.api.test_chat_checkpointer import TestBaseCheckpointerClass
from tests.factories.factory_creators import seed_agent_state

ENDPOINT = '/langchain-assistant'
SSE_HEADERS = {'Accept': 'text/event-stream'}

# `merge` gets its model and its knowledge service from its own module, which neither stub helper in
# `tests/agent_stubs.py` reaches.
PATH_TO_COMPARE_MODEL_FACTORY = 'app.ai_assistant_langchain.graphs.compare_programs.nodes.get_model_factory'
PATH_TO_COMPARE_KNOWLEDGE_SERVICE = 'app.ai_assistant_langchain.graphs.compare_programs.nodes.get_knowledge_service'

# §4.4's node name, patched by the test that the whole `ToolMessage` still answers when the tool's
# graph streams nothing this chat recognises.
PATH_TO_DIRECT_ANSWER_NODE = 'app.ai_assistant_langchain.service.DIRECT_ANSWER_NODE'

PATH_TO_CHECKPOINTER = 'app.lifespan.get_checkpointer'
PATH_TO_WAIT_FOR_STREAMING_TURNS = 'app.lifespan.wait_for_streaming_turns'


@pytest.fixture
def knowledge() -> Generator[StubKnowledgeService, None, None]:
    stub = StubKnowledgeService('Bachelor of Computer Science, 4 years, 12000 EUR per year.')
    with patch(PATH_TO_KNOWLEDGE_SERVICE, return_value=stub):
        yield stub


@dataclass(frozen=True, slots=True)
class Event:
    name: str
    data: str


async def read_events(client: AsyncClient, payload: dict[str, str | list[dict]]) -> list[Event]:
    events: list[Event] = []
    async with client.stream('POST', ENDPOINT, json=payload, headers=SSE_HEADERS) as response:
        assert response.status_code == 200, (await response.aread()).decode()
        assert response.headers['content-type'].startswith('text/event-stream')
        name = ''
        async for line in response.aiter_lines():
            if line.startswith('event:'):
                name = line.removeprefix('event:').strip()
            elif line.startswith('data:'):
                events.append(Event(name, line.removeprefix('data:').strip()))
    return events


def streamed_text(events: list[Event]) -> str:
    return ''.join(TokenEvent.model_validate_json(event.data).text for event in events if event.name == 'token')


class TestTheStubStreams(TestBaseCheckpointerClass):
    async def _parts(self, user_id: str, query: str) -> list[dict]:
        return [
            part
            async for part in self.agent.astream(
                {'messages': [HumanMessage(content=query)]},
                config={'configurable': {'thread_id': user_id}},
                context=CustomContext(user_id=user_id),
                stream_mode=['messages', 'values'],
                subgraphs=True,
                version='v2',
            )
        ]

    async def test_the_model_node_streams_chunks(self) -> None:
        user_id = str(uuid4())
        self.stub_model.replies = [AIMessage(content='The CS bachelor runs for four years.')]

        parts = await self._parts(user_id, 'How long is the CS bachelor?')

        chunks = [
            part['data'][0]
            for part in parts
            if part['type'] == 'messages' and part['data'][1].get('langgraph_node') == MODEL_NODE
        ]
        assert len(chunks) > 1, 'the stub answered in one piece — `_stream` is not being used'
        assert all(isinstance(chunk, AIMessageChunk) for chunk in chunks)
        assert ''.join(chunk.text for chunk in chunks) == 'The CS bachelor runs for four years.'

    async def test_the_router_call_streams_its_json_at_the_root(self) -> None:
        user_id = str(uuid4())
        self.stub_model.replies = [AIMessage(content='Hello!')]

        parts = await self._parts(user_id, 'Hi there!')

        routing = [
            part['data'][0].text
            for part in parts
            if part['type'] == 'messages' and part['ns'] == () and part['data'][1].get('langgraph_node') == '__start__'
        ]
        assert ''.join(routing) == '{"destination": "qa"}'


class TestTokenFilters:
    @staticmethod
    def _part(message: BaseMessage, ns: tuple[str, ...], node: str) -> dict:
        return {'type': 'messages', 'ns': ns, 'data': (message, {'langgraph_node': node})}

    def test_a_subagent_token_is_forwarded(self) -> None:
        part = self._part(AIMessageChunk(content='Four years.'), ('qa:1',), 'model')
        assert ChatService._is_answer_token(part) is True

    def test_the_router_json_is_not(self) -> None:
        """§4.1 — it streams at the root on every single turn."""
        part = self._part(AIMessageChunk(content='{"destination": "qa"}'), (), '__start__')
        assert ChatService._is_answer_token(part) is False

    def test_a_graph_nested_in_a_tool_is_not(self) -> None:
        """§4.1 — `compare_programs` runs its own graph one namespace deeper (P3)."""
        part = self._part(AIMessageChunk(content='### Economics'), ('qa:1', 'tools:2'), 'merge')
        assert ChatService._is_answer_token(part) is False

    def test_a_model_node_inside_a_tools_graph_is_not(self) -> None:
        part = self._part(AIMessageChunk(content='### Economics'), ('qa:1', 'tools:2'), 'model')
        assert ChatService._is_answer_token(part) is False

    def test_the_summariser_is_not(self) -> None:
        """§4.2 — same namespace, same message class, different node."""
        part = self._part(
            AIMessageChunk(content='Here is a summary'), ('qa:1',), 'SummarizationMiddleware.before_model'
        )
        assert ChatService._is_answer_token(part) is False

    def test_a_tool_result_is_not(self) -> None:
        part = self._part(ToolMessage(content='passages', tool_call_id='1', name='retriever'), ('qa:1',), 'tools')
        assert ChatService._is_answer_token(part) is False

    def test_an_empty_chunk_is_not(self) -> None:
        part = self._part(AIMessageChunk(content=''), ('qa:1',), 'model')
        assert ChatService._is_answer_token(part) is False

    def test_a_chunk_of_blocks_with_no_text_is_not(self) -> None:
        part = self._part(AIMessageChunk(content=[{'type': 'reasoning', 'reasoning': 'hmm'}]), ('qa:1',), 'model')
        assert ChatService._is_answer_token(part) is False

    def test_the_comparison_graphs_own_tokens_are_the_answer(self) -> None:
        """§4.4 — the one node one namespace down whose text is what the applicant asked for."""
        part = self._part(AIMessageChunk(content='### Economics'), ('qa:1', 'tools:2'), 'merge')
        assert ChatService._is_direct_answer_token(part) is True

    def test_another_node_in_a_tools_graph_is_not_a_direct_answer_token(self) -> None:
        """A tool running a `create_agent` of its own has a node called `model`. §4.4 is one named
        node, not "anything a tool streams"."""
        part = self._part(AIMessageChunk(content='### Economics'), ('qa:1', 'tools:2'), 'model')
        assert ChatService._is_direct_answer_token(part) is False

    def test_the_subagents_own_tokens_are_not_direct_answer_tokens(self) -> None:
        part = self._part(AIMessageChunk(content='Four years.'), ('qa:1',), 'merge')
        assert ChatService._is_direct_answer_token(part) is False

    def test_an_empty_chunk_from_the_comparison_graph_is_not(self) -> None:
        part = self._part(AIMessageChunk(content=''), ('qa:1', 'tools:2'), 'merge')
        assert ChatService._is_direct_answer_token(part) is False

    def test_a_return_direct_result_from_a_nested_graph_is_not_the_answer(self) -> None:
        part = self._part(
            ToolMessage(content='Economics vs Business Analytics', tool_call_id='1', name='compare_programs'),
            ('qa:1', 'tools:2'),
            'tools',
        )
        assert ChatService._is_direct_answer(part) is False

    def test_a_return_direct_result_is_the_answer(self) -> None:
        part = self._part(
            ToolMessage(content='Economics vs Business Analytics', tool_call_id='1', name='compare_programs'),
            ('qa:1',),
            'tools',
        )
        assert ChatService._is_direct_answer(part) is True

    def test_an_ordinary_tool_result_is_not_a_direct_answer(self) -> None:
        part = self._part(ToolMessage(content='passages', tool_call_id='1', name='retriever'), ('qa:1',), 'tools')
        assert ChatService._is_direct_answer(part) is False

    def test_a_failed_return_direct_result_is_not_the_answer(self) -> None:
        part = self._part(
            ToolMessage(
                content='1 validation error for CompareProgramsToolInput\nprograms\n  Field required',
                tool_call_id='1',
                name='compare_programs',
                status='error',
            ),
            ('qa:1',),
            'tools',
        )
        assert ChatService._is_direct_answer(part) is False


class TestWirePayloads:
    def test_a_token_event_is_json_not_a_repr(self) -> None:
        assert TokenEvent(text='Four years.').model_dump_json() == '{"text":"Four years."}'

    def test_an_error_event_carries_a_detail(self) -> None:
        assert ErrorEvent(detail='Something went wrong.').model_dump_json() == '{"detail":"Something went wrong."}'


class TestAgentServiceStreams(TestBaseCheckpointerClass):
    async def test_stream_agent_ends_on_a_root_values_part(self) -> None:
        user_id = str(uuid4())
        self.stub_model.replies = [AIMessage(content='Four years.')]
        service = AgentService(self.agent)

        parts = [part async for part in service.stream_agent([HumanMessage(content='How long?')], user_id=user_id)]

        assert parts[-1]['type'] == 'values'
        assert parts[-1]['ns'] == ()
        assert [message.text for message in parts[-1]['data']['messages']] == ['How long?', 'Four years.']

    async def test_stream_agent_refuses_a_paused_thread(self) -> None:
        user_id = str(uuid4())
        self.stub_model.routes = [GraphNode.BOOKING]
        self.stub_model.replies = [booking_tool_call_reply()]
        service = AgentService(self.agent)

        async for _ in service.stream_agent([HumanMessage(content='Book me a slot')], user_id=user_id):
            pass

        with pytest.raises(ConflictingStatusError):
            async for _ in service.stream_agent([HumanMessage(content='Actually, hello')], user_id=user_id):
                pass


class TestRunRegistry:
    @pytest.fixture(autouse=True)
    async def _a_empty_registry(self) -> AsyncGenerator[None, None]:
        yield
        await runs.wait_for_streaming_turns(timeout=5.0)

    async def test_a_second_turn_on_the_same_thread_is_refused(self) -> None:
        user_id = str(uuid4())
        runs.register(user_id, asyncio.create_task(asyncio.sleep(0.2)))

        with pytest.raises(ConflictingStatusError):
            runs.assert_free(user_id)

    async def test_another_thread_is_free(self) -> None:
        runs.register(str(uuid4()), asyncio.create_task(asyncio.sleep(0.2)))

        runs.assert_free(str(uuid4()))

    async def test_the_entry_clears_when_the_run_finishes(self) -> None:
        user_id = str(uuid4())
        task = asyncio.create_task(asyncio.sleep(0))
        runs.register(user_id, task)

        await task

        assert not runs._RUNS
        runs.assert_free(user_id)

    async def test_the_drain_waits_for_a_pending_run(self) -> None:
        finished = False

        async def _slow_turn() -> None:
            nonlocal finished
            await asyncio.sleep(0.2)
            finished = True

        runs.register(str(uuid4()), asyncio.create_task(_slow_turn()))

        await runs.wait_for_streaming_turns(timeout=5.0)

        assert finished is True, 'the drain returned before the run had written anything'
        assert not runs._RUNS

    async def test_a_claimed_thread_is_busy_until_the_turn_ends(self) -> None:
        user_id = str(uuid4())

        async with runs.claimed(user_id):
            with pytest.raises(ConflictingStatusError):
                runs.assert_free(user_id)

        runs.assert_free(user_id)

    async def test_a_claim_is_released_even_when_the_turn_raises(self) -> None:
        user_id = str(uuid4())

        try:
            async with runs.claimed(user_id):
                raise ConflictingStatusError('the thread is paused')
        except ConflictingStatusError:
            pass

        runs.assert_free(user_id)

    async def test_a_finished_run_does_not_release_a_later_turn(self) -> None:
        user_id = str(uuid4())
        abandoned = asyncio.create_task(asyncio.sleep(0))
        runs.register(user_id, abandoned)
        later = asyncio.create_task(asyncio.sleep(0.2))
        runs.register(user_id, later)

        await abandoned

        with pytest.raises(ConflictingStatusError):
            runs.assert_free(user_id)
        later.cancel()

    async def test_the_drain_gives_up_after_its_timeout(self) -> None:
        task = asyncio.create_task(asyncio.sleep(30))
        runs.register(str(uuid4()), task)

        await runs.wait_for_streaming_turns(timeout=0.1)

        assert len(runs._RUNS) == 1, 'a run past the timeout is left alone, not cancelled'
        task.cancel()


class TestStreamingTurn(TestBaseCheckpointerClass):
    @pytest.fixture(autouse=True)
    def _a_provide_repository(self, dynamo_client: DynamoDBClient, conversations_table: DynamoDBService) -> None:
        self.repository = ConversationHistoryRepository(dynamo_client, get_settings())

    @pytest.fixture(autouse=True)
    async def _a_drain(self) -> AsyncGenerator[None, None]:
        yield
        await runs.wait_for_streaming_turns(timeout=10.0)

    async def test_the_json_representation_is_untouched(self) -> None:
        self.stub_model.replies = [AIMessage(content='Four years.')]

        response = await self.not_auth_client.post(
            ENDPOINT, json={'query': 'How long is the CS bachelor?', 'user_id': str(uuid4())}
        )

        assert response.status_code == 200
        assert response.headers['content-type'] == 'application/json'
        assert ChatSchemaOut.model_validate(response.json()).message == 'Four years.'

    async def test_tokens_arrive_in_order_and_spell_the_reply(self) -> None:
        reply = 'The Computer Science bachelor runs for four years and costs 12000 EUR per year.'
        self.stub_model.replies = [AIMessage(content=reply)]

        events = await read_events(
            self.not_auth_client, {'query': 'Tell me about the CS bachelor', 'user_id': str(uuid4())}
        )

        assert len([event for event in events if event.name == 'token']) > 1, 'the answer arrived in one piece'
        assert streamed_text(events) == reply

    async def test_done_carries_the_whole_result_and_comes_last(self) -> None:
        user_id = str(uuid4())
        self.stub_model.replies = [AIMessage(content='Four years.')]

        events = await read_events(self.not_auth_client, {'query': 'How long?', 'user_id': user_id})

        assert events[-1].name == 'done'
        result = ChatSchemaOut.model_validate_json(events[-1].data)
        assert result.status == 'answer'
        assert result.message == 'Four years.'

    async def test_done_is_what_was_stored(self) -> None:
        user_id = str(uuid4())
        self.stub_model.replies = [AIMessage(content='Four years.')]

        events = await read_events(self.not_auth_client, {'query': 'How long?', 'user_id': user_id})

        stored = await self.repository.list_message_by_user_id(user_id)
        assert [(message.role, message.content) for message in stored] == [
            (Role.USER, 'How long?'),
            (Role.ASSISTANT, ChatSchemaOut.model_validate_json(events[-1].data).message),
        ]


class TestNothingButTheAnswerIsForwarded(TestBaseCheckpointerClass):
    @pytest.fixture(autouse=True)
    async def _a_drain(self) -> AsyncGenerator[None, None]:
        yield
        await runs.wait_for_streaming_turns(timeout=10.0)

    async def test_the_router_json_never_reaches_the_client(self) -> None:
        self.stub_model.replies = [AIMessage(content='Hello!')]

        events = await read_events(self.not_auth_client, {'query': 'Hi there!', 'user_id': str(uuid4())})

        assert 'destination' not in streamed_text(events)
        assert streamed_text(events) == 'Hello!'

    async def test_a_tool_result_never_reaches_the_client(self, knowledge: StubKnowledgeService) -> None:
        self.stub_model.replies = [
            tool_call_reply('computer science tuition'),
            AIMessage(content='Tuition is 12000 EUR per year.'),
        ]

        events = await read_events(self.not_auth_client, {'query': 'How much is CS tuition?', 'user_id': str(uuid4())})

        assert knowledge.passages not in streamed_text(events)
        assert streamed_text(events) == 'Tuition is 12000 EUR per year.'

    async def test_the_summariser_never_reaches_the_client(self) -> None:
        user_id = str(uuid4())
        padding = 'We went over campus life, student housing, clubs and the sports centre in detail. ' * 24
        await seed_agent_state(
            self.agent,
            user_id,
            [
                message
                for turn in range(12)
                for message in (
                    HumanMessage(content=f'Earlier question {turn}. {padding}'),
                    AIMessage(content=f'Earlier answer {turn}. {padding}'),
                )
            ],
        )
        self.stub_model.replies = [
            AIMessage(content='## SESSION INTENT The applicant asked about campus life and housing.'),
            AIMessage(content='You can apply for the STEM scholarship.'),
        ]

        events = await read_events(
            self.not_auth_client, {'query': 'What scholarships can I apply for?', 'user_id': user_id}
        )

        assert 'SESSION INTENT' not in streamed_text(events)
        assert streamed_text(events) == 'You can apply for the STEM scholarship.'

    async def test_a_return_direct_answer_streams_as_its_own_graph_writes_it(
        self, knowledge: StubKnowledgeService
    ) -> None:
        """§4.4 — the comparison arrives while it is being written, not in one piece at the end."""
        comparison = '### Economics vs Business Analytics\n- Economics is broader.'
        self.stub_model.replies = [
            compare_tool_call_reply('Economics', 'Business Analytics'),
            AIMessage(content=comparison),
        ]

        with (
            patch(PATH_TO_COMPARE_MODEL_FACTORY, return_value=self.stub_model),
            patch(PATH_TO_COMPARE_KNOWLEDGE_SERVICE, return_value=knowledge),
        ):
            events = await read_events(
                self.not_auth_client,
                {'query': 'Compare Economics and Business Analytics', 'user_id': str(uuid4())},
            )

        assert [event.name for event in events][-1] == 'done'
        assert len([event for event in events if event.name == 'token']) > 1
        # Exactly once: the `ToolMessage` carries the same text again, and forwarding both would
        # print the comparison twice.
        assert streamed_text(events) == comparison

    async def test_a_return_direct_answer_falls_back_to_one_token(self, knowledge: StubKnowledgeService) -> None:
        """With the tool's graph streaming nothing this chat recognises — a node renamed under us —
        the whole `ToolMessage` still goes out, so the applicant gets an answer either way."""
        comparison = '### Economics vs Business Analytics\n- Economics is broader.'
        self.stub_model.replies = [
            compare_tool_call_reply('Economics', 'Business Analytics'),
            AIMessage(content=comparison),
        ]

        with (
            patch(PATH_TO_COMPARE_MODEL_FACTORY, return_value=self.stub_model),
            patch(PATH_TO_COMPARE_KNOWLEDGE_SERVICE, return_value=knowledge),
            patch(PATH_TO_DIRECT_ANSWER_NODE, 'renamed'),
        ):
            events = await read_events(
                self.not_auth_client,
                {'query': 'Compare Economics and Business Analytics', 'user_id': str(uuid4())},
            )

        assert [event.name for event in events] == ['token', 'done']
        assert streamed_text(events) == comparison


class TestPausesAndFailures(TestBaseCheckpointerClass):
    @pytest.fixture(autouse=True)
    def _a_provide_repository(self, dynamo_client: DynamoDBClient, conversations_table: DynamoDBService) -> None:
        self.repository = ConversationHistoryRepository(dynamo_client, get_settings())

    @pytest.fixture(autouse=True)
    async def _a_drain(self) -> AsyncGenerator[None, None]:
        yield
        await runs.wait_for_streaming_turns(timeout=10.0)

    async def _pause_a_booking(self, user_id: str) -> list[Event]:
        self.stub_model.routes = [GraphNode.BOOKING]
        self.stub_model.replies = [booking_tool_call_reply()]
        return await read_events(self.not_auth_client, {'query': 'Book me a slot', 'user_id': user_id})

    async def test_an_interrupt_ends_the_stream_on_pending(self) -> None:
        events = await self._pause_a_booking(str(uuid4()))

        assert events[-1].name == 'pending'
        result = ChatSchemaOut.model_validate_json(events[-1].data)
        assert result.status == 'pending_approval'
        assert result.pending is not None
        assert [approval.action for approval in result.pending] == ['book_appointment']

    async def test_the_pause_is_the_same_one_json_returns(self) -> None:
        streamed = await self._pause_a_booking(str(uuid4()))

        self.stub_model.reset()
        self.stub_model.routes = [GraphNode.BOOKING]
        self.stub_model.replies = [booking_tool_call_reply()]
        json_response = await self.not_auth_client.post(
            ENDPOINT, json={'query': 'Book me a slot', 'user_id': str(uuid4())}
        )

        assert json_response.json() == ChatSchemaOut.model_validate_json(streamed[-1].data).model_dump()

    async def test_a_paused_thread_refuses_a_new_message_with_409(self) -> None:
        user_id = str(uuid4())
        await self._pause_a_booking(user_id)

        response = await self.not_auth_client.post(
            ENDPOINT, json={'query': 'Actually, never mind', 'user_id': user_id}, headers=SSE_HEADERS
        )

        assert response.status_code == 409
        assert response.headers['content-type'] == 'application/json'
        assert 'paused' in response.json()['detail']

    async def test_decisions_stream_too(self) -> None:
        user_id = str(uuid4())
        await self._pause_a_booking(user_id)
        self.stub_model.replies = [*self.stub_model.replies, AIMessage(content='No problem, nothing was booked.')]

        events = await read_events(
            self.not_auth_client,
            {'user_id': user_id, 'decisions': [{'type': 'reject', 'message': 'The slot is taken.'}]},
        )

        assert events[-1].name == 'done'
        assert ChatSchemaOut.model_validate_json(events[-1].data).message == 'No problem, nothing was booked.'

    async def test_wrong_decisions_are_refused_with_409_before_the_stream_starts(self) -> None:
        user_id = str(uuid4())
        await self._pause_a_booking(user_id)

        response = await self.not_auth_client.post(
            ENDPOINT,
            json={'user_id': user_id, 'decisions': [{'type': 'approve'}, {'type': 'approve'}]},
            headers=SSE_HEADERS,
        )

        assert response.status_code == 409
        assert response.headers['content-type'] == 'application/json'
        assert 'send one per action' in response.json()['detail']

    async def test_a_tool_failure_stores_nothing_on_the_json_path_either(self) -> None:
        user_id = str(uuid4())
        self.stub_model.replies = [
            AIMessage(content='', tool_calls=[{'id': 'call_1', 'name': 'compare_programs', 'args': {}}])
        ]

        response = await self.not_auth_client.post(
            ENDPOINT, json={'query': 'Compare the programmes', 'user_id': user_id}
        )

        assert ChatSchemaOut.model_validate(response.json()).message == TOOL_FAILURE_MESSAGE
        assert await self.repository.list_message_by_user_id(user_id) == []

    async def test_a_tool_failure_replaces_the_answer_and_stores_nothing(self) -> None:
        user_id = str(uuid4())
        self.stub_model.replies = [
            AIMessage(content='', tool_calls=[{'id': 'call_1', 'name': 'compare_programs', 'args': {}}])
        ]

        events = await read_events(self.not_auth_client, {'query': 'Compare the programmes', 'user_id': user_id})

        assert [event.name for event in events] == ['done'], 'the failed tool streamed its error to the applicant'
        assert ChatSchemaOut.model_validate_json(events[-1].data).message == TOOL_FAILURE_MESSAGE
        assert await self.repository.list_message_by_user_id(user_id) == []


class TestDetachedRuns(TestBaseCheckpointerClass):
    @pytest.fixture(autouse=True)
    def _a_provide_repository(self, dynamo_client: DynamoDBClient, conversations_table: DynamoDBService) -> None:
        self.repository = ConversationHistoryRepository(dynamo_client, get_settings())

    @pytest.fixture(autouse=True)
    async def _a_drain(self) -> AsyncGenerator[None, None]:
        yield
        await runs.wait_for_streaming_turns(timeout=10.0)

    async def test_the_run_finishes_with_nobody_consuming_the_stream(self) -> None:

        user_id = uuid4()
        reply = 'The Computer Science bachelor runs for four years.'
        self.stub_model.replies = [AIMessage(content=reply)]
        service = ChatService(AgentService(self.agent), ConversationHistoryService(self.repository))

        await service.start_streaming_turn(UserQuery(user_id=user_id, query='How long?'))
        await runs.wait_for_streaming_turns(timeout=10.0)

        stored = await self.repository.list_message_by_user_id(str(user_id))
        assert [(message.role, message.content) for message in stored] == [
            (Role.USER, 'How long?'),
            (Role.ASSISTANT, reply),
        ]

    async def test_the_producer_is_not_owned_by_the_request(self) -> None:
        user_id = str(uuid4())
        self.stub_model.replies = [AIMessage(content='Four years.')]

        await read_events(self.not_auth_client, {'query': 'How long?', 'user_id': user_id})
        await runs.wait_for_streaming_turns(timeout=10.0)

        assert not runs._RUNS

    async def test_a_second_turn_while_one_is_in_flight_is_refused(self) -> None:
        user_id = str(uuid4())
        in_flight = asyncio.create_task(asyncio.sleep(2))
        runs.register(user_id, in_flight)

        response = await self.not_auth_client.post(
            ENDPOINT, json={'query': 'How long?', 'user_id': user_id}, headers=SSE_HEADERS
        )

        assert response.status_code == 409
        assert 'still running' in response.json()['detail']
        in_flight.cancel()

    async def test_a_json_turn_is_refused_while_a_run_is_in_flight(self) -> None:
        user_id = str(uuid4())
        in_flight = asyncio.create_task(asyncio.sleep(2))
        runs.register(user_id, in_flight)

        response = await self.not_auth_client.post(ENDPOINT, json={'query': 'How long?', 'user_id': user_id})

        assert response.status_code == 409
        assert 'still running' in response.json()['detail']
        in_flight.cancel()


class _StubCheckpointer:
    def __init__(self) -> None:
        self.closed_after_drain = False

    @asynccontextmanager
    async def opened(self) -> AsyncGenerator['_StubCheckpointer', None]:
        try:
            yield self
        finally:
            self.closed_after_drain = not runs._RUNS


class TestLifespanDrainsRuns:
    async def test_the_lifespan_waits_for_a_run_before_it_closes_the_checkpointer(self) -> None:
        finished = False

        async def _slow_turn() -> None:
            nonlocal finished
            await asyncio.sleep(0.2)
            finished = True

        with patch(PATH_TO_CHECKPOINTER, return_value=_StubCheckpointer()) as checkpointer:
            async with lifespan(FastAPI()):
                runs.register(str(uuid4()), asyncio.create_task(_slow_turn()))

        assert finished is True, 'shutdown returned while a turn was still writing'
        assert checkpointer.return_value.closed_after_drain is True

    async def test_the_drain_budget_comes_from_settings(self) -> None:
        with (
            patch(PATH_TO_CHECKPOINTER, return_value=_StubCheckpointer()),
            patch(PATH_TO_WAIT_FOR_STREAMING_TURNS, new_callable=AsyncMock) as drain,
        ):
            async with lifespan(FastAPI()):
                pass

        drain.assert_awaited_once_with(get_settings().DRAIN_TIMEOUT_SECONDS)
        assert get_settings().DRAIN_TIMEOUT_SECONDS <= 30, 'longer than a default Kubernetes grace period'
