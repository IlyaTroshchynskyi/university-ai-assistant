"""Tool-routing evaluation: which tool the agent reaches for, given the question."""

from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import AsyncGenerator, Generator
from unittest.mock import patch
import uuid

from deepeval.test_case import LLMTestCase, ToolCall
from langchain_core.messages import AIMessage
import pytest
from qdrant_client.models import ScoredPoint

from app.ai_assistant_langchain.main_graph import build_main_graph
from app.core.dynamodb.schemas import Place, Professor
from tests.conftest import TestBaseClientClass
from tests.integration.metrics import assert_metrics, routing_metrics

# Session loop for the same reason as the agent suite: the tests drive the app through the
# session-scoped ``not_auth_client``, and the agent's cached clients bind to whatever loop the
# first call ran on.
# ``eval_checkpointer`` but not ``knowledge_base_populated``: every tool backend is stubbed, the
# compare graph's own retrieval included, so nothing here reaches Qdrant — but the graph still runs,
# and the graph still writes checkpoints.
pytestmark = [
    pytest.mark.evaluation,
    pytest.mark.asyncio(loop_scope='session'),
    pytest.mark.usefixtures('eval_checkpointer', 'eval_conversations_table'),
]

ENDPOINT = '/langchain-assistant'
PATH_TO_KNOWLEDGE_SERVICE = 'app.ai_assistant_langchain.tools.get_knowledge_service'
PATH_TO_PROFESSORS_REPOSITORY = 'app.ai_assistant_langchain.tools.open_professors_repository'
PATH_TO_PLACES_REPOSITORY = 'app.ai_assistant_langchain.tools.open_places_repository'
# The compare graph searches the knowledge base itself rather than through the `retriever` tool, so
# its own module-level lookup is a second place to stub — patching the tool's leaves both gather
# branches talking to a Qdrant this suite is not allowed to need.
PATH_TO_COMPARE_KNOWLEDGE_SERVICE = 'app.ai_assistant_langchain.graphs.compare_programs.nodes.get_knowledge_service'


@dataclass
class RoutingCase:
    """A question and the tools it has to reach, in order. Empty means: answer without a lookup."""

    question: str
    expected_tools: list[str]


ROUTING_CASES = [
    # A named person: their contact details and office hours live in DynamoDB, not the handbook.
    RoutingCase("What is Dr. Alan Whitfield's email?", ['find_person']),
    RoutingCase('When are the office hours of Dr. Sophia Martens?', ['find_person']),
    # # A named place. "office hours" is the phrase that separates the two finders: it belongs to
    # # both a professor and a building, so a description that stops distinguishing them fails here.
    RoutingCase('When does the Main Library open?', ['find_place']),
    RoutingCase('Where is the Cafeteria?', ['find_place']),
    # # Everything factual that is not a named person or place.
    RoutingCase('How much does the Computer Science bachelor cost per year?', ['retriever']),
    # Two named programmes weighed against each other. The one case the compare graph exists for:
    # the whole question goes to `compare_programs`, which fans out to both programmes at once —
    # `retriever` twice in a row would be the same answer built the slow way, and reads here as a
    # failure.
    RoutingCase('Compare the Economics and Business Analytics programs.', ['compare_programs']),
    # # Small talk: the first prompt bullet says answer directly. A lookup here is money spent on a
    # # greeting, and it is the regression a tool description written too broadly causes.
    RoutingCase('Hi there!', []),
]

PROFESSOR = Professor(
    id=1,
    full_name='Dr. Alan Whitfield',
    title='Professor',
    faculty_id=1,
    email='a.whitfield@nit.edu',
    room_id=8,
    office_hours='Mon 14:00-16:00',
)
PLACE = Place(id=2, name='Main Library', building='Main Library', floor='1-3', opening_hours='Mon-Fri 08:00-22:00')
PASSAGE = 'Tuition for the BSc in Computer Science is $22,500 per year. Applications close on March 1.'


class StubKnowledgeService:
    """Answers every query with the same passage, in the hit shape ``join_passages`` expects."""

    async def search(self, query: str) -> list[ScoredPoint]:
        return [ScoredPoint(id=1, version=0, score=1.0, payload={'text': PASSAGE})]


PROGRAM_PASSAGES = {
    'Economics': (
        'The BSc in Economics runs three years. Annual tuition is $19,000. It covers micro- and '
        'macroeconomics, econometrics and public policy, and leads to analyst and policy roles.'
    ),
    'Business Analytics': (
        'The BSc in Business Analytics runs three years. Annual tuition is $21,500. It covers '
        'statistics, data visualisation and business modelling, and leads to data analyst roles.'
    ),
}


class StubProgramKnowledgeService:
    """The retrieval behind the compare graph: a different passage per programme.

    Its own stub rather than ``StubKnowledgeService``, which answers everything with ``PASSAGE``:
    both gather branches search, and handing both the same text gives the model two programmes that
    read identically — precisely when it goes for a second opinion from ``retriever`` and the
    assertion fails on a tool the agent called only because the fixture confused it.
    """

    async def search(self, query: str) -> list[ScoredPoint]:
        text = next(
            (passage for program, passage in PROGRAM_PASSAGES.items() if program.lower() in query.lower()),
            PASSAGE,
        )
        return [ScoredPoint(id=1, version=0, score=1.0, payload={'text': text})]


class StubProfessorsRepository:
    """Answers every lookup with a match, whatever the name. A miss would send the model looking
    for a second tool, and the fallback path is not what this suite measures.

    The name asked for is echoed back into the record. Returning a fixed professor to every query
    is a miss wearing a match: asked about Sophia Martens and handed Alan Whitfield, the model
    reasonably assumes it spelled the name wrong and calls `find_person` a second time — which the
    suite then reads as choosing the tool twice.
    """

    async def find_professors_by_name(self, name: str) -> list[Professor]:
        return [PROFESSOR.model_copy(update={'full_name': name})]


class StubPlacesRepository:
    """The place half of the same idea — two stubs now that the two lookups read two tables, and
    the same echo, for the same reason."""

    async def find_place_by_name(self, name: str) -> Place:
        return PLACE.model_copy(update={'name': name})


@asynccontextmanager
async def stub_professors_repository() -> AsyncGenerator[StubProfessorsRepository, None]:
    yield StubProfessorsRepository()


@asynccontextmanager
async def stub_places_repository() -> AsyncGenerator[StubPlacesRepository, None]:
    yield StubPlacesRepository()


async def get_called_tools(user_id: str) -> list[ToolCall]:
    """The tools the model asked for, in the order it asked for them, read back off the thread."""
    snapshot = await build_main_graph().aget_state({'configurable': {'thread_id': user_id}})
    called_tools: list[ToolCall] = []
    for message in snapshot.values['messages']:
        if not isinstance(message, AIMessage):
            continue
        for call in message.tool_calls:
            called_tools.append(ToolCall(name=call['name']))
    return called_tools


class TestToolRouting(TestBaseClientClass):
    @pytest.fixture(autouse=True)
    def _stub_tool_backends(self) -> Generator[None, None, None]:
        with (
            patch(PATH_TO_KNOWLEDGE_SERVICE, return_value=StubKnowledgeService()),
            patch(PATH_TO_COMPARE_KNOWLEDGE_SERVICE, return_value=StubProgramKnowledgeService()),
            patch(PATH_TO_PROFESSORS_REPOSITORY, stub_professors_repository),
            patch(PATH_TO_PLACES_REPOSITORY, stub_places_repository),
        ):
            yield

    @pytest.mark.parametrize('case', ROUTING_CASES)
    async def test_question_routes_to_expected_tool(self, case: RoutingCase) -> None:
        user_id = str(uuid.uuid4())

        response = await self.not_auth_client.post(ENDPOINT, json={'query': case.question, 'user_id': user_id})

        test_case = LLMTestCase(
            input=case.question,
            actual_output=response.json()['message'],
            tools_called=await get_called_tools(user_id),
            expected_tools=[ToolCall(name=name) for name in case.expected_tools],
        )

        await assert_metrics(test_case, routing_metrics())
