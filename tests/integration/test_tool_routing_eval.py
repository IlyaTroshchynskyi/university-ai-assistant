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

from app.ai_assistant_langchain.agent import create_assistant_agent
from app.core.dynamodb.schemas import Place, Professor
from tests.conftest import TestBaseAgentClass
from tests.integration.conftest import assert_metrics, routing_metrics

# Session loop for the same reason as the agent suite: the tests drive the app through the
# session-scoped ``not_auth_client``, and the agent's cached clients bind to whatever loop the
# first call ran on.
pytestmark = [pytest.mark.evaluation, pytest.mark.asyncio(loop_scope='session')]

ENDPOINT = '/langchain-assistant'
PATH_TO_KNOWLEDGE_SERVICE = 'app.ai_assistant_langchain.tools.get_knowledge_service'
PATH_TO_UNIVERSITY_REPOSITORY = 'app.ai_assistant_langchain.tools.open_university_repository'


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


class StubUniversityRepository:
    """Answers every lookup with a match, whatever the name. A miss would send the model looking
    for a second tool, and the fallback path is not what this suite measures."""

    async def find_professors_by_name(self, name: str) -> list[Professor]:
        return [PROFESSOR]

    async def find_place_by_name(self, name: str) -> Place:
        return PLACE


@asynccontextmanager
async def stub_repository() -> AsyncGenerator[StubUniversityRepository, None]:
    yield StubUniversityRepository()


async def get_called_tools(user_id: str) -> list[ToolCall]:
    """The tools the model asked for, in the order it asked for them, read back off the thread."""
    snapshot = await create_assistant_agent().aget_state({'configurable': {'thread_id': user_id}})
    called_tools: list[ToolCall] = []
    for message in snapshot.values['messages']:
        if not isinstance(message, AIMessage):
            continue
        for call in message.tool_calls:
            called_tools.append(ToolCall(name=call['name']))
    return called_tools


class TestToolRouting(TestBaseAgentClass):
    @pytest.fixture(autouse=True)
    def _stub_tool_backends(self) -> Generator[None, None, None]:
        with (
            patch(PATH_TO_KNOWLEDGE_SERVICE, return_value=StubKnowledgeService()),
            patch(PATH_TO_UNIVERSITY_REPOSITORY, stub_repository),
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
