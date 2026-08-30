import logging
import uuid

from deepeval.test_case import ConversationalTestCase, Turn
from httpx import AsyncClient
import pytest
import pytest_asyncio
from types_aiobotocore_dynamodb import DynamoDBClient

from app.ai_assistant.university_knowladge.knowledge_service import NO_RESULTS
from app.ai_assistant_langchain.main_graph import build_main_graph
from tests.conftest import TestBaseClientClass
from tests.factories.factory_creators import create_test_place_row, create_test_professor_row
from tests.integration.conversations import ConversationCase, CONVERSATIONS
from tests.integration.metrics import assert_metrics, conversation_metrics
from tests.integration.thread_state import get_tool_responses

# Session loop for the same reason as the neighbouring suites: the tests drive the session-scoped
# ``not_auth_client``.
pytestmark = [
    pytest.mark.evaluation,
    pytest.mark.asyncio(loop_scope='session'),
    pytest.mark.usefixtures('eval_checkpointer', 'knowledge_base_populated', 'eval_conversations_table'),
]

logger = logging.getLogger(__name__)

ENDPOINT = '/langchain-assistant'


@pytest_asyncio.fixture(scope='session', loop_scope='session', autouse=True)
async def _seed_university_records(session_dynamo_client: DynamoDBClient) -> None:
    await create_test_professor_row(session_dynamo_client)
    await create_test_place_row(session_dynamo_client)


async def run_conversation(client: AsyncClient, user_id: str, questions: list[str]) -> list[Turn]:
    """Post every question in order on one thread and return the turns, grounds attached."""
    turns: list[Turn] = []
    seen = 0
    for question in questions:
        response = await client.post(ENDPOINT, json={'query': question, 'user_id': user_id})
        assert response.status_code == 200, response.text

        grounds, seen = await get_tool_responses(build_main_graph(), user_id, seen, ignore={NO_RESULTS})
        turns.append(Turn(role='user', content=question))
        turns.append(Turn(role='assistant', content=response.json()['message'], retrieval_context=grounds or None))
    return turns


class TestAgentConversations(TestBaseClientClass):
    @pytest.mark.parametrize('case', CONVERSATIONS, ids=lambda case: case.name)
    async def test_conversation(self, case: ConversationCase) -> None:
        turns = await run_conversation(self.not_auth_client, str(uuid.uuid4()), case.questions)

        # Without the transcript a red run cannot be diagnosed: three scores cannot tell a wrong
        # answer from a judge that slipped, and the thread is dropped with the test tables.
        logger.info('%s transcript\n%s', case.name, '\n'.join(f'  {t.role:>9}: {t.content}' for t in turns))

        test_case = ConversationalTestCase(
            name=case.name,
            expected_outcome=case.expected_outcome,
            turns=turns,
        )

        await assert_metrics(test_case, conversation_metrics())
