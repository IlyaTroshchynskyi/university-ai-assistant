import logging
import uuid

from deepeval.test_case import ConversationalTestCase, Turn
from httpx import AsyncClient
from langchain_core.messages import ToolMessage
import pytest
import pytest_asyncio

from app.ai_assistant.university_knowladge.knowledge_service import NO_RESULTS
from app.ai_assistant_langchain.agent import create_assistant_agent
from app.core.dynamodb.client import get_aioboto_session, open_dynamo_client
from app.settings import get_settings
from tests.conftest import TestBaseClientClass
from tests.db_utils import TableSpecs
from tests.factories.factory_creators import create_test_place_row, create_test_professor_row
from tests.integration.conversations import ConversationCase, CONVERSATIONS
from tests.integration.metrics import assert_metrics, conversation_metrics

# Session loop for the same reason as the neighbouring suites: the tests drive the session-scoped
# ``not_auth_client``.
pytestmark = [
    pytest.mark.evaluation,
    pytest.mark.asyncio(loop_scope='session'),
    pytest.mark.usefixtures('eval_checkpointer', 'knowledge_base_populated'),
]

logger = logging.getLogger(__name__)

ENDPOINT = '/langchain-assistant'


@pytest_asyncio.fixture(scope='session', loop_scope='session', autouse=True)
async def _seed_university_records(tables: TableSpecs) -> None:
    async with open_dynamo_client(get_aioboto_session(), get_settings()) as client:
        await create_test_professor_row(client)
        await create_test_place_row(client)


async def tool_grounds(user_id: str, seen: int) -> tuple[list[str], int]:
    """Everything the tools returned since message ``seen`` of the thread, and the new watermark."""
    snapshot = await create_assistant_agent().aget_state({'configurable': {'thread_id': user_id}})
    messages = snapshot.values['messages']
    grounds = [
        str(message.content)
        for message in messages[seen:]
        if isinstance(message, ToolMessage) and message.content != NO_RESULTS
    ]
    return grounds, len(messages)


async def run_conversation(client: AsyncClient, user_id: str, questions: list[str]) -> list[Turn]:
    """Post every question in order on one thread and return the turns, grounds attached."""
    turns: list[Turn] = []
    seen = 0
    for question in questions:
        response = await client.post(ENDPOINT, json={'query': question, 'user_id': user_id})
        assert response.status_code == 200, response.text

        grounds, seen = await tool_grounds(user_id, seen)
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
