from typing import AsyncGenerator

import pytest
import pytest_asyncio

from app.ai_assistant.university_knowladge.knowledge_service import get_knowledge_service
from app.ai_assistant_langchain.checkpointer.saver import get_checkpointer
from tests.db_utils import TableSpecs


@pytest_asyncio.fixture(scope='session', loop_scope='session')
async def eval_checkpointer(tables: TableSpecs) -> AsyncGenerator[None, None]:
    """Open the agent's DynamoDB client once, on the session loop."""
    async with get_checkpointer().opened():
        yield


@pytest_asyncio.fixture(scope='session', loop_scope='session')
async def knowledge_base_populated() -> None:
    """Skip the suite, with a reason, when there is nothing to retrieve."""
    try:
        hits = await get_knowledge_service().search('university programs and tuition')
    except Exception as error:  # noqa: BLE001 — infrastructure absence is a skip, not a failure
        pytest.skip(f'knowledge base is unreachable ({error.__class__.__name__}: {error})')

    if not hits:
        pytest.skip('knowledge base is empty — ingest the handbook via POST /documents first')
