from asyncio import AbstractEventLoop, DefaultEventLoopPolicy
from contextlib import AsyncExitStack
import os
from typing import AsyncGenerator, Callable, TypeAlias

from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient
import pytest
import pytest_asyncio
from types_aiobotocore_dynamodb import DynamoDBClient

from app.ai_assistant_langchain.checkpointer.saver import get_checkpointer
from app.core.dynamodb.base_service import DynamoDBService
from app.core.dynamodb.client import get_aioboto_session, open_dynamo_client
from app.settings import get_settings
from tests.db_utils import _table, get_dynamo_base_service, get_table_specs, TableSpecs
from tests.dependencies import override_app_test_dependencies

TEST_HOST = 'http://test'
LoopFactory: TypeAlias = Callable[[], AbstractEventLoop]


RUN_EVAL_OPTION = '--run-eval'
EVALUATION_MARKER = 'evaluation'
TABLES_MARKER = 'tables'


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        RUN_EVAL_OPTION,
        action='store_true',
        default=False,
        help='run the DeepEval suites in tests/integration (real OpenAI and Qdrant calls, costs money)',
    )


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        'markers',
        f'{EVALUATION_MARKER}: DeepEval suite — skipped unless {RUN_EVAL_OPTION} is passed',
    )
    config.addinivalue_line(
        'markers',
        f'{TABLES_MARKER}: golden whose answer lives inside a table — the cases a chunking change '
        f'splitting tables breaks first (`make eval-tables`)',
    )

    # The stub key would shadow the real one: an environment variable outranks .env in
    # pydantic-settings, and the evaluations need a key that actually works.
    if not config.getoption(RUN_EVAL_OPTION):
        os.environ.setdefault('OPENAI_API_KEY', 'test-key')

    os.environ.setdefault('DYNAMODB_ENDPOINT_URL', 'http://localhost:8001')
    os.environ.setdefault('AWS_REGION', 'us-east-1')
    os.environ.setdefault('AWS_ACCESS_KEY_ID', 'dummy')
    os.environ.setdefault('AWS_SECRET_ACCESS_KEY', 'dummy')
    os.environ['DYNAMODB_UNIVERSITY_TABLE'] = 'university_test'
    os.environ['DYNAMODB_SLOTS_TABLE'] = 'appointment_slots_test'
    os.environ['DYNAMODB_CHECKPOINTS_TABLE'] = 'agent_checkpoints_test'

    # Imported here, after the env is in place, so nothing can build (and cache) a Settings that
    # still points at the development tables.
    from app.settings import get_settings

    get_settings.cache_clear()


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """Keep a plain ``pytest`` free and offline: the evaluations are collected, then skipped with
    a reason, unless the flag asks for them."""
    if config.getoption(RUN_EVAL_OPTION):
        return

    skip_eval = pytest.mark.skip(reason=f'needs {RUN_EVAL_OPTION} (real OpenAI and Qdrant calls)')
    for item in items:
        if EVALUATION_MARKER in item.keywords:
            item.add_marker(skip_eval)


@pytest_asyncio.fixture(scope='session', loop_scope='session')
async def app() -> AsyncGenerator[FastAPI, None]:
    from app.main import create_app

    _app = create_app()
    override_app_test_dependencies(_app)

    yield _app


@pytest_asyncio.fixture(scope='session', loop_scope='session')
async def not_auth_client(app: FastAPI) -> AsyncGenerator[AsyncClient, None]:
    async with AsyncClient(transport=ASGITransport(app=app), base_url=TEST_HOST) as client:
        yield client


def pytest_asyncio_loop_factories(config: pytest.Config, item: pytest.Item) -> dict[str, LoopFactory]:
    """Run the suite on the stdlib asyncio loop rather than whatever policy happens to be installed
    (uvloop ships in this venv). The ``event_loop_policy`` fixture used to do this; pytest-asyncio
    1.x deprecated overriding it in favour of this hook."""
    return {'asyncio': DefaultEventLoopPolicy().new_event_loop}


@pytest_asyncio.fixture(scope='session', loop_scope='session')
async def tables() -> AsyncGenerator[TableSpecs, None]:
    """Create every test table once for the whole session; drop them all when it ends."""
    specs = get_table_specs()
    async with open_dynamo_client(get_aioboto_session(), get_settings()) as client, AsyncExitStack() as stack:
        for spec in specs.values():
            await stack.enter_async_context(_table(client, spec))
        yield specs


@pytest.fixture
async def dynamo_client() -> AsyncGenerator[DynamoDBClient, None]:
    async with open_dynamo_client(get_aioboto_session(), get_settings()) as client:
        yield client


@pytest.fixture
async def university_table(dynamo_client: DynamoDBClient, tables: TableSpecs) -> AsyncGenerator[DynamoDBService, None]:
    async with get_dynamo_base_service(dynamo_client, tables['university']) as table:
        yield table


@pytest.fixture
async def slots_table(dynamo_client: DynamoDBClient, tables: TableSpecs) -> AsyncGenerator[DynamoDBService, None]:
    async with get_dynamo_base_service(dynamo_client, tables['slots']) as table:
        yield table


@pytest.fixture
async def issues_table(dynamo_client: DynamoDBClient, tables: TableSpecs) -> AsyncGenerator[DynamoDBService, None]:
    async with get_dynamo_base_service(dynamo_client, tables['issues']) as table:
        yield table


class TestBaseClientClass:
    @pytest.fixture(autouse=True)
    def _a_provide_client(
        self,
        not_auth_client: AsyncClient,
    ):
        self.not_auth_client = not_auth_client


class TestBaseAgentClass(TestBaseClientClass):
    """For a test that drives the **real** graph, whatever it stubs underneath.
    The agent's checkpointer is DynamoDB-backed, so the first ``ainvoke`` writes to
    ``agent_checkpoints_test`` — a table only the ``tables`` fixture creates.`.
    """

    @pytest.fixture(autouse=True)
    async def _a_provide_checkpointer(self, tables: TableSpecs) -> AsyncGenerator[None, None]:
        # What the app's lifespan does at startup, per test — each test gets its own event loop, and
        # the client has to be created and closed on the one that uses it.
        async with get_checkpointer().opened():
            yield


class TestBaseDBClass:
    @pytest.fixture(autouse=True)
    def _a_provide_db(
        self, university_table: DynamoDBService, tables: TableSpecs, dynamo_client: DynamoDBClient
    ) -> None:
        self.university_table = university_table
        self.tables = tables
        self.dynamo_client = dynamo_client


class TestBaseClientDBClass(TestBaseClientClass, TestBaseDBClass): ...
