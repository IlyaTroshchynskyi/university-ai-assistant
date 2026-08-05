from asyncio import AbstractEventLoop, DefaultEventLoopPolicy
from contextlib import AsyncExitStack
import os
from typing import AsyncGenerator, Callable, TypeAlias

from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient
import pytest
import pytest_asyncio
from types_aiobotocore_dynamodb import DynamoDBClient

from app.core.dynamodb.base_service import DynamoDBService
from app.core.dynamodb.client import get_aioboto_session, open_dynamo_client
from app.settings import get_settings
from tests.db_utils import _table, get_dynamo_base_service, get_table_specs, TableSpecs
from tests.dependencies import override_app_test_dependencies

TEST_HOST = 'http://test'
LoopFactory: TypeAlias = Callable[[], AbstractEventLoop]


def pytest_configure(config: pytest.Config) -> None:
    os.environ.setdefault('OPENAI_API_KEY', 'test-key')
    os.environ.setdefault('DYNAMODB_ENDPOINT_URL', 'http://localhost:8001')
    os.environ.setdefault('AWS_REGION', 'us-east-1')
    os.environ.setdefault('AWS_ACCESS_KEY_ID', 'dummy')
    os.environ.setdefault('AWS_SECRET_ACCESS_KEY', 'dummy')
    os.environ['DYNAMODB_UNIVERSITY_TABLE'] = 'university_test'
    os.environ['DYNAMODB_SLOTS_TABLE'] = 'appointment_slots_test'

    # Imported here, after the env is in place, so nothing can build (and cache) a Settings that
    # still points at the development tables.
    from app.settings import get_settings

    get_settings.cache_clear()


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


class TestBaseDBClass:
    @pytest.fixture(autouse=True)
    def _a_provide_db(
        self, university_table: DynamoDBService, tables: TableSpecs, dynamo_client: DynamoDBClient
    ) -> None:
        self.university_table = university_table
        self.tables = tables
        self.dynamo_client = dynamo_client


class TestBaseClientDBClass(TestBaseClientClass, TestBaseDBClass): ...
