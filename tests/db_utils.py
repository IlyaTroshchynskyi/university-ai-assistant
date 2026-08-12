from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Any, AsyncGenerator, Awaitable, Callable, Mapping, Sequence

from types_aiobotocore_dynamodb import DynamoDBClient
from types_aiobotocore_dynamodb.type_defs import WriteRequestOutputTypeDef, WriteRequestTypeDef

from db.load_dynamodb import table_definition

from app.core.dynamodb.base_service import DynamoDBService, Item
from app.settings import get_settings

ISSUES_TABLE = 'reported_issues_test'

BATCH_WRITE_LIMIT = 25

Seeder = Callable[[DynamoDBService, list[Item]], Awaitable[None]]


def university_service(db_client: DynamoDBClient) -> DynamoDBService:
    """The generic service pointed at the university table — what a factory uses to read or write a
    row directly, without going through a repository."""
    return DynamoDBService(db_client, get_settings().DYNAMODB_UNIVERSITY_TABLE)


@dataclass(frozen=True)
class TableSpec:
    name: str
    gsi_numbers: list[int]
    extra_gsis: list[tuple[str, str, str]] = field(default_factory=list)
    ttl_attribute: str | None = None


TableSpecs = dict[str, TableSpec]


def get_table_specs() -> TableSpecs:
    """The tables the suite runs against, keyed by the fixture that hands each one out."""
    settings = get_settings()
    return {
        'university': TableSpec(
            name=settings.DYNAMODB_UNIVERSITY_TABLE,
            gsi_numbers=[1, 2],
            extra_gsis=[('GSI_NAME', 'gsi_name_pk', 'gsi_name_sk')],
        ),
        'slots': TableSpec(name=settings.DYNAMODB_SLOTS_TABLE, gsi_numbers=[1, 2], ttl_attribute='expires_at'),
        'issues': TableSpec(name=ISSUES_TABLE, gsi_numbers=[1]),
        'checkpoints': TableSpec(name=settings.DYNAMODB_CHECKPOINTS_TABLE, gsi_numbers=[]),
    }


async def _drop_table(client: DynamoDBClient, name: str) -> None:
    await client.delete_table(TableName=name)
    await client.get_waiter('table_not_exists').wait(TableName=name)


async def _clear_table(client: DynamoDBClient, name: str) -> None:
    """Delete every row of ``name``, leaving the table and its indexes standing."""
    keys: list[dict[str, Any]] = []
    kwargs: dict[str, Any] = {'TableName': name, 'ProjectionExpression': 'pk, sk'}
    while True:
        page = await client.scan(**kwargs)
        keys.extend(page.get('Items', []))
        last_key = page.get('LastEvaluatedKey')
        if last_key is None:
            break
        kwargs['ExclusiveStartKey'] = last_key

    for start in range(0, len(keys), BATCH_WRITE_LIMIT):
        batch = keys[start : start + BATCH_WRITE_LIMIT]
        request: Mapping[str, Sequence[WriteRequestTypeDef | WriteRequestOutputTypeDef]] = {
            name: [{'DeleteRequest': {'Key': key}} for key in batch]
        }
        while request:  # DynamoDB may decline part of a batch and hand the rest back to be retried
            response = await client.batch_write_item(RequestItems=request)
            request = response.get('UnprocessedItems') or {}


@asynccontextmanager
async def _table(client: DynamoDBClient, spec: TableSpec) -> AsyncGenerator[TableSpec, None]:
    if spec.name in (await client.list_tables()).get('TableNames', []):
        await _drop_table(client, spec.name)

    await client.create_table(**table_definition(spec.name, spec.gsi_numbers, spec.extra_gsis))
    await client.get_waiter('table_exists').wait(TableName=spec.name)
    if spec.ttl_attribute:
        await client.update_time_to_live(
            TableName=spec.name,
            TimeToLiveSpecification={'Enabled': True, 'AttributeName': spec.ttl_attribute},
        )
    try:
        yield spec
    finally:
        await _drop_table(client, spec.name)


@asynccontextmanager
async def get_dynamo_base_service(client: DynamoDBClient, spec: TableSpec) -> AsyncGenerator[DynamoDBService, None]:
    try:
        yield DynamoDBService(client, spec.name)
    finally:
        await _clear_table(client, spec.name)
