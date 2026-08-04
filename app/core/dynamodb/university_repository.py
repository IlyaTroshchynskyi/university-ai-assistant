from contextlib import asynccontextmanager
from typing import AsyncGenerator

from boto3.dynamodb.conditions import Attr, Key
from types_aiobotocore_dynamodb import DynamoDBClient

from app.core.dynamodb.base_service import DynamoDBService
from app.core.dynamodb.client import get_aioboto_session, open_dynamo_client
from app.core.dynamodb.indexes import Index, KeyAttr, normalize_name_key
from app.core.dynamodb.schemas import Place, Professor
from app.settings import get_settings, Settings


class UniversityRepository(DynamoDBService):
    def __init__(self, client: DynamoDBClient, settings: Settings):
        super().__init__(client, settings.DYNAMODB_UNIVERSITY_TABLE)

    async def find_professors_by_name(self, name: str) -> list[Professor] | None:
        prefix = normalize_name_key(name)
        if not prefix:
            return None

        rows = await self.query(
            key_condition=Key(KeyAttr.GSI_NAME_PK).eq('PROF') & Key(KeyAttr.GSI_NAME_SK).begins_with(prefix),
            index_name=Index.GSI_NAME,
            limit=5,
        )
        return [Professor.model_validate(row) for row in rows] if rows else None

    async def find_place_by_name(self, name: str) -> Place | None:
        query = name.strip().lower()
        rows = await self.query(
            key_condition=Key(KeyAttr.GSI1_PK).eq('TYPE#PLACE'),
            filter_expression=Attr('name_lower').contains(query),
            index_name=Index.GSI1,
        )
        return Place.model_validate(rows[0]) if rows else None


@asynccontextmanager
async def open_university_repository() -> AsyncGenerator[UniversityRepository, None]:
    async with open_dynamo_client(get_aioboto_session(), get_settings()) as client:
        yield UniversityRepository(client, get_settings())
