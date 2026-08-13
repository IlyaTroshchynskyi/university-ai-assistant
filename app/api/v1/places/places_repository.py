from contextlib import asynccontextmanager
from typing import AsyncGenerator

from boto3.dynamodb.conditions import Attr
from types_aiobotocore_dynamodb import DynamoDBClient

from app.core.dynamodb.base_service import DynamoDBService
from app.core.dynamodb.client import get_aioboto_session, open_dynamo_client
from app.core.dynamodb.schemas import Place
from app.settings import get_settings, Settings


class PlacesRepository(DynamoDBService):
    """Reads of the ``places`` table — the campus locations the assistant answers 'where is…' with."""

    def __init__(self, client: DynamoDBClient, settings: Settings):
        super().__init__(client, settings.DYNAMODB_PLACES_TABLE)

    async def find_place_by_name(self, name: str) -> Place | None:
        """The first place whose name contains ``name``, case-insensitively, or ``None``."""
        rows = await self.scan(filter_expression=Attr('name_lower').contains(name.strip().lower()))
        return Place.model_validate(rows[0]) if rows else None


@asynccontextmanager
async def open_places_repository() -> AsyncGenerator[PlacesRepository, None]:
    async with open_dynamo_client(get_aioboto_session(), get_settings()) as client:
        yield PlacesRepository(client, get_settings())
