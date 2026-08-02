from types_aiobotocore_dynamodb import DynamoDBClient

from app.api.v1.rooms.repository import RoomsRepository
from app.api.v1.rooms.schemas import CreateRoom, Room
from app.settings import Settings


async def create_test_room(creation: CreateRoom, db_client: DynamoDBClient, settings: Settings) -> Room:
    repo = RoomsRepository(db_client, settings)
    return await repo.create_room(creation)
