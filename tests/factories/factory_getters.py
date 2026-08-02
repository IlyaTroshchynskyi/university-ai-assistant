from types_aiobotocore_dynamodb import DynamoDBClient

from app.api.v1.rooms.repository import RoomsRepository
from app.api.v1.rooms.schemas import Room
from app.settings import Settings


async def get_test_room(room_id: str, db_client: DynamoDBClient, settings: Settings) -> Room | None:
    repo = RoomsRepository(db_client, settings)
    return await repo.find_room(room_id)
