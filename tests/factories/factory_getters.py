from types_aiobotocore_dynamodb import DynamoDBClient

from app.api.v1.faculty.schemas import FacultyItem, FacultyNameItem
from app.api.v1.rooms.repository import RoomsRepository
from app.api.v1.rooms.schemas import Room
from app.core.dynamodb.base_service import DynamoDBService
from app.settings import Settings


async def get_test_room(room_id: str, db_client: DynamoDBClient, settings: Settings) -> Room | None:
    repo = RoomsRepository(db_client, settings)
    return await repo.find_room(room_id)


async def get_test_faculty(faculty_id: str, db_client: DynamoDBClient, settings: Settings) -> FacultyItem | None:
    service = DynamoDBService(db_client, settings.DYNAMODB_UNIVERSITY_TABLE)
    row = await service.get_item(FacultyItem.key(faculty_id), consistent_read=True)
    return FacultyItem.model_validate(row) if row is not None else None


async def get_test_faculty_name_reservation(
    name: str, db_client: DynamoDBClient, settings: Settings
) -> FacultyNameItem | None:
    service = DynamoDBService(db_client, settings.DYNAMODB_UNIVERSITY_TABLE)
    row = await service.get_item(FacultyNameItem.key(name), consistent_read=True)
    return FacultyNameItem.model_validate(row) if row is not None else None
