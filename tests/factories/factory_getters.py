from types_aiobotocore_dynamodb import DynamoDBClient

from app.api.v1.faculty.schemas import FacultyItem, FacultyNameItem
from app.api.v1.programs.schemas import ProgramItem, ProgramNameItem
from app.api.v1.rooms.repository import RoomsRepository
from app.api.v1.rooms.schemas import Room
from app.settings import get_settings
from tests.db_utils import university_service


async def get_test_room(room_id: str, db_client: DynamoDBClient) -> Room | None:
    repo = RoomsRepository(db_client, get_settings())
    return await repo.find_room(room_id)


async def get_test_faculty(faculty_id: str, db_client: DynamoDBClient) -> FacultyItem | None:
    row = await university_service(db_client).get_item(FacultyItem.key(faculty_id), consistent_read=True)
    return FacultyItem.model_validate(row) if row is not None else None


async def get_test_faculty_name_reservation(name: str, db_client: DynamoDBClient) -> FacultyNameItem | None:
    row = await university_service(db_client).get_item(FacultyNameItem.key(name), consistent_read=True)
    return FacultyNameItem.model_validate(row) if row is not None else None


async def get_test_program(program_id: str, db_client: DynamoDBClient) -> ProgramItem | None:
    row = await university_service(db_client).get_item(ProgramItem.key(program_id), consistent_read=True)
    return ProgramItem.model_validate(row) if row is not None else None


async def get_test_program_name_reservation(
    faculty_id: str, name: str, db_client: DynamoDBClient
) -> ProgramNameItem | None:
    row = await university_service(db_client).get_item(ProgramNameItem.key(faculty_id, name), consistent_read=True)
    return ProgramNameItem.model_validate(row) if row is not None else None
