from types_aiobotocore_dynamodb import DynamoDBClient

from app.api.v1.faculty.repository import FacultyRepository
from app.api.v1.faculty.schemas import FacultyCreate, FacultyItem
from app.api.v1.rooms.repository import RoomsRepository
from app.api.v1.rooms.schemas import CreateRoom, Room
from app.core.dynamodb.base_service import DynamoDBService
from app.settings import Settings
from tests.factories.program_factory import ProgramRow


async def create_test_room(creation: CreateRoom, db_client: DynamoDBClient, settings: Settings) -> Room:
    repo = RoomsRepository(db_client, settings)
    return await repo.create_room(creation)


async def create_test_faculty(creation: FacultyCreate, db_client: DynamoDBClient, settings: Settings) -> FacultyItem:
    repo = FacultyRepository(db_client, settings)
    faculty = await repo.create_faculty(creation)
    return faculty


async def create_test_program(faculty_id: str, db_client: DynamoDBClient, settings: Settings) -> ProgramRow:
    """Put a programme belonging to ``faculty_id`` straight into the table — programmes have no
    repository of their own yet, so the row goes in through the generic service."""
    program = ProgramRow(gsi1pk=f'FACULTY#{faculty_id}')
    await DynamoDBService(db_client, settings.DYNAMODB_UNIVERSITY_TABLE).put_item(program)
    return program
