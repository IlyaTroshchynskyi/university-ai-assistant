from types_aiobotocore_dynamodb import DynamoDBClient

from app.api.v1.faculty.repository import FacultyRepository
from app.api.v1.faculty.schemas import FacultyCreate, FacultyItem
from app.api.v1.programs.repository import ProgramsRepository
from app.api.v1.programs.schemas import ProgramItem
from app.api.v1.rooms.repository import RoomsRepository
from app.api.v1.rooms.schemas import CreateRoom, Room
from app.settings import get_settings
from tests.db_utils import university_service
from tests.factories.group_factory import GroupRow
from tests.factories.program_factory import ProgramCreationFactory


async def create_test_room(creation: CreateRoom, db_client: DynamoDBClient) -> Room:
    repo = RoomsRepository(db_client, get_settings())
    return await repo.create_room(creation)


async def create_test_faculty(creation: FacultyCreate, db_client: DynamoDBClient) -> FacultyItem:
    repo = FacultyRepository(db_client, get_settings())
    faculty = await repo.create_faculty(creation)
    return faculty


async def create_test_program(faculty_id: str, db_client: DynamoDBClient, **overrides: str | int) -> ProgramItem:
    repo = ProgramsRepository(db_client, get_settings())
    _, program = await repo.create_program(ProgramCreationFactory.build(faculty_id=faculty_id, **overrides))
    return program


async def create_test_group_row(program_id: str, db_client: DynamoDBClient) -> GroupRow:
    group = GroupRow(gsi1pk=f'PROGRAM#{program_id}')
    await university_service(db_client).put_item(group)
    return group
