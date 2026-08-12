from langchain_core.messages import BaseMessage
from langgraph.graph.state import CompiledStateGraph
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
from tests.factories.university_rows_factory import PlaceFactory, PlaceRow, ProfessorFactory, ProfessorRow


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


async def seed_agent_state(agent: CompiledStateGraph, user_id: str, messages: list[BaseMessage]) -> None:
    """Write messages straight into the agent's checkpointer under ``thread_id == user_id``, so a
    test can start from an existing conversation without driving the model through it. Goes through
    the graph rather than the raw saver because the ``add_messages`` reducer and the checkpoint
    bookkeeping are what make the state loadable again."""
    await agent.aupdate_state({'configurable': {'thread_id': user_id}}, {'messages': messages}, as_node='model')


async def create_test_group_row(program_id: str, db_client: DynamoDBClient) -> GroupRow:
    group = GroupRow(gsi1pk=f'PROGRAM#{program_id}')
    await university_service(db_client).put_item(group)
    return group


async def create_test_professor_row(db_client: DynamoDBClient, **overrides: str | int) -> ProfessorRow:
    """Seed the professor the multi-turn evaluation asks about."""
    row = ProfessorRow.from_entity(ProfessorFactory.build(**overrides))
    await university_service(db_client).put_item(row)
    return row


async def create_test_place_row(db_client: DynamoDBClient, **overrides: str | int) -> PlaceRow:
    """Seed the campus place the multi-turn evaluation asks about."""
    row = PlaceRow.from_entity(PlaceFactory.build(**overrides))
    await university_service(db_client).put_item(row)
    return row
