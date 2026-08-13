from langchain_core.messages import BaseMessage
from langgraph.graph.state import CompiledStateGraph
from types_aiobotocore_dynamodb import DynamoDBClient

from app.api.v1.faculty.repository import FacultyRepository
from app.api.v1.faculty.schemas import FacultyCreate, FacultyItem
from app.api.v1.programs.repository import ProgramsRepository
from app.api.v1.programs.schemas import ProgramItem
from app.api.v1.rooms.repository import RoomsRepository
from app.api.v1.rooms.schemas import CreateRoom, Room
from app.core.dynamodb.base_items import DEPENDANTS
from app.settings import get_settings
from tests.db_utils import table_service
from tests.factories.entity_rows_factory import PlaceFactory, PlaceRow, ProfessorFactory, ProfessorRow
from tests.factories.group_factory import GroupRow
from tests.factories.program_factory import ProgramCreationFactory


async def create_test_room(creation: CreateRoom, db_client: DynamoDBClient) -> Room:
    repo = RoomsRepository(db_client, get_settings())
    return await repo.create_room(creation)


async def create_test_faculty(creation: FacultyCreate, db_client: DynamoDBClient) -> FacultyItem:
    repo = FacultyRepository(db_client, get_settings())
    faculty = await repo.create_faculty(creation)
    assert faculty is not None, 'faculty creation failed — the name was already taken'
    return faculty


async def create_test_program(faculty_id: str, db_client: DynamoDBClient, **overrides: str | int) -> ProgramItem:
    repo = ProgramsRepository(db_client, get_settings())
    outcome, program = await repo.create_program(ProgramCreationFactory.build(faculty_id=faculty_id, **overrides))
    assert program is not None, f'program creation failed with {outcome}'
    return program


async def seed_agent_state(agent: CompiledStateGraph, user_id: str, messages: list[BaseMessage]) -> None:
    """Write messages straight into the agent's checkpointer under ``thread_id == user_id``, so a
    test can start from an existing conversation without driving the model through it. Goes through
    the graph rather than the raw saver because the ``add_messages`` reducer and the checkpoint
    bookkeeping are what make the state loadable again."""
    await agent.aupdate_state({'configurable': {'thread_id': user_id}}, {'messages': messages}, as_node='model')


async def create_test_group_row(program_id: str, db_client: DynamoDBClient) -> GroupRow:
    settings = get_settings()
    group = GroupRow(gsi1pk=f'PROGRAM#{program_id}')
    await table_service(db_client, settings.DYNAMODB_GROUPS_TABLE).put_item(group)
    await table_service(db_client, settings.DYNAMODB_PROGRAMS_TABLE).update_item(
        ProgramItem.key(program_id),
        f'ADD {DEPENDANTS} :one',
        {':one': 1},
    )
    return group


async def create_test_professor_row(db_client: DynamoDBClient, **overrides: str | int) -> ProfessorRow:
    """Seed the professor the multi-turn evaluation asks about."""
    row = ProfessorRow.from_entity(ProfessorFactory.build(**overrides))
    await table_service(db_client, get_settings().DYNAMODB_PROFESSORS_TABLE).put_item(row)
    return row


async def create_test_place_row(db_client: DynamoDBClient, **overrides: str | int) -> PlaceRow:
    """Seed the campus place the multi-turn evaluation asks about."""
    row = PlaceRow.from_entity(PlaceFactory.build(**overrides))
    await table_service(db_client, get_settings().DYNAMODB_PLACES_TABLE).put_item(row)
    return row
