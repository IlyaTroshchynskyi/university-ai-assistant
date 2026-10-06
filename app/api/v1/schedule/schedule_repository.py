from boto3.dynamodb.conditions import Attr, Key
from types_aiobotocore_dynamodb import DynamoDBClient

from app.api.v1.rooms.schemas import GroupItem, ProfessorItem, Schedule
from app.core.dynamodb.base_service import DynamoDBService
from app.core.dynamodb.indexes import Index, KeyAttr, normalize_name
from app.settings import Settings

CLASS_SK_PREFIX = 'SCHED#'


class ScheduleRepository(DynamoDBService):
    def __init__(self, client: DynamoDBClient, settings: Settings):
        super().__init__(client, settings.DYNAMODB_GROUPS_TABLE)

    async def find_groups_by_name(self, name: str) -> dict[str, str]:
        name = normalize_name(name)
        rows = await self.scan(filter_expression=Attr(KeyAttr.SK).eq(GroupItem.meta_sk))
        return {str(row['id']): str(row['name']) for row in rows if normalize_name(row['name']) == name}

    async def list_group_classes(self, group_id: str) -> list[Schedule]:
        rows = await self.query(
            key_condition=(
                Key(KeyAttr.PK).eq(GroupItem(id=group_id).pk) & Key(KeyAttr.SK).begins_with(CLASS_SK_PREFIX)
            ),
        )
        return [Schedule.model_validate(row) for row in rows]

    async def list_professor_classes(self, professor_id: str) -> list[Schedule]:
        rows = await self.query(
            key_condition=(
                Key(KeyAttr.GSI1_PK).eq(ProfessorItem(id=professor_id).pk)
                & Key(KeyAttr.GSI1_SK).begins_with(CLASS_SK_PREFIX)
            ),
            index_name=Index.GSI1,
        )
        return [Schedule.model_validate(row) for row in rows]

    async def list_classes(self) -> list[Schedule]:
        rows = await self.scan(filter_expression=Attr(KeyAttr.SK).begins_with(CLASS_SK_PREFIX))
        return [Schedule.model_validate(row) for row in rows]
