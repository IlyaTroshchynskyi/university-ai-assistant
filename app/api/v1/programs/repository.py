from typing import Annotated

from boto3.dynamodb.conditions import Attr, Key
from fastapi import Depends
from types_aiobotocore_dynamodb import DynamoDBClient

from app.api.v1.faculty.schemas import FacultyItem
from app.api.v1.programs.enums import ProgramCreateOutcome
from app.api.v1.programs.schemas import ProgramCreate, ProgramItem, ProgramNameItem
from app.core.dynamodb.base_service import DynamoDBService
from app.core.dynamodb.client import get_dynamo_client
from app.core.dynamodb.indexes import Index, KeyAttr
from app.core.dynamodb.schemas import TransactConditionCheck, TransactDelete, TransactPut
from app.settings import get_settings, Settings

# Positions of the actions ``create_program`` sends. A cancelled transaction reports its failures by
# position, so these are what tell 'no such faculty' apart from 'that name is taken'.
_FACULTY_EXISTS = 0
_NAME_RESERVATION = 2


class ProgramsRepository(DynamoDBService):
    def __init__(
        self,
        dynamo_db_client: Annotated[DynamoDBClient, Depends(get_dynamo_client)],
        settings: Annotated[Settings, Depends(get_settings)],
    ) -> None:
        super().__init__(dynamo_db_client, settings.DYNAMODB_UNIVERSITY_TABLE)

    async def create_program(self, creation: ProgramCreate) -> tuple[ProgramCreateOutcome, ProgramItem | None]:
        item = ProgramItem(**creation.model_dump())
        try:
            await self.transact_write(
                TransactConditionCheck(FacultyItem.key(creation.faculty_id), Attr(KeyAttr.PK).exists()),
                TransactPut(item, condition=Attr(KeyAttr.PK).not_exists()),
                TransactPut(
                    ProgramNameItem(faculty_id=item.faculty_id, name=item.name, program_id=item.id),
                    condition=Attr(KeyAttr.PK).not_exists(),
                ),
            )
        except self._client.exceptions.TransactionCanceledException as exc:
            failed = self.get_failed_condition_indexes(exc)
            if _FACULTY_EXISTS in failed:
                return ProgramCreateOutcome.FACULTY_NOT_FOUND, None
            if _NAME_RESERVATION not in failed:
                raise
            return ProgramCreateOutcome.NAME_TAKEN, None
        return ProgramCreateOutcome.CREATED, item

    async def list_programs(self) -> list[ProgramItem]:
        res = await self.scan(filter_expression=Attr('entity_type').eq(ProgramItem.entity_type_value()))
        return [ProgramItem.model_validate(r) for r in res]

    async def has_dependants(self, program_id: str) -> bool:
        rows = await self.query(
            Key(KeyAttr.GSI1_PK).eq(f'{ProgramItem.entity}#{program_id}'),
            index_name=Index.GSI1,
            limit=1,
        )
        return bool(rows)

    async def delete_program(self, program_id: str) -> bool:
        row = await self.get_item(ProgramItem.key(program_id), consistent_read=True)
        if row is None:
            return False

        program = ProgramItem.model_validate(row)
        try:
            await self.transact_write(
                TransactDelete(ProgramItem.key(program_id), condition=Attr(KeyAttr.PK).exists()),
                TransactDelete(ProgramNameItem.key(program.faculty_id, program.name)),
            )
        except self._client.exceptions.TransactionCanceledException:
            return False
        return True
