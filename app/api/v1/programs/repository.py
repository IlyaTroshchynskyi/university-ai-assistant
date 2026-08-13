from typing import Annotated

from boto3.dynamodb.conditions import Attr
from fastapi import Depends
from types_aiobotocore_dynamodb import DynamoDBClient

from app.api.v1.faculty.schemas import FacultyItem
from app.api.v1.programs.schemas import ProgramCreate, ProgramItem, ProgramNameItem
from app.core.dynamodb.base_items import DEPENDANTS
from app.core.dynamodb.base_service import DynamoDBService
from app.core.dynamodb.client import get_dynamo_client
from app.core.dynamodb.indexes import KeyAttr
from app.core.dynamodb.schemas import TransactDelete, TransactPut, TransactUpdate
from app.core.enums import CreateOutcome, DeleteOutcome
from app.settings import get_settings, Settings

# Positions of the actions ``create_program`` sends. A cancelled transaction reports its failures by
# position, so these are what tell 'no such faculty' apart from 'that name is taken'.
_PARENT_EXISTS = 0
_NAME_RESERVATION = 2

# Position of the programme row in ``delete_program`` — the one whose condition covers both 'gone'
# and 'still has groups'.
_PROGRAM_ROW = 0


class ProgramsRepository(DynamoDBService):
    """The ``programs`` table, plus the one attribute it maintains elsewhere."""

    def __init__(
        self,
        dynamo_db_client: Annotated[DynamoDBClient, Depends(get_dynamo_client)],
        settings: Annotated[Settings, Depends(get_settings)],
    ) -> None:
        super().__init__(dynamo_db_client, settings.DYNAMODB_PROGRAMS_TABLE)
        # A table name, not a second service: the faculty row is only ever touched as one action of
        # a transaction this repository is already sending, and an action can name its own table.
        self._faculties_table = settings.DYNAMODB_FACULTIES_TABLE

    async def create_program(self, creation: ProgramCreate) -> tuple[CreateOutcome, ProgramItem | None]:
        """Store a new programme, its name reservation, and the fact of it on its faculty."""
        item = ProgramItem(**creation.model_dump())
        try:
            await self.transact_write(
                TransactUpdate(
                    FacultyItem.key(creation.faculty_id),
                    f'ADD {DEPENDANTS} :one',
                    {':one': 1},
                    condition=Attr(KeyAttr.PK).exists(),
                    table=self._faculties_table,
                ),
                TransactPut(item, condition=Attr(KeyAttr.PK).not_exists()),
                TransactPut(
                    ProgramNameItem(faculty_id=item.faculty_id, name=item.name, program_id=item.id),
                    condition=Attr(KeyAttr.PK).not_exists(),
                ),
            )
        except self._client.exceptions.TransactionCanceledException as exc:
            failed = self.get_failed_condition_indexes(exc)
            if _PARENT_EXISTS in failed:
                return CreateOutcome.PARENT_NOT_FOUND, None
            if _NAME_RESERVATION not in failed:
                raise
            return CreateOutcome.NAME_TAKEN, None
        return CreateOutcome.CREATED, item

    async def list_programs(self) -> list[ProgramItem]:
        """Every programme. A scan of a table holding programmes only, filtered to the ``#META``
        rows so the name reservations beside them are stepped over."""
        res = await self.scan(filter_expression=Attr(KeyAttr.SK).eq(ProgramItem.meta_sk))
        return [ProgramItem.model_validate(r) for r in res]

    async def delete_program(self, program_id: str) -> DeleteOutcome:
        """Delete the programme and its name reservation, and take it off its faculty's counter."""
        row = await self.get_item(ProgramItem.key(program_id), consistent_read=True)
        if row is None:
            return DeleteOutcome.NOT_FOUND

        program = ProgramItem.model_validate(row)
        try:
            await self.transact_write(
                TransactDelete(
                    ProgramItem.key(program_id),
                    condition=Attr(KeyAttr.PK).exists() & (Attr(DEPENDANTS).not_exists() | Attr(DEPENDANTS).eq(0)),
                ),
                TransactDelete(ProgramNameItem.key(program.faculty_id, program.name)),
                TransactUpdate(
                    FacultyItem.key(program.faculty_id),
                    f'ADD {DEPENDANTS} :minus',
                    {':minus': -1},
                    table=self._faculties_table,
                ),
            )
        except self._client.exceptions.TransactionCanceledException as exc:
            if _PROGRAM_ROW not in self.get_failed_condition_indexes(exc):
                raise
            still_there = await self.get_item(ProgramItem.key(program_id), consistent_read=True)
            if still_there is None:
                return DeleteOutcome.NOT_FOUND
            return DeleteOutcome.HAS_DEPENDANTS
        return DeleteOutcome.DELETED
