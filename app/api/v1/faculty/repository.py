from typing import Annotated

from boto3.dynamodb.conditions import Attr
from fastapi import Depends
from types_aiobotocore_dynamodb import DynamoDBClient

from app.api.v1.faculty.schemas import FacultyCreate, FacultyItem, FacultyNameItem
from app.core.dynamodb.base_items import DEPENDANTS
from app.core.dynamodb.base_service import DynamoDBService
from app.core.dynamodb.client import get_dynamo_client
from app.core.dynamodb.indexes import KeyAttr
from app.core.dynamodb.schemas import TransactDelete, TransactPut
from app.core.enums import DeleteOutcome
from app.settings import get_settings, Settings

# Position of the name reservation among the actions ``create_faculty`` sends, which is how a
# cancelled transaction is read back: a condition that failed *there* means the name is taken.
_NAME_RESERVATION = 1

# Position of the faculty row itself in ``delete_faculty``. Its condition fails for two different
# reasons, which is why reading it back takes one more request — see ``delete_faculty``.
_FACULTY_ROW = 0


class FacultyRepository(DynamoDBService):
    def __init__(
        self,
        dynamo_db_client: Annotated[DynamoDBClient, Depends(get_dynamo_client)],
        settings: Annotated[Settings, Depends(get_settings)],
    ) -> None:
        super().__init__(dynamo_db_client, settings.DYNAMODB_FACULTIES_TABLE)

    async def create_faculty(self, creation: FacultyCreate) -> FacultyItem | None:
        """Store a new faculty together with the row reserving its name, or return ``None`` if that
        name is already taken.

        The two rows go in one transaction because neither is any use alone: a faculty whose name
        was never reserved lets the next create take it as well, and a reservation without its
        """
        item = FacultyItem(**creation.model_dump())
        try:
            await self.transact_write(
                TransactPut(item, condition=Attr(KeyAttr.PK).not_exists()),
                TransactPut(
                    FacultyNameItem(name=item.name, faculty_id=item.id),
                    condition=Attr(KeyAttr.PK).not_exists(),
                ),
            )
        except self._client.exceptions.TransactionCanceledException as exc:
            if _NAME_RESERVATION not in self.get_failed_condition_indexes(exc):
                raise
            return None
        return item

    async def list_faculties(self) -> list[FacultyItem]:
        """Every faculty in the table."""
        res = await self.scan(filter_expression=Attr(KeyAttr.SK).eq(FacultyItem.meta_sk))
        return [FacultyItem.model_validate(r) for r in res]

    async def delete_faculty(self, faculty_id: str) -> DeleteOutcome:
        """Delete the faculty along with its name reservation, unless something is still filed under it."""
        row = await self.get_item(FacultyItem.key(faculty_id), consistent_read=True)
        if row is None:
            return DeleteOutcome.NOT_FOUND

        try:
            await self.transact_write(
                TransactDelete(
                    FacultyItem.key(faculty_id),
                    condition=Attr(KeyAttr.PK).exists() & (Attr(DEPENDANTS).not_exists() | Attr(DEPENDANTS).eq(0)),
                ),
                TransactDelete(FacultyNameItem.key(FacultyItem.model_validate(row).name)),
            )
        except self._client.exceptions.TransactionCanceledException as exc:
            if _FACULTY_ROW not in self.get_failed_condition_indexes(exc):
                raise
            still_there = await self.get_item(FacultyItem.key(faculty_id), consistent_read=True)
            if still_there is None:
                return DeleteOutcome.NOT_FOUND
            return DeleteOutcome.HAS_DEPENDANTS
        return DeleteOutcome.DELETED
