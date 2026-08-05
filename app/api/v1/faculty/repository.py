from typing import Annotated

from boto3.dynamodb.conditions import Attr, Key
from fastapi import Depends
from types_aiobotocore_dynamodb import DynamoDBClient

from app.api.v1.faculty.schemas import FacultyCreate, FacultyItem, FacultyNameItem
from app.core.dynamodb.base_service import DynamoDBService
from app.core.dynamodb.client import get_dynamo_client
from app.core.dynamodb.indexes import Index, KeyAttr
from app.core.dynamodb.schemas import TransactDelete, TransactPut
from app.settings import get_settings, Settings

# Position of the name reservation among the actions ``create_faculty`` sends, which is how a
# cancelled transaction is read back: a condition that failed *there* means the name is taken.
_NAME_RESERVATION = 1


class FacultyRepository(DynamoDBService):
    def __init__(
        self,
        dynamo_db_client: Annotated[DynamoDBClient, Depends(get_dynamo_client)],
        settings: Annotated[Settings, Depends(get_settings)],
    ) -> None:
        super().__init__(dynamo_db_client, settings.DYNAMODB_UNIVERSITY_TABLE)

    async def create_faculty(self, creation: FacultyCreate) -> FacultyItem | None:
        """Store a new faculty together with the row reserving its name, or return ``None`` if that
        name is already taken.

        The two rows go in one transaction because neither is any use alone: a faculty whose name
        was never reserved lets the next create take it as well, and a reservation without its
        faculty blocks a name nobody holds."""
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
        """Every faculty in the table.

        Faculties carry no GSI keys, so there is no partition to query them by — hence a scan,
        narrowed by ``entity_type`` so the other entities sharing the table are dropped. A filter
        applies after the read, so it trims the result, not the amount of table read."""
        res = await self.scan(filter_expression=Attr('entity_type').eq(FacultyItem.entity_type_value()))
        return [FacultyItem.model_validate(r) for r in res]

    async def has_dependants(self, faculty_id: str) -> bool:
        """Whether anything is still filed under this faculty."""
        rows = await self.query(
            Key(KeyAttr.GSI1_PK).eq(f'{FacultyItem.entity}#{faculty_id}'),
            index_name=Index.GSI1,
            limit=1,
        )
        return bool(rows)

    async def delete_faculty(self, faculty_id: str) -> bool:
        """Delete the faculty along with its name reservation, reporting whether it was there to delete."""
        row = await self.get_item(FacultyItem.key(faculty_id), consistent_read=True)
        if row is None:
            return False

        try:
            await self.transact_write(
                TransactDelete(FacultyItem.key(faculty_id), condition=Attr(KeyAttr.PK).exists()),
                TransactDelete(FacultyNameItem.key(FacultyItem.model_validate(row).name)),
            )
        except self._client.exceptions.TransactionCanceledException:
            return False
        return True
