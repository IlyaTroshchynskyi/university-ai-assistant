from contextlib import asynccontextmanager
from typing import AsyncGenerator

from boto3.dynamodb.conditions import Attr, Key
from botocore.exceptions import ClientError
from types_aiobotocore_dynamodb import DynamoDBClient

from app.core.dynamodb.base_service import DynamoDBService
from app.core.dynamodb.client import get_aioboto_session, open_dynamo_client
from app.core.dynamodb.indexes import Index, KeyAttr
from app.core.dynamodb.schemas import Slot, SlotStatus
from app.settings import get_settings, Settings


class SlotsRepository(DynamoDBService):
    def __init__(self, client: DynamoDBClient, settings: Settings):
        super().__init__(client, settings.DYNAMODB_SLOTS_TABLE)

    async def list_open_slots_on_date(self, date: str) -> list[Slot]:
        rows = await self.query(
            key_condition=Key(KeyAttr.PK).eq(date),
            filter_expression=Attr('status').eq(SlotStatus.OPEN),
        )
        return [Slot.model_validate(row) for row in rows]

    async def list_slots_by_status(self, status: SlotStatus) -> list[Slot]:
        rows = await self.query(
            key_condition=Key(KeyAttr.GSI1_PK).eq(f'STATUS#{status}'),
            index_name=Index.GSI1,
        )
        return [Slot.model_validate(row) for row in rows]

    async def list_student_bookings(self, email: str) -> list[Slot]:
        rows = await self.query(
            key_condition=Key(KeyAttr.GSI2_PK).eq(f'STUDENT#{email}'),
            index_name=Index.GSI2,
        )
        return [Slot.model_validate(row) for row in rows]

    async def book_slot(
        self, date: str, start_time: str, slot_id: int, booked_by: str, topic: str | None = None
    ) -> Slot | None:
        set_parts = ['#st = :booked', 'booked_by = :by', 'gsi1pk = :g1', 'gsi2pk = :g2', 'gsi2sk = :g2s']
        values: dict = {
            ':booked': SlotStatus.BOOKED,
            ':by': booked_by,
            ':g1': f'STATUS#{SlotStatus.BOOKED}',
            ':g2': f'STUDENT#{booked_by}',
            ':g2s': f'{date}#{start_time}',
        }
        if topic:
            set_parts.append('topic = :topic')
            values[':topic'] = topic
        try:
            updated = await self.update_item(
                key={KeyAttr.PK: date, KeyAttr.SK: f'{start_time}#{slot_id}'},
                update_expression='SET ' + ', '.join(set_parts),
                expression_values=values,
                expression_names={'#st': 'status'},
                condition=Attr('status').eq(SlotStatus.OPEN),
            )
        except ClientError as error:
            if error.response['Error']['Code'] == 'ConditionalCheckFailedException':
                return None  # not open (already booked / doesn't exist)
            raise
        return Slot.model_validate(updated)

    async def cancel_slot(self, date: str, start_time: str, slot_id: int) -> Slot | None:
        """Cancel a booked slot and free it again."""
        try:
            updated = await self.update_item(
                key={KeyAttr.PK: date, KeyAttr.SK: f'{start_time}#{slot_id}'},
                update_expression='SET #st = :open, gsi1pk = :g1 REMOVE booked_by, gsi2pk, gsi2sk',
                expression_values={':open': SlotStatus.OPEN, ':g1': f'STATUS#{SlotStatus.OPEN}'},
                expression_names={'#st': 'status'},
                condition=Attr('status').eq(SlotStatus.BOOKED),
            )
        except ClientError as error:
            if error.response['Error']['Code'] == 'ConditionalCheckFailedException':
                return None  # not booked
            raise
        return Slot.model_validate(updated)


@asynccontextmanager
async def open_slots_repository() -> AsyncGenerator[SlotsRepository, None]:
    """Open a ``SlotsRepository`` with its DynamoDB client managed around the block. Builds from the
    cached session + settings singletons; for use outside FastAPI DI (e.g. a CrewAI tool)."""
    async with open_dynamo_client(get_aioboto_session(), get_settings()) as client:
        yield SlotsRepository(client, get_settings())
