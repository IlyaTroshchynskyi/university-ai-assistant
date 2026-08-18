from contextlib import asynccontextmanager
from typing import AsyncGenerator

from boto3.dynamodb.conditions import Attr, Key
from botocore.exceptions import ClientError
from pydantic import EmailStr
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

    async def list_earliest_open_slots(self, from_date: str, limit: int = 5) -> list[Slot]:
        """The soonest open slots on ``from_date`` or after it, earliest first.

        How "the nearest day with anything free" is answered without probing one day at a time:
        the next open slot can be weeks out, and ``list_open_slots_on_date`` would need a call per
        day to reach it. GSI1 already sorts a status partition by ``{date}#{start_time}``, so the
        rows arrive in the order they are to be proposed in and a range condition on the sort key
        drops the past days inside the query rather than after it.

        ``from_date`` is a bare date against a ``{date}#{start_time}`` sort key, which is what makes
        the whole of that day match: every slot on it shares the prefix and sorts after the plain
        date. ``limit`` caps what DynamoDB reads, and here that is exactly what we want — with no
        filter expression in play the first rows read are the first rows wanted.
        """
        rows = await self.query(
            key_condition=Key(KeyAttr.GSI1_PK).eq(f'STATUS#{SlotStatus.OPEN}') & Key(KeyAttr.GSI1_SK).gte(from_date),
            index_name=Index.GSI1,
            limit=limit,
        )
        return [Slot.model_validate(row) for row in rows]

    async def list_student_bookings(self, email: EmailStr) -> list[Slot]:
        rows = await self.query(
            key_condition=Key(KeyAttr.GSI2_PK).eq(f'STUDENT#{email}'),
            index_name=Index.GSI2,
        )
        return [Slot.model_validate(row) for row in rows]

    async def book_slot(self, date: str, start_time: str, slot_id: int, booked_by: EmailStr, topic: str) -> Slot | None:
        # `topic` is written like every other attribute of a booking: a booked slot has one, and the
        # CrewAI flow's '' for "they never said" is storable as it is — DynamoDB rejects empty
        # strings only in key attributes, and this is not one.
        set_parts = [
            '#st = :booked',
            'booked_by = :by',
            'topic = :topic',
            'gsi1pk = :g1',
            'gsi2pk = :g2',
            'gsi2sk = :g2s',
        ]
        values: dict = {
            ':booked': SlotStatus.BOOKED,
            ':by': booked_by,
            ':topic': topic,
            ':g1': f'STATUS#{SlotStatus.BOOKED}',
            ':g2': f'STUDENT#{booked_by}',
            ':g2s': f'{date}#{start_time}',
        }
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

    async def cancel_slot(self, date: str, start_time: str, slot_id: int, booked_by: EmailStr) -> Slot | None:
        """Cancel ``booked_by``'s slot and free it again. ``None`` if there is no such booking."""
        try:
            updated = await self.update_item(
                key={KeyAttr.PK: date, KeyAttr.SK: f'{start_time}#{slot_id}'},
                update_expression='SET #st = :open, gsi1pk = :g1 REMOVE booked_by, gsi2pk, gsi2sk, topic',
                expression_values={':open': SlotStatus.OPEN, ':g1': f'STATUS#{SlotStatus.OPEN}'},
                expression_names={'#st': 'status'},
                condition=Attr('status').eq(SlotStatus.BOOKED) & Attr('booked_by').eq(booked_by),
            )
        except ClientError as error:
            if error.response['Error']['Code'] == 'ConditionalCheckFailedException':
                return None  # not booked, or booked by somebody else
            raise
        return Slot.model_validate(updated)


@asynccontextmanager
async def open_slots_repository() -> AsyncGenerator[SlotsRepository, None]:
    """Open a ``SlotsRepository`` with its DynamoDB client managed around the block. Builds from the
    cached session + settings singletons; for use outside FastAPI DI (e.g. a CrewAI tool)."""
    async with open_dynamo_client(get_aioboto_session(), get_settings()) as client:
        yield SlotsRepository(client, get_settings())
