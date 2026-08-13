from operator import attrgetter
from typing import Annotated

from boto3.dynamodb.conditions import Attr
from fastapi import Depends
from types_aiobotocore_dynamodb import DynamoDBClient

from app.api.v1.rooms.schemas import CreateRoom, Room, RoomItem
from app.core.dynamodb.base_service import DynamoDBService
from app.core.dynamodb.client import get_dynamo_client
from app.core.dynamodb.indexes import KeyAttr
from app.settings import get_settings, Settings


class RoomsRepository(DynamoDBService):
    def __init__(
        self,
        dynamo_db_client: Annotated[DynamoDBClient, Depends(get_dynamo_client)],
        settings: Annotated[Settings, Depends(get_settings)],
    ) -> None:
        super().__init__(dynamo_db_client, settings.DYNAMODB_ROOMS_TABLE)

    async def create_room(self, creation: CreateRoom) -> Room:
        """Store a new room under a freshly minted id, and return it as the API sees it."""
        item = RoomItem(**creation.model_dump())
        await self.put_item(item, condition=Attr(KeyAttr.PK).not_exists())
        return Room(**item.model_dump())

    async def find_room(self, room_id: str) -> Room | None:
        """The room with this id, or ``None`` if there is no such room."""
        row = await self.get_item(RoomItem.key(room_id))
        return Room(**row) if row is not None else None

    async def delete_room(self, room_id: str) -> bool:
        """Delete the room, reporting whether it was there to delete."""
        try:
            await self.delete_item(RoomItem.key(room_id), condition=Attr(KeyAttr.PK).exists())
        except self._client.exceptions.ConditionalCheckFailedException:
            return False
        return True

    async def list_rooms(self) -> list[Room]:
        """Every room, ordered by building and door number."""
        rows = await self.scan(filter_expression=Attr(KeyAttr.SK).eq(RoomItem.meta_sk))
        return sorted((Room(**row) for row in rows), key=attrgetter('building', 'number'))
