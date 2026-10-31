from typing import Annotated

from fastapi import Depends

from app.api.v1.rooms.repository import RoomsRepository
from app.api.v1.rooms.schemas import Room, RoomCreate
from app.core.exceptions import NotFoundError


class RoomsService:
    def __init__(self, repository: Annotated[RoomsRepository, Depends(RoomsRepository)]) -> None:
        self._repository = repository

    async def create_room(self, creation: RoomCreate) -> Room:
        room = await self._repository.create_room(creation)
        return room

    async def find_room(self, room_id: str) -> Room:
        room = await self._repository.find_room(room_id)
        if room is None:
            raise NotFoundError(f'Room not found with id = {room_id}')
        return room

    async def delete_room(self, room_id: str) -> None:
        if not await self._repository.delete_room(room_id):
            raise NotFoundError(f'Room not found with id = {room_id}')

    async def list_rooms(self) -> list[Room]:
        return await self._repository.list_rooms()
