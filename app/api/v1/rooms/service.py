from typing import Annotated

from fastapi import Depends

from app.api.v1.rooms.repository import RoomsRepository
from app.api.v1.rooms.schemas import CreateRoom, Room
from app.core.exceptions import NotFoundError


class RoomsService:
    def __init__(self, repo: Annotated[RoomsRepository, Depends(RoomsRepository)]) -> None:
        self._repo = repo

    async def create_room(self, creation: CreateRoom) -> Room:
        room = await self._repo.create_room(creation)
        return room

    async def find_room(self, room_id: str) -> Room:
        room = await self._repo.find_room(room_id)
        if room is None:
            raise NotFoundError(f'Room not found with id = {room_id}')
        return room

    async def delete_room(self, room_id: str) -> None:
        if not await self._repo.delete_room(room_id):
            raise NotFoundError(f'Room not found with id = {room_id}')

    async def list_rooms(self) -> list[Room]:
        return await self._repo.list_rooms()
