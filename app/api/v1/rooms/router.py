from fastapi import APIRouter, Depends

from app.api.v1.rooms.schemas import CreateRoom, Room
from app.api.v1.rooms.service import RoomsService

router = APIRouter(tags=['Rooms'])


@router.post('/rooms', status_code=201)
async def create_rooms(
    creation: CreateRoom,
    service: RoomsService = Depends(),
) -> Room:
    return await service.create_room(creation)


@router.get('/rooms')
async def get_all_rooms(
    service: RoomsService = Depends(),
) -> list[Room]:
    return await service.list_rooms()


@router.get('/rooms/{room_id}')
async def get_room(
    room_id: str,
    service: RoomsService = Depends(),
) -> Room:
    return await service.find_room(room_id)


@router.delete('/rooms/{room_id}', status_code=204)
async def delete_room(
    room_id: str,
    service: RoomsService = Depends(),
) -> None:
    await service.delete_room(room_id)
