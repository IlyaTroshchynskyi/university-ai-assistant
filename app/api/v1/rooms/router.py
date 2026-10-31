from fastapi import APIRouter, Depends

from app.api.v1.rooms.schemas import Room, RoomCreate
from app.api.v1.rooms.service import RoomsService

router = APIRouter(tags=['Rooms'])


@router.post('/rooms', status_code=201)
async def create_room(
    creation: RoomCreate,
    service: RoomsService = Depends(),
) -> Room:
    return await service.create_room(creation)


@router.get('/rooms')
async def list_rooms(
    service: RoomsService = Depends(),
) -> list[Room]:
    return await service.list_rooms()


@router.get('/rooms/{room_id}')
async def find_room(
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
