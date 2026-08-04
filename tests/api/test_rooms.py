from operator import itemgetter

from app.api.v1.rooms.schemas import Room
from app.settings import get_settings
from tests.conftest import TestBaseClientDBClass
from tests.factories.factory_creators import create_test_room
from tests.factories.factory_getters import get_test_room
from tests.factories.rooms_factory import RoomCreationFactory


class TestRooms(TestBaseClientDBClass):
    async def test_create_room(self):
        room = RoomCreationFactory.build()

        response = await self.not_auth_client.post('/rooms', json=room.model_dump())

        assert response.status_code == 201
        assert Room.model_validate(response.json()).model_dump(exclude={'id'}) == room.model_dump()

    async def test_list_room(self):
        room1 = RoomCreationFactory.build()
        room2 = RoomCreationFactory.build()
        settings = get_settings()
        await create_test_room(room1, self.dynamo_client, settings)
        await create_test_room(room2, self.dynamo_client, settings)

        response = await self.not_auth_client.get('/rooms')

        assert response.status_code == 200

        # The endpoint orders by GSI1's sort key (building, then door number), which the factory
        # fills at random — so both sides are sorted the same way before comparing.
        result = sorted(
            (Room.model_validate(i).model_dump(exclude={'id'}) for i in response.json()),
            key=itemgetter('building', 'number'),
        )
        assert result == sorted((room1.model_dump(), room2.model_dump()), key=itemgetter('building', 'number'))

    async def test_list_rooms_orders_by_building_then_number(self):
        settings = get_settings()
        for building, number in (('Turing Hall', 201), ('Ada Wing', 3), ('Turing Hall', 105)):
            room = RoomCreationFactory.build(building=building, number=number)
            await create_test_room(room, self.dynamo_client, settings)

        response = await self.not_auth_client.get('/rooms')

        assert response.status_code == 200
        # 105 before 201: the GSI1 sort key zero-pads the number, so it compares as '0105' < '0201'
        # rather than as '105' > '201'.
        assert [(r['building'], r['number']) for r in response.json()] == [
            ('Ada Wing', 3),
            ('Turing Hall', 105),
            ('Turing Hall', 201),
        ]

    async def test_find_room(self):
        created = await create_test_room(RoomCreationFactory.build(), self.dynamo_client, get_settings())

        response = await self.not_auth_client.get(f'/rooms/{created.id}')

        assert response.status_code == 200
        assert Room.model_validate(response.json()) == created

    async def test_find_room_missing(self):
        response = await self.not_auth_client.get('/rooms/no-such-id')

        assert response.status_code == 404
        assert response.json() == {'detail': 'Room not found with id = no-such-id'}

    async def test_delete_room(self):
        created = await create_test_room(RoomCreationFactory.build(), self.dynamo_client, get_settings())

        response = await self.not_auth_client.delete(f'/rooms/{created.id}')

        assert response.status_code == 204
        assert await get_test_room(created.id, self.dynamo_client, get_settings()) is None

    async def test_delete_room_missing(self):
        response = await self.not_auth_client.delete('/rooms/no-such-id')

        assert response.status_code == 404
        assert response.json() == {'detail': 'Room not found with id = no-such-id'}
