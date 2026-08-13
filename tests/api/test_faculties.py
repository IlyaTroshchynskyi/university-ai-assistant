from app.api.v1.faculty.schemas import FacultyItem
from tests.conftest import TestBaseClientDBClass
from tests.factories.factory_creators import create_test_faculty, create_test_program
from tests.factories.factory_deleters import delete_test_faculty
from tests.factories.factory_getters import (
    get_test_faculty,
    get_test_faculty_dependants,
    get_test_faculty_name_reservation,
)
from tests.factories.faculty_factory import FacultyCreationFactory


class TestFaculties(TestBaseClientDBClass):
    async def test_create_faculty(self) -> None:
        faculty = FacultyCreationFactory.build()

        response = await self.not_auth_client.post('/faculties', json=faculty.model_dump())

        assert response.status_code == 201
        body = response.json()
        assert body == FacultyItem(id=body['id'], name=faculty.name).model_dump()
        assert await get_test_faculty(body['id'], self.dynamo_client) is not None

    async def test_create_faculty_reserves_its_name(self) -> None:
        faculty = FacultyCreationFactory.build()

        response = await self.not_auth_client.post('/faculties', json=faculty.model_dump())

        assert response.status_code == 201
        reservation = await get_test_faculty_name_reservation(faculty.name, self.dynamo_client)
        assert reservation is not None
        assert reservation.faculty_id == response.json()['id']

    async def test_create_faculty_rejects_duplicate_name(self) -> None:
        faculty = FacultyCreationFactory.build()
        await create_test_faculty(faculty, self.dynamo_client)

        response = await self.not_auth_client.post('/faculties', json=faculty.model_dump())

        assert response.status_code == 409
        assert response.json() == {'detail': f'Faculty already exists with name = {faculty.name}'}

    async def test_create_faculty_rejects_name_differing_only_in_case_or_spacing(self) -> None:
        faculty = FacultyCreationFactory.build(name='Computer Science')
        await create_test_faculty(faculty, self.dynamo_client)

        response = await self.not_auth_client.post('/faculties', json={'name': 'computer  science'})

        assert response.status_code == 409

    async def test_create_faculty_with_name_of_a_deleted_one(self) -> None:
        faculty = FacultyCreationFactory.build()
        created = await create_test_faculty(faculty, self.dynamo_client)
        await delete_test_faculty(created.id, self.dynamo_client)

        response = await self.not_auth_client.post('/faculties', json=faculty.model_dump())

        assert response.status_code == 201

    async def test_list_faculties(self) -> None:
        faculty1 = FacultyCreationFactory.build()
        faculty2 = FacultyCreationFactory.build()
        await create_test_faculty(faculty1, self.dynamo_client)
        await create_test_faculty(faculty2, self.dynamo_client)

        response = await self.not_auth_client.get('/faculties')

        assert response.status_code == 200
        # The listing is a scan, which returns items in no defined order — so both sides are sorted
        # by name before comparing.
        assert sorted(f['name'] for f in response.json()) == sorted([faculty1.name, faculty2.name])

    async def test_list_faculties_skips_name_reservations(self) -> None:
        """The reservation row shares the table with the faculty it reserves, so the listing has to
        step over it — the same ``#UNIQUE``/``#META`` split every entity table uses."""
        faculty = await create_test_faculty(FacultyCreationFactory.build(), self.dynamo_client)
        reservation = await get_test_faculty_name_reservation(faculty.name, self.dynamo_client)
        assert reservation is not None, 'the fixture is meaningless if no reservation was written'

        response = await self.not_auth_client.get('/faculties')

        assert response.status_code == 200
        assert [f['name'] for f in response.json()] == [faculty.name]

    async def test_delete_faculty(self) -> None:
        created = await create_test_faculty(FacultyCreationFactory.build(), self.dynamo_client)

        response = await self.not_auth_client.delete(f'/faculties/{created.id}')

        assert response.status_code == 204
        assert await get_test_faculty(created.id, self.dynamo_client) is None

    async def test_delete_faculty_frees_its_name(self) -> None:
        faculty = FacultyCreationFactory.build()
        created = await create_test_faculty(faculty, self.dynamo_client)

        response = await self.not_auth_client.delete(f'/faculties/{created.id}')

        assert response.status_code == 204
        assert await get_test_faculty_name_reservation(faculty.name, self.dynamo_client) is None

    async def test_delete_faculty_with_dependants(self) -> None:
        """The programme is created through the API, because that is what maintains the counter the
        delete refuses on. A programme row written directly would leave the faculty deletable."""
        created = await create_test_faculty(FacultyCreationFactory.build(), self.dynamo_client)
        await create_test_program(created.id, self.dynamo_client)

        response = await self.not_auth_client.delete(f'/faculties/{created.id}')

        assert response.status_code == 409
        assert response.json() == {
            'detail': f'Faculty with id = {created.id} still has programs, professors or courses'
        }
        assert await get_test_faculty(created.id, self.dynamo_client) is not None

    async def test_creating_a_program_counts_it_on_its_faculty(self) -> None:
        created = await create_test_faculty(FacultyCreationFactory.build(), self.dynamo_client)
        assert await get_test_faculty_dependants(created.id, self.dynamo_client) == 0

        await create_test_program(created.id, self.dynamo_client)
        await create_test_program(created.id, self.dynamo_client)

        assert await get_test_faculty_dependants(created.id, self.dynamo_client) == 2

    async def test_deleting_the_last_program_re_permits_deleting_its_faculty(self) -> None:
        """The counter has to come back down, or the 409 above becomes permanent: a faculty that
        once had a programme would stay undeletable after the programme was gone."""
        created = await create_test_faculty(FacultyCreationFactory.build(), self.dynamo_client)
        program = await create_test_program(created.id, self.dynamo_client)

        assert (await self.not_auth_client.delete(f'/programs/{program.id}')).status_code == 204
        assert await get_test_faculty_dependants(created.id, self.dynamo_client) == 0

        response = await self.not_auth_client.delete(f'/faculties/{created.id}')

        assert response.status_code == 204
        assert await get_test_faculty(created.id, self.dynamo_client) is None

    async def test_delete_faculty_missing(self) -> None:
        response = await self.not_auth_client.delete('/faculties/no-such-id')

        assert response.status_code == 404
        assert response.json() == {'detail': 'Faculty not found with id = no-such-id'}
