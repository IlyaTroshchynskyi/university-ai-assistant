from app.api.v1.programs.schemas import ProgramItem, ProgramNameItem
from tests.conftest import TestBaseClientDBClass
from tests.factories.factory_creators import (
    create_test_faculty,
    create_test_group_row,
    create_test_program,
    create_test_room,
)
from tests.factories.factory_getters import get_test_program, get_test_program_name_reservation
from tests.factories.faculty_factory import FacultyCreationFactory
from tests.factories.program_factory import ProgramCreationFactory
from tests.factories.rooms_factory import RoomCreationFactory


class TestPrograms(TestBaseClientDBClass):
    async def test_create_program(self) -> None:
        faculty = await create_test_faculty(FacultyCreationFactory.build(), self.dynamo_client)
        program = ProgramCreationFactory.build(faculty_id=faculty.id)

        response = await self.not_auth_client.post('/programs', json=program.model_dump())

        assert response.status_code == 201
        body = response.json()
        assert body == ProgramItem(id=body['id'], **program.model_dump()).model_dump()

    async def test_create_program_files_it_under_its_faculty(self) -> None:
        faculty = await create_test_faculty(FacultyCreationFactory.build(), self.dynamo_client)
        program = ProgramCreationFactory.build(faculty_id=faculty.id)

        response = await self.not_auth_client.post('/programs', json=program.model_dump())

        assert response.status_code == 201
        program_id = response.json()['id']
        stored = await get_test_program(program_id, self.dynamo_client)
        assert stored == ProgramItem(id=program_id, **program.model_dump())

    async def test_create_program_reserves_its_name(self) -> None:
        faculty = await create_test_faculty(FacultyCreationFactory.build(), self.dynamo_client)
        program = ProgramCreationFactory.build(faculty_id=faculty.id)

        response = await self.not_auth_client.post('/programs', json=program.model_dump())

        assert response.status_code == 201
        reservation = await get_test_program_name_reservation(faculty.id, program.name, self.dynamo_client)
        assert reservation == ProgramNameItem(
            faculty_id=faculty.id, name=program.name, program_id=response.json()['id']
        )

    async def test_create_program_rejects_duplicate_name_in_the_same_faculty(self) -> None:
        faculty = await create_test_faculty(FacultyCreationFactory.build(), self.dynamo_client)
        program = ProgramCreationFactory.build(faculty_id=faculty.id)
        await self.not_auth_client.post('/programs', json=program.model_dump())

        response = await self.not_auth_client.post('/programs', json=program.model_dump())

        assert response.status_code == 409
        assert response.json() == {
            'detail': f'Program already exists with name = {program.name} in faculty = {faculty.id}'
        }

    async def test_create_program_allows_the_same_name_in_another_faculty(self) -> None:
        first = await create_test_faculty(FacultyCreationFactory.build(), self.dynamo_client)
        second = await create_test_faculty(FacultyCreationFactory.build(), self.dynamo_client)
        await create_test_program(first.id, self.dynamo_client, name='Data Science')
        program = ProgramCreationFactory.build(faculty_id=second.id, name='Data Science')

        response = await self.not_auth_client.post('/programs', json=program.model_dump())

        assert response.status_code == 201
        body = response.json()
        assert body == ProgramItem(id=body['id'], **program.model_dump()).model_dump()

    async def test_create_program_rejects_name_differing_only_in_case_or_spacing(self) -> None:
        faculty = await create_test_faculty(FacultyCreationFactory.build(), self.dynamo_client)
        program = ProgramCreationFactory.build(faculty_id=faculty.id, name='Software Engineering')
        await self.not_auth_client.post('/programs', json=program.model_dump())

        response = await self.not_auth_client.post(
            '/programs', json=program.model_copy(update={'name': 'software  ENGINEERING'}).model_dump()
        )

        assert response.status_code == 409
        assert response.json() == {
            'detail': f'Program already exists with name = software  ENGINEERING in faculty = {faculty.id}'
        }

    async def test_create_program_strips_padding_from_its_name(self) -> None:
        faculty = await create_test_faculty(FacultyCreationFactory.build(), self.dynamo_client)
        program = ProgramCreationFactory.build(faculty_id=faculty.id, name='Data Science')

        response = await self.not_auth_client.post(
            '/programs', json=program.model_copy(update={'name': '  Data Science  '}).model_dump()
        )

        assert response.status_code == 201
        body = response.json()
        assert body == ProgramItem(id=body['id'], **program.model_dump()).model_dump()

    async def test_create_program_rejects_a_whitespace_only_name(self) -> None:
        faculty = await create_test_faculty(FacultyCreationFactory.build(), self.dynamo_client)
        program = ProgramCreationFactory.build(faculty_id=faculty.id)

        response = await self.not_auth_client.post(
            '/programs', json=program.model_copy(update={'name': '  '}).model_dump()
        )

        assert response.status_code == 422

    async def test_create_program_rejects_unknown_faculty(self) -> None:
        program = ProgramCreationFactory.build(faculty_id='no-such-faculty')

        response = await self.not_auth_client.post('/programs', json=program.model_dump())

        assert response.status_code == 404
        assert response.json() == {'detail': 'Faculty not found with id = no-such-faculty'}

    async def test_list_programs(self) -> None:
        faculty = await create_test_faculty(FacultyCreationFactory.build(), self.dynamo_client)
        first = await create_test_program(faculty.id, self.dynamo_client)
        second = await create_test_program(faculty.id, self.dynamo_client)

        response = await self.not_auth_client.get('/programs')

        assert response.status_code == 200
        expected = [first.model_dump(), second.model_dump()]
        assert sorted(response.json(), key=lambda p: p['id']) == sorted(expected, key=lambda p: p['id'])

    async def test_list_programs_skips_other_entities(self) -> None:
        faculty = await create_test_faculty(FacultyCreationFactory.build(), self.dynamo_client)
        program = await create_test_program(faculty.id, self.dynamo_client)
        await create_test_room(RoomCreationFactory.build(), self.dynamo_client)

        response = await self.not_auth_client.get('/programs')

        assert response.status_code == 200
        assert response.json() == [program.model_dump()]

    async def test_delete_program(self) -> None:
        faculty = await create_test_faculty(FacultyCreationFactory.build(), self.dynamo_client)
        created = await create_test_program(faculty.id, self.dynamo_client)

        response = await self.not_auth_client.delete(f'/programs/{created.id}')

        assert response.status_code == 204
        assert await get_test_program(created.id, self.dynamo_client) is None

    async def test_delete_program_frees_its_name(self) -> None:
        faculty = await create_test_faculty(FacultyCreationFactory.build(), self.dynamo_client)
        created = await create_test_program(faculty.id, self.dynamo_client)

        response = await self.not_auth_client.delete(f'/programs/{created.id}')

        assert response.status_code == 204
        reservation = await get_test_program_name_reservation(faculty.id, created.name, self.dynamo_client)
        assert reservation is None

    async def test_delete_program_with_dependants(self) -> None:
        faculty = await create_test_faculty(FacultyCreationFactory.build(), self.dynamo_client)
        created = await create_test_program(faculty.id, self.dynamo_client)
        await create_test_group_row(created.id, self.dynamo_client)

        response = await self.not_auth_client.delete(f'/programs/{created.id}')

        assert response.status_code == 409
        assert response.json() == {'detail': f'Program with id = {created.id} still has groups'}
        assert await get_test_program(created.id, self.dynamo_client) == created

    async def test_delete_program_missing(self) -> None:
        response = await self.not_auth_client.delete('/programs/no-such-id')

        assert response.status_code == 404
        assert response.json() == {'detail': 'Program not found with id = no-such-id'}
