from typing import Annotated

from fastapi import Depends

from app.api.v1.programs.repository import ProgramsRepository
from app.api.v1.programs.schemas import ProgramCreate, ProgramItem
from app.core.enums import CreateOutcome, DeleteOutcome
from app.core.exceptions import AlreadyExistError, NotFoundError


class ProgramService:
    def __init__(self, repo: Annotated[ProgramsRepository, Depends(ProgramsRepository)]) -> None:
        self._repo = repo

    async def create_program(self, creation: ProgramCreate) -> ProgramItem:
        outcome, program = await self._repo.create_program(creation)
        # The enum says only 'the parent was missing'; naming *which* parent is the service's job,
        # and the reason the message did not move into the enum with the outcome.
        if outcome is CreateOutcome.PARENT_NOT_FOUND:
            raise NotFoundError(f'Faculty not found with id = {creation.faculty_id}')
        if outcome is CreateOutcome.NAME_TAKEN:
            raise AlreadyExistError(
                f'Program already exists with name = {creation.name} in faculty = {creation.faculty_id}'
            )
        assert program is not None
        return program

    async def list_programs(self) -> list[ProgramItem]:
        return await self._repo.list_programs()

    async def delete_program(self, program_id: str) -> None:
        """Delete a programme that nothing depends on."""
        outcome = await self._repo.delete_program(program_id)
        if outcome is DeleteOutcome.HAS_DEPENDANTS:
            raise AlreadyExistError(f'Program with id = {program_id} still has groups')
        if outcome is DeleteOutcome.NOT_FOUND:
            raise NotFoundError(f'Program not found with id = {program_id}')
