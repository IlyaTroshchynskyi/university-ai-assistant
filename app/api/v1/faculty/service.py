from typing import Annotated

from fastapi import Depends

from app.api.v1.faculty.repository import FacultyRepository
from app.api.v1.faculty.schemas import FacultyCreate, FacultyItem
from app.core.exceptions import AlreadyExistError, NotFoundError


class FacultyService:
    def __init__(self, repo: Annotated[FacultyRepository, Depends(FacultyRepository)]) -> None:
        self._repo = repo

    async def create_faculty(self, creation: FacultyCreate) -> FacultyItem:
        faculty = await self._repo.create_faculty(creation)
        if faculty is None:
            raise AlreadyExistError(f'Faculty already exists with name = {creation.name}')
        return faculty

    async def list_faculties(self) -> list[FacultyItem]:
        return await self._repo.list_faculties()

    async def delete_faculty(self, faculty_id: str) -> None:
        """Delete a faculty that nothing depends on."""
        if await self._repo.has_dependants(faculty_id):
            raise AlreadyExistError(f'Faculty with id = {faculty_id} still has programs, professors or courses')
        if not await self._repo.delete_faculty(faculty_id):
            raise NotFoundError(f'Faculty not found with id = {faculty_id}')
