from fastapi import APIRouter, Depends

from app.api.v1.faculty.schemas import FacultyCreate, FacultyItem
from app.api.v1.faculty.service import FacultyService

router = APIRouter(tags=['Faculty'])


@router.post('/faculties', status_code=201)
async def create_faculty(
    creation: FacultyCreate,
    service: FacultyService = Depends(),
) -> FacultyItem:
    return await service.create_faculty(creation)


@router.get('/faculties')
async def get_all_faculties(
    service: FacultyService = Depends(),
) -> list[FacultyItem]:
    return await service.list_faculties()


@router.delete('/faculties/{faculty_id}', status_code=204)
async def delete_faculty(
    faculty_id: str,
    service: FacultyService = Depends(),
) -> None:
    await service.delete_faculty(faculty_id)
