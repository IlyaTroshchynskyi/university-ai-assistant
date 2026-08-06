from fastapi import APIRouter, Depends

from app.api.v1.programs.schemas import ProgramCreate, ProgramItem
from app.api.v1.programs.service import ProgramService

router = APIRouter(tags=['Programs'])


@router.post('/programs', status_code=201)
async def create_program(
    creation: ProgramCreate,
    service: ProgramService = Depends(),
) -> ProgramItem:
    return await service.create_program(creation)


@router.get('/programs')
async def get_all_programs(
    service: ProgramService = Depends(),
) -> list[ProgramItem]:
    return await service.list_programs()


@router.delete('/programs/{program_id}', status_code=204)
async def delete_program(
    program_id: str,
    service: ProgramService = Depends(),
) -> None:
    await service.delete_program(program_id)
