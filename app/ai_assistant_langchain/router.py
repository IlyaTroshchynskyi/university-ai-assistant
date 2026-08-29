from uuid import UUID

from fastapi import APIRouter, Depends

from app.ai_assistant_langchain.schemas import ChatSchemaOut, HistoryMessage, UserQuery
from app.ai_assistant_langchain.service import ChatService

router = APIRouter(tags=['Langchain assistant'])


@router.post(
    '/langchain-assistant',
)
async def chat_with_user(
    input_data: UserQuery,
    service: ChatService = Depends(),
) -> ChatSchemaOut:
    return await service.process_turn(input_data)


@router.get('/langchain-assistant/{user_id}/history')
async def get_chat_history(
    user_id: UUID,
    service: ChatService = Depends(),
) -> list[HistoryMessage]:
    # Todo paginate this. It returns the whole thread today, and a long one is both a slow response
    return await service.get_history(user_id)
