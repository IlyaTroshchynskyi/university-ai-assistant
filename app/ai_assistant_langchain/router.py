from fastapi import APIRouter, Depends

from app.ai_assistant_langchain.schemas import ChatSchemaOut, UserQuery
from app.ai_assistant_langchain.service import ChatService

router = APIRouter(tags=['Langchain assistant'])


@router.post(
    '/langchain-assistant',
)
async def chat_with_user(
    input_data: UserQuery,
    service: ChatService = Depends(),
) -> ChatSchemaOut:
    return await service.process_user_query(input_data.user_id, input_data.query)
