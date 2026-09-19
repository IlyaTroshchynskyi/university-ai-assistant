from uuid import UUID

from fastapi import APIRouter, Depends, Request
from sse_starlette import EventSourceResponse

from app.ai_assistant_langchain.schemas import ChatSchemaOut, HistoryMessage, UserQuery
from app.ai_assistant_langchain.service import ChatService, SSE_MEDIA_TYPE

router = APIRouter(tags=['Langchain assistant'])


@router.post(
    '/langchain-assistant',
    response_model=ChatSchemaOut,
    responses={200: {'content': {SSE_MEDIA_TYPE: {'schema': {'type': 'string'}}}}},
)
async def chat_with_user(
    input_data: UserQuery,
    request: Request,
    service: ChatService = Depends(),
) -> ChatSchemaOut | EventSourceResponse:
    """One path, two representations. A client that asks for `text/event-stream` gets SSE; a client
    that asks for anything else — including nothing at all — gets the JSON it has always got.
    """
    if SSE_MEDIA_TYPE not in request.headers.get('accept', ''):
        return await service.process_turn(input_data)

    queue = await service.start_streaming_turn(input_data)
    return EventSourceResponse(service.stream_events(queue))


@router.get('/langchain-assistant/{user_id}/history')
async def get_chat_history(
    user_id: UUID,
    service: ChatService = Depends(),
) -> list[HistoryMessage]:
    # Todo paginate this. It returns the whole thread today, and a long one is both a slow response
    return await service.get_history(user_id)
