from fastapi import FastAPI
from pydantic import BaseModel

from app.ai_assistant.main import answer_question
from app.ai_assistant.tools.booking_tools import SLOTS

app = FastAPI(title='University Assistant')


class AskRequest(BaseModel):
    question: str
    session_id: str = 'default'


class AskResponse(BaseModel):
    answer: str


@app.get('/health')
async def health() -> dict[str, str]:
    return {'status': 'ok'}


@app.post('/ask', response_model=AskResponse)
async def ask(request: AskRequest) -> AskResponse:
    answer = await answer_question(request.question, request.session_id)
    return AskResponse(answer=answer)


@app.get('/slots')
async def slots(status: str | None = None) -> list[dict]:
    """Inspect consultation slots (in-memory). Optionally filter by status, e.g. ?status=booked."""
    if status:
        return [s for s in SLOTS if s['status'] == status]
    return SLOTS
