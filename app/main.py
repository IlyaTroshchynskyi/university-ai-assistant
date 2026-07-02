from fastapi import FastAPI
from pydantic import BaseModel

from app.ai_assistant.main import answer_question

app = FastAPI(title='University Assistant')


class AskRequest(BaseModel):
    question: str


class AskResponse(BaseModel):
    answer: str


@app.get('/health')
async def health() -> dict[str, str]:
    return {'status': 'ok'}


@app.post('/ask', response_model=AskResponse)
async def ask(request: AskRequest) -> AskResponse:
    answer = await answer_question(request.question)
    return AskResponse(answer=answer)
