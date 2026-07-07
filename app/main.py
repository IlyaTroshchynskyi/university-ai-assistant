import asyncio
from typing import Annotated

from fastapi import FastAPI, File, Form, UploadFile
from fastapi.openapi.utils import get_openapi
from pydantic import BaseModel

from app.ai_assistant.main_flow_service import MainFlowService
from app.ai_assistant.tools.booking_tools import SLOTS
from app.settings import build_llm

app = FastAPI(title='University Assistant')


def _walk(node: object):
    """Yield every dict inside a nested JSON structure."""
    if isinstance(node, dict):
        yield node
        for value in node.values():
            yield from _walk(value)
    elif isinstance(node, list):
        for value in node:
            yield from _walk(value)


def _openapi() -> dict:
    """Swagger UI renders a file picker for a single UploadFile but not for the items of a
    list[UploadFile]: pydantic v2 emits the OpenAPI 3.1 ``contentMediaType`` encoding, which
    Swagger UI doesn't recognise inside arrays. Rewrite it to the older ``format: binary`` so
    multi-file upload works in the docs UI."""
    if not app.openapi_schema:
        schema = get_openapi(title=app.title, version=app.version, routes=app.routes)
        for node in _walk(schema):
            if node.get('type') == 'string' and node.pop('contentMediaType', None):
                node['format'] = 'binary'
        app.openapi_schema = schema
    return app.openapi_schema


app.openapi = _openapi


class AskResponse(BaseModel):
    answer: str


@app.get('/health')
async def health() -> dict[str, str]:
    return {'status': 'ok'}


@app.post('/ask', response_model=AskResponse)
async def ask(
    question: str = Form(''),
    session_id: str = Form('default'),
    files: Annotated[list[UploadFile] | None, File()] = None,
) -> AskResponse:
    """Ask a question and/or upload documents (multipart/form-data). If any files are
    attached, the turn is routed to document verification."""
    images = await asyncio.gather(*(f.read() for f in files or []))
    answer = await MainFlowService(build_llm()).answer_question(question, session_id, images)
    return AskResponse(answer=answer)


@app.get('/slots')
async def slots(status: str | None = None) -> list[dict]:
    """Inspect consultation slots (in-memory). Optionally filter by status, e.g. ?status=booked."""
    if status:
        return [s for s in SLOTS if s['status'] == status]
    return SLOTS
