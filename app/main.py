import asyncio
from typing import Annotated

from fastapi import Depends, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.openapi.utils import get_openapi
from pydantic import BaseModel

from app.ai_assistant.main_flow_service import get_main_flow_service, MainFlowService
from app.ai_assistant.university_knowladge.knowledge_service import get_knowledge_service, KnowledgeService
from app.api.v1.rooms.router import router as rooms_router
from app.core.dynamodb.schemas import Slot, SlotStatus
from app.core.dynamodb.slots_repository import open_slots_repository
from app.core.execption_handler import include_exception_handlers


def create_app() -> FastAPI:
    app = FastAPI(title='University Assistant')
    include_exception_handlers(app)
    app.include_router(rooms_router)

    return app


app = create_app()


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


class IngestResponse(BaseModel):
    source: str
    chunks: int


@app.get('/health')
async def health() -> dict[str, str]:
    return {'status': 'ok'}


@app.post('/ask', response_model=AskResponse)
async def ask(
    service: Annotated[MainFlowService, Depends(get_main_flow_service)],
    question: str = Form(''),
    session_id: str = Form(),
    files: Annotated[list[UploadFile] | None, File()] = None,
) -> AskResponse:
    """Ask a question and/or upload documents (multipart/form-data). If any files are
    attached, the turn is routed to document verification."""
    images = await asyncio.gather(*(f.read() for f in files or []))
    answer = await service.answer_question(question, session_id, images)
    return AskResponse(answer=answer)


@app.post('/documents', response_model=IngestResponse)
async def ingest_document(
    file: Annotated[UploadFile, File()],
    knowledge: Annotated[KnowledgeService, Depends(get_knowledge_service)],
    doc_type: str | None = Form('general'),
) -> IngestResponse:
    """Ingest a PDF into the knowledge base: extract text, split into chunks, embed and store
    them in Qdrant so the retriever can find them. ``doc_type`` (e.g. ``program`` / ``policy`` /
    ``admissions``) is an optional facet a search can later filter on."""
    if file.content_type != 'application/pdf' and not (file.filename or '').lower().endswith('.pdf'):
        raise HTTPException(status_code=400, detail='Only PDF files are supported.')

    source = file.filename or 'upload.pdf'
    chunks = await knowledge.ingest_pdf(await file.read(), source, doc_type=doc_type)
    return IngestResponse(source=source, chunks=chunks)


@app.get('/slots')
async def slots(status: str | None = None) -> list[dict]:
    """Inspect consultation slots from DynamoDB. Optionally filter by status, e.g. ?status=booked."""
    # Todo refactor to avoid repo here
    async with open_slots_repository() as repo:
        if status:
            try:
                wanted = SlotStatus(status)
            except ValueError:
                raise HTTPException(status_code=400, detail=f'Unknown status {status!r}.')
            found = await repo.list_slots_by_status(wanted)
        else:
            found = [Slot.model_validate(item) for item in await repo.scan()]
    return [slot.model_dump() for slot in found]
