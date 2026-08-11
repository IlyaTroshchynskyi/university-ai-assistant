from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI

from app.ai_assistant_langchain.checkpointer.saver import get_checkpointer


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncGenerator[None, None]:
    """Hold the checkpointer's DynamoDB client open for the life of the process.

    Unlike ``get_dynamo_client``, which opens one per request, the checkpointer lives inside the
    cached graph and outlives every request — so its client is opened here, once, and closed here.
    Startup is also where a bad endpoint or missing table should surface, rather than on somebody's
    first message.
    """
    async with get_checkpointer().opened():
        yield
