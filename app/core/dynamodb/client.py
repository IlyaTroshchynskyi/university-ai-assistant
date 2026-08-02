"""Async DynamoDB client wiring, in the aiobotocore session + FastAPI-dependency style.

A low-level ``DynamoDBClient`` is an **async context manager**: it must be entered before use and
closed after. Rather than manage that lifecycle by hand, we lean on the framework — a cached
``AioSession`` (sessions are cheap and hold no connections) plus a small ``@asynccontextmanager``
that opens one client, and a FastAPI dependency that yields it for the lifetime of a request.
"""

from contextlib import asynccontextmanager
from functools import lru_cache
from typing import AsyncGenerator

from aiobotocore.session import AioSession, get_session
from fastapi import Depends
from types_aiobotocore_dynamodb import DynamoDBClient

from app.settings import get_settings, Settings


@lru_cache
def get_aioboto_session() -> AioSession:
    return get_session()


@asynccontextmanager
async def open_dynamo_client(session: AioSession, settings: Settings) -> AsyncGenerator[DynamoDBClient, None]:
    """Open one low-level DynamoDB client and close it on exit. The reusable core (no ``Depends``,
    so it works outside a request): ``async with open_dynamo_client(session, settings) as client``.
    ``get_dynamo_client`` wraps it for FastAPI DI."""
    async with session.create_client(
        'dynamodb',
        endpoint_url=settings.DYNAMODB_ENDPOINT_URL,
        aws_access_key_id=settings.AWS_ACCESS_KEY_ID,
        aws_secret_access_key=settings.AWS_SECRET_ACCESS_KEY,
        region_name=settings.AWS_REGION,
    ) as client:
        yield client


async def get_dynamo_client(
    session: AioSession = Depends(get_aioboto_session),
    settings: Settings = Depends(get_settings),
) -> AsyncGenerator[DynamoDBClient, None]:
    """FastAPI dependency: yields a request-scoped DynamoDB client (opened for the request, closed
    when it ends). Inject it into a route or build a ``DynamoDBService`` around it."""
    async with open_dynamo_client(session=session, settings=settings) as client:
        yield client
