import asyncio
from contextlib import asynccontextmanager
import logging
from typing import AsyncGenerator

from app.core.exceptions import ConflictingStatusError

logger = logging.getLogger(__name__)

_BUSY_USER_IDS: set[str] = set()
_STREAMING_TURNS_BY_USER_ID: dict[str, asyncio.Task[None]] = {}


def refuse_busy_thread(user_id: str) -> None:
    if user_id in _BUSY_USER_IDS:
        raise ConflictingStatusError(
            'A turn is still running on this thread, so it cannot take another message yet. A turn '
            'whose connection dropped keeps running for a few seconds so that it can be recorded — '
            'wait for it and send the message again.'
        )


@asynccontextmanager
async def claim_thread(user_id: str) -> AsyncGenerator[None, None]:
    refuse_busy_thread(user_id)
    _BUSY_USER_IDS.add(user_id)
    try:
        yield
    finally:
        _BUSY_USER_IDS.discard(user_id)


def register_streaming_turn(user_id: str, task: asyncio.Task[None]) -> None:
    _BUSY_USER_IDS.add(user_id)
    _STREAMING_TURNS_BY_USER_ID[user_id] = task
    task.add_done_callback(lambda finished: _release_streaming_turn(user_id, finished))


def _release_streaming_turn(user_id: str, task: asyncio.Task[None]) -> None:
    if _STREAMING_TURNS_BY_USER_ID.get(user_id) is task:
        del _STREAMING_TURNS_BY_USER_ID[user_id]
        _BUSY_USER_IDS.discard(user_id)


async def wait_for_streaming_turns(timeout: float) -> None:
    if not _STREAMING_TURNS_BY_USER_ID:
        return

    logger.info('Waiting up to %.0fs for %d streaming turn(s) to finish', timeout, len(_STREAMING_TURNS_BY_USER_ID))
    _, pending = await asyncio.wait(_STREAMING_TURNS_BY_USER_ID.values(), timeout=timeout)
    if pending:
        logger.error(
            '%d streaming turn(s) did not finish within %.0fs; their transcript rows are lost, '
            'though the turns themselves are checkpointed',
            len(pending),
            timeout,
        )
