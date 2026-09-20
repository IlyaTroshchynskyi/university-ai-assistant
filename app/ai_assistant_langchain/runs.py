import asyncio
from contextlib import asynccontextmanager
import logging
from typing import AsyncGenerator

from app.core.exceptions import ConflictingStatusError

logger = logging.getLogger(__name__)

_BUSY: set[str] = set()
_RUNS: dict[str, asyncio.Task[None]] = {}


def assert_free(user_id: str) -> None:
    if user_id in _BUSY:
        raise ConflictingStatusError(
            'A turn is still running on this thread, so it cannot take another message yet. A turn '
            'whose connection dropped keeps running for a few seconds so that it can be recorded — '
            'wait for it and send the message again.'
        )


@asynccontextmanager
async def claimed(user_id: str) -> AsyncGenerator[None, None]:
    assert_free(user_id)
    _BUSY.add(user_id)
    try:
        yield
    finally:
        _BUSY.discard(user_id)


def register(user_id: str, task: asyncio.Task[None]) -> None:
    _BUSY.add(user_id)
    _RUNS[user_id] = task
    task.add_done_callback(lambda finished: _release(user_id, finished))


def _release(user_id: str, task: asyncio.Task[None]) -> None:
    if _RUNS.get(user_id) is task:
        del _RUNS[user_id]
        _BUSY.discard(user_id)


async def wait_for_streaming_turns(timeout: float) -> None:
    if not _RUNS:
        return

    logger.info('Waiting up to %.0fs for %d streaming turn(s) to finish', timeout, len(_RUNS))
    _, pending = await asyncio.wait(_RUNS.values(), timeout=timeout)
    if pending:
        logger.error(
            '%d streaming turn(s) did not finish within %.0fs; their transcript rows are lost, '
            'though the turns themselves are checkpointed',
            len(pending),
            timeout,
        )
