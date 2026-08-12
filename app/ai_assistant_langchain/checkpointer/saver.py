"""
A ``BaseCheckpointSaver`` over DynamoDB — the agent's conversation memory.
The layout is ``PostgresSaver``'s, collapsed onto one table: a checkpoint row, one blob row per
channel *version*, and one row per pending write.
"""

import asyncio
from contextlib import asynccontextmanager
from functools import lru_cache
import logging
import random
from typing import Any, AsyncGenerator, AsyncIterator, Sequence
import zlib

from aiobotocore.session import AioSession
from boto3.dynamodb.conditions import Attr, Key
from botocore.exceptions import ClientError
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import (
    BaseCheckpointSaver,
    ChannelVersions,
    Checkpoint,
    CheckpointMetadata,
    CheckpointTuple,
    get_checkpoint_id,
    get_checkpoint_metadata,
    PendingWrite,
    SerializerProtocol,
    WRITES_IDX_MAP,
)

from app.ai_assistant_langchain.checkpointer.items import (
    BlobRow,
    CheckpointRow,
    EMPTY_TYPE,
    row_key,
    thread_key,
    WriteRow,
)
from app.core.dynamodb.base_service import DynamoDBService
from app.core.dynamodb.client import get_aioboto_session, open_dynamo_client
from app.core.dynamodb.schemas import TransactAction, TransactPut
from app.settings import get_settings, Settings

logger = logging.getLogger(__name__)

CONDITIONAL_CHECK_FAILED = 'ConditionalCheckFailedException'

# Level 6 is zlib's default: the knee of the ratio/CPU curve. The payload is tens of KB, so this
# costs microseconds against a network round trip — see the measurements in the design doc.
COMPRESSION_LEVEL = 6

SYNC_NOT_SUPPORTED = (
    'DynamoDBCheckpointer is async-only: aiobotocore has no synchronous client. Drive the graph '
    'with ainvoke/astream, which reach aget_tuple/aput/aput_writes/alist.'
)


class DynamoDBCheckpointer(BaseCheckpointSaver[str]):
    """Persists LangGraph state in one DynamoDB table, one partition per thread."""

    def __init__(
        self,
        session: AioSession,
        settings: Settings,
        *,
        serde: SerializerProtocol | None = None,
    ) -> None:
        super().__init__(serde=serde)
        self._session = session
        self._settings = settings
        self._table: DynamoDBService | None = None

    @asynccontextmanager
    async def opened(self) -> AsyncGenerator['DynamoDBCheckpointer', None]:
        """Hold the DynamoDB client open for the duration of the block.

        The app opens this once, in its lifespan; a test opens it around each test. That is the
        whole lifecycle — the client is created and closed on one event loop, by one owner, at a
        point in the code you can see.

        Opening lazily on first use would be more convenient and was tried. It cost a lock (to keep
        two first requests from opening two clients), a record of which loop the client belonged to,
        and a branch that deliberately leaked a client it could not close from the wrong loop — none
        of which the app needed, and one of which was itself loop-bound and broken.
        """
        table_name = self._settings.DYNAMODB_CHECKPOINTS_TABLE
        async with open_dynamo_client(self._session, self._settings) as client:
            self._table = DynamoDBService(client, table_name)
            logger.info('Checkpointer opened on table %s', table_name)
            try:
                yield self
            finally:
                self._table = None
                logger.info('Checkpointer closed')

    @property
    def table(self) -> DynamoDBService:
        """The open client's service, or a clear error naming who was supposed to open it."""
        if self._table is None:
            raise RuntimeError(
                'DynamoDBCheckpointer is closed. The app opens it in its lifespan and tests open it '
                'per test — see TestBaseAgentClass. Wrap the work in `async with checkpointer.opened()`.'
            )
        return self._table

    def get_next_version(self, current: str | int | None, channel: None) -> str:
        """The next version of a channel: an ordered counter plus a random tail.

        The base class would answer ``current + 1``, and that is not safe here. A blob row is keyed
        ``CHECKPOINT_BLOBS#{ns}#{channel}#{version}`` and written unconditionally, so two
        checkpoints forked from the same parent — time travel, ``aupdate_state`` — would both call
        the channel version 2, land on one key, and the second would overwrite the first. Loading
        the first branch afterwards would return the other branch's state, with nothing raised.

        The zero-padded counter keeps versions sorted; the random tail keeps siblings apart. This is
        ``InMemorySaver.get_next_version`` verbatim, and for the same reason — a saver that stores
        blobs per version needs versions that are unique per *write*, not merely increasing.

        The ``isinstance(current, int)`` branch is what carries threads written before this: their
        stored versions are plain integers, and they keep resolving.
        """
        if current is None:
            current_version = 0
        elif isinstance(current, int):
            current_version = current
        else:
            current_version = int(current.split('.')[0])
        return f'{current_version + 1:032}.{random.random():016}'

    # --- serialization ---------------------------------------------------------------------------

    def _pack(self, value: Any) -> tuple[str, bytes]:
        """Serialize, then compress. The type tag is left alone — it routes deserialization, and it
        is a short string. Compression is what keeps a conversation clear of the 400 KB item limit
        and off a per-KB write bill; the design doc has the numbers."""
        type_, payload = self.serde.dumps_typed(value)
        return type_, zlib.compress(payload, COMPRESSION_LEVEL)

    def _unpack(self, type_: str, payload: Any) -> Any:
        """The inverse. ``payload`` arrives as a boto3 ``Binary``, hence ``bytes()``."""
        return self.serde.loads_typed((type_, zlib.decompress(bytes(payload))))

    # --- reads -----------------------------------------------------------------------------------

    async def aget_tuple(self, config: RunnableConfig) -> CheckpointTuple | None:
        thread_id: str = config['configurable']['thread_id']
        checkpoint_ns: str = config['configurable'].get('checkpoint_ns', '')
        table = self.table

        checkpoint_id = get_checkpoint_id(config)
        if checkpoint_id:
            row = await table.get_item(
                row_key(thread_id, CheckpointRow.sort_key(checkpoint_ns, checkpoint_id)),
                consistent_read=True,
            )
        else:
            # Descending with limit=1: DynamoDB reads the single newest row rather than the whole
            # partition. uuid6 ids sort chronologically, so the last one is the latest.
            rows = await table.query(
                Key('pk').eq(thread_key(thread_id)) & Key('sk').begins_with(CheckpointRow.scan_prefix(checkpoint_ns)),
                ascending=False,
                limit=1,
                consistent_read=True,
            )
            row = rows[0] if rows else None

        if row is None:
            return None
        return await self._to_tuple(table, row)

    async def alist(
        self,
        config: RunnableConfig | None,
        *,
        filter: dict[str, Any] | None = None,
        before: RunnableConfig | None = None,
        limit: int | None = None,
    ) -> AsyncIterator[CheckpointTuple]:
        table = self.table
        if config is None:
            # Every thread in the table. An inspection path, not a request path.
            rows = await table.scan(
                filter_expression=Attr('sk').begins_with(CheckpointRow.scan_prefix(None)),
                consistent_read=True,
            )
            rows.sort(key=lambda item: item['checkpoint_id'], reverse=True)
        else:
            rows = await table.query(
                Key('pk').eq(thread_key(config['configurable']['thread_id']))
                & Key('sk').begins_with(CheckpointRow.scan_prefix(config['configurable'].get('checkpoint_ns'))),
                ascending=False,
                consistent_read=True,
            )

        wanted_id = get_checkpoint_id(config) if config else None
        before_id = get_checkpoint_id(before) if before else None
        yielded = 0
        for row in rows:
            if wanted_id and row['checkpoint_id'] != wanted_id:
                continue
            if before_id and row['checkpoint_id'] >= before_id:
                continue
            if limit is not None and yielded >= limit:
                return

            checkpoint_tuple = await self._to_tuple(table, row)
            # Metadata is an opaque serde blob, so DynamoDB cannot filter on it — this is the one
            # filter that has to happen after the read.
            if filter and not all(checkpoint_tuple.metadata.get(key) == value for key, value in filter.items()):
                continue

            yielded += 1
            yield checkpoint_tuple

    async def _to_tuple(self, table: DynamoDBService, row: dict[str, Any]) -> CheckpointTuple:
        """Rebuild a full ``CheckpointTuple`` from a checkpoint row.

        The channel values and the pending writes are two more round trips, and neither needs the
        other, so they go out together. Sequentially this would make every ``aget_tuple`` three
        serial round trips instead of two — and ``alist``, which calls this once per row, twice that
        per checkpoint listed.
        """
        thread_id: str = row['thread_id']
        checkpoint_ns: str = row['checkpoint_ns']
        checkpoint_id: str = row['checkpoint_id']

        checkpoint: Checkpoint = self._unpack(row['checkpoint_type'], row['checkpoint_value'])
        async with asyncio.TaskGroup() as group:
            blobs = group.create_task(self._load_blobs(table, thread_id, checkpoint_ns, checkpoint['channel_versions']))
            writes = group.create_task(self._load_writes(table, thread_id, checkpoint_ns, checkpoint_id))

        checkpoint['channel_values'] = blobs.result()
        parent_id = row.get('parent_checkpoint_id')

        return CheckpointTuple(
            config={
                'configurable': {
                    'thread_id': thread_id,
                    'checkpoint_ns': checkpoint_ns,
                    'checkpoint_id': checkpoint_id,
                }
            },
            checkpoint=checkpoint,
            metadata=self._unpack(row['metadata_type'], row['metadata_value']),
            parent_config=(
                {
                    'configurable': {
                        'thread_id': thread_id,
                        'checkpoint_ns': checkpoint_ns,
                        'checkpoint_id': parent_id,
                    }
                }
                if parent_id
                else None
            ),
            pending_writes=writes.result(),
        )

    async def _load_blobs(
        self,
        table: DynamoDBService,
        thread_id: str,
        checkpoint_ns: str,
        versions: ChannelVersions,
    ) -> dict[str, Any]:
        """The channel values a checkpoint references, by exact key.

        By key rather than by ``begins_with(sk, 'CHECKPOINT_BLOBS#')``: the prefix query would drag
        back every version ever written for the thread, and that set only grows.
        """
        if not versions:
            return {}

        keys = [
            row_key(thread_id, BlobRow.sort_key(checkpoint_ns, channel, version))
            for channel, version in versions.items()
        ]
        rows = await table.batch_get(keys, consistent_read=True)
        return {
            row['channel']: self._unpack(row['value_type'], row['value'])
            for row in rows
            if row['value_type'] != EMPTY_TYPE
        }

    async def _load_writes(
        self,
        table: DynamoDBService,
        thread_id: str,
        checkpoint_ns: str,
        checkpoint_id: str,
    ) -> list[PendingWrite]:
        rows = await table.query(
            Key('pk').eq(thread_key(thread_id))
            & Key('sk').begins_with(WriteRow.scan_prefix(checkpoint_ns, checkpoint_id)),
            consistent_read=True,
        )
        # Sorted here rather than by the sort key: idx can be negative, which a string sort gets
        # wrong. The order is the one Pregel replays the writes in, so it has to be deterministic.
        rows.sort(key=lambda item: (item['task_path'], item['task_id'], int(item['idx'])))
        return [(row['task_id'], row['channel'], self._unpack(row['value_type'], row['value'])) for row in rows]

    # --- writes ----------------------------------------------------------------------------------

    async def aput(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: ChannelVersions,
    ) -> RunnableConfig:
        thread_id: str = config['configurable']['thread_id']
        checkpoint_ns: str = config['configurable'].get('checkpoint_ns', '')
        table = self.table

        payload = checkpoint.copy()
        values: dict[str, Any] = payload.pop('channel_values')

        # Only the channels that changed this superstep. Everything else stays where it is, and the
        # new checkpoint's channel_versions goes on pointing at those older blob rows.
        actions: list[TransactAction] = [
            TransactPut(item=self._blob_row(thread_id, checkpoint_ns, channel, version, values))
            for channel, version in new_versions.items()
        ]

        checkpoint_type, checkpoint_value = self._pack(payload)
        # get_checkpoint_metadata folds config['metadata'] and config['configurable'] into what is
        # stored; skipping it loses run_id and every custom config key.
        metadata_type, metadata_value = self._pack(get_checkpoint_metadata(config, metadata))
        actions.append(
            TransactPut(
                item=CheckpointRow(
                    thread_id=thread_id,
                    checkpoint_ns=checkpoint_ns,
                    checkpoint_id=checkpoint['id'],
                    # The checkpoint being written *from* is the parent of the one being written.
                    parent_checkpoint_id=config['configurable'].get('checkpoint_id'),
                    checkpoint_type=checkpoint_type,
                    checkpoint_value=checkpoint_value,
                    metadata_type=metadata_type,
                    metadata_value=metadata_value,
                )
            )
        )

        # One transaction, so a checkpoint row can never reference a blob row whose write failed.
        await table.transact_write(*actions)
        return {
            'configurable': {
                'thread_id': thread_id,
                'checkpoint_ns': checkpoint_ns,
                'checkpoint_id': checkpoint['id'],
            }
        }

    def _blob_row(
        self,
        thread_id: str,
        checkpoint_ns: str,
        channel: str,
        version: Any,
        values: dict[str, Any],
    ) -> BlobRow:
        """One channel's value at one version. A channel listed in ``new_versions`` but absent from
        ``channel_values`` is stored as an explicit ``empty`` marker rather than skipped, so the
        read side can tell "no value here" from "row missing"."""
        value_type, value = self._pack(values[channel]) if channel in values else (EMPTY_TYPE, b'')
        return BlobRow(
            thread_id=thread_id,
            checkpoint_ns=checkpoint_ns,
            channel=channel,
            version=str(version),
            value_type=value_type,
            value=value,
        )

    async def aput_writes(
        self,
        config: RunnableConfig,
        writes: Sequence[tuple[str, Any]],
        task_id: str,
        task_path: str = '',
    ) -> None:
        thread_id: str = config['configurable']['thread_id']
        checkpoint_ns: str = config['configurable'].get('checkpoint_ns', '')
        checkpoint_id: str = config['configurable']['checkpoint_id']
        table = self.table

        rows: list[tuple[WriteRow, bool]] = []
        for position, (channel, value) in enumerate(writes):
            idx = WRITES_IDX_MAP.get(channel, position)
            value_type, value_bytes = self._pack(value)
            rows.append(
                (
                    WriteRow(
                        thread_id=thread_id,
                        checkpoint_ns=checkpoint_ns,
                        checkpoint_id=checkpoint_id,
                        task_id=task_id,
                        task_path=task_path,
                        idx=idx,
                        channel=channel,
                        value_type=value_type,
                        value=value_bytes,
                    ),
                    # Regular writes: whoever got there first wins, so a retried task cannot
                    # overwrite what its earlier attempt committed. Special writes (negative idx:
                    # error, interrupt, resume, scheduled) replace the earlier one.
                    idx >= 0,
                )
            )

        # Independent rows, so they go out together. Not a transaction: a failed condition here
        # means "skip this one", not "abandon the batch".
        try:
            async with asyncio.TaskGroup() as group:
                for row, first_wins in rows:
                    group.create_task(self._put_write(table, row, first_wins=first_wins))
        except* ClientError as failures:
            # A TaskGroup reports every failure as an ExceptionGroup, so an upstream
            # ``except ClientError`` — a LangGraph retry policy keyed on the botocore type among
            # them — would not match, and a transient throttle would surface as something opaque
            # rather than as the retryable error it is. Re-raise the first one as itself.
            #
            # Anything that is not a ClientError stays a group: that is a bug rather than a
            # storage condition, and flattening it would hide the others. Losing the sibling
            # ClientErrors is acceptable here — they are the same condition, reported per row, and
            # the whole superstep is about to be retried anyway.
            raise failures.exceptions[0]

        # Cancelling the siblings on the first failure is safe, and deliberately so: a retry calls
        # this again with the same task_id, the rows that landed are kept by the first-wins
        # condition above, and the rest are written then.

    @staticmethod
    async def _put_write(table: DynamoDBService, row: WriteRow, *, first_wins: bool) -> None:
        try:
            await table.put_item(row, condition=Attr('sk').not_exists() if first_wins else None)
        except ClientError as error:
            if error.response['Error']['Code'] != CONDITIONAL_CHECK_FAILED:
                raise
            # The write is already there. That is the de-duplication doing its job, not a failure.

    # --- the synchronous half ----------------------------------------------------------------------

    def get_tuple(self, config: RunnableConfig) -> CheckpointTuple | None:
        raise NotImplementedError(SYNC_NOT_SUPPORTED)

    def list(self, config: RunnableConfig | None, **kwargs: Any) -> Any:
        raise NotImplementedError(SYNC_NOT_SUPPORTED)

    def put(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: ChannelVersions,
    ) -> RunnableConfig:
        raise NotImplementedError(SYNC_NOT_SUPPORTED)

    def put_writes(
        self,
        config: RunnableConfig,
        writes: Sequence[tuple[str, Any]],
        task_id: str,
        task_path: str = '',
    ) -> None:
        raise NotImplementedError(SYNC_NOT_SUPPORTED)


@lru_cache
def get_checkpointer() -> DynamoDBCheckpointer:
    return DynamoDBCheckpointer(get_aioboto_session(), get_settings())
