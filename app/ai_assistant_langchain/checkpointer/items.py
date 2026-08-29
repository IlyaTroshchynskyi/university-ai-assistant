"""The three row types of the ``agent_checkpoints`` table, and the keys that place them.

The prefixes are ``PostgresSaver``'s three table names — ``checkpoints``, ``checkpoint_blobs``,
``checkpoint_writes`` — upper-cased, the way this project already namespaces a single-table sort
key (``SCHED#``, ``TYPE#ROOM``). One row type here is one Postgres table there.
"""

from typing import ClassVar

from pydantic import BaseModel, computed_field

from app.core.dynamodb.indexes import thread_key

EMPTY_TYPE = 'empty'


def row_key(thread_id: str, sort_key: str) -> dict[str, str]:
    """The primary key of a single row — what ``get_item``/``batch_get`` need."""
    return {'pk': thread_key(thread_id), 'sk': sort_key}


class Row(BaseModel):
    """Shared partition key. ``prefix`` is the sort-key namespace of the subclass."""

    prefix: ClassVar[str]
    thread_id: str
    checkpoint_ns: str

    @computed_field  # type: ignore[prop-decorator]
    @property
    def pk(self) -> str:
        return thread_key(self.thread_id)


class CheckpointRow(Row):
    """One checkpoint, minus its channel values — those are ``BlobRow``s.

    ``checkpoint_id`` is a uuid6: monotonically increasing and lexicographically sortable, so
    within one namespace this sort key is chronological and the newest checkpoint is the last.
    """

    prefix: ClassVar[str] = 'CHECKPOINTS'
    checkpoint_id: str
    parent_checkpoint_id: str | None
    checkpoint_type: str
    checkpoint_value: bytes
    metadata_type: str
    metadata_value: bytes

    @classmethod
    def sort_key(cls, checkpoint_ns: str, checkpoint_id: str) -> str:
        return f'{cls.prefix}#{checkpoint_ns}#{checkpoint_id}'

    @classmethod
    def scan_prefix(cls, checkpoint_ns: str | None) -> str:
        return f'{cls.prefix}#' if checkpoint_ns is None else f'{cls.prefix}#{checkpoint_ns}#'

    @computed_field  # type: ignore[prop-decorator]
    @property
    def sk(self) -> str:
        return self.sort_key(self.checkpoint_ns, self.checkpoint_id)


class BlobRow(Row):
    """One channel's value at one version."""

    prefix: ClassVar[str] = 'CHECKPOINT_BLOBS'

    channel: str
    version: str
    value_type: str
    value: bytes

    @classmethod
    def sort_key(cls, checkpoint_ns: str, channel: str, version: str | int | float) -> str:
        return f'{cls.prefix}#{checkpoint_ns}#{channel}#{version}'

    @computed_field  # type: ignore[prop-decorator]
    @property
    def sk(self) -> str:
        return self.sort_key(self.checkpoint_ns, self.channel, self.version)


class WriteRow(Row):
    """One pending write of one task."""

    prefix: ClassVar[str] = 'CHECKPOINT_WRITES'

    checkpoint_id: str
    task_id: str
    task_path: str
    idx: int
    channel: str
    value_type: str
    value: bytes

    @classmethod
    def sort_key(cls, checkpoint_ns: str, checkpoint_id: str, task_id: str, idx: int) -> str:
        return f'{cls.prefix}#{checkpoint_ns}#{checkpoint_id}#{task_id}#{idx}'

    @classmethod
    def scan_prefix(cls, checkpoint_ns: str, checkpoint_id: str) -> str:
        """Every write belonging to one checkpoint."""
        return f'{cls.prefix}#{checkpoint_ns}#{checkpoint_id}#'

    @computed_field  # type: ignore[prop-decorator]
    @property
    def sk(self) -> str:
        return self.sort_key(self.checkpoint_ns, self.checkpoint_id, self.task_id, self.idx)
