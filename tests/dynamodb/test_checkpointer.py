from datetime import datetime, timezone
import random
from typing import Any, AsyncGenerator
from unittest.mock import patch
from uuid import uuid4
import zlib

from botocore.exceptions import ClientError
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import ChannelVersions, Checkpoint, CheckpointMetadata, LATEST_VERSION
from langgraph.checkpoint.base.id import uuid6
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.checkpoint.serde.types import ERROR
import pytest

from app.ai_assistant_langchain.checkpointer.saver import DynamoDBCheckpointer
from app.core.dynamodb.client import get_aioboto_session
from app.settings import get_settings
from tests.db_utils import TableSpecs

ITEM_LIMIT = 400 * 1024


def new_checkpoint_id() -> str:
    """A uuid6, as LangGraph generates them: monotonic, so ids sort chronologically."""
    return str(uuid6(clock_seq=-2))


def make_checkpoint(channel_values: dict[str, Any], versions: ChannelVersions) -> Checkpoint:
    return Checkpoint(
        v=LATEST_VERSION,
        id=new_checkpoint_id(),
        ts=datetime.now(timezone.utc).isoformat(),
        channel_values=channel_values,
        channel_versions=versions,
        versions_seen={},
        updated_channels=None,
    )


def config_for(thread_id: str, checkpoint_id: str | None = None) -> RunnableConfig:
    configurable: dict[str, Any] = {'thread_id': thread_id, 'checkpoint_ns': ''}
    if checkpoint_id is not None:
        configurable['checkpoint_id'] = checkpoint_id
    return {'configurable': configurable}


def metadata(step: int) -> CheckpointMetadata:
    return CheckpointMetadata(source='loop', step=step, parents={})


class TestDynamoDBCheckpointer:
    @pytest.fixture(autouse=True)
    async def _a_provide_saver(self, tables: TableSpecs) -> AsyncGenerator[None, None]:
        async with DynamoDBCheckpointer(get_aioboto_session(), get_settings()).opened() as saver:
            self.saver = saver
            self.thread_id = str(uuid4())
            yield

    async def _put(
        self,
        channel_values: dict[str, Any],
        versions: ChannelVersions,
        new_versions: ChannelVersions | None = None,
        parent_id: str | None = None,
        step: int = 1,
    ) -> str:
        """Write one checkpoint and answer with its id. ``new_versions`` defaults to *everything*
        changing; pass a narrower one to model a channel that stayed put."""
        checkpoint = make_checkpoint(channel_values, versions)
        await self.saver.aput(
            config_for(self.thread_id, parent_id),
            checkpoint,
            metadata(step),
            new_versions if new_versions is not None else versions,
        )
        return checkpoint['id']

    async def test_checkpoint_round_trips(self) -> None:
        """The values, the metadata and the id all survive the trip through DynamoDB."""
        checkpoint_id = await self._put({'messages': [HumanMessage(content='Hello')]}, {'messages': 1})

        loaded = await self.saver.aget_tuple(config_for(self.thread_id))

        assert loaded is not None
        assert loaded.checkpoint['id'] == checkpoint_id
        assert [message.content for message in loaded.checkpoint['channel_values']['messages']] == ['Hello']
        assert loaded.metadata['step'] == 1
        assert loaded.parent_config is None

    async def test_unknown_thread_has_no_checkpoint(self) -> None:
        assert await self.saver.aget_tuple(config_for(str(uuid4()))) is None

    async def test_latest_checkpoint_wins_and_carries_its_parent(self) -> None:
        """Without a checkpoint_id in the config, the newest one comes back — and points at the one
        it was written from."""
        first = await self._put({'messages': [HumanMessage(content='One')]}, {'messages': 1}, step=1)
        second = await self._put(
            {'messages': [HumanMessage(content='One'), AIMessage(content='Two')]},
            {'messages': 2},
            parent_id=first,
            step=2,
        )

        loaded = await self.saver.aget_tuple(config_for(self.thread_id))

        assert loaded is not None
        assert loaded.checkpoint['id'] == second
        assert loaded.parent_config['configurable']['checkpoint_id'] == first

    async def test_checkpoint_can_be_fetched_by_id(self) -> None:
        """Naming a checkpoint_id reads that checkpoint, not the newest — this is what time travel
        is built on."""
        first = await self._put({'messages': [HumanMessage(content='One')]}, {'messages': 1}, step=1)
        await self._put({'messages': [AIMessage(content='Two')]}, {'messages': 2}, parent_id=first, step=2)

        loaded = await self.saver.aget_tuple(config_for(self.thread_id, first))

        assert loaded is not None
        assert loaded.checkpoint['id'] == first
        assert [message.content for message in loaded.checkpoint['channel_values']['messages']] == ['One']

    async def test_unchanged_channel_is_read_from_the_earlier_blob(self) -> None:
        """The point of splitting blobs out of the checkpoint.

        The second write reports only ``messages`` as changed, so no blob row is written for
        ``call_count`` — yet the loaded checkpoint still has its value, resolved through the version
        the first write left behind.
        """
        first = await self._put(
            {'messages': [HumanMessage(content='One')], 'call_count': 7},
            {'messages': 1, 'call_count': 1},
            step=1,
        )
        await self._put(
            {'messages': [AIMessage(content='Two')]},
            {'messages': 2, 'call_count': 1},
            new_versions={'messages': 2},
            parent_id=first,
            step=2,
        )

        loaded = await self.saver.aget_tuple(config_for(self.thread_id))

        assert loaded is not None
        assert loaded.checkpoint['channel_values']['call_count'] == 7

    async def test_sibling_branches_get_distinct_channel_versions(self) -> None:
        """Two checkpoints forked from one parent must not share a version for the same channel.

        A blob row is keyed by ``(channel, version)`` and written unconditionally, so equal versions
        mean one key and a silent overwrite. The base class's ``current + 1`` produces exactly that;
        the override adds the random tail that keeps siblings apart.
        """
        root = self.saver.get_next_version(None, None)

        assert self.saver.get_next_version(root, None) != self.saver.get_next_version(root, None)

    async def test_versions_still_sort_by_age(self) -> None:
        """The random tail must not cost the ordering the counter provides — ``alist`` and the
        newest-checkpoint query both lean on versions being comparable."""
        first = self.saver.get_next_version(None, None)
        second = self.saver.get_next_version(first, None)

        assert first < second < self.saver.get_next_version(second, None)

    async def test_a_forked_branch_keeps_its_own_channel_value(self) -> None:
        """The bug end to end: write two children of one parent, then read the first back.

        With colliding versions the second write lands on the first's blob row, and this returns
        ``'Branch B'``.
        """
        parent_version = self.saver.get_next_version(None, None)
        parent = await self._put({'messages': [HumanMessage(content='Root')]}, {'messages': parent_version})

        branches = {}
        for label in ('Branch A', 'Branch B'):
            version = self.saver.get_next_version(parent_version, None)
            branches[label] = await self._put(
                {'messages': [AIMessage(content=label)]},
                {'messages': version},
                parent_id=parent,
                step=2,
            )

        loaded = await self.saver.aget_tuple(config_for(self.thread_id, branches['Branch A']))

        assert loaded is not None
        assert [message.content for message in loaded.checkpoint['channel_values']['messages']] == ['Branch A']

    async def test_versions_written_as_plain_integers_still_resolve(self) -> None:
        """Threads stored before the version format changed keep working: the override reads an int
        as the counter it was, and ``_load_blobs`` still finds a blob keyed by ``1``."""
        await self._put({'messages': [HumanMessage(content='Old thread')]}, {'messages': 1})

        loaded = await self.saver.aget_tuple(config_for(self.thread_id))

        assert loaded is not None
        assert [message.content for message in loaded.checkpoint['channel_values']['messages']] == ['Old thread']
        assert self.saver.get_next_version(1, None).startswith('0' * 31 + '2.')

    async def test_channel_without_a_value_comes_back_missing_not_broken(self) -> None:
        """A channel in ``new_versions`` but not in ``channel_values`` is stored as an explicit
        empty marker, and is simply absent on the way back."""
        await self._put({'messages': [HumanMessage(content='One')]}, {'messages': 1, 'pending': 1})

        loaded = await self.saver.aget_tuple(config_for(self.thread_id))

        assert loaded is not None
        assert 'pending' not in loaded.checkpoint['channel_values']
        assert 'messages' in loaded.checkpoint['channel_values']

    async def test_threads_do_not_see_each_other(self) -> None:
        await self._put({'messages': [HumanMessage(content='Mine')]}, {'messages': 1})
        async with DynamoDBCheckpointer(get_aioboto_session(), get_settings()).opened() as other:
            checkpoint = make_checkpoint({'messages': [HumanMessage(content='Theirs')]}, {'messages': 1})
            await other.aput(config_for(str(uuid4())), checkpoint, metadata(1), {'messages': 1})

        loaded = await self.saver.aget_tuple(config_for(self.thread_id))
        assert loaded is not None
        assert [message.content for message in loaded.checkpoint['channel_values']['messages']] == ['Mine']


class TestPendingWrites:
    @pytest.fixture(autouse=True)
    async def _a_provide_saver(self, tables: TableSpecs) -> AsyncGenerator[None, None]:
        async with DynamoDBCheckpointer(get_aioboto_session(), get_settings()).opened() as saver:
            self.saver = saver
            self.thread_id = str(uuid4())
            checkpoint = make_checkpoint({'messages': [HumanMessage(content='Hello')]}, {'messages': 1})
            await saver.aput(config_for(self.thread_id), checkpoint, metadata(1), {'messages': 1})
            self.config = config_for(self.thread_id, checkpoint['id'])
            yield

    async def _pending(self) -> list[tuple[str, str, Any]]:
        loaded = await self.saver.aget_tuple(config_for(self.thread_id))
        assert loaded is not None
        return list(loaded.pending_writes or [])

    async def test_writes_are_stored_with_their_task(self) -> None:
        await self.saver.aput_writes(self.config, [('messages', 'alpha'), ('other', 'beta')], 'task-1')

        assert await self._pending() == [('task-1', 'messages', 'alpha'), ('task-1', 'other', 'beta')]

    async def test_a_repeated_write_keeps_the_first_value(self) -> None:
        """A retried task must not overwrite what its earlier attempt committed — Pregel depends on
        it, and here it is a conditional put rather than a dictionary check."""
        await self.saver.aput_writes(self.config, [('messages', 'first')], 'task-1')
        await self.saver.aput_writes(self.config, [('messages', 'second')], 'task-1')

        assert await self._pending() == [('task-1', 'messages', 'first')]

    async def test_two_tasks_writing_the_same_channel_both_survive(self) -> None:
        await self.saver.aput_writes(self.config, [('messages', 'from-one')], 'task-1')
        await self.saver.aput_writes(self.config, [('messages', 'from-two')], 'task-2')

        assert await self._pending() == [
            ('task-1', 'messages', 'from-one'),
            ('task-2', 'messages', 'from-two'),
        ]

    async def test_an_error_write_replaces_the_earlier_one(self) -> None:
        """``__error__`` maps to a negative index, and those are last-write-wins — the opposite of
        the rule above."""
        await self.saver.aput_writes(self.config, [(ERROR, 'first failure')], 'task-1')
        await self.saver.aput_writes(self.config, [(ERROR, 'second failure')], 'task-1')

        assert await self._pending() == [('task-1', ERROR, 'second failure')]

    async def test_a_storage_failure_surfaces_as_itself_not_as_a_group(self) -> None:
        """``aput_writes`` fans its rows out through a ``TaskGroup``, and a TaskGroup reports every
        failure as an ``ExceptionGroup``. An upstream ``except ClientError`` — a LangGraph retry
        policy keyed on the botocore type among them — has to keep matching, so the group is
        unwrapped before it leaves.
        """
        throttled = ClientError({'Error': {'Code': 'ProvisionedThroughputExceededException'}}, 'PutItem')

        with patch.object(self.saver.table, 'put_item', side_effect=throttled):
            with pytest.raises(ClientError) as raised:
                await self.saver.aput_writes(self.config, [('messages', 'alpha')], 'task-1')

        assert raised.value.response['Error']['Code'] == 'ProvisionedThroughputExceededException'

    async def test_a_conditional_failure_is_still_swallowed(self) -> None:
        """The unwrapping must not turn de-duplication into an error: a conditional check failing is
        the first-wins rule working, and never reaches the caller."""
        already_there = ClientError({'Error': {'Code': 'ConditionalCheckFailedException'}}, 'PutItem')

        with patch.object(self.saver.table, 'put_item', side_effect=already_there):
            await self.saver.aput_writes(self.config, [('messages', 'alpha')], 'task-1')

    async def test_writes_belong_to_their_own_checkpoint(self) -> None:
        """A later checkpoint starts with a clean slate rather than inheriting pending writes."""
        await self.saver.aput_writes(self.config, [('messages', 'alpha')], 'task-1')
        later = make_checkpoint({'messages': [AIMessage(content='Later')]}, {'messages': 2})
        await self.saver.aput(
            config_for(self.thread_id, self.config['configurable']['checkpoint_id']),
            later,
            metadata(2),
            {'messages': 2},
        )

        assert await self._pending() == []


class TestListingCheckpoints:
    @pytest.fixture(autouse=True)
    async def _a_provide_history(self, tables: TableSpecs) -> AsyncGenerator[None, None]:
        async with DynamoDBCheckpointer(get_aioboto_session(), get_settings()).opened() as saver:
            self.saver = saver
            self.thread_id = str(uuid4())
            self.ids: list[str] = []
            parent: str | None = None
            for step in range(4):
                checkpoint = make_checkpoint(
                    {'messages': [HumanMessage(content=f'Step {step}')]}, {'messages': step + 1}
                )
                await saver.aput(config_for(self.thread_id, parent), checkpoint, metadata(step), {'messages': step + 1})
                self.ids.append(checkpoint['id'])
                parent = checkpoint['id']
            yield

    async def _listed(self, **kwargs: Any) -> list[str]:
        return [tuple_.checkpoint['id'] async for tuple_ in self.saver.alist(config_for(self.thread_id), **kwargs)]

    async def test_newest_first(self) -> None:
        assert await self._listed() == list(reversed(self.ids))

    async def test_limit_caps_the_result(self) -> None:
        assert await self._listed(limit=2) == list(reversed(self.ids))[:2]

    async def test_before_excludes_the_named_checkpoint_and_everything_after(self) -> None:
        assert await self._listed(before=config_for(self.thread_id, self.ids[2])) == [self.ids[1], self.ids[0]]

    async def test_metadata_filter_applies(self) -> None:
        assert await self._listed(filter={'step': 2}) == [self.ids[2]]

    async def test_listing_carries_the_channel_values(self) -> None:
        """Listing is not a shortcut that skips the blobs — a time-travel consumer needs the state,
        not just the ids."""
        newest = [tuple_ async for tuple_ in self.saver.alist(config_for(self.thread_id), limit=1)][0]

        assert [message.content for message in newest.checkpoint['channel_values']['messages']] == ['Step 3']


class TestLargePayloads:
    @pytest.fixture(autouse=True)
    async def _a_provide_saver(self, tables: TableSpecs) -> AsyncGenerator[None, None]:
        async with DynamoDBCheckpointer(get_aioboto_session(), get_settings()).opened() as saver:
            self.saver = saver
            self.thread_id = str(uuid4())
            yield

    async def test_a_history_too_big_to_store_raw_round_trips(self) -> None:
        """The test that proves compression is on the write path rather than merely configured.

        The message list is built from non-repeating text — repetitive filler would compress
        absurdly well and prove nothing — and is checked to exceed the 400 KB item limit before
        compression, so the write can only succeed because it is compressed.
        """
        random.seed(7)
        vocabulary = [f'{prefix}{index}' for prefix in ('faculty', 'tuition', 'seminar') for index in range(4000)]
        messages = [HumanMessage(content=' '.join(random.choice(vocabulary) for _ in range(400))) for _ in range(160)]

        raw = JsonPlusSerializer().dumps_typed(messages)[1]
        assert len(raw) > ITEM_LIMIT, f'fixture too small to prove anything: {len(raw)} B'
        assert len(zlib.compress(raw, 6)) < ITEM_LIMIT

        await self.saver.aput(
            config_for(self.thread_id),
            make_checkpoint({'messages': messages}, {'messages': 1}),
            metadata(1),
            {'messages': 1},
        )
        loaded = await self.saver.aget_tuple(config_for(self.thread_id))

        assert loaded is not None
        restored = loaded.checkpoint['channel_values']['messages']
        assert len(restored) == len(messages)
        assert restored[0].content == messages[0].content
        assert restored[-1].content == messages[-1].content
