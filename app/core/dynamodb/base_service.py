"""A thin, generic CRUD layer over a single DynamoDB table, on the low-level async client.

The low-level client speaks DynamoDB's wire format (``{'pk': {'S': 'u-1'}}``), so this service
translates at the boundary: callers pass and receive plain Python dicts (``{'pk': 'u-1'}``), and a
``TypeSerializer``/``TypeDeserializer`` pair converts to/from ``AttributeValue`` on the way in and
out. ``Key``/``Attr`` condition objects are rendered with ``ConditionExpressionBuilder`` (the
convenience the high-level resource gives you for free), so the query/scan API stays expressive.
"""

from typing import Any, TypeVar

from boto3.dynamodb.conditions import BuiltConditionExpression, ConditionBase, ConditionExpressionBuilder
from boto3.dynamodb.types import TypeDeserializer, TypeSerializer
from botocore.exceptions import ClientError
from pydantic import BaseModel
from types_aiobotocore_dynamodb import DynamoDBClient
from types_aiobotocore_dynamodb.type_defs import TransactWriteItemTypeDef

from app.core.dynamodb.indexes import Index
from app.core.dynamodb.schemas import (
    TransactAction,
    TransactConditionCheck,
    TransactDelete,
    TransactPut,
    TransactUpdate,
)

Item = TypeVar('Item', bound=BaseModel)

_SERIALIZER = TypeSerializer()
_DESERIALIZER = TypeDeserializer()

# How many keys one ``BatchGetItem`` accepts; ``batch_get`` chunks anything longer.
BATCH_GET_LIMIT = 100

# Which member of a ``TransactWriteItems`` entry each action becomes. Keyed by type rather than
# branched on, so a fourth action (``Update``) is one line here rather than another elif.
_TRANSACT_MEMBER: dict[type, str] = {
    TransactPut: 'Put',
    TransactDelete: 'Delete',
    TransactConditionCheck: 'ConditionCheck',
    TransactUpdate: 'Update',
}


class DynamoDBService:
    """Reads from and writes to one DynamoDB table via the low-level client. The client is injected
    (DI) rather than built inside, so callers control its lifecycle and tests can pass a fake."""

    def __init__(self, client: DynamoDBClient, table_name: str):
        self._client = client
        self._table_name = table_name

    async def put_item(self, item: Item, condition: ConditionBase | None = None) -> None:
        """Create or overwrite an item. Pass ``condition`` (e.g. ``Attr('pk').not_exists()``) to
        make the write conditional — it raises ``ConditionalCheckFailedException`` if it fails.

        Empty attributes are dropped rather than written, as ``db/load_dynamodb._clean`` drops them
        when seeding: absent and NULL read back the same through ``.get()``, and for a GSI key they
        do not — DynamoDB has no NULL type for one, so an open slot carrying ``gsi2pk: None`` is
        refused outright with ``Invalid attribute value type``.
        """
        kwargs: dict[str, Any] = {
            'TableName': self._table_name,
            'Item': self._serialize(item.model_dump(exclude_none=True)),
            'ReturnConsumedCapacity': 'TOTAL',
        }
        if condition is not None:
            expr, names, values = self._build_condition(condition)
            kwargs['ConditionExpression'] = expr
            self._set_expressions(kwargs, names, values)
        await self._client.put_item(**kwargs)

    async def get_item(self, key: dict[str, Any], consistent_read: bool = False) -> dict[str, Any] | None:
        """The item matching ``key`` (its primary key, e.g. ``TableItem.key(id)``), or ``None`` if it
        doesn't exist. Set ``consistent_read`` for a strongly-consistent read instead of the default
        eventual one."""
        response = await self._client.get_item(
            TableName=self._table_name,
            Key=self._serialize(key),
            ConsistentRead=consistent_read,
            ReturnConsumedCapacity='TOTAL',
        )
        item = response.get('Item')
        return self._deserialize(item) if item is not None else None

    async def batch_get(self, keys: list[dict[str, Any]], consistent_read: bool = False) -> list[dict[str, Any]]:
        """The items at ``keys``, fetched in one round trip per 100 keys.

        Unlike ``get_item`` this says nothing about *which* key each item came from and returns no
        placeholder for a key that matched nothing — a caller that needs the correspondence reads
        it back off the items. Order is not preserved either: DynamoDB is free to answer in any
        order, and to answer partially, which is what the ``UnprocessedKeys`` loop is for."""
        items: list[dict[str, Any]] = []
        for start in range(0, len(keys), BATCH_GET_LIMIT):
            chunk = keys[start : start + BATCH_GET_LIMIT]
            request: dict[str, Any] = {
                self._table_name: {
                    'Keys': [self._serialize(key) for key in chunk],
                    'ConsistentRead': consistent_read,
                }
            }
            while request:
                response = await self._client.batch_get_item(RequestItems=request)
                found = response.get('Responses', {}).get(self._table_name, [])
                items.extend(self._deserialize(raw) for raw in found)
                request = response.get('UnprocessedKeys') or {}
        return items

    async def update_item(
        self,
        key: dict[str, Any],
        update_expression: str,
        expression_values: dict[str, Any],
        expression_names: dict[str, str] | None = None,
        condition: ConditionBase | None = None,
    ) -> dict:
        """Apply an update expression (e.g. ``'SET #n = :name'``) to the item at ``key`` and return
        the new attributes. ``expression_names`` aliases reserved words; ``condition`` guards the
        write. Creates the item if it doesn't exist (DynamoDB upsert semantics).

        ``condition``'s generated placeholders (``#n0`` / ``:v0``) are merged in — keep your own
        aliases away from that pattern to avoid a clash."""
        names = dict(expression_names or {})
        values = self._serialize(expression_values)
        kwargs: dict[str, Any] = {
            'TableName': self._table_name,
            'Key': self._serialize(key),
            'UpdateExpression': update_expression,
            'ReturnValues': 'ALL_NEW',
        }
        if condition is not None:
            expr, c_names, c_values = self._build_condition(condition)
            kwargs['ConditionExpression'] = expr
            names.update(c_names)
            values.update(c_values)
        self._set_expressions(kwargs, names, values)
        response = await self._client.update_item(**kwargs)
        return self._deserialize(response.get('Attributes', {}))

    async def delete_item(self, key: dict[str, Any], condition: ConditionBase | None = None) -> None:
        """Delete the item at ``key`` (a no-op if it's absent). Pass ``condition`` to delete only
        when it holds."""
        kwargs: dict[str, Any] = {'TableName': self._table_name, 'Key': self._serialize(key)}
        if condition is not None:
            expr, names, values = self._build_condition(condition)
            kwargs['ConditionExpression'] = expr
            self._set_expressions(kwargs, names, values)
        await self._client.delete_item(**kwargs)

    async def transact_write(self, *actions: TransactAction) -> None:
        """Apply several writes as one all-or-nothing ``TransactWriteItems``.

        A ``TransactConditionCheck`` writes nothing — it only asserts something about a row the
        transaction leaves alone (that a parent still exists, say). It still counts as one of the
        100 actions, and its position still shows up in ``get_failed_condition_indexes``.

        The transaction is **not confined to this service's table**: an action carrying a ``table``
        of its own is sent against that one instead, which is how a repository maintains a counter
        on a row belonging to another entity (``ProgramsRepository`` and the faculty's
        ``dependants``) without a second service or a second, non-atomic write."""
        transact_items: list[TransactWriteItemTypeDef] = []
        for action in actions:
            entry: dict[str, Any] = {'TableName': action.table or self._table_name}
            # One name/value namespace per entry: an update's own placeholders and the condition's
            # generated ones (#n0/:v0) end up in the same two maps, so they are collected here
            # rather than attached twice.
            names: dict[str, str] = {}
            values: dict[str, Any] = {}
            if isinstance(action, TransactPut):
                entry['Item'] = self._serialize(action.item.model_dump())
            else:
                entry['Key'] = self._serialize(action.key)
            if isinstance(action, TransactUpdate):
                entry['UpdateExpression'] = action.update_expression
                names.update(action.expression_names or {})
                values.update(self._serialize(action.expression_values))
            if action.condition is not None:
                expr, c_names, c_values = self._build_condition(action.condition)
                entry['ConditionExpression'] = expr
                names.update(c_names)
                values.update(c_values)
            self._set_expressions(entry, names, values)
            transact_items.append({_TRANSACT_MEMBER[type(action)]: entry})  # type: ignore[misc]

        await self._client.transact_write_items(TransactItems=transact_items)

    @staticmethod
    def get_failed_condition_indexes(error: ClientError) -> set[int]:
        """Which actions of a cancelled transaction failed their condition, by position in the call."""
        reasons = error.response.get('CancellationReasons', [])
        return {i for i, reason in enumerate(reasons) if reason.get('Code') == 'ConditionalCheckFailed'}

    async def query(
        self,
        key_condition: ConditionBase,
        filter_expression: ConditionBase | None = None,
        index_name: Index | None = None,
        limit: int | None = None,
        ascending: bool = True,
        consistent_read: bool = False,
    ) -> list[dict[str, Any]]:
        """Items matching a ``key_condition`` (``Key('pk').eq(...)``), transparently paginating so
        the full result set comes back. ``filter_expression`` post-filters, ``limit`` caps the total
        returned.

        ``index_name`` picks what is queried: an ``Index`` member targets that GSI, and ``None`` —
        the default — queries the base table, which is what a key condition on ``pk``/``sk`` needs
        (e.g. ``SlotsRepository.list_open_slots_on_date``).

        ``ascending=False`` walks the sort key backwards, so ``limit=1`` returns the *last* item of
        the partition rather than the first — how the checkpointer finds a thread's newest
        checkpoint without reading the ones before it.

        ``consistent_read`` reads the latest committed data rather than whatever a replica has, at
        twice the RCU. **DynamoDB rejects it on a GSI**, so it cannot be combined with
        ``index_name`` — a global secondary index is only ever eventually consistent."""
        if consistent_read and index_name is not None:
            raise ValueError(f'consistent_read is not available on a GSI ({index_name}); query the base table')
        # One builder for both expressions so their placeholders (#n0/:v0, #n1/:v1) never collide.
        builder = ConditionExpressionBuilder()
        names: dict[str, str] = {}
        values: dict[str, Any] = {}

        key: BuiltConditionExpression = builder.build_expression(key_condition, is_key_condition=True)
        names.update(key.attribute_name_placeholders)
        values.update(key.attribute_value_placeholders)
        kwargs: dict[str, Any] = {
            'TableName': self._table_name,
            'KeyConditionExpression': key.condition_expression,
            'ScanIndexForward': ascending,
            'ConsistentRead': consistent_read,
        }

        if filter_expression is not None:
            flt: BuiltConditionExpression = builder.build_expression(filter_expression, is_key_condition=False)
            kwargs['FilterExpression'] = flt.condition_expression
            names.update(flt.attribute_name_placeholders)
            values.update(flt.attribute_value_placeholders)

        if index_name is not None:
            kwargs['IndexName'] = index_name

        if limit is not None:
            # Caps what DynamoDB *reads* per page, not merely what we keep, so an existence check
            # (limit=1) costs one row instead of a full 1 MB page. The cap applies before any
            # FilterExpression, hence a page can come back short — which is why ``_paginate`` keeps
            # following LastEvaluatedKey rather than treating one page as the whole answer.
            kwargs['Limit'] = limit

        self._set_expressions(kwargs, names, {ph: _SERIALIZER.serialize(v) for ph, v in values.items()})
        return await self._paginate(self._client.query, kwargs, limit)

    async def scan(
        self,
        filter_expression: ConditionBase | None = None,
        limit: int | None = None,
        consistent_read: bool = False,
    ) -> list[dict[str, Any]]:
        """Every item in the table (optionally ``filter_expression``-filtered), paginating to
        completion. A full scan is expensive — prefer ``query`` on a key/index where possible.

        ``consistent_read`` as in ``query``: latest committed data, twice the RCU."""
        kwargs: dict[str, Any] = {'TableName': self._table_name, 'ConsistentRead': consistent_read}
        if filter_expression is not None:
            expr, names, values = self._build_condition(filter_expression)
            kwargs['FilterExpression'] = expr
            self._set_expressions(kwargs, names, values)

        return await self._paginate(self._client.scan, kwargs, limit)

    async def _paginate(self, operation: Any, kwargs: dict[str, Any], limit: int | None) -> list[dict[str, Any]]:
        """Follow ``LastEvaluatedKey`` until the table is exhausted (or ``limit`` items collected),
        deserialising each page. ``operation`` is the client's ``query`` or ``scan``."""
        items: list[dict[str, Any]] = []
        while True:
            response = await operation(**kwargs)
            items.extend(self._deserialize(raw) for raw in response.get('Items', []))
            last_key = response.get('LastEvaluatedKey')
            if last_key is None or (limit is not None and len(items) >= limit):
                break

            kwargs['ExclusiveStartKey'] = last_key
        return items[:limit] if limit is not None else items

    @staticmethod
    def _serialize(item: dict[str, Any]) -> dict[str, Any]:
        return {key: _SERIALIZER.serialize(value) for key, value in item.items()}

    @staticmethod
    def _deserialize(item: dict[str, Any]) -> dict:
        return {key: _DESERIALIZER.deserialize(value) for key, value in item.items()}

    @staticmethod
    def _build_condition(
        condition: ConditionBase, *, is_key: bool = False
    ) -> tuple[str, dict[str, str], dict[str, Any]]:
        """Render a ``Key``/``Attr`` condition into its expression string, name placeholders and
        serialised value placeholders."""
        built: BuiltConditionExpression = ConditionExpressionBuilder().build_expression(
            condition, is_key_condition=is_key
        )
        values = {ph: _SERIALIZER.serialize(v) for ph, v in built.attribute_value_placeholders.items()}
        return built.condition_expression, built.attribute_name_placeholders, values

    @staticmethod
    def _set_expressions(kwargs: dict[str, Any], names: dict[str, str], values: dict[str, Any]) -> None:
        """Attach name/value placeholder maps to a request — but only when non-empty. DynamoDB rejects
        an empty ``ExpressionAttributeValues`` (e.g. a value-less ``Attr(...).not_exists()`` condition)."""
        if names:
            kwargs['ExpressionAttributeNames'] = names
        if values:
            kwargs['ExpressionAttributeValues'] = values
