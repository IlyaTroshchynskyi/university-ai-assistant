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
from app.core.dynamodb.schemas import TransactAction, TransactPut

Item = TypeVar('Item', bound=BaseModel)

_SERIALIZER = TypeSerializer()
_DESERIALIZER = TypeDeserializer()


class DynamoDBService:
    """Reads from and writes to one DynamoDB table via the low-level client. The client is injected
    (DI) rather than built inside, so callers control its lifecycle and tests can pass a fake."""

    def __init__(self, client: DynamoDBClient, table_name: str):
        self._client = client
        self._table_name = table_name

    async def put_item(self, item: Item, condition: ConditionBase | None = None) -> None:
        """Create or overwrite an item. Pass ``condition`` (e.g. ``Attr('pk').not_exists()``) to
        make the write conditional — it raises ``ConditionalCheckFailedException`` if it fails."""
        kwargs: dict[str, Any] = {
            'TableName': self._table_name,
            'Item': self._serialize(item.model_dump()),
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
        """Apply several writes as one all-or-nothing ``TransactWriteItems``."""
        transact_items: list[TransactWriteItemTypeDef] = []
        for action in actions:
            entry: dict[str, Any] = {'TableName': self._table_name}
            if isinstance(action, TransactPut):
                entry['Item'] = self._serialize(action.item.model_dump())
            else:
                entry['Key'] = self._serialize(action.key)
            if action.condition is not None:
                expr, names, values = self._build_condition(action.condition)
                entry['ConditionExpression'] = expr
                self._set_expressions(entry, names, values)
            transact_items.append({'Put': entry} if isinstance(action, TransactPut) else {'Delete': entry})

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
    ) -> list[dict[str, Any]]:
        """Items matching a ``key_condition`` (``Key('pk').eq(...)``), transparently paginating so
        the full result set comes back. ``filter_expression`` post-filters, ``limit`` caps the total
        returned.

        ``index_name`` picks what is queried: an ``Index`` member targets that GSI, and ``None`` —
        the default — queries the base table, which is what a key condition on ``pk``/``sk`` needs
        (e.g. ``SlotsRepository.list_open_slots_on_date``)."""
        # One builder for both expressions so their placeholders (#n0/:v0, #n1/:v1) never collide.
        builder = ConditionExpressionBuilder()
        names: dict[str, str] = {}
        values: dict[str, Any] = {}

        key: BuiltConditionExpression = builder.build_expression(key_condition, is_key_condition=True)
        names.update(key.attribute_name_placeholders)
        values.update(key.attribute_value_placeholders)
        kwargs: dict[str, Any] = {'TableName': self._table_name, 'KeyConditionExpression': key.condition_expression}

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
        self, filter_expression: ConditionBase | None = None, limit: int | None = None
    ) -> list[dict[str, Any]]:
        """Every item in the table (optionally ``filter_expression``-filtered), paginating to
        completion. A full scan is expensive — prefer ``query`` on a key/index where possible."""
        kwargs: dict[str, Any] = {'TableName': self._table_name}
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
