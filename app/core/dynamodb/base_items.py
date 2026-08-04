"""The single-table key layout, as a model mixin.

Every entity shares one table, so a primary key is namespaced by entity type rather than being the
bare id: ``pk='ROOM#{id}'``, ``sk='#META'`` — the layout ``db/load_dynamodb.py`` seeds and the
repositories query. ``TableItem`` derives that key from the id instead of leaving it to each call
site, so a write can't reach DynamoDB with a missing or hand-rolled key (which is what
``ValidationException: One of the required keys was not given a value`` means).

The key attributes are ``@computed_field``s: they belong to ``model_dump()`` — and so to the item
``DynamoDBService.put_item`` sends — but are read-only, so nothing outside can set them. ``id``
stays an ordinary field: generated on create, accepted back on read.
"""

from typing import ClassVar
from uuid import uuid4

from pydantic import BaseModel, computed_field, Field


class TableItem(BaseModel):
    """An entity's single-table keys, derived from its ``id``."""

    # The pk prefix ('ROOM', 'PROF', …). A ClassVar, so it labels the subclass without becoming a
    # field — it describes the item rather than being written to the table alongside it.
    entity: ClassVar[str]
    # Sort key of the row carrying the entity's own attributes, as opposed to any related rows
    # filed under the same partition (a schedule's SCHED#… entries, say).
    meta_sk: ClassVar[str] = '#META'

    id: str = Field(default_factory=lambda: str(uuid4()))

    @classmethod
    def key(cls, item_id: str) -> dict[str, str]:
        """The primary key of one item, built from its id alone — what ``get_item``/``delete_item``
        need, and the one thing they can't get from an item they don't have yet."""
        return {'pk': f'{cls.entity}#{item_id}', 'sk': cls.meta_sk}

    @computed_field
    @property
    def pk(self) -> str:
        return f'{self.entity}#{self.id}'

    @computed_field
    @property
    def sk(self) -> str:
        return self.meta_sk

    @computed_field
    @property
    def entity_type(self) -> str:
        return self.entity.lower()


class ListedItem(TableItem):
    """A ``TableItem`` that can also be listed whole, via a constant GSI1 partition.

    ``Query`` needs an equality match on a partition key, and ``pk`` gives every item one of its
    own — so 'every room' is unaskable on the base table. Putting them all under one
    ``gsi1pk='TYPE#ROOM'`` makes the list a single query, over an index holding only that entity
    (a GSI is sparse: an item without its key attributes never enters it). Same shape the seed uses
    for places, ``db/load_dynamodb.py``.

    Subclasses declare ``gsi1sk``, since that is what orders the listing.
    """

    @computed_field
    @property
    def gsi1pk(self) -> str:
        return f'TYPE#{self.entity}'
