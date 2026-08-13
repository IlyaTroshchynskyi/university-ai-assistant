"""The shared key layout, as a model mixin.

Each entity has a table of its own (``db/03-table-split.md``), but the key *values* still carry the
entity prefix: ``pk='ROOM#{id}'``, ``sk='#META'`` — the layout ``db/load_dynamodb.py`` seeds and the
repositories query. That is deliberate rather than left over from the single table: a prefixed key
reads unambiguously in a log or a dump, and it keeps a row movable between tables, which is what
made the split a pure move in the first place.

``sk`` earns its keep in a one-entity table too: ``#META`` marks the entity's own row, ``#UNIQUE``
a name reservation filed beside it — so every listing is the same scan filtered on ``sk = #META``,
whichever entity it lists.

``TableItem`` derives the key from the id instead of leaving it to each call site, so a write can't
reach DynamoDB with a missing or hand-rolled key (which is what ``ValidationException: One of the
required keys was not given a value`` means).

The key attributes are ``@computed_field``s: they belong to ``model_dump()`` — and so to the item
``DynamoDBService.put_item`` sends — but are read-only, so nothing outside can set them. ``id``
stays an ordinary field: generated on create, accepted back on read.

Each one carries ``# type: ignore[prop-decorator]``, here and in the other item/schema modules.
mypy refuses any decorator stacked on ``@property`` (mypy issue #1362) and has no setting to soften
it; the ignore on the ``@computed_field`` line is what pydantic's own docs prescribe. Dropping
``@property`` would silence it too, but then a type checker reads ``item.pk`` as the *method*
rather than as ``str`` — which is why pydantic recommends keeping it.
"""

from typing import ClassVar
from uuid import uuid4

from pydantic import BaseModel, computed_field, Field

# The counter a parent row carries: how many children are filed under it. Faculties count their
# programmes, professors and courses; programmes count their groups.
#
# A bare constant rather than a field on any model, and deliberately so. It is maintained by ``ADD``
# and read only by a delete condition, never deserialised — putting it on ``FacultyItem`` would leak
# it into every API response, which is the one thing the table split promised not to do. What it
# needs instead is a single spelling, since a repository writing 'dependants' and a guard reading
# 'dependents' would silently never refuse anything.
DEPENDANTS = 'dependants'


class TableItem(BaseModel):
    """An entity's keys, derived from its ``id``."""

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

    @computed_field  # type: ignore[prop-decorator]
    @property
    def pk(self) -> str:
        return f'{self.entity}#{self.id}'

    @computed_field  # type: ignore[prop-decorator]
    @property
    def sk(self) -> str:
        return self.meta_sk

    @classmethod
    def entity_type_value(cls) -> str:
        """The ``entity_type`` an item of this class is written with — derived from the class alone.

        Nothing queries it any more: a listing filters on ``sk = #META`` now that a table holds one
        entity. It stays on the row because it costs nothing and names the item in a dump, a log or
        the seeder's own bookkeeping."""
        return cls.entity.lower()

    @computed_field  # type: ignore[prop-decorator]
    @property
    def entity_type(self) -> str:
        return self.entity_type_value()
