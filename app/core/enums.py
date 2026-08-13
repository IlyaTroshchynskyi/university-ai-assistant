from enum import StrEnum


class CreateOutcome(StrEnum):
    """What a create that has a parent and a reserved name did, as the repository saw it.
    Maps to HTTP: ``PARENT_NOT_FOUND`` → 404, ``NAME_TAKEN`` → 409."""

    CREATED = 'created'
    PARENT_NOT_FOUND = 'parent_not_found'
    NAME_TAKEN = 'name_taken'


class DeleteOutcome(StrEnum):
    """What a guarded delete did, as the repository saw it.

    Maps to HTTP: ``NOT_FOUND`` → 404, ``HAS_DEPENDANTS`` → 409."""

    DELETED = 'deleted'
    NOT_FOUND = 'not_found'
    HAS_DEPENDANTS = 'has_dependants'
