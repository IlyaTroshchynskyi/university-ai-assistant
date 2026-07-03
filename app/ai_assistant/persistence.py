"""Flow persistence backends.

`@human_feedback` needs a `FlowPersistence` so a paused flow can be saved and later
restored (`from_pending`). CrewAI's default is SQLite; here we use a process-memory
store instead. To move to a real store (e.g. DynamoDB), implement the same five methods
against that backend and swap `FLOW_PERSISTENCE` below — nothing in the flow changes.
"""

from __future__ import annotations

from typing import Any, TYPE_CHECKING

from crewai.flow.persistence.base import FlowPersistence
from pydantic import BaseModel

if TYPE_CHECKING:
    from crewai.flow.async_feedback.types import PendingFeedbackContext

# Process-wide in-memory stores, keyed by flow id. Reset on restart. TODO: replace with
# a durable backend (DynamoDB, etc.) by implementing FlowPersistence, not by editing this.
_STATES: dict[str, dict[str, Any]] = {}
_PENDING: dict[str, tuple[dict[str, Any], 'PendingFeedbackContext']] = {}


class InMemoryFlowPersistence(FlowPersistence):
    """Keeps flow state and pending-feedback context in process memory (no disk)."""

    persistence_type: str = 'in_memory'

    def init_db(self) -> None:
        # Nothing to set up for an in-memory store.
        pass

    @staticmethod
    def _to_dict(state_data: dict[str, Any] | BaseModel) -> dict[str, Any]:
        return state_data.model_dump() if isinstance(state_data, BaseModel) else dict(state_data)

    def save_state(self, flow_uuid: str, method_name: str, state_data: dict[str, Any] | BaseModel) -> None:
        _STATES[flow_uuid] = self._to_dict(state_data)

    def load_state(self, flow_uuid: str) -> dict[str, Any] | None:
        return _STATES.get(flow_uuid)

    def save_pending_feedback(
        self,
        flow_uuid: str,
        context: PendingFeedbackContext,
        state_data: dict[str, Any] | BaseModel,
    ) -> None:
        state = self._to_dict(state_data)
        _STATES[flow_uuid] = state
        _PENDING[flow_uuid] = (state, context)

    def load_pending_feedback(
        self,
        flow_uuid: str,
    ) -> tuple[dict[str, Any], PendingFeedbackContext] | None:
        return _PENDING.get(flow_uuid)

    def clear_pending_feedback(self, flow_uuid: str) -> None:
        _PENDING.pop(flow_uuid, None)


# The single instance the flow uses for both @persist and from_pending. Swap this line
# for a DynamoFlowPersistence() later.
FLOW_PERSISTENCE = InMemoryFlowPersistence()
