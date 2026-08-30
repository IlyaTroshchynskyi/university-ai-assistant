from typing import Annotated, Sequence

from boto3.dynamodb.conditions import Attr, Key
from fastapi import Depends
from types_aiobotocore_dynamodb import DynamoDBClient

from app.ai_assistant_langchain.history.schemas import MessageRow, NewMessage
from app.core.dynamodb.base_service import DynamoDBService
from app.core.dynamodb.client import get_dynamo_client
from app.core.dynamodb.indexes import KeyAttr, thread_key
from app.core.dynamodb.schemas import TransactPut
from app.settings import get_settings, Settings


class ConversationHistoryRepository(DynamoDBService):
    """The ``conversation_history`` table: what was asked and what was answered, a row per message."""

    def __init__(
        self,
        dynamo_db_client: Annotated[DynamoDBClient, Depends(get_dynamo_client)],
        settings: Annotated[Settings, Depends(get_settings)],
    ) -> None:
        super().__init__(dynamo_db_client, settings.DYNAMODB_CONVERSATIONS_TABLE)

    async def append_turn(self, user_id: str, messages: Sequence[NewMessage]) -> list[MessageRow]:
        rows = [
            MessageRow(
                user_id=user_id,
                role=message.role,
                content=message.content,
                created_at=message.created_at,
            )
            for message in messages
        ]
        await self.transact_write(*(TransactPut(row, condition=Attr(KeyAttr.PK).not_exists()) for row in rows))
        return rows

    async def list_message_by_user_id(self, user_id: str) -> list[MessageRow]:
        rows = await self.query(key_condition=Key(KeyAttr.PK).eq(thread_key(user_id)), consistent_read=True)
        return [MessageRow.model_validate(row) for row in rows]
