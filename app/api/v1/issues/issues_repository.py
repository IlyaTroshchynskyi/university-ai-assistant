import asyncio
from contextlib import asynccontextmanager
from typing import AsyncGenerator
from uuid import uuid4

from boto3.dynamodb.conditions import Attr, Key
from types_aiobotocore_dynamodb import DynamoDBClient

from app.api.v1.rooms.enums import IssueStatus
from app.api.v1.rooms.schemas import Issue, IssueCreate, IssueItem
from app.core.dynamodb.base_service import DynamoDBService
from app.core.dynamodb.client import get_aioboto_session, open_dynamo_client
from app.core.dynamodb.indexes import build_issue_status_key, Index, KeyAttr
from app.settings import get_settings, Settings


class IssuesRepository(DynamoDBService):
    def __init__(self, client: DynamoDBClient, settings: Settings):
        super().__init__(client, settings.DYNAMODB_ISSUES_TABLE)

    async def list_most_voted_issues(self, limit: int) -> list[Issue]:
        async with asyncio.TaskGroup() as task_group:
            status_queries = [
                task_group.create_task(self._list_most_voted_issues_with_status(status, limit))
                for status in IssueStatus
            ]
        issues = [issue for status_query in status_queries for issue in status_query.result()]
        return sorted(issues, key=lambda issue: issue.votes, reverse=True)[:limit]

    async def create_issue(self, creation: IssueCreate) -> Issue:
        issue_item = IssueItem(id=str(uuid4()), status=IssueStatus.OPEN, votes=0, **creation.model_dump())
        await self.put_item(issue_item, condition=Attr(KeyAttr.PK).not_exists())
        return Issue.model_validate(issue_item.model_dump())

    async def _list_most_voted_issues_with_status(self, status: IssueStatus, limit: int) -> list[Issue]:
        rows = await self.query(
            key_condition=Key(KeyAttr.GSI1_PK).eq(build_issue_status_key(status)),
            index_name=Index.GSI1,
            limit=limit,
            ascending=False,
        )
        return [Issue.model_validate(row) for row in rows]


@asynccontextmanager
async def open_issues_repository() -> AsyncGenerator[IssuesRepository, None]:
    async with open_dynamo_client(get_aioboto_session(), get_settings()) as client:
        yield IssuesRepository(client, get_settings())
