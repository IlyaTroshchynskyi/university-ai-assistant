from polyfactory.factories.pydantic_factory import ModelFactory
from pydantic import BaseModel

from app.api.v1.rooms.enums import IssueCategory, IssueStatus
from app.api.v1.rooms.schemas import Issue
from app.core.dynamodb.indexes import build_issue_status_key, build_issue_votes_key

ISSUE_TEXT = 'Wi-Fi keeps dropping in the Turing Hall lecture halls during class.'


class IssueFactory(ModelFactory[Issue]):
    __model__ = Issue

    category = IssueCategory.FACILITIES
    text = ISSUE_TEXT
    status = IssueStatus.OPEN
    votes = 0


class IssueRow(BaseModel):
    pk: str
    sk: str
    gsi1pk: str
    gsi1sk: str
    id: str
    category: str
    text: str
    status: str
    votes: int

    @classmethod
    def from_entity(cls, issue: Issue) -> 'IssueRow':
        votes_key = build_issue_votes_key(issue.votes, issue.id)
        return cls(
            pk=issue.category,
            sk=votes_key,
            gsi1pk=build_issue_status_key(issue.status),
            gsi1sk=votes_key,
            **issue.model_dump(),
        )
