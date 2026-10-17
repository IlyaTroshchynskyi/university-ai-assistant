from pydantic import ValidationError
import pytest
from types_aiobotocore_dynamodb import DynamoDBClient

from app.ai_assistant_langchain.agents.qa.tools import list_common_issues, report_issue
from app.api.v1.rooms.enums import IssueCategory, IssueStatus
from app.api.v1.rooms.schemas import Issue
from tests.factory_helpers.factory_creators import create_test_issue_row
from tests.factory_helpers.factory_getters import get_test_issue_rows
from tests.factory_models.issue_factory import IssueRow

REPORTED_TEXT = 'The air conditioning in room 210 of Turing Hall is broken.'
REPORT = {'category': IssueCategory.FACILITIES, 'text': REPORTED_TEXT}


@pytest.mark.usefixtures('issues_table')
class TestListCommonIssues:
    async def test_issues_come_most_voted_first_whatever_their_category_and_status(
        self, dynamo_client: DynamoDBClient
    ) -> None:
        for category, status, votes in [
            (IssueCategory.FACILITIES, IssueStatus.OPEN, 52),
            (IssueCategory.FOOD, IssueStatus.IN_PROGRESS, 38),
            (IssueCategory.ADMIN, IssueStatus.RESOLVED, 60),
            (IssueCategory.OTHER, IssueStatus.OPEN, 7),
        ]:
            await create_test_issue_row(dynamo_client, category=category, status=status, votes=votes)

        issues = await list_common_issues.ainvoke({})

        assert [issue['votes'] for issue in issues] == [60, 52, 38, 7]

    async def test_only_as_many_issues_as_asked_for_are_returned(self, dynamo_client: DynamoDBClient) -> None:
        for status, votes in [
            (IssueStatus.OPEN, 9),
            (IssueStatus.OPEN, 31),
            (IssueStatus.IN_PROGRESS, 27),
            (IssueStatus.IN_PROGRESS, 4),
            (IssueStatus.RESOLVED, 19),
        ]:
            await create_test_issue_row(dynamo_client, status=status, votes=votes)

        issues = await list_common_issues.ainvoke({'limit': 2})

        assert [issue['votes'] for issue in issues] == [31, 27]

    async def test_issue_is_returned_with_its_category_text_status_and_votes(
        self, dynamo_client: DynamoDBClient
    ) -> None:
        row = await create_test_issue_row(dynamo_client, status=IssueStatus.IN_PROGRESS, votes=52)

        issues = await list_common_issues.ainvoke({})

        assert issues == [{'category': row.category, 'text': row.text, 'status': 'in_progress', 'votes': 52}]

    async def test_table_with_no_issues_is_reported_as_a_fact(self) -> None:
        result = await list_common_issues.ainvoke({})

        assert result == 'No student issues have been reported.'


@pytest.mark.usefixtures('issues_table')
class TestReportIssue:
    async def test_report_is_filed_as_an_open_issue_with_no_votes(self, dynamo_client: DynamoDBClient) -> None:
        result = await report_issue.ainvoke(REPORT)

        filed_issue = Issue(id=result['id'], status=IssueStatus.OPEN, votes=0, **REPORT)
        assert result == filed_issue.model_dump()
        assert await get_test_issue_rows(dynamo_client) == [IssueRow.from_entity(filed_issue)]

    async def test_filed_issue_is_listed_among_the_reported_ones(self, dynamo_client: DynamoDBClient) -> None:
        reported_row = await create_test_issue_row(dynamo_client, votes=3)

        await report_issue.ainvoke(REPORT)
        issues = await list_common_issues.ainvoke({})

        assert [issue['text'] for issue in issues] == [reported_row.text, REPORTED_TEXT]

    async def test_category_outside_the_allowed_ones_is_refused(self, dynamo_client: DynamoDBClient) -> None:
        with pytest.raises(ValidationError):
            await report_issue.ainvoke({**REPORT, 'category': 'parking'})

        assert await get_test_issue_rows(dynamo_client) == []
