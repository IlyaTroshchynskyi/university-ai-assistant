import re

from langchain_core.tools import BaseTool
import pytest

from app.ai_assistant_langchain.agents.booking.prompts import BOOKING_SYSTEM_PROMPT_TEMPLATE
from app.ai_assistant_langchain.agents.booking.tools import BOOKING_TOOLS
from app.ai_assistant_langchain.agents.qa.prompts import MAIN_CHAT_PROMPT
from app.ai_assistant_langchain.agents.qa.tools import ASSISTANT_TOOLS
from app.ai_assistant_langchain.applicant_documents.document_rules import ISSUE_CODES
from app.ai_assistant_langchain.applicant_documents.enums import FileOutcome
from app.ai_assistant_langchain.applicant_documents.schemas import Verdict
from app.ai_assistant_langchain.schemas import ATTACHMENT_LINE_TEMPLATE

ASSISTANT_TOOL_NAMES = [tool.name for tool in ASSISTANT_TOOLS]
BOOKING_TOOL_NAMES = [tool.name for tool in BOOKING_TOOLS]
ALL_TOOLS = [*ASSISTANT_TOOLS, *BOOKING_TOOLS]
ALL_TOOL_NAMES = [*ASSISTANT_TOOL_NAMES, *BOOKING_TOOL_NAMES]


def collect_tool_texts(tool: BaseTool) -> list[str]:
    return [tool.description, *(argument['description'] for argument in tool.args.values())]


class TestToolTexts:
    @pytest.mark.parametrize('tool', ALL_TOOLS, ids=ALL_TOOL_NAMES)
    def test_tool_and_its_arguments_name_no_other_tool(self, tool: BaseTool) -> None:
        tool_texts = ' '.join(collect_tool_texts(tool))

        assert [name for name in ALL_TOOL_NAMES if name != tool.name and name in tool_texts] == []

    @pytest.mark.parametrize('name', ASSISTANT_TOOL_NAMES)
    def test_qa_prompt_has_a_section_for_the_tool(self, name: str) -> None:
        assert re.search(rf'^ *"{name}" — .+:$', MAIN_CHAT_PROMPT, flags=re.MULTILINE)

    @pytest.mark.parametrize('name', BOOKING_TOOL_NAMES)
    def test_booking_prompt_names_the_tool(self, name: str) -> None:
        assert f'"{name}"' in BOOKING_SYSTEM_PROMPT_TEMPLATE

    def test_qa_prompt_shows_the_line_a_message_with_files_ends_with(self) -> None:
        assert ATTACHMENT_LINE_TEMPLATE.format(file_count=2) in MAIN_CHAT_PROMPT

    @pytest.mark.parametrize('issue_code', ISSUE_CODES)
    def test_issue_code_is_explained_wherever_the_model_reads_about_issues(self, issue_code: str) -> None:
        assert f'`{issue_code}' in str(Verdict.model_fields['issues'].description)
        assert f'"{issue_code}' in MAIN_CHAT_PROMPT

    @pytest.mark.parametrize('file_outcome', list(FileOutcome))
    def test_file_outcome_is_explained_wherever_the_model_reads_about_outcomes(self, file_outcome: FileOutcome) -> None:
        assert f'`{file_outcome}`' in str(Verdict.model_fields['file_outcomes'].description)
        assert f'"{file_outcome}"' in MAIN_CHAT_PROMPT
