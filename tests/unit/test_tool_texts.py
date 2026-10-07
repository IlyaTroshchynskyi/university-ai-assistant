import re

from langchain_core.tools import BaseTool
import pytest

from app.ai_assistant_langchain.booking_tools import BOOKING_TOOLS
from app.ai_assistant_langchain.prompts import BOOKING_SYSTEM_PROMPT_TEMPLATE, MAIN_CHAT_PROMPT
from app.ai_assistant_langchain.tools import ASSISTANT_TOOLS

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
