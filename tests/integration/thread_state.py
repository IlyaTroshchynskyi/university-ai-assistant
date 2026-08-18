from typing import Container

from langchain_core.messages import ToolMessage
from langgraph.graph.state import CompiledStateGraph


async def get_tool_responses(
    graph: CompiledStateGraph,
    user_id: str,
    seen: int,
    ignore: Container[str] = (),
) -> tuple[list[str], int]:
    """Everything the tools returned since message ``seen`` of the thread, and the new watermark."""
    snapshot = await graph.aget_state({'configurable': {'thread_id': user_id}})
    messages = snapshot.values['messages']
    grounds = [
        str(message.content)
        for message in messages[seen:]
        if isinstance(message, ToolMessage) and message.content not in ignore
    ]
    return grounds, len(messages)
