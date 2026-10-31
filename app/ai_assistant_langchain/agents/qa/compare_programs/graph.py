from functools import lru_cache

from langgraph.constants import END, START
from langgraph.graph.state import CompiledStateGraph, StateGraph
from langgraph.types import Send

from app.ai_assistant_langchain.agents.qa.compare_programs.nodes import (
    gather_program_facts,
    merge_program_facts,
)
from app.ai_assistant_langchain.agents.qa.compare_programs.schemas import CompareState
from app.ai_assistant_langchain.graph.schemas import AgentRunContext

GATHER_NODE = 'gather'
MERGE_NODE = 'merge'


def fan_out_programs(state: CompareState) -> list[Send]:
    """One ``gather`` branch per programme, all started at once.

    ``Send`` rather than a node per programme: the branches differ only in which name they look up,
    and ``results`` already reduces with ``add``, so the number of them is data. Adding a third
    programme is then a longer list, not a new node and two new edges.
    """
    return [Send(GATHER_NODE, {'program': program}) for program in state['programs']]


@lru_cache
def get_compare_programs_graph() -> CompiledStateGraph:
    builder = StateGraph(CompareState, context_schema=AgentRunContext)
    builder.add_node(GATHER_NODE, gather_program_facts)
    builder.add_node(MERGE_NODE, merge_program_facts)

    builder.add_conditional_edges(START, fan_out_programs, [GATHER_NODE])
    builder.add_edge(GATHER_NODE, MERGE_NODE)
    builder.add_edge(MERGE_NODE, END)

    return builder.compile()
