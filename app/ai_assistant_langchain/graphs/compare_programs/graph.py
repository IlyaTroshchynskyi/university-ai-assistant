from functools import lru_cache

from langgraph.constants import END, START
from langgraph.graph.state import CompiledStateGraph, StateGraph
from langgraph.types import Send

from app.ai_assistant_langchain.agent_schemas import CustomContext
from app.ai_assistant_langchain.graphs.compare_programs.nodes import gather, merge
from app.ai_assistant_langchain.graphs.compare_programs.schemas import CompareState

GATHER = 'gather'
MERGE = 'merge'


def fan_out(state: CompareState) -> list[Send]:
    """One ``gather`` branch per programme, all started at once.

    ``Send`` rather than a node per programme: the branches differ only in which name they look up,
    and ``results`` already reduces with ``add``, so the number of them is data. Adding a third
    programme is then a longer list, not a new node and two new edges.
    """
    return [Send(GATHER, {'program': program}) for program in state['programs']]


@lru_cache
def get_compare_programs_graph() -> CompiledStateGraph:
    builder = StateGraph(CompareState, context_schema=CustomContext)
    builder.add_node(GATHER, gather)
    builder.add_node(MERGE, merge)

    builder.add_conditional_edges(START, fan_out, [GATHER])
    builder.add_edge(GATHER, MERGE)
    builder.add_edge(MERGE, END)

    return builder.compile()
