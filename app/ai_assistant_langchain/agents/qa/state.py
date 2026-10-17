from langchain.agents.middleware.types import AgentState


class AssistantAgentState(AgentState):
    documents_on_file: list[dict[str, object]]
