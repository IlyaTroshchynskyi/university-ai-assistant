from crewai import Agent, Crew, Task
from crewai.agents.agent_builder.base_agent import BaseAgent
from crewai.project import agent, crew, CrewBase, task

from app.ai_assistant.tools.booking_tools import ListFreeSlotsTool, ProposedSlot
from app.settings import build_llm


@CrewBase
class BookingCrew:
    """Booking subagent: proposes a consultation slot. It never books — the write is
    a separate, human-gated step handled deterministically by the flow."""

    agents: list[BaseAgent]
    tasks: list[Task]

    agents_config = 'config/agents.yaml'
    tasks_config = 'config/tasks.yaml'

    @agent
    def booking_assistant(self) -> Agent:
        return Agent(
            config=self.agents_config['booking_assistant'],  # type: ignore[index]
            llm=build_llm(),
            tools=[
                ListFreeSlotsTool(),
            ],
        )

    @task
    def propose_slot_task(self) -> Task:
        return Task(
            config=self.tasks_config['propose_slot_task'],  # type: ignore[index]
            output_pydantic=ProposedSlot,
        )

    @crew
    def crew(self) -> Crew:
        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            verbose=True,
        )
