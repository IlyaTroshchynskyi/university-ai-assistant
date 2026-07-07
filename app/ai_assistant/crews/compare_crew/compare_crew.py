from crewai import Agent, Crew, Task
from crewai.agents.agent_builder.base_agent import BaseAgent
from crewai.project import agent, crew, CrewBase, task

from app.ai_assistant.tools.retriever_tool import RetrieverTool
from app.settings import build_llm


@CrewBase
class ProgramResearchCrew:
    """Researches ONE program. The flow fans this out — one kickoff per program — so the
    two research runs happen in parallel (that is the fan-out)."""

    agents: list[BaseAgent]
    tasks: list[Task]

    agents_config = 'config/research_agents.yaml'
    tasks_config = 'config/research_tasks.yaml'

    @agent
    def program_researcher(self) -> Agent:
        return Agent(
            config=self.agents_config['program_researcher'],  # type: ignore[index]
            llm=build_llm(),
            tools=[
                RetrieverTool(),
            ],
        )

    @task
    def research_program_task(self) -> Task:
        return Task(config=self.tasks_config['research_program_task'])  # type: ignore[index]

    @crew
    def crew(self) -> Crew:
        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            verbose=True,
        )
