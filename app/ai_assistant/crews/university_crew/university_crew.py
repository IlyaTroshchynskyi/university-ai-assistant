from crewai import Agent, Crew, Process, Task
from crewai.agents.agent_builder.base_agent import BaseAgent
from crewai.project import agent, crew, CrewBase, task

from app.ai_assistant.tools.finder_tools import FindPersonTool, FindPlaceTool
from app.ai_assistant.tools.retriever_tool import RetrieverTool


@CrewBase
class UniversityCrew:
    """University Assistant Crew"""

    agents: list[BaseAgent]
    tasks: list[Task]

    agents_config = 'config/agents.yaml'
    tasks_config = 'config/tasks.yaml'

    @agent
    def university_assistant(self) -> Agent:
        return Agent(
            config=self.agents_config['university_assistant'],  # type: ignore[index]
            tools=[
                RetrieverTool(),
                FindPersonTool(),
                FindPlaceTool(),
            ],
        )

    @task
    def answer_question_task(self) -> Task:
        return Task(
            config=self.tasks_config['answer_question_task'],  # type: ignore[index]
        )

    @crew
    def crew(self) -> Crew:
        """Creates the University Assistant Crew"""
        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            process=Process.sequential,
            verbose=True,
        )
