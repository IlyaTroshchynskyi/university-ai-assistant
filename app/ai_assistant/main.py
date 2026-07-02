#!/usr/bin/env python
import asyncio
import logging

from crewai.flow import Flow, listen, start
from dotenv import load_dotenv
from pydantic import BaseModel

from app.ai_assistant.crews.university_crew.university_crew import UniversityCrew

load_dotenv()  # load OPENAI_API_KEY etc. from .env

logger = logging.getLogger(__name__)

DEFAULT_QUESTION = 'What programs does the university offer?'


class AssistantState(BaseModel):
    question: str = ''
    answer: str = ''


class UniversityAssistantFlow(Flow[AssistantState]):
    @start()
    def receive_question(self, crewai_trigger_payload: dict = None):
        if crewai_trigger_payload:
            self.state.question = crewai_trigger_payload.get('question', DEFAULT_QUESTION)
            logger.info('Using trigger payload: %s', crewai_trigger_payload)
        elif not self.state.question:
            self.state.question = DEFAULT_QUESTION

        logger.info('Question: %s', self.state.question)

    @listen(receive_question)
    async def answer_question(self):
        logger.info('Answering: %s', self.state.question)
        result = await UniversityCrew().crew().kickoff_async(inputs={'question': self.state.question})
        self.state.answer = result.raw
        logger.info('Answer generated')

    @listen(answer_question)
    def show_answer(self):
        logger.info('Answer: %s', self.state.answer)


async def answer_question(question: str) -> str:
    """Answer a single university question. Await this directly from FastAPI."""
    flow = UniversityAssistantFlow()
    await flow.kickoff_async(inputs={'question': question})
    return flow.state.answer


# --- Sync entrypoints (for the CLI scripts in pyproject.toml) ---


def kickoff():
    logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')
    flow = UniversityAssistantFlow()
    asyncio.run(flow.kickoff_async())


def plot():
    flow = UniversityAssistantFlow()
    flow.plot()


def run_with_trigger():
    """
    Run the flow with a trigger payload, e.g.:

        run_with_trigger '{"question": "How do I apply for a scholarship?"}'
    """
    import json
    import sys

    logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')

    if len(sys.argv) < 2:
        raise Exception('No trigger payload provided. Please provide JSON payload as argument.')

    try:
        trigger_payload = json.loads(sys.argv[1])
    except json.JSONDecodeError:
        raise Exception('Invalid JSON payload provided as argument')

    flow = UniversityAssistantFlow()

    try:
        return asyncio.run(flow.kickoff_async({'crewai_trigger_payload': trigger_payload}))
    except Exception as e:
        raise Exception(f'An error occurred while running the flow with trigger: {e}')


if __name__ == '__main__':
    kickoff()
