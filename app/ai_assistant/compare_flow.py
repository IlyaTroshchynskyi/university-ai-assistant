"""Compare-two-programs flow: an explicit fan-out / fan-in graph on CrewAI Flow.

    extract_programs (@start)          pull the two program names from the question
           |
      +----+----+                     fan-out: both listeners of the SAME trigger run
      |         |                      CONCURRENTLY (CrewAI runs them via asyncio.gather)
  research_a  research_b               each researches ONE program -> its OWN state key
      +----+----+
           |
        merge   @listen(and_(research_a, research_b))   fan-in: waits for BOTH, then
           |                                             writes the side-by-side answer
        answer

Concurrency note (the CrewAI take on LangGraph's "reducer"): CrewAI Flow has no reducer
to merge concurrent writes to one key. We don't need one — the two branches write to
SEPARATE keys (`info_a` / `info_b`), so there is nothing to merge and nothing to race.
"""

import logging

from crewai.flow import and_, Flow, listen, start
from pydantic import BaseModel, Field

from app.ai_assistant.crews.compare_crew.compare_crew import ProgramResearchCrew
from app.settings import build_llm

logger = logging.getLogger(__name__)


class ProgramPair(BaseModel):
    """Structured output for the extractor: the two programs the user wants to compare."""

    program_a: str = Field(description='The first program to compare.')
    program_b: str = Field(default='', description='The second program, empty if only one was named.')


class CompareState(BaseModel):
    question: str = ''
    program_a: str = ''
    program_b: str = ''
    # Each branch writes ONLY its own key -> no shared-key write, so no reducer needed.
    info_a: str = ''  # written only by research_a
    info_b: str = ''  # written only by research_b
    answer: str = ''


class CompareProgramsFlow(Flow[CompareState]):
    @start()
    async def extract_programs(self):
        prompt = (
            'The user wants to compare two university programs. Extract the two program '
            'names they are comparing from the message below. If only one program is named, '
            'leave the second empty.\n\n'
            f'Message: "{self.state.question}"'
        )
        pair = await build_llm().acall(prompt, response_model=ProgramPair)
        self.state.program_a = pair.program_a.strip()
        self.state.program_b = pair.program_b.strip()
        logger.info('Comparing %r vs %r', self.state.program_a, self.state.program_b)

    @listen(extract_programs)
    async def research_a(self):
        self.state.info_a = await self._research(self.state.program_a, self.state.question)
        logger.info('Research done for program A: %r', self.state.program_a)

    @listen(extract_programs)
    async def research_b(self):
        self.state.info_b = await self._research(self.state.program_b, self.state.question)
        logger.info('Research done for program B: %r', self.state.program_b)

    @listen(and_(research_a, research_b))
    async def merge(self):
        # Fan-in: a plain LLM call (no tools) turns the two summaries into one side-by-side
        # answer. It must not invent facts beyond info_a / info_b — that is why it has no
        # retriever tool, unlike the research agent.
        prompt = (
            'You are a university admissions assistant. Write a side-by-side comparison of '
            'two programs for an applicant, using ONLY the two summaries below — do not '
            'invent any fact beyond them.\n\n'
            f'Applicant request: "{self.state.question}"\n\n'
            f'Program A — {self.state.program_a}:\n{self.state.info_a}\n\n'
            f'Program B — {self.state.program_b}:\n{self.state.info_b}\n\n'
            'Write a clear, concise comparison in plain, readable prose — short paragraphs or '
            'a simple bulleted list. Do NOT use a wide markdown table. Cover focus, courses, '
            'duration, admission, tuition and career outcomes only where the summaries actually '
            'have information; if a summary lacks details, say so briefly in one sentence '
            'instead of listing every dimension as "Unknown". Finish with a short, balanced '
            'takeaway. Write the whole answer in the SAME language the applicant used.'
        )
        self.state.answer = (await build_llm().acall(prompt)).strip()
        logger.info('Comparison merged for %r vs %r', self.state.program_a, self.state.program_b)

    @staticmethod
    async def _research(program: str, question: str) -> str:
        """Run the single-program research crew for one program. Empty program -> empty info."""
        if not program:
            return ''
        result = await ProgramResearchCrew().crew().kickoff_async(inputs={'program': program, 'question': question})
        return result.raw
