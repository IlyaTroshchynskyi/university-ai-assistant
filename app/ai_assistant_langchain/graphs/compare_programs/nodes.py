import logging

from langchain_core.messages import HumanMessage, SystemMessage

from app.ai_assistant.university_knowladge.knowledge_service import get_knowledge_service, join_passages, NO_RESULTS
from app.ai_assistant_langchain.agent_model import get_model_factory
from app.ai_assistant_langchain.graphs.compare_programs.schemas import (
    CompareState,
    GatherInput,
    GatherUpdate,
    MergeUpdate,
)
from app.ai_assistant_langchain.prompts import COMPARE_PROGRAMS_PROMPT, NOT_IN_KB

logger = logging.getLogger(__name__)


async def gather(state: GatherInput) -> GatherUpdate:
    program = state['program']
    logger.info('Gather started for %r', program)
    passages = join_passages(await get_knowledge_service().search(program))

    summary = '' if passages == NO_RESULTS else passages
    logger.info('Gather finished for %r (%s)', program, 'found' if summary else 'nothing found')
    return {'results': [{'program': program, 'summary': summary}]}


async def merge(state: CompareState) -> MergeUpdate:
    logger.info('Merging %s', ' vs '.join(repr(program) for program in state['programs']))
    model = get_model_factory()
    response = await model.ainvoke([SystemMessage(COMPARE_PROGRAMS_PROMPT), HumanMessage(_format_results(state))])
    return {'comparison': response.text.strip()}


def _format_results(state: CompareState) -> str:
    summaries = {result['program']: result['summary'] for result in state['results']}
    return '\n\n'.join(
        f'Programme: {program}\nWhat the knowledge base returned about it:\n{summaries[program] or NOT_IN_KB}'
        for program in state['programs']
    )
