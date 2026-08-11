"""Agent evaluation: the answer ``POST /langchain-assistant`` gives to each golden, scored on
relevance, faithfulness to the retrieved passages, and factual agreement with the golden.

**A red run here is not automatically a regression.** Every metric is an LLM judge, and the
judge's spread is wider than the gap between a good answer and a mediocre one. Across five full
runs of the same code the failing set was, in order: ``{10}``, ``{1, 4, 8, 10, 12}``, ``{}``,
``{6}``, ``{4, 12, …}``. Read the score table that ``assert_metrics`` logs, not just the colour:
if the judge's reason describes the answer as correct, it is the judge that slipped.

Cases known to sit near a threshold, with the scores measured on them (2026-08-10):

============  ==================  ==========================================================
golden        metric              observed
============  ==================  ==========================================================
4  fee/       Correctness         0.676 · 0.557 · 0.845 · 0.639 · 0.576 — straddles 0.6, the
   discount                       agent's answer being the same each time
12 English    Answer Relevancy    1.000 · 0.500 · 1.000 · 1.000 · 0.500 — the 0.500 runs judge
   test                           a correct "yes, IELTS 6.5 / TOEFL 90" as not answering
1  DS         Correctness         0.461 once, 1.000 otherwise. The 0.461 reason itself read
   tuition                        "the key fact and figure are present and accurate"
6  Dean's     Answer Relevancy    0.667 · 0.714 · 0.857 · 1.000 — seven statements, so it can
   Award                          only land on 0.714 or 0.857 near the bar
16 housing    Answer Relevancy    0.750 · 0.778 and 0.857 · 0.875 — thin on both. Separately,
               / Faithfulness     the agent misdates the housing window (see below)
============  ==================  ==========================================================

One real defect the suite has found and nobody has fixed: on golden 16 the agent answers that
housing applications open on October 1, 2026. That is when *admission* applications open;
housing opens only once the tuition deposit is paid. Faithfulness scores it 0.857 — above the
bar, so the case passes and the error stands.

Thresholds live in ``tests/integration/config.py``; each carries the measurements behind it.
"""

import uuid

from deepeval.test_case import LLMTestCase
from langchain_core.messages import ToolMessage
import pytest

from app.ai_assistant.university_knowladge.knowledge_service import NO_RESULTS
from app.ai_assistant_langchain.agent import create_assistant_agent
from tests.conftest import TestBaseClientClass
from tests.integration.conftest import agent_metrics, assert_metrics
from tests.integration.goldens import golden_params, GoldenCase, Layer

# The suite drives the app through the session-scoped ``not_auth_client``, so the test has to run
# on the session loop too. Left on the default function loop it hangs: the client and the agent's
# cached clients belong to a loop nobody is running any more.
pytestmark = [
    pytest.mark.evaluation,
    pytest.mark.asyncio(loop_scope='session'),
    pytest.mark.usefixtures('knowledge_base_populated'),
]

ENDPOINT = '/langchain-assistant'
RETRIEVER_TOOL = 'retriever'
GOLDENS = golden_params(Layer.AGENT)


async def retrieved_chunks(user_id: str) -> list[str]:
    snapshot = await create_assistant_agent().aget_state({'configurable': {'thread_id': user_id}})
    return [
        message.content
        for message in snapshot.values['messages']
        if isinstance(message, ToolMessage) and message.name == RETRIEVER_TOOL and message.content != NO_RESULTS
    ]


class TestAgent(TestBaseClientClass):
    @pytest.mark.parametrize('golden', GOLDENS)
    async def test_agent_answer(self, golden: GoldenCase) -> None:
        user_id = str(uuid.uuid4())
        payload = {'query': golden.question, 'user_id': user_id}

        response = await self.not_auth_client.post(ENDPOINT, json=payload)
        assert response.status_code == 200, response.text

        chunks = await retrieved_chunks(user_id)
        test_case = LLMTestCase(
            input=golden.question,
            actual_output=response.json()['message'],
            expected_output=golden.expected_answer,
            retrieval_context=chunks or None,
        )

        # No chunks means the agent answered without the tool — small talk, or a question it
        # declined. Neither faithfulness nor answer relevancy means anything there, so
        # ``agent_metrics`` drops both and scores the reply on correctness alone.
        await assert_metrics(test_case, agent_metrics(has_context=bool(chunks)))
