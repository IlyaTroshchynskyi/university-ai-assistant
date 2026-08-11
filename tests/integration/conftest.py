import asyncio
from functools import lru_cache
import logging

from deepeval.metrics import (
    AnswerRelevancyMetric,
    BaseMetric,
    ContextualPrecisionMetric,
    ContextualRecallMetric,
    ContextualRelevancyMetric,
    FaithfulnessMetric,
    GEval,
    ToolCorrectnessMetric,
)
from deepeval.models import GPTModel
from deepeval.test_case import LLMTestCase, LLMTestCaseParams
import pytest
import pytest_asyncio

from app.ai_assistant.university_knowladge.knowledge_service import get_knowledge_service
from tests.integration.config import get_eval_settings

logger = logging.getLogger(__name__)

# Broad enough that any ingested handbook answers it. The preflight only asks "is there anything in
# there at all", so it deliberately does not use a golden — a golden that stops matching is a
# retrieval finding, not a reason to skip the suite.
_PREFLIGHT_QUERY = 'university programs and tuition'


@pytest_asyncio.fixture(scope='session', loop_scope='session')
async def knowledge_base_populated() -> None:
    """Skip the suite, with a reason, when there is nothing to retrieve.

    Without this an empty or unreachable collection hands ``retrieval_context=[]`` to every metric,
    and 41 goldens fail on low judge scores — which reads as "retrieval regressed" when it actually
    means "nobody ingested the handbook". It also spends real judge calls to say so.

    Requested per module rather than ``autouse``: the two scoring suites retrieve for real and need
    it, ``test_tool_routing_eval.py`` stubs the tools out and would be gated on infrastructure it
    never touches.
    """
    try:
        hits = await get_knowledge_service().search(_PREFLIGHT_QUERY)
    except Exception as error:  # noqa: BLE001 — infrastructure absence is a skip, not a failure
        pytest.skip(f'knowledge base is unreachable ({error.__class__.__name__}: {error})')

    if not hits:
        pytest.skip('knowledge base is empty — ingest the handbook via POST /documents first')


CORRECTNESS_CRITERIA = (
    'Decide whether the actual output states the same facts as the expected output. Every number, '
    'amount, date and name that appears in the expected output must appear in the actual output '
    'with the same value; a wrong or missing figure is a failure. Wording, ordering and extra '
    'helpful phrasing are free — do not penalise them.'
)


@lru_cache
def judge_model() -> GPTModel:
    settings = get_eval_settings()
    return GPTModel(model=settings.EVAL_JUDGE_MODEL, api_key=settings.EVAL_MODEL_API_KEY or None)


def retriever_metrics() -> list[BaseMetric]:
    """How good the passages the retriever returned are. Three metrics, and each one compares a
    different pair — the name says what it *scores*, not what it reads:

    ===================  ======================================  ===================================
    metric               compares                                a low score means
    ===================  ======================================  ===================================
    ContextualRelevancy  question ↔ passages                     the passages carry a lot the
                                                                 question never asked about
    ContextualRecall     expected answer ↔ passages              the passages are missing something
                                                                 the answer needs
    ContextualPrecision  question + expected answer ↔ the        the useful passages are ranked
                         *order* of the passages                 below the useless ones
    ===================  ======================================  ===================================

    **None of the three looks at the generated answer.** ``actual_output`` plays no part here,
    which is what makes this suite a measurement of retrieval alone — and why the cases in
    ``test_retriever_eval.py`` are built without one. DeepEval's docs list ``actual_output`` as a
    required param for all three; that is stale. In the installed 3.9.6 the string does not appear
    anywhere in the three metric packages, prompt templates included, and ``_required_params``
    does not contain it either.

    ``_required_params`` over-declares in the other direction too: it lists ``INPUT`` for recall,
    which in fact reads only ``expected_output`` and ``retrieval_context``. Take the table above,
    not the declaration, as the answer to "what would change this score".
    """
    settings, model = get_eval_settings(), judge_model()
    return [
        ContextualRelevancyMetric(threshold=settings.EVAL_CONTEXTUAL_RELEVANCY_THRESHOLD, model=model),
        ContextualRecallMetric(threshold=settings.EVAL_CONTEXTUAL_RECALL_THRESHOLD, model=model),
        ContextualPrecisionMetric(threshold=settings.EVAL_CONTEXTUAL_PRECISION_THRESHOLD, model=model),
    ]


def agent_metrics(has_context: bool) -> list[BaseMetric]:
    """How good the agent's answer is. Three metrics that deliberately pull in different
    directions, because each is blind to what the other two check:

    ===============  =========================================  ====================================
    metric           compares                                   a low score means
    ===============  =========================================  ====================================
    Correctness      answer ↔ the golden's expected answer      a figure, date or name is wrong or
                                                                missing
    AnswerRelevancy  question ↔ answer                          the answer padded, dodged, or
                                                                wandered off the question
    Faithfulness     answer ↔ the passages the tool returned    the answer states something the
                                                                passages do not
    ===============  =========================================  ====================================

    What each one *cannot* see matters as much as what it checks:

    * ``Correctness`` never receives the question — its ``evaluation_params`` are the two outputs
      and nothing else — so it cannot notice that an accurate answer ignored what was asked.
    * ``AnswerRelevancy`` never receives the expected answer or the passages, so a fluent, wholly
      invented answer scores 1.000 on it. It is the metric that polices padding, which
      ``CORRECTNESS_CRITERIA`` explicitly forgives; that division of labour is on purpose.
    * ``Faithfulness`` never receives the expected answer. Grounded is not the same as true: on
      golden 16 the agent quotes a date that really is in the retrieved passages but belongs to a
      different deadline, and passes at 0.857. Only ``Correctness`` catches that one.

    The consequence for dataset design: a golden's question has to be as wide as its expected
    answer. Ask narrowly while expecting four facts and the two metrics conflict with no
    resolution — answer narrowly and ``Correctness`` fails, answer fully and ``AnswerRelevancy``
    does, because neither can see the ground the other is standing on.

    (``Faithfulness`` also *declares* ``INPUT`` in its ``_required_params`` while reading only
    ``actual_output`` and ``retrieval_context``, same over-declaration as ``ContextualRecall``.)

    ``has_context`` is false when the agent answered without calling the retriever — small talk,
    for one. Both ready-made metrics are dropped there, and only correctness is left:

    * Faithfulness has nothing to check against, and scoring it against an empty context measures
      nothing.
    * Answer relevancy needs a question for the answer to be relevant *to*. A greeting is not one.
      Measured: "Hi there!" answered with "Hello! How can I assist you with your university
      questions today?" scores 0.500, the judge's reason being that the reply "uses 'Hello!'
      instead of matching the input greeting 'Hi there!'". No greeting can pass that.

    Correctness still runs, so a no-tool answer that invents facts is caught by the golden's
    expected answer — which is where the small-talk cases state what a good reply looks like."""
    settings = get_eval_settings()
    model = judge_model()
    metrics: list[BaseMetric] = [
        # answer ↔ expected answer. ``evaluation_params`` is the whole of what the judge sees, so
        # leaving INPUT out is what keeps this metric about facts rather than about relevance.
        GEval(
            name='Correctness',
            criteria=CORRECTNESS_CRITERIA,
            evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT, LLMTestCaseParams.EXPECTED_OUTPUT],
            threshold=settings.EVAL_CORRECTNESS_THRESHOLD,
            model=model,
        ),
    ]
    if has_context:
        metrics += [
            # question ↔ answer. Share of the answer's statements that address the question, so
            # the score is quantised by how many statements it has.
            AnswerRelevancyMetric(
                threshold=settings.EVAL_ANSWER_RELEVANCY_THRESHOLD,
                model=model,
                include_reason=True,
            ),
            # answer ↔ retrieved passages. The hallucination check: a ratio of the answer's claims
            # that the passages support.
            FaithfulnessMetric(
                threshold=settings.EVAL_FAITHFULNESS_THRESHOLD,
                model=model,
                include_reason=True,
            ),
        ]
    return metrics


def routing_metrics() -> list[BaseMetric]:
    """Which tool the agent reached for, given the question."""
    return [ToolCorrectnessMetric(threshold=1.0, should_exact_match=True)]


async def assert_metrics(test_case: LLMTestCase, metrics: list[BaseMetric]) -> None:
    """Score every metric against ``test_case``, report the whole table, then fail on the ones
    that came in under their threshold.

    Replaces ``deepeval.assert_test``, which measures on *copies* of the metrics (``copy_metrics``
    inside ``a_execute_test_cases``) and mentions only the failures — so a passing score cannot be
    read back out of it, and a green run tells you nothing about how close it was. Measuring here
    keeps every score on the metric objects themselves.

    Each metric is an independent judge call, so they run concurrently.

    Scores are logged rather than printed: ``pytest`` shows them under *Captured log call* on a
    failure, and ``--log-cli-level=INFO`` shows them on a pass too.
    """
    async with asyncio.TaskGroup() as task_group:
        for metric in metrics:
            task_group.create_task(metric.a_measure(test_case))

    logger.info('Metric scores for %r\n%s', test_case.input, _format_scores(metrics))

    failed = [metric for metric in metrics if not metric.success]
    assert not failed, 'Below threshold — ' + '; '.join(
        f'{metric.__name__} {metric.score:.3f} < {metric.threshold} ({metric.reason})' for metric in failed
    )


def _format_scores(metrics: list[BaseMetric]) -> str:
    """One aligned line per metric: name, score against its threshold, verdict."""
    width = max(len(metric.__name__) for metric in metrics)
    return '\n'.join(
        f'  {metric.__name__:<{width}}  {metric.score:.3f} / {metric.threshold:<5} '
        f'{"PASS" if metric.success else "FAIL"}'
        for metric in metrics
    )
