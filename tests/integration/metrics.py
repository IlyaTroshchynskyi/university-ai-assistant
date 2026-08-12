import asyncio
from functools import lru_cache
import logging
from typing import Sequence, TypeAlias

from deepeval.metrics import (
    AnswerRelevancyMetric,
    BaseConversationalMetric,
    BaseMetric,
    BiasMetric,
    ContextualPrecisionMetric,
    ContextualRecallMetric,
    ContextualRelevancyMetric,
    ConversationalGEval,
    FaithfulnessMetric,
    GEval,
    ToolCorrectnessMetric,
    TurnFaithfulnessMetric,
)
from deepeval.models import GPTModel
from deepeval.test_case import ConversationalTestCase, LLMTestCase, LLMTestCaseParams, TurnParams

from tests.integration.config import get_eval_settings

logger = logging.getLogger(__name__)


AnyMetric: TypeAlias = BaseMetric | BaseConversationalMetric


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


OUTCOME_STEPS = [
    'Read the expected outcome and list every number, amount, date, name and yes/no claim it makes.',
    'For each item on that list, check whether the assistant turns state it with the same value. An '
    'item that is missing, or stated with a different value, is a failure.',
    'Check any figure the assistant gave early that the expected outcome says no longer stands: the '
    'assistant must have withdrawn or corrected it, not merely stopped repeating it.',
    'Ignore wording, ordering, tone, role consistency, helpfulness, and any extra facts beyond the '
    'expected outcome. Score only whether the expected facts were established.',
]


def conversation_metrics() -> list[BaseConversationalMetric]:
    """How good a whole conversation is. Two metrics, each blind to the other:

    ==================  =========================================  ==================================
    metric              compares                                   a low score means
    ==================  =========================================  ==================================
    TurnFaithfulness    assistant turns ↔ everything the tools      the agent stated something no
                        returned across the window                  passage or record supports
    Outcome             the turns ↔ the case's expected outcome     a fact is wrong, missing, or was
                                                                    not revised when it had to be
    ==================  =========================================  ==================================
    """
    settings, model = get_eval_settings(), judge_model()
    return [
        TurnFaithfulnessMetric(
            threshold=settings.EVAL_TURN_FAITHFULNESS_THRESHOLD,
            model=model,
            include_reason=True,
            window_size=20,
        ),
        ConversationalGEval(
            name='Outcome',
            evaluation_steps=OUTCOME_STEPS,
            evaluation_params=[TurnParams.CONTENT, TurnParams.EXPECTED_OUTCOME],
            threshold=settings.EVAL_CONVERSATION_OUTCOME_THRESHOLD,
            model=model,
        ),
    ]


def bias_metrics() -> list[BaseMetric]:
    """Whether the answer carries gender, political, racial/ethnic or geographical bias."""
    return [BiasMetric(threshold=get_eval_settings().EVAL_BIAS_THRESHOLD, model=judge_model(), include_reason=True)]


async def assert_metrics(test_case: LLMTestCase | ConversationalTestCase, metrics: Sequence[AnyMetric]) -> None:
    """Score every metric against ``test_case``, report the whole table, then fail on the ones that
    did not meet their threshold.

    Takes a ``ConversationalTestCase`` as readily as an ``LLMTestCase``: the conversational metrics
    expose the same ``a_measure``, ``threshold`` and ``success``.

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

    # ``input`` is an LLMTestCase field; a ConversationalTestCase has no single input, so it is
    # labelled by its name instead.
    label = test_case.input if isinstance(test_case, LLMTestCase) else test_case.name
    logger.info('Metric scores for %r\n%s', label, _format_scores(metrics))

    failed = [metric for metric in metrics if not metric.success]
    if failed:
        # Logged as well as asserted. pytest truncates a long assertion message, and the judge's
        # reason is the only thing that separates "the agent was wrong" from "the judge slipped" —
        # losing it means the run has to be paid for twice.
        logger.info('\n'.join(f'{metric.__name__} reason: {metric.reason}' for metric in failed))

    assert not failed, 'Failed — ' + '; '.join(
        f'{metric.__name__} {metric.score:.3f} vs {metric.threshold}' for metric in failed
    )


def _format_scores(metrics: Sequence[AnyMetric]) -> str:
    """One aligned line per metric: name, score against its threshold, verdict."""
    width = max(len(metric.__name__) for metric in metrics)
    return '\n'.join(
        f'  {metric.__name__:<{width}}  {metric.score:.3f} / {metric.threshold:<5} '
        f'{"PASS" if metric.success else "FAIL"}'
        for metric in metrics
    )
