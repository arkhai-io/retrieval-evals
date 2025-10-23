"""Base module for ungrounded evaluations.

This module handles evaluations where gold standard answers are not available.
It uses alternative metrics like consistency, coherence, or model-based scoring.
"""

from pathlib import Path

from retrieval_evals.evals.ungrounded.metrics import AnswerLength, Metric
from retrieval_evals.types import EvalResult, QAPair

# Default metrics to run
DEFAULT_METRICS: list[Metric] = [
    AnswerLength(unit="words"),
]


def evaluate_ungrounded(
    data: Path | str | list[QAPair],
    metrics: list[Metric] | None = None,
) -> list[EvalResult]:
    """Run ungrounded evaluation without gold standard answers.

    Args:
        data: Path to JSON file OR list of Q&A pairs
        metrics: List of metric instances to run. If None, runs default metrics.

    Returns:
        List of evaluation results with scores

    Examples:
        # Run with default metrics
        results = evaluate_ungrounded("data.json")

        # Run with custom metrics
        from retrieval_evals.evals.ungrounded.metrics import AnswerLength

        results = evaluate_ungrounded(
            "data.json",
            metrics=[AnswerLength(unit="characters")]
        )

        # Pass data directly
        qa_pairs = [{"question": "...", "answer": "..."}]
        results = evaluate_ungrounded(qa_pairs)
    """
    # Use default metrics if none specified
    metrics_to_run = metrics if metrics is not None else DEFAULT_METRICS

    # Run all metrics and collect results
    results: list[EvalResult] = []
    for metric in metrics_to_run:
        results.extend(metric.evaluate(data))

    return results
