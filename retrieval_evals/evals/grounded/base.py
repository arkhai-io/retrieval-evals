"""Base module for grounded evaluations.

This module handles evaluations where gold standard answers are available.
It compares generated answers against known correct answers.
"""

from pathlib import Path

from retrieval_evals.evals.grounded.metrics import ExactMatch, Metric
from retrieval_evals.types import EvalResult, QAPairWithGold

# Default metrics to run
DEFAULT_METRICS: list[Metric] = [
    ExactMatch(),
]


def evaluate_grounded(
    data: Path | str | list[QAPairWithGold],
    metrics: list[Metric] | None = None,
) -> list[EvalResult]:
    """Run grounded evaluation with gold standard answers.

    Args:
        data: Path to JSON file OR list of Q&A pairs with gold answers
        metrics: List of metric instances to run. If None, runs default metrics.

    Returns:
        List of evaluation results with scores

    Examples:
        # Run with default metrics
        results = evaluate_grounded("data.json")

        # Run with custom metrics
        from retrieval_evals.evals.grounded.metrics import ExactMatch

        results = evaluate_grounded(
            "data.json",
            metrics=[ExactMatch(case_sensitive=True)]
        )

        # Pass data directly
        qa_pairs = [{"question": "...", "answer": "...", "gold_answer": "..."}]
        results = evaluate_grounded(qa_pairs)
    """
    # Use default metrics if none specified
    metrics_to_run = metrics if metrics is not None else DEFAULT_METRICS

    # Run all metrics and collect results
    results: list[EvalResult] = []
    for metric in metrics_to_run:
        results.extend(metric.evaluate(data))

    return results
