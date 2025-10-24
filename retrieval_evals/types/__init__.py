"""Type definitions for retrieval evaluations."""

from typing import Any, TypedDict


class QAPair(TypedDict):
    """Question-answer pair."""

    question: str
    answer: str


class QAPairWithGold(QAPair):
    """Question-answer pair with gold standard answer."""

    gold_answer: str


class EvalResult(TypedDict):
    """Result from an evaluation."""

    metric_name: str
    score: float
    individual_scores: list[Any]
    metadata: dict[str, Any]
