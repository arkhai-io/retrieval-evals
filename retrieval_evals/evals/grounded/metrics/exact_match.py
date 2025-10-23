"""Exact match metric for grounded evaluation."""

from retrieval_evals.evals.grounded.metrics.base import Metric
from retrieval_evals.types import EvalResult, QAPairWithGold


class ExactMatch(Metric):
    """Exact match metric - checks if answer exactly matches gold answer."""

    def __init__(self, case_sensitive: bool = False, normalize_whitespace: bool = True):
        """Initialize exact match metric.

        Args:
            case_sensitive: Whether to perform case-sensitive matching (default: False)
            normalize_whitespace: Whether to normalize whitespace before comparison (default: True)
        """
        super().__init__(case_sensitive=case_sensitive, normalize_whitespace=normalize_whitespace)
        self.case_sensitive = case_sensitive
        self.normalize_whitespace = normalize_whitespace

    @property
    def name(self) -> str:
        """Return metric name."""
        return "exact_match"

    def _normalize(self, text: str) -> str:
        """Normalize text based on config.

        Args:
            text: Text to normalize

        Returns:
            Normalized text
        """
        if self.normalize_whitespace:
            text = " ".join(text.split())

        if not self.case_sensitive:
            text = text.lower()

        return text.strip()

    def compute(self, qa_pairs: list[QAPairWithGold]) -> list[EvalResult]:
        """Compute exact match scores.

        Args:
            qa_pairs: List of Q&A pairs with gold standard answers

        Returns:
            List with single evaluation result
        """
        if not qa_pairs:
            return [
                {
                    "metric_name": self.name,
                    "score": 0.0,
                    "individual_scores": [],
                    "metadata": {"num_pairs": 0, "num_matches": 0, "config": self.config},
                }
            ]

        # Compute individual scores (1.0 for match, 0.0 for no match)
        individual_scores = []
        for pair in qa_pairs:
            answer = self._normalize(pair["answer"])
            gold_answer = self._normalize(pair["gold_answer"])

            if answer == gold_answer:
                individual_scores.append(1.0)
            else:
                individual_scores.append(0.0)

        # Calculate overall accuracy (average of individual scores)
        accuracy = sum(individual_scores) / len(individual_scores)
        matches = sum(individual_scores)

        return [
            {
                "metric_name": self.name,
                "score": accuracy,
                "individual_scores": individual_scores,
                "metadata": {
                    "num_pairs": len(qa_pairs),
                    "num_matches": int(matches),
                    "config": self.config,
                },
            }
        ]
