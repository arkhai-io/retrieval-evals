"""Answer length metric for ungrounded evaluation."""

from retrieval_evals.evals.ungrounded.metrics.base import Metric
from retrieval_evals.types import EvalResult, QAPair


class AnswerLength(Metric):
    """Answer length metric - computes statistics about answer lengths."""

    def __init__(self, unit: str = "words"):
        """Initialize answer length metric.

        Args:
            unit: Unit to measure length in ("words" or "characters", default: "words")
        """
        super().__init__(unit=unit)
        if unit not in ["words", "characters"]:
            raise ValueError(f"Unit must be 'words' or 'characters', got: {unit}")
        self.unit = unit

    @property
    def name(self) -> str:
        """Return metric name."""
        return f"answer_length_{self.unit}"

    def _compute_length(self, text: str) -> int:
        """Compute length based on unit.

        Args:
            text: Text to measure

        Returns:
            Length in specified unit
        """
        if self.unit == "words":
            return len(text.split())
        else:  # characters
            return len(text)

    def compute(self, qa_pairs: list[QAPair]) -> list[EvalResult]:
        """Compute answer length statistics.

        Args:
            qa_pairs: List of Q&A pairs

        Returns:
            List with single evaluation result containing length statistics
        """
        if not qa_pairs:
            return [
                {
                    "metric_name": self.name,
                    "score": 0.0,
                    "individual_scores": [],
                    "metadata": {
                        "num_pairs": 0,
                        "min": 0,
                        "max": 0,
                        "config": self.config,
                    },
                }
            ]

        # Compute individual lengths for all answers
        individual_scores = [float(self._compute_length(pair["answer"])) for pair in qa_pairs]

        # Calculate statistics
        avg_length = sum(individual_scores) / len(individual_scores)
        min_length = min(individual_scores)
        max_length = max(individual_scores)

        return [
            {
                "metric_name": self.name,
                "score": avg_length,
                "individual_scores": individual_scores,
                "metadata": {
                    "num_pairs": len(qa_pairs),
                    "min": min_length,
                    "max": max_length,
                    "unit": self.unit,
                    "config": self.config,
                },
            }
        ]
