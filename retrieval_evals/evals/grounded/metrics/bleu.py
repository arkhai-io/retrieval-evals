"""BLEU metric for lexical overlap evaluation.

BLEU (Bilingual Evaluation Understudy) measures n-gram precision between the
generated answer and reference answer. It computes how many n-grams (1-4 by default)
in the generated answer appear in the reference answer.

Key characteristics:
- Precision-oriented: penalizes missing words from reference
- Uses geometric mean of n-gram precisions
- Applies brevity penalty for short answers
- Smoothing helps with zero n-gram counts (especially for short texts)

Score range: 0.0 (no overlap) to 1.0 (perfect match)
Best for: Comparing literal word/phrase overlap
"""

from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu

from retrieval_evals.evals.grounded.metrics.base import Metric
from retrieval_evals.types import EvalResult, QAPairWithGold


class BLEU(Metric):
    """BLEU score metric measuring n-gram precision overlap."""

    def __init__(self, max_n: int = 4, smoothing: bool = True):
        """Initialize BLEU metric.

        Args:
            max_n: Maximum n-gram size (default: 4)
            smoothing: Apply smoothing for zero counts (default: True)
        """
        super().__init__(max_n=max_n, smoothing=smoothing)
        self.max_n = max_n
        self.smoothing = smoothing
        self.smooth_fn = SmoothingFunction().method1 if smoothing else None

    @property
    def name(self) -> str:
        """Return metric name."""
        return f"bleu_{self.max_n}"

    def compute(self, qa_pairs: list[QAPairWithGold]) -> list[EvalResult]:
        """Compute BLEU scores.

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
                    "metadata": {"num_pairs": 0, "config": self.config},
                }
            ]

        individual_scores = []
        for pair in qa_pairs:
            reference = [pair["gold_answer"].split()]
            hypothesis = pair["answer"].split()

            # Compute BLEU score
            weights = tuple([1.0 / self.max_n] * self.max_n)
            score = sentence_bleu(
                reference, hypothesis, weights=weights, smoothing_function=self.smooth_fn
            )
            individual_scores.append(float(score))

        avg_score = sum(individual_scores) / len(individual_scores)

        return [
            {
                "metric_name": self.name,
                "score": avg_score,
                "individual_scores": individual_scores,
                "metadata": {
                    "num_pairs": len(qa_pairs),
                    "max_n": self.max_n,
                    "config": self.config,
                },
            }
        ]
