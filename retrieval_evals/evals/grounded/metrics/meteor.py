"""METEOR metric for lexical overlap evaluation.

METEOR (Metric for Evaluation of Translation with Explicit ORdering) is more
sophisticated than BLEU/ROUGE. It considers:
1. Exact word matches
2. Stem matches (e.g., "running" matches "run")
3. Synonym matches (using WordNet)
4. Word order via chunk alignment

Key characteristics:
- Balances precision and recall (configurable via alpha)
- Penalizes fragmentation (words matched out of order)
- Handles paraphrasing better than pure n-gram methods
- More computationally expensive than BLEU/ROUGE

Score range: 0.0 (no match) to 1.0 (perfect match)
Best for: Evaluating semantic similarity with paraphrasing
"""

from nltk.translate.meteor_score import meteor_score

from retrieval_evals.evals.grounded.metrics.base import Metric
from retrieval_evals.types import EvalResult, QAPairWithGold


class METEOR(Metric):
    """METEOR score metric with synonym and stemming support."""

    def __init__(self, alpha: float = 0.9, beta: float = 3.0, gamma: float = 0.5):
        """Initialize METEOR metric.

        Args:
            alpha: Weight for precision vs recall (default: 0.9)
            beta: Penalty weight for fragmentation (default: 3.0)
            gamma: Weight for fragmentation penalty (default: 0.5)
        """
        super().__init__(alpha=alpha, beta=beta, gamma=gamma)
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

    @property
    def name(self) -> str:
        """Return metric name."""
        return "meteor"

    def compute(self, qa_pairs: list[QAPairWithGold]) -> list[EvalResult]:
        """Compute METEOR scores.

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
            reference = pair["gold_answer"].split()
            hypothesis = pair["answer"].split()

            # Compute METEOR score
            score = meteor_score(
                [reference],
                hypothesis,
                alpha=self.alpha,
                beta=self.beta,
                gamma=self.gamma,
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
                    "config": self.config,
                },
            }
        ]
